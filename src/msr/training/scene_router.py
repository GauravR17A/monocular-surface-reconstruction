"""Offline supervision and training helpers for conservative scene routing.

The router must answer a narrow question: *which already-frozen endpoint is
safer for this complete scene?*  Its target is consequently derived from the
two endpoints' measured utility against held-out reference labels.  Dataset
source and landscape are retained only for balanced sampling and reporting;
neither is accepted as a routing target.

This module intentionally operates on precomputed, detached scene descriptors.
Training it therefore cannot update either endpoint or the shared height model.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


@dataclass(frozen=True)
class SceneUtilityConfig:
    """Weights for label-derived endpoint utility (lower is better).

    Height RMSE is divided by ``height_scale_m`` before it is combined with
    semantic balanced error, keeping the two components on comparable scales.
    A component with zero weight is ignored.  If a scene lacks valid labels for
    one component, the remaining valid component determines its utility.
    """

    height_rmse_weight: float = 1.0
    semantic_error_weight: float = 1.0
    height_scale_m: float = 10.0

    def __post_init__(self) -> None:
        if self.height_rmse_weight < 0 or self.semantic_error_weight < 0:
            raise ValueError("scene utility weights cannot be negative")
        if self.height_rmse_weight + self.semantic_error_weight <= 0:
            raise ValueError("at least one scene utility component must be enabled")
        if self.height_scale_m <= 0:
            raise ValueError("height_scale_m must be positive")


@dataclass(frozen=True)
class SceneEndpointUtility:
    """Auditable per-scene score computed from reference labels."""

    utility: float
    height_rmse_m: float | None
    semantic_balanced_error: float | None
    valid_height_pixels: int
    valid_semantic_pixels: int
    observed_semantic_classes: int

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class SceneRouterDecision:
    """Binary route label plus the measured reason for that label."""

    candidate_target: bool
    candidate_gain: float
    fallback_utility: float
    candidate_utility: float
    minimum_candidate_gain: float


def _single_scene_tensor(
    value: torch.Tensor | np.ndarray,
    *,
    name: str,
    channels: int | None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if channels is None:
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.ndim == 3 and tensor.shape[0] != 1:
            raise ValueError(f"{name} must describe exactly one scene")
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape (H, W) or (1, H, W)")
        return tensor

    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != channels:
        raise ValueError(
            f"{name} must have shape ({channels}, H, W) or "
            f"(1, {channels}, H, W)"
        )
    return tensor


def compute_scene_endpoint_utility(
    endpoint: Mapping[str, torch.Tensor],
    *,
    height_target: torch.Tensor | np.ndarray | None = None,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    domain_target: torch.Tensor | np.ndarray | None = None,
    domain_valid_mask: torch.Tensor | np.ndarray | None = None,
    config: SceneUtilityConfig | None = None,
    height_key: str = "height",
    domain_logits_key: str = "domain_logits",
) -> SceneEndpointUtility:
    """Measure one endpoint on one scene using only held-out labels.

    Semantic error is one minus macro recall over the reference classes that
    actually occur in this scene.  This avoids letting a large ground area hide
    complete failure on a smaller building or vegetation class.
    """

    config = config or SceneUtilityConfig()
    weighted_components: list[tuple[float, float]] = []
    height_rmse_m: float | None = None
    semantic_error: float | None = None
    valid_height_pixels = 0
    valid_semantic_pixels = 0
    observed_semantic_classes = 0

    if config.height_rmse_weight > 0 and height_target is not None:
        if height_key not in endpoint:
            raise KeyError(f"endpoint is missing height output {height_key!r}")
        prediction = _single_scene_tensor(
            endpoint[height_key], name=height_key, channels=1
        )[0].to(torch.float64)
        target = _single_scene_tensor(
            height_target, name="height_target", channels=None
        ).to(torch.float64)
        if prediction.shape != target.shape:
            raise ValueError("height prediction and target spatial shapes differ")
        # Reference support is defined without looking at endpoint output so
        # both frozen endpoints are scored on the exact same pixels.
        height_valid = torch.isfinite(target)
        if valid_mask is not None:
            supplied_valid = _single_scene_tensor(
                valid_mask, name="valid_mask", channels=None
            ).bool()
            if supplied_valid.shape != target.shape:
                raise ValueError("valid_mask and height_target shapes differ")
            height_valid &= supplied_valid
        valid_height_pixels = int(torch.count_nonzero(height_valid))
        if valid_height_pixels:
            if not bool(torch.all(torch.isfinite(prediction[height_valid]))):
                raise ValueError(
                    f"{height_key} contains non-finite values on valid reference support"
                )
            error = prediction[height_valid] - target[height_valid]
            height_rmse_m = float(torch.sqrt(torch.mean(torch.square(error))))
            weighted_components.append(
                (
                    config.height_rmse_weight,
                    height_rmse_m / config.height_scale_m,
                )
            )

    if config.semantic_error_weight > 0 and domain_target is not None:
        if domain_logits_key not in endpoint:
            raise KeyError(
                f"endpoint is missing semantic output {domain_logits_key!r}"
            )
        logits_value = torch.as_tensor(endpoint[domain_logits_key])
        if logits_value.ndim == 4 and logits_value.shape[0] == 1:
            logits_value = logits_value[0]
        if logits_value.ndim != 3 or logits_value.shape[0] < 2:
            raise ValueError(
                f"{domain_logits_key} must have shape (classes, H, W) or "
                "(1, classes, H, W)"
            )
        logits = logits_value.detach().cpu()
        target = _single_scene_tensor(
            domain_target, name="domain_target", channels=None
        ).long()
        if logits.shape[-2:] != target.shape:
            raise ValueError("semantic logits and target spatial shapes differ")
        semantic_valid = (target >= 0) & (target < logits.shape[0])
        if domain_valid_mask is not None:
            supplied_valid = _single_scene_tensor(
                domain_valid_mask,
                name="domain_valid_mask",
                channels=None,
            ).bool()
            if supplied_valid.shape != target.shape:
                raise ValueError(
                    "domain_valid_mask and domain_target shapes differ"
                )
            semantic_valid &= supplied_valid
        valid_semantic_pixels = int(torch.count_nonzero(semantic_valid))
        if valid_semantic_pixels:
            if not bool(torch.all(torch.isfinite(logits[:, semantic_valid]))):
                raise ValueError(
                    f"{domain_logits_key} contains non-finite values on valid "
                    "reference support"
                )
            prediction = logits.argmax(dim=0)
            recalls: list[torch.Tensor] = []
            for class_index in range(logits.shape[0]):
                class_mask = semantic_valid & (target == class_index)
                support = torch.count_nonzero(class_mask)
                if int(support):
                    recalls.append(
                        torch.count_nonzero(
                            class_mask & (prediction == class_index)
                        ).to(torch.float64)
                        / support
                    )
            observed_semantic_classes = len(recalls)
            semantic_error = float(1.0 - torch.stack(recalls).mean())
            weighted_components.append(
                (config.semantic_error_weight, semantic_error)
            )

    if not weighted_components:
        raise ValueError("scene has no valid enabled reference labels")
    weight_sum = sum(weight for weight, _ in weighted_components)
    utility = sum(weight * value for weight, value in weighted_components) / weight_sum
    return SceneEndpointUtility(
        utility=float(utility),
        height_rmse_m=height_rmse_m,
        semantic_balanced_error=semantic_error,
        valid_height_pixels=valid_height_pixels,
        valid_semantic_pixels=valid_semantic_pixels,
        observed_semantic_classes=observed_semantic_classes,
    )


def derive_scene_router_target(
    fallback: SceneEndpointUtility | float,
    candidate: SceneEndpointUtility | float,
    *,
    minimum_candidate_gain: float = 0.0,
) -> SceneRouterDecision:
    """Choose candidate only when reference utility proves sufficient gain.

    Ties and improvements smaller than ``minimum_candidate_gain`` deliberately
    remain on the protected fallback endpoint.
    """

    if minimum_candidate_gain < 0:
        raise ValueError("minimum_candidate_gain cannot be negative")
    fallback_utility = float(
        fallback.utility if isinstance(fallback, SceneEndpointUtility) else fallback
    )
    candidate_utility = float(
        candidate.utility if isinstance(candidate, SceneEndpointUtility) else candidate
    )
    if not math.isfinite(fallback_utility) or not math.isfinite(candidate_utility):
        raise ValueError("endpoint utilities must be finite")
    gain = fallback_utility - candidate_utility
    return SceneRouterDecision(
        candidate_target=bool(gain > minimum_candidate_gain),
        candidate_gain=float(gain),
        fallback_utility=fallback_utility,
        candidate_utility=candidate_utility,
        minimum_candidate_gain=float(minimum_candidate_gain),
    )


@dataclass(frozen=True)
class SceneRouterRecord:
    """One detached descriptor with label-derived endpoint utilities."""

    sample_id: str
    descriptor: torch.Tensor
    fallback_utility: float
    candidate_utility: float
    source: str
    landscape: str
    partition: str = "train"

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id cannot be empty")
        descriptor = torch.as_tensor(self.descriptor).detach().cpu().float()
        if descriptor.ndim != 1 or descriptor.numel() <= 0:
            raise ValueError("descriptor must be a non-empty one-dimensional tensor")
        if not bool(torch.all(torch.isfinite(descriptor))):
            raise ValueError("descriptor must contain only finite values")
        for name in ("source", "landscape", "partition"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        if not math.isfinite(float(self.fallback_utility)) or not math.isfinite(
            float(self.candidate_utility)
        ):
            raise ValueError("record utilities must be finite")
        object.__setattr__(self, "descriptor", descriptor.clone())
        object.__setattr__(self, "source", self.source.strip().lower())
        object.__setattr__(self, "landscape", self.landscape.strip().lower())
        object.__setattr__(self, "partition", self.partition.strip().lower())

    def decision(self, minimum_candidate_gain: float = 0.0) -> SceneRouterDecision:
        return derive_scene_router_target(
            self.fallback_utility,
            self.candidate_utility,
            minimum_candidate_gain=minimum_candidate_gain,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "descriptor": self.descriptor.tolist(),
            "fallback_utility": float(self.fallback_utility),
            "candidate_utility": float(self.candidate_utility),
            "source": self.source,
            "landscape": self.landscape,
            "partition": self.partition,
        }

    @classmethod
    def from_json_dict(cls, value: Mapping[str, object]) -> "SceneRouterRecord":
        required = {
            "sample_id",
            "descriptor",
            "fallback_utility",
            "candidate_utility",
            "source",
            "landscape",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"router record is missing fields: {missing}")
        # Deliberately ignore/recompute any supplied target.  A source label or
        # hand-authored binary target can never override measured utilities.
        return cls(
            sample_id=str(value["sample_id"]),
            descriptor=torch.tensor(value["descriptor"], dtype=torch.float32),
            fallback_utility=float(value["fallback_utility"]),
            candidate_utility=float(value["candidate_utility"]),
            source=str(value["source"]),
            landscape=str(value["landscape"]),
            partition=str(value.get("partition", "train")),
        )


def build_scene_router_record(
    *,
    sample_id: str,
    descriptor: torch.Tensor | np.ndarray,
    fallback_endpoint: Mapping[str, torch.Tensor],
    candidate_endpoint: Mapping[str, torch.Tensor],
    source: str,
    landscape: str,
    partition: str,
    height_target: torch.Tensor | np.ndarray | None = None,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    domain_target: torch.Tensor | np.ndarray | None = None,
    domain_valid_mask: torch.Tensor | np.ndarray | None = None,
    utility_config: SceneUtilityConfig | None = None,
    height_key: str = "height",
    domain_logits_key: str = "domain_logits",
) -> tuple[SceneRouterRecord, dict[str, dict[str, float | int | None]]]:
    """Build one auditable record directly from two frozen endpoint outputs.

    The returned diagnostics preserve component-level evidence for reports.  A
    binary target is still intentionally absent from the stored record and is
    derived only when :class:`SceneRouterDataset` is indexed.
    """

    fallback_utility = compute_scene_endpoint_utility(
        fallback_endpoint,
        height_target=height_target,
        valid_mask=valid_mask,
        domain_target=domain_target,
        domain_valid_mask=domain_valid_mask,
        config=utility_config,
        height_key=height_key,
        domain_logits_key=domain_logits_key,
    )
    candidate_utility = compute_scene_endpoint_utility(
        candidate_endpoint,
        height_target=height_target,
        valid_mask=valid_mask,
        domain_target=domain_target,
        domain_valid_mask=domain_valid_mask,
        config=utility_config,
        height_key=height_key,
        domain_logits_key=domain_logits_key,
    )
    descriptor_tensor = torch.as_tensor(descriptor).detach().cpu()
    if descriptor_tensor.ndim == 2 and descriptor_tensor.shape[0] == 1:
        descriptor_tensor = descriptor_tensor[0]
    record = SceneRouterRecord(
        sample_id=sample_id,
        descriptor=descriptor_tensor,
        fallback_utility=fallback_utility.utility,
        candidate_utility=candidate_utility.utility,
        source=source,
        landscape=landscape,
        partition=partition,
    )
    return record, {
        "fallback": fallback_utility.to_dict(),
        "candidate": candidate_utility.to_dict(),
    }


def write_scene_router_records(
    records: Sequence[SceneRouterRecord], path: str | Path
) -> None:
    """Write portable JSONL descriptors without executable pickle payloads."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")


def read_scene_router_records(path: str | Path) -> list[SceneRouterRecord]:
    records: list[SceneRouterRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("record must be a JSON object")
                records.append(SceneRouterRecord.from_json_dict(value))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid router record at line {line_number}: {error}"
                ) from error
    if not records:
        raise ValueError("scene-router record file is empty")
    descriptor_sizes = {record.descriptor.numel() for record in records}
    if len(descriptor_sizes) != 1:
        raise ValueError("all scene-router descriptors must have the same size")
    return records


class SceneRouterDataset(Dataset[dict[str, Any]]):
    """Dataset that always derives its target from endpoint utility."""

    def __init__(
        self,
        records: Sequence[SceneRouterRecord],
        *,
        minimum_candidate_gain: float = 0.0,
    ) -> None:
        if not records:
            raise ValueError("SceneRouterDataset requires at least one record")
        if minimum_candidate_gain < 0:
            raise ValueError("minimum_candidate_gain cannot be negative")
        descriptor_sizes = {record.descriptor.numel() for record in records}
        if len(descriptor_sizes) != 1:
            raise ValueError("all router descriptors must have the same size")
        self.records = tuple(records)
        self.minimum_candidate_gain = float(minimum_candidate_gain)
        self.descriptor_size = descriptor_sizes.pop()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        decision = record.decision(self.minimum_candidate_gain)
        return {
            "descriptor": record.descriptor.clone(),
            "candidate_target": torch.tensor(
                [float(decision.candidate_target)], dtype=torch.float32
            ),
            "fallback_utility": torch.tensor(
                [record.fallback_utility], dtype=torch.float32
            ),
            "candidate_utility": torch.tensor(
                [record.candidate_utility], dtype=torch.float32
            ),
            "candidate_gain": torch.tensor(
                [decision.candidate_gain], dtype=torch.float32
            ),
            "sample_id": record.sample_id,
            "source": record.source,
            "landscape": record.landscape,
            "partition": record.partition,
        }


class BalancedSourceLandscapeSampler(Sampler[int]):
    """Deterministically give each observed source/landscape stratum equal weight.

    Labels are intentionally absent from the grouping key so sampling cannot
    manufacture an artificially calibrated candidate rate.
    """

    def __init__(
        self,
        records: Sequence[SceneRouterRecord],
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        if not records:
            raise ValueError("balanced router sampling requires records")
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            groups[(record.source, record.landscape)].append(index)
        if num_samples is None:
            num_samples = max(len(indices) for indices in groups.values()) * len(groups)
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_samples % len(groups):
            raise ValueError(
                "num_samples must be divisible by the number of observed "
                "source/landscape strata for exact balance"
            )
        self.groups = {key: tuple(value) for key, value in sorted(groups.items())}
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

    @property
    def samples_per_stratum(self) -> int:
        return self.num_samples // len(self.groups)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    @staticmethod
    def _cycles(
        population: Sequence[int], count: int, generator: torch.Generator
    ) -> list[int]:
        draws: list[int] = []
        while len(draws) < count:
            permutation = torch.randperm(
                len(population), generator=generator
            ).tolist()
            draws.extend(population[index] for index in permutation)
        return draws[:count]

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws: list[int] = []
        for population in self.groups.values():
            draws.extend(
                self._cycles(population, self.samples_per_stratum, generator)
            )
        order = torch.randperm(len(draws), generator=generator).tolist()
        return iter(draws[index] for index in order)


class SceneRouterObjective(nn.Module):
    """Asymmetric BCE with an optional Brier calibration regularizer."""

    def __init__(
        self,
        *,
        candidate_weight: float = 1.0,
        fallback_weight: float = 2.0,
        brier_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if candidate_weight <= 0 or fallback_weight <= 0:
            raise ValueError("BCE class weights must be positive")
        if brier_weight < 0:
            raise ValueError("brier_weight cannot be negative")
        self.candidate_weight = float(candidate_weight)
        self.fallback_weight = float(fallback_weight)
        self.brier_weight = float(brier_weight)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if logits.shape != targets.shape:
            raise ValueError("router logits and targets must have identical shapes")
        if not bool(torch.all((targets == 0) | (targets == 1))):
            raise ValueError("router targets must be binary")
        weights = torch.where(
            targets.bool(),
            torch.as_tensor(
                self.candidate_weight, device=logits.device, dtype=logits.dtype
            ),
            torch.as_tensor(
                self.fallback_weight, device=logits.device, dtype=logits.dtype
            ),
        )
        # Normalize the batch weights so changing conservatism does not also
        # silently change the effective learning rate.
        weighted_bce = (
            F.binary_cross_entropy_with_logits(
                logits, targets.to(logits.dtype), reduction="none"
            )
            * weights
        ).sum() / weights.sum().clamp_min(1.0)
        brier = torch.mean(
            torch.square(logits.sigmoid() - targets.to(logits.dtype))
        )
        return {
            "loss": weighted_bce + self.brier_weight * brier,
            "bce": weighted_bce,
            "brier": brier,
        }


def expected_calibration_error(
    probabilities: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    *,
    bins: int = 10,
) -> float:
    """Equal-width binary expected calibration error."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if probability.shape != target.shape or probability.size == 0:
        raise ValueError("probabilities and targets must be equally sized and non-empty")
    if np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    if np.any((target != 0) & (target != 1)):
        raise ValueError("targets must be binary")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(probability.size)
    error = 0.0
    for index in range(bins):
        if index == bins - 1:
            in_bin = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            in_bin = (probability >= edges[index]) & (probability < edges[index + 1])
        count = int(np.count_nonzero(in_bin))
        if count:
            error += (count / total) * abs(
                float(probability[in_bin].mean()) - float(target[in_bin].mean())
            )
    return float(error)


def compute_scene_router_metrics(
    probabilities: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    *,
    threshold: float,
    fallback_utilities: np.ndarray | torch.Tensor | None = None,
    candidate_utilities: np.ndarray | torch.Tensor | None = None,
    calibration_bins: int = 10,
) -> dict[str, float | int]:
    """Return classification, calibration, and routed-utility diagnostics."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be within (0, 1]")
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if probability.shape != target.shape or probability.size == 0:
        raise ValueError("probabilities and targets must be equally sized and non-empty")
    if np.any(~np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must be finite and within [0, 1]")
    if np.any((target != 0) & (target != 1)):
        raise ValueError("targets must be binary")
    truth = target.astype(bool)
    selected = probability >= threshold
    tp = int(np.count_nonzero(selected & truth))
    fp = int(np.count_nonzero(selected & ~truth))
    fn = int(np.count_nonzero(~selected & truth))
    tn = int(np.count_nonzero(~selected & ~truth))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    true_negative_rate = tn / (tn + fp) if tn + fp else 1.0
    clipped = np.clip(probability, 1.0e-7, 1.0 - 1.0e-7)
    metrics: dict[str, float | int] = {
        "scene_count": int(probability.size),
        "threshold": float(threshold),
        "candidate_selected": int(np.count_nonzero(selected)),
        "candidate_selection_rate": float(selected.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": float((tp + tn) / probability.size),
        "balanced_accuracy": float((recall + true_negative_rate) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier_score": float(np.mean(np.square(probability - target))),
        "log_loss": float(
            -np.mean(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
        ),
        "expected_calibration_error": expected_calibration_error(
            probability, target, bins=calibration_bins
        ),
        "mean_probability_minus_candidate_rate": float(
            probability.mean() - target.mean()
        ),
    }
    if (fallback_utilities is None) != (candidate_utilities is None):
        raise ValueError("both fallback and candidate utilities must be supplied together")
    if fallback_utilities is not None and candidate_utilities is not None:
        fallback = np.asarray(fallback_utilities, dtype=np.float64).reshape(-1)
        candidate = np.asarray(candidate_utilities, dtype=np.float64).reshape(-1)
        if fallback.shape != probability.shape or candidate.shape != probability.shape:
            raise ValueError("utility arrays must match probability shape")
        if np.any(~np.isfinite(fallback)) or np.any(~np.isfinite(candidate)):
            raise ValueError("utility arrays must be finite")
        routed = np.where(selected, candidate, fallback)
        oracle = np.minimum(fallback, candidate)
        metrics.update(
            {
                "fallback_mean_utility": float(fallback.mean()),
                "candidate_mean_utility": float(candidate.mean()),
                "routed_mean_utility": float(routed.mean()),
                "oracle_mean_utility": float(oracle.mean()),
                "routed_gain_vs_fallback": float(fallback.mean() - routed.mean()),
                "routed_regret_vs_oracle": float(routed.mean() - oracle.mean()),
                "selected_candidate_mean_gain": float(
                    np.mean((fallback - candidate)[selected])
                    if np.any(selected)
                    else 0.0
                ),
            }
        )
    return metrics


@dataclass(frozen=True)
class ConservativeThresholdResult:
    threshold: float
    eligible: bool
    reason: str
    metrics: dict[str, float | int]
    evaluated_thresholds: int


def select_conservative_threshold(
    probabilities: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    fallback_utilities: np.ndarray | torch.Tensor,
    candidate_utilities: np.ndarray | torch.Tensor,
    *,
    minimum_threshold: float = 0.5,
    minimum_precision: float = 0.9,
    maximum_mean_utility_regression: float = 0.0,
    minimum_candidate_selections: int = 1,
    calibration_bins: int = 10,
) -> ConservativeThresholdResult:
    """Choose a safe operating threshold on a separate calibration split.

    Eligible thresholds must meet precision and mean-utility safety guards.
    Among them, the largest measured gain over always using the protected
    fallback wins.  Ties prefer higher precision and then a higher threshold.
    If none qualify, the returned near-one threshold selects no candidate for
    finite sigmoid logits.
    """

    if not 0.0 < minimum_threshold < 1.0:
        raise ValueError("minimum_threshold must be within (0, 1)")
    if not 0.0 <= minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be within [0, 1]")
    if maximum_mean_utility_regression < 0:
        raise ValueError("maximum_mean_utility_regression cannot be negative")
    if minimum_candidate_selections <= 0:
        raise ValueError("minimum_candidate_selections must be positive")
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    fallback_threshold = float(np.nextafter(1.0, 0.0))
    candidate_thresholds = sorted(
        {
            float(minimum_threshold),
            *(float(value) for value in probability if minimum_threshold <= value < 1.0),
        }
    )
    eligible: list[tuple[tuple[float, float, float], dict[str, float | int]]] = []
    for threshold in candidate_thresholds:
        metrics = compute_scene_router_metrics(
            probability,
            targets,
            threshold=threshold,
            fallback_utilities=fallback_utilities,
            candidate_utilities=candidate_utilities,
            calibration_bins=calibration_bins,
        )
        if int(metrics["candidate_selected"]) < minimum_candidate_selections:
            continue
        if float(metrics["precision"]) < minimum_precision:
            continue
        if (
            float(metrics["routed_gain_vs_fallback"])
            < -maximum_mean_utility_regression
        ):
            continue
        score = (
            float(metrics["routed_gain_vs_fallback"]),
            float(metrics["precision"]),
            float(threshold),
        )
        eligible.append((score, metrics))
    if eligible:
        _, best_metrics = max(eligible, key=lambda item: item[0])
        return ConservativeThresholdResult(
            threshold=float(best_metrics["threshold"]),
            eligible=True,
            reason="precision and routed-utility safety guards passed",
            metrics=best_metrics,
            evaluated_thresholds=len(candidate_thresholds),
        )
    fallback_metrics = compute_scene_router_metrics(
        probability,
        targets,
        threshold=fallback_threshold,
        fallback_utilities=fallback_utilities,
        candidate_utilities=candidate_utilities,
        calibration_bins=calibration_bins,
    )
    return ConservativeThresholdResult(
        threshold=fallback_threshold,
        eligible=False,
        reason="no candidate threshold passed; route all scenes to fallback",
        metrics=fallback_metrics,
        evaluated_thresholds=len(candidate_thresholds),
    )


def freeze_for_scene_router_training(*modules: nn.Module) -> None:
    """Freeze and put endpoint/shared modules in deterministic evaluation mode."""

    for module in modules:
        module.requires_grad_(False)
        module.eval()


def _router_logits_from_descriptor(router: nn.Module, descriptor: torch.Tensor) -> torch.Tensor:
    forward_descriptor = getattr(router, "forward_descriptor", None)
    if callable(forward_descriptor):
        result = forward_descriptor(descriptor)
        if isinstance(result, Mapping):
            result = result.get("candidate_logit")
        if not isinstance(result, torch.Tensor):
            raise TypeError("router.forward_descriptor must return logits or a mapping")
        return result
    network = getattr(router, "network", None)
    if isinstance(network, nn.Module):
        return network(descriptor)
    return router(descriptor)


@dataclass(frozen=True)
class SceneRouterTrainingConfig:
    epochs: int = 20
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    candidate_weight: float = 1.0
    fallback_weight: float = 2.0
    brier_weight: float = 0.05
    minimum_threshold: float = 0.75
    minimum_precision: float = 0.9
    maximum_mean_utility_regression: float = 0.0
    minimum_candidate_selections: int = 1

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid scene-router optimizer configuration")


@dataclass(frozen=True)
class SceneRouterTrainingResult:
    history: tuple[dict[str, float | int], ...]
    threshold: ConservativeThresholdResult
    best_validation_bce: float


def _collect_router_validation(
    router: nn.Module,
    loader: DataLoader,
    device: torch.device,
    objective: SceneRouterObjective,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    router.eval()
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    fallback_parts: list[torch.Tensor] = []
    candidate_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            descriptor = batch["descriptor"].to(device)
            target = batch["candidate_target"].to(device)
            logits = _router_logits_from_descriptor(router, descriptor)
            if logits.shape != target.shape:
                raise ValueError("router output shape does not match scene targets")
            logits_parts.append(logits.cpu())
            target_parts.append(target.cpu())
            fallback_parts.append(batch["fallback_utility"].cpu())
            candidate_parts.append(batch["candidate_utility"].cpu())
    if not logits_parts:
        raise ValueError("validation loader produced no scenes")
    logits = torch.cat(logits_parts).float()
    targets = torch.cat(target_parts).float()
    objective_values = objective(logits, targets)
    arrays = {
        "probabilities": logits.sigmoid().numpy(),
        "targets": targets.numpy(),
        "fallback_utilities": torch.cat(fallback_parts).numpy(),
        "candidate_utilities": torch.cat(candidate_parts).numpy(),
    }
    metrics = compute_scene_router_metrics(
        arrays["probabilities"],
        arrays["targets"],
        threshold=0.5,
        fallback_utilities=arrays["fallback_utilities"],
        candidate_utilities=arrays["candidate_utilities"],
    )
    metrics.update(
        {
            "objective": float(objective_values["loss"]),
            "bce": float(objective_values["bce"]),
        }
    )
    return metrics, arrays


def train_scene_router(
    router: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    config: SceneRouterTrainingConfig | None = None,
    device: str | torch.device = "cpu",
    frozen_modules: Iterable[nn.Module] = (),
) -> SceneRouterTrainingResult:
    """Train only a router on detached descriptors, then choose a safe threshold."""

    config = config or SceneRouterTrainingConfig()
    device = torch.device(device)
    frozen_modules = tuple(frozen_modules)
    freeze_for_scene_router_training(*frozen_modules)
    nested_router = getattr(router, "router", None)
    routing_module = nested_router if isinstance(nested_router, nn.Module) else router
    router.to(device)
    # Accept either the tiny router itself or a complete routed wrapper.  Start
    # by freezing the entire supplied object, then reopen only its router.  This
    # prevents a caller from accidentally optimizing the frozen endpoints or
    # shared height trunk merely by passing the outer wrapper here.
    router.requires_grad_(False)
    routing_module.requires_grad_(True)
    trainable_ids = {id(parameter) for parameter in routing_module.parameters()}
    trainable = [parameter for parameter in routing_module.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("router has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    objective = SceneRouterObjective(
        candidate_weight=config.candidate_weight,
        fallback_weight=config.fallback_weight,
        brier_weight=config.brier_weight,
    )
    history: list[dict[str, float | int]] = []
    best_validation_bce = math.inf
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(config.epochs):
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        router.eval()
        routing_module.train()
        total_loss = 0.0
        scenes = 0
        for batch in train_loader:
            descriptor = batch["descriptor"].to(device)
            target = batch["candidate_target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _router_logits_from_descriptor(routing_module, descriptor)
            values = objective(logits, target)
            values["loss"].backward()
            optimizer.step()
            batch_scenes = int(descriptor.shape[0])
            total_loss += float(values["loss"].detach()) * batch_scenes
            scenes += batch_scenes
        if not scenes:
            raise ValueError("training loader produced no scenes")
        validation_metrics, _ = _collect_router_validation(
            routing_module, validation_loader, device, objective
        )
        validation_bce = float(validation_metrics["bce"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_objective": total_loss / scenes,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
        )
        if validation_bce < best_validation_bce:
            best_validation_bce = validation_bce
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in routing_module.state_dict().items()
            }
        if any(
            parameter.grad is not None
            for parameter in router.parameters()
            if id(parameter) not in trainable_ids
        ):
            raise RuntimeError("a non-router parameter received a gradient")
        if any(parameter.grad is not None for module in frozen_modules for parameter in module.parameters()):
            raise RuntimeError("a frozen endpoint/shared parameter received a gradient")

    assert best_state is not None
    routing_module.load_state_dict(best_state)
    _, arrays = _collect_router_validation(
        routing_module, validation_loader, device, objective
    )
    threshold = select_conservative_threshold(
        arrays["probabilities"],
        arrays["targets"],
        arrays["fallback_utilities"],
        arrays["candidate_utilities"],
        minimum_threshold=config.minimum_threshold,
        minimum_precision=config.minimum_precision,
        maximum_mean_utility_regression=config.maximum_mean_utility_regression,
        minimum_candidate_selections=config.minimum_candidate_selections,
    )
    decision_threshold = getattr(routing_module, "decision_threshold", None)
    if isinstance(decision_threshold, torch.Tensor):
        decision_threshold.fill_(threshold.threshold)
    router.eval()
    return SceneRouterTrainingResult(
        history=tuple(history),
        threshold=threshold,
        best_validation_bce=float(best_validation_bce),
    )


__all__ = [
    "BalancedSourceLandscapeSampler",
    "ConservativeThresholdResult",
    "SceneEndpointUtility",
    "SceneRouterDataset",
    "SceneRouterDecision",
    "SceneRouterObjective",
    "SceneRouterRecord",
    "SceneRouterTrainingConfig",
    "SceneRouterTrainingResult",
    "SceneUtilityConfig",
    "build_scene_router_record",
    "compute_scene_endpoint_utility",
    "compute_scene_router_metrics",
    "derive_scene_router_target",
    "expected_calibration_error",
    "freeze_for_scene_router_training",
    "read_scene_router_records",
    "select_conservative_threshold",
    "train_scene_router",
    "write_scene_router_records",
]
