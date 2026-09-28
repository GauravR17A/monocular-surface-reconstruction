"""Versioned Stage-4 records and isolated dual endpoint-gain routers.

Stage-3 records intentionally retain only one aggregate endpoint utility.  That
is insufficient to learn different routing policies for semantic outputs and
height outputs.  This module therefore defines a new, incompatible rich-record
schema which retains the two measured utility components for both frozen
endpoints.  Old records fail closed rather than being reinterpreted.

Only detached, all-pixel scene descriptors are model inputs.  Reference labels,
source, group ID, landscape, and endpoint scores are supervision/audit metadata and
are never passed to either router network.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .scene_router import (
    SceneRouterRecord,
    SceneUtilityConfig,
    build_scene_router_record,
)


STAGE4_RECORD_SCHEMA = "msr.stage4_dual_router_record.v2"
STAGE4_RECORD_STORE_SCHEMA = "msr.stage4_dual_router_records.v2"
STAGE4_ARTIFACT_SCHEMA = "msr.stage4_dual_endpoint_router.v1"
STAGE4_REPORT_SCHEMA = "msr.stage4_dual_endpoint_router_report.v1"
SEMANTIC_CLASS_COUNT = 3

ComponentName = Literal["height", "semantic"]


@dataclass(frozen=True)
class EndpointComponentUtilities:
    """Reference-measured endpoint errors retained for independent routing."""

    aggregate_utility: float
    height_sse: float | None
    height_rmse_m: float | None
    semantic_confusion_3x3: tuple[tuple[int, int, int], ...]
    semantic_balanced_error: float | None
    valid_height_pixels: int
    valid_semantic_pixels: int
    observed_semantic_classes: int

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.aggregate_utility)):
            raise ValueError("aggregate endpoint utility must be finite")
        for name in (
            "valid_height_pixels",
            "valid_semantic_pixels",
            "observed_semantic_classes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.valid_height_pixels:
            if (
                self.height_sse is None
                or not math.isfinite(float(self.height_sse))
                or float(self.height_sse) < 0
                or self.height_rmse_m is None
                or not math.isfinite(float(self.height_rmse_m))
                or float(self.height_rmse_m) < 0
            ):
                raise ValueError("valid height support requires finite SSE and RMSE")
            recomputed_rmse = math.sqrt(
                float(self.height_sse) / self.valid_height_pixels
            )
            if not math.isclose(
                recomputed_rmse,
                float(self.height_rmse_m),
                rel_tol=1e-6,
                abs_tol=1e-7,
            ):
                raise ValueError("height SSE/count disagree with height RMSE")
        elif self.height_rmse_m is not None or self.height_sse is not None:
            raise ValueError("height SSE/RMSE must be null without valid height pixels")
        confusion = self.semantic_confusion_3x3
        if (
            not isinstance(confusion, tuple)
            or len(confusion) != SEMANTIC_CLASS_COUNT
            or any(
                not isinstance(row, tuple)
                or len(row) != SEMANTIC_CLASS_COUNT
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in row
                )
                for row in confusion
            )
        ):
            raise ValueError("semantic_confusion_3x3 must be a non-negative 3x3 matrix")
        confusion_count = sum(sum(row) for row in confusion)
        supported_classes = sum(int(sum(row) > 0) for row in confusion)
        if confusion_count != self.valid_semantic_pixels:
            raise ValueError("semantic confusion count disagrees with valid pixels")
        if supported_classes != self.observed_semantic_classes:
            raise ValueError("semantic confusion support disagrees with observed classes")
        if self.valid_semantic_pixels:
            if (
                self.semantic_balanced_error is None
                or not math.isfinite(float(self.semantic_balanced_error))
                or not 0.0 <= float(self.semantic_balanced_error) <= 1.0
                or self.observed_semantic_classes <= 0
            ):
                raise ValueError(
                    "valid semantic support requires bounded balanced error and classes"
                )
            recalls = [
                confusion[index][index] / sum(confusion[index])
                for index in range(SEMANTIC_CLASS_COUNT)
                if sum(confusion[index]) > 0
            ]
            recomputed_error = 1.0 - sum(recalls) / len(recalls)
            if not math.isclose(
                recomputed_error,
                float(self.semantic_balanced_error),
                rel_tol=1e-6,
                abs_tol=1e-7,
            ):
                raise ValueError(
                    "semantic confusion matrix disagrees with balanced error"
                )
        elif (
            self.semantic_balanced_error is not None
            or self.observed_semantic_classes != 0
        ):
            raise ValueError(
                "semantic error/classes must be empty without valid semantic pixels"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "EndpointComponentUtilities":
        required = {
            "height_sse",
            "height_rmse_m",
            "semantic_confusion_3x3",
            "semantic_balanced_error",
            "valid_height_pixels",
            "valid_semantic_pixels",
            "observed_semantic_classes",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"endpoint component utility is missing fields: {missing}")
        aggregate = value.get("aggregate_utility", value.get("utility"))
        if aggregate is None:
            raise ValueError("endpoint component utility has no aggregate utility")
        return cls(
            aggregate_utility=float(aggregate),
            height_sse=(
                None if value["height_sse"] is None else float(value["height_sse"])
            ),
            height_rmse_m=(
                None
                if value["height_rmse_m"] is None
                else float(value["height_rmse_m"])
            ),
            semantic_balanced_error=(
                None
                if value["semantic_balanced_error"] is None
                else float(value["semantic_balanced_error"])
            ),
            semantic_confusion_3x3=tuple(
                tuple(int(item) for item in row)
                for row in value["semantic_confusion_3x3"]  # type: ignore[union-attr]
            ),
            valid_height_pixels=int(value["valid_height_pixels"]),
            valid_semantic_pixels=int(value["valid_semantic_pixels"]),
            observed_semantic_classes=int(value["observed_semantic_classes"]),
        )

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    def component(self, name: ComponentName) -> float | None:
        if name == "height":
            return self.height_rmse_m
        if name == "semantic":
            return self.semantic_balanced_error
        raise ValueError(f"unsupported Stage-4 routing component: {name!r}")


@dataclass(frozen=True)
class Stage4DualRouterRecord:
    """One authenticated descriptor with both endpoint component utilities."""

    sample_id: str
    descriptor: torch.Tensor
    protected: EndpointComponentUtilities
    candidate: EndpointComponentUtilities
    source: str
    group_id: str
    landscape: str
    partition: str

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id cannot be empty")
        descriptor = torch.as_tensor(self.descriptor).detach().cpu().float()
        if descriptor.ndim != 1 or descriptor.numel() <= 0:
            raise ValueError("descriptor must be a non-empty one-dimensional tensor")
        if not bool(torch.all(torch.isfinite(descriptor))):
            raise ValueError("descriptor must contain only finite values")
        for name in ("source", "group_id", "landscape", "partition"):
            value = str(getattr(self, name)).strip().lower()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        if self.partition not in {"train", "calibration"}:
            raise ValueError("partition must be train or calibration")
        if self.protected.valid_height_pixels != self.candidate.valid_height_pixels:
            raise ValueError("endpoint height utilities must use identical support")
        if self.protected.valid_semantic_pixels != self.candidate.valid_semantic_pixels:
            raise ValueError("endpoint semantic utilities must use identical support")
        if (
            self.protected.observed_semantic_classes
            != self.candidate.observed_semantic_classes
        ):
            raise ValueError("endpoint semantic utilities must observe identical classes")
        object.__setattr__(self, "descriptor", descriptor.clone())

    @property
    def group_key(self) -> str:
        return f"{self.source}/{self.group_id}"

    @property
    def stratum_key(self) -> str:
        return f"{self.source}/{self.landscape}"

    def component_pair(self, name: ComponentName) -> tuple[float, float] | None:
        protected = self.protected.component(name)
        candidate = self.candidate.component(name)
        if protected is None or candidate is None:
            return None
        return float(protected), float(candidate)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema": STAGE4_RECORD_SCHEMA,
            "sample_id": self.sample_id,
            "descriptor": self.descriptor.tolist(),
            "protected": self.protected.to_json_dict(),
            "candidate": self.candidate.to_json_dict(),
            "source": self.source,
            "group_id": self.group_id,
            "landscape": self.landscape,
            "partition": self.partition,
        }

    @classmethod
    def from_json_dict(cls, value: Mapping[str, object]) -> "Stage4DualRouterRecord":
        if value.get("schema") != STAGE4_RECORD_SCHEMA:
            raise ValueError(
                "Stage-4 dual routing requires rich v2 records; aggregate Stage-3 "
                "records cannot be upgraded without reference-derived components"
            )
        required = {
            "sample_id",
            "descriptor",
            "protected",
            "candidate",
            "source",
            "group_id",
            "landscape",
            "partition",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"Stage-4 record is missing fields: {missing}")
        if not isinstance(value["protected"], Mapping) or not isinstance(
            value["candidate"], Mapping
        ):
            raise ValueError("Stage-4 endpoint utilities must be mappings")
        return cls(
            sample_id=str(value["sample_id"]),
            descriptor=torch.as_tensor(value["descriptor"], dtype=torch.float32),
            protected=EndpointComponentUtilities.from_mapping(value["protected"]),
            candidate=EndpointComponentUtilities.from_mapping(value["candidate"]),
            source=str(value["source"]),
            group_id=str(value["group_id"]),
            landscape=str(value["landscape"]),
            partition=str(value["partition"]),
        )


def _scene_2d(value: torch.Tensor | np.ndarray, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape (H, W) or (1, H, W)")
    return tensor


def _exact_component_evidence(
    endpoint: Mapping[str, torch.Tensor],
    *,
    height_target: torch.Tensor | np.ndarray | None,
    valid_mask: torch.Tensor | np.ndarray | None,
    domain_target: torch.Tensor | np.ndarray | None,
    domain_valid_mask: torch.Tensor | np.ndarray | None,
) -> dict[str, object]:
    height_sse: float | None = None
    height_count = 0
    if height_target is not None:
        if "height" not in endpoint:
            raise KeyError("endpoint is missing height")
        prediction = torch.as_tensor(endpoint["height"]).detach().cpu()
        if prediction.ndim == 4 and prediction.shape[:2] == (1, 1):
            prediction = prediction[0, 0]
        elif prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim != 2:
            raise ValueError("endpoint height must describe exactly one scene")
        target = _scene_2d(height_target, name="height_target").to(torch.float64)
        if prediction.shape != target.shape:
            raise ValueError("height prediction and target shapes differ")
        support = torch.isfinite(target)
        if valid_mask is not None:
            supplied = _scene_2d(valid_mask, name="valid_mask").bool()
            if supplied.shape != target.shape:
                raise ValueError("height valid mask and target shapes differ")
            support &= supplied
        height_count = int(torch.count_nonzero(support))
        if height_count:
            prediction = prediction.to(torch.float64)
            if not bool(torch.all(torch.isfinite(prediction[support]))):
                raise ValueError("endpoint height is non-finite on reference support")
            height_sse = float(torch.square(prediction[support] - target[support]).sum())

    confusion = torch.zeros(
        (SEMANTIC_CLASS_COUNT, SEMANTIC_CLASS_COUNT), dtype=torch.int64
    )
    semantic_count = 0
    if domain_target is not None:
        if "domain_logits" not in endpoint:
            raise KeyError("endpoint is missing domain_logits")
        logits = torch.as_tensor(endpoint["domain_logits"]).detach().cpu()
        if logits.ndim == 4 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim != 3 or logits.shape[0] != SEMANTIC_CLASS_COUNT:
            raise ValueError("endpoint domain logits must have shape (3, H, W)")
        target = _scene_2d(domain_target, name="domain_target").long()
        if logits.shape[-2:] != target.shape:
            raise ValueError("semantic logits and target shapes differ")
        support = (target >= 0) & (target < SEMANTIC_CLASS_COUNT)
        if domain_valid_mask is not None:
            supplied = _scene_2d(
                domain_valid_mask, name="domain_valid_mask"
            ).bool()
            if supplied.shape != target.shape:
                raise ValueError("semantic valid mask and target shapes differ")
            support &= supplied
        semantic_count = int(torch.count_nonzero(support))
        if semantic_count:
            if not bool(torch.all(torch.isfinite(logits[:, support]))):
                raise ValueError(
                    "endpoint domain logits are non-finite on reference support"
                )
            prediction = logits.argmax(dim=0)
            flat = SEMANTIC_CLASS_COUNT * target[support] + prediction[support]
            confusion = torch.bincount(
                flat, minlength=SEMANTIC_CLASS_COUNT**2
            ).reshape(SEMANTIC_CLASS_COUNT, SEMANTIC_CLASS_COUNT)
    return {
        "height_sse": height_sse,
        "semantic_confusion_3x3": tuple(
            tuple(int(value) for value in row) for row in confusion.tolist()
        ),
    }


def build_stage4_dual_router_record(
    *,
    sample_id: str,
    descriptor: torch.Tensor | np.ndarray,
    protected_endpoint: Mapping[str, torch.Tensor],
    candidate_endpoint: Mapping[str, torch.Tensor],
    source: str,
    group_id: str,
    landscape: str,
    partition: str,
    height_target: torch.Tensor | np.ndarray | None = None,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    domain_target: torch.Tensor | np.ndarray | None = None,
    domain_valid_mask: torch.Tensor | np.ndarray | None = None,
    utility_config: SceneUtilityConfig | None = None,
) -> Stage4DualRouterRecord:
    """Measure both endpoints once and retain every Stage-4 target component."""

    aggregate, diagnostics = build_scene_router_record(
        sample_id=sample_id,
        descriptor=descriptor,
        fallback_endpoint=protected_endpoint,
        candidate_endpoint=candidate_endpoint,
        source=source,
        landscape=landscape,
        partition=partition,
        height_target=height_target,
        valid_mask=valid_mask,
        domain_target=domain_target,
        domain_valid_mask=domain_valid_mask,
        utility_config=utility_config,
    )
    exact = {
        "fallback": _exact_component_evidence(
            protected_endpoint,
            height_target=height_target,
            valid_mask=valid_mask,
            domain_target=domain_target,
            domain_valid_mask=domain_valid_mask,
        ),
        "candidate": _exact_component_evidence(
            candidate_endpoint,
            height_target=height_target,
            valid_mask=valid_mask,
            domain_target=domain_target,
            domain_valid_mask=domain_valid_mask,
        ),
    }
    return stage4_record_from_stage3_diagnostics(
        aggregate, diagnostics=diagnostics, exact_evidence=exact, group_id=group_id
    )


def stage4_record_from_stage3_diagnostics(
    record: SceneRouterRecord,
    *,
    diagnostics: Mapping[str, Mapping[str, object]],
    exact_evidence: Mapping[str, Mapping[str, object]],
    group_id: str,
) -> Stage4DualRouterRecord:
    """Convert in-memory Stage-3 build evidence before its components are lost."""

    if set(diagnostics) != {"fallback", "candidate"}:
        raise ValueError("endpoint diagnostics must contain fallback and candidate")
    if set(exact_evidence) != {"fallback", "candidate"}:
        raise ValueError("exact evidence must contain fallback and candidate")
    protected = EndpointComponentUtilities.from_mapping(
        {**diagnostics["fallback"], **exact_evidence["fallback"]}
    )
    candidate = EndpointComponentUtilities.from_mapping(
        {**diagnostics["candidate"], **exact_evidence["candidate"]}
    )
    if not math.isclose(
        protected.aggregate_utility, record.fallback_utility, rel_tol=0.0, abs_tol=1e-12
    ) or not math.isclose(
        candidate.aggregate_utility, record.candidate_utility, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("component diagnostics disagree with aggregate Stage-3 utility")
    return Stage4DualRouterRecord(
        sample_id=record.sample_id,
        descriptor=record.descriptor,
        protected=protected,
        candidate=candidate,
        source=record.source,
        group_id=group_id,
        landscape=record.landscape,
        partition=record.partition,
    )


def read_stage4_dual_router_records(path: str | Path) -> list[Stage4DualRouterRecord]:
    records: list[Stage4DualRouterRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("record must be a JSON object")
                records.append(Stage4DualRouterRecord.from_json_dict(value))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid Stage-4 record at line {line_number}: {error}"
                ) from error
    if not records:
        raise ValueError("Stage-4 record file is empty")
    if len({record.sample_id for record in records}) != len(records):
        raise ValueError("Stage-4 record file contains duplicate sample IDs")
    if len({record.descriptor.numel() for record in records}) != 1:
        raise ValueError("all Stage-4 descriptors must have the same size")
    return records


def write_stage4_dual_router_records(
    records: Sequence[Stage4DualRouterRecord], path: str | Path
) -> None:
    if not records:
        raise ValueError("cannot write an empty Stage-4 record file")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")


def partition_group_held_out_records(
    records: Sequence[Stage4DualRouterRecord],
    *,
    require_group_disjoint: bool = True,
) -> tuple[list[Stage4DualRouterRecord], list[Stage4DualRouterRecord], dict[str, object]]:
    """Return train/calibration records and prove source+group isolation."""

    train = [record for record in records if record.partition == "train"]
    calibration = [
        record
        for record in records
        if record.partition == "calibration"
    ]
    if not train or not calibration:
        raise ValueError("Stage-4 records require train and held-out calibration rows")
    train_groups = {record.group_key for record in train}
    calibration_groups = {record.group_key for record in calibration}
    overlap = sorted(train_groups & calibration_groups)
    if require_group_disjoint and overlap:
        raise ValueError(
            "source/group IDs cross train and calibration partitions: "
            + ", ".join(overlap[:5])
        )
    diagnostics = {
        "group_key": "source/group_id",
        "required_disjoint": bool(require_group_disjoint),
        "train_groups": sorted(train_groups),
        "calibration_groups": sorted(calibration_groups),
        "overlap": overlap,
    }
    return train, calibration, diagnostics


class Stage4GainDataset(Dataset[dict[str, Any]]):
    """Expose descriptors and one measured endpoint gain, never audit metadata."""

    def __init__(
        self, records: Sequence[Stage4DualRouterRecord], *, component: ComponentName
    ) -> None:
        if component not in {"height", "semantic"}:
            raise ValueError("component must be height or semantic")
        selected = [record for record in records if record.component_pair(component)]
        if not selected:
            raise ValueError(f"no records contain paired {component} endpoint utilities")
        if len({record.descriptor.numel() for record in selected}) != 1:
            raise ValueError("Stage-4 routing descriptors have inconsistent sizes")
        self.records = tuple(selected)
        self.component = component
        self.descriptor_size = self.records[0].descriptor.numel()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        pair = record.component_pair(self.component)
        assert pair is not None
        protected, candidate = pair
        result: dict[str, Any] = {
            "descriptor": record.descriptor.clone(),
            # These three values form one audited arithmetic identity. Keep
            # all of them in float64 so independently rounded endpoint values
            # cannot disagree with their stored difference at calibration.
            "target_gain": torch.tensor([protected - candidate], dtype=torch.float64),
            "protected_utility": torch.tensor([protected], dtype=torch.float64),
            "candidate_utility": torch.tensor([candidate], dtype=torch.float64),
            "sample_id": record.sample_id,
            "group": record.group_key,
            "source": record.source,
            "landscape": record.landscape,
            "source_landscape": record.stratum_key,
        }
        if self.component == "height":
            assert record.protected.height_sse is not None
            assert record.candidate.height_sse is not None
            result.update(
                {
                    "protected_exact": torch.tensor(
                        [record.protected.height_sse], dtype=torch.float64
                    ),
                    "candidate_exact": torch.tensor(
                        [record.candidate.height_sse], dtype=torch.float64
                    ),
                    "support_count": torch.tensor(
                        [record.protected.valid_height_pixels], dtype=torch.int64
                    ),
                }
            )
        else:
            result.update(
                {
                    "protected_exact": torch.tensor(
                        record.protected.semantic_confusion_3x3, dtype=torch.int64
                    ),
                    "candidate_exact": torch.tensor(
                        record.candidate.semantic_confusion_3x3, dtype=torch.int64
                    ),
                    "support_count": torch.tensor(
                        [record.protected.valid_semantic_pixels], dtype=torch.int64
                    ),
                }
            )
        return result


class BalancedStage4Sampler(Sampler[int]):
    """Balance source/landscape exposure without using targets or group IDs."""

    def __init__(self, records: Sequence[Stage4DualRouterRecord], *, seed: int) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            groups[record.stratum_key].append(index)
        if not groups:
            raise ValueError("balanced Stage-4 sampling requires records")
        self.groups = {key: tuple(value) for key, value in sorted(groups.items())}
        self.samples_per_stratum = max(len(value) for value in self.groups.values())
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.samples_per_stratum * len(self.groups)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws: list[int] = []
        for population in self.groups.values():
            stratum: list[int] = []
            while len(stratum) < self.samples_per_stratum:
                order = torch.randperm(len(population), generator=generator).tolist()
                stratum.extend(population[index] for index in order)
            draws.extend(stratum[: self.samples_per_stratum])
        order = torch.randperm(len(draws), generator=generator).tolist()
        return iter(draws[index] for index in order)


class EndpointGainRouter(nn.Module):
    """Tiny regression router predicting measured candidate gain in native units."""

    def __init__(
        self,
        descriptor_size: int,
        *,
        hidden_features: int = 16,
        target_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if descriptor_size <= 0 or hidden_features <= 0 or target_scale <= 0:
            raise ValueError("gain-router sizes and target scale must be positive")
        self.descriptor_size = int(descriptor_size)
        self.hidden_features = int(hidden_features)
        self.register_buffer("target_scale", torch.tensor(float(target_scale)))
        self.register_buffer("decision_threshold", torch.tensor(0.0))
        self.register_buffer("routing_enabled", torch.tensor(False))
        self.network = nn.Sequential(
            nn.Linear(self.descriptor_size, self.hidden_features),
            nn.GELU(),
            nn.Linear(self.hidden_features, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, -0.1)

    def predict_gain(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.ndim != 2 or descriptor.shape[1] != self.descriptor_size:
            raise ValueError(
                f"descriptor must have shape (batch, {self.descriptor_size})"
            )
        return self.network(descriptor.float()) * self.target_scale

    def forward_descriptor(self, descriptor: torch.Tensor) -> dict[str, torch.Tensor]:
        predicted = self.predict_gain(descriptor)
        selected = self.routing_enabled & (predicted >= self.decision_threshold)
        return {
            "predicted_gain": predicted,
            "candidate_selected": selected,
        }

    def set_operating_point(self, *, threshold: float, eligible: bool) -> None:
        if not math.isfinite(float(threshold)):
            raise ValueError("gain-router threshold must be finite")
        self.decision_threshold.fill_(float(threshold))
        self.routing_enabled.fill_(bool(eligible))


def compute_gain_route_metrics(
    predicted_gains: np.ndarray | torch.Tensor,
    actual_gains: np.ndarray | torch.Tensor,
    protected_utilities: np.ndarray | torch.Tensor,
    candidate_utilities: np.ndarray | torch.Tensor,
    *,
    threshold: float,
    minimum_candidate_gain: float,
) -> dict[str, float | int]:
    predicted = np.asarray(predicted_gains, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual_gains, dtype=np.float64).reshape(-1)
    protected = np.asarray(protected_utilities, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_utilities, dtype=np.float64).reshape(-1)
    if not predicted.size or not (
        predicted.shape == actual.shape == protected.shape == candidate.shape
    ):
        raise ValueError("gain and utility arrays must be equally sized and non-empty")
    if not all(np.all(np.isfinite(value)) for value in (predicted, actual, protected, candidate)):
        raise ValueError("gain and utility arrays must be finite")
    if minimum_candidate_gain < 0 or not math.isfinite(float(threshold)):
        raise ValueError("gain and threshold guards are invalid")
    if not np.allclose(actual, protected - candidate, rtol=1e-6, atol=1e-7):
        raise ValueError("actual gains disagree with endpoint utilities")
    beneficial = actual > minimum_candidate_gain
    selected = predicted >= threshold
    tp = int(np.count_nonzero(selected & beneficial))
    fp = int(np.count_nonzero(selected & ~beneficial))
    fn = int(np.count_nonzero(~selected & beneficial))
    tn = int(np.count_nonzero(~selected & ~beneficial))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    routed = np.where(selected, candidate, protected)
    return {
        "scene_count": int(predicted.size),
        "threshold": float(threshold),
        "candidate_selected": int(np.count_nonzero(selected)),
        "candidate_selection_rate": float(selected.mean()),
        "beneficial_candidate_count": int(np.count_nonzero(beneficial)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "prediction_mae": float(np.mean(np.abs(predicted - actual))),
        "prediction_rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
        "protected_mean_utility": float(protected.mean()),
        "candidate_mean_utility": float(candidate.mean()),
        "routed_mean_utility": float(routed.mean()),
        "routed_gain_vs_protected": float(protected.mean() - routed.mean()),
        "selected_candidate_mean_gain": float(actual[selected].mean() if np.any(selected) else 0.0),
    }


def _aggregate_exact_error(
    component: ComponentName,
    evidence: np.ndarray,
    support_counts: np.ndarray,
) -> float:
    counts = np.asarray(support_counts, dtype=np.int64).reshape(-1)
    if component == "height":
        squared_error = np.asarray(evidence, dtype=np.float64).reshape(-1)
        if squared_error.shape != counts.shape or np.any(squared_error < 0):
            raise ValueError("height SSE evidence has an invalid shape or value")
        count = int(counts.sum())
        if count <= 0:
            raise ValueError("height aggregation has no supported pixels")
        return float(math.sqrt(float(squared_error.sum()) / count))
    confusion = np.asarray(evidence, dtype=np.int64)
    if confusion.shape != (len(counts), SEMANTIC_CLASS_COUNT, SEMANTIC_CLASS_COUNT):
        raise ValueError("semantic evidence must be one 3x3 confusion matrix per scene")
    if np.any(confusion < 0) or not np.array_equal(
        confusion.sum(axis=(1, 2)), counts
    ):
        raise ValueError("semantic confusion evidence disagrees with support counts")
    combined = confusion.sum(axis=0)
    recalls = [
        float(combined[index, index] / combined[index].sum())
        for index in range(SEMANTIC_CLASS_COUNT)
        if int(combined[index].sum()) > 0
    ]
    if not recalls:
        raise ValueError("semantic aggregation has no supported classes")
    return float(1.0 - np.mean(recalls))


def compute_exact_gain_route_metrics(
    predicted_gains: np.ndarray | torch.Tensor,
    actual_gains: np.ndarray | torch.Tensor,
    protected_utilities: np.ndarray | torch.Tensor,
    candidate_utilities: np.ndarray | torch.Tensor,
    protected_exact: np.ndarray | torch.Tensor,
    candidate_exact: np.ndarray | torch.Tensor,
    support_counts: np.ndarray | torch.Tensor,
    *,
    component: ComponentName,
    threshold: float,
    minimum_candidate_gain: float,
) -> dict[str, float | int]:
    """Report exact pixel-aggregated RMSE or confusion-derived semantic error."""

    metrics = compute_gain_route_metrics(
        predicted_gains,
        actual_gains,
        protected_utilities,
        candidate_utilities,
        threshold=threshold,
        minimum_candidate_gain=minimum_candidate_gain,
    )
    predicted = np.asarray(predicted_gains, dtype=np.float64).reshape(-1)
    selected = predicted >= float(threshold)
    protected_evidence = np.asarray(protected_exact)
    candidate_evidence = np.asarray(candidate_exact)
    counts = np.asarray(support_counts, dtype=np.int64).reshape(-1)
    if len(counts) != len(predicted) or protected_evidence.shape != candidate_evidence.shape:
        raise ValueError("exact endpoint evidence does not match routed scenes")
    selector_shape = (len(selected),) + (1,) * (protected_evidence.ndim - 1)
    routed_evidence = np.where(
        selected.reshape(selector_shape), candidate_evidence, protected_evidence
    )
    protected_error = _aggregate_exact_error(component, protected_evidence, counts)
    candidate_error = _aggregate_exact_error(component, candidate_evidence, counts)
    routed_error = _aggregate_exact_error(component, routed_evidence, counts)
    metrics.update(
        {
            "protected_aggregate_error": protected_error,
            "candidate_aggregate_error": candidate_error,
            "routed_aggregate_error": routed_error,
            "routed_gain_vs_protected": protected_error - routed_error,
            "aggregate_error_kind": (
                "pixel_rmse_m" if component == "height" else "confusion_balanced_error"
            ),
            "supported_pixels": int(counts.sum()),
        }
    )
    return metrics


@dataclass(frozen=True)
class GroupGuardedThresholdResult:
    threshold: float
    eligible: bool
    reason: str
    metrics: dict[str, Any]
    evaluated_thresholds: int


def select_group_guarded_gain_threshold(
    predicted_gains: np.ndarray | torch.Tensor,
    actual_gains: np.ndarray | torch.Tensor,
    protected_utilities: np.ndarray | torch.Tensor,
    candidate_utilities: np.ndarray | torch.Tensor,
    protected_exact: np.ndarray | torch.Tensor,
    candidate_exact: np.ndarray | torch.Tensor,
    support_counts: np.ndarray | torch.Tensor,
    strata: Mapping[str, Sequence[str]] | Sequence[str],
    *,
    component: ComponentName,
    minimum_score: float = 0.0,
    minimum_candidate_gain: float = 0.0,
    minimum_precision: float = 0.98,
    minimum_candidate_selections: int = 32,
    minimum_stratum_support: int = 32,
    maximum_mean_utility_regression: float = 0.0,
    maximum_stratum_utility_regression: float = 0.0,
) -> GroupGuardedThresholdResult:
    """Calibrate one head; any unsafe stratum makes the threshold ineligible."""

    predicted = np.asarray(predicted_gains, dtype=np.float64).reshape(-1)
    actual = np.asarray(actual_gains, dtype=np.float64).reshape(-1)
    protected = np.asarray(protected_utilities, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_utilities, dtype=np.float64).reshape(-1)
    if isinstance(strata, Mapping):
        required_axes = {"source", "landscape", "source_landscape"}
        if set(strata) != required_axes:
            raise ValueError(
                "guard strata must contain source, landscape, and source_landscape"
            )
        axes = {
            axis: np.asarray(list(strata[axis]), dtype=object).reshape(-1)
            for axis in sorted(required_axes)
        }
    else:
        axes = {
            "source_landscape": np.asarray(list(strata), dtype=object).reshape(-1)
        }
    if any(
        labels.shape != predicted.shape
        or any(not str(value).strip() for value in labels)
        for labels in axes.values()
    ):
        raise ValueError("every calibration scene requires non-empty guard strata")
    if not math.isfinite(minimum_score) or minimum_candidate_gain < 0:
        raise ValueError("invalid gain threshold floor")
    if not 0.0 <= minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be within [0, 1]")
    if minimum_candidate_selections <= 0:
        raise ValueError("minimum_candidate_selections must be positive")
    if minimum_stratum_support <= 0:
        raise ValueError("minimum_stratum_support must be positive")
    if maximum_mean_utility_regression < 0 or maximum_stratum_utility_regression < 0:
        raise ValueError("utility regression allowances cannot be negative")
    if not predicted.size or np.any(~np.isfinite(predicted)):
        raise ValueError("predicted gains must be finite and non-empty")
    candidates = sorted(
        {
            float(np.float32(minimum_score)),
            *(
                float(np.float32(value))
                for value in predicted
                if value >= minimum_score
            ),
        }
    )
    eligible: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for threshold in candidates:
        threshold = float(np.float32(threshold))
        overall = compute_exact_gain_route_metrics(
            predicted,
            actual,
            protected,
            candidate,
            protected_exact,
            candidate_exact,
            support_counts,
            component=component,
            threshold=threshold,
            minimum_candidate_gain=minimum_candidate_gain,
        )
        stratum_metrics: dict[str, dict[str, dict[str, float | int | str | bool]]] = {}
        safe = True
        for axis, labels in axes.items():
            axis_metrics: dict[str, dict[str, float | int | str | bool]] = {}
            for name in sorted({str(value) for value in labels}):
                mask = labels == name
                metrics = compute_exact_gain_route_metrics(
                    predicted[mask],
                    actual[mask],
                    protected[mask],
                    candidate[mask],
                    np.asarray(protected_exact)[mask],
                    np.asarray(candidate_exact)[mask],
                    np.asarray(support_counts)[mask],
                    component=component,
                    threshold=threshold,
                    minimum_candidate_gain=minimum_candidate_gain,
                )
                guard_evaluated = int(metrics["scene_count"]) >= minimum_stratum_support
                metrics["non_regression_guard_evaluated"] = guard_evaluated
                axis_metrics[name] = metrics
                if guard_evaluated and (
                    float(metrics["routed_gain_vs_protected"])
                    < -maximum_stratum_utility_regression
                ):
                    safe = False
            stratum_metrics[axis] = axis_metrics
        combined: dict[str, Any] = {**overall, "strata": stratum_metrics}
        if int(overall["candidate_selected"]) < minimum_candidate_selections:
            continue
        if float(overall["precision"]) < minimum_precision:
            continue
        if (
            float(overall["routed_gain_vs_protected"])
            < -maximum_mean_utility_regression
            or not safe
        ):
            continue
        score = (
            float(overall["routed_gain_vs_protected"]),
            float(overall["precision"]),
            float(threshold),
        )
        eligible.append((score, combined))
    if eligible:
        _, metrics = max(eligible, key=lambda item: item[0])
        return GroupGuardedThresholdResult(
            threshold=float(metrics["threshold"]),
            eligible=True,
            reason=(
                "precision, selection, global non-regression, and every "
                "source/landscape non-regression guard passed"
            ),
            metrics=metrics,
            evaluated_thresholds=len(candidates),
        )
    fallback_threshold = float(
        np.nextafter(np.float32(predicted.max()), np.float32(math.inf))
    )
    fallback = compute_exact_gain_route_metrics(
        predicted,
        actual,
        protected,
        candidate,
        protected_exact,
        candidate_exact,
        support_counts,
        component=component,
        threshold=fallback_threshold,
        minimum_candidate_gain=minimum_candidate_gain,
    )
    fallback["strata"] = {}
    for axis, labels in axes.items():
        fallback["strata"][axis] = {}
        for name in sorted({str(value) for value in labels}):
            mask = labels == name
            metrics = compute_exact_gain_route_metrics(
                predicted[mask],
                actual[mask],
                protected[mask],
                candidate[mask],
                np.asarray(protected_exact)[mask],
                np.asarray(candidate_exact)[mask],
                np.asarray(support_counts)[mask],
                component=component,
                threshold=fallback_threshold,
                minimum_candidate_gain=minimum_candidate_gain,
            )
            metrics["non_regression_guard_evaluated"] = (
                int(metrics["scene_count"]) >= minimum_stratum_support
            )
            fallback["strata"][axis][name] = metrics
    return GroupGuardedThresholdResult(
        threshold=fallback_threshold,
        eligible=False,
        reason="no safe threshold passed; this head remains protected-only",
        metrics=fallback,
        evaluated_thresholds=len(candidates),
    )


@dataclass(frozen=True)
class GainRouterTrainingConfig:
    epochs: int = 30
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    huber_delta: float = 1.0
    minimum_score: float = 0.0
    minimum_candidate_gain: float = 0.0
    minimum_precision: float = 0.98
    minimum_candidate_selections: int = 32
    minimum_stratum_support: int = 32
    maximum_mean_utility_regression: float = 0.0
    maximum_stratum_utility_regression: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid Stage-4 optimizer configuration")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if self.minimum_candidate_selections <= 0 or self.minimum_stratum_support <= 0:
            raise ValueError("Stage-4 selection/support counts must be positive")


@dataclass(frozen=True)
class GainRouterTrainingResult:
    history: tuple[dict[str, float | int], ...]
    threshold: GroupGuardedThresholdResult
    calibration_gain_mae: float
    calibration_gain_rmse: float


def _collect_gain_validation(
    router: EndpointGainRouter,
    loader: DataLoader,
    device: torch.device,
    *,
    component: ComponentName,
) -> dict[str, Any]:
    router.eval()
    predicted: list[torch.Tensor] = []
    actual: list[torch.Tensor] = []
    protected: list[torch.Tensor] = []
    candidate: list[torch.Tensor] = []
    protected_exact: list[torch.Tensor] = []
    candidate_exact: list[torch.Tensor] = []
    support_counts: list[torch.Tensor] = []
    sources: list[str] = []
    landscapes: list[str] = []
    source_landscapes: list[str] = []
    groups: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            descriptor = batch["descriptor"].to(device)
            predicted.append(router.predict_gain(descriptor).cpu())
            actual.append(batch["target_gain"].double().cpu())
            protected.append(batch["protected_utility"].double().cpu())
            candidate.append(batch["candidate_utility"].double().cpu())
            protected_exact.append(batch["protected_exact"].cpu())
            candidate_exact.append(batch["candidate_exact"].cpu())
            support_counts.append(batch["support_count"].cpu())
            sources.extend(str(value) for value in batch["source"])
            landscapes.extend(str(value) for value in batch["landscape"])
            source_landscapes.extend(
                str(value) for value in batch["source_landscape"]
            )
            groups.extend(str(value) for value in batch["group"])
    if not predicted:
        raise ValueError("Stage-4 calibration loader produced no scenes")
    return {
        "predicted_gains": torch.cat(predicted).numpy(),
        "actual_gains": torch.cat(actual).numpy(),
        "protected_utilities": torch.cat(protected).numpy(),
        "candidate_utilities": torch.cat(candidate).numpy(),
        "protected_exact": torch.cat(protected_exact).numpy(),
        "candidate_exact": torch.cat(candidate_exact).numpy(),
        "support_counts": torch.cat(support_counts).numpy(),
        "strata": {
            "source": sources,
            "landscape": landscapes,
            "source_landscape": source_landscapes,
        },
        "groups": groups,
        "component": component,
    }


def train_gain_router(
    router: EndpointGainRouter,
    train_loader: DataLoader,
    calibration_loader: DataLoader,
    *,
    component: ComponentName,
    config: GainRouterTrainingConfig,
    device: str | torch.device = "cpu",
) -> GainRouterTrainingResult:
    """Fit one gain regressor and calibrate its guarded operating threshold."""

    device = torch.device(device)
    router.to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float | int]] = []
    scale = float(router.target_scale.item())
    for epoch in range(config.epochs):
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        router.train()
        loss_sum = 0.0
        scene_count = 0
        for batch in train_loader:
            descriptor = batch["descriptor"].to(device)
            # The network is deliberately float32, but audit utilities remain
            # float64 everywhere else. This is the only lossy target cast.
            target = batch["target_gain"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predicted = router.predict_gain(descriptor)
            loss = F.huber_loss(
                predicted / scale,
                target / scale,
                delta=config.huber_delta,
            )
            loss.backward()
            optimizer.step()
            count = int(descriptor.shape[0])
            loss_sum += float(loss.detach()) * count
            scene_count += count
        if not scene_count:
            raise ValueError("Stage-4 training loader produced no scenes")
        history.append(
            {
                "epoch": epoch + 1,
                "train_huber": loss_sum / scene_count,
            }
        )
    # The fixed final epoch is selected without observing calibration.  Read
    # the held-out calibration loader exactly once, only after fitting ends,
    # so the 0.98 precision threshold guard is not also a model-selection set.
    arrays = _collect_gain_validation(
        router, calibration_loader, device, component=component
    )
    calibration_error = arrays["predicted_gains"] - arrays["actual_gains"]
    calibration_gain_mae = float(np.mean(np.abs(calibration_error)))
    calibration_gain_rmse = float(
        np.sqrt(np.mean(np.square(calibration_error)))
    )
    threshold = select_group_guarded_gain_threshold(
        arrays["predicted_gains"],
        arrays["actual_gains"],
        arrays["protected_utilities"],
        arrays["candidate_utilities"],
        arrays["protected_exact"],
        arrays["candidate_exact"],
        arrays["support_counts"],
        arrays["strata"],
        component=component,
        minimum_score=config.minimum_score,
        minimum_candidate_gain=config.minimum_candidate_gain,
        minimum_precision=config.minimum_precision,
        minimum_candidate_selections=config.minimum_candidate_selections,
        minimum_stratum_support=config.minimum_stratum_support,
        maximum_mean_utility_regression=config.maximum_mean_utility_regression,
        maximum_stratum_utility_regression=(
            config.maximum_stratum_utility_regression
        ),
    )
    router.set_operating_point(
        threshold=threshold.threshold, eligible=threshold.eligible
    )
    router.eval()
    return GainRouterTrainingResult(
        history=tuple(history),
        threshold=threshold,
        calibration_gain_mae=calibration_gain_mae,
        calibration_gain_rmse=calibration_gain_rmse,
    )


__all__ = [
    "BalancedStage4Sampler",
    "EndpointComponentUtilities",
    "EndpointGainRouter",
    "GainRouterTrainingConfig",
    "GainRouterTrainingResult",
    "GroupGuardedThresholdResult",
    "STAGE4_ARTIFACT_SCHEMA",
    "STAGE4_RECORD_SCHEMA",
    "STAGE4_RECORD_STORE_SCHEMA",
    "STAGE4_REPORT_SCHEMA",
    "Stage4DualRouterRecord",
    "Stage4GainDataset",
    "build_stage4_dual_router_record",
    "compute_gain_route_metrics",
    "partition_group_held_out_records",
    "read_stage4_dual_router_records",
    "select_group_guarded_gain_threshold",
    "stage4_record_from_stage3_diagnostics",
    "train_gain_router",
    "write_stage4_dual_router_records",
]
