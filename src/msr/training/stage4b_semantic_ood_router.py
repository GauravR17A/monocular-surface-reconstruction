"""Stage-4b semantic-only domain/OOD endpoint routing.

This module is deliberately isolated from the Stage-3 and Stage-4 routers.  It
learns whether an all-pixel, deployable scene descriptor looks like the GAMUS
domain on which the candidate semantic endpoint is useful.  Source and group
metadata are supervision/audit fields only; they are never model inputs.

The controller is intentionally incapable of routing height.  A conservative
ensemble score, a shrinkage-Mahalanobis OOD check, and a class-distance margin
must all accept a scene before the semantic endpoint can be selected.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .scene_router import SceneRouterRecord
from .stage4_dual_router import (
    Stage4DualRouterRecord,
    compute_exact_gain_route_metrics,
)


STAGE4B_ARTIFACT_SCHEMA = "msr.stage4b_semantic_ood_router.v1"
STAGE4B_REPORT_SCHEMA = "msr.stage4b_semantic_ood_router_report.v1"
STAGE4B_COMPLETION_SCHEMA = "msr.stage4b_semantic_ood_router_completion.v1"
DESCRIPTOR_CONTRACT = "full_center_crop_all_pixels_no_reference_mask"
LOWER_ENSEMBLE_FORMULA = (
    "clamp(mean(member_probability) - z * population_std(member_probability), 0, 1)"
)
DEVELOPMENT_DISCLAIMER = (
    "Stage-4 v3 calibration is reused development evidence, not a fresh holdout; "
    "a new untouched geographic holdout is required before promotion."
)

_GAMUS_KEY = re.compile(r"^gamus/train/([A-Za-z0-9]+)_")


@dataclass(frozen=True)
class Stage4bDomainExample:
    """One Stage-3 TRAIN descriptor with an audit-only domain target."""

    sample_id: str
    descriptor: torch.Tensor
    candidate_domain: bool
    source: str
    landscape: str
    group_id: str

    def __post_init__(self) -> None:
        descriptor = torch.as_tensor(self.descriptor).detach().cpu().float()
        if descriptor.ndim != 1 or descriptor.numel() == 0:
            raise ValueError("Stage-4b descriptor must be a non-empty vector")
        if not bool(torch.all(torch.isfinite(descriptor))):
            raise ValueError("Stage-4b descriptor must contain only finite values")
        if self.source not in {"gamus", "legacy"}:
            raise ValueError("Stage-4b source must be gamus or legacy")
        if not self.sample_id or not self.group_id or not self.landscape:
            raise ValueError("Stage-4b audit metadata cannot be empty")
        if self.candidate_domain != (self.source == "gamus"):
            raise ValueError("candidate-domain target must be derived from source")
        object.__setattr__(self, "descriptor", descriptor.clone())

    @property
    def group_key(self) -> str:
        return f"{self.source}/{self.group_id}"


def gamus_city_group(sample_id: str) -> str:
    """Derive the exact GAMUS city without opening any source image/label."""

    match = _GAMUS_KEY.match(str(sample_id))
    if match is None:
        raise ValueError(f"GAMUS TRAIN key has no canonical city: {sample_id!r}")
    return f"gamus:{match.group(1).lower()}"


def build_stage4b_training_examples(
    stage3_records: Sequence[SceneRouterRecord],
    *,
    legacy_group_ids: Mapping[str, str],
    required_gamus_cities: Sequence[str] = ("dc", "nyc", "phl"),
) -> tuple[tuple[Stage4bDomainExample, ...], dict[str, Any]]:
    """Select only Stage-3 TRAIN rows and attach authenticated group IDs.

    ``legacy_group_ids`` must originate from the authenticated Stage-4 v3
    ancestry.  This avoids inventing an OpenCanopy spatial region from a chip
    filename, which does not carry the complete canonical region identity.
    """

    train = [record for record in stage3_records if record.partition == "train"]
    if not train:
        raise ValueError("Stage-4b requires Stage-3 TRAIN descriptors")
    if any("/test/" in record.sample_id.lower() for record in stage3_records):
        raise ValueError("official test keys are forbidden in Stage-4b inputs")
    descriptor_sizes = {int(record.descriptor.numel()) for record in train}
    if len(descriptor_sizes) != 1:
        raise ValueError("Stage-3 TRAIN descriptor sizes are inconsistent")
    seen: set[str] = set()
    examples: list[Stage4bDomainExample] = []
    missing_legacy: list[str] = []
    for record in train:
        if record.sample_id in seen:
            raise ValueError(f"duplicate Stage-3 key: {record.sample_id}")
        seen.add(record.sample_id)
        source = record.source.strip().lower()
        if source == "gamus":
            group_id = gamus_city_group(record.sample_id)
        elif source == "legacy":
            group_id = str(legacy_group_ids.get(record.sample_id, "")).strip().lower()
            if not group_id:
                missing_legacy.append(record.sample_id)
                continue
            if not group_id.startswith(("highbuild:", "opencanopy:")):
                raise ValueError(f"legacy group is not exact/authenticated: {group_id!r}")
        else:
            raise ValueError(f"unsupported Stage-4b source: {source!r}")
        examples.append(
            Stage4bDomainExample(
                sample_id=record.sample_id,
                descriptor=record.descriptor,
                candidate_domain=source == "gamus",
                source=source,
                landscape=record.landscape.strip().lower(),
                group_id=group_id,
            )
        )
    if missing_legacy:
        raise ValueError(
            "authenticated group IDs are missing for legacy TRAIN rows: "
            + ", ".join(missing_legacy[:5])
        )
    cities = sorted(
        {example.group_id.removeprefix("gamus:") for example in examples if example.source == "gamus"}
    )
    required = sorted(str(value).strip().lower() for value in required_gamus_cities)
    if cities != required:
        raise ValueError(
            f"Stage-4b GAMUS city inventory differs: expected {required}, found {cities}"
        )
    counts = Counter((example.source, example.group_id) for example in examples)
    return tuple(examples), {
        "record_count": len(examples),
        "descriptor_size": descriptor_sizes.pop(),
        "source_counts": dict(Counter(example.source for example in examples)),
        "landscape_counts": dict(Counter(example.landscape for example in examples)),
        "gamus_cities": cities,
        "group_count": len(counts),
        "stage3_partitions_consumed": ["train"],
        "official_test_used": False,
    }


@dataclass(frozen=True)
class GroupHeldOutFold:
    index: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]


def _balanced_group_bins(
    groups: Mapping[str, Sequence[int]], folds: int, *, seed: int
) -> list[list[str]]:
    """Greedily balance whole groups; a stable hash breaks equal-size ties."""

    bins: list[list[str]] = [[] for _ in range(folds)]
    loads = [0] * folds

    def tie_hash(name: str) -> str:
        return hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()

    ordered = sorted(groups, key=lambda name: (-len(groups[name]), tie_hash(name), name))
    for name in ordered:
        target = min(range(folds), key=lambda index: (loads[index], index))
        bins[target].append(name)
        loads[target] += len(groups[name])
    return bins


def make_group_held_out_folds(
    examples: Sequence[Stage4bDomainExample], *, folds: int = 3, seed: int = 0
) -> tuple[tuple[GroupHeldOutFold, ...], dict[str, Any]]:
    """Build three folds held out by GAMUS city and exact legacy region."""

    if folds != 3:
        raise ValueError("Stage-4b is contract-locked to three held-out folds")
    if not examples:
        raise ValueError("cannot split empty Stage-4b examples")
    groups: dict[str, list[int]] = defaultdict(list)
    group_landscape: dict[str, str] = {}
    for index, example in enumerate(examples):
        groups[example.group_key].append(index)
        prior = group_landscape.setdefault(example.group_key, example.landscape)
        if prior != example.landscape:
            raise ValueError("one geographic group spans multiple landscapes")
    gamus_groups = sorted(name for name in groups if name.startswith("gamus/gamus:"))
    if len(gamus_groups) != folds:
        raise ValueError("three-fold Stage-4b requires exactly three GAMUS cities")
    assigned: list[list[str]] = [[name] for name in gamus_groups]
    for landscape in sorted({group_landscape[name] for name in groups if name not in gamus_groups}):
        subset = {
            name: groups[name]
            for name in groups
            if name not in gamus_groups and group_landscape[name] == landscape
        }
        bins = _balanced_group_bins(subset, folds, seed=seed)
        for index, values in enumerate(bins):
            assigned[index].extend(values)
    all_indices = set(range(len(examples)))
    result: list[GroupHeldOutFold] = []
    seen_validation_groups: set[str] = set()
    for fold_index, validation_group_list in enumerate(assigned):
        validation_groups = set(validation_group_list)
        validation_indices = sorted(
            index for group in validation_groups for index in groups[group]
        )
        train_indices = sorted(all_indices - set(validation_indices))
        train_groups = {examples[index].group_key for index in train_indices}
        if not validation_indices or not train_indices or train_groups & validation_groups:
            raise ValueError("invalid Stage-4b group-held-out fold")
        validation_labels = {examples[index].candidate_domain for index in validation_indices}
        train_labels = {examples[index].candidate_domain for index in train_indices}
        if validation_labels != {False, True} or train_labels != {False, True}:
            raise ValueError("every Stage-4b fold must contain both domains")
        seen_validation_groups.update(validation_groups)
        result.append(
            GroupHeldOutFold(
                index=fold_index,
                train_indices=tuple(train_indices),
                validation_indices=tuple(validation_indices),
                train_groups=tuple(sorted(train_groups)),
                validation_groups=tuple(sorted(validation_groups)),
            )
        )
    if seen_validation_groups != set(groups):
        raise ValueError("every Stage-4b group must be held out exactly once")
    return tuple(result), {
        "folds": folds,
        "group_key": "source/authenticated_group_id",
        "all_groups_held_out_once": True,
        "zero_group_overlap_every_fold": True,
        "folds_detail": [
            {
                "index": fold.index,
                "train_count": len(fold.train_indices),
                "validation_count": len(fold.validation_indices),
                "train_groups": list(fold.train_groups),
                "validation_groups": list(fold.validation_groups),
                "overlap": [],
            }
            for fold in result
        ],
    }


class FitNormalizedLinearClassifier(nn.Module):
    """A regularized linear classifier with immutable fit-only normalization."""

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        scale = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
        if mean.shape != scale.shape or mean.numel() == 0 or bool(torch.any(scale <= 0)):
            raise ValueError("classifier normalizer has invalid mean/scale")
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_scale", scale.clone())
        self.linear = nn.Linear(mean.numel(), 1)

    @property
    def descriptor_size(self) -> int:
        return int(self.feature_mean.numel())

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.ndim != 2 or descriptor.shape[1] != self.descriptor_size:
            raise ValueError("classifier descriptor shape mismatch")
        if not bool(torch.all(torch.isfinite(descriptor))):
            raise ValueError("classifier descriptor contains non-finite values")
        normalized = (descriptor.float() - self.feature_mean) / self.feature_scale
        return self.linear(normalized).reshape(-1)


@dataclass(frozen=True)
class LinearTrainingConfig:
    epochs: int = 60
    learning_rate: float = 0.02
    weight_decay: float = 0.01
    batch_size: int = 256

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0 or self.weight_decay <= 0:
            raise ValueError("Stage-4b requires positive fixed training settings")
        if self.batch_size <= 0:
            raise ValueError("Stage-4b batch size must be positive")


def _fit_normalizer(descriptor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = descriptor.double().mean(dim=0).float()
    scale = descriptor.double().std(dim=0, unbiased=False).float().clamp_min(1.0e-6)
    return mean, scale


def train_linear_member(
    descriptors: torch.Tensor,
    targets: torch.Tensor,
    *,
    seed: int,
    config: LinearTrainingConfig,
) -> tuple[FitNormalizedLinearClassifier, tuple[dict[str, float | int], ...]]:
    """Fit one deterministic member; all statistics come from its fit split."""

    x = torch.as_tensor(descriptors).detach().cpu().float()
    y = torch.as_tensor(targets).detach().cpu().float().reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or set(y.tolist()) != {0.0, 1.0}:
        raise ValueError("linear member requires a two-class descriptor matrix")
    mean, scale = _fit_normalizer(x)
    torch.manual_seed(int(seed))
    model = FitNormalizedLinearClassifier(mean, scale)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    positive = int(torch.count_nonzero(y))
    negative = len(y) - positive
    positive_weight = torch.tensor(negative / positive, dtype=torch.float32)
    generator = torch.Generator().manual_seed(int(seed))
    history: list[dict[str, float | int]] = []
    for epoch in range(config.epochs):
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for start in range(0, len(x), config.batch_size):
            indices = order[start : start + config.batch_size]
            logits = model(x[indices])
            loss = F.binary_cross_entropy_with_logits(
                logits, y[indices], pos_weight=positive_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        history.append({"epoch": epoch + 1, "train_bce": total / len(x)})
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tuple(history)


@dataclass(frozen=True)
class MahalanobisFit:
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    gamus_center: torch.Tensor
    legacy_center: torch.Tensor
    precision: torch.Tensor

    def __post_init__(self) -> None:
        vectors = (
            self.feature_mean,
            self.feature_scale,
            self.gamus_center,
            self.legacy_center,
        )
        size = torch.as_tensor(vectors[0]).numel()
        if any(torch.as_tensor(value).numel() != size for value in vectors):
            raise ValueError("Mahalanobis vectors have inconsistent sizes")
        if torch.as_tensor(self.precision).shape != (size, size):
            raise ValueError("Mahalanobis precision has invalid shape")


def fit_shrinkage_mahalanobis(
    descriptors: torch.Tensor,
    targets: torch.Tensor,
    *,
    shrinkage: float,
) -> MahalanobisFit:
    """Fit a positive-definite pooled covariance using only supplied rows."""

    if not 0.0 < shrinkage <= 1.0:
        raise ValueError("Mahalanobis shrinkage must be within (0, 1]")
    x = torch.as_tensor(descriptors).detach().cpu().double()
    y = torch.as_tensor(targets).detach().cpu().bool().reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or set(y.tolist()) != {False, True}:
        raise ValueError("Mahalanobis fitting requires both domains")
    feature_mean = x.mean(dim=0)
    feature_scale = x.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    z = (x - feature_mean) / feature_scale
    gamus_center = z[y].mean(dim=0)
    legacy_center = z[~y].mean(dim=0)
    residual = torch.cat((z[y] - gamus_center, z[~y] - legacy_center), dim=0)
    covariance = residual.T @ residual / max(1, len(residual) - 2)
    spherical = torch.eye(x.shape[1], dtype=torch.float64) * (
        torch.trace(covariance) / x.shape[1]
    )
    covariance = (1.0 - shrinkage) * covariance + shrinkage * spherical
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = max(float(torch.trace(covariance) / x.shape[1]) * 1.0e-6, 1.0e-8)
    precision = (eigenvectors * eigenvalues.clamp_min(floor).reciprocal()) @ eigenvectors.T
    return MahalanobisFit(
        feature_mean=feature_mean.float(),
        feature_scale=feature_scale.float(),
        gamus_center=gamus_center.float(),
        legacy_center=legacy_center.float(),
        precision=precision.float(),
    )


def mahalanobis_distances(
    descriptors: torch.Tensor, fit: MahalanobisFit
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(descriptors).float()
    if x.ndim != 2 or x.shape[1] != fit.feature_mean.numel():
        raise ValueError("Mahalanobis descriptor shape mismatch")
    if not bool(torch.all(torch.isfinite(x))):
        raise ValueError("Mahalanobis descriptor contains non-finite values")
    z = (x - fit.feature_mean) / fit.feature_scale

    def distance(center: torch.Tensor) -> torch.Tensor:
        delta = z - center
        squared = torch.einsum("bi,ij,bj->b", delta, fit.precision, delta)
        return torch.sqrt(squared.clamp_min(0.0))

    return distance(fit.gamus_center), distance(fit.legacy_center)


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor keys, dtypes, shapes and bytes for audit reports."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = torch.as_tensor(state[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def mahalanobis_sha256(fit: MahalanobisFit) -> str:
    return state_sha256(
        {
            "feature_mean": fit.feature_mean,
            "feature_scale": fit.feature_scale,
            "gamus_center": fit.gamus_center,
            "legacy_center": fit.legacy_center,
            "precision": fit.precision,
        }
    )


def _higher_quantile(values: torch.Tensor, quantile: float) -> float:
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be within (0, 1]")
    ordered = torch.sort(torch.as_tensor(values).float().reshape(-1)).values
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def fit_ood_guard(
    descriptors: torch.Tensor,
    targets: torch.Tensor,
    folds: Sequence[GroupHeldOutFold],
    *,
    shrinkage: float,
    quantile: float,
    minimum_distance_margin: float = 0.0,
) -> tuple[MahalanobisFit, float, float, dict[str, Any]]:
    """Fit deployable OOD stats and conservative worst-fold thresholds."""

    x = torch.as_tensor(descriptors).float()
    y = torch.as_tensor(targets).bool().reshape(-1)
    fold_details: list[dict[str, Any]] = []
    cutoffs: list[float] = []
    legacy_margins: list[float] = []
    for fold in folds:
        train_index = torch.tensor(fold.train_indices, dtype=torch.long)
        validation_index = torch.tensor(fold.validation_indices, dtype=torch.long)
        fit = fit_shrinkage_mahalanobis(
            x[train_index], y[train_index], shrinkage=shrinkage
        )
        d_gamus, d_legacy = mahalanobis_distances(x[validation_index], fit)
        validation_y = y[validation_index]
        own = torch.where(validation_y, d_gamus, d_legacy)
        cutoff = _higher_quantile(own, quantile)
        margins = d_legacy - d_gamus
        maximum_legacy_margin = float(margins[~validation_y].max())
        cutoffs.append(cutoff)
        legacy_margins.append(maximum_legacy_margin)
        fold_details.append(
            {
                "fold": fold.index,
                "fit_state_sha256": mahalanobis_sha256(fit),
                "validation_count": len(validation_index),
                "own_distance_quantile": cutoff,
                "maximum_legacy_distance_margin": maximum_legacy_margin,
            }
        )
    cutoff = float(np.float32(max(cutoffs)))
    raw_margin = max(float(minimum_distance_margin), max(legacy_margins))
    margin = float(np.nextafter(np.float32(raw_margin), np.float32(math.inf)))
    full = fit_shrinkage_mahalanobis(x, y, shrinkage=shrinkage)
    return full, cutoff, margin, {
        "method": "fit-only shrinkage pooled Mahalanobis",
        "shrinkage": float(shrinkage),
        "cutoff_quantile": float(quantile),
        "cutoff_policy": "maximum held-out-fold own-class quantile",
        "distance_margin_policy": "fp32 nextafter(max held-out-fold legacy margin)",
        "ood_max_distance": cutoff,
        "minimum_gamus_distance_margin": margin,
        "deployment_fit_state_sha256": mahalanobis_sha256(full),
        "folds": fold_details,
    }


class Stage4bSemanticOODRouter(nn.Module):
    """Fifteen-member semantic controller with hard-disabled height routing."""

    def __init__(
        self,
        members: Sequence[FitNormalizedLinearClassifier],
        ood_fit: MahalanobisFit,
        *,
        ensemble_std_multiplier: float,
        ood_max_distance: float,
        minimum_distance_margin: float,
    ) -> None:
        super().__init__()
        if not members or len({member.descriptor_size for member in members}) != 1:
            raise ValueError("Stage-4b ensemble members are empty or incompatible")
        if members[0].descriptor_size != ood_fit.feature_mean.numel():
            raise ValueError("Stage-4b classifier and OOD descriptor sizes differ")
        if ensemble_std_multiplier < 0 or ood_max_distance <= 0:
            raise ValueError("Stage-4b conservative score/OOD settings are invalid")
        self.members = nn.ModuleList(members)
        self.register_buffer("ensemble_std_multiplier", torch.tensor(float(ensemble_std_multiplier)))
        self.register_buffer("ood_feature_mean", ood_fit.feature_mean.clone())
        self.register_buffer("ood_feature_scale", ood_fit.feature_scale.clone())
        self.register_buffer("ood_gamus_center", ood_fit.gamus_center.clone())
        self.register_buffer("ood_legacy_center", ood_fit.legacy_center.clone())
        self.register_buffer("ood_precision", ood_fit.precision.clone())
        self.register_buffer("ood_max_distance", torch.tensor(float(ood_max_distance)))
        self.register_buffer("minimum_distance_margin", torch.tensor(float(minimum_distance_margin)))
        self.register_buffer("semantic_decision_threshold", torch.tensor(1.0))
        self.register_buffer("semantic_route_enabled", torch.tensor(False))

    @property
    def descriptor_size(self) -> int:
        return self.members[0].descriptor_size

    @property
    def height_route_enabled(self) -> bool:
        # Deliberately not a mutable tensor/buffer: Stage-4b can never route height.
        return False

    def _distances(self, descriptor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fit = MahalanobisFit(
            self.ood_feature_mean,
            self.ood_feature_scale,
            self.ood_gamus_center,
            self.ood_legacy_center,
            self.ood_precision,
        )
        return mahalanobis_distances(descriptor, fit)

    def score_descriptor(self, descriptor: torch.Tensor) -> dict[str, torch.Tensor]:
        if descriptor.ndim != 2 or descriptor.shape[1] != self.descriptor_size:
            raise ValueError("Stage-4b descriptor shape mismatch")
        probabilities = torch.stack(
            [torch.sigmoid(member(descriptor)) for member in self.members], dim=1
        )
        mean = probabilities.mean(dim=1)
        std = probabilities.std(dim=1, unbiased=False)
        lower = (mean - self.ensemble_std_multiplier * std).clamp(0.0, 1.0)
        d_gamus, d_legacy = self._distances(descriptor)
        nearest = torch.minimum(d_gamus, d_legacy)
        margin = d_legacy - d_gamus
        ood_accepted = nearest <= self.ood_max_distance
        margin_accepted = margin >= self.minimum_distance_margin
        return {
            "member_probabilities": probabilities,
            "mean_probability": mean,
            "probability_std": std,
            "lower_probability": lower,
            "gamus_distance": d_gamus,
            "legacy_distance": d_legacy,
            "nearest_distance": nearest,
            "distance_margin": margin,
            "ood_accepted": ood_accepted,
            "distance_margin_accepted": margin_accepted,
        }

    def forward_descriptor(self, descriptor: torch.Tensor) -> dict[str, torch.Tensor]:
        score = self.score_descriptor(descriptor)
        selected = (
            self.semantic_route_enabled
            & score["ood_accepted"]
            & score["distance_margin_accepted"]
            & (score["lower_probability"] >= self.semantic_decision_threshold)
        )
        return {
            **score,
            "semantic_candidate_selected": selected,
            "height_candidate_selected": torch.zeros_like(selected, dtype=torch.bool),
        }

    def set_semantic_operating_point(self, *, threshold: float, enabled: bool) -> None:
        if not math.isfinite(float(threshold)):
            raise ValueError("semantic threshold must be finite")
        self.semantic_decision_threshold.fill_(float(np.float32(threshold)))
        self.semantic_route_enabled.fill_(bool(enabled))


@dataclass(frozen=True)
class DevelopmentGateConfig:
    minimum_gamus_coverage: float = 0.85
    minimum_gamus_selections: int = 400
    minimum_global_exact_gain: float = 0.05
    minimum_gamus_exact_gain: float = 0.08
    minimum_stratum_support: int = 32
    minimum_decision_consistency: float = 0.995
    minimum_candidate_gain_for_reporting: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_gamus_coverage <= 1.0:
            raise ValueError("GAMUS coverage gate must be within (0, 1]")
        if self.minimum_gamus_selections < 400:
            raise ValueError("Stage-4b requires at least 400 GAMUS selections")
        if self.minimum_global_exact_gain < 0.05 or self.minimum_gamus_exact_gain < 0.08:
            raise ValueError("Stage-4b semantic exact-gain gates cannot be relaxed")
        if self.minimum_stratum_support < 32:
            raise ValueError("Stage-4b stratum support cannot be below 32")
        if not 0.0 <= self.minimum_decision_consistency <= 1.0:
            raise ValueError("decision consistency gate must be within [0, 1]")


@dataclass(frozen=True)
class DevelopmentOperatingPoint:
    threshold: float
    eligible: bool
    reason: str
    metrics: dict[str, Any]
    evaluated_thresholds: int


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _semantic_arrays(records: Sequence[Stage4DualRouterRecord]) -> dict[str, Any]:
    selected = [record for record in records if record.component_pair("semantic") is not None]
    if not selected:
        raise ValueError("development proof has no semantic endpoint evidence")
    protected = np.asarray([record.protected.semantic_balanced_error for record in selected], dtype=np.float64)
    candidate = np.asarray([record.candidate.semantic_balanced_error for record in selected], dtype=np.float64)
    return {
        "records": selected,
        "actual": protected - candidate,
        "protected": protected,
        "candidate": candidate,
        "protected_exact": np.asarray([record.protected.semantic_confusion_3x3 for record in selected], dtype=np.int64),
        "candidate_exact": np.asarray([record.candidate.semantic_confusion_3x3 for record in selected], dtype=np.int64),
        "support": np.asarray([record.protected.valid_semantic_pixels for record in selected], dtype=np.int64),
        "source": np.asarray([record.source for record in selected], dtype=object),
        "landscape": np.asarray([record.landscape for record in selected], dtype=object),
        "source_landscape": np.asarray([record.stratum_key for record in selected], dtype=object),
    }


def _subset_arrays(values: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {
        key: (value[mask] if isinstance(value, np.ndarray) and len(value) == len(mask) else value)
        for key, value in values.items()
        if key != "records"
    }


def _exact_metrics(values: Mapping[str, Any], score: np.ndarray, threshold: float, config: DevelopmentGateConfig) -> dict[str, Any]:
    return compute_exact_gain_route_metrics(
        score,
        values["actual"],
        values["protected"],
        values["candidate"],
        values["protected_exact"],
        values["candidate_exact"],
        values["support"],
        component="semantic",
        threshold=threshold,
        minimum_candidate_gain=config.minimum_candidate_gain_for_reporting,
    )


def _jackknife_consistency(
    probabilities: np.ndarray,
    *,
    std_multiplier: float,
    threshold: float,
    accepted: np.ndarray,
) -> float:
    if probabilities.ndim != 2 or probabilities.shape[1] < 3:
        raise ValueError("decision consistency requires at least three members")
    full = probabilities.mean(axis=1) - std_multiplier * probabilities.std(axis=1)
    reference = accepted & (full >= threshold)
    matches = 0
    total = 0
    for index in range(probabilities.shape[1]):
        reduced = np.delete(probabilities, index, axis=1)
        decision = accepted & (
            reduced.mean(axis=1) - std_multiplier * reduced.std(axis=1) >= threshold
        )
        matches += int(np.count_nonzero(decision == reference))
        total += len(reference)
    return matches / total


def select_development_operating_point(
    records: Sequence[Stage4DualRouterRecord],
    *,
    lower_probabilities: np.ndarray | torch.Tensor,
    member_probabilities: np.ndarray | torch.Tensor,
    ood_accepted: np.ndarray | torch.Tensor,
    margin_accepted: np.ndarray | torch.Tensor,
    std_multiplier: float,
    config: DevelopmentGateConfig,
) -> DevelopmentOperatingPoint:
    """Choose a fail-closed semantic threshold on reused development evidence."""

    values = _semantic_arrays(records)
    count = len(values["actual"])
    lower = np.asarray(lower_probabilities, dtype=np.float64).reshape(-1)
    members = np.asarray(member_probabilities, dtype=np.float64)
    ood = np.asarray(ood_accepted, dtype=bool).reshape(-1)
    margin = np.asarray(margin_accepted, dtype=bool).reshape(-1)
    if lower.shape != (count,) or members.shape[0] != count or ood.shape != (count,) or margin.shape != (count,):
        raise ValueError("Stage-4b development scores differ from semantic records")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(members)):
        raise ValueError("Stage-4b development probabilities must be finite")
    accepted = ood & margin
    decision_score = np.where(accepted, lower, -1.0)
    thresholds = sorted(
        {
            float(np.float32(0.0)),
            *(float(np.float32(value)) for value in lower[accepted]),
        }
    )
    eligible: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    for threshold in thresholds:
        threshold = float(np.float32(threshold))
        overall = _exact_metrics(values, decision_score, threshold, config)
        selected = decision_score >= threshold
        gamus = values["source"] == "gamus"
        legacy = values["source"] == "legacy"
        gamus_selected = int(np.count_nonzero(selected & gamus))
        legacy_selected = int(np.count_nonzero(selected & legacy))
        coverage = gamus_selected / int(np.count_nonzero(gamus))
        source_ci = {
            "gamus_coverage_95ci": _wilson_interval(gamus_selected, int(np.count_nonzero(gamus))),
            "legacy_selection_rate_95ci": _wilson_interval(legacy_selected, int(np.count_nonzero(legacy))),
        }
        stratum_metrics: dict[str, dict[str, Any]] = {}
        safe_strata = True
        for axis in ("source", "landscape", "source_landscape"):
            labels = values[axis]
            stratum_metrics[axis] = {}
            for name in sorted({str(item) for item in labels}):
                mask = labels == name
                subset = _subset_arrays(values, mask)
                metric = _exact_metrics(subset, decision_score[mask], threshold, config)
                guarded = int(metric["scene_count"]) >= config.minimum_stratum_support
                metric["non_regression_guard_evaluated"] = guarded
                if guarded and float(metric["routed_gain_vs_protected"]) < 0.0:
                    safe_strata = False
                stratum_metrics[axis][name] = metric
        gamus_metrics = stratum_metrics["source"]["gamus"]
        consistency = _jackknife_consistency(
            members,
            std_multiplier=std_multiplier,
            threshold=threshold,
            accepted=accepted,
        )
        metrics = {
            **overall,
            "threshold": threshold,
            "gamus_scene_count": int(np.count_nonzero(gamus)),
            "legacy_scene_count": int(np.count_nonzero(legacy)),
            "gamus_selected": gamus_selected,
            "legacy_selected": legacy_selected,
            "gamus_coverage": coverage,
            "selected_source_confusion": {
                "selected_gamus": gamus_selected,
                "selected_legacy": legacy_selected,
                "rejected_gamus": int(np.count_nonzero(gamus)) - gamus_selected,
                "rejected_legacy": int(np.count_nonzero(legacy)) - legacy_selected,
            },
            **source_ci,
            "gamus_exact_gain": float(gamus_metrics["routed_gain_vs_protected"]),
            "jackknife_decision_consistency": consistency,
            "strata": stratum_metrics,
            "ood_accepted": int(np.count_nonzero(ood)),
            "distance_margin_accepted": int(np.count_nonzero(margin)),
            "development_evidence_reused": True,
            "fresh_holdout_required": True,
            "development_disclaimer": DEVELOPMENT_DISCLAIMER,
        }
        checks = {
            "zero_legacy_selections": legacy_selected == 0,
            "minimum_gamus_coverage": coverage >= config.minimum_gamus_coverage,
            "minimum_gamus_selections": gamus_selected >= config.minimum_gamus_selections,
            "minimum_global_exact_gain": float(overall["routed_gain_vs_protected"]) >= config.minimum_global_exact_gain,
            "minimum_gamus_exact_gain": float(gamus_metrics["routed_gain_vs_protected"]) >= config.minimum_gamus_exact_gain,
            "zero_supported_stratum_regression": safe_strata,
            "decision_consistency": consistency >= config.minimum_decision_consistency,
        }
        metrics["gate_checks"] = checks
        if all(checks.values()):
            score = (
                float(overall["routed_gain_vs_protected"]),
                coverage,
                threshold,
            )
            eligible.append((score, metrics))
    if eligible:
        _, metrics = max(eligible, key=lambda item: item[0])
        return DevelopmentOperatingPoint(
            threshold=float(metrics["threshold"]),
            eligible=True,
            reason="all semantic domain/OOD development gates passed",
            metrics=metrics,
            evaluated_thresholds=len(thresholds),
        )
    fallback = float(np.nextafter(np.float32(lower.max()), np.float32(math.inf)))
    fallback_score = np.full(count, -1.0, dtype=np.float64)
    metrics = _exact_metrics(values, fallback_score, fallback, config)
    metrics.update(
        {
            "gamus_selected": 0,
            "legacy_selected": 0,
            "gamus_coverage": 0.0,
            "development_evidence_reused": True,
            "fresh_holdout_required": True,
            "development_disclaimer": DEVELOPMENT_DISCLAIMER,
        }
    )
    return DevelopmentOperatingPoint(
        threshold=fallback,
        eligible=False,
        reason="no semantic threshold passed every domain/OOD development gate",
        metrics=metrics,
        evaluated_thresholds=len(thresholds),
    )


__all__ = [
    "DESCRIPTOR_CONTRACT",
    "DEVELOPMENT_DISCLAIMER",
    "DevelopmentGateConfig",
    "DevelopmentOperatingPoint",
    "FitNormalizedLinearClassifier",
    "GroupHeldOutFold",
    "LOWER_ENSEMBLE_FORMULA",
    "LinearTrainingConfig",
    "MahalanobisFit",
    "STAGE4B_ARTIFACT_SCHEMA",
    "STAGE4B_COMPLETION_SCHEMA",
    "STAGE4B_REPORT_SCHEMA",
    "Stage4bDomainExample",
    "Stage4bSemanticOODRouter",
    "build_stage4b_training_examples",
    "fit_ood_guard",
    "fit_shrinkage_mahalanobis",
    "gamus_city_group",
    "mahalanobis_distances",
    "make_group_held_out_folds",
    "select_development_operating_point",
    "state_sha256",
    "train_linear_member",
]
