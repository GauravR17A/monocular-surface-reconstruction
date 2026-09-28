"""Deterministic additive-logit calibration for dense multiclass decisions.

This module calibrates only the class chosen by ``argmax(logits + bias)``.  It
does not alter a model, does not alter its logits, and does not make softmax
scores calibrated probabilities of correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .classification_metrics import compute_multiclass_metrics


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def centered_biases(
    biases: Sequence[float], *, class_count: int, maximum_absolute_bias: float
) -> np.ndarray:
    """Return a mean-zero bias vector and enforce the declared search bound."""

    values = np.asarray(biases, dtype=np.float64)
    if values.shape != (class_count,) or not np.all(np.isfinite(values)):
        raise ValueError(f"biases must contain {class_count} finite values")
    if not math.isfinite(maximum_absolute_bias) or maximum_absolute_bias <= 0.0:
        raise ValueError("maximum_absolute_bias must be finite and positive")
    values = values - values.mean()
    if float(np.max(np.abs(values))) > maximum_absolute_bias + 1.0e-12:
        raise ValueError("centered decision biases exceed the configured bound")
    return values


def confusion_from_logits(
    logits: np.ndarray,
    target: np.ndarray,
    biases: Sequence[float],
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Create a row-reference confusion matrix without modifying ``logits``."""

    scores = np.asarray(logits)
    labels = np.asarray(target)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("logits must have shape (N, C) with N > 0 and C >= 2")
    if labels.shape != (scores.shape[0],):
        raise ValueError("target must have shape (N,)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits contain non-finite values")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.rint(labels)):
            raise ValueError("targets must contain integer class indices")
    labels = labels.astype(np.int64, copy=False)
    class_count = scores.shape[1]
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("target contains an out-of-range class index")
    offsets = np.asarray(biases, dtype=np.float64)
    if offsets.shape != (class_count,) or not np.all(np.isfinite(offsets)):
        raise ValueError(f"biases must contain {class_count} finite values")
    prediction = np.argmax(scores + offsets[None, :], axis=1)
    encoded = labels * class_count + prediction
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != labels.shape or not np.all(np.isfinite(weights)):
            raise ValueError("sample_weight must contain one finite value per target")
        if np.any(weights <= 0.0):
            raise ValueError("sample_weight values must be positive")
    counts = np.bincount(
        encoded, weights=weights, minlength=class_count**2
    ).reshape(class_count, class_count)
    # Downstream metrics intentionally use integer pixel counts.  Weighted
    # bounded samples estimate the original per-class support, so rounding is
    # both deterministic and substantially more faithful than pretending the
    # class-balanced cache has an equal-prior deployment distribution.
    return np.rint(counts).astype(np.int64)


def metrics_from_logits(
    logits: np.ndarray,
    target: np.ndarray,
    biases: Sequence[float],
    class_names: Sequence[str],
    sample_weight: np.ndarray | None = None,
) -> dict[str, object]:
    names = tuple(str(name) for name in class_names)
    if len(names) != np.asarray(logits).shape[1]:
        raise ValueError("class_names and logit channels disagree")
    return compute_multiclass_metrics(
        confusion_from_logits(logits, target, biases, sample_weight), names
    )


def _metric_value(
    metrics: Mapping[str, Any], metric_name: str, class_name: str | None = None
) -> float:
    if class_name is None:
        return _finite_number(metrics.get(metric_name), metric_name)
    per_class = metrics.get("per_class")
    if not isinstance(per_class, Mapping) or class_name not in per_class:
        raise ValueError(f"metrics do not contain class {class_name!r}")
    values = per_class[class_name]
    if not isinstance(values, Mapping):
        raise ValueError(f"metrics for {class_name!r} must be a mapping")
    return _finite_number(values.get(metric_name), f"{class_name} {metric_name}")


def decision_gate_report(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    gate_config: Mapping[str, Any],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Evaluate absolute floors and retention gates declared before fitting."""

    names = tuple(str(name) for name in class_names)
    absolute = gate_config.get("absolute_minimums")
    retention = gate_config.get("maximum_drops_from_uncalibrated")
    if not isinstance(absolute, Mapping) or not isinstance(retention, Mapping):
        raise ValueError("gate config needs absolute_minimums and maximum_drops_from_uncalibrated")

    checks: dict[str, dict[str, Any]] = {}

    def minimum(key: str, value: float, limit: float) -> None:
        if not 0.0 <= limit <= 1.0:
            raise ValueError(f"minimum gate {key} must be in [0, 1]")
        checks[key] = {
            "value": value,
            "minimum": limit,
            "margin": value - limit,
            "passes": value >= limit,
        }

    for metric_name, raw_limit in absolute.items():
        if metric_name == "per_class":
            continue
        minimum(
            str(metric_name),
            _metric_value(candidate, str(metric_name)),
            _finite_number(raw_limit, f"absolute {metric_name}"),
        )
    absolute_classes = absolute.get("per_class", {})
    if not isinstance(absolute_classes, Mapping):
        raise ValueError("absolute per_class gates must be a mapping")
    for class_name, raw_metrics in absolute_classes.items():
        if class_name not in names or not isinstance(raw_metrics, Mapping):
            raise ValueError(f"invalid absolute class gate {class_name!r}")
        for metric_name, raw_limit in raw_metrics.items():
            minimum(
                f"{class_name}_{metric_name}_minimum",
                _metric_value(candidate, str(metric_name), str(class_name)),
                _finite_number(raw_limit, f"absolute {class_name} {metric_name}"),
            )

    def maximum_drop(
        key: str, value: float, reference: float, allowed_drop: float
    ) -> None:
        if not 0.0 <= allowed_drop <= 1.0:
            raise ValueError(f"maximum drop gate {key} must be in [0, 1]")
        floor = reference - allowed_drop
        checks[key] = {
            "value": value,
            "uncalibrated_value": reference,
            "maximum_drop": allowed_drop,
            "minimum_retained_value": floor,
            "margin": value - floor,
            "passes": value >= floor,
        }

    for metric_name, raw_drop in retention.items():
        if metric_name == "per_class":
            continue
        maximum_drop(
            f"{metric_name}_retention",
            _metric_value(candidate, str(metric_name)),
            _metric_value(baseline, str(metric_name)),
            _finite_number(raw_drop, f"retention {metric_name}"),
        )
    retention_classes = retention.get("per_class", {})
    if not isinstance(retention_classes, Mapping):
        raise ValueError("retention per_class gates must be a mapping")
    for class_name, raw_metrics in retention_classes.items():
        if class_name not in names or not isinstance(raw_metrics, Mapping):
            raise ValueError(f"invalid retention class gate {class_name!r}")
        for metric_name, raw_drop in raw_metrics.items():
            maximum_drop(
                f"{class_name}_{metric_name}_retention",
                _metric_value(candidate, str(metric_name), str(class_name)),
                _metric_value(baseline, str(metric_name), str(class_name)),
                _finite_number(raw_drop, f"retention {class_name} {metric_name}"),
            )

    required_gain = _finite_number(
        gate_config.get("minimum_objective_gain", 0.0),
        "minimum_objective_gain",
    )
    if required_gain < 0.0:
        raise ValueError("minimum_objective_gain cannot be negative")
    objective_name = str(gate_config.get("objective", "macro_f1"))
    objective = _metric_value(candidate, objective_name)
    baseline_objective = _metric_value(baseline, objective_name)
    checks["objective_gain"] = {
        "value": objective - baseline_objective,
        "minimum": required_gain,
        "margin": objective - baseline_objective - required_gain,
        "passes": objective - baseline_objective >= required_gain,
    }
    return {
        "passes_all": all(bool(item["passes"]) for item in checks.values()),
        "objective": objective_name,
        "objective_value": objective,
        "uncalibrated_objective_value": baseline_objective,
        "checks": checks,
    }


def _rank_gate_report(report: Mapping[str, Any], bias: np.ndarray) -> tuple[Any, ...]:
    raw_checks = report.get("checks")
    if not isinstance(raw_checks, Mapping):
        raise ValueError("gate report has no checks")
    checks = tuple(raw_checks.values())
    passed = sum(bool(check["passes"]) for check in checks)
    violation = sum(max(0.0, -float(check["margin"])) for check in checks)
    return (
        bool(report.get("passes_all")),
        passed,
        -violation,
        float(report["objective_value"]),
        -float(np.linalg.norm(bias)),
        tuple(float(-abs(value)) for value in bias),
    )


def fit_additive_decision_biases(
    logits: np.ndarray,
    target: np.ndarray,
    class_names: Sequence[str],
    *,
    gate_config: Mapping[str, Any],
    maximum_absolute_bias: float,
    step_schedule: Sequence[float],
    maximum_sweeps_per_step: int,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit six external argmax offsets with deterministic coordinate search."""

    scores = np.asarray(logits, dtype=np.float32)
    names = tuple(str(name) for name in class_names)
    class_count = len(names)
    if scores.ndim != 2 or scores.shape[1] != class_count:
        raise ValueError("logits and class_names disagree")
    if maximum_sweeps_per_step <= 0:
        raise ValueError("maximum_sweeps_per_step must be positive")
    steps = tuple(float(step) for step in step_schedule)
    if not steps or any(not math.isfinite(step) or step <= 0.0 for step in steps):
        raise ValueError("step_schedule must contain positive finite values")
    if any(next_step >= step for step, next_step in zip(steps, steps[1:])):
        raise ValueError("step_schedule must be strictly decreasing")

    zero = np.zeros(class_count, dtype=np.float64)
    baseline = metrics_from_logits(scores, target, zero, names, sample_weight)
    current_bias = zero
    current_metrics = baseline
    current_report = decision_gate_report(
        current_metrics, baseline, gate_config, names
    )
    evaluations = 1

    for step in steps:
        for _ in range(maximum_sweeps_per_step):
            best_bias = current_bias
            best_metrics = current_metrics
            best_report = current_report
            best_rank = _rank_gate_report(best_report, best_bias)
            for class_index in range(class_count):
                for direction in (-1.0, 1.0):
                    proposal = current_bias.copy()
                    proposal[class_index] += direction * step
                    proposal -= proposal.mean()
                    if float(np.max(np.abs(proposal))) > maximum_absolute_bias + 1e-12:
                        continue
                    metrics = metrics_from_logits(
                        scores, target, proposal, names, sample_weight
                    )
                    report = decision_gate_report(metrics, baseline, gate_config, names)
                    evaluations += 1
                    rank = _rank_gate_report(report, proposal)
                    if rank > best_rank:
                        best_bias = proposal
                        best_metrics = metrics
                        best_report = report
                        best_rank = rank
            if np.array_equal(best_bias, current_bias):
                break
            current_bias = best_bias
            current_metrics = best_metrics
            current_report = best_report

    current_bias = centered_biases(
        current_bias,
        class_count=class_count,
        maximum_absolute_bias=maximum_absolute_bias,
    )
    return {
        "biases": [float(value) for value in current_bias],
        "biases_by_class": {
            name: float(current_bias[index]) for index, name in enumerate(names)
        },
        "uncalibrated_metrics": baseline,
        "calibrated_metrics": current_metrics,
        "gate_report": current_report,
        "search_evaluations": evaluations,
        "identifiability_constraint": "mean_zero",
        "decision_rule": "argmax(raw_logits + additive_biases)",
        "changes_model_or_height_tensors": False,
        "probability_confidence_calibration_performed": False,
    }


@dataclass
class _Reservoir:
    priority: np.ndarray
    logits: np.ndarray
    seen: int = 0


class DeterministicBoundedClassSampler:
    """Keep a deterministic, hard-bounded, class-balanced sample of logits."""

    def __init__(
        self,
        class_names: Sequence[str],
        *,
        seed: int,
        per_tile_class_cap: int,
        maximum_pixels_per_class: int,
    ) -> None:
        self.class_names = tuple(str(name) for name in class_names)
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(
            self.class_names
        ):
            raise ValueError("class_names must contain unique class labels")
        if per_tile_class_cap <= 0 or maximum_pixels_per_class <= 0:
            raise ValueError("sampling caps must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed = seed
        self.per_tile_class_cap = per_tile_class_cap
        self.maximum_pixels_per_class = maximum_pixels_per_class
        class_count = len(self.class_names)
        self._items = [
            _Reservoir(
                priority=np.empty((0,), dtype=np.float64),
                logits=np.empty((0, class_count), dtype=np.float32),
            )
            for _ in self.class_names
        ]
        self.tiles_seen = 0

    def _rng(self, sample_id: str, class_index: int) -> np.random.Generator:
        material = f"{self.seed}\0{sample_id}\0{class_index}".encode("utf-8")
        numeric = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        return np.random.default_rng(numeric)

    def update(
        self,
        logits: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray,
        *,
        sample_id: str,
    ) -> None:
        scores = np.asarray(logits, dtype=np.float32)
        labels = np.asarray(target)
        valid = np.asarray(valid_mask, dtype=bool)
        class_count = len(self.class_names)
        if scores.ndim != 3 or scores.shape[0] != class_count:
            raise ValueError("tile logits must have shape (C, H, W)")
        if labels.shape != scores.shape[1:] or valid.shape != labels.shape:
            raise ValueError("tile labels and valid mask must match logit spatial shape")
        if not sample_id or not isinstance(sample_id, str):
            raise ValueError("sample_id must be a non-empty string")
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"non-finite logits in {sample_id}")
        labels = labels.astype(np.int64, copy=False)
        flat_scores = scores.reshape(class_count, -1).T
        flat_labels = labels.reshape(-1)
        flat_valid = valid.reshape(-1)
        for class_index, item in enumerate(self._items):
            indices = np.flatnonzero(flat_valid & (flat_labels == class_index))
            item.seen += int(indices.size)
            if not indices.size:
                continue
            rng = self._rng(sample_id, class_index)
            if indices.size > self.per_tile_class_cap:
                indices = indices[
                    rng.choice(
                        indices.size,
                        size=self.per_tile_class_cap,
                        replace=False,
                    )
                ]
            priorities = rng.random(indices.size)
            selected = flat_scores[indices]
            combined_priority = np.concatenate([item.priority, priorities])
            combined_logits = np.concatenate([item.logits, selected], axis=0)
            if combined_priority.size > self.maximum_pixels_per_class:
                keep = np.argpartition(
                    combined_priority, self.maximum_pixels_per_class - 1
                )[: self.maximum_pixels_per_class]
                order = keep[np.argsort(combined_priority[keep], kind="stable")]
                combined_priority = combined_priority[order]
                combined_logits = combined_logits[order]
            item.priority = combined_priority
            item.logits = combined_logits
        self.tiles_seen += 1

    def finalize(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        missing = [
            name
            for name, item in zip(self.class_names, self._items, strict=True)
            if item.logits.shape[0] == 0
        ]
        if missing:
            raise ValueError(f"bounded sample has no pixels for classes: {missing}")
        logits = np.concatenate([item.logits for item in self._items], axis=0)
        target = np.concatenate(
            [
                np.full(item.logits.shape[0], index, dtype=np.int64)
                for index, item in enumerate(self._items)
            ]
        )
        sample_weight = np.concatenate(
            [
                np.full(
                    item.logits.shape[0],
                    item.seen / item.logits.shape[0],
                    dtype=np.float64,
                )
                for item in self._items
            ]
        )
        provenance = {
            "strategy": "per-tile deterministic subsample then bounded priority reservoir",
            "class_prior_correction": (
                "each retained class pixel is weighted by valid_pixels_seen / "
                "pixels_retained when fitting approximate deployment-prior metrics"
            ),
            "seed": self.seed,
            "per_tile_class_cap": self.per_tile_class_cap,
            "maximum_pixels_per_class": self.maximum_pixels_per_class,
            "hard_maximum_cached_pixels": (
                len(self.class_names) * self.maximum_pixels_per_class
            ),
            "retained_pixels": int(target.size),
            "tiles_seen": self.tiles_seen,
            "per_class": {
                name: {
                    "valid_pixels_seen": item.seen,
                    "pixels_retained": int(item.logits.shape[0]),
                }
                for name, item in zip(self.class_names, self._items, strict=True)
            },
        }
        return logits, target, sample_weight, provenance


__all__ = [
    "DeterministicBoundedClassSampler",
    "centered_biases",
    "confusion_from_logits",
    "decision_gate_report",
    "fit_additive_decision_biases",
    "metrics_from_logits",
]
