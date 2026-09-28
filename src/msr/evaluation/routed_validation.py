"""Read-only metrics and fail-closed guards for routed surface validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from msr.data.surface_dataset import LANDSCAPE_CLASSES
from msr.evaluation.classification_metrics import (
    StreamingMulticlassMetrics,
)
from msr.evaluation.metrics import StreamingRegressionMetrics
from msr.models.routed_surface import RoutedDomainGatedSurfaceNet


class RoutedValidationError(RuntimeError):
    """Raised when routed-model evaluation cannot be proven safe or complete."""


def _metric_at_path(metrics: Mapping[str, object], path: str) -> float:
    value: object = metrics
    for component in path.split("."):
        if not component:
            raise ValueError(f"Invalid empty component in metric path {path!r}")
        if not isinstance(value, Mapping) or component not in value:
            raise KeyError(
                f"Validation metric path {path!r} is missing at {component!r}"
            )
        value = value[component]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"Validation metric path {path!r} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Validation metric path {path!r} is not finite")
    return result


def evaluate_validation_guards(
    current_by_suite: Mapping[str, Mapping[str, object]],
    protected_by_suite: Mapping[str, Mapping[str, object]],
    validation_guards: Mapping[str, object],
) -> tuple[dict[str, object], bool]:
    """Evaluate the configured GAMUS and legacy guards against one baseline.

    Relative specifications must explicitly use ``baseline: initial``.  For a
    routed endpoint evaluation, "initial" is the freshly recomputed protected
    endpoint on the exact same loader and preprocessing policy.
    """

    if not validation_guards:
        raise ValueError("At least one validation-guard suite is required")
    supported = {"min", "max", "max_regression", "max_drop"}
    suite_results: dict[str, object] = {}
    all_pass = True
    for raw_suite, raw_suite_guards in validation_guards.items():
        suite = str(raw_suite)
        if suite not in current_by_suite:
            raise KeyError(f"Validation guard suite {suite!r} has no routed metrics")
        if suite not in protected_by_suite:
            raise KeyError(
                f"Validation guard suite {suite!r} has no protected baseline"
            )
        if not isinstance(raw_suite_guards, Mapping) or not raw_suite_guards:
            raise ValueError(
                f"Validation guard suite {suite!r} must be a non-empty mapping"
            )
        guard_results: dict[str, object] = {}
        suite_pass = True
        for raw_path, raw_spec in raw_suite_guards.items():
            path = str(raw_path)
            if not isinstance(raw_spec, Mapping):
                raise ValueError(f"Validation guard {suite}.{path} must be a mapping")
            comparisons = supported & set(raw_spec)
            if len(comparisons) != 1:
                raise ValueError(
                    f"Validation guard {suite}.{path} must define exactly one of "
                    f"{sorted(supported)}"
                )
            comparison = next(iter(comparisons))
            allowed = {comparison}
            if comparison in {"max_regression", "max_drop"}:
                allowed.add("baseline")
                if raw_spec.get("baseline") != "initial":
                    raise ValueError(
                        f"Relative validation guard {suite}.{path} must set "
                        "baseline: initial"
                    )
            elif "baseline" in raw_spec:
                raise ValueError(
                    f"Absolute validation guard {suite}.{path} cannot define a baseline"
                )
            unknown = set(raw_spec) - allowed
            if unknown:
                raise ValueError(
                    f"Validation guard {suite}.{path} has unsupported keys: "
                    f"{sorted(unknown)}"
                )
            threshold = float(raw_spec[comparison])
            if not math.isfinite(threshold) or (
                comparison in {"max_regression", "max_drop"} and threshold < 0
            ):
                raise ValueError(
                    f"Validation guard {suite}.{path} threshold must be finite "
                    "and non-negative"
                )
            current = _metric_at_path(current_by_suite[suite], path)
            reference: float | None = None
            delta: float | None = None
            if comparison == "min":
                passed = current >= threshold
            elif comparison == "max":
                passed = current <= threshold
            else:
                reference = _metric_at_path(protected_by_suite[suite], path)
                delta = current - reference
                passed = (
                    delta <= threshold
                    if comparison == "max_regression"
                    else delta >= -threshold
                )
            passed = bool(passed)
            guard_results[path] = {
                "comparison": comparison,
                "threshold": threshold,
                "current": current,
                "protected_reference": reference,
                "delta": delta,
                "passes": passed,
            }
            suite_pass &= passed
        suite_results[suite] = {
            "passes": bool(suite_pass),
            "guards": guard_results,
        }
        all_pass &= suite_pass
    return {
        "passes": bool(all_pass),
        "baseline": "protected_endpoint_recomputed_on_same_suite",
        "suites": suite_results,
    }, bool(all_pass)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision not in {"fp16", "bf16"}:
        raise ValueError("precision must be 'fp32', 'fp16', or 'bf16'")
    if device.type != "cuda":
        raise ValueError("fp16/bf16 routed validation requires CUDA")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _to_device(batch: Mapping[str, object], name: str, device: torch.device) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Validation batch is missing tensor {name!r}")
    return value.to(device, non_blocking=True)


def _batch_strings(batch: Mapping[str, object], name: str, size: int) -> list[str]:
    value = batch.get(name)
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = [str(item) for item in value]
    else:
        raise TypeError(f"Validation batch is missing text field {name!r}")
    if len(values) != size:
        raise ValueError(f"Validation field {name!r} does not match batch size")
    return values


@torch.inference_mode()
def evaluate_dense_surface_model(
    model: nn.Module,
    loader: Any,
    *,
    device: str | torch.device = "cpu",
    precision: str = "fp32",
) -> dict[str, object]:
    """Compute existing dense height/semantic metrics plus scene-route audit data."""

    device = torch.device(device)
    model.to(device).eval()
    overall = StreamingRegressionMetrics()
    domain_metrics: dict[str, StreamingRegressionMetrics] = {
        name: StreamingRegressionMetrics() for name in LANDSCAPE_CLASSES
    }
    landscape_metrics: dict[str, StreamingRegressionMetrics] = defaultdict(
        StreamingRegressionMetrics
    )
    region_metrics: dict[str, StreamingRegressionMetrics] = defaultdict(
        StreamingRegressionMetrics
    )
    building_expert = StreamingRegressionMetrics()
    vegetation_expert = StreamingRegressionMetrics()
    class_names = tuple(
        name for name, _ in sorted(LANDSCAPE_CLASSES.items(), key=lambda item: item[1])
    )
    identification = StreamingMulticlassMetrics(class_names)
    building_router = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    route_probabilities: list[float] = []
    route_rows: list[dict[str, object]] = []
    saw_route_outputs: bool | None = None
    batches = 0

    for batch in loader:
        if not isinstance(batch, Mapping):
            raise TypeError("Validation loader must emit mapping batches")
        image = _to_device(batch, "image", device)
        height = _to_device(batch, "height", device)
        regression_mask = _to_device(batch, "regression_mask", device).bool()
        domain_target = _to_device(batch, "domain_target", device).long()
        domain_valid = _to_device(batch, "domain_valid_mask", device).bool()
        relative_prior = _to_device(batch, "relative_prior", device)
        valid_mask = _to_device(batch, "valid_mask", device).bool()
        batch_size = int(image.shape[0])
        landscapes = _batch_strings(batch, "landscape", batch_size)
        regions = _batch_strings(batch, "region", batch_size)
        sample_ids = _batch_strings(batch, "sample_id", batch_size)

        with _autocast(device, precision):
            if isinstance(model, RoutedDomainGatedSurfaceNet):
                # The deployed app has no label/reference validity mask.  Route
                # descriptors therefore use every pixel of the deterministic
                # centre crop, exactly like Stage-3 record generation.
                output = model(image, relative_prior)
            else:
                output = model(image, relative_prior)
        required_outputs = {
            "height",
            "domain_logits",
            "domain_probabilities",
            "protected_building_logits",
            "building_height",
            "canopy_height",
        }
        missing_outputs = required_outputs - set(output)
        if missing_outputs:
            raise RoutedValidationError(
                f"Surface model omitted required outputs: {sorted(missing_outputs)}"
            )
        prediction_tensor = output["height"]
        if not bool(torch.isfinite(prediction_tensor).all()):
            raise FloatingPointError("Non-finite height during routed validation")
        prediction = prediction_tensor.float().cpu().numpy()
        target = height.float().cpu().numpy()
        regression = regression_mask.cpu().numpy()
        domains = domain_target.cpu().numpy()
        domain_valid_numpy = domain_valid.cpu().numpy()
        identification.update(
            output["domain_logits"].argmax(dim=1).cpu().numpy(),
            domains,
            domain_valid_numpy,
        )
        overall.update(prediction, target, regression)
        building_expert.update(
            output["building_height"].float().cpu().numpy(),
            target,
            regression
            & (domains[:, None] == LANDSCAPE_CLASSES["building"]),
        )
        vegetation_expert.update(
            output["canopy_height"].float().cpu().numpy(),
            target,
            regression
            & (domains[:, None] == LANDSCAPE_CLASSES["vegetation"]),
        )
        for name, code in LANDSCAPE_CLASSES.items():
            domain_metrics[name].update(
                prediction,
                target,
                regression & (domains[:, None] == code),
            )
        for index, landscape in enumerate(landscapes):
            landscape_metrics[landscape].update(
                prediction[index : index + 1],
                target[index : index + 1],
                regression[index : index + 1],
            )
        for index, region in enumerate(regions):
            region_metrics[region].update(
                prediction[index : index + 1],
                target[index : index + 1],
                regression[index : index + 1],
            )

        router_score = (
            output["protected_building_logits"].sigmoid()
            * output["domain_probabilities"][:, 1:2]
            * (1.0 - output["domain_probabilities"][:, 2:3])
        )
        router_prediction = router_score[:, 0] >= 0.5
        router_truth = domain_target == LANDSCAPE_CLASSES["building"]
        building_router["tp"] += int(
            torch.count_nonzero(domain_valid & router_truth & router_prediction)
        )
        building_router["fp"] += int(
            torch.count_nonzero(domain_valid & ~router_truth & router_prediction)
        )
        building_router["fn"] += int(
            torch.count_nonzero(domain_valid & router_truth & ~router_prediction)
        )
        building_router["tn"] += int(
            torch.count_nonzero(domain_valid & ~router_truth & ~router_prediction)
        )

        has_route = {
            "route_candidate_probability",
            "route_candidate_selected",
        }.issubset(output)
        if saw_route_outputs is None:
            saw_route_outputs = has_route
        elif saw_route_outputs != has_route:
            raise RoutedValidationError(
                "Model inconsistently exposed route outputs between batches"
            )
        if has_route:
            probabilities = (
                output["route_candidate_probability"].float().cpu().reshape(-1).numpy()
            )
            selections = (
                output["route_candidate_selected"].bool().cpu().reshape(-1).numpy()
            )
            if probabilities.size != batch_size or selections.size != batch_size:
                raise RoutedValidationError("Scene-route outputs do not match batch size")
            if np.any(~np.isfinite(probabilities)) or np.any(
                (probabilities < 0.0) | (probabilities > 1.0)
            ):
                raise RoutedValidationError("Router probabilities are invalid")
            route_probabilities.extend(float(value) for value in probabilities)
            route_rows.extend(
                {
                    "sample_id": sample_ids[index],
                    "landscape": landscapes[index],
                    "candidate_probability": float(probabilities[index]),
                    "selected_endpoint": (
                        "gamus_stage1" if selections[index] else "protected"
                    ),
                }
                for index in range(batch_size)
            )
        batches += 1

    if batches == 0 or overall.count == 0 or identification.count == 0:
        raise RoutedValidationError("Validation loader produced no scorable pixels")
    domains_result = {
        name: metric.compute() for name, metric in domain_metrics.items() if metric.count
    }
    landscapes_result = {
        name: metric.compute() for name, metric in sorted(landscape_metrics.items())
    }
    regions_result = {
        name: metric.compute() for name, metric in sorted(region_metrics.items())
    }
    experts_result: dict[str, object] = {}
    if building_expert.count:
        experts_result["building"] = building_expert.compute()
    if vegetation_expert.count:
        experts_result["vegetation"] = vegetation_expert.compute()
    tp = building_router["tp"]
    fp = building_router["fp"]
    fn = building_router["fn"]
    precision_value = tp / max(tp + fp, 1)
    recall_value = tp / max(tp + fn, 1)
    beta_squared = 0.25
    f0_5 = (
        (1.0 + beta_squared) * precision_value * recall_value
        / max(beta_squared * precision_value + recall_value, 1.0e-12)
    )
    result: dict[str, object] = {
        **overall.compute(),
        "domains": domains_result,
        "landscapes": landscapes_result,
        "regions": regions_result,
        "experts": experts_result,
        "domain_macro_rmse_m": float(
            np.mean([float(item["rmse_m"]) for item in domains_result.values()])
        ),
        "landscape_macro_rmse_m": float(
            np.mean([float(item["rmse_m"]) for item in landscapes_result.values()])
        ),
        "semantic_identification": identification.compute(),
        "building_router": {
            **building_router,
            "score_threshold": 0.5,
            "precision": float(precision_value),
            "recall": float(recall_value),
            "f0_5": float(f0_5),
        },
        "building_router_error": float(1.0 - f0_5),
    }
    if "building" in experts_result:
        result["building_expert_rmse_m"] = float(
            experts_result["building"]["rmse_m"]
        )
    if "vegetation" in experts_result:
        result["canopy_expert_rmse_m"] = float(
            experts_result["vegetation"]["rmse_m"]
        )
    if domains_result:
        worst_domain = max(
            domains_result, key=lambda name: float(domains_result[name]["rmse_m"])
        )
        result["worst_domain"] = worst_domain
        result["worst_domain_rmse_m"] = float(
            domains_result[worst_domain]["rmse_m"]
        )
    if landscapes_result:
        worst_landscape = max(
            landscapes_result,
            key=lambda name: float(landscapes_result[name]["rmse_m"]),
        )
        result["worst_landscape"] = worst_landscape
        result["worst_landscape_rmse_m"] = float(
            landscapes_result[worst_landscape]["rmse_m"]
        )
    if saw_route_outputs:
        probabilities = np.asarray(route_probabilities, dtype=np.float64)
        candidate_count = sum(
            row["selected_endpoint"] == "gamus_stage1" for row in route_rows
        )
        result["scene_routing"] = {
            "scene_count": len(route_rows),
            "protected_selected": len(route_rows) - candidate_count,
            "gamus_stage1_selected": candidate_count,
            "candidate_selection_rate": candidate_count / len(route_rows),
            "candidate_probability": {
                "mean": float(probabilities.mean()),
                "std": float(probabilities.std()),
                "min": float(probabilities.min()),
                "p05": float(np.quantile(probabilities, 0.05)),
                "median": float(np.quantile(probabilities, 0.50)),
                "p95": float(np.quantile(probabilities, 0.95)),
                "max": float(probabilities.max()),
            },
            "per_scene": route_rows,
        }
    return result


__all__ = [
    "RoutedValidationError",
    "evaluate_dense_surface_model",
    "evaluate_validation_guards",
]
