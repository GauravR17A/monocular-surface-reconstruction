"""Full-native, classification-only evaluation for the independent RGB pilot.

Report structure matches the fixed V3 DC+PHL replay. Acceptance requires the
same sample IDs and reference support, not merely a more attractive mean F1.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

from msr.data.gamus_rgb_segmentation import ALLOWED_CITIES, canonical_sha256
from msr.data.gamus_dataset import GAMUS_SIX_CLASS_NAMES
from .classification_metrics import (
    StreamingClassBoundaryMetrics,
    StreamingMulticlassMetrics,
    StreamingWaterDarkPixelProxy,
    compute_multiclass_metrics,
    multiclass_metric_deltas,
)


CLASS_NAMES = tuple(GAMUS_SIX_CLASS_NAMES)


@dataclass
class _MetricGroup:
    six: StreamingMulticlassMetrics
    road: StreamingClassBoundaryMetrics
    water: StreamingWaterDarkPixelProxy

    @classmethod
    def create(cls) -> "_MetricGroup":
        return cls(StreamingMulticlassMetrics(CLASS_NAMES),
                   StreamingClassBoundaryMetrics(CLASS_NAMES.index("roads"), tolerance_pixels=2),
                   StreamingWaterDarkPixelProxy(CLASS_NAMES.index("water")))

    def update(self, prediction, target, dark, valid) -> None:
        self.six.update(prediction, target, valid)
        self.road.update(prediction, target, valid)
        self.water.update(prediction, target, dark, valid)

    def compute(self) -> dict[str, Any]:
        return {"six_class_identification": self.six.compute(),
                "road_boundary_quality": self.road.compute(),
                "water_dark_pixel_proxy": {**self.water.compute(), "water_class_index": CLASS_NAMES.index("water")}}


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


@torch.inference_mode()
def evaluate_rgb_segmentation(model: torch.nn.Module, loader: Iterable, device: torch.device | str, *,
                              precision: str = "bf16", expected_ids: Sequence[str] | None = None,
                              native_size: int = 1024,
                              progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Evaluate six-class logits on every supplied native validation tile.

    ``model`` receives ONLY image tensors. Labels, masks and the dark-pixel
    proxy are used after inference for evaluation, never as model features.
    The native-size override allows small synthetic unit fixtures only.
    """
    device = torch.device(device)
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16 or fp16")
    if native_size <= 0:
        raise ValueError("native_size must be positive")
    if expected_ids is None:
        dataset = getattr(loader, "dataset", None)
        expected_ids = getattr(dataset, "sample_ids", None)
    if not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Unique expected validation IDs are required")
    expected_ids = tuple(str(value) for value in expected_ids)
    for sample_id in expected_ids:
        if sample_id.split("_", 1)[0] not in ALLOWED_CITIES:
            raise ValueError("Only DC/PHL development validation may be evaluated")
    groups = {city: _MetricGroup.create() for city in ALLOWED_CITIES}
    overall = _MetricGroup.create()
    observed = []
    support_hash_rows = []
    was_training = model.training
    model.eval()
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    try:
        for batch_index, batch in enumerate(loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            if image.ndim != 4 or image.shape[1:] != (3, native_size, native_size):
                raise ValueError("Evaluation must use unresized full-native RGB tiles")
            if not torch.isfinite(image).all() or torch.any((image < 0) | (image > 1)):
                raise ValueError("Model input must be finite raw RGB in [0,1]")
            amp = (torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
                   if device.type == "cuda" and precision != "fp32" else nullcontext())
            with amp:
                output = model(image)
            logits = output.get("logits") if isinstance(output, Mapping) else output
            if not isinstance(logits, torch.Tensor) or logits.shape != (image.shape[0], 6, native_size, native_size):
                raise ValueError("Model must return six logits per full-native pixel")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite segmentation logits")
            prediction = logits.argmax(1).cpu().numpy()
            labels = _numpy(batch["labels"])
            image_valid = _numpy(batch["image_valid_mask"]).astype(bool)
            class_valid = _numpy(batch["classification_valid_mask"]).astype(bool)
            dark = _numpy(batch["dark_pixel_proxy_mask"]).astype(bool)
            valid = image_valid & class_valid
            if not (prediction.shape == labels.shape == image_valid.shape == class_valid.shape == dark.shape):
                raise ValueError("Prediction, label and independent validity grids must match")
            if np.any(class_valid & (~np.isfinite(labels) | (labels < 0) | (labels >= 6) | (labels != np.rint(labels)))):
                raise ValueError("Classification-valid label outside the six-class taxonomy")
            if np.any(dark & ~image_valid):
                raise ValueError("Dark-pixel proxy cannot include invalid RGB")
            sample_ids = list(batch["sample_id"])
            cities = list(batch["city"])
            if len(sample_ids) != prediction.shape[0] or len(cities) != len(sample_ids):
                raise ValueError("Batch identity count differs from predictions")
            for i, sample_id in enumerate(sample_ids):
                city = str(sample_id).split("_", 1)[0]
                if city not in groups or city != cities[i] or sample_id not in expected_ids:
                    raise ValueError("Unexpected validation sample or city identity")
                observed.append(sample_id)
                overall.update(prediction[i], labels[i], dark[i], valid[i])
                groups[city].update(prediction[i], labels[i], dark[i], valid[i])
                # Pixel-position-sensitive support binding supplements pooled
                # support counts for repeat comparisons of this RGB evaluator.
                import hashlib
                support_hash_rows.append({"sample_id": sample_id,
                                          "valid_mask_sha256": hashlib.sha256(np.ascontiguousarray(valid[i], dtype=np.uint8).tobytes()).hexdigest(),
                                          "reference_labels_sha256": hashlib.sha256(np.ascontiguousarray(np.where(valid[i], labels[i], 255), dtype=np.uint8).tobytes()).hexdigest()})
            if progress_callback is not None:
                progress_callback({"completed_batches": batch_index, "total_batches": total_batches})
    finally:
        model.train(was_training)
    if tuple(observed) != expected_ids:
        raise ValueError("Validation must cover each sealed ID exactly once in fixed order")
    if overall.six.count == 0:
        raise ValueError("No valid classification references were evaluated")
    return {"schema": "msr.gamus_rgb_segmentation_evaluation.v1",
            "interpretation": "DC+PHL development validation, not system-unseen or official test",
            "height_evaluated": False, "model_inputs": ["RGB"],
            "validation_native_size": native_size,
            "evaluated_sample_count": len(observed),
            "evaluated_samples_by_city": dict(sorted(Counter(s.split("_", 1)[0] for s in observed).items())),
            "evaluated_ids_sha256": canonical_sha256(sorted(observed)),
            "evaluated_ordered_ids_sha256": canonical_sha256(observed),
            "evaluated_ids_by_city_sha256": {city: canonical_sha256(sorted(s for s in observed if s.startswith(f"{city}_"))) for city in ALLOWED_CITIES},
            "reference_grid_binding_sha256": canonical_sha256(support_hash_rows),
            "overall": overall.compute(), "by_city": {city: group.compute() for city, group in groups.items()}}


evaluate = evaluate_rgb_segmentation


def _validate_group(group: Mapping[str, Any], role: str) -> None:
    six = group["six_class_identification"]
    if tuple(six.get("class_names", ())) != CLASS_NAMES:
        raise ValueError(f"{role}: class order differs")
    matrix = np.asarray(six["confusion_matrix"])
    if matrix.shape != (6, 6) or matrix.dtype.kind not in "iu" or np.any(matrix < 0):
        raise ValueError(f"{role}: malformed confusion matrix")
    expected = compute_multiclass_metrics(matrix, CLASS_NAMES)
    # Equality is exact because both scoreboards use the same deterministic
    # integer-confusion implementation and JSON preserves these floats.
    for key in ("total_valid_pixels", "macro_f1", "macro_iou", "per_class"):
        if six.get(key) != expected[key]:
            raise ValueError(f"{role}: inconsistent {key} and confusion matrix")
    if expected["total_valid_pixels"] <= 0:
        raise ValueError(f"{role}: empty evaluation support")
    road = group["road_boundary_quality"]
    if road.get("class_index") != CLASS_NAMES.index("roads") or road.get("tolerance_pixels") != 2:
        raise ValueError(f"{role}: road-boundary protocol differs")
    water = group["water_dark_pixel_proxy"]
    if water.get("is_shadow_proxy_not_ground_truth") is not True:
        raise ValueError(f"{role}: dark pixels cannot be called shadow truth")
    for name, value in (("road_boundary_f1", road["f1"]), ("dark_water_rate", water["false_water_rate_on_dark_non_water"])):
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{role}: invalid {name}")


def classification_acceptance(candidate: Mapping[str, Any], baseline_report: Mapping[str, Any],
                              overall_floors: Mapping[str, float],
                              city_floors: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Evaluate predeclared per-class/city floors against the fixed V3 support.

    Passing this development gate never promotes the active app model or
    proves new-geography accuracy. No metric is manufactured for a missing
    class, and an incomparable support set fails closed.
    """
    baseline = baseline_report.get("v3", baseline_report)
    failed, checks, deltas = [], [], {}
    try:
        identity_keys = ("evaluated_sample_count", "evaluated_samples_by_city", "evaluated_ids_sha256", "evaluated_ids_by_city_sha256")
        for key in identity_keys:
            if candidate.get(key) != baseline.get(key) or key not in candidate:
                raise ValueError(f"Candidate/baseline {key} differs")
        if set(candidate["by_city"]) != set(ALLOWED_CITIES) or set(baseline["by_city"]) != set(ALLOWED_CITIES):
            raise ValueError("Both DC and PHL city reports are required")
        if set(city_floors) != set(ALLOWED_CITIES):
            raise ValueError("Both fixed per-city floor sets are required")
        groups = [("overall", candidate["overall"], baseline["overall"], overall_floors)]
        groups.extend((city, candidate["by_city"][city], baseline["by_city"][city], city_floors[city]) for city in ALLOWED_CITIES)
        for name, current, reference, floors in groups:
            _validate_group(current, name)
            _validate_group(reference, f"baseline {name}")
            current_six, baseline_six = current["six_class_identification"], reference["six_class_identification"]
            for class_name in CLASS_NAMES:
                a = current_six["per_class"][class_name]["support_pixels"]
                b = baseline_six["per_class"][class_name]["support_pixels"]
                if a != b or a <= 0:
                    raise ValueError(f"{name}: reference support differs or is empty for {class_name}")
            for metric, field in (("road_boundary_quality", "reference_boundary_pixels"), ("water_dark_pixel_proxy", "dark_non_water_pixels")):
                if current[metric][field] != reference[metric][field]:
                    raise ValueError(f"{name}: {field} differs from baseline")
            required = {"six_class_macro_f1_min", "road_boundary_f1_min", "dark_non_water_false_water_rate_max", *(f"{c}_f1_min" for c in CLASS_NAMES)}
            if set(floors) != required:
                raise ValueError(f"{name}: all six class floors, macro, road and dark-water guards must be predeclared")
            measurements = {"six_class_macro_f1_min": current_six["macro_f1"],
                            "road_boundary_f1_min": current["road_boundary_quality"]["f1"],
                            "dark_non_water_false_water_rate_max": current["water_dark_pixel_proxy"]["false_water_rate_on_dark_non_water"],
                            **{f"{c}_f1_min": current_six["per_class"][c]["f1"] for c in CLASS_NAMES}}
            for key in sorted(required):
                limit, measured = float(floors[key]), float(measurements[key])
                if not math.isfinite(limit) or not 0 <= limit <= 1:
                    raise ValueError(f"{name}: invalid acceptance threshold {key}")
                passed = measured <= limit + 1e-12 if key.endswith("_max") else measured >= limit - 1e-12
                check = {"scope": name, "metric": key, "value": measured, "threshold": limit, "passes": passed}
                checks.append(check)
                if not passed:
                    failed.append(f"{name}.{key}: {measured:.6f} vs {limit:.6f}")
            deltas[name] = multiclass_metric_deltas(current_six, baseline_six)
        comparable = True
    except (KeyError, TypeError, ValueError) as error:
        comparable = False
        failed.append(f"Incomparable or invalid evaluation: {error}")
    return {"passes": comparable and not failed, "eligible": comparable and not failed,
            "comparable_to_fixed_v3": comparable, "failed_checks": failed, "checks": checks,
            "candidate_minus_v3": deltas, "promotion_performed": False,
            "scope": "Development identification only; app integration and external evaluation remain required"}
