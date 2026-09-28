"""Per-building HighBuild evaluation with explicit annotation provenance.

The corrected HighBuild rasters deliberately treat pixels outside COCO
building footprints as *unknown*, rather than as zero-height ground.  This
module keeps that contract while adding object-level evaluation.  It validates
the COCO outlines against the corrected semantic and height-validity masks
before computing any score.

Detection precision and count error are only defined when an independent
classification-validity mask certifies that annotations are exhaustive over
the evaluated area.  With the current positive-only corrected artifacts the
module still reports annotated-building recall, matched outline quality and
height accuracy, but it will not turn unknown background into false positives.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from msr.data.highbuild import annotation_masks, regression_support_mask
from msr.evaluation.metrics import StreamingRegressionMetrics


class HighBuildInstanceEvaluationError(ValueError):
    """Raised when instance evaluation cannot be performed honestly."""


@dataclass(frozen=True)
class ReferenceBuilding:
    """One validated COCO building annotation rasterized on the model grid."""

    annotation_id: str
    mask: np.ndarray
    height_m: float | None
    is_estimated_height: bool
    estimated_flag_explicit: bool

    @property
    def provenance(self) -> str:
        if self.height_m is None:
            return "missing_height"
        return "estimated" if self.is_estimated_height else "measured"


@dataclass(frozen=True)
class _MatchedBuilding:
    reference_index: int
    prediction_label: int
    iou: float


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _harmonic_mean(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _regression_summary(
    predictions: Iterable[float], targets: Iterable[float]
) -> dict[str, int | float | None]:
    predicted = np.asarray(tuple(predictions), dtype=np.float64)
    observed = np.asarray(tuple(targets), dtype=np.float64)
    if predicted.shape != observed.shape:
        raise HighBuildInstanceEvaluationError(
            "Instance prediction and reference collections differ in length"
        )
    valid = np.isfinite(predicted) & np.isfinite(observed)
    predicted = predicted[valid]
    observed = observed[valid]
    if predicted.size == 0:
        return {
            "instance_count": 0,
            "rmse_m": None,
            "mae_m": None,
            "bias_m": None,
            "median_ae_m": None,
            "p90_ae_m": None,
            "correlation": None,
            "r2": None,
        }
    error = predicted - observed
    absolute = np.abs(error)
    target_centered = observed - observed.mean()
    prediction_centered = predicted - predicted.mean()
    target_ss = float(np.square(target_centered).sum())
    prediction_ss = float(np.square(prediction_centered).sum())
    correlation = (
        float(
            (target_centered * prediction_centered).sum()
            / math.sqrt(target_ss * prediction_ss)
        )
        if target_ss > 0.0 and prediction_ss > 0.0
        else None
    )
    return {
        "instance_count": int(error.size),
        "rmse_m": float(np.sqrt(np.square(error).mean())),
        "mae_m": float(absolute.mean()),
        "bias_m": float(error.mean()),
        "median_ae_m": float(np.median(absolute)),
        "p90_ae_m": float(np.percentile(absolute, 90)),
        "correlation": correlation,
        "r2": float(1.0 - np.square(error).sum() / target_ss)
        if target_ss > 0.0
        else None,
    }


def _tile_counting_summary(
    *,
    reference_count: int,
    prediction_count: int,
    exhaustive_reference: bool,
    prediction_is_instance_labels: bool,
) -> dict[str, Any]:
    """Report building counts independently from pixel and detection scores.

    HighBuild's corrected positive-only masks do not prove that every building
    in a tile is annotated.  A numerical count error is therefore emitted only
    when a separate classification-validity contract certifies exhaustive
    annotations.  The prediction representation is recorded because connected
    components from a semantic mask can merge touching buildings, while an
    explicit instance raster preserves the candidate's own partition.
    """

    representation = (
        "provided_instance_labels"
        if prediction_is_instance_labels
        else "connected_components_from_building_mask"
    )
    common: dict[str, Any] = {
        "metric_source": "instance_counts_scored_separately_from_pixel_metrics",
        "prediction_representation": representation,
        "annotated_reference_instances": reference_count,
        "predicted_instances": prediction_count,
    }
    if not exhaustive_reference:
        return {
            **common,
            "status": "unavailable_positive_annotations_are_not_exhaustive",
            "reference_count": None,
            "signed_error_predicted_minus_reference": None,
            "absolute_error": None,
            "absolute_percentage_error": None,
            "exact_count": None,
        }

    signed_error = prediction_count - reference_count
    return {
        **common,
        "status": "validated_against_exhaustive_instance_reference",
        "reference_count": reference_count,
        "signed_error_predicted_minus_reference": signed_error,
        "absolute_error": abs(signed_error),
        "absolute_percentage_error": (
            float(abs(signed_error) / reference_count)
            if reference_count > 0
            else None
        ),
        "exact_count": signed_error == 0,
    }


def parse_highbuild_instances(
    coco: Mapping[str, Any], *, height: int, width: int
) -> list[ReferenceBuilding]:
    """Parse and rasterize individual HighBuild COCO annotations.

    The parser is intentionally strict: a tile must describe exactly one image,
    every annotation must belong to that image and to the building category,
    annotation IDs must be unique, and any provided estimated-height flag must
    be boolean. Missing flags retain the corrected artifact's established
    "not marked estimated" meaning and are counted explicitly in reports.
    Crowd/RLE and malformed polygons are rejected by the shared rasterizer.
    Valid sub-pixel polygons that rasterize to no pixel are retained for audit
    and explicitly excluded from pixel-grid detection denominators.
    """

    images = coco.get("images")
    annotations = coco.get("annotations")
    categories = coco.get("categories")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise HighBuildInstanceEvaluationError(
            "HighBuild COCO tile must contain exactly one image record"
        )
    if int(images[0].get("height", -1)) != height or int(
        images[0].get("width", -1)
    ) != width:
        raise HighBuildInstanceEvaluationError(
            "HighBuild COCO dimensions do not match the evaluation grid"
        )
    if not isinstance(categories, list):
        raise HighBuildInstanceEvaluationError("HighBuild COCO categories are missing")
    building_ids = {
        category.get("id")
        for category in categories
        if isinstance(category, dict)
        and str(category.get("name", "")).strip().lower() == "building"
    }
    if len(building_ids) != 1:
        raise HighBuildInstanceEvaluationError(
            "HighBuild COCO must declare exactly one building category"
        )
    if not isinstance(annotations, list):
        raise HighBuildInstanceEvaluationError("HighBuild COCO annotations are missing")

    image_id = images[0].get("id")
    building_id = next(iter(building_ids))
    if building_id is None or isinstance(building_id, bool):
        raise HighBuildInstanceEvaluationError(
            "HighBuild building category has no valid ID"
        )
    seen_ids: set[str] = set()
    instances: list[ReferenceBuilding] = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            raise HighBuildInstanceEvaluationError("HighBuild annotation must be an object")
        raw_annotation_id = annotation.get("id")
        if raw_annotation_id is None or isinstance(raw_annotation_id, bool):
            raise HighBuildInstanceEvaluationError("HighBuild annotation ID is missing")
        annotation_id = str(raw_annotation_id).strip()
        if not annotation_id:
            raise HighBuildInstanceEvaluationError("HighBuild annotation ID is missing")
        if annotation_id in seen_ids:
            raise HighBuildInstanceEvaluationError(
                f"Duplicate HighBuild annotation ID: {annotation_id}"
            )
        seen_ids.add(annotation_id)
        if annotation.get("image_id") != image_id:
            raise HighBuildInstanceEvaluationError(
                f"Annotation {annotation_id} belongs to a different image"
            )
        if annotation.get("category_id") != building_id:
            raise HighBuildInstanceEvaluationError(
                f"Annotation {annotation_id} is not in the building category"
            )
        attributes = annotation.get("attributes")
        if not isinstance(attributes, dict):
            raise HighBuildInstanceEvaluationError(
                f"Annotation {annotation_id} has no attributes mapping"
            )
        estimated_flag_explicit = "is_estimated_height" in attributes
        estimated = attributes.get("is_estimated_height", False)
        if not isinstance(estimated, bool):
            raise HighBuildInstanceEvaluationError(
                f"Annotation {annotation_id} has a non-boolean estimated-height flag"
            )
        raw_height = attributes.get("height")
        reference_height: float | None
        if isinstance(raw_height, bool) or not isinstance(raw_height, (int, float)):
            reference_height = None
        else:
            parsed_height = float(raw_height)
            reference_height = (
                parsed_height if math.isfinite(parsed_height) and parsed_height > 0.0 else None
            )
        one_annotation_coco = {
            "images": images,
            "annotations": [annotation],
        }
        try:
            instance_mask, _, _ = annotation_masks(
                one_annotation_coco, height=height, width=width
            )
        except ValueError as error:
            raise HighBuildInstanceEvaluationError(
                f"Invalid outline for annotation {annotation_id}: {error}"
            ) from error
        mask = instance_mask.astype(bool)
        instances.append(
            ReferenceBuilding(
                annotation_id=annotation_id,
                mask=mask,
                height_m=reference_height,
                is_estimated_height=estimated,
                estimated_flag_explicit=estimated_flag_explicit,
            )
        )
    return instances


def _union_masks(instances: Iterable[ReferenceBuilding], shape: tuple[int, int]) -> np.ndarray:
    union = np.zeros(shape, dtype=bool)
    for instance in instances:
        union |= instance.mask
    return union


def validate_corrected_highbuild_contract(
    *,
    instances: list[ReferenceBuilding],
    reference_height_m: np.ndarray,
    building_footprints: np.ndarray,
    height_valid_mask: np.ndarray,
    height_protocol: str,
) -> None:
    """Verify COCO, corrected masks and height raster agree exactly."""

    target = np.asarray(reference_height_m, dtype=np.float32)
    semantic = np.asarray(building_footprints, dtype=bool)
    height_valid = np.asarray(height_valid_mask, dtype=bool)
    if target.ndim != 2 or semantic.shape != target.shape or height_valid.shape != target.shape:
        raise HighBuildInstanceEvaluationError(
            "Reference height, building footprint and height-validity grids must match"
        )
    all_union = _union_masks(instances, target.shape)
    if not np.array_equal(all_union, semantic):
        mismatch = int(np.count_nonzero(all_union != semantic))
        raise HighBuildInstanceEvaluationError(
            f"Corrected building mask disagrees with individual COCO outlines at {mismatch} pixels"
        )
    measured_union = _union_masks(
        (
            instance
            for instance in instances
            if instance.height_m is not None and not instance.is_estimated_height
        ),
        target.shape,
    )
    try:
        expected_valid = regression_support_mask(
            all_footprints=all_union,
            measured_footprints=measured_union,
            target=target,
            nodata=None,
            protocol=height_protocol,
        ).astype(bool)
    except ValueError as error:
        raise HighBuildInstanceEvaluationError(str(error)) from error
    if not np.array_equal(expected_valid, height_valid):
        mismatch = int(np.count_nonzero(expected_valid != height_valid))
        raise HighBuildInstanceEvaluationError(
            f"Corrected height-validity mask violates {height_protocol!r} at {mismatch} pixels"
        )

    # Do not require the sparse raster value under every polygon pixel to equal
    # that polygon's attribute. Overlapping HighBuild polygons and the upstream
    # rasterization order can legitimately place a neighbouring building's
    # value there. Exact semantic/validity-mask reproduction above is the
    # authoritative corrected contract; COCO attributes remain the per-object
    # height reference.


def _predicted_instance_masks(
    prediction: np.ndarray,
    *,
    valid_mask: np.ndarray,
    probability_threshold: float,
    minimum_pixels: int,
    labels_are_instances: bool,
) -> list[tuple[int, np.ndarray]]:
    values = np.asarray(prediction)
    if values.shape != valid_mask.shape:
        raise HighBuildInstanceEvaluationError(
            "Predicted building grid and image-validity mask differ"
        )
    if minimum_pixels <= 0:
        raise HighBuildInstanceEvaluationError("minimum_pixels must be positive")
    if labels_are_instances:
        if not np.all(np.isfinite(values[valid_mask])):
            raise HighBuildInstanceEvaluationError(
                "Predicted instance labels contain non-finite valid pixels"
            )
        safe_values = np.where(valid_mask, values, 0)
        rounded = np.rint(safe_values)
        if np.any(values[valid_mask] < 0) or not np.allclose(
            safe_values[valid_mask], rounded[valid_mask]
        ):
            raise HighBuildInstanceEvaluationError(
                "Predicted instance labels must be non-negative integers"
            )
        label_grid = rounded.astype(np.int64)
        label_grid[~valid_mask] = 0
    else:
        if not 0.0 <= probability_threshold <= 1.0:
            raise HighBuildInstanceEvaluationError(
                "building probability threshold must be in [0, 1]"
            )
        binary = valid_mask & np.isfinite(values) & (values >= probability_threshold)
        try:
            from scipy import ndimage
        except ImportError as error:  # pragma: no cover - train extra is installed in CI
            raise HighBuildInstanceEvaluationError(
                "Connected-component instance evaluation requires the training extra (scipy)"
            ) from error
        label_grid, _ = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))

    result: list[tuple[int, np.ndarray]] = []
    for label in np.unique(label_grid):
        label_int = int(label)
        if label_int <= 0:
            continue
        mask = label_grid == label_int
        if int(mask.sum()) >= minimum_pixels:
            result.append((label_int, mask))
    return result


def _intersection_over_union(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return float(intersection / union) if union else 0.0


def _match_instances(
    references: list[ReferenceBuilding],
    predictions: list[tuple[int, np.ndarray]],
    *,
    iou_threshold: float,
) -> list[_MatchedBuilding]:
    if not 0.0 < iou_threshold <= 1.0:
        raise HighBuildInstanceEvaluationError("IoU threshold must be in (0, 1]")
    if not references or not predictions:
        return []
    ious = np.zeros((len(references), len(predictions)), dtype=np.float64)
    for reference_index, reference in enumerate(references):
        for prediction_index, (_, prediction_mask) in enumerate(predictions):
            ious[reference_index, prediction_index] = _intersection_over_union(
                reference.mask, prediction_mask
            )
    eligible = ious >= iou_threshold
    # A bonus larger than any possible total IoU makes the assignment maximize
    # valid match count first and outline overlap second.
    cardinality_bonus = float(min(ious.shape) + 1)
    weights = np.where(eligible, cardinality_bonus + ious, 0.0)
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - train extra is installed in CI
        raise HighBuildInstanceEvaluationError(
            "Optimal instance matching requires the training extra (scipy)"
        ) from error
    row_indices, column_indices = linear_sum_assignment(weights, maximize=True)
    matches: list[_MatchedBuilding] = []
    for reference_index, prediction_index in zip(row_indices, column_indices):
        if not eligible[reference_index, prediction_index]:
            continue
        matches.append(
            _MatchedBuilding(
                reference_index=int(reference_index),
                prediction_label=predictions[int(prediction_index)][0],
                iou=float(ious[reference_index, prediction_index]),
            )
        )
    return matches


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover
        raise HighBuildInstanceEvaluationError(
            "Boundary evaluation requires the training extra (scipy)"
        ) from error
    eroded = ndimage.binary_erosion(
        mask, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    return mask & ~eroded


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 0:
        raise HighBuildInstanceEvaluationError("Boundary tolerance cannot be negative")
    if radius == 0:
        return mask
    try:
        from scipy import ndimage
    except ImportError as error:  # pragma: no cover
        raise HighBuildInstanceEvaluationError(
            "Boundary evaluation requires the training extra (scipy)"
        ) from error
    return ndimage.binary_dilation(
        mask, structure=np.ones((3, 3), dtype=bool), iterations=radius
    )


def _boundary_scores(
    prediction: np.ndarray, reference: np.ndarray, *, tolerance_pixels: int
) -> tuple[float | None, float | None, float | None]:
    predicted_boundary = _binary_boundary(prediction)
    reference_boundary = _binary_boundary(reference)
    predicted_count = int(predicted_boundary.sum())
    reference_count = int(reference_boundary.sum())
    precision = _safe_ratio(
        int(np.count_nonzero(predicted_boundary & _dilate(reference_boundary, tolerance_pixels))),
        predicted_count,
    )
    recall = _safe_ratio(
        int(np.count_nonzero(reference_boundary & _dilate(predicted_boundary, tolerance_pixels))),
        reference_count,
    )
    return precision, recall, _harmonic_mean(precision, recall)


def authenticated_pixel_area_m2(
    *, crs: CRS | None, transform: Affine | None
) -> float | None:
    """Return metric pixel area only when a projected CRS authenticates scale."""

    if crs is None or transform is None or not crs.is_projected:
        return None
    try:
        _, linear_to_metre = crs.linear_units_factor
    except Exception:
        return None
    pixel_area_native = abs(transform.a * transform.e - transform.b * transform.d)
    pixel_area_m2 = float(pixel_area_native * float(linear_to_metre) ** 2)
    return pixel_area_m2 if math.isfinite(pixel_area_m2) and pixel_area_m2 > 0.0 else None


def evaluate_highbuild_tile(
    *,
    sample_id: str,
    prediction_height_m: np.ndarray,
    prediction_building: np.ndarray,
    coco: Mapping[str, Any],
    reference_height_m: np.ndarray,
    building_footprints: np.ndarray,
    height_valid_mask: np.ndarray,
    image_valid_mask: np.ndarray,
    height_protocol: str,
    classification_valid_mask: np.ndarray | None = None,
    prediction_is_instance_labels: bool = False,
    building_probability_threshold: float = 0.5,
    instance_iou_threshold: float = 0.5,
    minimum_predicted_instance_pixels: int = 1,
    minimum_height_pixels: int = 1,
    boundary_tolerance_pixels: int = 2,
    crs: CRS | None = None,
    transform: Affine | None = None,
) -> dict[str, Any]:
    """Evaluate one HighBuild tile without treating unknown pixels as ground."""

    target = np.asarray(reference_height_m, dtype=np.float32)
    prediction_height = np.asarray(prediction_height_m, dtype=np.float32)
    prediction_building_values = np.asarray(prediction_building)
    image_valid = np.asarray(image_valid_mask, dtype=bool)
    semantic = np.asarray(building_footprints, dtype=bool)
    height_valid = np.asarray(height_valid_mask, dtype=bool)
    shape = target.shape
    if target.ndim != 2 or any(
        array.shape != shape
        for array in (
            prediction_height,
            prediction_building_values,
            image_valid,
            semantic,
            height_valid,
        )
    ):
        raise HighBuildInstanceEvaluationError(
            "All HighBuild evaluation arrays must be aligned two-dimensional grids"
        )
    if minimum_height_pixels <= 0:
        raise HighBuildInstanceEvaluationError("minimum_height_pixels must be positive")
    if np.any(height_valid & ~image_valid):
        raise HighBuildInstanceEvaluationError(
            "Height-valid support extends outside independent image-valid support"
        )
    if np.any(semantic & ~image_valid):
        raise HighBuildInstanceEvaluationError(
            "Building outlines extend outside independent image-valid support"
        )
    if np.any(height_valid & ~semantic):
        raise HighBuildInstanceEvaluationError(
            "Height-valid support extends outside building outlines"
        )
    if not np.all(np.isfinite(prediction_height[height_valid])):
        raise HighBuildInstanceEvaluationError(
            "Prediction height is missing on valid HighBuild height support"
        )
    if not np.all(np.isfinite(prediction_building_values[semantic])):
        raise HighBuildInstanceEvaluationError(
            "Building prediction is missing on annotated HighBuild outlines"
        )

    classification_valid: np.ndarray | None = None
    if classification_valid_mask is not None:
        classification_valid = np.asarray(classification_valid_mask, dtype=bool)
        if classification_valid.shape != shape:
            raise HighBuildInstanceEvaluationError(
                "Classification-validity mask is not aligned"
            )
        if np.any(classification_valid & ~image_valid):
            raise HighBuildInstanceEvaluationError(
                "Classification-validity support extends outside image-valid support"
            )
        if np.any(semantic & ~classification_valid):
            raise HighBuildInstanceEvaluationError(
                "A building annotation falls outside classification-valid support"
            )
        if not np.all(np.isfinite(prediction_building_values[classification_valid])):
            raise HighBuildInstanceEvaluationError(
                "Building prediction is missing inside classification-valid support"
            )

    source_references = parse_highbuild_instances(
        coco, height=shape[0], width=shape[1]
    )
    validate_corrected_highbuild_contract(
        instances=source_references,
        reference_height_m=target,
        building_footprints=semantic,
        height_valid_mask=height_valid,
        height_protocol=height_protocol,
    )
    references = [reference for reference in source_references if np.any(reference.mask)]
    unrasterizable_references = len(source_references) - len(references)
    instance_rows: list[dict[str, Any]] = []
    for reference in references:
        height_eligible = reference.height_m is not None and (
            height_protocol == "inclusive_annotated"
            or not reference.is_estimated_height
        )
        instance_rows.append(
            {
                "annotation_id": reference.annotation_id,
                "provenance": reference.provenance,
                "estimated_flag_explicit": reference.estimated_flag_explicit,
                "reference_area_pixels": int(reference.mask.sum()),
                "reference_height_m": reference.height_m,
                "height_protocol_eligible": height_eligible,
                "matched": False,
                "prediction_instance_label": None,
                "outline_iou": None,
                "boundary_precision": None,
                "boundary_recall": None,
                "boundary_f1": None,
                "height_support_pixels": 0,
                "predicted_height_m": None,
                "height_error_m": None,
                "height_evaluation_status": "unmatched_detection",
            }
        )
    for reference in source_references:
        if np.any(reference.mask):
            continue
        instance_rows.append(
            {
                "annotation_id": reference.annotation_id,
                "provenance": reference.provenance,
                "estimated_flag_explicit": reference.estimated_flag_explicit,
                "reference_area_pixels": 0,
                "reference_height_m": reference.height_m,
                "height_protocol_eligible": False,
                "matched": False,
                "prediction_instance_label": None,
                "outline_iou": None,
                "boundary_precision": None,
                "boundary_recall": None,
                "boundary_f1": None,
                "height_support_pixels": 0,
                "predicted_height_m": None,
                "height_error_m": None,
                "height_evaluation_status": "unrasterizable_at_grid_resolution",
            }
        )

    component_validity = classification_valid if classification_valid is not None else image_valid
    predictions = _predicted_instance_masks(
        prediction_building_values,
        valid_mask=component_validity,
        probability_threshold=building_probability_threshold,
        minimum_pixels=minimum_predicted_instance_pixels,
        labels_are_instances=prediction_is_instance_labels,
    )
    matches = _match_instances(
        references, predictions, iou_threshold=instance_iou_threshold
    )
    prediction_masks = {label: mask for label, mask in predictions}

    outline_ious: list[float] = []
    boundary_precision: list[float] = []
    boundary_recall: list[float] = []
    boundary_f1: list[float] = []
    height_predictions: dict[str, list[float]] = {
        "combined": [],
        "measured": [],
        "estimated": [],
    }
    height_targets: dict[str, list[float]] = {
        "combined": [],
        "measured": [],
        "estimated": [],
    }
    matched_by_provenance = {"measured": 0, "estimated": 0, "missing_height": 0}
    for match in matches:
        reference = references[match.reference_index]
        instance_row = instance_rows[match.reference_index]
        prediction_mask = prediction_masks[match.prediction_label]
        outline_ious.append(match.iou)
        precision, recall, f1 = _boundary_scores(
            prediction_mask,
            reference.mask,
            tolerance_pixels=boundary_tolerance_pixels,
        )
        if precision is not None:
            boundary_precision.append(precision)
        if recall is not None:
            boundary_recall.append(recall)
        if f1 is not None:
            boundary_f1.append(f1)
        matched_by_provenance[reference.provenance] += 1
        instance_row.update(
            {
                "matched": True,
                "prediction_instance_label": match.prediction_label,
                "outline_iou": match.iou,
                "boundary_precision": precision,
                "boundary_recall": recall,
                "boundary_f1": f1,
            }
        )

        height_eligible = reference.height_m is not None and (
            height_protocol == "inclusive_annotated"
            or not reference.is_estimated_height
        )
        if not height_eligible:
            instance_row["height_evaluation_status"] = (
                "excluded_missing_reference_height"
                if reference.height_m is None
                else "excluded_estimated_by_strict_protocol"
            )
            continue
        instance_support = reference.mask & height_valid & image_valid
        height_support_pixels = int(instance_support.sum())
        instance_row["height_support_pixels"] = height_support_pixels
        if height_support_pixels < minimum_height_pixels:
            instance_row["height_evaluation_status"] = (
                "insufficient_valid_height_pixels"
            )
            continue
        predicted_instance_height = float(np.median(prediction_height[instance_support]))
        provenance = "estimated" if reference.is_estimated_height else "measured"
        height_predictions["combined"].append(predicted_instance_height)
        height_targets["combined"].append(float(reference.height_m))
        height_predictions[provenance].append(predicted_instance_height)
        height_targets[provenance].append(float(reference.height_m))
        instance_row.update(
            {
                "predicted_height_m": predicted_instance_height,
                "height_error_m": predicted_instance_height
                - float(reference.height_m),
                "height_evaluation_status": "evaluated",
            }
        )

    reference_by_provenance = {
        provenance: sum(instance.provenance == provenance for instance in references)
        for provenance in ("measured", "estimated", "missing_height")
    }
    eligible_by_provenance = {
        "measured": reference_by_provenance["measured"],
        "estimated": reference_by_provenance["estimated"]
        if height_protocol == "inclusive_annotated"
        else 0,
    }
    implicit_not_estimated_count = sum(
        not instance.estimated_flag_explicit for instance in source_references
    )
    provenance_report = {
        provenance: {
            "reference_instances": reference_by_provenance[provenance],
            "matched_instances": matched_by_provenance[provenance],
            "annotated_building_recall": _safe_ratio(
                matched_by_provenance[provenance],
                reference_by_provenance[provenance],
            ),
        }
        for provenance in ("measured", "estimated", "missing_height")
    }

    pixel_book = StreamingRegressionMetrics()
    pixel_book.update(prediction_height, target, height_valid & image_valid)
    pixel_metrics = pixel_book.compute() if pixel_book.count else None

    matched_count = len(matches)
    reference_count = len(references)
    prediction_count = len(predictions)
    matched_prediction_labels = {match.prediction_label for match in matches}
    unmatched_prediction_labels = [
        label for label, _ in predictions if label not in matched_prediction_labels
    ]
    recall = _safe_ratio(matched_count, reference_count)
    precision = (
        _safe_ratio(matched_count, prediction_count)
        if classification_valid is not None
        else None
    )
    detection_status = (
        "exhaustive_classification_support"
        if classification_valid is not None
        else "positive_annotations_only_precision_and_count_unavailable"
    )
    counting = _tile_counting_summary(
        reference_count=reference_count,
        prediction_count=prediction_count,
        exhaustive_reference=classification_valid is not None,
        prediction_is_instance_labels=prediction_is_instance_labels,
    )
    result: dict[str, Any] = {
        "sample_id": sample_id,
        "height_protocol": height_protocol,
        "thresholds": {
            "building_probability": None
            if prediction_is_instance_labels
            else building_probability_threshold,
            "instance_iou": instance_iou_threshold,
            "minimum_predicted_instance_pixels": minimum_predicted_instance_pixels,
            "minimum_height_pixels": minimum_height_pixels,
            "boundary_tolerance_pixels": boundary_tolerance_pixels,
        },
        "support": {
            "image_valid_pixels": int(image_valid.sum()),
            "classification_valid_pixels": int(classification_valid.sum())
            if classification_valid is not None
            else None,
            "height_valid_pixels": int(height_valid.sum()),
            "source_annotations": len(source_references),
            "unrasterizable_annotations_excluded": unrasterizable_references,
            "reference_instances": reference_count,
            "predicted_instances": prediction_count,
            "matched_instances": matched_count,
            "reference_by_provenance": reference_by_provenance,
            "matched_by_provenance": matched_by_provenance,
            "height_eligible_by_provenance": eligible_by_provenance,
            "implicit_not_estimated_annotations": implicit_not_estimated_count,
        },
        "pixel_height": {
            "status": "available" if pixel_metrics is not None else "no_height_support",
            "metrics": pixel_metrics,
        },
        "detection": {
            "status": detection_status,
            "precision": precision,
            "recall": recall,
            "annotated_building_recall": recall,
            "f1": _harmonic_mean(precision, recall),
            "count_error_predicted_minus_reference": prediction_count - reference_count
            if classification_valid is not None
            else None,
            "unmatched_prediction_labels": unmatched_prediction_labels,
            "unmatched_predictions_interpretation": (
                "false_positives_within_exhaustive_support"
                if classification_valid is not None
                else "unknown_without_exhaustive_negative_labels"
            ),
            "per_reference_provenance": provenance_report,
        },
        "counting": counting,
        "outline": {
            "matched_instances": matched_count,
            "mean_iou": float(np.mean(outline_ious)) if outline_ious else None,
            "median_iou": float(np.median(outline_ious)) if outline_ious else None,
            "mean_boundary_precision": float(np.mean(boundary_precision))
            if boundary_precision
            else None,
            "mean_boundary_recall": float(np.mean(boundary_recall))
            if boundary_recall
            else None,
            "mean_boundary_f1": float(np.mean(boundary_f1)) if boundary_f1 else None,
        },
        "instance_height": {
            "aggregation": "median_prediction_over_reference_outline_and_valid_height_support",
            "combined": _regression_summary(
                height_predictions["combined"], height_targets["combined"]
            ),
            "measured": _regression_summary(
                height_predictions["measured"], height_targets["measured"]
            ),
            "estimated": {
                **_regression_summary(
                    height_predictions["estimated"], height_targets["estimated"]
                ),
                "status": "included_separately"
                if height_protocol == "inclusive_annotated"
                else "excluded_by_strict_measured_protocol",
            },
        },
        "per_building": instance_rows,
        # Private aggregation payload; the dataset accumulator removes it from
        # serialized scene rows.
        "_aggregation": {
            "height_predictions": height_predictions,
            "height_targets": height_targets,
            "outline_ious": outline_ious,
            "boundary_precision": boundary_precision,
            "boundary_recall": boundary_recall,
            "boundary_f1": boundary_f1,
        },
    }

    pixel_area_m2 = authenticated_pixel_area_m2(crs=crs, transform=transform)
    if pixel_area_m2 is None:
        result["coverage"] = {
            "status": "unavailable_without_projected_crs_and_authenticated_scale"
        }
    else:
        predicted_union = np.zeros(shape, dtype=bool)
        for _, mask in predictions:
            predicted_union |= mask
        result["coverage"] = {
            "status": "available",
            "pixel_area_m2": pixel_area_m2,
            "reference_building_area_m2": float(semantic.sum() * pixel_area_m2),
            "predicted_building_area_m2": float(predicted_union.sum() * pixel_area_m2),
        }
    return result


class HighBuildInstanceAccumulator:
    """Accumulate exact counts and reproducible macro instance metrics."""

    def __init__(self, *, height_protocol: str) -> None:
        self.height_protocol = height_protocol
        self.pixel_metrics = StreamingRegressionMetrics()
        self.scenes: list[dict[str, Any]] = []
        self.reference_instances = 0
        self.source_annotations = 0
        self.unrasterizable_annotations = 0
        self.predicted_instances = 0
        self.matched_instances = 0
        self.reference_by_provenance = {
            "measured": 0,
            "estimated": 0,
            "missing_height": 0,
        }
        self.matched_by_provenance = dict(self.reference_by_provenance)
        self.implicit_not_estimated_annotations = 0
        self.height_predictions = {"combined": [], "measured": [], "estimated": []}
        self.height_targets = {"combined": [], "measured": [], "estimated": []}
        self.outline_ious: list[float] = []
        self.boundary_precision: list[float] = []
        self.boundary_recall: list[float] = []
        self.boundary_f1: list[float] = []
        self._classification_is_exhaustive: bool | None = None
        self._count_reference_by_scene: list[int] = []
        self._count_prediction_by_scene: list[int] = []
        self._prediction_representations: set[str] = set()
        self._coverage_all_authenticated = True
        self._reference_area_m2 = 0.0
        self._predicted_area_m2 = 0.0

    def update(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("height_protocol") != self.height_protocol:
            raise HighBuildInstanceEvaluationError(
                "Accumulator and tile height protocols differ"
            )
        result = evaluate_highbuild_tile(**kwargs)
        support = result["support"]
        self.source_annotations += int(support["source_annotations"])
        self.unrasterizable_annotations += int(
            support["unrasterizable_annotations_excluded"]
        )
        self.reference_instances += int(support["reference_instances"])
        self.predicted_instances += int(support["predicted_instances"])
        self.matched_instances += int(support["matched_instances"])
        for provenance in self.reference_by_provenance:
            self.reference_by_provenance[provenance] += int(
                support["reference_by_provenance"][provenance]
            )
            self.matched_by_provenance[provenance] += int(
                support["matched_by_provenance"][provenance]
            )
        self.implicit_not_estimated_annotations += int(
            support["implicit_not_estimated_annotations"]
        )
        exhaustive = kwargs.get("classification_valid_mask") is not None
        if self._classification_is_exhaustive is None:
            self._classification_is_exhaustive = exhaustive
        elif self._classification_is_exhaustive != exhaustive:
            raise HighBuildInstanceEvaluationError(
                "Cannot mix exhaustive and positive-only annotation contracts in one report"
            )
        counting = result["counting"]
        self._prediction_representations.add(
            str(counting["prediction_representation"])
        )
        if exhaustive:
            self._count_reference_by_scene.append(int(counting["reference_count"]))
            self._count_prediction_by_scene.append(
                int(counting["predicted_instances"])
            )
        prediction_height = np.asarray(kwargs["prediction_height_m"])
        reference_height = np.asarray(kwargs["reference_height_m"])
        height_valid = np.asarray(kwargs["height_valid_mask"], dtype=bool)
        image_valid = np.asarray(kwargs["image_valid_mask"], dtype=bool)
        self.pixel_metrics.update(
            prediction_height, reference_height, height_valid & image_valid
        )
        aggregation = result.pop("_aggregation")
        for provenance in self.height_predictions:
            self.height_predictions[provenance].extend(
                aggregation["height_predictions"][provenance]
            )
            self.height_targets[provenance].extend(
                aggregation["height_targets"][provenance]
            )
        self.outline_ious.extend(aggregation["outline_ious"])
        self.boundary_precision.extend(aggregation["boundary_precision"])
        self.boundary_recall.extend(aggregation["boundary_recall"])
        self.boundary_f1.extend(aggregation["boundary_f1"])
        coverage = result["coverage"]
        if coverage["status"] != "available":
            self._coverage_all_authenticated = False
        else:
            self._reference_area_m2 += float(coverage["reference_building_area_m2"])
            self._predicted_area_m2 += float(coverage["predicted_building_area_m2"])
        self.scenes.append(result)
        return result

    def compute(self) -> dict[str, Any]:
        if not self.scenes:
            raise HighBuildInstanceEvaluationError("No HighBuild tiles were evaluated")
        recall = _safe_ratio(self.matched_instances, self.reference_instances)
        precision = (
            _safe_ratio(self.matched_instances, self.predicted_instances)
            if self._classification_is_exhaustive
            else None
        )
        coverage: dict[str, Any]
        if self._coverage_all_authenticated:
            coverage = {
                "status": "available",
                "reference_building_area_m2": self._reference_area_m2,
                "predicted_building_area_m2": self._predicted_area_m2,
            }
        else:
            coverage = {
                "status": "unavailable_unless_every_tile_has_projected_crs_and_authenticated_scale"
            }
        provenance_report = {
            provenance: {
                "reference_instances": self.reference_by_provenance[provenance],
                "matched_instances": self.matched_by_provenance[provenance],
                "annotated_building_recall": _safe_ratio(
                    self.matched_by_provenance[provenance],
                    self.reference_by_provenance[provenance],
                ),
            }
            for provenance in ("measured", "estimated", "missing_height")
        }
        if self._classification_is_exhaustive:
            reference_counts = np.asarray(
                self._count_reference_by_scene, dtype=np.float64
            )
            prediction_counts = np.asarray(
                self._count_prediction_by_scene, dtype=np.float64
            )
            if reference_counts.size != len(self.scenes) or (
                prediction_counts.shape != reference_counts.shape
            ):
                raise HighBuildInstanceEvaluationError(
                    "Exhaustive counting support is incomplete"
                )
            count_errors = prediction_counts - reference_counts
            absolute_count_errors = np.abs(count_errors)
            positive_reference = reference_counts > 0
            counting: dict[str, Any] = {
                "status": "validated_against_exhaustive_instance_reference",
                "metric_source": "instance_counts_scored_separately_from_pixel_metrics",
                "prediction_representations": sorted(
                    self._prediction_representations
                ),
                "scene_count": len(self.scenes),
                "total_reference_instances": int(reference_counts.sum()),
                "total_predicted_instances": int(prediction_counts.sum()),
                "total_signed_error_predicted_minus_reference": int(
                    count_errors.sum()
                ),
                "per_scene_mae": float(absolute_count_errors.mean()),
                "per_scene_rmse": float(
                    np.sqrt(np.square(count_errors).mean())
                ),
                "per_scene_median_absolute_error": float(
                    np.median(absolute_count_errors)
                ),
                "exact_count_scene_rate": float(np.mean(count_errors == 0)),
                "overcount_scene_count": int(np.count_nonzero(count_errors > 0)),
                "undercount_scene_count": int(np.count_nonzero(count_errors < 0)),
                "exact_count_scene_count": int(np.count_nonzero(count_errors == 0)),
                "mean_absolute_percentage_error": (
                    float(
                        np.mean(
                            absolute_count_errors[positive_reference]
                            / reference_counts[positive_reference]
                        )
                    )
                    if np.any(positive_reference)
                    else None
                ),
                "percentage_error_scene_count": int(
                    np.count_nonzero(positive_reference)
                ),
            }
        else:
            counting = {
                "status": "unavailable_positive_annotations_are_not_exhaustive",
                "metric_source": "instance_counts_scored_separately_from_pixel_metrics",
                "prediction_representations": sorted(
                    self._prediction_representations
                ),
                "scene_count": len(self.scenes),
                "annotated_reference_instances": self.reference_instances,
                "predicted_instances": self.predicted_instances,
                "total_reference_instances": None,
                "total_signed_error_predicted_minus_reference": None,
                "per_scene_mae": None,
                "per_scene_rmse": None,
                "per_scene_median_absolute_error": None,
                "exact_count_scene_rate": None,
                "reason": (
                    "HighBuild supplies positive outlines but does not certify "
                    "that every building in the evaluated area is annotated"
                ),
            }
        return {
            "height_protocol": self.height_protocol,
            "tile_count": len(self.scenes),
            "contract": {
                "outside_coco_footprints": "unknown_not_ground",
                "estimated_heights": "reported_separately",
                "classification_support": "exhaustive"
                if self._classification_is_exhaustive
                else "positive_annotations_only",
            },
            "support": {
                "source_annotations": self.source_annotations,
                "unrasterizable_annotations_excluded": (
                    self.unrasterizable_annotations
                ),
                "reference_instances": self.reference_instances,
                "predicted_instances": self.predicted_instances,
                "matched_instances": self.matched_instances,
                "reference_by_provenance": self.reference_by_provenance,
                "matched_by_provenance": self.matched_by_provenance,
                "implicit_not_estimated_annotations": (
                    self.implicit_not_estimated_annotations
                ),
                "height_valid_pixels": self.pixel_metrics.count,
            },
            "pixel_height": self.pixel_metrics.compute()
            if self.pixel_metrics.count
            else None,
            "detection": {
                "status": "exhaustive_classification_support"
                if self._classification_is_exhaustive
                else "positive_annotations_only_precision_and_count_unavailable",
                "precision": precision,
                "recall": recall,
                "annotated_building_recall": recall,
                "f1": _harmonic_mean(precision, recall),
                "count_error_predicted_minus_reference": self.predicted_instances
                - self.reference_instances
                if self._classification_is_exhaustive
                else None,
                "per_reference_provenance": provenance_report,
            },
            "counting": counting,
            "outline": {
                "matched_instances": self.matched_instances,
                "mean_iou": float(np.mean(self.outline_ious))
                if self.outline_ious
                else None,
                "median_iou": float(np.median(self.outline_ious))
                if self.outline_ious
                else None,
                "mean_boundary_precision": float(np.mean(self.boundary_precision))
                if self.boundary_precision
                else None,
                "mean_boundary_recall": float(np.mean(self.boundary_recall))
                if self.boundary_recall
                else None,
                "mean_boundary_f1": float(np.mean(self.boundary_f1))
                if self.boundary_f1
                else None,
            },
            "instance_height": {
                "aggregation": "median_prediction_over_reference_outline_and_valid_height_support",
                "combined": _regression_summary(
                    self.height_predictions["combined"], self.height_targets["combined"]
                ),
                "measured": _regression_summary(
                    self.height_predictions["measured"], self.height_targets["measured"]
                ),
                "estimated": {
                    **_regression_summary(
                        self.height_predictions["estimated"],
                        self.height_targets["estimated"],
                    ),
                    "status": "included_separately"
                    if self.height_protocol == "inclusive_annotated"
                    else "excluded_by_strict_measured_protocol",
                },
            },
            "coverage": coverage,
            "scenes": self.scenes,
        }
