"""Streaming multiclass-identification metrics for dense semantic rasters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def compute_multiclass_metrics(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, object]:
    """Compute dense-classification metrics from a row-truth confusion matrix."""

    names = tuple(str(name) for name in class_names)
    matrix = np.asarray(confusion_matrix, dtype=np.int64)
    expected_shape = (len(names), len(names))
    if matrix.shape != expected_shape:
        raise ValueError(
            f"confusion matrix must have shape {expected_shape}, found {matrix.shape}"
        )
    if np.any(matrix < 0):
        raise ValueError("confusion matrix counts cannot be negative")

    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    false_positive = predicted_count - true_positive
    false_negative = support - true_positive

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        values = np.zeros_like(numerator, dtype=np.float64)
        np.divide(
            numerator,
            denominator,
            out=values,
            where=denominator != 0,
        )
        return values

    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    f1 = ratio(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
    )
    iou = ratio(true_positive, true_positive + false_positive + false_negative)
    total = int(matrix.sum())
    correct = int(true_positive.sum())
    macro = {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "iou": float(iou.mean()),
    }
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "iou": float(iou[index]),
            "support_pixels": int(support[index]),
            "predicted_pixels": int(predicted_count[index]),
            "true_positive_pixels": int(true_positive[index]),
            "false_positive_pixels": int(false_positive[index]),
            "false_negative_pixels": int(false_negative[index]),
        }
        for index, name in enumerate(names)
    }
    accuracy = correct / total if total else 0.0
    return {
        "class_names": list(names),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": {
            "rows": "reference",
            "columns": "predicted",
        },
        "total_valid_pixels": total,
        "correct_pixels": correct,
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy if total else 0.0,
        "macro": macro,
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "macro_iou": macro["iou"],
        "per_class": per_class,
    }


class StreamingMulticlassMetrics:
    """Accumulate an exact confusion matrix without retaining prediction rasters.

    Rows in the stored confusion matrix are reference classes and columns are
    predicted classes. Invalid/ignored pixels are excluded before accumulation.
    Undefined per-class ratios use the conventional deterministic value ``0``.
    """

    def __init__(self, class_names: Sequence[str]) -> None:
        names = tuple(str(name) for name in class_names)
        if not names:
            raise ValueError("class_names must contain at least one class")
        if len(set(names)) != len(names):
            raise ValueError("class_names must be unique")
        self.class_names = names
        self.confusion_matrix = np.zeros(
            (len(names), len(names)), dtype=np.int64
        )

    @property
    def count(self) -> int:
        return int(self.confusion_matrix.sum())

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        predicted = np.asarray(prediction)
        reference = np.asarray(target)
        if predicted.shape != reference.shape:
            raise ValueError(
                "prediction and target must have identical shapes; "
                f"received {predicted.shape} and {reference.shape}"
            )
        if valid_mask is None:
            valid = np.ones(reference.shape, dtype=bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool).copy()
            if valid.shape != reference.shape:
                raise ValueError(
                    "valid_mask must match prediction and target shapes; "
                    f"received {valid.shape} and {reference.shape}"
                )

        class_count = len(self.class_names)
        valid &= np.isfinite(predicted) & np.isfinite(reference)
        valid &= (reference >= 0) & (reference < class_count)
        valid &= (predicted >= 0) & (predicted < class_count)
        if not np.any(valid):
            return

        encoded = (
            reference[valid].astype(np.int64, copy=False) * class_count
            + predicted[valid].astype(np.int64, copy=False)
        )
        self.confusion_matrix += np.bincount(
            encoded, minlength=class_count * class_count
        ).reshape(class_count, class_count)

    def compute(self) -> dict[str, object]:
        return compute_multiclass_metrics(self.confusion_matrix, self.class_names)


def _binary_boundaries(
    class_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return one-pixel boundaries without crossing invalid-label gaps."""

    foreground = np.asarray(class_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if foreground.shape != valid.shape or foreground.ndim != 2:
        raise ValueError("boundary inputs must be same-shaped two-dimensional arrays")
    boundary = np.zeros_like(foreground)
    horizontal_pair = valid[:, 1:] & valid[:, :-1]
    horizontal_change = horizontal_pair & (
        foreground[:, 1:] != foreground[:, :-1]
    )
    boundary[:, 1:] |= horizontal_change
    boundary[:, :-1] |= horizontal_change
    vertical_pair = valid[1:, :] & valid[:-1, :]
    vertical_change = vertical_pair & (foreground[1:, :] != foreground[:-1, :])
    boundary[1:, :] |= vertical_change
    boundary[:-1, :] |= vertical_change
    return boundary & valid


def _dilate_mask(
    mask: np.ndarray,
    radius: int,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or radius < 0:
        raise ValueError("dilation needs a two-dimensional mask and nonnegative radius")
    valid = np.ones_like(values) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != values.shape:
        raise ValueError("valid_mask must match the mask being dilated")
    result = values & valid
    for _ in range(radius):
        expanded = result.copy()
        expanded[1:, :] |= result[:-1, :]
        expanded[:-1, :] |= result[1:, :]
        expanded[:, 1:] |= result[:, :-1]
        expanded[:, :-1] |= result[:, 1:]
        expanded[1:, 1:] |= result[:-1, :-1]
        expanded[1:, :-1] |= result[:-1, 1:]
        expanded[:-1, 1:] |= result[1:, :-1]
        expanded[:-1, :-1] |= result[1:, 1:]
        # Do not let a tolerance neighbourhood tunnel through unknown/no-data
        # label gaps and create a false road-boundary match on the other side.
        result = expanded & valid
    return result


class StreamingClassBoundaryMetrics:
    """Accumulate tolerance-aware boundary quality for one semantic class."""

    def __init__(self, class_index: int, *, tolerance_pixels: int = 2) -> None:
        if class_index < 0 or tolerance_pixels < 0:
            raise ValueError("class_index and tolerance_pixels must be nonnegative")
        self.class_index = int(class_index)
        self.tolerance_pixels = int(tolerance_pixels)
        self.predicted_boundary_pixels = 0
        self.reference_boundary_pixels = 0
        self.matched_predicted_pixels = 0
        self.matched_reference_pixels = 0
        self.dilated_intersection_pixels = 0
        self.dilated_union_pixels = 0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        predicted = np.asarray(prediction)
        reference = np.asarray(target)
        valid = np.asarray(valid_mask, dtype=bool)
        if predicted.shape != reference.shape or predicted.shape != valid.shape:
            raise ValueError("boundary prediction, target, and valid mask must match")
        if predicted.ndim == 2:
            predicted = predicted[None]
            reference = reference[None]
            valid = valid[None]
        if predicted.ndim != 3:
            raise ValueError("boundary inputs must have shape (H, W) or (N, H, W)")
        for predicted_item, reference_item, valid_item in zip(
            predicted, reference, valid, strict=True
        ):
            predicted_boundary = _binary_boundaries(
                predicted_item == self.class_index, valid_item
            )
            reference_boundary = _binary_boundaries(
                reference_item == self.class_index, valid_item
            )
            predicted_count = int(np.count_nonzero(predicted_boundary))
            reference_count = int(np.count_nonzero(reference_boundary))
            self.predicted_boundary_pixels += predicted_count
            self.reference_boundary_pixels += reference_count
            self.matched_predicted_pixels += int(
                np.count_nonzero(
                    predicted_boundary
                    & _dilate_mask(
                        reference_boundary,
                        self.tolerance_pixels,
                        valid_mask=valid_item,
                    )
                )
            )
            self.matched_reference_pixels += int(
                np.count_nonzero(
                    reference_boundary
                    & _dilate_mask(
                        predicted_boundary,
                        self.tolerance_pixels,
                        valid_mask=valid_item,
                    )
                )
            )
            dilated_prediction = _dilate_mask(
                predicted_boundary,
                self.tolerance_pixels,
                valid_mask=valid_item,
            )
            dilated_reference = _dilate_mask(
                reference_boundary,
                self.tolerance_pixels,
                valid_mask=valid_item,
            )
            self.dilated_intersection_pixels += int(
                np.count_nonzero(dilated_prediction & dilated_reference)
            )
            self.dilated_union_pixels += int(
                np.count_nonzero(dilated_prediction | dilated_reference)
            )

    def compute(self) -> dict[str, object]:
        precision = self.matched_predicted_pixels / max(
            self.predicted_boundary_pixels, 1
        )
        recall = self.matched_reference_pixels / max(
            self.reference_boundary_pixels, 1
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return {
            "class_index": self.class_index,
            "tolerance_pixels": self.tolerance_pixels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "dilated_boundary_iou": (
                self.dilated_intersection_pixels
                / max(self.dilated_union_pixels, 1)
            ),
            "predicted_boundary_pixels": self.predicted_boundary_pixels,
            "reference_boundary_pixels": self.reference_boundary_pixels,
            "matched_predicted_pixels": self.matched_predicted_pixels,
            "matched_reference_pixels": self.matched_reference_pixels,
            "dilated_intersection_pixels": self.dilated_intersection_pixels,
            "dilated_union_pixels": self.dilated_union_pixels,
        }


class StreamingWaterDarkPixelProxy:
    """Measure water false positives on dark non-water pixels.

    GAMUS contains water labels but no shadow labels.  Dark RGB pixels are only
    a reproducible proxy for likely shadow-like ambiguity, so the returned
    metadata states this limitation explicitly.
    """

    def __init__(self, water_class_index: int) -> None:
        if water_class_index < 0:
            raise ValueError("water_class_index must be nonnegative")
        self.water_class_index = int(water_class_index)
        self.dark_non_water_pixels = 0
        self.false_water_on_dark_non_water_pixels = 0
        self.all_false_water_pixels = 0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        dark_pixel_proxy_mask: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        predicted = np.asarray(prediction)
        reference = np.asarray(target)
        dark = np.asarray(dark_pixel_proxy_mask, dtype=bool)
        valid = np.asarray(valid_mask, dtype=bool)
        if not (
            predicted.shape == reference.shape == dark.shape == valid.shape
        ):
            raise ValueError("water proxy inputs must have identical shapes")
        non_water = valid & (reference != self.water_class_index)
        false_water = non_water & (predicted == self.water_class_index)
        dark_non_water = non_water & dark
        self.dark_non_water_pixels += int(np.count_nonzero(dark_non_water))
        self.false_water_on_dark_non_water_pixels += int(
            np.count_nonzero(false_water & dark)
        )
        self.all_false_water_pixels += int(np.count_nonzero(false_water))

    def compute(self) -> dict[str, object]:
        return {
            "is_shadow_proxy_not_ground_truth": True,
            "definition": (
                "water false positives on valid dark RGB pixels whose GAMUS "
                "reference class is not water"
            ),
            "limitation": (
                "GAMUS has no shadow class; dark pavement, roofs, and deep "
                "vegetation can also enter this proxy"
            ),
            "dark_non_water_pixels": self.dark_non_water_pixels,
            "false_water_on_dark_non_water_pixels": (
                self.false_water_on_dark_non_water_pixels
            ),
            "all_false_water_pixels": self.all_false_water_pixels,
            "false_water_rate_on_dark_non_water": (
                self.false_water_on_dark_non_water_pixels
                / max(self.dark_non_water_pixels, 1)
            ),
            "dark_share_of_all_water_false_positives": (
                self.false_water_on_dark_non_water_pixels
                / max(self.all_false_water_pixels, 1)
            ),
        }


def multiclass_metric_deltas(
    current: Mapping[str, object],
    reference: Mapping[str, object],
) -> dict[str, object]:
    """Return current-minus-reference deltas for higher-is-better metrics."""

    accuracy_delta = float(current["accuracy"]) - float(reference["accuracy"])
    reference_error = float(reference.get("error_rate", 1.0 - float(reference["accuracy"])))
    current_error = float(current.get("error_rate", 1.0 - float(current["accuracy"])))
    deltas: dict[str, object] = {
        "positive_score_delta_means_candidate_improved": True,
        "accuracy_delta": accuracy_delta,
        "accuracy_percentage_points": 100.0 * accuracy_delta,
        "relative_error_reduction": (
            (reference_error - current_error) / reference_error
            if reference_error > 0.0
            else 0.0
        ),
    }
    current_macro = current.get("macro")
    reference_macro = reference.get("macro")
    if not isinstance(current_macro, Mapping) or not isinstance(
        reference_macro, Mapping
    ):
        raise ValueError("Both metric mappings must contain macro mappings")
    deltas["macro"] = {
        metric_name: {
            "delta": float(current_macro[metric_name])
            - float(reference_macro[metric_name]),
            "percentage_points": 100.0
            * (
                float(current_macro[metric_name])
                - float(reference_macro[metric_name])
            ),
        }
        for metric_name in ("precision", "recall", "f1", "iou")
    }
    current_classes = current.get("per_class")
    reference_classes = reference.get("per_class")
    if not isinstance(current_classes, Mapping) or not isinstance(
        reference_classes, Mapping
    ):
        raise ValueError("Both metric mappings must contain per_class mappings")
    shared_classes = current_classes.keys() & reference_classes.keys()
    deltas["per_class"] = {
        str(class_name): {
            metric_name: {
                "delta": float(current_classes[class_name][metric_name])
                - float(reference_classes[class_name][metric_name]),
                "percentage_points": 100.0
                * (
                    float(current_classes[class_name][metric_name])
                    - float(reference_classes[class_name][metric_name])
                ),
            }
            for metric_name in ("precision", "recall", "f1", "iou")
        }
        for class_name in shared_classes
    }
    deltas["direction"] = "current_minus_initial; positive means improvement"
    return deltas
