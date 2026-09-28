"""Streaming statistics used by validation error analysis.

The validation set contains hundreds of millions of pixels, so these helpers
retain sufficient statistics instead of complete prediction rasters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegressionSums:
    """Sufficient statistics for exact regression metrics and affine transforms."""

    count: int = 0
    sum_prediction: float = 0.0
    sum_target: float = 0.0
    sum_prediction_squared: float = 0.0
    sum_target_squared: float = 0.0
    sum_cross: float = 0.0
    sum_error: float = 0.0
    sum_absolute_error: float = 0.0
    sum_squared_error: float = 0.0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes differ")
        valid = np.isfinite(prediction) & np.isfinite(target)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != target.shape:
                raise ValueError("Mask and target shapes differ")
            valid &= mask
        if not np.any(valid):
            return

        predicted = prediction[valid]
        observed = target[valid]
        error = predicted - observed
        self.count += int(error.size)
        self.sum_prediction += float(predicted.sum())
        self.sum_target += float(observed.sum())
        self.sum_prediction_squared += float(np.square(predicted).sum())
        self.sum_target_squared += float(np.square(observed).sum())
        self.sum_cross += float((predicted * observed).sum())
        self.sum_error += float(error.sum())
        self.sum_absolute_error += float(np.abs(error).sum())
        self.sum_squared_error += float(np.square(error).sum())

    def merge(self, other: "RegressionSums") -> "RegressionSums":
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    def affine_fit(self) -> tuple[float, float]:
        """Return least-squares ``target ~= scale * prediction + offset``."""

        if self.count == 0:
            raise ValueError("No values were accumulated")
        count = float(self.count)
        prediction_variance_sum = (
            self.sum_prediction_squared - self.sum_prediction**2 / count
        )
        covariance_sum = self.sum_cross - self.sum_prediction * self.sum_target / count
        if prediction_variance_sum <= 0.0:
            raise ValueError("Cannot calibrate a constant prediction")
        scale = covariance_sum / prediction_variance_sum
        offset = (self.sum_target - scale * self.sum_prediction) / count
        return float(scale), float(offset)

    def transformed(self, scale: float, offset: float) -> "RegressionSums":
        """Return sums after applying an affine transform to predictions.

        Absolute error cannot be reconstructed from sufficient statistics after
        a transform, so it is represented as NaN. RMSE, bias, correlation, and
        R-squared remain exact.
        """

        count = float(self.count)
        sum_prediction = scale * self.sum_prediction + offset * count
        sum_prediction_squared = (
            scale**2 * self.sum_prediction_squared
            + 2.0 * scale * offset * self.sum_prediction
            + offset**2 * count
        )
        sum_cross = scale * self.sum_cross + offset * self.sum_target
        sum_error = sum_prediction - self.sum_target
        sum_squared_error = (
            sum_prediction_squared + self.sum_target_squared - 2.0 * sum_cross
        )
        return RegressionSums(
            count=self.count,
            sum_prediction=sum_prediction,
            sum_target=self.sum_target,
            sum_prediction_squared=sum_prediction_squared,
            sum_target_squared=self.sum_target_squared,
            sum_cross=sum_cross,
            sum_error=sum_error,
            sum_absolute_error=float("nan"),
            sum_squared_error=max(float(sum_squared_error), 0.0),
        )

    def metrics(self) -> dict[str, int | float | None]:
        if self.count == 0:
            raise ValueError("No values were accumulated")
        count = float(self.count)
        prediction_variance_sum = max(
            self.sum_prediction_squared - self.sum_prediction**2 / count, 0.0
        )
        target_variance_sum = max(
            self.sum_target_squared - self.sum_target**2 / count, 0.0
        )
        covariance_sum = self.sum_cross - self.sum_prediction * self.sum_target / count
        correlation = None
        if prediction_variance_sum > 0.0 and target_variance_sum > 0.0:
            correlation = covariance_sum / np.sqrt(
                prediction_variance_sum * target_variance_sum
            )
        r2 = None
        if target_variance_sum > 0.0:
            r2 = 1.0 - self.sum_squared_error / target_variance_sum
        mae = (
            None
            if not np.isfinite(self.sum_absolute_error)
            else self.sum_absolute_error / count
        )
        return {
            "pixel_count": self.count,
            "rmse_m": float(np.sqrt(max(self.sum_squared_error, 0.0) / count)),
            "mae_m": float(mae) if mae is not None else None,
            "bias_m": self.sum_error / count,
            "correlation": float(correlation) if correlation is not None else None,
            "r2": float(r2) if r2 is not None else None,
        }


@dataclass
class BinarySums:
    """Streaming binary-classification confusion counts."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(
        self,
        probability: np.ndarray,
        target: np.ndarray,
        valid: np.ndarray,
        *,
        threshold: float,
    ) -> None:
        predicted = np.asarray(probability) >= threshold
        observed = np.asarray(target, dtype=bool)
        valid = np.asarray(valid, dtype=bool)
        self.true_positive += int(np.count_nonzero(valid & predicted & observed))
        self.false_positive += int(np.count_nonzero(valid & predicted & ~observed))
        self.false_negative += int(np.count_nonzero(valid & ~predicted & observed))
        self.true_negative += int(np.count_nonzero(valid & ~predicted & ~observed))

    def metrics(self) -> dict[str, int | float]:
        tp = self.true_positive
        fp = self.false_positive
        fn = self.false_negative
        tn = self.true_negative
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "f1": f1,
        }
