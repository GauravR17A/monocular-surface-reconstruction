"""Masked, metric-unit evaluation for dense height rasters."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HeightMetrics:
    pixel_count: int
    rmse_m: float
    mae_m: float
    bias_m: float
    median_ae_m: float
    p90_ae_m: float
    p95_ae_m: float
    correlation: float | None
    r2: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


class StreamingRegressionMetrics:
    """Accumulate exact scalar metrics without retaining full validation rasters."""

    def __init__(self) -> None:
        self.count = 0
        self.sum_error = 0.0
        self.sum_absolute_error = 0.0
        self.sum_squared_error = 0.0
        self.sum_prediction = 0.0
        self.sum_target = 0.0
        self.sum_prediction_squared = 0.0
        self.sum_target_squared = 0.0
        self.sum_cross = 0.0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes differ")
        valid = np.isfinite(prediction) & np.isfinite(target)
        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            if valid_mask.shape != target.shape:
                raise ValueError("Validity mask and target shapes differ")
            valid &= valid_mask
        if not np.any(valid):
            return

        predicted = prediction[valid]
        observed = target[valid]
        error = predicted - observed
        self.count += int(error.size)
        self.sum_error += float(error.sum())
        self.sum_absolute_error += float(np.abs(error).sum())
        self.sum_squared_error += float(np.square(error).sum())
        self.sum_prediction += float(predicted.sum())
        self.sum_target += float(observed.sum())
        self.sum_prediction_squared += float(np.square(predicted).sum())
        self.sum_target_squared += float(np.square(observed).sum())
        self.sum_cross += float((predicted * observed).sum())

    def compute(self) -> dict[str, int | float | None]:
        if self.count == 0:
            raise ValueError("No valid pixels were accumulated")
        count = float(self.count)
        prediction_variance_sum = self.sum_prediction_squared - self.sum_prediction**2 / count
        target_variance_sum = self.sum_target_squared - self.sum_target**2 / count
        covariance_sum = self.sum_cross - self.sum_prediction * self.sum_target / count
        if prediction_variance_sum > 0.0 and target_variance_sum > 0.0:
            correlation = covariance_sum / np.sqrt(
                prediction_variance_sum * target_variance_sum
            )
        else:
            correlation = None
        r2 = (
            1.0 - self.sum_squared_error / target_variance_sum
            if target_variance_sum > 0.0
            else None
        )
        return {
            "pixel_count": self.count,
            "rmse_m": float(np.sqrt(self.sum_squared_error / count)),
            "mae_m": self.sum_absolute_error / count,
            "bias_m": self.sum_error / count,
            "correlation": float(correlation) if correlation is not None else None,
            "r2": float(r2) if r2 is not None else None,
        }


def compute_height_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> HeightMetrics:
    """Compute regression metrics after excluding invalid prediction/target pixels.

    Bias is defined as ``prediction - target``; positive bias means systematic
    overestimation. Correlation and R-squared are undefined for a constant target
    and are returned as ``None`` in that case.
    """

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )

    valid = np.isfinite(prediction) & np.isfinite(target)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != target.shape:
            raise ValueError(
                f"Validity mask and target shapes differ: {valid_mask.shape} != {target.shape}"
            )
        valid &= valid_mask
    if not np.any(valid):
        raise ValueError("No valid pixels remain for evaluation")

    predicted = prediction[valid]
    observed = target[valid]
    error = predicted - observed
    absolute_error = np.abs(error)
    squared_error = np.square(error)

    target_centered = observed - observed.mean()
    prediction_centered = predicted - predicted.mean()
    target_ss = float(np.sum(np.square(target_centered)))
    prediction_ss = float(np.sum(np.square(prediction_centered)))
    if target_ss > 0.0 and prediction_ss > 0.0:
        correlation = float(
            np.sum(target_centered * prediction_centered)
            / np.sqrt(target_ss * prediction_ss)
        )
    else:
        correlation = None
    r2 = float(1.0 - squared_error.sum() / target_ss) if target_ss > 0.0 else None

    return HeightMetrics(
        pixel_count=int(error.size),
        rmse_m=float(np.sqrt(squared_error.mean())),
        mae_m=float(absolute_error.mean()),
        bias_m=float(error.mean()),
        median_ae_m=float(np.median(absolute_error)),
        p90_ae_m=float(np.percentile(absolute_error, 90)),
        p95_ae_m=float(np.percentile(absolute_error, 95)),
        correlation=correlation,
        r2=r2,
    )
