"""Target-free blending of a general height model and a building specialist."""

from __future__ import annotations

import numpy as np


def specialist_weight(
    building_probability: np.ndarray,
    *,
    threshold: float,
    transition: float = 0.0,
) -> np.ndarray:
    """Return a hard or smooth specialist gate from predicted probability."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if transition < 0.0:
        raise ValueError("transition must be non-negative")
    probability = np.asarray(building_probability, dtype=np.float32)
    if transition == 0.0:
        return (probability >= threshold).astype(np.float32)
    lower = threshold - transition / 2.0
    upper = threshold + transition / 2.0
    scaled = np.clip((probability - lower) / max(upper - lower, 1e-6), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def blend_height_predictions(
    baseline_height: np.ndarray,
    specialist_height: np.ndarray,
    building_probability: np.ndarray,
    *,
    threshold: float,
    transition: float = 0.0,
    maximum_delta_m: float | None = 20.0,
) -> np.ndarray:
    """Blend predictions using only model outputs available at inference time."""

    baseline = np.asarray(baseline_height, dtype=np.float32)
    specialist = np.asarray(specialist_height, dtype=np.float32)
    probability = np.asarray(building_probability, dtype=np.float32)
    if baseline.shape != specialist.shape or baseline.shape != probability.shape:
        raise ValueError("baseline, specialist, and probability must share a grid")
    delta = specialist - baseline
    if maximum_delta_m is not None:
        if maximum_delta_m <= 0:
            raise ValueError("maximum_delta_m must be positive when supplied")
        delta = np.clip(delta, -maximum_delta_m, maximum_delta_m)
    weight = specialist_weight(
        probability,
        threshold=threshold,
        transition=transition,
    )
    blended = baseline + weight * delta
    valid = np.isfinite(baseline) & np.isfinite(specialist) & np.isfinite(probability)
    blended[~valid] = np.nan
    return blended.astype(np.float32, copy=False)
