"""Radiometric transforms shared by training and presentation previews.

Satellite products from different sensors can encode the same visible scene at
very different numeric brightness ranges. These helpers keep sensor style
separate from geometric augmentation so robustness is explicit and testable.
"""

from __future__ import annotations

import numpy as np
import torch


RADIOMETRIC_POLICIES = {"raw", "percentile_stretch", "random_sensor"}


def percentile_stretch_rgb(
    image: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
    output_scale: float = 255.0,
) -> np.ndarray:
    """Stretch each CHW RGB band robustly while preserving invalid pixels."""

    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError(f"Expected CHW RGB data, received {values.shape}")
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    if output_scale <= 0:
        raise ValueError("output_scale must be positive")

    spatial_shape = values.shape[-2:]
    valid = np.ones(spatial_shape, dtype=bool)
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != spatial_shape:
            raise ValueError("valid_mask must match the RGB spatial dimensions")

    output = np.zeros_like(values, dtype=np.float32)
    for band_index in range(3):
        band = values[band_index]
        finite = valid & np.isfinite(band)
        if not np.any(finite):
            continue
        low, high = np.percentile(band[finite], [low_percentile, high_percentile])
        span = float(high - low)
        if span <= np.finfo(np.float32).eps:
            continue
        output[band_index, finite] = (
            np.clip((band[finite] - low) / span, 0.0, 1.0) * output_scale
        )
    return output


def apply_radiometric_policy(
    image: np.ndarray,
    *,
    valid_mask: np.ndarray | None,
    policy: str,
    rgb_scale: float = 255.0,
) -> np.ndarray:
    """Apply a deterministic or stochastic sensor-style RGB transform."""

    if policy not in RADIOMETRIC_POLICIES:
        raise ValueError(
            f"Unknown radiometric policy {policy!r}; expected "
            f"one of {sorted(RADIOMETRIC_POLICIES)}"
        )
    values = np.asarray(image, dtype=np.float32)
    if policy == "raw":
        return values.copy()
    stretched = percentile_stretch_rgb(
        values,
        valid_mask=valid_mask,
        output_scale=rgb_scale,
    )
    if policy == "percentile_stretch":
        return stretched

    # Retain raw radiometry regularly so robustness does not cost performance
    # on original sensor products. Other branches remove the brightness
    # shortcut between the HighBuild and Open-Canopy sensors.
    choice = float(torch.rand(()))
    if choice < 0.35:
        normalized = np.clip(values / rgb_scale, 0.0, 1.0)
    else:
        normalized = np.clip(stretched / rgb_scale, 0.0, 1.0)
        if choice >= 0.70:
            black_level = float(torch.empty(()).uniform_(0.0, 0.10))
            dynamic_range = float(torch.empty(()).uniform_(0.20, 0.90))
            normalized = black_level + dynamic_range * normalized

    gamma = float(torch.empty(()).uniform_(0.70, 1.45))
    normalized = np.clip(normalized, 0.0, 1.0) ** gamma
    contrast = float(torch.empty(()).uniform_(0.75, 1.30))
    brightness = float(torch.empty(()).uniform_(-0.10, 0.10))
    channel_gain = torch.empty(3).uniform_(0.82, 1.18).numpy()[:, None, None]
    channel_bias = torch.empty(3).uniform_(-0.035, 0.035).numpy()[:, None, None]
    mean = np.mean(normalized, axis=(-2, -1), keepdims=True)
    normalized = (
        (normalized - mean) * contrast + mean + brightness
    ) * channel_gain + channel_bias
    if float(torch.rand(())) < 0.30:
        noise_sigma = float(torch.empty(()).uniform_(0.0, 0.012))
        normalized = normalized + torch.randn(values.shape).numpy() * noise_sigma
    output = np.clip(normalized, 0.0, 1.0) * rgb_scale
    if valid_mask is not None:
        output[:, ~np.asarray(valid_mask, dtype=bool)] = 0.0
    return output.astype(np.float32, copy=False)
