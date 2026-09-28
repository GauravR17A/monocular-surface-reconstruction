"""Licensed pretrained monocular relative-depth extraction.

The output is deliberately dimensionless. Metric elevation is established by
the separate DEM/GCP calibration module; naming raw relative depth "metres"
would be scientifically incorrect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
import torch


DEFAULT_RELATIVE_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


@dataclass(frozen=True)
class RelativeDepthResult:
    relative_surface: np.ndarray
    raw_depth: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any]


def robust_normalize_depth(
    depth: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
    near_is_high: bool = True,
) -> np.ndarray:
    """Map scale-ambiguous depth to a robust [0, 1] relative-surface score."""

    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Depth must be 2-D, received shape {values.shape}")
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")

    valid = np.isfinite(values)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must have the same shape as depth")
        valid &= mask
    if not np.any(valid):
        raise ValueError("Depth contains no valid pixels")

    low, high = np.percentile(values[valid], [low_percentile, high_percentile])
    span = float(high - low)
    if not np.isfinite(span) or span <= np.finfo(np.float32).eps:
        normalized = np.zeros_like(values, dtype=np.float32)
    else:
        normalized = np.clip((values - low) / span, 0.0, 1.0).astype(np.float32)
    if not near_is_high:
        normalized = 1.0 - normalized
    normalized[~valid] = np.nan
    return normalized


class DepthAnythingV2Predictor:
    """Thin lazy-loading adapter around the Apache-2.0 Small HF checkpoint."""

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_RELATIVE_DEPTH_MODEL,
        device: str | torch.device = "cuda",
        near_is_high: bool = True,
    ) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as error:
            raise RuntimeError(
                "Depth Anything support is optional; install with "
                "`pip install -e .[foundation]`."
            ) from error

        self.device = torch.device(device)
        self.model_id = model_id
        self.near_is_high = near_is_high
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(
        self,
        rgb: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
    ) -> RelativeDepthResult:
        """Infer relative geometry at the original raster resolution."""

        array = np.asarray(rgb)
        if array.ndim != 3:
            raise ValueError(f"RGB input must be 3-D, received shape {array.shape}")
        if array.shape[0] == 3:
            hwc = np.moveaxis(array, 0, -1)
        elif array.shape[-1] >= 3:
            hwc = array[..., :3]
        else:
            raise ValueError(f"Could not find three RGB bands in shape {array.shape}")

        finite = np.all(np.isfinite(hwc), axis=-1)
        if valid_mask is not None:
            finite &= np.asarray(valid_mask, dtype=bool)
        clean = np.nan_to_num(hwc, copy=True)
        if clean.dtype != np.uint8:
            maximum = float(np.nanmax(clean)) if clean.size else 0.0
            scale = 255.0 if maximum <= 1.0 else 1.0
            clean = np.clip(clean * scale, 0, 255).astype(np.uint8)
        image = Image.fromarray(clean, mode="RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        processed = self.processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        raw = processed[0]["predicted_depth"].float().cpu().numpy().astype(np.float32)
        relative = robust_normalize_depth(
            raw,
            valid_mask=finite,
            near_is_high=self.near_is_high,
        )
        raw[~finite] = np.nan
        return RelativeDepthResult(
            relative_surface=relative,
            raw_depth=raw,
            valid_mask=finite,
            metadata={
                "model": self.model_id,
                "task": "relative_monocular_depth",
                "units": "dimensionless",
                "near_is_high": self.near_is_high,
                "license": "Apache-2.0",
                "warning": "Not metric elevation until calibrated with DEM or GCPs.",
            },
        )
