"""The inference experiment must compare the same inputs and pixel support."""
import numpy as np
import pytest
import torch

from scripts.audit_protected_inference_protocol_v1 import (
    metric_deltas,
    require_same_normalized_rgb,
)
from msr.data.raster_dataset import IMAGENET_MEAN, IMAGENET_STD


def test_raw_source_must_reproduce_authenticated_normalized_input():
    raw = np.random.default_rng(42).integers(0, 256, (3, 17, 13), dtype=np.uint8)
    sample = {"image": torch.from_numpy((raw.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)}
    require_same_normalized_rgb(raw, sample)
    changed = raw.copy()
    changed[0, 0, 0] = (int(changed[0, 0, 0]) + 1) % 256
    with pytest.raises(ValueError, match="Raw RGB differs"):
        require_same_normalized_rgb(changed, sample)


def test_metric_delta_retains_pixel_count_diagnostic_and_domains():
    native = {"pixel_count": 12, "rmse_m": 5.0, "domains": {"building": {"rmse_m": 7.0}}}
    tiled = {"pixel_count": 12, "rmse_m": 4.0, "domains": {"building": {"rmse_m": 6.5}}}
    assert metric_deltas(tiled, native) == {"pixel_count": 0, "rmse_m": -1.0, "domains": {"building": {"rmse_m": -0.5}}}
