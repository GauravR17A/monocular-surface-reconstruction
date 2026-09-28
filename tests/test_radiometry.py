import numpy as np
import pytest
import torch

from msr.api.app import _preview_rgb
from msr.data.radiometry import (
    apply_radiometric_policy,
    percentile_stretch_rgb,
)


def test_percentile_stretch_respects_mask_and_range() -> None:
    image = np.stack(
        [
            np.arange(16, dtype=np.float32).reshape(4, 4),
            np.arange(16, dtype=np.float32).reshape(4, 4) * 2,
            np.arange(16, dtype=np.float32).reshape(4, 4) * 3,
        ]
    )
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    result = percentile_stretch_rgb(
        image,
        valid_mask=valid,
        low_percentile=0,
        high_percentile=100,
    )
    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.all(result[:, ~valid] == 0)
    assert float(result.min()) >= 0
    assert float(result.max()) <= 255
    assert np.allclose(result[:, -1, -1], 255)


def test_random_sensor_policy_is_seeded_and_bounded() -> None:
    image = np.linspace(8, 73, 3 * 32 * 32, dtype=np.float32).reshape(3, 32, 32)
    valid = np.ones((32, 32), dtype=bool)
    torch.manual_seed(17)
    first = apply_radiometric_policy(
        image, valid_mask=valid, policy="random_sensor"
    )
    torch.manual_seed(17)
    second = apply_radiometric_policy(
        image, valid_mask=valid, policy="random_sensor"
    )
    assert np.array_equal(first, second)
    assert float(first.min()) >= 0
    assert float(first.max()) <= 255
    assert not np.array_equal(first, image)


def test_preview_uses_shared_one_to_ninety_nine_percent_stretch() -> None:
    image = np.linspace(0, 1000, 3 * 20 * 20, dtype=np.float32).reshape(3, 20, 20)
    expected = np.moveaxis(
        percentile_stretch_rgb(
            image,
            low_percentile=1,
            high_percentile=99,
        ).astype(np.uint8),
        0,
        -1,
    )
    assert np.array_equal(_preview_rgb(image), expected)


def test_unknown_radiometric_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown radiometric policy"):
        apply_radiometric_policy(
            np.zeros((3, 2, 2), dtype=np.float32),
            valid_mask=np.ones((2, 2), dtype=bool),
            policy="mystery",
        )
