import numpy as np
import pytest

from msr.inference.relative_depth import robust_normalize_depth


def test_robust_normalize_depth_clips_outliers_and_preserves_invalid() -> None:
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    depth[0, 0] = -1000
    valid = np.ones((10, 10), dtype=bool)
    valid[3, 4] = False

    result = robust_normalize_depth(depth, valid_mask=valid, low_percentile=5, high_percentile=95)

    assert result.shape == depth.shape
    assert np.nanmin(result) == pytest.approx(0.0)
    assert np.nanmax(result) == pytest.approx(1.0)
    assert np.isnan(result[3, 4])
    assert result[8, 8] > result[2, 2]


def test_robust_normalize_depth_can_invert_orientation() -> None:
    depth = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    forward = robust_normalize_depth(depth, low_percentile=0, high_percentile=100)
    inverted = robust_normalize_depth(
        depth, low_percentile=0, high_percentile=100, near_is_high=False
    )
    np.testing.assert_allclose(forward + inverted, 1.0)
