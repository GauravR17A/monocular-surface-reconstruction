import numpy as np
import pytest

from msr.inference.hybrid import blend_height_predictions, specialist_weight


def test_hard_gate_uses_specialist_only_above_threshold() -> None:
    baseline = np.array([[1.0, 2.0]], dtype=np.float32)
    specialist = np.array([[10.0, 20.0]], dtype=np.float32)
    probability = np.array([[0.2, 0.8]], dtype=np.float32)
    blended = blend_height_predictions(
        baseline,
        specialist,
        probability,
        threshold=0.5,
        maximum_delta_m=None,
    )
    np.testing.assert_allclose(blended, [[1.0, 20.0]])


def test_smooth_gate_is_bounded_and_delta_is_clipped() -> None:
    weight = specialist_weight(
        np.array([0.3, 0.5, 0.7], dtype=np.float32),
        threshold=0.5,
        transition=0.4,
    )
    np.testing.assert_allclose(weight, [0.0, 0.5, 1.0], atol=1e-6)
    blended = blend_height_predictions(
        np.zeros(3, dtype=np.float32),
        np.full(3, 100.0, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        threshold=0.5,
        maximum_delta_m=12.0,
    )
    np.testing.assert_allclose(blended, 12.0)


def test_hybrid_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="share a grid"):
        blend_height_predictions(
            np.zeros((2, 2)),
            np.zeros((3, 3)),
            np.zeros((2, 2)),
            threshold=0.5,
        )
