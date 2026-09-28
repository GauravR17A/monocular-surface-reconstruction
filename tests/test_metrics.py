import numpy as np
import pytest

from msr.evaluation.metrics import StreamingRegressionMetrics, compute_height_metrics


def test_metrics_are_in_metric_units_and_bias_has_documented_sign():
    target = np.array([[0.0, 10.0], [20.0, 30.0]])
    prediction = np.array([[1.0, 8.0], [23.0, 26.0]])

    result = compute_height_metrics(prediction, target)

    assert result.pixel_count == 4
    assert result.mae_m == pytest.approx(2.5)
    assert result.rmse_m == pytest.approx(np.sqrt(7.5))
    assert result.bias_m == pytest.approx(-0.5)
    assert result.correlation is not None
    assert result.r2 is not None


def test_invalid_values_and_mask_are_excluded():
    target = np.array([0.0, 1.0, np.nan, 3.0])
    prediction = np.array([0.0, 3.0, 5.0, np.inf])
    valid_mask = np.array([True, False, True, True])

    result = compute_height_metrics(prediction, target, valid_mask)

    assert result.pixel_count == 1
    assert result.rmse_m == 0.0
    assert result.correlation is None


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        compute_height_metrics(np.zeros((2, 2)), np.zeros((4,)))


def test_streaming_primary_metrics_match_full_metrics():
    target = np.arange(12, dtype=np.float64).reshape(3, 4)
    prediction = target + np.array(
        [[0, 1, 0, -1], [2, 0, -2, 0], [1, 1, -1, -1]], dtype=np.float64
    )
    full = compute_height_metrics(prediction, target)
    streaming = StreamingRegressionMetrics()
    streaming.update(prediction[:2], target[:2])
    streaming.update(prediction[2:], target[2:])
    result = streaming.compute()

    assert result["rmse_m"] == pytest.approx(full.rmse_m)
    assert result["mae_m"] == pytest.approx(full.mae_m)
    assert result["bias_m"] == pytest.approx(full.bias_m)
    assert result["correlation"] == pytest.approx(full.correlation)
    assert result["r2"] == pytest.approx(full.r2)

