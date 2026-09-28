import numpy as np

from msr.evaluation.error_analysis import BinarySums, RegressionSums


def test_regression_sums_matches_direct_metrics() -> None:
    prediction = np.array([1.0, 3.0, 7.0, 9.0])
    target = np.array([2.0, 2.0, 6.0, 10.0])
    stats = RegressionSums()
    stats.update(prediction[:2], target[:2])
    stats.update(prediction[2:], target[2:])
    metrics = stats.metrics()

    error = prediction - target
    assert np.isclose(metrics["rmse_m"], np.sqrt(np.mean(error**2)))
    assert np.isclose(metrics["mae_m"], np.mean(np.abs(error)))
    assert np.isclose(metrics["bias_m"], np.mean(error))
    assert np.isclose(metrics["correlation"], np.corrcoef(prediction, target)[0, 1])


def test_affine_fit_and_transformed_metrics() -> None:
    prediction = np.array([1.0, 2.0, 3.0, 4.0])
    target = 2.5 * prediction - 3.0
    stats = RegressionSums()
    stats.update(prediction, target)

    scale, offset = stats.affine_fit()
    transformed = stats.transformed(scale, offset).metrics()

    assert np.isclose(scale, 2.5)
    assert np.isclose(offset, -3.0)
    assert np.isclose(transformed["rmse_m"], 0.0)
    assert np.isclose(transformed["bias_m"], 0.0)
    assert transformed["mae_m"] is None


def test_binary_sums() -> None:
    probability = np.array([0.9, 0.6, 0.4, 0.1])
    target = np.array([True, False, True, False])
    stats = BinarySums()
    stats.update(probability, target, np.ones(4, dtype=bool), threshold=0.5)
    metrics = stats.metrics()

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
    assert np.isclose(metrics["iou"], 1.0 / 3.0)
