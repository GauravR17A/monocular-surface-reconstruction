import numpy as np
import pytest

from msr.evaluation.classification_metrics import (
    StreamingClassBoundaryMetrics,
    StreamingMulticlassMetrics,
    StreamingWaterDarkPixelProxy,
    multiclass_metric_deltas,
)


def test_streaming_multiclass_metrics_reports_exact_dense_scores() -> None:
    metric = StreamingMulticlassMetrics(("ground", "building", "vegetation"))
    metric.update(
        np.asarray([[0, 1, 1], [2, 2, 2]]),
        np.asarray([[0, 0, 1], [1, 2, 2]]),
    )

    result = metric.compute()

    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 1], [0, 0, 2]]
    assert result["total_valid_pixels"] == 6
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["macro_precision"] == pytest.approx((1 + 0.5 + 2 / 3) / 3)
    assert result["macro_recall"] == pytest.approx((0.5 + 0.5 + 1) / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 0.8) / 3)
    assert result["macro_iou"] == pytest.approx((0.5 + 1 / 3 + 2 / 3) / 3)
    assert result["per_class"]["building"] == {
        "precision": pytest.approx(0.5),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(0.5),
        "iou": pytest.approx(1 / 3),
        "support_pixels": 2,
        "predicted_pixels": 2,
        "true_positive_pixels": 1,
        "false_positive_pixels": 1,
        "false_negative_pixels": 1,
    }


def test_streaming_multiclass_metrics_masks_invalid_and_accumulates() -> None:
    metric = StreamingMulticlassMetrics(("ground", "building", "vegetation"))
    metric.update(
        np.asarray([0, 1, 2, 2]),
        np.asarray([0, 1, 9, 2]),
        np.asarray([True, False, True, True]),
    )
    metric.update(np.asarray([1]), np.asarray([1]))

    result = metric.compute()

    assert result["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert result["total_valid_pixels"] == 3
    assert result["accuracy"] == 1.0
    assert result["macro_iou"] == 1.0


def test_empty_classes_are_finite_and_metric_deltas_are_explicit() -> None:
    baseline_metric = StreamingMulticlassMetrics(("ground", "building", "vegetation"))
    baseline_metric.update(np.asarray([0, 0]), np.asarray([0, 0]))
    current_metric = StreamingMulticlassMetrics(("ground", "building", "vegetation"))
    current_metric.update(np.asarray([0, 1]), np.asarray([0, 0]))

    baseline = baseline_metric.compute()
    current = current_metric.compute()
    delta = multiclass_metric_deltas(current, baseline)

    assert np.isfinite(list(baseline["per_class"]["vegetation"].values())[0])
    assert delta["accuracy_delta"] == pytest.approx(-0.5)
    assert delta["accuracy_percentage_points"] == pytest.approx(-50.0)
    assert delta["per_class"]["ground"]["recall"]["delta"] == pytest.approx(
        -0.5
    )
    assert "positive means improvement" in delta["direction"]


def test_streaming_multiclass_metrics_rejects_shape_mismatch() -> None:
    metric = StreamingMulticlassMetrics(("ground", "building", "vegetation"))
    with pytest.raises(ValueError, match="identical shapes"):
        metric.update(np.zeros((2, 2)), np.zeros((3, 2)))


def test_road_boundary_metric_is_tolerance_aware() -> None:
    target = np.zeros((1, 7, 7), dtype=np.int64)
    prediction = np.zeros_like(target)
    target[:, :, 2:4] = 3
    prediction[:, :, 3:5] = 3
    valid = np.ones_like(target, dtype=bool)
    exact = StreamingClassBoundaryMetrics(3, tolerance_pixels=0)
    tolerant = StreamingClassBoundaryMetrics(3, tolerance_pixels=1)

    exact.update(prediction, target, valid)
    tolerant.update(prediction, target, valid)

    assert tolerant.compute()["f1"] > exact.compute()["f1"]
    assert tolerant.compute()["f1"] == pytest.approx(1.0)
    assert tolerant.compute()["dilated_boundary_iou"] > 0.0
    assert tolerant.compute()["tolerance_pixels"] == 1


def test_road_boundary_tolerance_does_not_bridge_invalid_label_gaps() -> None:
    target = np.zeros((7, 7), dtype=np.int64)
    prediction = np.zeros_like(target)
    target[:, :2] = 3
    prediction[:, 5:] = 3
    valid = np.ones_like(target, dtype=bool)
    valid[:, 3] = False
    metric = StreamingClassBoundaryMetrics(3, tolerance_pixels=2)

    metric.update(prediction, target, valid)

    assert metric.compute()["f1"] == 0.0


def test_water_dark_pixel_metric_is_explicitly_only_a_shadow_proxy() -> None:
    # Water is class 2 in the stable six-class output order.
    target = np.asarray([[2, 0, 1, 3], [2, 4, 5, 0]])
    prediction = np.asarray([[2, 2, 1, 2], [0, 2, 5, 0]])
    dark = np.asarray([[True, True, False, True], [True, True, False, False]])
    valid = np.ones_like(target, dtype=bool)
    metric = StreamingWaterDarkPixelProxy(2)

    metric.update(prediction, target, dark, valid)
    result = metric.compute()

    assert result["is_shadow_proxy_not_ground_truth"] is True
    assert result["dark_non_water_pixels"] == 3
    assert result["false_water_on_dark_non_water_pixels"] == 3
    assert result["all_false_water_pixels"] == 3
    assert result["false_water_rate_on_dark_non_water"] == 1.0
