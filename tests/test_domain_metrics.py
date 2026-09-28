import numpy as np
import pytest

from msr.evaluation.domain_metrics import (
    DOMAIN_CODES,
    build_domain_masks,
    evaluate_by_domain,
    terrain_slope_degrees,
)


def test_domain_masks_are_exclusive_and_cover_every_valid_pixel():
    valid = np.ones((3, 4), dtype=bool)
    building = np.zeros_like(valid)
    building[0, :2] = True
    vegetation = np.zeros_like(valid)
    vegetation[0, 1:3] = True  # Overlap is intentionally assigned to urban.
    slope = np.zeros(valid.shape, dtype=np.float32)
    slope[1, :] = 12.0

    masks, domain_map = build_domain_masks(
        valid,
        building_mask=building,
        vegetation_mask=vegetation,
        terrain_slope_deg=slope,
        hilly_threshold_deg=5.0,
    )

    stacked = np.stack(list(masks.values()))
    assert np.all(stacked.sum(axis=0) == 1)
    assert masks["urban"][0, 1]
    assert not masks["forest"][0, 1]
    assert set(np.unique(domain_map)) == set(DOMAIN_CODES.values())


def test_domain_evaluation_exposes_failure_hidden_by_overall_average():
    reference = np.zeros((10, 10), dtype=np.float32)
    prediction = np.zeros_like(reference)
    building = np.zeros_like(reference, dtype=bool)
    building[:2, :] = True
    reference[building] = 20.0
    prediction[building] = 10.0

    result = evaluate_by_domain(
        prediction,
        reference,
        building_mask=building,
    ).metrics

    assert result["overall"]["rmse_m"] == pytest.approx(np.sqrt(20.0))
    assert result["domains"]["urban"]["rmse_m"] == pytest.approx(10.0)
    assert result["domains"]["plains"]["rmse_m"] == 0.0
    assert result["worst_domain"] == "urban"


def test_metric_terrain_slope_is_reported_in_degrees():
    # One metre rise per one metre in x is a 45-degree plane.
    terrain = np.tile(np.arange(5, dtype=np.float32), (5, 1))
    slope = terrain_slope_degrees(
        terrain,
        np.ones_like(terrain, dtype=bool),
        pixel_size_x_m=1.0,
        pixel_size_y_m=1.0,
    )

    np.testing.assert_allclose(slope, 45.0, atol=1e-5)
