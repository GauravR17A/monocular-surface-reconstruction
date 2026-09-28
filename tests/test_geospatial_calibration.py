from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from msr.geospatial.calibration import (
    compose_absolute_dsm,
    fit_affine_calibration,
    reproject_dem_to_grid,
)


def test_robust_affine_calibration_rejects_large_outlier() -> None:
    relative = np.linspace(0, 1, 30)
    elevation = 40.0 * relative + 100.0
    elevation[10] += 500.0

    fit = fit_affine_calibration(relative, elevation)

    assert fit.scale == pytest.approx(40.0, abs=1e-6)
    assert fit.offset_m == pytest.approx(100.0, abs=1e-6)
    assert fit.points_used == 29
    assert fit.rmse_m < 1e-5


def test_compose_absolute_dsm_adds_nonnegative_object_height() -> None:
    ground = np.array([[100.0, 101.0], [102.0, np.nan]], dtype=np.float32)
    height = np.array([[5.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    result = compose_absolute_dsm(ground, height)
    np.testing.assert_allclose(result[:2, :2], [[105.0, 101.0], [105.0, np.nan]], equal_nan=True)


def test_reproject_dem_to_reference_grid(tmp_path: Path) -> None:
    path = tmp_path / "dem.tif"
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 2,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32632",
        "transform": Affine.translation(500000, 1000) * Affine.scale(10, -10),
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(np.array([[10, 20], [30, 40]], dtype=np.float32), 1)

    aligned, valid = reproject_dem_to_grid(path, profile)

    np.testing.assert_allclose(aligned, [[10, 20], [30, 40]])
    assert valid.all()
