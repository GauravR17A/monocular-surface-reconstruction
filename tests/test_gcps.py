from pathlib import Path

import numpy as np
from affine import Affine

from msr.geospatial.gcps import sample_raster_at_gcps


def test_sample_raster_at_pixel_gcps(tmp_path: Path) -> None:
    csv_path = tmp_path / "gcps.csv"
    csv_path.write_text("row,col,elevation_m\n0,1,100\n2,2,120\n", encoding="utf-8")
    raster = np.arange(9, dtype=np.float32).reshape(3, 3)

    values, elevations = sample_raster_at_gcps(
        raster,
        csv_path,
        reference_profile={"transform": Affine.identity()},
    )

    np.testing.assert_allclose(values, [1, 8])
    np.testing.assert_allclose(elevations, [100, 120])


def test_pixel_gcps_scale_with_an_overview_grid(tmp_path: Path) -> None:
    csv_path = tmp_path / "overview_gcps.csv"
    csv_path.write_text("row,col,elevation_m\n6,8,125\n", encoding="utf-8")
    raster = np.arange(25, dtype=np.float32).reshape(5, 5)

    values, elevations = sample_raster_at_gcps(
        raster,
        csv_path,
        reference_profile={"transform": Affine.identity()},
        pixel_coordinate_scale=(0.5, 0.5),
    )

    np.testing.assert_allclose(values, [19])
    np.testing.assert_allclose(elevations, [125])
