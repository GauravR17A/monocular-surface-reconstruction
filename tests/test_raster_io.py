from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.transform import array_bounds

from msr.io.raster import pixel_spacing_m, read_rgb_raster, write_float_raster


def test_raster_round_trip_preserves_spatial_grid(tmp_path: Path) -> None:
    source_path = tmp_path / "rgb.tif"
    transform = Affine.translation(400000, 3000000) * Affine.scale(0.5, -0.5)
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 9,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:32643",
        "transform": transform,
    }
    with rasterio.open(source_path, "w", **profile) as destination:
        destination.write(np.full((3, 8, 9), 127, dtype=np.uint8))

    source = read_rgb_raster(source_path)
    copied_profile = dict(source.profile)
    copied_profile.update(blockxsize=9, blockysize=1)
    output_path = write_float_raster(
        tmp_path / "rdsm.tif",
        np.ones((8, 9), dtype=np.float32),
        reference_profile=copied_profile,
    )

    assert source.georeferenced
    assert source.rgb.shape == (3, 8, 9)
    with rasterio.open(output_path) as output:
        assert output.crs == source.profile["crs"]
        assert output.transform == transform
        assert output.count == 1
        assert output.dtypes == ("float32",)
        assert float(output.tags()["MSR_PIXEL_SIZE_X_M"]) == pytest.approx(0.5, abs=0.001)
        assert float(output.tags()["MSR_PIXEL_SIZE_Y_M"]) == pytest.approx(0.5, abs=0.001)


def test_pixel_spacing_converts_geographic_degrees_to_metres() -> None:
    profile = {
        "height": 100,
        "width": 100,
        "crs": "EPSG:4326",
        "transform": Affine.translation(77.0, 28.0) * Affine.scale(0.00001, -0.00001),
    }

    spacing = pixel_spacing_m(profile)

    assert spacing is not None
    assert spacing[0] == pytest.approx(0.98, abs=0.02)
    assert spacing[1] == pytest.approx(1.11, abs=0.02)


def test_web_mercator_uses_ground_distances_for_joshimath() -> None:
    profile = {
        "width": 1024, "height": 1024, "crs": "EPSG:3857",
        "transform": Affine(38.21851414258708, 0, 8844681.416934315,
                            0, -38.2185141425889, 3600489.7803449426),
    }
    x, y = pixel_spacing_m(profile)
    # Independent WGS84 geodesic distances at the centre of the downloaded TIFF.
    assert x == pytest.approx(32.925, abs=0.005)
    assert y == pytest.approx(32.761, abs=0.005)
    assert x < 38.21851414258708


def test_web_mercator_ground_scale_handles_rotated_pixels() -> None:
    profile = {"width": 1, "height": 1, "crs": "EPSG:3857",
               "transform": Affine(0, 10, -10, 10, 0, -10)}
    x, y = pixel_spacing_m(profile)
    assert x == pytest.approx(9.9330562, abs=0.0001)
    assert y == pytest.approx(10, abs=0.0001)


def test_raster_pixel_budget_preserves_geospatial_extent(tmp_path: Path) -> None:
    source_path = tmp_path / "large_rgb.tif"
    transform = Affine.translation(400000, 3000000) * Affine.scale(0.5, -0.5)
    profile = {
        "driver": "GTiff",
        "height": 80,
        "width": 120,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:32643",
        "transform": transform,
    }
    with rasterio.open(source_path, "w", **profile) as destination:
        destination.write(np.full((3, 80, 120), 127, dtype=np.uint8))

    source = read_rgb_raster(source_path, max_pixels=2400, max_dimension=64)

    assert source.resampled
    assert (source.original_width, source.original_height) == (120, 80)
    assert source.rgb.shape[1] * source.rgb.shape[2] <= 2400
    original_bounds = array_bounds(80, 120, transform)
    processed_bounds = array_bounds(
        source.profile["height"], source.profile["width"], source.profile["transform"]
    )
    assert np.allclose(original_bounds, processed_bounds)
