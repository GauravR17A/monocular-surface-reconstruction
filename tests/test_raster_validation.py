from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from msr.evaluation.raster_validation import evaluate_reference_raster


def _write_reference(path: Path, values: np.ndarray, *, transform: Affine) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=transform,
        nodata=np.nan,
    ) as target:
        target.write(values.astype(np.float32), 1)


def test_reference_validation_reports_signed_metric_error(tmp_path: Path) -> None:
    transform = Affine.translation(500000, 2000000) * Affine.scale(1.5, -1.5)
    reference = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    prediction = reference + 1.0
    path = tmp_path / "reference.tif"
    _write_reference(path, reference, transform=transform)

    result = evaluate_reference_raster(
        prediction,
        path,
        prediction_profile={
            "height": 2,
            "width": 2,
            "crs": "EPSG:32643",
            "transform": transform,
        },
    )

    assert result.alignment == "exact_geospatial_grid"
    assert result.metrics.rmse_m == 1.0
    assert result.metrics.mae_m == 1.0
    assert result.metrics.bias_m == 1.0
    assert np.allclose(result.reference_m, reference)
    assert np.allclose(result.error_m, 1.0)


def test_reference_validation_resamples_unreferenced_pixel_grid(tmp_path: Path) -> None:
    reference = np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    path = tmp_path / "reference.tif"
    _write_reference(path, reference, transform=Affine.identity())
    prediction = np.full((4, 4), 2.0, dtype=np.float32)

    result = evaluate_reference_raster(
        prediction,
        path,
        prediction_profile={
            "height": 4,
            "width": 4,
            "crs": None,
            "transform": Affine.identity(),
        },
    )

    assert result.alignment == "pixel_space_resampled"
    assert result.metrics.pixel_count == 16
    assert np.isfinite(result.reference_m).all()
    assert np.isfinite(result.error_m).all()
