"""Build reproducible Manali hilly-scene evidence against independent SRTM.

The operational absolute DSM is calibrated with Copernicus GLO-30.  This script
uses a separate SRTM tile only for evaluation, aligns it to the model grid, and
writes the prediction/reference/error assets consumed by the reviewer-facing UI.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from msr.evaluation.raster_validation import evaluate_reference_raster


def _write_float_raster(
    path: Path,
    values: np.ndarray,
    profile: dict[str, object],
) -> None:
    output_profile = profile.copy()
    output_profile.update(
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=3,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(values.astype(np.float32, copy=False), 1)


def _materialize_srtm(hgt_gz: Path, destination: Path) -> None:
    with gzip.open(hgt_gz, "rb") as source:
        payload = source.read()
    values = np.frombuffer(payload, dtype=">i2")
    side = int(round(np.sqrt(values.size)))
    if side * side != values.size or side not in {1201, 3601}:
        raise ValueError(f"Unexpected HGT dimensions: {values.size} samples")
    tile = values.reshape(side, side).astype(np.int16)
    # HGT samples lie on a 1/3600-degree point grid including both tile edges.
    # GeoTIFF stores pixel areas, so place the first sample at the centre of the
    # north-west pixel rather than shrinking the grid with ``from_bounds``.
    resolution = 1.0 / float(side - 1)
    transform = (
        Affine.translation(77.0 - resolution / 2.0, 33.0 + resolution / 2.0)
        * Affine.scale(resolution, -resolution)
    )
    profile = {
        "driver": "GTiff",
        "height": side,
        "width": side,
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -32768,
        "compress": "deflate",
        "predictor": 2,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as target:
        target.write(tile, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction",
        type=Path,
        default=Path("viewer/public/demo/landscapes/hilly/absolute_dsm_m.tif"),
    )
    parser.add_argument(
        "--texture",
        type=Path,
        default=Path("viewer/public/demo/landscapes/hilly/texture.jpg"),
    )
    parser.add_argument(
        "--srtm-gzip",
        type=Path,
        default=Path("data/external/srtm/N32E077.hgt.gz"),
    )
    parser.add_argument(
        "--srtm-geotiff",
        type=Path,
        default=Path("data/external/srtm/N32E077_srtm_m.tif"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("viewer/public/demo/hilly_validation"),
    )
    args = parser.parse_args()

    _materialize_srtm(args.srtm_gzip, args.srtm_geotiff)
    args.output.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.prediction) as source:
        prediction = source.read(1).astype(np.float32, copy=False)
        prediction_mask = source.read_masks(1) > 0
        prediction_mask &= np.isfinite(prediction)
        profile = source.profile.copy()

    result = evaluate_reference_raster(
        prediction,
        args.srtm_geotiff,
        prediction_profile=profile,
        prediction_valid_mask=prediction_mask,
    )
    _write_float_raster(args.output / "model_prediction_m.tif", prediction, profile)
    _write_float_raster(
        args.output / "srtm_reference_aligned_m.tif",
        result.reference_m,
        profile,
    )
    _write_float_raster(args.output / "signed_error_m.tif", result.error_m, profile)
    shutil.copy2(args.texture, args.output / "satellite_rgb.jpg")

    valid_reference = result.reference_m[result.valid_mask]
    reference_min = float(valid_reference.min())
    reference_max = float(valid_reference.max())
    reference_relief = reference_max - reference_min
    metrics = {
        **result.metrics.to_dict(),
        "reference_kind": "dsm",
        "prediction_product": "absolute_surface_dsm",
        "alignment": result.alignment,
        "scene": "Manali, Himachal Pradesh",
        "prediction_calibration": "Copernicus GLO-30 terrain datum",
        "evaluation_reference": "Independent SRTM 1 arc-second elevation",
        "reference_resolution": "approximately 30 m",
        "reference_min_m": reference_min,
        "reference_max_m": reference_max,
        "relief_normalized_rmse_percent": (
            100.0 * result.metrics.rmse_m / reference_relief
            if reference_relief > 0.0
            else None
        ),
        "vertical_datum_note": (
            "SRTM is EGM96-referenced while the Copernicus calibration source uses "
            "EGM2008; no post-hoc offset was fitted to the evaluation reference."
        ),
        "limitations": (
            "Coarse DEM cross-check of final absolute surface elevation; not a "
            "survey-grade or LiDAR validation of individual objects."
        ),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "source.json").write_text(
        json.dumps(
            {
                "dataset": "SRTM-style HGT from the AWS Terrain Tiles public dataset",
                "tile": "N32E077",
                "download_url": (
                    "https://s3.amazonaws.com/elevation-tiles-prod/skadi/"
                    "N32/N32E077.hgt.gz"
                ),
                "used_for_inference": False,
                "used_for_calibration": False,
                "role": "independent coarse elevation evaluation only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
