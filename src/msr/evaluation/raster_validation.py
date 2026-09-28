"""Align an uploaded reference surface and compute honest dense-height metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import reproject

from msr.evaluation.metrics import HeightMetrics, compute_height_metrics


@dataclass(frozen=True)
class RasterValidationResult:
    metrics: HeightMetrics
    reference_m: np.ndarray
    error_m: np.ndarray
    valid_mask: np.ndarray
    alignment: str


def _same_transform(left: Any, right: Any) -> bool:
    return left is not None and right is not None and np.allclose(tuple(left), tuple(right))


def evaluate_reference_raster(
    prediction: np.ndarray,
    reference_path: str | Path,
    *,
    prediction_profile: dict[str, Any],
    prediction_valid_mask: np.ndarray | None = None,
) -> RasterValidationResult:
    """Evaluate a prediction after aligning a reference onto its exact grid.

    Georeferenced rasters are reprojected by CRS/transform. Unreferenced rasters
    are accepted in pixel space and resampled only when their dimensions differ.
    The signed error convention is prediction minus reference.
    """

    predicted = np.asarray(prediction, dtype=np.float32)
    expected_shape = (
        int(prediction_profile["height"]),
        int(prediction_profile["width"]),
    )
    if predicted.shape != expected_shape:
        raise ValueError(
            f"Prediction shape {predicted.shape} does not match its grid {expected_shape}"
        )

    destination = np.full(expected_shape, np.nan, dtype=np.float32)
    destination_valid = np.zeros(expected_shape, dtype=np.uint8)
    destination_crs = prediction_profile.get("crs")
    destination_transform = prediction_profile.get("transform")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(Path(reference_path).expanduser().resolve()) as source:
            if source.count < 1:
                raise ValueError("Reference raster has no height band")
            exact_geospatial_grid = (
                destination_crs is not None
                and source.crs is not None
                and destination_crs == source.crs
                and source.shape == expected_shape
                and _same_transform(source.transform, destination_transform)
            )
            pixel_aligned_grid = (
                source.shape == expected_shape
                and (destination_crs is None or source.crs is None)
            )

            if exact_geospatial_grid or pixel_aligned_grid:
                destination[:] = source.read(1).astype(np.float32, copy=False)
                destination_valid[:] = source.read_masks(1) > 0
                alignment = (
                    "exact_geospatial_grid"
                    if exact_geospatial_grid
                    else "pixel_aligned_same_shape"
                )
            elif destination_crs is not None and source.crs is not None:
                reproject(
                    source=rasterio.band(source, 1),
                    destination=destination,
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=source.nodata,
                    dst_transform=destination_transform,
                    dst_crs=destination_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                source_mask = source.read_masks(1)
                reproject(
                    source=source_mask,
                    destination=destination_valid,
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=0,
                    dst_transform=destination_transform,
                    dst_crs=destination_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
                alignment = "reprojected_to_prediction_grid"
            else:
                destination[:] = source.read(
                    1,
                    out_shape=expected_shape,
                    resampling=Resampling.bilinear,
                ).astype(np.float32, copy=False)
                destination_valid[:] = source.read_masks(
                    1,
                    out_shape=expected_shape,
                    resampling=Resampling.nearest,
                ) > 0
                alignment = "pixel_space_resampled"

            if source.nodata is not None:
                destination_valid &= ~np.isclose(
                    destination, source.nodata, equal_nan=True
                )

    valid = destination_valid.astype(bool)
    valid &= np.isfinite(destination) & np.isfinite(predicted)
    if prediction_valid_mask is not None:
        prediction_valid = np.asarray(prediction_valid_mask, dtype=bool)
        if prediction_valid.shape != expected_shape:
            raise ValueError("Prediction validity mask and raster shape differ")
        valid &= prediction_valid
    metrics = compute_height_metrics(predicted, destination, valid)
    error = np.full(expected_shape, np.nan, dtype=np.float32)
    error[valid] = predicted[valid] - destination[valid]
    aligned_reference = np.full(expected_shape, np.nan, dtype=np.float32)
    aligned_reference[valid] = destination[valid]
    return RasterValidationResult(
        metrics=metrics,
        reference_m=aligned_reference,
        error_m=error,
        valid_mask=valid,
        alignment=alignment,
    )
