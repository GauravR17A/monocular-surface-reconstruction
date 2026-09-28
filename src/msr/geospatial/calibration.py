"""Conservative conversion from relative geometry to metric elevation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


@dataclass(frozen=True)
class AffineCalibration:
    """A robust scene-level mapping from relative values to metres."""

    scale: float
    offset_m: float
    rmse_m: float
    correlation: float
    points_used: int
    points_total: int

    def apply(self, relative: np.ndarray) -> np.ndarray:
        return (self.scale * np.asarray(relative) + self.offset_m).astype(np.float32)


def fit_affine_calibration(
    relative_values: np.ndarray,
    elevation_m: np.ndarray,
    *,
    max_iterations: int = 5,
    mad_threshold: float = 3.5,
    minimum_points: int = 3,
) -> AffineCalibration:
    """Fit elevation = scale * relative + offset with iterative MAD trimming."""

    x = np.asarray(relative_values, dtype=np.float64).ravel()
    y = np.asarray(elevation_m, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("relative_values and elevation_m must have the same shape")
    finite = np.isfinite(x) & np.isfinite(y)
    points_total = int(finite.sum())
    if points_total < minimum_points:
        raise ValueError(f"At least {minimum_points} finite calibration points are required")

    keep = finite.copy()
    for _ in range(max_iterations):
        design = np.column_stack((x[keep], np.ones(int(keep.sum()))))
        scale, offset = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        residual = y - (scale * x + offset)
        centre = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - centre))
        if mad <= np.finfo(np.float64).eps:
            break
        robust_sigma = 1.4826 * mad
        updated = finite & (np.abs(residual - centre) <= mad_threshold * robust_sigma)
        if int(updated.sum()) < minimum_points or np.array_equal(updated, keep):
            break
        keep = updated

    design = np.column_stack((x[keep], np.ones(int(keep.sum()))))
    scale, offset = np.linalg.lstsq(design, y[keep], rcond=None)[0]
    fitted = scale * x[keep] + offset
    rmse = float(np.sqrt(np.mean((fitted - y[keep]) ** 2)))
    if np.std(x[keep]) <= 1e-12 or np.std(y[keep]) <= 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(x[keep], y[keep])[0, 1])
    return AffineCalibration(
        scale=float(scale),
        offset_m=float(offset),
        rmse_m=rmse,
        correlation=correlation,
        points_used=int(keep.sum()),
        points_total=points_total,
    )


def reproject_dem_to_grid(
    dem_path: str | Path,
    reference_profile: dict[str, Any],
    *,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject a DEM into the exact CRS, transform and shape of an RGB raster."""

    destination_crs = reference_profile.get("crs")
    destination_transform = reference_profile.get("transform")
    if destination_crs is None or destination_transform is None:
        raise ValueError("Reference raster needs a CRS and affine transform")

    shape = (int(reference_profile["height"]), int(reference_profile["width"]))
    destination = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(Path(dem_path)) as source:
        if source.crs is None:
            raise ValueError("DEM must have a CRS")
        source_data = source.read(1)
        source_nodata = source.nodata
        reproject(
            source=source_data,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source_nodata,
            dst_transform=destination_transform,
            dst_crs=destination_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    valid = np.isfinite(destination)
    return destination, valid


def compose_absolute_dsm(
    ground_elevation_m: np.ndarray,
    object_height_m: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Create an absolute DSM as terrain elevation plus non-negative nDSM."""

    ground = np.asarray(ground_elevation_m, dtype=np.float32)
    height = np.asarray(object_height_m, dtype=np.float32)
    if ground.shape != height.shape:
        raise ValueError("ground_elevation_m and object_height_m must share a grid")
    valid = np.isfinite(ground) & np.isfinite(height)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != ground.shape:
            raise ValueError("valid_mask must share the raster grid")
        valid &= mask
    dsm = ground + np.maximum(height, 0.0)
    dsm[~valid] = np.nan
    return dsm.astype(np.float32, copy=False)
