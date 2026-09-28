"""Ground-control-point parsing and raster sampling."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.transform import rowcol


def sample_raster_at_gcps(
    raster: np.ndarray,
    csv_path: str | Path,
    *,
    reference_profile: dict[str, Any],
    pixel_coordinate_scale: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a raster at GCPs supplied as x/y or row/col plus elevation_m."""

    values: list[float] = []
    elevations: list[float] = []
    height, width = raster.shape
    transform = reference_profile.get("transform")
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        names = set(reader.fieldnames or [])
        if "elevation_m" not in names:
            raise ValueError("GCP CSV needs an elevation_m column")
        uses_pixels = {"row", "col"}.issubset(names)
        uses_map = {"x", "y"}.issubset(names)
        if not uses_pixels and not uses_map:
            raise ValueError("GCP CSV needs either row,col or x,y columns")
        if uses_map and transform is None:
            raise ValueError("Map-coordinate GCPs require a raster transform")

        for record in reader:
            if uses_pixels:
                row_scale, col_scale = pixel_coordinate_scale or (1.0, 1.0)
                row = int(round(float(record["row"]) * row_scale))
                col = int(round(float(record["col"]) * col_scale))
            else:
                row, col = rowcol(transform, float(record["x"]), float(record["y"]))
            if 0 <= row < height and 0 <= col < width:
                value = float(raster[row, col])
                elevation = float(record["elevation_m"])
                if np.isfinite(value) and np.isfinite(elevation):
                    values.append(value)
                    elevations.append(elevation)
    if not values:
        raise ValueError("No finite GCPs fall inside the raster")
    return np.asarray(values, dtype=np.float64), np.asarray(elevations, dtype=np.float64)
