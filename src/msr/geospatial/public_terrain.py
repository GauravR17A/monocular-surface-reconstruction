"""Small-area terrain-datum download for interactive georeferenced demos."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import math
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds


TERRAIN_TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{z}/{x}/{y}.tif"


@dataclass(frozen=True)
class PublicTerrainMetadata:
    provider: str
    attribution: str
    zoom: int
    tile_count: int
    urls: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tile_xy(longitude: float, latitude: float, zoom: int) -> tuple[int, int]:
    latitude = min(85.05112878, max(-85.05112878, latitude))
    scale = 1 << zoom
    x = int(math.floor((longitude + 180.0) / 360.0 * scale))
    latitude_rad = math.radians(latitude)
    y = int(
        math.floor(
            (1.0 - math.asinh(math.tan(latitude_rad)) / math.pi) / 2.0 * scale
        )
    )
    return min(scale - 1, max(0, x)), min(scale - 1, max(0, y))


def terrain_tiles_for_bounds(
    bounds_wgs84: tuple[float, float, float, float],
    *,
    zoom: int = 12,
    maximum_tiles: int = 16,
) -> list[tuple[int, int, int]]:
    west, south, east, north = bounds_wgs84
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid WGS84 bounds for public terrain lookup")
    x_min, y_min = _tile_xy(west, north, zoom)
    x_max, y_max = _tile_xy(east - 1e-10, south + 1e-10, zoom)
    tile_count = (abs(x_max - x_min) + 1) * (abs(y_max - y_min) + 1)
    if tile_count > maximum_tiles:
        raise ValueError(
            f"Interactive terrain lookup needs {tile_count} tiles; crop the image or attach a DEM."
        )
    tiles = [
        (zoom, x, y)
        for x in range(min(x_min, x_max), max(x_min, x_max) + 1)
        for y in range(min(y_min, y_max), max(y_min, y_max) + 1)
    ]
    return tiles


def choose_terrain_tiles(
    bounds_wgs84: tuple[float, float, float, float], *, preferred_zoom: int = 12,
    minimum_zoom: int = 8, maximum_tiles: int = 16,
) -> list[tuple[int, int, int]]:
    """Fit larger imported scenes to the tile budget; record the actual source zoom."""
    for zoom in range(preferred_zoom, minimum_zoom - 1, -1):
        try:
            return terrain_tiles_for_bounds(bounds_wgs84, zoom=zoom, maximum_tiles=maximum_tiles)
        except ValueError as error:
            if "crop the image" not in str(error) or zoom == minimum_zoom:
                raise
    raise ValueError("Invalid public terrain zoom range")


def download_public_terrain_dem(
    reference_profile: dict[str, Any],
    output_path: str | Path,
    *,
    zoom: int = 12,
    timeout_seconds: int = 20,
    maximum_tile_bytes: int = 8 * 1024 * 1024,
) -> PublicTerrainMetadata:
    """Download and mosaic a coarse bare-earth terrain datum for one image grid."""

    crs = reference_profile.get("crs")
    transform = reference_profile.get("transform")
    if crs is None or transform is None:
        raise ValueError("Automatic terrain lookup requires a georeferenced image")
    height = int(reference_profile["height"])
    width = int(reference_profile["width"])
    native_bounds = array_bounds(height, width, transform)
    wgs84_bounds = transform_bounds(crs, "EPSG:4326", *native_bounds, densify_pts=21)
    tiles = choose_terrain_tiles(wgs84_bounds, preferred_zoom=zoom, minimum_zoom=min(8, zoom))
    zoom = tiles[0][0]

    memory_files: list[MemoryFile] = []
    datasets: list[rasterio.io.DatasetReader] = []
    urls: list[str] = []
    try:
        for tile_zoom, tile_x, tile_y in tiles:
            url = TERRAIN_TILE_URL.format(z=tile_zoom, x=tile_x, y=tile_y)
            request = Request(url, headers={"User-Agent": "Monocular Surface Reconstruction/0.1"})
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read(maximum_tile_bytes + 1)
            if len(payload) > maximum_tile_bytes:
                raise ValueError(f"Public terrain tile exceeded {maximum_tile_bytes} bytes")
            memory_file = MemoryFile(BytesIO(payload))
            dataset = memory_file.open()
            memory_files.append(memory_file)
            datasets.append(dataset)
            urls.append(url)

        mosaic, mosaic_transform = merge(
            datasets,
            nodata=np.nan,
            dtype="float32",
            resampling=rasterio.enums.Resampling.bilinear,
        )
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            width=mosaic.shape[2],
            height=mosaic.shape[1],
            transform=mosaic_transform,
            nodata=np.nan,
            compress="deflate",
            predictor=3,
        )
        with rasterio.open(output, "w", **profile) as destination:
            destination.write(mosaic[0], 1)
            destination.update_tags(
                MSR_PRODUCT="coarse_public_terrain_datum",
                MSR_SOURCE="Mapzen Terrain Tiles via AWS Open Data",
                MSR_WARNING="coarse_datum_not_structure_height_truth",
            )
    finally:
        for dataset in datasets:
            dataset.close()
        for memory_file in memory_files:
            memory_file.close()

    return PublicTerrainMetadata(
        provider="Mapzen Terrain Tiles via AWS Open Data",
        attribution="Public bare-earth terrain datum; source mix and licences are documented by the provider.",
        zoom=zoom,
        tile_count=len(tiles),
        urls=urls,
    )
