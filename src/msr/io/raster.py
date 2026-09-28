"""Uniform RGB raster I/O for ordinary images and georeferenced imagery."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, atan, cos, exp, hypot, pi, radians, sin, sqrt
from pathlib import Path
from typing import Any
from uuid import uuid4
import warnings

import numpy as np
from PIL import Image
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform as transform_coordinates


@dataclass(frozen=True)
class RasterImage:
    """An RGB image plus the raster profile needed to preserve its grid."""

    rgb: np.ndarray
    valid_mask: np.ndarray
    profile: dict[str, Any]
    georeferenced: bool
    source: Path
    original_width: int
    original_height: int
    resampled: bool


def _has_spatial_reference(profile: dict[str, Any]) -> bool:
    transform = profile.get("transform")
    crs = profile.get("crs")
    return crs is not None and transform is not None and transform != Affine.identity()


def pixel_spacing_m(reference_profile: dict[str, Any]) -> tuple[float, float] | None:
    """Estimate centre-pixel x/y spacing in metres for any raster CRS."""

    crs = reference_profile.get("crs")
    affine = reference_profile.get("transform")
    width = int(reference_profile.get("width", 0))
    height = int(reference_profile.get("height", 0))
    if crs is None or affine is None or width <= 0 or height <= 0:
        return None

    crs_object = rasterio.crs.CRS.from_user_input(crs)
    if crs_object.to_epsg() == 3857:
        # Web Mercator map metres are enlarged by latitude. Convert the local
        # grid vectors to WGS84 ground distances, including ellipsoid axes.
        _, centre_y = affine * (width / 2 + 0.5, height / 2 + 0.5)
        latitude = 2 * atan(exp(centre_y / 6_378_137.0)) - pi / 2
        eccentricity_squared = 6.6943799901413165e-3
        denominator = 1 - eccentricity_squared * sin(latitude) ** 2
        east_factor = cos(latitude) / sqrt(denominator)
        north_factor = (1 - eccentricity_squared) * cos(latitude) / denominator ** 1.5
        spacing_x = hypot(affine.a * east_factor, affine.d * north_factor)
        spacing_y = hypot(affine.b * east_factor, affine.e * north_factor)
        return (spacing_x, spacing_y) if spacing_x > 0 and spacing_y > 0 else None
    if crs_object.is_projected:
        _, unit_to_metre = crs_object.linear_units_factor
        spacing_x = hypot(affine.a, affine.d) * unit_to_metre
        spacing_y = hypot(affine.b, affine.e) * unit_to_metre
        if spacing_x > 0 and spacing_y > 0:
            return float(spacing_x), float(spacing_y)

    center_col = width / 2
    center_row = height / 2
    centres = (
        affine * (center_col + 0.5, center_row + 0.5),
        affine * (center_col + 1.5, center_row + 0.5),
        affine * (center_col + 0.5, center_row + 1.5),
    )
    try:
        longitudes, latitudes = transform_coordinates(
            crs,
            "EPSG:4326",
            [point[0] for point in centres],
            [point[1] for point in centres],
        )
    except Exception:
        return None

    def distance_m(first: int, second: int) -> float:
        lon1, lat1 = radians(longitudes[first]), radians(latitudes[first])
        lon2, lat2 = radians(longitudes[second]), radians(latitudes[second])
        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1
        haversine = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
        return 2 * 6_371_008.8 * asin(min(1.0, sqrt(max(0.0, haversine))))

    spacing_x = distance_m(0, 1)
    spacing_y = distance_m(0, 2)
    if not np.isfinite(spacing_x) or not np.isfinite(spacing_y):
        return None
    if spacing_x <= 0 or spacing_y <= 0:
        return None
    return float(spacing_x), float(spacing_y)


def _processing_shape(
    width: int,
    height: int,
    *,
    max_pixels: int | None,
    max_dimension: int | None,
) -> tuple[int, int]:
    scale = 1.0
    if max_pixels is not None:
        if max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        scale = min(scale, float(np.sqrt(max_pixels / (width * height))))
    if max_dimension is not None:
        if max_dimension <= 0:
            raise ValueError("max_dimension must be positive")
        scale = min(scale, max_dimension / width, max_dimension / height)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def read_rgb_raster(
    path: str | Path,
    *,
    max_pixels: int | None = None,
    max_dimension: int | None = None,
) -> RasterImage:
    """Read the first three bands as CHW float32 RGB.

    Rasterio supports GeoTIFF as well as ordinary PNG/JPEG files. A file is
    considered georeferenced only when both a CRS and non-identity transform
    are present; a TIFF suffix alone is not enough.
    """

    source_path = Path(path).expanduser().resolve()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(source_path) as source:
            ordinary_monochrome = source.count < 3 and source.crs is None and source_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            if source.count < 3 and not ordinary_monochrome:
                raise ValueError("Imagery must contain at least three RGB bands. For a single-band elevation TIFF, use 'Load existing height map'.")
            original_width = source.width
            original_height = source.height
            width, height = _processing_shape(
                original_width,
                original_height,
                max_pixels=max_pixels,
                max_dimension=max_dimension,
            )
            out_shape = (3, height, width)
            if ordinary_monochrome:
                # Clipboard images can be grayscale or palette PNGs. Interpret
                # their colours as an image, never reinterpret DEM values as RGB.
                with Image.open(source_path) as image:
                    rgba = image.convert("RGBA").resize((width, height), Image.Resampling.BILINEAR)
                    pixels = np.asarray(rgba)
                    rgb = pixels[:, :, :3].transpose(2, 0, 1).astype(np.float32)
                    band_masks = np.broadcast_to(pixels[:, :, 3] > 0, out_shape)
            else:
                rgb = source.read(
                    (1, 2, 3),
                    out_shape=out_shape,
                    resampling=Resampling.bilinear,
                ).astype(np.float32, copy=False)
                band_masks = source.read_masks(
                    (1, 2, 3),
                    out_shape=out_shape,
                    resampling=Resampling.nearest,
                ) > 0
            profile = source.profile.copy()
            if (width, height) != (original_width, original_height):
                profile.update(
                    width=width,
                    height=height,
                    transform=source.transform
                    * Affine.scale(original_width / width, original_height / height),
                )

    valid_mask = np.all(band_masks, axis=0) & np.all(np.isfinite(rgb), axis=0)
    return RasterImage(
        rgb=rgb,
        valid_mask=valid_mask,
        profile=profile,
        georeferenced=_has_spatial_reference(profile),
        source=source_path,
        original_width=original_width,
        original_height=original_height,
        resampled=(width, height) != (original_width, original_height),
    )


def write_float_raster(
    path: str | Path,
    array: np.ndarray,
    *,
    reference_profile: dict[str, Any],
    nodata: float = float("nan"),
    tags: dict[str, str] | None = None,
) -> Path:
    """Write one float32 band on exactly the reference image grid."""

    output = Path(path).expanduser().resolve()
    data = np.asarray(array, dtype=np.float32)
    expected = (int(reference_profile["height"]), int(reference_profile["width"]))
    if data.shape != expected:
        raise ValueError(f"Output shape {data.shape} does not match reference grid {expected}")

    profile = dict(reference_profile)
    profile.pop("photometric", None)
    # JPEG/PNG drivers often report scanline block sizes (for example height 1).
    # Those values are invalid when copied into a tiled GeoTIFF profile.
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    use_tiles = data.shape[0] >= 256 and data.shape[1] >= 256
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=nodata,
        compress="deflate",
        predictor=3,
        tiled=use_tiles,
    )
    if use_tiles:
        profile.update(blockxsize=256, blockysize=256)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid4().hex}.tmp{output.suffix}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(temporary, "w", **profile) as destination:
                destination.write(data, 1)
                output_tags = dict(tags or {})
                spacing = pixel_spacing_m(reference_profile)
                if spacing is not None:
                    output_tags.update(
                        MSR_PIXEL_SIZE_X_M=f"{spacing[0]:.9f}",
                        MSR_PIXEL_SIZE_Y_M=f"{spacing[1]:.9f}",
                    )
                if output_tags:
                    destination.update_tags(**output_tags)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
