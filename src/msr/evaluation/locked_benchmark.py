"""Shared helpers for reproducible, app-equivalent height benchmarks.

This module intentionally contains no checkpoint-selection or training logic.  A
benchmark may read a checkpoint, but it must never promote or mutate one.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import reproject

from msr.data.highbuild import validate_highbuild_manifest_contract
from msr.data.radiometry import percentile_stretch_rgb
from msr.data.surface_dataset import SurfaceSampleRecord
from msr.io.raster import RasterImage


RADIOMETRIC_VARIANTS = (
    "raw",
    "stretch01_99",
    "jpeg92",
    "gamma070",
    "gamma145",
    "exposure075",
    "exposure125",
    "channel_gain",
)
RADIOMETRIC_VARIANT_DESCRIPTIONS = {
    "raw": "unchanged raster values",
    "stretch01_99": "per-band 1st--99th percentile stretch to the uint8 range",
    "jpeg92": "the same stretch encoded and decoded at JPEG quality 92",
    "gamma070": "raw RGB normalized to [0,1] with gamma 0.70",
    "gamma145": "raw RGB normalized to [0,1] with gamma 1.45",
    "exposure075": "raw RGB multiplied by 0.75 and clipped to [0,255]",
    "exposure125": "raw RGB multiplied by 1.25 and clipped to [0,255]",
    "channel_gain": "raw RGB gains [1.18,0.82,1.08] and clipping to [0,255]",
}


@dataclass(frozen=True)
class BenchmarkReference:
    """Full-resolution target and masks aligned to an app input grid."""

    target_m: np.ndarray
    valid_mask: np.ndarray
    regression_mask: np.ndarray
    building_mask: np.ndarray
    vegetation_mask: np.ndarray
    alignments: dict[str, str]


def sha256_file(path: str | Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = sha256()
    with Path(path).resolve().open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path_tree(path: str | Path) -> str:
    """Hash one file or a directory tree using relative filenames and bytes."""

    source = Path(path).expanduser().resolve()
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    digest = sha256()
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Cannot hash an empty model directory: {source}")
    for item in files:
        relative = item.relative_to(source).as_posix()
        digest.update(f"{relative}\0{sha256_file(item)}\n".encode("utf-8"))
    return digest.hexdigest()


def benchmark_dataset_digest(
    records: Iterable[SurfaceSampleRecord],
) -> tuple[str, dict[str, dict[str, str]]]:
    """Hash every benchmark input that can affect labels or inference.

    Absolute paths are deliberately excluded from the aggregate digest so that a
    verified C:-to-D: migration does not invalidate the benchmark identity.
    """

    aggregate = sha256()
    per_sample: dict[str, dict[str, str]] = {}
    roles = (
        "rgb_path",
        "surface_path",
        "dtm_path",
        "building_mask_path",
        "vegetation_mask_path",
        "valid_mask_path",
    )
    for record in sorted(records, key=lambda item: item.sample_id):
        hashes: dict[str, str] = {}
        header = (
            f"{record.sample_id}\0{record.region}\0{record.landscape}\0"
            f"{record.target_kind}\n"
        )
        aggregate.update(header.encode("utf-8"))
        for role in roles:
            path = getattr(record, role)
            if path is None:
                continue
            file_hash = sha256_file(path)
            hashes[role] = file_hash
            aggregate.update(f"{role}\0{file_hash}\n".encode("ascii"))
        per_sample[record.sample_id] = hashes
    return aggregate.hexdigest(), per_sample


def make_radiometric_variant(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    variant: str,
) -> np.ndarray:
    """Create deterministic in-memory variants of the exact app RGB array.

    ``jpeg92`` mirrors the browser texture path: per-band 1--99 percentile
    stretch followed by Pillow JPEG quality 92 with optimization enabled.
    """

    values = np.asarray(rgb, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError(f"Expected CHW RGB data, received {values.shape}")
    if valid.shape != values.shape[-2:]:
        raise ValueError("valid_mask must match the RGB spatial dimensions")
    if variant not in RADIOMETRIC_VARIANTS:
        raise ValueError(
            f"Unknown radiometric variant {variant!r}; expected "
            f"one of {RADIOMETRIC_VARIANTS}"
        )

    if variant == "raw":
        output = values.copy()
    elif variant in {"stretch01_99", "jpeg92"}:
        stretched = percentile_stretch_rgb(
            values,
            valid_mask=valid,
            low_percentile=1.0,
            high_percentile=99.0,
        )
        if variant == "stretch01_99":
            output = stretched
        else:
            image = Image.fromarray(
                np.moveaxis(stretched[:3].astype(np.uint8), 0, -1), mode="RGB"
            )
            encoded = BytesIO()
            image.save(encoded, format="JPEG", quality=92, optimize=True)
            encoded.seek(0)
            with Image.open(encoded) as decoded:
                output = np.moveaxis(
                    np.asarray(decoded.convert("RGB"), dtype=np.float32), -1, 0
                )
    elif variant.startswith("gamma"):
        gamma = {"gamma070": 0.70, "gamma145": 1.45}[variant]
        normalized = np.clip(values[:3] / 255.0, 0.0, 1.0)
        output = np.power(normalized, gamma) * 255.0
    elif variant.startswith("exposure"):
        factor = {"exposure075": 0.75, "exposure125": 1.25}[variant]
        output = np.clip(values[:3] * factor, 0.0, 255.0)
    else:
        gains = np.asarray([1.18, 0.82, 1.08], dtype=np.float32)[:, None, None]
        output = np.clip(values[:3] * gains, 0.0, 255.0)
    output = output[:3].astype(np.float32, copy=False)
    output[:, ~valid] = 0.0
    return output


def _same_transform(left: Any, right: Any) -> bool:
    return left is not None and right is not None and np.allclose(
        tuple(left), tuple(right), rtol=0.0, atol=1e-9
    )


def _read_aligned_band(
    path: Path,
    source: RasterImage,
    *,
    resampling: Resampling,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Read one raster band on the app's processing grid."""

    expected_shape = (
        int(source.profile["height"]),
        int(source.profile["width"]),
    )
    destination_crs = source.profile.get("crs")
    destination_transform = source.profile.get("transform")
    destination = np.full(expected_shape, np.nan, dtype=np.float32)
    destination_valid = np.zeros(expected_shape, dtype=np.uint8)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as raster:
            if raster.count < 1:
                raise ValueError(f"Raster contains no bands: {path}")
            exact_geospatial = (
                destination_crs is not None
                and raster.crs is not None
                and destination_crs == raster.crs
                and raster.shape == expected_shape
                and _same_transform(raster.transform, destination_transform)
            )
            pixel_aligned = (
                raster.shape == expected_shape
                and (destination_crs is None or raster.crs is None)
            )
            if exact_geospatial or pixel_aligned:
                destination[:] = raster.read(1).astype(np.float32, copy=False)
                destination_valid[:] = raster.read_masks(1) > 0
                alignment = (
                    "exact_geospatial_grid"
                    if exact_geospatial
                    else "pixel_aligned_same_shape"
                )
            elif destination_crs is not None and raster.crs is not None:
                reproject(
                    source=rasterio.band(raster, 1),
                    destination=destination,
                    src_transform=raster.transform,
                    src_crs=raster.crs,
                    src_nodata=raster.nodata,
                    dst_transform=destination_transform,
                    dst_crs=destination_crs,
                    dst_nodata=np.nan,
                    resampling=resampling,
                )
                source_mask = raster.read_masks(1)
                reproject(
                    source=source_mask,
                    destination=destination_valid,
                    src_transform=raster.transform,
                    src_crs=raster.crs,
                    src_nodata=0,
                    dst_transform=destination_transform,
                    dst_crs=destination_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
                alignment = "reprojected_to_app_grid"
            elif (
                raster.width == source.original_width
                and raster.height == source.original_height
                and source.resampled
            ):
                destination[:] = raster.read(
                    1, out_shape=expected_shape, resampling=resampling
                ).astype(np.float32, copy=False)
                destination_valid[:] = raster.read_masks(
                    1, out_shape=expected_shape, resampling=Resampling.nearest
                ) > 0
                alignment = "resampled_with_app_input"
            else:
                raise ValueError(
                    f"Unaligned raster {path}: {raster.shape} cannot be safely mapped "
                    f"to app grid {expected_shape} without shared georeferencing"
                )

            if raster.nodata is not None:
                destination_valid &= ~np.isclose(
                    destination, raster.nodata, equal_nan=True
                )
    valid = destination_valid.astype(bool) & np.isfinite(destination)
    return destination, valid, alignment


def load_benchmark_reference(
    record: SurfaceSampleRecord,
    source: RasterImage,
    *,
    height_max_m: float = 200.0,
    building_threshold_m: float = 2.0,
) -> BenchmarkReference:
    """Load a full reference surface using the training manifest semantics."""

    if height_max_m <= 0 or building_threshold_m < 0:
        raise ValueError("height_max_m must be positive and threshold nonnegative")
    validate_highbuild_manifest_contract(
        sample_id=record.sample_id,
        surface_path=record.surface_path,
        target_kind=record.target_kind,
        building_mask_path=record.building_mask_path,
        valid_mask_path=record.valid_mask_path,
    )
    target, valid, surface_alignment = _read_aligned_band(
        record.surface_path, source, resampling=Resampling.bilinear
    )
    alignments = {"surface": surface_alignment}
    valid &= source.valid_mask

    if record.target_kind == "dsm":
        if record.dtm_path is None:
            raise ValueError(f"DSM record {record.sample_id} requires a DTM")
        terrain, terrain_valid, terrain_alignment = _read_aligned_band(
            record.dtm_path, source, resampling=Resampling.bilinear
        )
        alignments["dtm"] = terrain_alignment
        valid &= terrain_valid
        target = target - terrain
    target = np.maximum(target, 0.0)
    valid &= np.isfinite(target) & (target <= height_max_m)

    if record.valid_mask_path is not None:
        mask, mask_valid, alignment = _read_aligned_band(
            record.valid_mask_path, source, resampling=Resampling.nearest
        )
        alignments["valid_mask"] = alignment
        valid &= mask_valid & (mask > 0)

    explicit_building = None
    if record.building_mask_path is not None:
        mask, mask_valid, alignment = _read_aligned_band(
            record.building_mask_path, source, resampling=Resampling.nearest
        )
        alignments["building_mask"] = alignment
        explicit_building = mask_valid & (mask > 0)
    explicit_vegetation = None
    if record.vegetation_mask_path is not None:
        mask, mask_valid, alignment = _read_aligned_band(
            record.vegetation_mask_path, source, resampling=Resampling.nearest
        )
        alignments["vegetation_mask"] = alignment
        explicit_vegetation = mask_valid & (mask > 0)

    if record.target_kind == "building_height" and explicit_building is not None:
        # Keep semantic building evidence independent of the selected height
        # protocol. Estimated footprints may be valid semantic labels while
        # remaining excluded from strict measured-height regression.
        building = explicit_building
    elif explicit_building is not None:
        building = explicit_building & valid
    elif record.landscape == "urban" or record.target_kind == "building_height":
        building = valid & (target >= building_threshold_m)
    else:
        building = np.zeros_like(valid)

    if explicit_vegetation is not None:
        vegetation = explicit_vegetation & valid & ~building
    elif record.landscape == "forest":
        vegetation = valid & ~building
    else:
        vegetation = np.zeros_like(valid)

    regression = valid.copy()
    if record.target_kind == "building_height":
        # HighBuild's zero background is not observed terrain/canopy height.
        regression = valid & building

    target_output = np.full(target.shape, np.nan, dtype=np.float32)
    target_output[valid] = target[valid].astype(np.float32, copy=False)
    return BenchmarkReference(
        target_m=target_output,
        valid_mask=valid,
        regression_mask=regression,
        building_mask=building,
        vegetation_mask=vegetation,
        alignments=alignments,
    )
