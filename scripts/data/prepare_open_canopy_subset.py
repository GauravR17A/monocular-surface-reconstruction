"""Stream a leakage-safe Open-Canopy pilot subset instead of downloading 423 GB."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from tqdm import tqdm

from msr.data.open_canopy import (
    OPEN_CANOPY_CRS,
    OPEN_CANOPY_DATASET_URL,
    OPEN_CANOPY_LICENSE,
    OpenCanopyFeature,
    forest_semantic_masks,
    load_open_canopy_features,
    open_canopy_urls,
    stored_canopy_to_metres,
)


MANIFEST_FIELDS = [
    "sample_id",
    "region",
    "landscape",
    "rgb_path",
    "surface_path",
    "target_kind",
    "dtm_path",
    "building_mask_path",
    "vegetation_mask_path",
    "valid_mask_path",
    "relative_prior_path",
    "gsd_m",
]


def _chip_bounds(
    bounds: tuple[float, float, float, float], chip_size_m: float
) -> tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    centre_x = (left + right) / 2
    centre_y = (bottom + top) / 2
    half = chip_size_m / 2
    if chip_size_m > min(right - left, top - bottom):
        raise ValueError("chip_size_m must fit inside the official split cell")
    return centre_x - half, centre_y - half, centre_x + half, centre_y + half


def _read_remote_grid(
    url: str,
    bounds: tuple[float, float, float, float],
    output_size: int,
    indexes: int | tuple[int, ...],
    resampling: Resampling,
) -> tuple[np.ndarray, np.ndarray]:
    transform = from_bounds(*bounds, output_size, output_size)
    with rasterio.open(url) as source:
        if str(source.crs).upper() != OPEN_CANOPY_CRS:
            raise ValueError(f"Unexpected CRS {source.crs} for {url}")
        with WarpedVRT(
            source,
            crs=OPEN_CANOPY_CRS,
            transform=transform,
            width=output_size,
            height=output_size,
            resampling=resampling,
        ) as aligned:
            values = aligned.read(indexes)
            masks = aligned.read_masks(indexes)
    if masks.ndim == 3:
        valid = np.all(masks > 0, axis=0)
    else:
        valid = masks > 0
    return values, valid


def _write_raster_atomic(
    path: Path,
    values: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    dtype: str,
    nodata: float | int | None,
    tags: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.tif")
    bands = values if values.ndim == 3 else values[None]
    profile = {
        "driver": "GTiff",
        "width": bands.shape[2],
        "height": bands.shape[1],
        "count": bands.shape[0],
        "dtype": dtype,
        "crs": OPEN_CANOPY_CRS,
        "transform": from_bounds(*bounds, bands.shape[2], bands.shape[1]),
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": min(256, bands.shape[2]),
        "blockysize": min(256, bands.shape[1]),
        "nodata": nodata,
    }
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(bands.astype(dtype, copy=False))
        destination.update_tags(**tags)
    temporary.replace(path)


def _record_for(sample_dir: Path, feature: OpenCanopyFeature, gsd_m: float) -> dict[str, str]:
    return {
        "sample_id": feature.sample_id,
        "region": feature.region,
        "landscape": "forest",
        "rgb_path": str((sample_dir / "rgb.tif").resolve()),
        "surface_path": str((sample_dir / "canopy_height_m.tif").resolve()),
        "target_kind": "ndsm",
        "dtm_path": "",
        "building_mask_path": "",
        "vegetation_mask_path": str((sample_dir / "vegetation_mask.tif").resolve()),
        "valid_mask_path": str((sample_dir / "valid_mask.tif").resolve()),
        "relative_prior_path": "",
        "gsd_m": f"{gsd_m:.6f}",
    }


def _existing_record(
    sample_dir: Path, feature: OpenCanopyFeature, gsd_m: float
) -> dict[str, str] | None:
    required = [
        sample_dir / "rgb.tif",
        sample_dir / "canopy_height_m.tif",
        sample_dir / "vegetation_mask.tif",
        sample_dir / "valid_mask.tif",
        sample_dir / "source.json",
    ]
    return _record_for(sample_dir, feature, gsd_m) if all(path.is_file() for path in required) else None


def _materialize_feature(
    feature: OpenCanopyFeature,
    output_root: Path,
    *,
    output_size: int,
    chip_size_m: float,
) -> tuple[dict[str, str], float, float]:
    sample_dir = output_root / feature.split / feature.sample_id
    gsd_m = chip_size_m / output_size
    existing = _existing_record(sample_dir, feature, gsd_m)
    if existing is not None:
        with (sample_dir / "source.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return existing, float(metadata["valid_fraction"]), float(metadata["vegetation_fraction"])

    urls = open_canopy_urls(feature)
    bounds = _chip_bounds(feature.bounds, chip_size_m)
    rgb, rgb_valid = _read_remote_grid(
        urls["rgb"], bounds, output_size, (1, 2, 3), Resampling.bilinear
    )
    canopy_stored, canopy_valid = _read_remote_grid(
        urls["canopy"], bounds, output_size, 1, Resampling.bilinear
    )
    classification, classification_valid = _read_remote_grid(
        urls["classification"], bounds, output_size, 1, Resampling.nearest
    )
    canopy_m = stored_canopy_to_metres(canopy_stored)
    source_valid = rgb_valid & canopy_valid & classification_valid & np.isfinite(canopy_m)
    valid, vegetation = forest_semantic_masks(classification, source_valid)
    canopy_m = np.where(valid, np.clip(canopy_m, 0.0, 100.0), -9999.0).astype(np.float32)
    valid_fraction = float(valid.mean())
    vegetation_fraction = float(vegetation.mean())
    vegetation_heights = canopy_m[vegetation & valid]
    canopy_statistics = {
        "vegetation_mean_height_m": (
            float(np.mean(vegetation_heights)) if vegetation_heights.size else 0.0
        ),
        "vegetation_p90_height_m": (
            float(np.percentile(vegetation_heights, 90))
            if vegetation_heights.size
            else 0.0
        ),
        "vegetation_p99_height_m": (
            float(np.percentile(vegetation_heights, 99))
            if vegetation_heights.size
            else 0.0
        ),
    }

    common_tags = {
        "MSR_SOURCE": OPEN_CANOPY_DATASET_URL,
        "MSR_LICENSE": OPEN_CANOPY_LICENSE,
        "MSR_SAMPLE_ID": feature.sample_id,
        "MSR_OFFICIAL_SPLIT": feature.split,
    }
    _write_raster_atomic(
        sample_dir / "rgb.tif",
        rgb,
        bounds=bounds,
        dtype="uint8",
        nodata=None,
        tags={**common_tags, "MSR_PRODUCT": "rgb"},
    )
    _write_raster_atomic(
        sample_dir / "canopy_height_m.tif",
        canopy_m,
        bounds=bounds,
        dtype="float32",
        nodata=-9999.0,
        tags={
            **common_tags,
            "MSR_PRODUCT": "canopy_height",
            "MSR_UNITS": "metres",
            "MSR_SOURCE_SCALE": "stored_uint16_decimetres_x_0.1",
        },
    )
    _write_raster_atomic(
        sample_dir / "vegetation_mask.tif",
        vegetation.astype(np.uint8),
        bounds=bounds,
        dtype="uint8",
        nodata=0,
        tags={**common_tags, "MSR_PRODUCT": "lidar_vegetation_classes_3_4_5"},
    )
    _write_raster_atomic(
        sample_dir / "valid_mask.tif",
        valid.astype(np.uint8),
        bounds=bounds,
        dtype="uint8",
        nodata=0,
        tags={**common_tags, "MSR_PRODUCT": "forest_valid_classes_2_3_4_5_9"},
    )
    metadata = {
        "sample_id": feature.sample_id,
        "official_split": feature.split,
        "region": feature.region,
        "year": feature.year,
        "image_name": feature.image_name,
        "imagery_acquisition_date": feature.imagery_acquisition_date,
        "lidar_acquisition_date": feature.lidar_acquisition_date,
        "lidar_source_url": feature.lidar_url,
        "lidar_points_in_official_cell": feature.lidar_points,
        "bounds": bounds,
        "output_size": output_size,
        "gsd_m": gsd_m,
        "valid_fraction": valid_fraction,
        "vegetation_fraction": vegetation_fraction,
        **canopy_statistics,
        "urls": urls,
        "source": OPEN_CANOPY_DATASET_URL,
        "license": OPEN_CANOPY_LICENSE,
    }
    metadata_path = sample_dir / "source.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    temporary.replace(metadata_path)
    return _record_for(sample_dir, feature, gsd_m), valid_fraction, vegetation_fraction


def _round_robin(features: list[OpenCanopyFeature], seed: int) -> list[OpenCanopyFeature]:
    groups: dict[tuple[int, str], list[OpenCanopyFeature]] = {}
    for feature in features:
        groups.setdefault((feature.year, feature.image_name), []).append(feature)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    rng.shuffle(keys)
    ordered: list[OpenCanopyFeature] = []
    while keys:
        next_keys: list[tuple[int, str]] = []
        for key in keys:
            values = groups[key]
            if values:
                ordered.append(values.pop())
            if values:
                next_keys.append(key)
        keys = next_keys
    return ordered


def _write_manifest(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry_index", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--train", type=int, default=48)
    parser.add_argument("--validation", type=int, default=12)
    parser.add_argument("--test", type=int, default=12)
    parser.add_argument("--output-size", type=int, default=384)
    parser.add_argument("--chip-size-m", type=float, default=576.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.50)
    parser.add_argument("--min-vegetation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--max-attempt-multiplier", type=int, default=8)
    args = parser.parse_args()
    if min(args.train, args.validation, args.test) < 0:
        raise ValueError("Split counts cannot be negative")
    if args.output_size <= 0 or args.chip_size_m <= 0:
        raise ValueError("output-size and chip-size-m must be positive")

    features = load_open_canopy_features(args.geometry_index)
    requested = {"train": args.train, "val": args.validation, "test": args.test}
    output_root = args.output_root.resolve()
    records_by_split: dict[str, list[dict[str, str]]] = {name: [] for name in requested}
    claimed_regions: dict[str, str] = {}

    env_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "50000000",
    }
    total_requested = sum(requested.values())
    progress = tqdm(total=total_requested, unit="chip", desc="Open-Canopy subset")
    try:
        with rasterio.Env(**env_options):
            # Test and validation claim coarse regions first. Training then cannot
            # silently reuse a nearby cell even if a future index is malformed.
            for split in ("test", "val", "train"):
                count = requested[split]
                candidates = _round_robin(
                    [feature for feature in features if feature.split == split],
                    args.seed + {"train": 1, "val": 2, "test": 3}[split],
                )
                attempts = 0
                maximum_attempts = max(count, count * args.max_attempt_multiplier)
                for feature in candidates:
                    if len(records_by_split[split]) >= count or attempts >= maximum_attempts:
                        break
                    owner = claimed_regions.get(feature.region)
                    if owner is not None and owner != split:
                        continue
                    attempts += 1
                    error: Exception | None = None
                    for retry in range(3):
                        try:
                            record, valid_fraction, vegetation_fraction = _materialize_feature(
                                feature,
                                output_root,
                                output_size=args.output_size,
                                chip_size_m=args.chip_size_m,
                            )
                            error = None
                            break
                        except Exception as exc:  # network/GDAL failures are resumable
                            error = exc
                            if retry < 2:
                                time.sleep(2 ** retry)
                    if error is not None:
                        tqdm.write(f"skip {feature.sample_id}: {error}")
                        continue
                    if (
                        valid_fraction < args.min_valid_fraction
                        or vegetation_fraction < args.min_vegetation_fraction
                    ):
                        tqdm.write(
                            f"reject {feature.sample_id}: valid={valid_fraction:.1%}, "
                            f"vegetation={vegetation_fraction:.1%}"
                        )
                        continue
                    claimed_regions[feature.region] = split
                    records_by_split[split].append(record)
                    progress.update(1)
                    progress.set_postfix(
                        split=split,
                        valid=f"{valid_fraction:.0%}",
                        vegetation=f"{vegetation_fraction:.0%}",
                    )
                if len(records_by_split[split]) != count:
                    raise RuntimeError(
                        f"Only found {len(records_by_split[split])}/{count} acceptable {split} "
                        f"chips after {attempts} attempts; completed files remain resumable"
                    )
    finally:
        progress.close()

    manifest_dir = output_root / "manifests"
    for split, records in records_by_split.items():
        name = "validation" if split == "val" else split
        _write_manifest(manifest_dir / f"{name}.csv", records)
    summary = {
        "source": OPEN_CANOPY_DATASET_URL,
        "license": OPEN_CANOPY_LICENSE,
        "official_split_preserved": True,
        "coarse_regions_disjoint": True,
        "counts": {("validation" if key == "val" else key): len(value) for key, value in records_by_split.items()},
        "output_size": args.output_size,
        "chip_size_m": args.chip_size_m,
        "gsd_m": args.chip_size_m / args.output_size,
        "minimum_valid_fraction": args.min_valid_fraction,
        "minimum_vegetation_fraction": args.min_vegetation_fraction,
    }
    summary_path = output_root / "subset_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    temporary.replace(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
