"""Backfill aligned Open-Canopy LiDAR class IDs into a versioned sidecar root.

Existing RGB/height/mask files and accepted manifests are read-only inputs.  By
default the locked official test inventory is excluded.  This script streams
only the requested classification windows; it never downloads full mosaics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from tqdm import tqdm

from msr.data.open_canopy import OPEN_CANOPY_CRS
from msr.data.open_canopy_classification import (
    IGN_LIDAR_HD_CLASS_NAMES,
    OPEN_CANOPY_CLASSIFICATION_NODATA,
    OPEN_CANOPY_CLASSIFICATION_SIDECAR_SCHEMA,
    RICH_VEGETATION_CLASS_IDS,
    classification_histogram,
    preserve_classification_ids,
)


REPORT_SCHEMA = "msr.open_canopy.classification_backfill_report.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_aligned_classification(
    url: str,
    rgb_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(rgb_path) as reference:
        if str(reference.crs).upper() != OPEN_CANOPY_CRS:
            raise ValueError(f"unexpected prepared RGB CRS {reference.crs}")
        target = {
            "crs": reference.crs,
            "transform": reference.transform,
            "width": reference.width,
            "height": reference.height,
            "bounds": list(reference.bounds),
        }
    with rasterio.open(url) as source:
        if str(source.crs).upper() != OPEN_CANOPY_CRS:
            raise ValueError(f"unexpected classification CRS {source.crs} for {url}")
        with WarpedVRT(
            source,
            crs=target["crs"],
            transform=target["transform"],
            width=target["width"],
            height=target["height"],
            resampling=Resampling.nearest,
        ) as aligned:
            raw = aligned.read(1)
            source_valid = aligned.read_masks(1) > 0
    return preserve_classification_ids(raw, source_valid), target


def write_classification_artifact(
    output_dir: Path,
    *,
    sample_id: str,
    split: str,
    source_url: str,
    rgb_path: Path,
    values: np.ndarray,
    reference: dict[str, Any],
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one immutable-style class raster plus its provenance sidecar."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raster_path = output_dir / "lidar_classification_id.tif"
    temporary = output_dir / "lidar_classification_id.partial.tif"
    width = int(reference["width"])
    height = int(reference["height"])
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint8",
        "crs": reference["crs"],
        "transform": reference["transform"],
        "compress": "deflate",
        "predictor": 2,
        "nodata": OPEN_CANOPY_CLASSIFICATION_NODATA,
    }
    if width >= 16 and height >= 16:
        profile.update(
            tiled=True,
            blockxsize=max(16, min(256, (width // 16) * 16)),
            blockysize=max(16, min(256, (height // 16) * 16)),
        )
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(values.astype(np.uint8, copy=False), 1)
        destination.update_tags(
            MSR_PRODUCT="open_canopy_lidar_classification_ids",
            MSR_SCHEMA=OPEN_CANOPY_CLASSIFICATION_SIDECAR_SCHEMA,
            MSR_SAMPLE_ID=sample_id,
            MSR_OFFICIAL_SPLIT=split,
            MSR_SOURCE_URL=source_url,
            MSR_ALIGNMENT_REFERENCE=str(rgb_path),
        )
    temporary.replace(raster_path)

    histogram = classification_histogram(values)
    context = source_context or {}
    metadata = {
        "schema": OPEN_CANOPY_CLASSIFICATION_SIDECAR_SCHEMA,
        "sample_id": sample_id,
        "official_split": split,
        "classification_raster": str(raster_path.resolve()),
        "classification_raster_sha256": _sha256(raster_path),
        "aligned_rgb": str(rgb_path.resolve()),
        "aligned_rgb_sha256": _sha256(rgb_path),
        "source_url": source_url,
        "source_identity": {
            "region": context.get("region"),
            "image_name": context.get("image_name"),
            "imagery_acquisition_date": context.get("imagery_acquisition_date"),
            "lidar_acquisition_date": context.get("lidar_acquisition_date"),
            "source_metadata_path": context.get("source_metadata_path"),
            "source_metadata_sha256": context.get("source_metadata_sha256"),
        },
        "crs": str(reference["crs"]),
        "bounds": [float(value) for value in reference["bounds"]],
        "width": int(reference["width"]),
        "height": int(reference["height"]),
        "nodata_id": OPEN_CANOPY_CLASSIFICATION_NODATA,
        "preservation_contract": (
            "source-valid upstream byte IDs are retained exactly; only source-invalid "
            "pixels become 255"
        ),
        "known_ign_lidar_hd_class_names": {
            str(key): value for key, value in sorted(IGN_LIDAR_HD_CLASS_NAMES.items())
        },
        "candidate_rich_vegetation_ids": sorted(RICH_VEGETATION_CLASS_IDS),
        "class_pixel_counts": histogram,
    }
    _atomic_json(output_dir / "classification_source.json", metadata)
    return metadata


def _selected_records(
    report: dict[str, Any], splits: set[str]
) -> Iterable[dict[str, Any]]:
    for record in report.get("records", []):
        if str(record.get("split")) in splits:
            yield record


def backfill(
    provenance_report: str | Path,
    output_root: str | Path,
    *,
    splits: set[str],
    sample_ids: set[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    report_path = Path(provenance_report).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    provenance = json.loads(report_path.read_text(encoding="utf-8"))
    if provenance.get("schema") != "msr.open_canopy.provenance.v1":
        raise ValueError("unsupported Open-Canopy provenance report schema")
    prepared_root = Path(provenance["prepared_subset_root"]).expanduser().resolve()
    if destination == prepared_root or prepared_root in destination.parents:
        raise ValueError(
            "classification sidecars require a separate versioned output root; "
            "the prepared binary subset is read-only"
        )
    records = list(_selected_records(provenance, splits))
    if sample_ids:
        records = [record for record in records if record.get("sample_id") in sample_ids]
        found = {str(record["sample_id"]) for record in records}
        missing = sorted(sample_ids - found)
        if missing:
            raise ValueError(
                f"requested sample IDs are absent from selected accepted splits: {missing}"
            )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        records = records[:limit]
    results: list[dict[str, Any]] = []
    issues: list[str] = []
    aggregate = Counter()
    env_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "50000000",
    }
    with rasterio.Env(**env_options):
        for record in tqdm(records, unit="chip", desc="Open-Canopy class sidecars"):
            sample_id = str(record["sample_id"])
            split = str(record["split"])
            source_metadata = json.loads(
                Path(record["source_metadata_path"]).read_text(encoding="utf-8")
            )
            rgb_path = Path(source_metadata.get("rgb_path", ""))
            if not rgb_path.is_file():
                rgb_path = (
                    Path(record["source_metadata_path"]).resolve().parent / "rgb.tif"
                )
            source_url = str(source_metadata["urls"]["classification"])
            output_dir = destination / split / sample_id
            existing_metadata = output_dir / "classification_source.json"
            if existing_metadata.is_file():
                existing = json.loads(existing_metadata.read_text(encoding="utf-8"))
                existing_raster = Path(existing["classification_raster"])
                if (
                    existing.get("schema")
                    == OPEN_CANOPY_CLASSIFICATION_SIDECAR_SCHEMA
                    and existing.get("sample_id") == sample_id
                    and existing.get("official_split") == split
                    and existing.get("source_url") == source_url
                    and existing.get("known_ign_lidar_hd_class_names")
                    == {
                        str(key): value
                        for key, value in sorted(IGN_LIDAR_HD_CLASS_NAMES.items())
                    }
                    and existing_raster.is_file()
                    and _sha256(existing_raster)
                    == existing.get("classification_raster_sha256")
                    and rgb_path.is_file()
                    and _sha256(rgb_path) == existing.get("aligned_rgb_sha256")
                ):
                    results.append(existing)
                    aggregate.update(existing.get("class_pixel_counts", {}))
                    continue
            try:
                values, reference = _read_aligned_classification(source_url, rgb_path)
                metadata = write_classification_artifact(
                    output_dir,
                    sample_id=sample_id,
                    split=split,
                    source_url=source_url,
                    rgb_path=rgb_path,
                    values=values,
                    reference=reference,
                    source_context=record,
                )
            except Exception as error:  # network/GDAL failures remain resumable
                issues.append(f"{split}/{sample_id}: {error}")
                continue
            results.append(metadata)
            aggregate.update(metadata["class_pixel_counts"])

    completed_by_split = Counter(item["official_split"] for item in results)
    summary = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_report": str(report_path),
        "provenance_report_sha256": _sha256(report_path),
        "output_root": str(destination),
        "requested_splits": sorted(splits),
        "requested_sample_ids": sorted(sample_ids) if sample_ids else None,
        "limit": limit,
        "locked_test_excluded": "test" not in splits,
        "requested_samples": len(records),
        "completed_samples": len(results),
        "completed_by_split": dict(sorted(completed_by_split.items())),
        "class_pixel_counts": dict(sorted(aggregate.items(), key=lambda item: int(item[0]))),
        "issues": issues,
        "passes": len(results) == len(records) and not issues,
    }
    _atomic_json(destination / "backfill_report.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provenance_report", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val"),
    )
    parser.add_argument(
        "--confirm-locked-test",
        action="store_true",
        help="Required if --splits includes the official locked test inventory.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Backfill only this accepted sample ID; repeat for multiple samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N selected provenance records (smoke tests).",
    )
    args = parser.parse_args()
    splits = set(args.splits)
    if "test" in splits and not args.confirm_locked_test:
        raise ValueError(
            "locked test sidecars require explicit --confirm-locked-test"
        )
    result = backfill(
        args.provenance_report,
        args.output_root,
        splits=splits,
        sample_ids=set(args.sample_ids or []),
        limit=args.limit,
    )
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
