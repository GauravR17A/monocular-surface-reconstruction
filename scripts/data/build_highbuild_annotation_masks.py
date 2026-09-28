"""Build trustworthy HighBuild masks and corrected versioned manifests.

HighBuild height rasters contain building annotations on a zero-filled
background.  The preserved COCO polygons define building support, while the
``is_estimated_height`` flag distinguishes measured/reference heights from
estimated ones.  This script rasterizes those annotations and creates a new
manifest contract without modifying any historical data or report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any

import numpy as np
import rasterio

from msr.data.highbuild import (
    HEIGHT_PROTOCOLS,
    annotation_masks,
    regression_support_mask,
)


SCHEMA = "msr.highbuild_annotation_contract.v2"
SPLITS = ("train", "validation", "test")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(
    path: Path, *, required_field: str = "sample_id"
) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or required_field not in fields:
        raise ValueError(f"invalid manifest: {path}")
    return fields, rows


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_mask(path: Path, mask: np.ndarray, profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    output_profile = dict(profile)
    output_profile.update(
        driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate", predictor=1
    )
    with rasterio.open(temporary, "w", **output_profile) as destination:
        destination.write(mask.astype(np.uint8, copy=False), 1)
    os.replace(temporary, path)


def build_contract(
    *,
    source_manifest_dir: Path,
    split_root: Path,
    output_root: Path,
    height_protocol: str = "strict_measured",
) -> dict[str, Any]:
    if height_protocol not in HEIGHT_PROTOCOLS:
        raise ValueError(
            f"height_protocol must be one of {list(HEIGHT_PROTOCOLS)}, "
            f"got {height_protocol!r}"
        )
    source_manifest_dir = source_manifest_dir.resolve()
    split_root = split_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite corrected contract: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.pending-", dir=output_root.parent)
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "source_manifest_dir": str(source_manifest_dir),
        "source_split_root": str(split_root),
        "output_root": str(output_root),
        "historical_artifacts_modified": False,
        "height_protocol": height_protocol,
        "height_contract": (
            "COCO footprint AND finite positive raster height AND "
            "is_estimated_height=false"
            if height_protocol == "strict_measured"
            else "COCO footprint AND finite positive raster height, including "
            "estimated annotations"
        ),
        "semantic_contract": (
            "All COCO footprints are building labels; outside-building pixels "
            "remain unknown. Semantic support is independent of height support."
        ),
        "test_role": "previously_inspected_regression_evidence_only_not_fresh_holdout",
        "splits": {},
    }
    try:
        split_ids: dict[str, set[str]] = {}
        for split in SPLITS:
            source_manifest = source_manifest_dir / f"{split}.csv"
            fields, rows = _read_csv(source_manifest)
            _, source_rows = _read_csv(
                split_root / f"{split}.csv", required_field="webdataset_key"
            )
            source_by_id = {row["webdataset_key"]: row for row in source_rows}
            corrected: list[dict[str, str]] = []
            split_audit = {
                "rows": len(rows),
                "highbuild_rows": 0,
                "annotations": 0,
                "trusted_annotations": 0,
                "estimated_annotations_excluded": 0,
                "estimated_annotations_included": 0,
                "rejected_annotations": 0,
                "all_footprint_pixels": 0,
                "regression_height_pixels": 0,
                "source_manifest_sha256": file_sha256(source_manifest),
            }
            archives: dict[Path, tarfile.TarFile] = {}
            try:
                for source_row in rows:
                    row = dict(source_row)
                    surface = Path(row["surface_path"])
                    is_highbuild = (
                        row.get("landscape", "").strip().lower() == "urban"
                        and surface.name.lower() == "building_height_m.tif"
                    )
                    if not is_highbuild:
                        corrected.append(row)
                        continue
                    sample_id = row["sample_id"]
                    metadata = source_by_id.get(sample_id)
                    if metadata is None:
                        raise ValueError(f"HighBuild provenance missing for {split}/{sample_id}")
                    archive_path = (split_root / metadata["msr_shard"]).resolve()
                    archive = archives.get(archive_path)
                    if archive is None:
                        archive = tarfile.open(archive_path, "r")
                        archives[archive_path] = archive
                    member = archive.extractfile(metadata["webdataset_json_member"])
                    if member is None:
                        raise FileNotFoundError(metadata["webdataset_json_member"])
                    coco = json.load(member)
                    with rasterio.open(surface) as target_source:
                        target = target_source.read(1).astype(np.float32)
                        profile = target_source.profile
                        nodata = target_source.nodata
                    all_mask, trusted_polygon, counts = annotation_masks(
                        coco, height=target.shape[0], width=target.shape[1]
                    )
                    regression_support = regression_support_mask(
                        all_footprints=all_mask,
                        measured_footprints=trusted_polygon,
                        target=target,
                        nodata=nodata,
                        protocol=height_protocol,
                    )
                    relative_dir = Path("masks") / split / sample_id
                    building_mask = staging / relative_dir / "building_footprints.tif"
                    validity_name = (
                        "measured_height_valid.tif"
                        if height_protocol == "strict_measured"
                        else "annotated_height_valid.tif"
                    )
                    height_valid_mask = staging / relative_dir / validity_name
                    _write_mask(building_mask, all_mask, profile)
                    _write_mask(height_valid_mask, regression_support, profile)
                    row["target_kind"] = "building_height"
                    row["building_mask_path"] = str(
                        output_root / relative_dir / building_mask.name
                    )
                    row["valid_mask_path"] = str(
                        output_root / relative_dir / height_valid_mask.name
                    )
                    corrected.append(row)
                    split_audit["highbuild_rows"] += 1
                    split_audit["annotations"] += counts["annotations"]
                    split_audit["trusted_annotations"] += counts["trusted_annotations"]
                    split_audit["estimated_annotations_excluded"] += counts[
                        "estimated_annotations"
                    ] if height_protocol == "strict_measured" else 0
                    split_audit["estimated_annotations_included"] += counts[
                        "estimated_annotations"
                    ] if height_protocol == "inclusive_annotated" else 0
                    split_audit["rejected_annotations"] += counts[
                        "rejected_nonpositive_or_missing_height_annotations"
                    ]
                    split_audit["all_footprint_pixels"] += int(all_mask.sum())
                    split_audit["regression_height_pixels"] += int(
                        regression_support.sum()
                    )
            finally:
                for archive in archives.values():
                    archive.close()
            destination_manifest = staging / "manifests" / f"{split}.csv"
            _write_csv(destination_manifest, fields, corrected)
            split_audit["output_manifest"] = str(
                output_root / "manifests" / f"{split}.csv"
            )
            split_audit["output_manifest_sha256"] = file_sha256(destination_manifest)
            report["splits"][split] = split_audit
            split_ids[split] = {row["sample_id"] for row in corrected}
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                overlap = split_ids[left] & split_ids[right]
                if overlap:
                    raise ValueError(
                        f"sample leakage between {left} and {right}: {sorted(overlap)[:5]}"
                    )
        report_path = staging / "contract_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest-dir", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--height-protocol",
        choices=HEIGHT_PROTOCOLS,
        default="strict_measured",
    )
    args = parser.parse_args()
    report = build_contract(
        source_manifest_dir=args.source_manifest_dir,
        split_root=args.split_root,
        output_root=args.output_root,
        height_protocol=args.height_protocol,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
