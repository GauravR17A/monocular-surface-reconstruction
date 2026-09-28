"""Extract a small city-balanced urban surface subset from protected TAR shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import rasterio
from tqdm import tqdm

from msr.data.highbuild import (
    HEIGHT_PROTOCOLS,
    annotation_masks,
    regression_support_mask,
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

# HighBuild supplies annotated building heights, not an observed full-surface
# nDSM.  Keeping this semantic distinction in the manifest is critical: areas
# outside the annotated buildings are unknown, rather than confirmed zero-height
# ground/canopy targets.
HIGHBUILD_TARGET_KIND = "building_height"


def _stable_score(seed: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def _balanced_rows(rows: list[dict[str, str]], per_region: int, seed: int) -> list[dict[str, str]]:
    by_region: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_region.setdefault(row["public_dir"], []).append(row)
    selected: list[dict[str, str]] = []
    for region, values in sorted(by_region.items()):
        # Interleave sparse, medium, and dense building scenes, then use a stable
        # hash inside each bucket. The adapter needs routing diversity, not only
        # the visually busiest city blocks.
        buckets = {"sparse": [], "medium": [], "dense": []}
        for row in values:
            count = int(row.get("num_annotations") or 0)
            bucket = "sparse" if count <= 3 else "medium" if count <= 10 else "dense"
            buckets[bucket].append(row)
        for bucket in buckets.values():
            bucket.sort(key=lambda row: _stable_score(seed, row["webdataset_key"]))
        region_selected: list[dict[str, str]] = []
        names = ["dense", "medium", "sparse"]
        while len(region_selected) < per_region and any(buckets.values()):
            for name in names:
                if buckets[name] and len(region_selected) < per_region:
                    region_selected.append(buckets[name].pop())
        if len(region_selected) < per_region:
            raise ValueError(
                f"Region {region} contains only {len(region_selected)} usable rows; "
                f"requested {per_region}"
            )
        selected.extend(region_selected)
    return selected


def _atomic_member_write(archive: tarfile.TarFile, member_name: str, destination: Path) -> None:
    if destination.is_file():
        return
    member = archive.getmember(member_name)
    source = archive.extractfile(member)
    if source is None:
        raise FileNotFoundError(f"Cannot read {member_name} from {archive.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as handle:
        while chunk := source.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(destination)


def _validate_pair(rgb_path: Path, surface_path: Path) -> None:
    with rasterio.open(rgb_path) as rgb, rasterio.open(surface_path) as surface:
        if (rgb.width, rgb.height) != (surface.width, surface.height):
            raise ValueError(f"Unaligned pair: {rgb_path} and {surface_path}")
        if rgb.count < 3 or surface.count != 1:
            raise ValueError(f"Unexpected bands: RGB={rgb.count}, surface={surface.count}")


def _write_mask(path: Path, mask: np.ndarray, profile: dict[str, object]) -> None:
    """Write a resumable mask, rejecting conflicting existing content."""

    if path.is_file():
        with rasterio.open(path) as source:
            existing = source.read(1)
        if existing.shape != mask.shape or not np.array_equal(existing > 0, mask > 0):
            raise ValueError(f"Existing HighBuild contract mask differs: {path}")
        return
    output_profile = dict(profile)
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        predictor=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with rasterio.open(temporary, "w", **output_profile) as destination:
        destination.write(mask.astype(np.uint8, copy=False), 1)
    temporary.replace(path)


def _build_contract_masks(
    *,
    annotation_path: Path,
    surface_path: Path,
    building_mask_path: Path,
    valid_mask_path: Path,
    height_protocol: str,
) -> dict[str, int]:
    with annotation_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)
    with rasterio.open(surface_path) as surface:
        target = surface.read(1).astype(np.float32)
        profile = surface.profile
        nodata = surface.nodata
    footprints, measured, counts = annotation_masks(
        coco, height=target.shape[0], width=target.shape[1]
    )
    regression_support = regression_support_mask(
        all_footprints=footprints,
        measured_footprints=measured,
        target=target,
        nodata=nodata,
        protocol=height_protocol,
    )
    _write_mask(building_mask_path, footprints, profile)
    _write_mask(valid_mask_path, regression_support, profile)
    return {
        **counts,
        "semantic_building_pixels": int(footprints.sum()),
        "regression_height_pixels": int(regression_support.sum()),
    }


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
    parser.add_argument("split_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--train-per-region", type=int, default=24)
    parser.add_argument("--validation-per-region", type=int, default=12)
    parser.add_argument("--test-per-region", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument(
        "--height-protocol",
        choices=HEIGHT_PROTOCOLS,
        default="strict_measured",
        help=(
            "strict_measured excludes estimated heights from regression; "
            "inclusive_annotated includes every annotated positive raster height"
        ),
    )
    args = parser.parse_args()
    split_root = args.split_root.resolve()
    output_root = args.output_root.resolve()
    requested = {
        "train": args.train_per_region,
        "validation": args.validation_per_region,
        "test": args.test_per_region,
    }
    summary: dict[str, object] = {
        "source": str(split_root),
        "height_protocol": args.height_protocol,
        "target_kind": HIGHBUILD_TARGET_KIND,
        "semantic_contract": "all COCO building footprints",
        "regression_contract": (
            "measured non-estimated positive annotated heights"
            if args.height_protocol == "strict_measured"
            else "all positive annotated heights, including estimated heights"
        ),
        "splits": {},
    }
    for split, per_region in requested.items():
        manifest_path = split_root / f"{split}.csv"
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        selected = _balanced_rows(rows, per_region, args.seed)
        by_shard: dict[Path, list[dict[str, str]]] = {}
        for row in selected:
            shard = (split_root / row["msr_shard"]).resolve()
            if not shard.is_file():
                raise FileNotFoundError(shard)
            by_shard.setdefault(shard, []).append(row)

        records: list[dict[str, str]] = []
        progress = tqdm(total=len(selected), desc=f"HighBuild {split}", unit="sample")
        for shard, shard_rows in sorted(by_shard.items()):
            with tarfile.open(shard, "r") as archive:
                for row in shard_rows:
                    sample_id = row["webdataset_key"]
                    sample_dir = output_root / split / sample_id
                    rgb_path = sample_dir / "rgb.jpg"
                    surface_path = sample_dir / "building_height_m.tif"
                    annotation_path = sample_dir / "annotations.json"
                    building_mask_path = sample_dir / "building_footprints.tif"
                    validity_name = (
                        "measured_height_valid.tif"
                        if args.height_protocol == "strict_measured"
                        else "annotated_height_valid.tif"
                    )
                    valid_mask_path = sample_dir / validity_name
                    _atomic_member_write(
                        archive, row["webdataset_image_member"], rgb_path
                    )
                    _atomic_member_write(
                        archive, row["webdataset_mask_member"], surface_path
                    )
                    _atomic_member_write(
                        archive, row["webdataset_json_member"], annotation_path
                    )
                    _validate_pair(rgb_path, surface_path)
                    mask_audit = _build_contract_masks(
                        annotation_path=annotation_path,
                        surface_path=surface_path,
                        building_mask_path=building_mask_path,
                        valid_mask_path=valid_mask_path,
                        height_protocol=args.height_protocol,
                    )
                    source_path = sample_dir / "source.json"
                    if not source_path.is_file():
                        temporary = source_path.with_suffix(".json.tmp")
                        with temporary.open("w", encoding="utf-8") as handle:
                            json.dump(
                                {
                                    "sample_id": sample_id,
                                    "city": row["public_dir"],
                                    "source_shard": str(shard),
                                    "source_members": {
                                        "rgb": row["webdataset_image_member"],
                                        "height": row["webdataset_mask_member"],
                                        "annotations": row["webdataset_json_member"],
                                    },
                                    "height_protocol": args.height_protocol,
                                    "mask_audit": mask_audit,
                                    "license_audit": str(
                                        (split_root.parent / "LICENSES.md").resolve()
                                    ),
                                },
                                handle,
                                indent=2,
                            )
                        temporary.replace(source_path)
                    records.append(
                        {
                            "sample_id": sample_id,
                            "region": row["public_dir"],
                            "landscape": "urban",
                            "rgb_path": str(rgb_path),
                            "surface_path": str(surface_path),
                            "target_kind": HIGHBUILD_TARGET_KIND,
                            "dtm_path": "",
                            # Semantic footprints and height-validity are
                            # deliberately separate: estimated annotations can
                            # still teach "building" without entering the
                            # strict measured-height regression protocol.
                            "building_mask_path": str(building_mask_path),
                            "vegetation_mask_path": "",
                            "valid_mask_path": str(valid_mask_path),
                            "relative_prior_path": "",
                            "gsd_m": "",
                        }
                    )
                    progress.update(1)
        progress.close()
        _write_manifest(output_root / "manifests" / f"{split}.csv", records)
        summary["splits"][split] = {
            "samples": len(records),
            "regions": sorted({row["region"] for row in records}),
        }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "subset_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    temporary.replace(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
