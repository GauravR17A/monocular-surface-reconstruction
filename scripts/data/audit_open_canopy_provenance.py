"""Create a fail-closed provenance index for a prepared Open-Canopy subset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from msr.data.open_canopy import (
    OPEN_CANOPY_DATASET_URL,
    OPEN_CANOPY_LICENSE,
    OpenCanopyFeature,
    load_open_canopy_features,
    open_canopy_urls,
)


SCHEMA = "msr.open_canopy.provenance.v1"
SPLIT_ROLES = {
    "train": "learning",
    "val": "development_validation",
    "test": "locked_evaluation_inventory_only",
}
MANIFEST_NAMES = {
    "train": "train.csv",
    "val": "validation.csv",
    "test": "test.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_index(features: list[OpenCanopyFeature]) -> dict[str, OpenCanopyFeature]:
    counts = Counter(feature.sample_id for feature in features)
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate official Open-Canopy sample IDs: {duplicates[:5]}")
    return {feature.sample_id: feature for feature in features}


def _bounds_within(
    inner: list[float] | tuple[float, ...],
    outer: tuple[float, float, float, float],
    *,
    tolerance: float = 1.0e-6,
) -> bool:
    if len(inner) != 4:
        return False
    left, bottom, right, top = (float(value) for value in inner)
    return (
        left >= outer[0] - tolerance
        and bottom >= outer[1] - tolerance
        and right <= outer[2] + tolerance
        and top <= outer[3] + tolerance
        and right > left
        and top > bottom
    )


def _selected_sample_ids(root: Path, split: str) -> list[str]:
    manifest_path = root / "manifests" / MANIFEST_NAMES[split]
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Open-Canopy provenance requires the accepted {split} manifest: "
            f"{manifest_path}"
        )
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values = [str(row.get("sample_id", "")).strip() for row in rows]
    if any(not value for value in values):
        raise ValueError(f"Empty sample_id in {manifest_path}")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate sample_id in {manifest_path}")
    return values


def build_provenance_report(
    geometry_index: str | Path,
    subset_root: str | Path,
) -> dict[str, Any]:
    """Audit metadata only; no official test image or label is opened."""

    geometry_path = Path(geometry_index).expanduser().resolve()
    root = Path(subset_root).expanduser().resolve()
    features = load_open_canopy_features(geometry_path)
    official = _feature_index(features)
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    regions_by_split: dict[str, set[str]] = defaultdict(set)
    image_scenes_by_split: dict[str, set[str]] = defaultdict(set)
    seen_subset_ids: set[str] = set()
    unselected_materialized: dict[str, int] = {}

    for split in SPLIT_ROLES:
        split_root = root / split
        selected_ids = _selected_sample_ids(root, split)
        materialized_ids = {
            path.parent.name for path in split_root.glob("*/source.json")
        }
        unselected_materialized[split] = len(materialized_ids - set(selected_ids))
        for selected_id in selected_ids:
            source_path = split_root / selected_id / "source.json"
            if not source_path.is_file():
                issues.append(
                    f"Accepted {split} sample has no source metadata: {selected_id}"
                )
                continue
            try:
                metadata = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                issues.append(f"Unreadable source metadata {source_path}: {error}")
                continue
            sample_id = str(metadata.get("sample_id", "")).strip()
            if not sample_id:
                issues.append(f"Missing sample_id in {source_path}")
                continue
            if sample_id in seen_subset_ids:
                issues.append(f"Duplicate prepared sample {sample_id}")
                continue
            seen_subset_ids.add(sample_id)
            feature = official.get(sample_id)
            if feature is None:
                issues.append(f"Prepared sample absent from official geometry: {sample_id}")
                continue
            if split != feature.split or metadata.get("official_split") != split:
                issues.append(
                    f"Split mismatch for {sample_id}: directory={split}, "
                    f"metadata={metadata.get('official_split')}, official={feature.split}"
                )
            if int(metadata.get("year", -1)) != feature.year:
                issues.append(f"Year mismatch for {sample_id}")
            if str(metadata.get("image_name", "")) != feature.image_name:
                issues.append(f"Image identity mismatch for {sample_id}")
            if str(metadata.get("region", "")) != feature.region:
                issues.append(f"Region mismatch for {sample_id}")
            if not _bounds_within(metadata.get("bounds", []), feature.bounds):
                issues.append(f"Prepared bounds escape official cell for {sample_id}")
            expected_urls = open_canopy_urls(feature)
            if metadata.get("urls") != expected_urls:
                issues.append(f"Product URL mismatch for {sample_id}")
            if metadata.get("source") != OPEN_CANOPY_DATASET_URL:
                issues.append(f"Dataset source mismatch for {sample_id}")
            if metadata.get("license") != OPEN_CANOPY_LICENSE:
                issues.append(f"Licence mismatch for {sample_id}")

            regions_by_split[split].add(feature.region)
            image_scenes_by_split[split].add(feature.image_name)
            records.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "protocol_role": SPLIT_ROLES[split],
                    "region": feature.region,
                    "official_cell_bounds_epsg2154": list(feature.bounds),
                    "prepared_bounds_epsg2154": [
                        float(value) for value in metadata.get("bounds", [])
                    ],
                    "gsd_m": float(metadata["gsd_m"]),
                    "lidar_year": feature.year,
                    "lidar_acquisition_date": feature.lidar_acquisition_date,
                    "imagery_acquisition_date": feature.imagery_acquisition_date,
                    "image_name": feature.image_name,
                    "lidar_points_in_official_cell": feature.lidar_points,
                    "lidar_source_url": feature.lidar_url,
                    "product_urls": expected_urls,
                    "source_metadata_path": str(source_path),
                    "source_metadata_sha256": _sha256(source_path),
                }
            )

    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    overlap_report: dict[str, dict[str, list[str]]] = {}
    for left, right in split_pairs:
        region_overlap = sorted(regions_by_split[left] & regions_by_split[right])
        sample_key = f"{left}__{right}"
        overlap_report[sample_key] = {
            "regions": region_overlap,
            "source_images": sorted(
                image_scenes_by_split[left] & image_scenes_by_split[right]
            ),
        }
        if region_overlap:
            issues.append(
                f"Geographic region leakage between {left} and {right}: "
                f"{region_overlap[:5]}"
            )

    counts: dict[str, dict[str, int]] = {}
    for split in SPLIT_ROLES:
        subset = [record for record in records if record["split"] == split]
        counts[split] = {
            "samples": len(subset),
            "unique_regions": len(regions_by_split[split]),
            "unique_source_images": len(image_scenes_by_split[split]),
            "unique_imagery_dates": len(
                {record["imagery_acquisition_date"] for record in subset}
            ),
            "unique_lidar_dates": len(
                {record["lidar_acquisition_date"] for record in subset}
            ),
        }

    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passes": not issues,
        "audit_scope": (
            "Prepared metadata and official geometry only; locked test pixels "
            "were not opened or used for decisions"
        ),
        "official_geometry": {
            "path": str(geometry_path),
            "sha256": _sha256(geometry_path),
            "feature_count": len(features),
        },
        "prepared_subset_root": str(root),
        "dataset_source": OPEN_CANOPY_DATASET_URL,
        "license": OPEN_CANOPY_LICENSE,
        "counts": counts,
        "unselected_materialized_candidates": unselected_materialized,
        "cross_split_overlaps": overlap_report,
        "issues": issues,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry_index", type=Path)
    parser.add_argument("subset_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = build_provenance_report(args.geometry_index, args.subset_root)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({key: report[key] for key in ("passes", "counts", "issues")}, indent=2))
    if not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
