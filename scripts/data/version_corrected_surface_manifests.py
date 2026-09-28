"""Create immutable manifests with corrected HighBuild label semantics.

Historical Monocular Surface Reconstruction manifests tagged exported HighBuild
``building_height_m.tif`` rasters as complete nDSMs.  Those rasters only
supervise annotated buildings; their zero background is not observed ground or
canopy height.  This tool creates a separately versioned manifest set that marks
those rows as ``building_height`` and records exact input/output hashes.

It never opens image, height, validation, or test raster content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


SCHEMA = "msr.corrected_surface_label_contract.v2"
SPLITS = ("train", "validation", "test")
REQUIRED_FIELDS = {
    "sample_id",
    "region",
    "landscape",
    "surface_path",
    "target_kind",
    "dtm_path",
    "building_mask_path",
    "vegetation_mask_path",
    "valid_mask_path",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def is_exported_highbuild_building_height(row: dict[str, str]) -> bool:
    """Recognize only the known HighBuild building-only export contract."""

    surface_name = Path((row.get("surface_path") or "").strip()).name.lower()
    return (
        (row.get("landscape") or "").strip().lower() == "urban"
        and surface_name == "building_height_m.tif"
        and not (row.get("dtm_path") or "").strip()
        and not (row.get("building_mask_path") or "").strip()
        and not (row.get("vegetation_mask_path") or "").strip()
        and not (row.get("valid_mask_path") or "").strip()
    )


def correct_rows(
    rows: list[dict[str, str]], *, split: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    corrected: list[dict[str, str]] = []
    changed_ids: list[str] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"{split} has an empty or duplicate sample_id: {sample_id!r}")
        seen.add(sample_id)
        if is_exported_highbuild_building_height(row):
            old_kind = (row.get("target_kind") or "").strip().lower()
            if old_kind not in {"ndsm", "building_height"}:
                raise ValueError(
                    f"HighBuild row {sample_id} has unexpected target_kind {old_kind!r}"
                )
            row["target_kind"] = "building_height"
            # HighBuild's exported height raster itself encodes the annotated
            # building support: positive values are labelled buildings and
            # zeros outside that support are unknown.  Reusing it as the
            # explicit mask preserves sub-2 m labels and avoids an inferred
            # threshold mask without duplicating raster data.
            row["building_mask_path"] = row["surface_path"]
            if old_kind != "building_height":
                changed_ids.append(sample_id)
        corrected.append(row)
    return corrected, {
        "rows": len(corrected),
        "changed_rows": len(changed_ids),
        "changed_sample_ids_sha256": hashlib.sha256(
            json.dumps(changed_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = REQUIRED_FIELDS - set(fields)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        return fields, list(reader)


def write_manifest(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_versioned_manifests(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite corrected manifests: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.pending-", dir=output_dir.parent)
    )
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "source_manifest_directory": str(input_dir),
        "output_manifest_directory": str(output_dir),
        "rule": (
            "Known HighBuild building_height_m.tif rows are building-only; "
            "their positive raster support is the explicit building mask, and "
            "outside-building pixels are unknown and excluded from height and "
            "semantic metrics."
        ),
        "raster_content_opened": False,
        "official_test_content_opened": False,
        "test_manifest_metadata_versioned": True,
        "historical_scores_directly_comparable": False,
        "splits": {},
    }
    try:
        all_ids: dict[str, set[str]] = {}
        for split in SPLITS:
            source_path = input_dir / f"{split}.csv"
            fields, rows = read_manifest(source_path)
            corrected, audit = correct_rows(rows, split=split)
            destination = staging / f"{split}.csv"
            write_manifest(destination, fields, corrected)
            ids = {(row.get("sample_id") or "").strip() for row in corrected}
            all_ids[split] = ids
            summary["splits"][split] = {
                **audit,
                "source_path": str(source_path),
                "source_sha256": file_sha256(source_path),
                "output_path": str(output_dir / f"{split}.csv"),
                "output_sha256": file_sha256(destination),
            }
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                overlap = all_ids[left] & all_ids[right]
                if overlap:
                    raise ValueError(
                        f"sample leakage between {left} and {right}: {sorted(overlap)[:5]}"
                    )
        contract_path = staging / "label_contract.json"
        contract_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_versioned_manifests(args.input_dir, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
