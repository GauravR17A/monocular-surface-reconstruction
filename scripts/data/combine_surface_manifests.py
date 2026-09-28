"""Combine explicit surface manifests while preserving every provenance field."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from msr.data.surface_dataset import load_surface_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    seen: set[str] = set()
    for source_path in args.inputs:
        # Strict validation occurs before anything is written.
        load_surface_manifest(source_path)
        with source_path.resolve().open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                sample_id = row["sample_id"].strip()
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample_id across manifests: {sample_id}")
                seen.add(sample_id)
                # Resolve paths now so combining manifests in another directory
                # cannot silently break relative source paths.
                base = source_path.resolve().parent
                for key in (
                    "rgb_path",
                    "surface_path",
                    "dtm_path",
                    "building_mask_path",
                    "vegetation_mask_path",
                    "valid_mask_path",
                    "relative_prior_path",
                ):
                    value = (row.get(key) or "").strip()
                    if value:
                        path = Path(value)
                        row[key] = str((path if path.is_absolute() else base / path).resolve())
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    load_surface_manifest(args.output)
    print(f"Combined {len(rows)} samples into {args.output.resolve()}")


if __name__ == "__main__":
    main()
