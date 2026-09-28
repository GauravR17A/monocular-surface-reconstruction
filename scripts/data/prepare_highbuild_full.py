"""Create conservative, city-disjoint Monocular Surface Reconstruction splits for HighBuild-1M."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CONFIRMED_OPEN_CITIES = {
    "Europe_Denmark_Aarhus",
    "Europe_Denmark_Copenhagen",
    "Europe_Denmark_Odense",
    "Europe_France_Lyon",
    "Europe_France_Marseille",
    "Europe_France_Paris",
    "Europe_France_Strasbourg",
    "Europe_France_Toulouse",
    "Europe_Germany_Berlin",
    "Europe_Germany_Frankfurt",
    "Europe_Germany_Munich",
    "Europe_Netherlands_Amsterdam",
    "NorthAmerica_Canada_Toronto",
    "NorthAmerica_Canada_Vancouver",
}

# These cities are held out in their entirety. The choices give both validation
# and test multiple countries/sensors while leaving large cities for training.
VALIDATION_CITIES = {
    "Europe_Denmark_Copenhagen",
    "Europe_France_Lyon",
    "Europe_France_Strasbourg",
    "Europe_Germany_Munich",
}
TEST_CITIES = {
    "Europe_Denmark_Odense",
    "Europe_France_Marseille",
    "Europe_Germany_Frankfurt",
    "Europe_Netherlands_Amsterdam",
    "NorthAmerica_Canada_Vancouver",
}


def assign_split(city: str) -> str:
    if city in TEST_CITIES:
        return "test"
    if city in VALIDATION_CITIES:
        return "validation"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/highbuild_full/benchmark_v1/manifest_webdataset_tiles_1024.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/highbuild_full/msr_splits"),
    )
    args = parser.parse_args()

    with args.manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Full manifest has no header")
        required = {
            "public_dir",
            "webdataset_key",
            "webdataset_shard",
            "webdataset_image_member",
            "webdataset_mask_member",
            "webdataset_json_member",
        }
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Full manifest is missing columns: {sorted(missing)}")
        source_fields = list(reader.fieldnames)
        all_rows = list(reader)
        selected = [row for row in all_rows if row["public_dir"] in CONFIRMED_OPEN_CITIES]

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_fields = source_fields + ["msr_split", "msr_shard"]
    rows_by_split: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for row in selected:
        split = assign_split(row["public_dir"])
        source_stem = Path(row["webdataset_shard"]).stem
        output = dict(row)
        output["msr_split"] = split
        output["msr_shard"] = f"shards/{split}/{source_stem}-{split}.tar"
        rows_by_split[split].append(output)

    city_sets: dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        city_sets[split] = {row["public_dir"] for row in rows}
        with (args.output_root / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(rows)
        (args.output_root / f"{split}_keys.txt").write_text(
            "".join(f"{row['webdataset_key']}\n" for row in rows), encoding="utf-8"
        )

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = city_sets[left] & city_sets[right]
        if overlap:
            raise RuntimeError(f"City leakage between {left} and {right}: {sorted(overlap)}")

    summary = {
        "policy": "confirmed-open imagery cities only; conservative non-legal assessment",
        "source_license_inventory": "data/highbuild_full/LICENSES.md",
        "sample_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "city_counts": {
            split: dict(sorted(Counter(row["public_dir"] for row in rows).items()))
            for split, rows in rows_by_split.items()
        },
        "excluded_or_conditional_cities": sorted(
            {row["public_dir"] for row in all_rows} - CONFIRMED_OPEN_CITIES
        ),
        "notes": [
            "Upstream random split is intentionally ignored.",
            "No city appears in more than one Monocular Surface Reconstruction split.",
            "Re-check upstream source terms before commercial release; this is not legal advice.",
        ],
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
