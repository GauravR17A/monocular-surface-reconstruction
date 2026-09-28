"""Repack training samples into city-specific shards for weighted mixing.

The source train shards mix cities and therefore cannot be sampled with explicit
city probabilities. This one-time, atomic repack preserves every member byte and
creates several shards per city so multi-worker WebDataset loading stays useful.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


MEMBER_COLUMNS = (
    "webdataset_image_member",
    "webdataset_mask_member",
    "webdataset_json_member",
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def shard_index(sample_id: str, count: int) -> int:
    digest = hashlib.sha1(sample_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("data/highbuild_full/msr_splits"),
    )
    parser.add_argument("--shards-per-city", type=int, default=8)
    args = parser.parse_args()
    if args.shards_per_city <= 0:
        raise ValueError("shards-per-city must be positive")

    split_root = args.split_root.resolve()
    output_root = split_root / "balanced_train"
    summary_path = output_root / "summary.json"
    with (split_root / "train.csv").open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Training manifest is empty")

    cities = sorted({row["city"] for row in rows})
    final_paths = {
        (city, index): output_root / safe_name(city) / f"part-{index:03d}.tar"
        for city in cities
        for index in range(args.shards_per_city)
    }
    if summary_path.is_file() and all(path.is_file() for path in final_paths.values()):
        print(f"Balanced city shards already complete: {output_root}")
        return

    routes_by_source: dict[str, dict[str, tuple[str, int]]] = defaultdict(dict)
    expected_members: Counter[tuple[str, int]] = Counter()
    sample_counts: Counter[str] = Counter()
    for row in rows:
        city = row["city"]
        index = shard_index(row["webdataset_key"], args.shards_per_city)
        sample_counts[city] += 1
        for column in MEMBER_COLUMNS:
            member = row[column]
            routes_by_source[row["msr_shard"]][member] = (city, index)
            expected_members[(city, index)] += 1

    writers: dict[tuple[str, int], tarfile.TarFile] = {}
    partial_paths: dict[tuple[str, int], Path] = {}
    seen_members: Counter[tuple[str, int]] = Counter()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        for route, final_path in final_paths.items():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            partial = final_path.with_suffix(".tar.part")
            partial.unlink(missing_ok=True)
            partial_paths[route] = partial
            writers[route] = tarfile.open(partial, mode="w")

        for source_number, (source_relative, routes) in enumerate(
            sorted(routes_by_source.items()), start=1
        ):
            source_path = split_root / source_relative
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            with tarfile.open(source_path, mode="r") as source:
                for member in source:
                    route = routes.get(member.name)
                    if route is None:
                        continue
                    extracted = source.extractfile(member) if member.isfile() else None
                    writers[route].addfile(member, extracted)
                    seen_members[route] += 1
            print(
                f"Processed source shard {source_number}/{len(routes_by_source)}: "
                f"{source_relative}"
            )
    finally:
        for writer in writers.values():
            writer.close()

    mismatches = {
        route: (expected_members[route], seen_members[route])
        for route in final_paths
        if expected_members[route] != seen_members[route]
    }
    if mismatches:
        raise RuntimeError(f"Balanced shard member verification failed: {mismatches}")
    for route, partial in partial_paths.items():
        partial.replace(final_paths[route])

    summary = {
        "source_manifest": str((split_root / "train.csv").resolve()),
        "shards_per_city": args.shards_per_city,
        "sample_counts": dict(sorted(sample_counts.items())),
        "shard_patterns": {
            city: str((output_root / safe_name(city) / "*.tar").resolve())
            for city in cities
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
