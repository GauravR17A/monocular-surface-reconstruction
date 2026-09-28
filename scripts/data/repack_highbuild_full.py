"""Repack selected HighBuild samples into license-filtered, city-disjoint shards.

Each upstream shard is processed atomically. Completed output TARs are skipped,
so an interrupted multi-hour preparation can be restarted safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


MEMBER_COLUMNS = (
    "webdataset_image_member",
    "webdataset_mask_member",
    "webdataset_json_member",
)


def load_selected_rows(split_root: Path) -> dict[str, dict[str, list[dict[str, str]]]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for split in ("train", "validation", "test"):
        with (split_root / f"{split}.csv").open(
            "r", newline="", encoding="utf-8-sig"
        ) as handle:
            for row in csv.DictReader(handle):
                if row["msr_split"] != split:
                    raise ValueError(f"Split mismatch for {row['webdataset_key']}")
                grouped[row["webdataset_shard"]][split].append(row)
    return grouped


def repack_source(
    dataset_root: Path,
    source_relative: str,
    selected_by_split: dict[str, list[dict[str, str]]],
) -> dict[str, int]:
    source = dataset_root / source_relative
    if not source.is_file():
        raise FileNotFoundError(f"Missing upstream shard: {source}")

    writers: dict[str, tarfile.TarFile] = {}
    partials: dict[str, Path] = {}
    finals: dict[str, Path] = {}
    expected_by_split: dict[str, set[str]] = {}
    member_to_split: dict[str, str] = {}
    for split, rows in selected_by_split.items():
        final_relative = rows[0]["msr_shard"]
        if any(row["msr_shard"] != final_relative for row in rows):
            raise ValueError(f"Inconsistent output shard for {source_relative}/{split}")
        final = dataset_root / "msr_splits" / final_relative
        if final.is_file():
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_suffix(final.suffix + ".part")
        expected = {row[column] for row in rows for column in MEMBER_COLUMNS}
        for member in expected:
            if member in member_to_split:
                raise ValueError(f"Duplicate selected TAR member: {member}")
            member_to_split[member] = split
        expected_by_split[split] = expected
        partials[split] = partial
        finals[split] = final
        writers[split] = tarfile.open(partial, mode="w")

    if not writers:
        return {split: len(rows) for split, rows in selected_by_split.items()}

    seen: dict[str, set[str]] = {split: set() for split in writers}
    try:
        with tarfile.open(source, mode="r") as input_tar:
            for member in input_tar:
                split = member_to_split.get(member.name)
                if split is None:
                    continue
                extracted = input_tar.extractfile(member) if member.isfile() else None
                writers[split].addfile(member, extracted)
                seen[split].add(member.name)
    finally:
        for writer in writers.values():
            writer.close()

    for split, expected in expected_by_split.items():
        missing = expected - seen[split]
        unexpected = seen[split] - expected
        if missing or unexpected:
            raise RuntimeError(
                f"Member verification failed for {source_relative}/{split}: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        partials[split].replace(finals[split])
    return {split: len(rows) for split, rows in selected_by_split.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/highbuild_full"))
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("data/highbuild_full/msr_splits"),
    )
    args = parser.parse_args()

    grouped = load_selected_rows(args.split_root)
    completed: dict[str, dict[str, int]] = {}
    for index, (source, selected) in enumerate(sorted(grouped.items()), start=1):
        completed[source] = repack_source(args.dataset_root, source, selected)
        progress = {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "source_shards_complete": index,
            "source_shards_total": len(grouped),
            "last_source_shard": source,
            "sample_counts_by_source": completed,
        }
        (args.split_root / "repack_progress.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Repacked source shard {index}/{len(grouped)}: {source}")


if __name__ == "__main__":
    main()
