"""Download a deterministic GAMUS pilot before resuming the complete dataset.

The Hugging Face repository contains tens of thousands of small HDF5 files and
its default order can download one modality for hours before any complete RGB /
height / class triplet is usable.  This command selects geographically mixed,
evenly spaced triplets from every official split, downloads those first, and
leaves the local directory fully compatible with a later snapshot resume.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download


REPOSITORY = "earthflow/GAMUS"
ALLOWED_ROOT = Path(r"D:\MSRData")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=ALLOWED_ROOT / "GAMUS")
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--val", type=int, default=120)
    parser.add_argument("--test", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=12)
    return parser.parse_args()


def _evenly_spaced(values: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    if count >= len(values):
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]


def _balanced_selection(sample_ids: list[str], requested: int) -> list[str]:
    by_city: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(sample_ids):
        by_city[sample_id.split("_", maxsplit=1)[0]].append(sample_id)
    cities = sorted(by_city)
    if not cities:
        raise ValueError("No GAMUS cities were discovered")
    requested = min(requested, len(sample_ids))
    base, remainder = divmod(requested, len(cities))
    selected: list[str] = []
    for index, city in enumerate(cities):
        allocation = base + (1 if index < remainder else 0)
        selected.extend(_evenly_spaced(by_city[city], allocation))
    return sorted(selected)


def _paths_for(split: str, sample_ids: list[str]) -> list[str]:
    paths: list[str] = []
    for sample_id in sample_ids:
        paths.extend(
            (
                f"images/{split}/{sample_id}_RGB.h5",
                f"heights/{split}/{sample_id}_AGL.h5",
                f"classes/{split}/{sample_id}_CLS.h5",
            )
        )
    return paths


def main() -> None:
    args = _parse_args()
    target = args.target.expanduser().resolve()
    allowed = ALLOWED_ROOT.resolve()
    if allowed not in target.parents:
        raise ValueError(f"GAMUS target must stay below {allowed}; received {target}")
    if args.max_workers < 1 or args.max_workers > 8:
        raise ValueError("max-workers must be between 1 and 8")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    counts = {"train": args.train, "val": args.val, "test": args.test}
    if any(value < 1 for value in counts.values()):
        raise ValueError("Every priority split count must be positive")

    target.mkdir(parents=True, exist_ok=True)
    cache_root = allowed / ".hf_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_XET_CACHE"] = str(cache_root / "xet")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    repository_files = HfApi().list_repo_files(REPOSITORY, repo_type="dataset")
    available: dict[str, list[str]] = {}
    selections: dict[str, list[str]] = {}
    patterns = ["README.md"]
    for split, requested in counts.items():
        prefix = f"images/{split}/"
        suffix = "_RGB.h5"
        sample_ids = [
            path[len(prefix) : -len(suffix)]
            for path in repository_files
            if path.startswith(prefix) and path.endswith(suffix)
        ]
        available[split] = sorted(sample_ids)
        selections[split] = _balanced_selection(sample_ids, requested)
        patterns.extend(_paths_for(split, selections[split]))

    selection_path = target / "msr_priority_selection.json"
    selection_payload = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "balanced_by_city_evenly_spaced",
        "requested": counts,
        "available": {split: len(items) for split, items in available.items()},
        "selected": selections,
        "file_count": len(patterns) - 1,
    }
    temporary = selection_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(selection_path)
    print(
        "Priority GAMUS selection: "
        + ", ".join(f"{split}={len(items)}" for split, items in selections.items()),
        flush=True,
    )

    for attempt in range(1, args.max_attempts + 1):
        try:
            snapshot_download(
                repo_id=REPOSITORY,
                repo_type="dataset",
                local_dir=target,
                allow_patterns=patterns,
                max_workers=args.max_workers,
            )
            break
        except Exception as error:
            if attempt >= args.max_attempts:
                raise RuntimeError(
                    f"Priority GAMUS download failed after {attempt} resumable attempts"
                ) from error
            delay = min(60, attempt * 10)
            print(
                f"Attempt {attempt}/{args.max_attempts} paused: {type(error).__name__}. "
                f"Existing files are safe; retrying in {delay}s.",
                flush=True,
            )
            time.sleep(delay)

    missing = [path for path in patterns if path != "README.md" and not (target / path).is_file()]
    if missing:
        raise RuntimeError(
            f"Priority download returned without {len(missing)} selected files; "
            f"first missing file: {missing[0]}"
        )
    print(
        f"Priority GAMUS subset ready: {sum(map(len, selections.values()))} complete triplets",
        flush=True,
    )


if __name__ == "__main__":
    main()
