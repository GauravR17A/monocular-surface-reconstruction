"""Download a deterministic, geographically split HighBuild reviewer subset.

This is engineering smoke data, not a substitute for the full benchmark. The
upstream dataset card declares ``license: other``; review and approve the full
dataset terms before any production or commercial training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


REPOSITORY = "feifei140729/small-sample"
REVISION = "main"
API_URL = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}"
RESOLVE_URL = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def fetch_tree() -> list[dict]:
    request = Request(
        f"{API_URL}?recursive=true&expand=false",
        headers={"User-Agent": "Monocular Surface Reconstruction dataset preparation"},
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Hugging Face repository tree response")
    return payload


def paired_files(tree: list[dict], samples_per_region: int) -> dict[str, list[tuple[str, str]]]:
    images: dict[tuple[str, str], str] = {}
    heights: dict[tuple[str, str], str] = {}
    for item in tree:
        if item.get("type") != "file":
            continue
        remote_path = str(item["path"])
        parts = Path(remote_path).parts
        suffix = Path(remote_path).suffix.lower()
        if len(parts) == 3 and parts[0] == "images" and suffix in IMAGE_SUFFIXES:
            images[(parts[1], Path(parts[2]).stem)] = remote_path
        elif (
            len(parts) == 4
            and parts[0] == "masks"
            and parts[2] == "masks"
            and suffix in {".tif", ".tiff"}
        ):
            heights[(parts[1], Path(parts[3]).stem)] = remote_path

    common = sorted(set(images) & set(heights))
    grouped: dict[str, list[tuple[str, str]]] = {}
    for region, stem in common:
        grouped.setdefault(region, []).append((images[(region, stem)], heights[(region, stem)]))
    selected = {
        region: pairs[:samples_per_region]
        for region, pairs in grouped.items()
        if len(pairs) >= samples_per_region
    }
    if len(selected) < 3:
        raise RuntimeError("Need at least three complete regions for leakage-safe splits")
    return selected


def split_regions(regions: list[str], seed: int) -> dict[str, str]:
    shuffled = sorted(regions)
    random.Random(seed).shuffle(shuffled)
    evaluation_count = max(1, round(len(shuffled) * 0.15))
    validation_count = max(1, round(len(shuffled) * 0.15))
    test_regions = set(shuffled[:evaluation_count])
    validation_regions = set(shuffled[evaluation_count : evaluation_count + validation_count])
    return {
        region: (
            "test"
            if region in test_regions
            else "validation"
            if region in validation_regions
            else "train"
        )
        for region in shuffled
    }


def download_one(remote_path: str, destination: Path, expected_size: int | None) -> str:
    if destination.is_file() and (expected_size is None or destination.stat().st_size == expected_size):
        return "cached"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(
        f"{RESOLVE_URL}/{quote(remote_path, safe='/')}",
        headers={"User-Agent": "Monocular Surface Reconstruction dataset preparation"},
    )
    with urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise IOError(
            f"Size mismatch for {remote_path}: {partial.stat().st_size} != {expected_size}"
        )
    partial.replace(destination)
    return "downloaded"


def write_manifests(
    output_root: Path,
    grouped: dict[str, list[tuple[str, str]]],
    assignments: dict[str, str],
) -> dict[str, int]:
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    for region, pairs in sorted(grouped.items()):
        split = assignments[region]
        for image_path, height_path in pairs:
            stem = Path(image_path).stem
            rows[split].append(
                {
                    "sample_id": f"{region}__{stem}",
                    "region": region,
                    "rgb_path": f"../{image_path}",
                    "height_path": f"../{height_path}",
                    "mask_path": "",
                    "gsd_m": "",
                }
            )
    columns = ["sample_id", "region", "rgb_path", "height_path", "mask_path", "gsd_m"]
    for split, split_rows in rows.items():
        with (manifest_root / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(split_rows)
    return {split: len(split_rows) for split, split_rows in rows.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/highbuild_reviewer"))
    parser.add_argument("--samples-per-region", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if args.samples_per_region <= 0 or args.workers <= 0:
        raise ValueError("samples-per-region and workers must be positive")

    tree = fetch_tree()
    grouped = paired_files(tree, args.samples_per_region)
    assignments = split_regions(list(grouped), args.seed)
    size_by_path = {
        str(item["path"]): int(item["size"])
        for item in tree
        if item.get("type") == "file" and item.get("size") is not None
    }
    jobs = [path for pairs in grouped.values() for pair in pairs for path in pair]
    print(f"Preparing {len(jobs) // 2} pairs from {len(grouped)} regions ({len(jobs)} files)")

    counts = {"cached": 0, "downloaded": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one,
                remote_path,
                args.output_root / remote_path,
                size_by_path.get(remote_path),
            ): remote_path
            for remote_path in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            status = future.result()
            counts[status] += 1
            if index % 25 == 0 or index == len(futures):
                print(f"Files complete: {index}/{len(futures)}")

    split_counts = write_manifests(args.output_root, grouped, assignments)
    metadata = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "declared_license": "other",
        "intended_use": "reviewer subset; engineering and data-quality smoke tests only",
        "seed": args.seed,
        "samples_per_region": args.samples_per_region,
        "region_assignments": assignments,
        "sample_counts": split_counts,
        "download_counts": counts,
    }
    (args.output_root / "provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
