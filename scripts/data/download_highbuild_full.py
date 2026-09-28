"""Resumably download HighBuild-1M metadata and optional WebDataset shards."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY = "feifei140729/HighBuild-1M"
REVISION = "2f5d76f8c5b5b4b7e925871d46db7aae64fd3daa"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/highbuild_full"))
    parser.add_argument("--include-shards", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    patterns = ["README.md", "LICENSES.md", "docs/**", "benchmark_v1/**"]
    if args.include_shards:
        patterns.append("data/webdataset/**")
        print(
            "WARNING: HighBuild-1M contains mixed imagery licenses. Downloading "
            "does not make every city suitable for commercial model training."
        )
    downloaded = snapshot_download(
        repo_id=REPOSITORY,
        repo_type="dataset",
        revision=REVISION,
        local_dir=args.output_root,
        allow_patterns=patterns,
        max_workers=args.workers,
    )
    record = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "include_shards": args.include_shards,
        "local_snapshot": str(Path(downloaded).resolve()),
        "license_notice": (
            "Mixed per-source terms; consult LICENSES.md. Exclude sources marked "
            "EXCLUDE from production training unless permission is obtained."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "msr_download.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
