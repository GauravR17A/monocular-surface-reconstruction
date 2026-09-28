"""Fetch pinned research inference artifacts without deserializing downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--include-foundation", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "model-artifacts.json").read_text())
    print(manifest["usage"])
    for entry in manifest["artifacts"]:
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to((ROOT / "models").resolve()):
            raise ValueError("Artifact path must stay inside models/")
        if path.is_file() and path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"]:
            print(f"Verified {entry['name']}")
            continue
        if args.verify_only:
            raise SystemExit(f"Missing or changed artifact: {entry['name']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            with urlopen(Request(entry["url"], headers={"User-Agent": "MSR-research-release"}), timeout=120) as source, temporary.open("wb") as target:
                count = 0
                while block := source.read(1024 * 1024):
                    count += len(block)
                    if count > entry["bytes"]:
                        raise ValueError("Download exceeds declared artifact size")
                    target.write(block)
            if temporary.stat().st_size != entry["bytes"] or digest(temporary) != entry["sha256"]:
                raise ValueError(f"Integrity check failed: {entry['name']}")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded and verified {entry['name']}")
    if args.include_foundation:
        from huggingface_hub import snapshot_download
        foundation = manifest["foundation"]
        if args.verify_only:
            for entry in foundation["files"]:
                path = ROOT / foundation["path"] / entry["name"]
                if not path.is_file() or digest(path) != entry["sha256"]:
                    raise SystemExit(f"Missing or changed foundation file: {entry['name']}")
        else:
            snapshot_download(repo_id=foundation["repository"], revision=foundation["revision"],
                              local_dir=ROOT / foundation["path"],
                              allow_patterns=[item["name"] for item in foundation["files"]])
            for entry in foundation["files"]:
                if digest(ROOT / foundation["path"] / entry["name"]) != entry["sha256"]:
                    raise ValueError(f"Foundation identity mismatch: {entry['name']}")
        print("Verified foundation model")


if __name__ == "__main__":
    main()
