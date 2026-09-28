"""Download only the pinned ImageNet MiT-B0 backbone into the data volume."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download
import requests


REPOSITORY = "nvidia/mit-b0"
REVISION = "80983a413c30d36a39c20203974ae7807835e2b4"
WEIGHT_SHA256 = "4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc"
DESTINATION = Path("D:/MSRData/models/segformer-mit-b0-imagenet")
FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for name in FILES:
        path = Path(hf_hub_download(REPOSITORY, name, revision=REVISION, local_dir=DESTINATION))
        digest = sha256(path)
        if name == "pytorch_model.bin" and digest != WEIGHT_SHA256:
            raise RuntimeError("Pinned upstream weight SHA-256 mismatch")
        artifacts[name] = {"sha256": digest, "size_bytes": path.stat().st_size}
        print(f"Verified {name}: {path.stat().st_size:,} bytes", flush=True)
    license_url = "https://raw.githubusercontent.com/NVlabs/SegFormer/master/LICENSE"
    license_response = requests.get(license_url, timeout=30)
    license_response.raise_for_status()
    license_path = DESTINATION / "LICENSE.upstream.txt"
    license_path.write_bytes(license_response.content)
    artifacts[license_path.name] = {"sha256": sha256(license_path), "size_bytes": license_path.stat().st_size}
    manifest = {
        "schema": "msr.rgb_segmenter_backbone.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "revision": REVISION,
        "source": f"https://huggingface.co/{REPOSITORY}/tree/{REVISION}",
        "pretraining": "ImageNet-1k classification encoder; six-class decoder is newly initialized",
        "ground_truth_height_input": False,
        "license": "NVIDIA SegFormer license: non-commercial research/evaluation only; retain notices",
        "license_source": license_url,
        "files": artifacts,
    }
    destination = DESTINATION / "backbone_manifest.json"
    # Existing identical artifacts are retained; only the derived manifest updates.
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
