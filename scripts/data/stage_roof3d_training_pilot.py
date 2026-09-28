"""Fetch a bounded RGB/DSM/roof-label development pilot using strict ZIP ranges.

Does not fetch the complete 4.72 GB archive or access its validation pixels.
The pilot is for schema/label inspection, NOT a spatially independent split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import urllib.request
import zipfile
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspect_openearthmap_archive import RemoteArchive

ROOT = Path("D:/MSRData/data/roof3d_training_pilot_v1")
RECORD = "https://zenodo.org/api/records/8300629"


def file_identity(path):
    sha = hashlib.sha256()
    crc = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            sha.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return sha.hexdigest(), crc


def main():
    with urllib.request.urlopen(RECORD, timeout=30) as response:
        record = json.load(response)
    if record["metadata"].get("license", {}).get("id") != "cc-by-4.0":
        raise RuntimeError("Licence changed: review before acquisition")
    archive_file = next(item for item in record["files"] if item["key"] == "roof3d_newest.zip")
    remote = RemoteArchive(url=archive_file["links"]["self"], size=archive_file["size"])
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        rgb = sorted(name for name in names if name.startswith("ROOF3D/train/rgb/") and name.endswith(".tif"))
        if len(rgb) < 16:
            raise RuntimeError(f"Unexpected RGB archive layout: {names[:8]}")
        # Uniform index coverage only; geography and synthetic/real roles must
        # be recovered from upstream metadata before making training splits.
        selected_rgb = [rgb[i * (len(rgb) - 1) // 15] for i in range(16)]
        selected = ["ROOF3D/train/annotation_plane.json", "ROOF3D/train/annotation_sec.json"]
        selected += selected_rgb + [name.replace("/rgb/", "/dsm/") for name in selected_rgb]
        entries = [archive.getinfo(name) for name in selected]
        if sum(item.compress_size for item in entries) > 65_000_000:
            raise RuntimeError("Selected pilot exceeds 65 MB download budget")
        receipts = []
        for i, item in enumerate(entries):
            parts = PurePosixPath(item.filename).parts
            if parts[:2] != ("ROOF3D", "train") or ".." in parts or "\\" in item.filename:
                raise RuntimeError("Unsafe archive member")
            path = (ROOT / Path(*parts[1:])).resolve()
            if not path.is_relative_to(ROOT.resolve()):
                raise RuntimeError("Member escapes staging directory")
            if path.exists():
                sha, crc = file_identity(path)
                if path.stat().st_size != item.file_size or crc != item.CRC:
                    raise RuntimeError(f"Existing file mismatch: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(path.suffix + ".part")
                with archive.open(item) as source, partial.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
                sha, crc = file_identity(partial)
                if partial.stat().st_size != item.file_size or crc != item.CRC:
                    raise RuntimeError("ZIP CRC32/size mismatch")
                partial.rename(path)
            receipts.append({"member": item.filename, "path": str(path), "bytes": item.file_size, "crc32": f"{crc:08x}", "sha256": sha})
            print(f"Roof3D training pilot: {i+1}/{len(entries)} files | {remote.transferred/1e6:.1f} MB transferred", flush=True)
        manifest = {
            "schema": "msr.roof3d_training_pilot.v1", "record": RECORD,
            "license": "CC-BY-4.0", "attribution": "Philipp Schuegraf, Ksenia Bittner and collaborators, DLR; Roof3D, 2023.",
            "role": "training_source_schema_audit_only_no_training_started",
            "selection": "16 evenly spaced filenames from published train/rgb; not a geographic holdout",
            "validation_pixels_accessed": False, "train_rgb_count_in_archive": len(rgb),
            "whole_archive_bytes": archive_file["size"], "whole_archive_downloaded": False,
            "network_bytes": remote.transferred,
            "integrity": "Selected ZIP member CRC32 and length verified; local SHA256 recorded; whole archive MD5 NOT verified",
            "files": receipts,
        }
        (ROOT / "acquisition_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summaries = {}
    selected_names = {Path(name).name for name in selected_rgb}
    for kind in ("plane", "sec"):
        annotation = json.loads((ROOT / "train" / f"annotation_{kind}.json").read_text(encoding="utf-8"))
        images = [im for im in annotation["images"] if Path(im["file_name"]).name in selected_names]
        ids = {im["id"] for im in images}
        labels = [a for a in annotation["annotations"] if a["image_id"] in ids]
        summaries[kind] = {
            "keys": list(annotation), "categories": annotation.get("categories"),
            "all_images": len(annotation["images"]), "selected_images": images,
            "selected_annotations": len(labels), "annotation_fields": list(labels[0]) if labels else [],
            "segmentation_example": {k: v for k,v in labels[0].get("segmentation",{}).items() if k != "counts"} if labels and isinstance(labels[0].get("segmentation"),dict) else "polygon_or_missing",
            "info": annotation.get("info"), "licenses": annotation.get("licenses"),
        }
        subset = {**annotation, "images": images, "annotations": labels}
        (ROOT / "train" / f"pilot_annotation_{kind}.json").write_text(json.dumps(subset) + "\n", encoding="utf-8")
    (ROOT / "label_schema_audit.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(ROOT), "network_mb": remote.transferred/1e6, "labels": {k:v["selected_annotations"] for k,v in summaries.items()}, "training_started": False}), flush=True)


if __name__ == "__main__":
    main()
