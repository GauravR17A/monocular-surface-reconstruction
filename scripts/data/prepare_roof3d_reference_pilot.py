"""Stage the small official Roof3D v3 reference tile, without training on it.

Uses published checksums, verifies raster grids, and records the dataset licence.
It is a *development diagnostic*, not a claim of an untouched final holdout.
No production checkpoint, old dataset, or application inference file is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import rasterio

RECORD = "https://zenodo.org/api/records/10910492"
NAMES = {"koeln0rgb.tif", "koeln0dsm.tif", "koeln0dtm.tif", "GT_roi_30.tif"}


def digest(path: Path) -> str:
    value = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "md5:" + value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("D:/MSRData/data/roof3d_reference_pilot_v1"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD, timeout=30) as response:
        record = json.load(response)
    files = {item["key"]: item for item in record["files"]}
    if set(files) != NAMES or record["metadata"].get("license", {}).get("id") != "cc-by-4.0":
        raise RuntimeError("Upstream file set/licence changed: inspect before downloading")
    size = sum(item["size"] for item in files.values())
    if size > 65_000_000:
        raise RuntimeError("Pilot size unexpectedly exceeds 65 MB")
    report = {
        "schema": "msr.roof3d_reference_pilot.v1",
        "record": RECORD, "doi": "10.5281/zenodo.10910492", "license": "CC-BY-4.0",
        "attribution": "Schuegraf, Philipp; project leader Bittner, Ksenia; German Aerospace Center (DLR). Roof3D - 3D Reconstruction (v3), 2024.",
        "role": "development_reference_diagnostic_not_training_or_untouched_final_holdout",
        "upstream_role": "published_test_area_with_3D_ground_truth",
        "height_datum": "Do not subtract GT from DTM until GT meaning/datum is independently verified.",
        "inference_contract": "RGB-only production inference; DSM/DTM/GT files must never be model inputs when reporting RGB-only accuracy.",
        "files": {}, "total_bytes": size,
    }
    done = 0
    for name in sorted(NAMES):
        item = files[name]
        path = root / name
        if path.exists():
            if path.stat().st_size != item["size"] or digest(path) != item["checksum"]:
                raise RuntimeError(f"Existing file differs from upstream: preserve and inspect {path}")
        else:
            temporary = root / (name + ".part")
            # A failed partial download is disposable; completed sources are not.
            with urllib.request.urlopen(item["links"]["self"], timeout=60) as response, temporary.open("wb") as stream:
                received = 0
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    received += len(chunk)
                    print(f"Roof3D pilot: {(done + received) / 1e6:.1f} / {size / 1e6:.1f} MB", flush=True)
            if temporary.stat().st_size != item["size"] or digest(temporary) != item["checksum"]:
                raise RuntimeError(f"Checksum mismatch, not promoted: {temporary}")
            temporary.rename(path)
        done += item["size"]
        with rasterio.open(path) as dataset:
            masked = dataset.read(1, masked=True)
            values = masked.compressed()
            values = values[np.isfinite(values)]
            report["files"][name] = {
                "path": str(path), "bytes": item["size"], "checksum": item["checksum"],
                "source": item["links"]["self"], "count": dataset.count,
                "shape": [dataset.height, dataset.width], "crs": str(dataset.crs),
                "transform": list(dataset.transform), "nodata": dataset.nodata,
                "dtype": dataset.dtypes, "band1_valid_pixels": int(values.size),
                "band1_range": [float(values.min()), float(values.max())] if values.size else None,
            }
    rasters = list(report["files"].values())
    report["matching_grid"] = all(all(r[key] == rasters[0][key] for key in ("shape", "crs", "transform")) for r in rasters)
    report["ready_for_training"] = False
    report["remaining"] = [
        "Obtain geographically separated RGB/roof-plane training annotations (Roof3D v2 is a separate 4.72 GB archive).",
        "Verify ground-truth units, vertical datum, nodata/background semantics and acquisition alignment.",
        "Preserve building/roof-part instances and unknown masks; do not infer planes from one scalar building height.",
        "Train an isolated RGB roof-part/ridge model; freeze production height inference.",
        "Validate flat/gable/hip/complex roofs and rooftop additions with coverage and rejection rates.",
    ]
    (root / "provenance_and_grid_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(root), "bytes": done, "matching_grid": report["matching_grid"], "training_started": False}), flush=True)


if __name__ == "__main__":
    main()
