"""Bounded, resumable OEM RGB/label acquisition; never downloads the whole archive."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import zipfile
import numpy as np
import rasterio
import yaml
from stage_oem_all_six_scenes import BoundedCachedArchive, download_member, digest, save_json, entry, URL
from evaluate_christchurch_rgb_v1 import validate_raster_headers

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTIONS = {
    "aachen": "GeoNRW / North Rhine-Westphalia; DL-DE-BY-2.0",
    "chisinau": "HTCD / Lightcyphers; CC BY 4.0",
    "accra": "Open Cities AI / GFDRR; CC BY 4.0",
    "kitsap": "USGS; public domain",
    "bogota": "MaptimeBogota; CC BY 4.0",
    "niamey": "Open Cities AI / GFDRR; CC BY 4.0",
    "melbourne": "City of Melbourne; CC BY 4.0",
    "vienna": "City of Vienna; public domain",
}

def validate_splits(data):
    groups = [set(data[k]) for k in ("train_regions", "validation_regions", "reserved_final_regions", "diagnostic_exclusions")]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i)):
        raise ValueError("Training, development, final reserve and diagnostic regions must be disjoint")
    if set(data["train_regions"]) | set(data["validation_regions"]) != set(ATTRIBUTIONS):
        raise ValueError("Source attribution required for every selected region")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rgb_multiregion_v3.yaml")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    validate_splits(data)
    destination = Path(data["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    contract_path = destination / "contract.json"
    contract = {"schema":config["schema"], "config_sha256":digest(config_path), "data":data,
                "source":URL, "attribution":ATTRIBUTIONS, "attribution_url":"https://open-earth-map.org/attribution.html",
                "scope":"Classification development data. Not LiDAR, height supervision or a final test set.",
                "crosswalk_caveat":"Bareland/developed-space -> ground; rangeland -> low vegetation; agriculture/unknown ignored. Taxonomies are approximate."}
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise RuntimeError("Preparation contract changed; use a new versioned directory")
    save_json(contract_path, contract)
    remote = BoundedCachedArchive()
    with zipfile.ZipFile(remote) as archive:
        members = {x.filename:x for x in archive.infolist()}
    jobs = []
    for split, key in (("train","train_regions"), ("val","validation_regions")):
        for region, count in data[key].items():
            prefix = f"OpenEarthMap_wo_xBD/{region}"
            labels = [n for n in members if n.startswith(prefix+"/labels/") and n.endswith(".tif")]
            labels.sort(key=lambda n: hashlib.sha256(PurePosixPath(n).name.encode()).hexdigest())
            paired = [n for n in labels if n.replace("/labels/", "/images/") in members]
            if len(paired) < count:
                raise RuntimeError(f"Insufficient pairs for predeclared region {region}")
            for label in paired[:count]:
                image = label.replace("/labels/", "/images/")
                jobs.append({"sample_id":PurePosixPath(label).stem,"region":region,"split":split,
                             "image_member":image,"label_member":label})
    expected_bytes = sum(members[r[k]].file_size for r in jobs for k in ("image_member","label_member"))
    if expected_bytes > data["max_download_bytes"]:
        raise RuntimeError("Selected content exceeds predeclared bounded download budget")
    plan = {"contract_sha256":digest(contract_path), "pairs":jobs,"total_file_bytes":expected_bytes}
    plan_path = destination / "download_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Remote selection changed since preparation started")
    save_json(plan_path, plan)
    completed = []
    def progress(stage, **extra):
        save_json(destination / "status.json", {"stage":stage,"pairs":len(completed),"total_pairs":len(jobs),
            "downloaded_file_bytes":sum(r["file_bytes"] for r in completed),"total_file_bytes":expected_bytes,
            "elapsed_seconds":time.monotonic()-started, **extra})
    def fetch(row):
        row = dict(row)
        transfer = BoundedCachedArchive()
        with zipfile.ZipFile(transfer) as archive:
            for role in ("image","label"):
                name = row[f"{role}_member"]
                path = destination / row["split"] / row["region"] / (role+"s") / PurePosixPath(name).name
                receipt = download_member(archive, members[name], path)
                row[f"{role}_path"], row[f"{role}_sha256"] = str(path), receipt["sha256"]
        with rasterio.open(row["image_path"]) as image, rasterio.open(row["label_path"]) as label:
            validate_raster_headers(image, label)
            values = label.read(1)
            if values.dtype.kind not in "iu" or np.any(values > 8):
                raise RuntimeError("Invalid OEM class IDs")
            mapped = np.asarray(data["source_crosswalk"], np.uint8)[values]
            valid = np.all(image.read_masks()>0,axis=0) & (label.read_masks(1)>0) & (mapped != 255)
            row["class_pixels"] = np.bincount(mapped[valid],minlength=6).tolist()
            row["shape"] = list(image.shape)
            if not valid.any():
                raise RuntimeError("Empty paired supervision")
        row["file_bytes"] = sum(Path(row[f"{role}_path"]).stat().st_size for role in ("image","label"))
        return row
    progress("downloading")
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(fetch, row) for row in jobs]
            for future in as_completed(futures):
                completed.append(future.result())
                progress("downloading")
                if len(completed)%10 == 0 or len(completed)==len(jobs):
                    print(f"OEM pairs {len(completed)}/{len(jobs)} | {sum(r['file_bytes'] for r in completed)/1e6:.1f}/{expected_bytes/1e6:.1f} MB", flush=True)
        rows = sorted(completed,key=lambda r:(r["split"],r["region"],r["sample_id"]))
        for split in ("train","val"):
            if np.any(np.sum([r["class_pixels"] for r in rows if r["split"]==split],axis=0)==0):
                raise RuntimeError(f"Missing class support in {split}")
        save_json(destination / "manifest.json", {"contract_sha256":digest(contract_path), "pairs":rows,
            "split_counts":{s:sum(r["split"]==s for r in rows) for s in ("train","val")}})
        progress("ready", manifest_sha256=digest(destination / "manifest.json"))
    except Exception as exc:
        progress("failed",error=repr(exc))
        raise

if __name__ == "__main__":
    main()
