"""Download a bounded, label-selected six-category challenge set, never by model score."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
import zipfile

import numpy as np
import rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning

from inspect_openearthmap_archive import RemoteArchive, URL, SIZE

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("duesseldorf", "dhaka", "chiangmai")
PER_REGION = 4
MIN_CLASS_PIXELS = 256
CROSSWALK = np.array([255, 0, 4, 0, 3, 5, 2, 255, 1], dtype=np.uint8)
DESTINATION = Path("D:/MSRData/evaluation/oem_all_six_20260919_v1")
ATTRIBUTION = {
    "dataset": "OpenEarthMap v1, Xia et al., WACV 2023; DOI 10.5281/zenodo.7223446",
    "source": "https://zenodo.org/records/7223446",
    "attribution_url": "https://open-earth-map.org/attribution.html",
    "duesseldorf": {"provider": "German state North Rhine-Westphalia, GeoNRW", "license": "DL-DE-BY-2.0", "license_url": "https://www.govdata.de/dl-de/by-2-0"},
    "dhaka": {"provider": "AIGEO Center, OpenAerialMap", "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/"},
    "chiangmai": {"provider": "UR Field Lab Chiang Mai, OpenAerialMap", "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/"},
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, path)


class BoundedCachedArchive(RemoteArchive):
    """Cache short ranges; reject full responses before reading their body."""
    def __init__(self):
        super().__init__()
        self.cache = b""
        self.cache_start = 0

    def read(self, size=-1):
        size = self.size-self.position if size < 0 else min(size, self.size-self.position)
        if size > 16*1024*1024:
            raise ValueError("Oversized archive read refused")
        if not size:
            return b""
        offset = self.position-self.cache_start
        if not (offset >= 0 and offset+size <= len(self.cache)):
            start = self.position
            for attempt in range(4):
                try:
                    self.cache = super().read(min(max(size, 256*1024), self.size-start))
                    break
                except Exception:
                    self.position = start
                    if attempt == 3:
                        raise
                    time.sleep(2*(attempt+1))
            self.cache_start, self.position, offset = start, start, 0
        result = self.cache[offset:offset+size]
        self.position += size
        return result


def member_parts(name):
    parts = PurePosixPath(name).parts
    if (len(parts) != 4 or parts[0] != "OpenEarthMap_wo_xBD" or parts[1] not in REGIONS
            or parts[2] not in {"images", "labels"} or not parts[3].endswith(".tif")
            or not parts[3].startswith(parts[1]+"_") or "\\" in name):
        raise ValueError(f"Unsupported or unsafe member: {name}")
    return parts


def label_support(labels, valid):
    if labels.shape != valid.shape or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels > 8)):
        raise ValueError("Expected OEM integer label IDs 0..8 on same validity grid")
    mapped = CROSSWALK[labels]
    support = np.bincount(mapped[valid & (mapped != 255)].astype(np.int64), minlength=6)
    return support.tolist(), bool(np.all(support >= MIN_CLASS_PIXELS))


def entry(item):
    return {"name": item.filename, "bytes": item.file_size, "compressed_bytes": item.compress_size,
            "crc32": f"{item.CRC:08x}", "header_offset": item.header_offset}


def download_member(archive, info, destination):
    expected = entry(info)
    receipt_path = destination.with_suffix(".receipt.json")
    if destination.exists():
        if not receipt_path.exists():
            raise RuntimeError(f"Existing data lacks receipt: {destination}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["source"] != expected or receipt["sha256"] != digest(destination) or destination.stat().st_size != info.file_size:
            raise RuntimeError("Cached source identity mismatch")
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tif.partial")
    with archive.open(info) as source, temporary.open("wb") as stream:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            stream.write(chunk)
    if temporary.stat().st_size != info.file_size:
        raise RuntimeError("Extracted length mismatch")
    receipt = {"source": expected, "sha256": digest(temporary), "crc32_verified": True}
    save_json(receipt_path, receipt)
    os.replace(temporary, destination)
    return receipt


def main():
    destination = DESTINATION
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "EVALUATED.json").exists():
        raise RuntimeError("Already evaluated; preserve these files and results")
    contract = {"schema": "msr.oem_all_six_selection.v1", "archive_url": URL,
        "archive_bytes": SIZE, "published_archive_md5_not_verified": "64155d1dc9d3b68536063f79878e1a67",
        "regions_in_order": list(REGIONS), "target_per_region": PER_REGION,
        "min_pixels_per_mapped_class": MIN_CLASS_PIXELS, "crosswalk": CROSSWALK.tolist(),
        "selection": "First four qualifying labels per region in ascending SHA256(filename) order. Qualification uses labels only, never model predictions. No substitution by model quality.",
        "scope": "All-six-present challenge sample, not representative city/global accuracy. Every label-scanned candidate is disclosed and no longer untouched.",
        "checkpoint_sha256": "4940b2d4ae8817370384ccb54da9f73aee1b6bf3614435d8c077bf3e3363d021",
        "source_script_sha256": digest(Path(__file__)), "attribution": ATTRIBUTION,
        "network_budget_bytes_this_attempt": 200*1024*1024}
    contract_path = destination / "selection_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Selection contract changed; refusing silent resume")
    else:
        save_json(contract_path, contract)
    remote = BoundedCachedArchive()
    started = time.monotonic()
    candidates, selected = [], []
    def status(stage, **details):
        save_json(destination / "status.json", {"stage":stage, "network_bytes_this_attempt":remote.transferred,
            "selected_pairs":len(selected), "candidate_labels_inspected":len(candidates),
            "elapsed_seconds":time.monotonic()-started, **details})
        if remote.transferred > contract["network_budget_bytes_this_attempt"]:
            raise RuntimeError("Bounded download budget reached; partial progress saved")
    try:
        status("reading_archive_metadata")
        with zipfile.ZipFile(remote) as archive:
            infos = {item.filename:item for item in archive.infolist() if not item.is_dir()}
            for region in REGIONS:
                labels = [item for name,item in infos.items() if name.startswith(f"OpenEarthMap_wo_xBD/{region}/labels/") and name.endswith(".tif")]
                labels.sort(key=lambda item:hashlib.sha256(item.filename.encode()).hexdigest())
                qualified = 0
                for info in labels:
                    _, _, _, filename = member_parts(info.filename)
                    image_name = info.filename.replace("/labels/", "/images/")
                    if image_name not in infos:
                        raise RuntimeError("Label without distributed image")
                    target = destination / "candidate_labels" / region / filename
                    receipt = download_member(archive, info, target)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", NotGeoreferencedWarning)
                        with rasterio.open(target) as label:
                            counts, acceptable = label_support(label.read(1), label.read_masks(1)>0)
                            shape = list(label.shape)
                    record = {"region":region, "sample_id":Path(filename).stem, "label_path":str(target),
                        "label_receipt":receipt, "class_support_pixels":counts, "shape":shape,
                        "qualifies_all_six":acceptable}
                    candidates.append(record)
                    if acceptable:
                        selected.append({**record, "image_entry":entry(infos[image_name])})
                        qualified += 1
                    save_json(destination / "candidate_labels.json", candidates)
                    status("selecting_by_reference_labels", region=region, selected_in_region=qualified)
                    if len(candidates)%5 == 0 or acceptable:
                        print(f"Labels checked {len(candidates)} | all-six scenes {len(selected)} | {region} {qualified}/{PER_REGION} | network {remote.transferred/1e6:.1f} MB", flush=True)
                    if qualified == PER_REGION:
                        break
            if not selected:
                raise RuntimeError("No scenes met the predefined all-six criterion")
            # Seal chosen IDs before any RGB decode/model inference.
            save_json(destination / "selected_before_rgb.json", selected)
            pairs = []
            for index, row in enumerate(selected, 1):
                info = infos[row["image_entry"]["name"]]
                filename = member_parts(info.filename)[3]
                target = destination / "images" / filename
                receipt = download_member(archive, info, target)
                pairs.append({**row, "image_path":str(target), "image_receipt":receipt})
                status("downloading_selected_rgb", images_downloaded=index, total_selected=len(selected))
                print(f"RGB downloaded {index}/{len(selected)} | network {remote.transferred/1e6:.1f} MB", flush=True)
            save_json(destination / "manifest.json", {"pairs":pairs, "selection_contract_sha256":digest(contract_path),
                "attribution":ATTRIBUTION, "scanned_candidates":len(candidates),
                "network_bytes_this_attempt":remote.transferred, "rgb_pixels_decoded":False,
                "selected_zip_members_crc_verified":True, "whole_archive_md5_verified":False})
            status("ready_for_frozen_evaluation", pairs=len(pairs))
            print(json.dumps({"destination":str(destination), "pairs":len(pairs), "network_mb":remote.transferred/1e6}), flush=True)
    except BaseException as error:
        save_json(destination / "status.json", {"stage":"failed", "error":str(error), "network_bytes_this_attempt":remote.transferred,
            "selected_pairs":len(selected), "candidate_labels_inspected":len(candidates)})
        raise


if __name__ == "__main__":
    main()
