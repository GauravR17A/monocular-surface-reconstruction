"""Stage every Christchurch TIFF without decoding any pixels or running a model."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import time
import zipfile
from datetime import datetime, timezone

from inspect_openearthmap_archive import RemoteArchive, SIZE, URL, inventory

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = Path("D:/MSRData/external_holdouts/oem_christchurch_rgb_v1")
PROTOCOL = ROOT / "docs/RGB_SEGMENTER_V1_CHRISTCHURCH_PROTOCOL.md"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    os.replace(temporary, path)


class CachedArchive(RemoteArchive):
    """Small bounded cache avoids repeated network requests for ZIP headers."""
    def __init__(self):
        super().__init__()
        self.cache = b""
        self.cache_start = 0

    def read(self, size=-1):
        size = SIZE-self.position if size < 0 else min(size, SIZE-self.position)
        if size > 16*1024*1024:
            raise ValueError("Oversized archive read")
        if not size:
            return b""
        offset = self.position-self.cache_start
        if not (offset >= 0 and offset+size <= len(self.cache)):
            start = self.position
            for attempt in range(4):
                try:
                    self.cache = super().read(min(max(size, 4*1024*1024), SIZE-start))
                    break
                except Exception:
                    self.position = start
                    if attempt == 3:
                        raise
                    time.sleep(5*(attempt+1))
            self.cache_start = start
            self.position = start
            offset = 0
        result = self.cache[offset:offset+size]
        self.position += size
        return result


def relative_member(name):
    parts = PurePosixPath(name).parts
    if len(parts) != 4 or parts[:2] != ("OpenEarthMap_wo_xBD", "christchurch") or parts[2] not in {"images", "labels"}:
        raise ValueError(f"Unexpected member: {name}")
    if not parts[3].startswith("christchurch_") or not parts[3].endswith(".tif") or "\\" in name:
        raise ValueError("Invalid region TIFF name")
    return Path(parts[2]) / parts[3]


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if (DESTINATION / "CONSUMED.json").exists():
        raise RuntimeError("Holdout consumed; staging is locked")
    protocol_hash = digest(PROTOCOL)
    remote = CachedArchive()
    with zipfile.ZipFile(remote) as archive:
        selected = sorted(inventory(archive), key=lambda item: item["header_offset"])
        groups = {group: {Path(relative_member(row["name"])).stem for row in selected
                          if relative_member(row["name"]).parts[0] == group} for group in ("images", "labels")}
        if not groups["labels"] or not groups["labels"].issubset(groups["images"]) or len(selected) != len(groups["images"])+len(groups["labels"]):
            raise RuntimeError("Region does not have a complete unique paired inventory")
        full_inventory = selected
        excluded_unlabelled = sorted(groups["images"]-groups["labels"])
        selected = [item for item in selected if relative_member(item["name"]).stem in groups["labels"]]
        contract = {"source": URL, "archive_bytes": SIZE, "protocol_sha256": protocol_hash,
                    "entries": selected, "pair_count": len(groups["labels"]),
                    "full_region_inventory": full_inventory, "excluded_no_released_label": excluded_unlabelled,
                    "integrity": "Selected ZIP CRC32/length + local SHA256; whole archive MD5 NOT verified"}
        contract_path = DESTINATION / "acquisition_contract.json"
        if contract_path.exists():
            if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
                raise RuntimeError("Acquisition contract changed; refusing resume")
        else:
            write_json(contract_path, contract)
        rows = []
        total = sum(row["bytes"] for row in selected)
        completed = 0
        for index, item in enumerate(selected, 1):
            target = (DESTINATION / relative_member(item["name"])).resolve()
            if not target.is_relative_to(DESTINATION.resolve()):
                raise RuntimeError("Member escaped holdout directory")
            receipt_path = target.with_suffix(".receipt.json")
            if target.exists():
                if not receipt_path.exists():
                    raise RuntimeError(f"Existing file without receipt: {target}")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt["source"] != item or receipt["sha256"] != digest(target) or target.stat().st_size != item["bytes"]:
                    raise RuntimeError("Existing staged data failed identity check")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".tif.partial")
                with archive.open(item["name"]) as source, temporary.open("wb") as output:
                    for chunk in iter(lambda: source.read(1024*1024), b""):
                        output.write(chunk)
                # Reading ZipExtFile to EOF validates the publisher ZIP's CRC32.
                if temporary.stat().st_size != item["bytes"]:
                    raise RuntimeError("Extracted size mismatch")
                receipt = {"source": item, "sha256": digest(temporary), "crc32_verified": True}
                write_json(receipt_path, receipt)
                os.replace(temporary, target)
            rows.append({"path": str(target.relative_to(DESTINATION)), **receipt})
            completed += item["bytes"]
            status = {"stage": "staging_no_pixel_decode", "files": index, "total_files": len(selected),
                      "extracted_bytes": completed, "total_extracted_bytes": total,
                      "network_bytes_this_attempt": remote.transferred,
                      "updated_utc": datetime.now(timezone.utc).isoformat()}
            write_json(DESTINATION / "status.json", status)
            if index % 10 == 0 or index == len(selected):
                print(f"Staging {index}/{len(selected)} files | {completed/1e6:.1f}/{total/1e6:.1f} MB extracted", flush=True)
        if digest(PROTOCOL) != protocol_hash:
            raise RuntimeError("Protocol changed during acquisition")
        write_json(DESTINATION / "manifest.json", {"contract_sha256": digest(contract_path),
                   "protocol_sha256": protocol_hash, "files": sorted(rows, key=lambda row: row["path"]),
                   "pair_count": len(groups["labels"]), "pixels_decoded": False})
        status["stage"] = "staged_not_consumed"
        write_json(DESTINATION / "status.json", status)
        print("Complete: all pairs verified; no TIFF pixels decoded; external evaluation pending.", flush=True)


if __name__ == "__main__":
    main()
