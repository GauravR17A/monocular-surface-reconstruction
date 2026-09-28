"""Height-independent RGB/semantic GAMUS inputs for a controlled DC+PHL pilot.

Only RGB and CLS HDF5 datasets are opened. AGL, relative depth, and existing
height-model outputs are deliberately absent from this data contract.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .gamus_dataset import (
    GAMUS_APPROVED_INDEX_SCHEMA,
    GAMUS_SIX_CLASS_NAMES,
    GAMUS_SOURCE_ID_TO_SIX_CLASS,
    _declared_nodata,
    _image_dataset,
    _require_h5py,
)
from .raster_dataset import _crop_origin


ALLOWED_CITIES = ("DC", "PHL")
IGNORE_INDEX = 255
EXPECTED_SPLIT_COUNTS = {"train": 3837, "val": 859}
_SAMPLE_ID = re.compile(r"(?:DC|PHL)_[A-Za-z0-9_]+\Z")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authenticated_json(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    source = Path(path).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)):
        raise ValueError("An exact lowercase SHA-256 is required")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {source}")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError(f"JSON must contain an object: {source}")
    return result


@dataclass(frozen=True)
class GamusRgbRecord:
    sample_id: str
    city: str
    image_path: Path
    class_path: Path

    @property
    def region(self) -> str:
        return f"GAMUS_{self.city}"


class GamusRgbSegmentationDataset(Dataset):
    """Raw 0..1 RGB and independent label/image validity; no height reads.

    Validation is the whole native tile, never a resized/central crop. The
    count/native-size overrides exist for small isolated CPU fixtures; the
    production factory fixes the approved 3837/859 and 1024-pixel contract.
    """

    def __init__(self, root: str | Path, split: str, *, approved_index_path: str | Path,
                 approved_index_sha256: str, patch_size: int = 384,
                 random_crop: bool = False, augment: bool = False,
                 rgb_scale: float = 255.0, dark_pixel_threshold: float = 0.15,
                 expected_split_counts: Mapping[str, int] | None = None,
                 native_shape: int = 1024) -> None:
        if split not in ("train", "val"):
            raise ValueError("Only approved train/val are permitted; no test or holdout access")
        if patch_size <= 0 or native_shape <= 0 or patch_size > native_shape:
            raise ValueError("patch_size must be positive and no larger than native_shape")
        if not math.isfinite(rgb_scale) or rgb_scale <= 0:
            raise ValueError("rgb_scale must be finite and positive")
        if not 0 <= dark_pixel_threshold <= 1:
            raise ValueError("dark_pixel_threshold must lie in [0,1]")
        if split == "val" and (random_crop or augment or patch_size != native_shape):
            raise ValueError("Validation must be raw full-native tiles, without crops or augmentation")
        expected = dict(EXPECTED_SPLIT_COUNTS if expected_split_counts is None else expected_split_counts)
        if set(expected) != {"train", "val"} or any(int(v) <= 0 for v in expected.values()):
            raise ValueError("Expected train/val counts must be positive")
        index = authenticated_json(approved_index_path, approved_index_sha256)
        if index.get("schema") != GAMUS_APPROVED_INDEX_SCHEMA or index.get("dataset") != "GAMUS":
            raise ValueError("Unexpected approved GAMUS index schema/dataset")
        splits = index.get("splits", {})
        if set(splits) != {"train", "val"}:
            raise ValueError("Approved index must contain only train/val, never test")
        ids_by_split = {}
        for name, payload in splits.items():
            ids = payload.get("approved_sample_ids")
            if not isinstance(ids, list) or not all(isinstance(s, str) and _SAMPLE_ID.fullmatch(s) for s in ids):
                raise ValueError("Only safe DC/PHL sample IDs are permitted; no NYC or traversal")
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate approved sample IDs")
            if len(ids) != expected[name] or payload.get("approved_count") != expected[name]:
                raise ValueError(f"Approved {name} count differs from the fixed split")
            if payload.get("semantic_eligible_sample_ids") != ids:
                raise ValueError("All approved records must be semantic-eligible in exact index order")
            ids_by_split[name] = tuple(ids)
        if set(ids_by_split["train"]) & set(ids_by_split["val"]):
            raise ValueError("Train/validation sample IDs overlap")
        self.root = Path(root).expanduser().resolve()
        self.approved_index_path = Path(approved_index_path).resolve()
        self.approved_index_sha256 = approved_index_sha256
        self.split, self.patch_size = split, int(patch_size)
        self.random_crop, self.augment = bool(random_crop), bool(augment)
        self.rgb_scale, self.dark_pixel_threshold = float(rgb_scale), float(dark_pixel_threshold)
        self.native_shape = int(native_shape)
        self.records = []
        # Direct paths from approved IDs: do not glob/discover unrelated cities.
        for sample_id in ids_by_split[split]:
            rgb = (self.root / "images" / split / f"{sample_id}_RGB.h5").resolve()
            cls = (self.root / "classes" / split / f"{sample_id}_CLS.h5").resolve()
            for path in (rgb, cls):
                if not path.is_relative_to(self.root):
                    raise ValueError(f"Source escaped approved root: {path}")
                if not path.is_file():
                    raise FileNotFoundError(path)
            self.records.append(GamusRgbRecord(sample_id, sample_id.split("_", 1)[0], rgb, cls))
        self.sample_ids = tuple(record.sample_id for record in self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        h5 = _require_h5py()
        with h5.File(record.image_path, "r") as image_handle, h5.File(record.class_path, "r") as class_handle:
            rgb = _image_dataset(image_handle, record.image_path)
            cls = _image_dataset(class_handle, record.class_path)
            if rgb.ndim != 3 or rgb.shape[-1] < 3 or cls.ndim != 2 or tuple(rgb.shape[:2]) != tuple(cls.shape):
                raise ValueError(f"Unaligned HWC RGB and 2D CLS: {record.sample_id}")
            if tuple(cls.shape) != (self.native_shape, self.native_shape):
                raise ValueError(f"Unexpected native tile shape: {record.sample_id}: {cls.shape}")
            row = _crop_origin(cls.shape[0], self.patch_size, self.random_crop)
            col = _crop_origin(cls.shape[1], self.patch_size, self.random_crop)
            window = np.s_[row:row + self.patch_size, col:col + self.patch_size]
            image = np.asarray(rgb[window[0], window[1], :3], dtype=np.float32).transpose(2, 0, 1)
            classes = np.asarray(cls[window], dtype=np.float32)
            image_nodata = _declared_nodata(rgb, image_handle)
        image_valid = np.isfinite(image).all(axis=0)
        for nodata in image_nodata:
            image_valid &= ~np.all(np.isclose(image, nodata, equal_nan=True), axis=0)
        # Preserve original raw radiometry; fail instead of silently clipping
        # an incompatible sensor's valid data into the expected byte range.
        if np.any((image[:, image_valid] < 0) | (image[:, image_valid] > self.rgb_scale)):
            raise ValueError(f"Valid RGB values outside [0,rgb_scale]: {record.sample_id}")
        rounded = np.rint(classes)
        class_available = np.isfinite(classes)
        legal = np.isclose(classes, rounded, rtol=0, atol=1e-5) & (rounded >= 0) & (rounded <= 6)
        if np.any(class_available & ~legal):
            unexpected = np.unique(classes[class_available & ~legal])[:8]
            raise ValueError(f"Unexpected source class IDs in {record.sample_id}: {unexpected}")
        class_valid = class_available & legal & (rounded != 0)
        labels = np.full(classes.shape, IGNORE_INDEX, dtype=np.int64)
        for source_id, target in GAMUS_SOURCE_ID_TO_SIX_CLASS.items():
            labels[class_valid & (rounded == source_id)] = target
        image[:, ~image_valid] = 0
        image /= self.rgb_scale
        arrays = {"image": image, "labels": labels,
                  "image_valid_mask": image_valid, "classification_valid_mask": class_valid,
                  "dark_pixel_proxy_mask": image_valid & (image.mean(axis=0) <= self.dark_pixel_threshold)}
        if self.augment:
            turns = int(torch.randint(0, 4, (1,)).item())
            flip = bool(torch.randint(0, 2, (1,)).item())
            for name, value in arrays.items():
                if turns:
                    value = np.rot90(value, turns, axes=(-2, -1))
                if flip:
                    value = np.flip(value, axis=-1)
                arrays[name] = value
        return {**{key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in arrays.items()},
                "sample_id": record.sample_id, "city": record.city, "region": record.region}


def make_datasets(config: Mapping[str, Any]) -> tuple[GamusRgbSegmentationDataset, GamusRgbSegmentationDataset, dict[str, Any]]:
    data = config["data"]
    if data.get("train_radiometric_policy", "raw") != "raw" or data.get("validation_radiometric_policy", "raw") != "raw":
        raise ValueError("This versioned pilot requires raw RGB radiometry")
    kwargs = {"approved_index_path": data["approved_index_path"],
              "approved_index_sha256": data["approved_index_file_sha256"],
              "rgb_scale": float(data.get("rgb_scale", 255)),
              "dark_pixel_threshold": float(data.get("dark_pixel_threshold", 0.15)),
              "native_shape": 1024}
    train = GamusRgbSegmentationDataset(data["root"], "train", patch_size=int(data.get("patch_size", 384)),
                                       random_crop=True, augment=True, **kwargs)
    val = GamusRgbSegmentationDataset(data["root"], "val", patch_size=int(data.get("validation_patch_size", 1024)), **kwargs)
    contract = {"schema": "msr.gamus_rgb_only_inputs.v1", "rgb_scale": kwargs["rgb_scale"],
                "dark_pixel_threshold": kwargs["dark_pixel_threshold"],
                "approved_index_sha256": kwargs["approved_index_sha256"], "class_names": list(GAMUS_SIX_CLASS_NAMES),
                "train_count": len(train), "validation_count": len(val),
                "train_ids_sha256": canonical_sha256(list(train.sample_ids)),
                "validation_ids_sha256": canonical_sha256(list(val.sample_ids)),
                "height_or_prior_read": False, "official_test_or_nyc_read": False,
                "validation_native_size": 1024, "radiometry": "raw RGB/rgb_scale, invalid RGB zeroed",
                "validity": "image_valid AND classification_valid; independent of height",
                "source_metadata": {"train": source_metadata_snapshot(train), "val": source_metadata_snapshot(val)}}
    contract["contract_sha256"] = canonical_sha256(contract)
    return train, val, contract


def source_metadata_snapshot(dataset: GamusRgbSegmentationDataset) -> dict[str, Any]:
    rows = []
    for record in sorted(dataset.records, key=lambda r: r.sample_id):
        for role, path in (("image", record.image_path), ("class", record.class_path)):
            stat = path.stat()
            rows.append({"sample_id": record.sample_id, "role": role, "path": str(path),
                         "size_bytes": stat.st_size, "modified_time_ns": stat.st_mtime_ns})
    return {"file_count": len(rows), "metadata_manifest_sha256": canonical_sha256(rows)}


def build_train_sampler(dataset: GamusRgbSegmentationDataset, config: Mapping[str, Any], generator: torch.Generator) -> WeightedRandomSampler:
    """Exact V3 city-mass-preserving 1+2*water-hit weighting (384 crops)."""
    if dataset.split != "train" or dataset.patch_size != 384:
        raise ValueError("Sealed water-hit sampler is defined only for 384px training crops")
    data = config["data"]
    path = Path(data["fine_class_sampling_index_path"]).resolve()
    if file_sha256(path) != data["fine_class_sampling_index_sha256"]:
        raise ValueError("Sampling index SHA-256 mismatch")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = row.get("sample_id")
        if row.get("schema") != "msr.gamus.train_six_class_tile_index.v1" or sample_id in rows:
            raise ValueError("Sampling index schema or duplicate ID invalid")
        rows[sample_id] = row
    if set(rows) != set(dataset.sample_ids):
        raise ValueError("Sampling index must match exact approved training IDs")
    boost = float(data.get("water_sampling_boost", 2.0))
    if not math.isfinite(boost) or boost < 0 or data.get("preserve_city_sampling_mass", True) is not True:
        raise ValueError("Finite nonnegative boost and city-mass preservation required")
    weights = []
    city_indices = defaultdict(list)
    for i, record in enumerate(dataset.records):
        row = rows[record.sample_id]
        if row.get("city") != record.city:
            raise ValueError("Sampling city identity mismatch")
        p = float(row["uniform_random_crop_384"]["water_hit_probability"])
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid water hit probability")
        weights.append(1 + boost * p)
        city_indices[record.city].append(i)
    for indices in city_indices.values():
        mean = sum(weights[i] for i in indices) / len(indices)
        for i in indices:
            weights[i] /= mean
    count = int(config.get("training", {}).get("samples_per_epoch", len(dataset)))
    if count <= 0:
        raise ValueError("samples_per_epoch must be positive")
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), count, replacement=True, generator=generator)


def authenticate_validation_content(dataset: GamusRgbSegmentationDataset,
                                    baseline_report_path: str | Path, expected_sha256: str,
                                    cache_path: str | Path | None = None) -> dict[str, Any]:
    """Match RGB/CLS bytes to the independent V3 replay, without AGL/prior reads.

    Cache reuse requires exact source path/size/mtime and all cached identities
    must still aggregate to the authenticated independent replay's SHA values.
    This is a local accidental-drift guard, not protection from an adversary
    who can rewrite source files while restoring their timestamps.
    """
    if dataset.split != "val" or dataset.random_crop or dataset.augment:
        raise ValueError("Content binding requires deterministic validation dataset")
    baseline = authenticated_json(baseline_report_path, expected_sha256)
    if baseline.get("schema") != "msr.gamus_dcphl_paired_independent_replay.v1" or baseline.get("passes") is not True:
        raise ValueError("A successful independent V3 baseline replay is required")
    if baseline["approved_index"]["sha256"] != dataset.approved_index_sha256:
        raise ValueError("Candidate and baseline approved indices differ")
    if baseline["validation_contract"]["validation_ids_sha256"] != canonical_sha256(list(dataset.sample_ids)):
        raise ValueError("Candidate and baseline validation order differs")
    baseline_contract = baseline["validation_contract"]
    if float(baseline_contract.get("rgb_scale", dataset.rgb_scale)) != dataset.rgb_scale:
        raise ValueError("Candidate and baseline RGB scale differs")
    if int(baseline_contract.get("validation_patch_size", dataset.native_shape)) != dataset.native_shape:
        raise ValueError("Candidate and baseline native resolution differs")
    if dataset.dark_pixel_threshold != 0.15:
        raise ValueError("Independent baseline uses a fixed 0.15 mean-RGB dark-pixel threshold")
    expected_roles = baseline["source_content_binding"]["per_role"]
    cache = {}
    cache_file = Path(cache_path).resolve() if cache_path is not None else None
    if cache_file is not None and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("baseline_sha256") == expected_sha256:
                cache = {row["path"]: row for row in cached.get("files", [])}
        except (ValueError, KeyError, TypeError):
            cache = {}
    metadata_before = source_metadata_snapshot(dataset)
    source_paths = {path.resolve() for record in dataset.records for path in (record.image_path, record.class_path)}
    if cache_file in source_paths or cache_file in {Path(baseline_report_path).resolve(), dataset.approved_index_path}:
        raise ValueError("Hash cache cannot overwrite source or sealed metadata")
    files, role_results = [], {}
    hashed_files = 0
    for role in ("class", "image"):
        rows = []
        for record in sorted(dataset.records, key=lambda r: r.sample_id):
            path = record.image_path if role == "image" else record.class_path
            stat = path.stat()
            identity = {"path": str(path), "size_bytes": stat.st_size, "modified_time_ns": stat.st_mtime_ns}
            old = cache.get(str(path), {})
            if all(old.get(k) == v for k, v in identity.items()) and re.fullmatch(r"[0-9a-f]{64}", str(old.get("sha256", ""))):
                digest = old["sha256"]
            else:
                digest = file_sha256(path)
                hashed_files += 1
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"Source changed during hashing: {path}")
            files.append({**identity, "sha256": digest})
            rows.append({"role": role, "sample_id": record.sample_id, "size_bytes": stat.st_size, "sha256": digest})
        actual = {"file_count": len(rows), "total_bytes": sum(row["size_bytes"] for row in rows),
                  "content_manifest_sha256": canonical_sha256(rows)}
        if actual != expected_roles[role]:
            raise ValueError(f"Candidate {role} content differs from the fixed independent baseline")
        role_results[role] = actual
    if source_metadata_snapshot(dataset) != metadata_before:
        raise RuntimeError("Validation source metadata changed during binding")
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
        temporary.write_text(json.dumps({"baseline_sha256": expected_sha256, "files": files}, sort_keys=True), encoding="utf-8")
        temporary.replace(cache_file)
    return {"baseline_sha256": expected_sha256, "validation_ids_sha256": canonical_sha256(list(dataset.sample_ids)),
            "per_role": role_results, "metadata": metadata_before, "newly_hashed_file_count": hashed_files,
            "rgb_and_class_content_matches_baseline": True, "height_and_prior_files_read": False}
