"""Create or verify the fail-closed GAMUS V4 runtime/data provenance seal.

The seal deliberately discovers records from the approved DC + Philadelphia
index instead of globbing GAMUS directories.  Consequently it never opens,
hashes, or even enumerates NYC or official-test inputs.  Each admitted train
and validation sample is bound to its RGB, class, AGL-height, and cached DAV2
prior bytes with SHA-256.

The contract also records the exact behavior-critical source files, control
artifacts, and stable Python/PyTorch/NumPy/CUDA/GPU runtime identity.  ``create``
publishes one JSON file atomically; ``verify`` reconstructs the inventory and
fails on any drift.  ``estimate`` performs metadata-only discovery to estimate
the one-time hashing cost.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import math
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "msr.gamus_v4_runtime_data_seal.v1"
EXPECTED_PROTOCOL_SCHEMA = "msr.gamus_six_class_hierarchical_dcphl.v4"
EXPECTED_INDEPENDENT_REPLAY_SCHEMA = (
    "msr.gamus_dcphl_paired_independent_replay.v1"
)
APPROVED_INDEX_SCHEMA = "msr.gamus.approved_samples.v1"
SAMPLING_INDEX_SCHEMA = "msr.gamus.train_six_class_tile_index.v1"
ALLOWED_CITIES = ("DC", "PHL")
ALLOWED_SPLITS = ("train", "val")
DATA_ROLES = ("image", "class", "height", "relative_prior")
SAMPLE_ID_PATTERN = re.compile(r"^(DC|PHL)_[A-Za-z0-9_]+$")
EXPECTED_CITY_SPLIT_COUNTS = {
    "train": {"DC": 1439, "PHL": 2398},
    "val": {"DC": 359, "PHL": 500},
}
EXPECTED_PRIOR_MODEL = r"models\foundation\depth-anything-v2-small-hf"

# Imported and/or executed by the V4 train + evaluation path.  This explicit
# allow-list is intentional: changes are reviewed instead of silently admitted
# by a dynamic import crawler.  The launcher may pin this script's digest too.
BEHAVIOR_SOURCE_PATHS = (
    "scripts/run_gamus_hierarchical_v4.ps1",
    "scripts/build_gamus_hierarchical_v4_config.py",
    "scripts/preflight_gamus_hierarchical_v4.py",
    "scripts/train_multidomain.py",
    "scripts/audit_gamus_hierarchical_v4_checkpoint.py",
    "scripts/audit_height_output_identity.py",
    "scripts/evaluate_gamus_dcphl_paired_replay.py",
    "scripts/seal_gamus_v4_runtime_data.py",
    "src/msr/__init__.py",
    "src/msr/data/__init__.py",
    "src/msr/data/gamus_dataset.py",
    "src/msr/data/highbuild.py",
    "src/msr/data/manifest.py",
    "src/msr/data/mixed_replay.py",
    "src/msr/data/open_canopy.py",
    "src/msr/data/radiometry.py",
    "src/msr/data/raster_dataset.py",
    "src/msr/data/surface_dataset.py",
    "src/msr/evaluation/__init__.py",
    "src/msr/evaluation/classification_metrics.py",
    "src/msr/evaluation/domain_metrics.py",
    "src/msr/evaluation/metrics.py",
    "src/msr/evaluation/routed_artifact.py",
    "src/msr/evaluation/routed_validation.py",
    "src/msr/inference/__init__.py",
    "src/msr/inference/predict.py",
    "src/msr/inference/relative_depth.py",
    "src/msr/inference/tiling.py",
    "src/msr/models/__init__.py",
    "src/msr/models/domain_surface_net.py",
    "src/msr/models/height_net.py",
    "src/msr/models/routed_surface.py",
    "src/msr/models/stage3_endpoints.py",
    "src/msr/training/__init__.py",
    "src/msr/training/losses.py",
    "src/msr/training/scene_router.py",
    "src/msr/training/scene_router_record_store.py",
)


class SealError(RuntimeError):
    """Raised when the provenance contract cannot be created or verified."""


def _mapping(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SealError(f"{role} must be a mapping")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SealError(f"unable to load V4 config {path}: {error}") from error
    if not isinstance(value, dict):
        raise SealError(f"V4 config root must be a mapping: {path}")
    return value


def _load_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SealError(f"unable to load {role} {path}: {error}") from error
    if not isinstance(value, dict):
        raise SealError(f"{role} root must be a mapping: {path}")
    return value


def _assert_regular_file(path: Path, role: str) -> None:
    if not path.is_file():
        raise SealError(f"missing {role}: {path}")


def _assert_expected_hash(path: Path, expected: object, role: str) -> str:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise SealError(f"{role} has no valid pinned SHA-256")
    actual = file_sha256(path)
    if actual != expected.lower():
        raise SealError(
            f"{role} SHA-256 mismatch: expected {expected.lower()}, found {actual}"
        )
    return actual


def _relative_posix(path: Path, root: Path, role: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise SealError(f"{role} escaped its approved root: {path}") from error


def _validate_v4_config(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    protocol = _mapping(config.get("protocol"), "config.protocol")
    data = _mapping(config.get("data"), "config.data")
    model = _mapping(config.get("model"), "config.model")
    if protocol.get("schema") != EXPECTED_PROTOCOL_SCHEMA:
        raise SealError(
            "the config is not the sealed hierarchical DC+PHL V4 protocol"
        )
    for field in ("learning_cities", "development_validation_cities"):
        if tuple(protocol.get(field, ())) != ALLOWED_CITIES:
            raise SealError(f"protocol.{field} must be exactly {list(ALLOWED_CITIES)}")
    if protocol.get("locked_classifier_holdout_city") != "NYC":
        raise SealError("protocol must keep NYC as the locked classifier holdout")
    if protocol.get("official_test_policy") != (
        "previously_consumed_forbidden_for_reuse"
    ):
        raise SealError("official GAMUS test must remain forbidden for reuse")
    for field in (
        "fixed_v3_independent_replay",
        "fixed_v3_independent_replay_sha256",
    ):
        if not protocol.get(field):
            raise SealError(f"protocol.{field} is required by the V4 seal")
    if not bool(data.get("require_approved_index")):
        raise SealError("V4 must require its approved DC+PHL index")
    if not bool(data.get("require_relative_priors")):
        raise SealError("V4 must require every cached relative prior")
    if not bool(data.get("require_complete_official_splits")):
        raise SealError("V4 must retain the complete-development-split guard")
    if not data.get("fine_class_sampling_index_path"):
        raise SealError("V4 must pin its train sampling index")
    expected_data_values = {
        "patch_size": 384,
        "validation_patch_size": 1024,
        "water_sampling_boost": 2.0,
        "preserve_city_sampling_mass": True,
        "rgb_scale": 255.0,
        "train_radiometric_policy": "raw",
        "validation_radiometric_policy": "raw",
        "height_max_m": 200.0,
    }
    for field, expected in expected_data_values.items():
        if data.get(field) != expected:
            raise SealError(f"data.{field} must remain {expected!r}")
    if not model.get("initial_checkpoint") or not model.get("base_checkpoint"):
        raise SealError("V4 must declare its initial and base checkpoints")
    return protocol, data, model


def _load_approved_ids(index_path: Path) -> dict[str, list[str]]:
    payload = _load_json(index_path, "approved index")
    if payload.get("schema") != APPROVED_INDEX_SCHEMA:
        raise SealError(f"unsupported approved-index schema: {payload.get('schema')!r}")
    splits = _mapping(payload.get("splits"), "approved index splits")
    if set(splits) != set(ALLOWED_SPLITS):
        raise SealError(
            "approved index must contain only train and val; test/holdout input is forbidden"
        )
    result: dict[str, list[str]] = {}
    seen: set[str] = set()
    for split in ALLOWED_SPLITS:
        split_payload = _mapping(splits.get(split), f"approved index {split}")
        raw_ids = split_payload.get("approved_sample_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise SealError(f"approved index {split} sample IDs must be a non-empty list")
        if not all(isinstance(item, str) for item in raw_ids):
            raise SealError(f"approved index {split} contains a non-string sample ID")
        ids = sorted(raw_ids)
        if len(ids) != len(set(ids)):
            raise SealError(f"approved index {split} contains duplicate sample IDs")
        declared = split_payload.get("approved_count")
        if declared is None or int(declared) != len(ids):
            raise SealError(f"approved index {split} count does not match its IDs")
        for sample_id in ids:
            match = SAMPLE_ID_PATTERN.fullmatch(sample_id)
            if match is None or match.group(1) not in ALLOWED_CITIES:
                raise SealError(
                    f"forbidden/non-DC+PHL sample in approved {split}: {sample_id!r}"
                )
            if sample_id in seen:
                raise SealError(f"sample appears in both train and val: {sample_id}")
            seen.add(sample_id)
        city_counts = {
            city: sum(item.startswith(f"{city}_") for item in ids)
            for city in ALLOWED_CITIES
        }
        if city_counts != EXPECTED_CITY_SPLIT_COUNTS[split]:
            raise SealError(
                f"approved {split} city counts changed: expected "
                f"{EXPECTED_CITY_SPLIT_COUNTS[split]}, found {city_counts}"
            )
        result[split] = ids
    return result


def _validate_sampling_index(path: Path, train_ids: Sequence[str], class_root: Path) -> None:
    observed: dict[str, str] = {}
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as error:
        raise SealError(f"unable to open train sampling index {path}: {error}") from error
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise SealError(f"blank sampling-index row at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SealError(
                    f"invalid sampling-index JSON at line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict) or row.get("schema") != SAMPLING_INDEX_SCHEMA:
                raise SealError(f"invalid sampling-index schema at line {line_number}")
            sample_id = row.get("sample_id")
            city = row.get("city")
            if not isinstance(sample_id, str) or SAMPLE_ID_PATTERN.fullmatch(sample_id) is None:
                raise SealError(f"invalid sampling sample ID at line {line_number}")
            expected_city = sample_id.split("_", maxsplit=1)[0]
            if city != expected_city or city not in ALLOWED_CITIES:
                raise SealError(f"forbidden/mismatched city at sampling line {line_number}")
            if sample_id in observed:
                raise SealError(f"duplicate sampling record: {sample_id}")
            expected_class = (class_root / "train" / f"{sample_id}_CLS.h5").resolve()
            actual_class = _resolve(str(row.get("class_path", "")))
            if actual_class != expected_class:
                raise SealError(
                    f"sampling class path does not match approved input for {sample_id}"
                )
            crop = row.get("uniform_random_crop_384")
            if not isinstance(crop, Mapping):
                raise SealError(
                    f"sampling row has no 384-pixel crop statistics at line {line_number}"
                )
            try:
                water_probability = float(crop["water_hit_probability"])
            except (KeyError, TypeError, ValueError) as error:
                raise SealError(
                    f"sampling row has invalid water probability at line {line_number}"
                ) from error
            if not math.isfinite(water_probability) or not 0.0 <= water_probability <= 1.0:
                raise SealError(
                    f"sampling water probability is outside [0,1] at line {line_number}"
                )
            observed[sample_id] = str(actual_class)
    if set(observed) != set(train_ids):
        missing = sorted(set(train_ids) - set(observed))[:5]
        extra = sorted(set(observed) - set(train_ids))[:5]
        raise SealError(
            f"sampling index is not an exact train-ID match; missing={missing}, extra={extra}"
        )


def _image_path(image_split_root: Path, sample_id: str) -> Path:
    candidates = (
        (image_split_root / f"{sample_id}_RGB.h5").resolve(),
        (image_split_root / f"{sample_id}_IMG.h5").resolve(),
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise SealError(
            f"expected exactly one approved RGB/IMG input for {sample_id}, found {len(existing)}"
        )
    return existing[0]


def discover_data_entries(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover exact inputs from approved IDs, never from dataset globbing."""

    _protocol, data, _model = _validate_v4_config(config)
    dataset_root = _resolve(str(data.get("root", "")))
    prior_root = _resolve(str(data.get("relative_prior_root", "")))
    approved_index = _resolve(str(data.get("approved_index_path", "")))
    sampling_index = _resolve(str(data.get("fine_class_sampling_index_path", "")))
    for path, role in (
        (approved_index, "approved index"),
        (sampling_index, "train sampling index"),
    ):
        _assert_regular_file(path, role)
    _assert_expected_hash(
        approved_index, data.get("approved_index_file_sha256"), "approved index"
    )
    _assert_expected_hash(
        sampling_index,
        data.get("fine_class_sampling_index_sha256"),
        "train sampling index",
    )
    approved = _load_approved_ids(approved_index)
    class_root = (dataset_root / "classes").resolve()
    _validate_sampling_index(sampling_index, approved["train"], class_root)

    roots = {
        "image": (dataset_root / "images").resolve(),
        "class": class_root,
        "height": (dataset_root / "heights").resolve(),
        "relative_prior": prior_root,
    }
    entries: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        for sample_id in approved[split]:
            city = sample_id.split("_", maxsplit=1)[0]
            paths = {
                "image": _image_path(roots["image"] / split, sample_id),
                "class": (roots["class"] / split / f"{sample_id}_CLS.h5").resolve(),
                "height": (roots["height"] / split / f"{sample_id}_AGL.h5").resolve(),
                "relative_prior": (
                    roots["relative_prior"] / split / f"{sample_id}_REL.h5"
                ).resolve(),
            }
            for role in DATA_ROLES:
                path = paths[role]
                _assert_regular_file(path, f"{split} {sample_id} {role}")
                entries.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "city": city,
                        "role": role,
                        "path": str(path),
                        "relative_path": _relative_posix(path, roots[role], role),
                        "size_bytes": int(path.stat().st_size),
                    }
                )
    entries.sort(
        key=lambda item: (
            ALLOWED_SPLITS.index(str(item["split"])),
            str(item["sample_id"]),
            DATA_ROLES.index(str(item["role"])),
        )
    )
    expected_count = 4 * sum(len(values) for values in approved.values())
    if len(entries) != expected_count:
        raise SealError("internal error: incomplete four-role data inventory")
    metadata = {
        "dataset_root": str(dataset_root),
        "relative_prior_root": str(prior_root),
        "allowed_cities": list(ALLOWED_CITIES),
        "allowed_splits": list(ALLOWED_SPLITS),
        "discovery_policy": (
            "approved_ids_only_no_directory_glob_no_nyc_no_official_test"
        ),
        "split_sample_counts": {
            split: len(approved[split]) for split in ALLOWED_SPLITS
        },
        "sample_count": sum(len(values) for values in approved.values()),
        "file_count": len(entries),
        "total_bytes": sum(int(item["size_bytes"]) for item in entries),
    }
    return entries, metadata


def _hdf_attr_text(value: object, role: str) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SealError(f"{role} is not valid UTF-8") from error
    if isinstance(value, str):
        return value
    raise SealError(f"{role} must be text")


def _validate_hdf5_sample(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(entries) != len(DATA_ROLES):
        raise SealError("HDF5 validation received an incomplete sample")
    by_role = {str(item["role"]): item for item in entries}
    if set(by_role) != set(DATA_ROLES):
        raise SealError("HDF5 validation received duplicate or unknown roles")
    sample_id = str(entries[0]["sample_id"])
    expected = {
        "image": ((1024, 1024, 3), {"uint8"}),
        "class": ((1024, 1024), {"uint8", "float32"}),
        "height": ((1024, 1024), {"float32"}),
        "relative_prior": ((1024, 1024), {"float16"}),
    }
    dtypes: dict[str, str] = {}
    prior_metadata: dict[str, str] = {}
    for role in DATA_ROLES:
        path = Path(str(by_role[role]["path"]))
        try:
            with h5py.File(path, "r") as handle:
                if "image" not in handle or not isinstance(handle["image"], h5py.Dataset):
                    raise SealError(f"{role} HDF5 has no /image dataset: {path}")
                dataset = handle["image"]
                shape = tuple(int(value) for value in dataset.shape)
                dtype = str(dataset.dtype)
                expected_shape, allowed_dtypes = expected[role]
                if shape != expected_shape:
                    raise SealError(
                        f"{role} grid mismatch for {sample_id}: {shape} != {expected_shape}"
                    )
                if dtype not in allowed_dtypes:
                    raise SealError(
                        f"{role} dtype mismatch for {sample_id}: {dtype} not in "
                        f"{sorted(allowed_dtypes)}"
                    )
                dtypes[role] = dtype
                if role == "relative_prior":
                    for attribute in ("units", "model", "source_rgb"):
                        if attribute not in dataset.attrs:
                            raise SealError(
                                f"relative prior lacks {attribute!r} for {sample_id}"
                            )
                        prior_metadata[attribute] = _hdf_attr_text(
                            dataset.attrs[attribute],
                            f"relative prior {attribute}",
                        )
        except OSError as error:
            raise SealError(f"unable to open approved HDF5 input {path}: {error}") from error
    if prior_metadata["units"] != "relative_0_1":
        raise SealError(f"relative prior units are not relative_0_1 for {sample_id}")
    if prior_metadata["model"] != EXPECTED_PRIOR_MODEL:
        raise SealError(f"relative prior model identity changed for {sample_id}")
    recorded_source = _resolve(prior_metadata["source_rgb"])
    expected_source = Path(str(by_role["image"]["path"])).resolve()
    if recorded_source != expected_source:
        raise SealError(f"relative prior source RGB mismatch for {sample_id}")
    return {
        "dtypes": dtypes,
        "prior_model": prior_metadata["model"],
        "prior_units": prior_metadata["units"],
    }


def validate_hdf5_entries(
    entries: Sequence[Mapping[str, Any]], *, jobs: int
) -> dict[str, Any]:
    """Validate DC+PHL HDF5 structure without reading raster pixel arrays."""

    if jobs <= 0:
        raise SealError("HDF5 validation worker count must be positive")
    grouped: list[list[Mapping[str, Any]]] = []
    current_key: tuple[str, str] | None = None
    current: list[Mapping[str, Any]] = []
    for entry in entries:
        key = (str(entry["split"]), str(entry["sample_id"]))
        if current_key is not None and key != current_key:
            grouped.append(current)
            current = []
        current_key = key
        current.append(entry)
    if current:
        grouped.append(current)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="v4-hdf") as executor:
        results.extend(executor.map(_validate_hdf5_sample, grouped))
    dtype_counts: dict[str, dict[str, int]] = {role: {} for role in DATA_ROLES}
    for result in results:
        for role, dtype in _mapping(result["dtypes"], "HDF5 dtypes").items():
            role_counts = dtype_counts[str(role)]
            role_counts[str(dtype)] = role_counts.get(str(dtype), 0) + 1
    models = sorted({str(result["prior_model"]) for result in results})
    units = sorted({str(result["prior_units"]) for result in results})
    if models != [EXPECTED_PRIOR_MODEL] or units != ["relative_0_1"]:
        raise SealError("approved relative priors do not share one source contract")
    return {
        "passes": True,
        "dataset_key": "/image",
        "grid_height": 1024,
        "grid_width": 1024,
        "samples_checked": len(grouped),
        "dtype_counts": dtype_counts,
        "relative_prior_model": EXPECTED_PRIOR_MODEL,
        "relative_prior_units": "relative_0_1",
        "relative_prior_source_rgb_checked_per_sample": True,
        "pixel_arrays_read_for_metadata_check": False,
    }


def _hash_one_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(entry["path"]))
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise SealError(f"input changed while being hashed: {path}")
    if int(entry["size_bytes"]) != after.st_size:
        raise SealError(f"input size changed before hashing: {path}")
    return {**dict(entry), "sha256": digest}


def hash_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    jobs: int,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    if jobs <= 0:
        raise SealError("hash worker count must be positive")
    result: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="v4-seal") as executor:
        for index, value in enumerate(executor.map(_hash_one_entry, entries), start=1):
            result.append(value)
            if progress is not None:
                progress(index, len(entries))
    return result


def _file_entry(path: Path, roles: Iterable[str]) -> dict[str, Any]:
    _assert_regular_file(path, "/".join(roles))
    return {
        "path": str(path),
        "roles": sorted(set(roles)),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _validate_fixed_v3_independent_replay(path: Path) -> None:
    """Authenticate the pinned V3 replay and the evaluator that produced it."""

    report = _load_json(path, "fixed V3 independent replay")
    if report.get("schema") != EXPECTED_INDEPENDENT_REPLAY_SCHEMA:
        raise SealError("fixed V3 independent replay has an unsupported schema")
    if report.get("mode") != "baseline_v3_only" or report.get("passes") is not True:
        raise SealError("fixed V3 independent replay is not a passing baseline-only run")

    access = _mapping(
        report.get("access_policy"), "fixed V3 independent replay access policy"
    )
    if tuple(access.get("cities_opened", ())) != ALLOWED_CITIES:
        raise SealError("fixed V3 independent replay must open exactly DC and PHL")
    for field in (
        "nyc_opened",
        "official_test_opened_or_reused",
        "promotion_performed",
    ):
        if access.get(field) is not False:
            raise SealError(
                f"fixed V3 independent replay access_policy.{field} must be false"
            )
    if access.get("split_constructed") != "val":
        raise SealError("fixed V3 independent replay must use only the validation split")

    implementation = _mapping(
        report.get("implementation_sha256"),
        "fixed V3 independent replay implementation hashes",
    )
    if implementation.get("unchanged_during_replay") is not True:
        raise SealError("replay implementation was not stable during evaluation")
    evaluator_path = _resolve("scripts/evaluate_gamus_dcphl_paired_replay.py")
    _assert_regular_file(evaluator_path, "independent replay evaluator")
    evaluator_sha256 = file_sha256(evaluator_path)
    for phase in ("before", "after"):
        recorded = _mapping(
            implementation.get(phase),
            f"fixed V3 independent replay implementation {phase}",
        ).get("evaluator")
        if recorded != evaluator_sha256:
            raise SealError(
                "fixed V3 independent replay evaluator SHA-256 does not match "
                f"the sealed evaluator ({phase})"
            )


def _control_artifacts(
    config_path: Path,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    protocol, data, model = _validate_v4_config(config)
    declarations: list[tuple[str, Path]] = [
        ("v4_config", config_path),
        ("approved_dc_phl_index", _resolve(str(data["approved_index_path"]))),
        ("train_sampling_index", _resolve(str(data["fine_class_sampling_index_path"]))),
        ("initial_protected_checkpoint", _resolve(str(model["initial_checkpoint"]))),
        ("base_checkpoint", _resolve(str(model["base_checkpoint"]))),
        ("source_comparator_config", _resolve(str(protocol["source_comparator_config"]))),
        ("fixed_v3_baseline_audit", _resolve(str(protocol["fixed_v3_baseline_audit"]))),
        (
            "fixed_v3_independent_replay",
            _resolve(str(protocol["fixed_v3_independent_replay"])),
        ),
        ("live_production_pointer", _resolve(str(protocol["live_pointer_file"]))),
    ]
    prior_provenance = (
        _resolve(str(data["relative_prior_root"]))
        / "msr_prior_provenance.json"
    ).resolve()
    declarations.append(("dav2_prior_provenance", prior_provenance))

    expected = {
        "approved_dc_phl_index": data.get("approved_index_file_sha256"),
        "train_sampling_index": data.get("fine_class_sampling_index_sha256"),
        "initial_protected_checkpoint": protocol.get("protected_checkpoint_sha256"),
        "source_comparator_config": protocol.get("source_comparator_config_sha256"),
        "fixed_v3_baseline_audit": protocol.get("fixed_v3_baseline_audit_sha256"),
        "fixed_v3_independent_replay": protocol.get(
            "fixed_v3_independent_replay_sha256"
        ),
        "live_production_pointer": protocol.get("live_pointer_file_sha256"),
    }
    grouped: dict[Path, list[str]] = {}
    for role, path in declarations:
        _assert_regular_file(path, role)
        if role in expected:
            _assert_expected_hash(path, expected[role], role)
        grouped.setdefault(path, []).append(role)
    replay_path = _resolve(str(protocol["fixed_v3_independent_replay"]))
    _validate_fixed_v3_independent_replay(replay_path)
    return [
        _file_entry(path, grouped[path])
        for path in sorted(grouped, key=lambda item: str(item).casefold())
    ]


def _behavior_sources(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    resolved: list[Path] = []
    for value in paths:
        path = _resolve(value)
        _relative_posix(path, PROJECT_ROOT, "behavior source")
        resolved.append(path)
    if len(resolved) != len(set(resolved)):
        raise SealError("behavior-source allow-list contains duplicates")
    result = []
    for path in sorted(resolved, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix()):
        entry = _file_entry(path, ("behavior_source",))
        entry["project_relative_path"] = path.relative_to(PROJECT_ROOT).as_posix()
        result.append(entry)
    return result


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as error:
        raise SealError(f"required runtime package is missing: {name}") from error


def _nvidia_smi_path() -> Path | None:
    candidates: list[Path] = []
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidates.append(Path(system_root) / "System32" / "nvidia-smi.exe")
    candidates.extend(
        [
            Path(r"C:\Windows\System32\nvidia-smi.exe"),
            Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"),
        ]
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _nvidia_smi_inventory() -> list[dict[str, Any]]:
    executable = _nvidia_smi_path()
    if executable is None:
        raise SealError("nvidia-smi is unavailable; GPU driver identity cannot be sealed")
    command = [
        str(executable),
        "--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id",
        "--format=csv,noheader,nounits",
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
            creationflags=flags,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SealError(f"unable to query NVIDIA driver identity: {error}") from error
    rows = []
    for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            raise SealError(f"unexpected nvidia-smi identity row: {row!r}")
        rows.append(
            {
                "index": int(row[0]),
                "name": row[1].strip(),
                "uuid": row[2].strip(),
                "driver_version": row[3].strip(),
                "memory_total_mib": int(row[4]),
                "pci_bus_id": row[5].strip(),
            }
        )
    if not rows:
        raise SealError("nvidia-smi reported no GPUs")
    return sorted(rows, key=lambda item: int(item["index"]))


def runtime_inventory(*, require_cuda: bool = True) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise SealError("CUDA is unavailable; refusing to seal a V4 GPU run")
    torch_devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            torch_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "uuid": str(getattr(properties, "uuid", "")),
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": int(properties.total_memory),
                    "multiprocessor_count": int(properties.multi_processor_count),
                }
            )
    nvidia = _nvidia_smi_inventory() if cuda_available else []
    if cuda_available and len(nvidia) != len(torch_devices):
        raise SealError("PyTorch and nvidia-smi disagree on GPU count")
    for torch_device, driver_device in zip(torch_devices, nvidia):
        torch_uuid = str(torch_device["uuid"]).lower().removeprefix("gpu-")
        driver_uuid = str(driver_device["uuid"]).lower().removeprefix("gpu-")
        if (
            int(torch_device["index"]) != int(driver_device["index"])
            or str(torch_device["name"]).strip() != str(driver_device["name"]).strip()
            or (torch_uuid and driver_uuid and torch_uuid != driver_uuid)
        ):
            raise SealError("PyTorch and nvidia-smi disagree on GPU identity/order")
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "torch", "torchvision", "h5py", "PyYAML", "timm")
        },
        "numpy_version": np.__version__,
        "pytorch": {
            "version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": cuda_available,
            "devices": torch_devices,
        },
        "nvidia_driver": nvidia,
        "behavior_environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTHONHASHSEED",
            )
        },
    }


def _data_aggregate(entries: Sequence[Mapping[str, Any]]) -> str:
    compact = [
        {
            key: item[key]
            for key in (
                "split",
                "sample_id",
                "city",
                "role",
                "relative_path",
                "size_bytes",
                "sha256",
            )
        }
        for item in entries
    ]
    return _sha256_bytes(_canonical_json(compact))


def _artifact_aggregate(entries: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json(list(entries)))


def _progress_printer(every: int, label: str) -> Callable[[int, int], None]:
    def report(done: int, total: int) -> None:
        if done == total or (every > 0 and done % every == 0):
            print(f"{label}: {done:,}/{total:,} files", file=sys.stderr, flush=True)

    return report


def build_contract(
    config_path: Path,
    *,
    jobs: int = 4,
    behavior_source_paths: Sequence[str | Path] = BEHAVIOR_SOURCE_PATHS,
    runtime: Mapping[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    _assert_regular_file(config_path, "V4 config")
    config = _load_yaml(config_path)
    # Validate small controls/runtime first so a stale pointer, environment, or
    # recipe fails before the expensive ~47 GiB data pass begins.
    controls = _control_artifacts(config_path, config)
    sources = _behavior_sources(behavior_source_paths)
    runtime_value = dict(runtime) if runtime is not None else runtime_inventory()
    data_entries, data_metadata = discover_data_entries(config)
    hdf5_contract = validate_hdf5_entries(data_entries, jobs=jobs)
    hashed_data = hash_entries(data_entries, jobs=jobs, progress=progress)
    dataset = {
        **data_metadata,
        "hdf5_contract": hdf5_contract,
        "manifest_sha256": _data_aggregate(hashed_data),
        "files": hashed_data,
    }
    return {
        "policy": {
            "cities": list(ALLOWED_CITIES),
            "splits": list(ALLOWED_SPLITS),
            "roles_per_sample": list(DATA_ROLES),
            "nyc_inputs_opened": False,
            "official_test_inputs_opened": False,
            "verification": "full_file_sha256",
            "evidence_scope": (
                "pre_v4_current_state_attestation_not_retroactive_v3_byte_proof"
            ),
            "external_seal_sha256_required_by_launcher": True,
        },
        "runtime": runtime_value,
        "behavior_sources": sources,
        "behavior_sources_sha256": _artifact_aggregate(sources),
        "control_artifacts": controls,
        "control_artifacts_sha256": _artifact_aggregate(controls),
        "dataset": dataset,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise SealError(f"refusing to overwrite existing seal: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.pending-", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and not replace:
            raise SealError(f"refusing to overwrite existing seal: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_seal(
    config_path: Path,
    output_path: Path,
    *,
    jobs: int = 4,
    behavior_source_paths: Sequence[str | Path] = BEHAVIOR_SOURCE_PATHS,
    runtime: Mapping[str, Any] | None = None,
    replace: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    contract = build_contract(
        config_path,
        jobs=jobs,
        behavior_source_paths=behavior_source_paths,
        runtime=runtime,
        progress=progress,
    )
    payload = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": _sha256_bytes(_canonical_json(contract)),
        "contract": contract,
    }
    _atomic_write_json(output_path.expanduser().resolve(), payload, replace=replace)
    return payload


def _artifact_descriptors(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "path": item.get("path"),
            "roles": item.get("roles"),
            "project_relative_path": item.get("project_relative_path"),
        }
        for item in entries
    ]


def _verify_artifacts(
    sealed: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]], role: str
) -> None:
    if list(sealed) != list(current):
        raise SealError(f"{role} drifted from the sealed contract")


def verify_seal(
    seal_path: Path,
    *,
    config_path: Path | None = None,
    expected_seal_sha256: str | None = None,
    jobs: int = 4,
    runtime: Mapping[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    seal_path = seal_path.expanduser().resolve()
    _assert_regular_file(seal_path, "runtime/data seal")
    seal_file_hash = file_sha256(seal_path)
    if expected_seal_sha256 is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_seal_sha256):
            raise SealError("expected external seal SHA-256 is malformed")
        if seal_file_hash != expected_seal_sha256.lower():
            raise SealError(
                "runtime/data seal file differs from its external SHA-256 anchor"
            )
    payload = _load_json(seal_path, "runtime/data seal")
    if payload.get("schema") != SCHEMA:
        raise SealError(f"unsupported runtime/data seal schema: {payload.get('schema')!r}")
    contract = _mapping(payload.get("contract"), "sealed contract")
    stored_contract_hash = payload.get("contract_sha256")
    actual_contract_hash = _sha256_bytes(_canonical_json(contract))
    if stored_contract_hash != actual_contract_hash:
        raise SealError("sealed contract JSON fingerprint is invalid")

    policy = _mapping(contract.get("policy"), "sealed policy")
    if policy != {
        "cities": list(ALLOWED_CITIES),
        "splits": list(ALLOWED_SPLITS),
        "roles_per_sample": list(DATA_ROLES),
        "nyc_inputs_opened": False,
        "official_test_inputs_opened": False,
        "verification": "full_file_sha256",
        "evidence_scope": (
            "pre_v4_current_state_attestation_not_retroactive_v3_byte_proof"
        ),
        "external_seal_sha256_required_by_launcher": True,
    }:
        raise SealError("sealed policy is not the strict DC+PHL-only policy")

    controls = contract.get("control_artifacts")
    sources = contract.get("behavior_sources")
    dataset = _mapping(contract.get("dataset"), "sealed dataset")
    if not isinstance(controls, list) or not isinstance(sources, list):
        raise SealError("sealed artifact inventories must be lists")
    if contract.get("control_artifacts_sha256") != _artifact_aggregate(controls):
        raise SealError("sealed control-artifact aggregate is invalid")
    if contract.get("behavior_sources_sha256") != _artifact_aggregate(sources):
        raise SealError("sealed behavior-source aggregate is invalid")

    config_entries = [
        item for item in controls if "v4_config" in item.get("roles", [])
    ]
    if len(config_entries) != 1:
        raise SealError("sealed controls do not identify exactly one V4 config")
    sealed_config = Path(str(config_entries[0]["path"])).resolve()
    if config_path is not None and config_path.expanduser().resolve() != sealed_config:
        raise SealError("requested verification config differs from the sealed config")

    # Cheap, behavior-critical checks precede the large data read.
    current_runtime = dict(runtime) if runtime is not None else runtime_inventory()
    if current_runtime != contract.get("runtime"):
        raise SealError("runtime identity drifted from the sealed contract")
    current_sources = _behavior_sources(
        [str(item["path"]) for item in sources]
    )
    _verify_artifacts(sources, current_sources, "behavior source inventory")
    config = _load_yaml(sealed_config)
    current_controls = _control_artifacts(sealed_config, config)
    _verify_artifacts(controls, current_controls, "control artifact inventory")

    discovered, metadata_now = discover_data_entries(config)
    sealed_files = dataset.get("files")
    if not isinstance(sealed_files, list):
        raise SealError("sealed dataset file inventory must be a list")
    sealed_descriptors = [
        {key: item.get(key) for key in item if key != "sha256"}
        for item in sealed_files
    ]
    if discovered != sealed_descriptors:
        raise SealError("approved data membership/path/size drifted from the seal")
    for key, value in metadata_now.items():
        if dataset.get(key) != value:
            raise SealError(f"sealed dataset metadata drifted: {key}")
    hdf5_contract_now = validate_hdf5_entries(discovered, jobs=jobs)
    if dataset.get("hdf5_contract") != hdf5_contract_now:
        raise SealError("approved HDF5 structure/provenance drifted from the seal")
    current_files = hash_entries(discovered, jobs=jobs, progress=progress)
    if current_files != sealed_files:
        changed = [
            str(before.get("path"))
            for before, after in zip(sealed_files, current_files)
            if before != after
        ][:10]
        raise SealError(f"data file content drifted from the seal: {changed}")
    current_manifest_hash = _data_aggregate(current_files)
    if dataset.get("manifest_sha256") != current_manifest_hash:
        raise SealError("data manifest aggregate is invalid")
    return {
        "schema": "msr.gamus_v4_runtime_data_verification.v1",
        "passes": True,
        "seal_path": str(seal_path),
        "seal_file_sha256": seal_file_hash,
        "external_seal_sha256_verified": expected_seal_sha256 is not None,
        "contract_sha256": actual_contract_hash,
        "data_manifest_sha256": current_manifest_hash,
        "sample_count": metadata_now["sample_count"],
        "file_count": metadata_now["file_count"],
        "total_bytes": metadata_now["total_bytes"],
        "nyc_inputs_opened": False,
        "official_test_inputs_opened": False,
    }


def estimate(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path.expanduser().resolve())
    entries, metadata_value = discover_data_entries(config)
    total_bytes = int(metadata_value["total_bytes"])
    overhead_seconds = len(entries) * 0.0005
    estimates = {}
    for throughput in (100, 250, 500):
        seconds = total_bytes / (throughput * 1024 * 1024) + overhead_seconds
        estimates[f"at_{throughput}_mib_per_second"] = round(seconds, 1)
    return {
        "schema": "msr.gamus_v4_runtime_data_seal_estimate.v1",
        **metadata_value,
        "hashing_seconds_estimate": estimates,
        "note": "metadata-only estimate; actual speed depends on D-volume storage and concurrent load",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate_parser = subparsers.add_parser("estimate", help="metadata-only cost estimate")
    estimate_parser.add_argument("--config", type=Path, required=True)

    create_parser = subparsers.add_parser("create", help="atomically create a new seal")
    create_parser.add_argument("--config", type=Path, required=True)
    create_parser.add_argument("--output", type=Path, required=True)
    create_parser.add_argument("--jobs", type=int, default=4)
    create_parser.add_argument("--replace", action="store_true")
    create_parser.add_argument("--allow-no-cuda", action="store_true")
    create_parser.add_argument("--progress-every", type=int, default=500)

    verify_parser = subparsers.add_parser("verify", help="fail on any sealed drift")
    verify_parser.add_argument("--seal", type=Path, required=True)
    verify_parser.add_argument("--expected-seal-sha256", required=True)
    verify_parser.add_argument("--config", type=Path)
    verify_parser.add_argument("--jobs", type=int, default=4)
    verify_parser.add_argument("--allow-no-cuda", action="store_true")
    verify_parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "estimate":
        result = estimate(args.config)
    elif args.command == "create":
        runtime = runtime_inventory(require_cuda=not args.allow_no_cuda)
        result = create_seal(
            args.config,
            args.output,
            jobs=args.jobs,
            runtime=runtime,
            replace=args.replace,
            progress=_progress_printer(args.progress_every, "V4 seal"),
        )
        result = {
            "schema": result["schema"],
            "output": str(args.output.expanduser().resolve()),
            "seal_file_sha256": file_sha256(args.output.expanduser().resolve()),
            "contract_sha256": result["contract_sha256"],
            "sample_count": result["contract"]["dataset"]["sample_count"],
            "file_count": result["contract"]["dataset"]["file_count"],
            "total_bytes": result["contract"]["dataset"]["total_bytes"],
        }
    else:
        runtime = runtime_inventory(require_cuda=not args.allow_no_cuda)
        result = verify_seal(
            args.seal,
            config_path=args.config,
            expected_seal_sha256=args.expected_seal_sha256,
            jobs=args.jobs,
            runtime=runtime,
            progress=_progress_printer(args.progress_every, "V4 verify"),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
