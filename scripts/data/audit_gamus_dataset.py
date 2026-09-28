"""Build an immutable, approved-only quality contract for native GAMUS data.

The source HDF5 files are never changed or moved.  Structurally or numerically
unsafe tiles, plus upstream-reported corrupt tiles, are recorded in a logical
quarantine and omitted from ``approved_samples.json``.  The complete result is
published by one directory rename so interrupted audits cannot leave a partial
training index behind.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
from tqdm import tqdm

from msr.data.gamus_dataset import (
    GAMUS_CLASSES,
    GAMUS_OFFICIAL_SPLIT_COUNTS,
    GAMUS_REGRESSION_CLASS_IDS,
    GAMUS_SPLITS,
)

try:
    import h5py
except ImportError:  # pragma: no cover - the CLI gives a clearer error below.
    h5py = None


AUDIT_SCHEMA = "msr.gamus.quality_audit.v1.1"
APPROVED_INDEX_SCHEMA = "msr.gamus.approved_samples.v1"
QUARANTINE_SCHEMA = "msr.gamus.logical_quarantine.v1"
SPLIT_ORDER = ("train", "val", "test")
DEFAULT_HEIGHT_MAX_M = 200.0
KNOWN_QUARANTINE: dict[tuple[str, str], dict[str, str]] = {
    ("test", "PHL_4001"): {
        "code": "upstream_reported_corrupt_height",
        "detail": (
            "PHL_4001_AGL.h5 is excluded because the released test tile was "
            "reported as corrupt upstream. The original files remain untouched."
        ),
        "source": "https://huggingface.co/datasets/earthflow/GAMUS/discussions/3",
    }
}
UNIT_CONTRACT = {
    "field": "AGL/nDSM above-ground surface height",
    "training_unit": "metre",
    "raw_to_training_scale": 1.0,
    "status": "explicit_project_assumption",
    "explanation": (
        "Monocular Surface Reconstruction treats raw GAMUS AGL/nDSM values as metres without rescaling. "
        "The local HDF5 files and dataset README do not declare a numeric unit, so "
        "this is recorded as an assumption rather than claimed as file-metadata proof."
    ),
    "provenance": [
        {
            "claim": "GAMUS pairs RGB imagery with nDSM/above-ground height labels",
            "source": "https://arxiv.org/html/2305.14914",
        },
        {
            "claim": "The official loader reads the AGL array directly without scaling",
            "source": (
                "https://github.com/EarthNets/RSI-MMSegmentation/"
                "blob/main/gamus_dataset.py"
            ),
        },
        {
            "claim": "The local dataset provenance identifies earthflow/GAMUS",
            "source": "msr_provenance.json",
        },
    ],
}


def _require_h5py() -> Any:
    if h5py is None:
        raise RuntimeError("GAMUS QC requires h5py (install the train extra)")
    return h5py


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id(path: Path, suffix: str) -> str | None:
    if not path.name.endswith(suffix):
        return None
    value = path.name[: -len(suffix)]
    return value or None


def _role_files(directory: Path, suffixes: tuple[str, ...]) -> tuple[dict[str, list[Path]], list[Path]]:
    roles: dict[str, list[Path]] = {}
    unrecognized: list[Path] = []
    for path in sorted(directory.glob("*.h5")):
        matched = False
        for suffix in suffixes:
            sample_id = _sample_id(path, suffix)
            if sample_id is not None:
                roles.setdefault(sample_id, []).append(path)
                matched = True
                break
        if not matched:
            unrecognized.append(path)
    return roles, unrecognized


def discover_split(root: Path, split: str) -> dict[str, Any]:
    """Discover every role and preserve duplicate/orphan evidence."""

    image_files, image_unknown = _role_files(
        root / "images" / split, ("_RGB.h5", "_IMG.h5")
    )
    height_files, height_unknown = _role_files(
        root / "heights" / split, ("_AGL.h5",)
    )
    class_files, class_unknown = _role_files(
        root / "classes" / split, ("_CLS.h5",)
    )
    all_ids = sorted(set(image_files) | set(height_files) | set(class_files))
    return {
        "images": image_files,
        "heights": height_files,
        "classes": class_files,
        "sample_ids": all_ids,
        "unrecognized": sorted(image_unknown + height_unknown + class_unknown),
    }


def _relative_file_record(root: Path, split: str, role: str, sample_id: str, path: Path) -> dict[str, Any]:
    return {
        "split": split,
        "role": role,
        "sample_id": sample_id,
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
    }


def inventory_fingerprint(root: Path, discoveries: Mapping[str, dict[str, Any]]) -> str:
    records: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        discovered = discoveries[split]
        for role in ("images", "heights", "classes"):
            for sample_id in sorted(discovered[role]):
                for path in sorted(discovered[role][sample_id]):
                    records.append(
                        _relative_file_record(root, split, role, sample_id, path)
                    )
        for path in discovered["unrecognized"]:
            records.append(
                {
                    "split": split,
                    "role": "unrecognized",
                    "sample_id": "",
                    "path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                }
            )
    return _canonical_sha256(records)


def _dataset(handle: Any, path: Path) -> Any:
    if "image" not in handle:
        raise ValueError(f"missing root image dataset: {path.name}")
    return handle["image"]


def _is_numeric(dtype: np.dtype[Any]) -> bool:
    return bool(np.issubdtype(dtype, np.number))


def _declared_nodata(dataset: Any, handle: Any) -> tuple[float, ...]:
    values: list[float] = []
    for attributes in (dataset.attrs, handle.attrs):
        for name in ("_FillValue", "fill_value", "nodata", "NoDataValue"):
            if name not in attributes:
                continue
            for raw in np.asarray(attributes[name]).reshape(-1):
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    continue
    return tuple(values)


def _one(paths: list[Path], role: str, sample_id: str) -> tuple[Path | None, list[str]]:
    if not paths:
        return None, [f"missing_{role}_file"]
    if len(paths) != 1:
        return None, [f"duplicate_{role}_files"]
    return paths[0], []


def audit_triplet(
    *,
    split: str,
    sample_id: str,
    image_paths: list[Path],
    height_paths: list[Path],
    class_paths: list[Path],
    height_max_m: float,
    block_rows: int,
) -> dict[str, Any]:
    """Audit one triplet without ever modifying it."""

    image_path, reasons = _one(image_paths, "image", sample_id)
    height_path, more = _one(height_paths, "height", sample_id)
    reasons.extend(more)
    class_path, more = _one(class_paths, "class", sample_id)
    reasons.extend(more)
    record: dict[str, Any] = {
        "split": split,
        "sample_id": sample_id,
        "status": "quarantined" if reasons else "pending",
        "reasons": reasons,
        "warnings": [],
        "paths": {
            "image": str(image_path) if image_path is not None else None,
            "height": str(height_path) if height_path is not None else None,
            "class": str(class_path) if class_path is not None else None,
        },
    }
    if reasons:
        return record

    assert image_path is not None and height_path is not None and class_path is not None
    h5 = _require_h5py()
    try:
        with (
            h5.File(image_path, "r") as image_handle,
            h5.File(height_path, "r") as height_handle,
            h5.File(class_path, "r") as class_handle,
        ):
            image = _dataset(image_handle, image_path)
            height = _dataset(height_handle, height_path)
            classes = _dataset(class_handle, class_path)
            record["arrays"] = {
                "image": {"shape": list(image.shape), "dtype": str(image.dtype)},
                "height": {"shape": list(height.shape), "dtype": str(height.dtype)},
                "class": {"shape": list(classes.shape), "dtype": str(classes.dtype)},
            }
            if not _is_numeric(image.dtype):
                reasons.append("image_dtype_not_numeric")
            if not _is_numeric(height.dtype):
                reasons.append("height_dtype_not_numeric")
            if not _is_numeric(classes.dtype):
                reasons.append("class_dtype_not_numeric")
            if image.ndim != 3 or image.shape[-1] < 3:
                reasons.append("image_not_hwc_rgb")
            if height.ndim != 2:
                reasons.append("height_not_2d")
            if classes.ndim != 2:
                reasons.append("class_not_2d")
            if reasons:
                record["status"] = "quarantined"
                return record
            spatial = tuple(image.shape[:2])
            if tuple(height.shape) != spatial or tuple(classes.shape) != spatial:
                reasons.append("array_shape_mismatch")
                record["status"] = "quarantined"
                return record

            total = int(height.size)
            counters: Counter[str] = Counter()
            class_counts: Counter[int] = Counter()
            valid_height_min: float | None = None
            valid_height_max: float | None = None
            valid_height_sum = 0.0
            valid_height_sum_squares = 0.0
            nodata_values = _declared_nodata(height, height_handle)
            for start in range(0, spatial[0], block_rows):
                stop = min(start + block_rows, spatial[0])
                height_block = np.asarray(height[start:stop], dtype=np.float64)
                class_block = np.asarray(classes[start:stop], dtype=np.float64)

                finite_height = np.isfinite(height_block)
                declared_nodata = np.zeros(height_block.shape, dtype=bool)
                for nodata in nodata_values:
                    if np.isnan(nodata):
                        declared_nodata |= np.isnan(height_block)
                    else:
                        declared_nodata |= np.isclose(
                            height_block, nodata, rtol=0.0, atol=1.0e-6,
                            equal_nan=True,
                        )
                valid_height = (
                    finite_height
                    & ~declared_nodata
                    & (height_block >= 0.0)
                    & (height_block <= height_max_m)
                )
                counters["height_finite"] += int(finite_height.sum())
                counters["height_declared_nodata"] += int(declared_nodata.sum())
                counters["height_negative"] += int(
                    (finite_height & ~declared_nodata & (height_block < 0.0)).sum()
                )
                counters["height_above_limit"] += int(
                    (
                        finite_height
                        & ~declared_nodata
                        & (height_block > height_max_m)
                    ).sum()
                )
                counters["height_valid"] += int(valid_height.sum())
                if valid_height.any():
                    valid_values = height_block[valid_height]
                    block_min = float(valid_values.min())
                    block_max = float(valid_values.max())
                    valid_height_min = (
                        block_min
                        if valid_height_min is None
                        else min(valid_height_min, block_min)
                    )
                    valid_height_max = (
                        block_max
                        if valid_height_max is None
                        else max(valid_height_max, block_max)
                    )
                    valid_height_sum += float(valid_values.sum(dtype=np.float64))
                    valid_height_sum_squares += float(
                        np.square(valid_values).sum(dtype=np.float64)
                    )

                finite_class = np.isfinite(class_block)
                rounded = np.rint(class_block)
                integral_class = finite_class & np.isclose(
                    class_block, rounded, rtol=0.0, atol=1.0e-5
                )
                in_range = (
                    integral_class
                    & (rounded >= min(GAMUS_CLASSES))
                    & (rounded <= max(GAMUS_CLASSES))
                )
                counters["class_finite"] += int(finite_class.sum())
                counters["class_noninteger"] += int(
                    (finite_class & ~integral_class).sum()
                )
                counters["class_out_of_range"] += int(
                    (integral_class & ~in_range).sum()
                )
                counters["semantic_supervised"] += int(
                    (in_range & (rounded != 0)).sum()
                )
                for class_id in GAMUS_CLASSES:
                    count = int((in_range & (rounded == class_id)).sum())
                    if count:
                        class_counts[class_id] += count
                regression_class = in_range & np.isin(
                    rounded, tuple(GAMUS_REGRESSION_CLASS_IDS)
                )
                counters["regression_supervised"] += int(
                    (valid_height & regression_class).sum()
                )

            invalid_class = (
                total - counters["class_finite"]
                + counters["class_noninteger"]
                + counters["class_out_of_range"]
            )
            if invalid_class:
                record["warnings"].append("invalid_class_pixels_masked_as_ignore")
            if counters["height_valid"] == 0:
                record["warnings"].append(
                    "no_valid_height_pixels_semantic_supervision_only"
                )
            if counters["regression_supervised"] == 0:
                record["warnings"].append(
                    "no_regression_supervision_semantic_supervision_only"
                )
            if (
                counters["semantic_supervised"] == 0
                and counters["regression_supervised"] == 0
            ):
                reasons.append("no_usable_semantic_or_height_supervision")
            if counters["height_negative"]:
                record["warnings"].append("negative_height_pixels_masked_as_invalid")
            if counters["height_above_limit"]:
                record["warnings"].append("height_pixels_above_policy_limit_masked")
            if counters["height_declared_nodata"]:
                record["warnings"].append("declared_nodata_pixels_masked")

            valid_count = counters["height_valid"]
            valid_mean = valid_height_sum / valid_count if valid_count else None
            valid_variance = (
                max(0.0, valid_height_sum_squares / valid_count - valid_mean**2)
                if valid_count and valid_mean is not None
                else None
            )
            record["numeric"] = {
                "pixels": total,
                "height_finite_pixels": counters["height_finite"],
                "height_valid_pixels": valid_count,
                "height_negative_pixels": counters["height_negative"],
                "height_above_limit_pixels": counters["height_above_limit"],
                "height_declared_nodata_pixels": counters["height_declared_nodata"],
                "height_valid_min": valid_height_min,
                "height_valid_max": valid_height_max,
                "height_valid_mean": valid_mean,
                "height_valid_std": (
                    valid_variance**0.5 if valid_variance is not None else None
                ),
                "class_finite_pixels": counters["class_finite"],
                "class_noninteger_pixels": counters["class_noninteger"],
                "class_out_of_range_pixels": counters["class_out_of_range"],
                "semantic_supervised_pixels": counters["semantic_supervised"],
                "class_counts": {
                    str(class_id): class_counts[class_id]
                    for class_id in sorted(GAMUS_CLASSES)
                },
                "regression_supervised_pixels": counters[
                    "regression_supervised"
                ],
            }
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        reasons.append(f"hdf5_read_error:{type(error).__name__}:{error}")

    known = KNOWN_QUARANTINE.get((split, sample_id))
    if known is not None:
        reasons.append(known["code"])
        record["known_quarantine"] = known
    record["eligible_tasks"] = []
    if record.get("numeric", {}).get("semantic_supervised_pixels", 0) > 0:
        record["eligible_tasks"].append("semantic")
    if record.get("numeric", {}).get("regression_supervised_pixels", 0) > 0:
        record["eligible_tasks"].append("height_regression")
    record["status"] = "approved" if not reasons else "quarantined"
    return record


def apply_task_eligibility_policy(record: dict[str, Any]) -> dict[str, Any]:
    """Upgrade earlier strict scan records without rereading the HDF5 arrays.

    A tile with sound classes but no valid height-regression pixels remains
    useful to the semantic loss.  It must be approved as semantic-only rather
    than mislabeled as corrupt.
    """

    updated = json.loads(json.dumps(record))
    reasons = [
        str(reason)
        for reason in updated.get("reasons", [])
        if reason
        not in {
            "invalid_class_values",
            "no_valid_height_pixels",
            "no_regression_supervision",
        }
    ]
    warnings = [str(value) for value in updated.get("warnings", [])]
    numeric = updated.get("numeric") or {}
    class_counts = numeric.get("class_counts") or {}
    semantic_pixels = int(
        numeric.get(
            "semantic_supervised_pixels",
            sum(int(class_counts.get(str(class_id), 0)) for class_id in range(1, 7)),
        )
    )
    regression_pixels = int(numeric.get("regression_supervised_pixels", 0))
    numeric["semantic_supervised_pixels"] = semantic_pixels
    updated["numeric"] = numeric
    if int(numeric.get("height_valid_pixels", 0)) == 0:
        warning = "no_valid_height_pixels_semantic_supervision_only"
        if warning not in warnings:
            warnings.append(warning)
    if regression_pixels == 0:
        warning = "no_regression_supervision_semantic_supervision_only"
        if warning not in warnings:
            warnings.append(warning)
    invalid_class_pixels = (
        int(numeric.get("pixels", 0))
        - int(numeric.get("class_finite_pixels", 0))
        + int(numeric.get("class_noninteger_pixels", 0))
        + int(numeric.get("class_out_of_range_pixels", 0))
    )
    if invalid_class_pixels > 0:
        warning = "invalid_class_pixels_masked_as_ignore"
        if warning not in warnings:
            warnings.append(warning)
    if semantic_pixels == 0 and regression_pixels == 0:
        reason = "no_usable_semantic_or_height_supervision"
        if reason not in reasons:
            reasons.append(reason)
    known = KNOWN_QUARANTINE.get((updated["split"], updated["sample_id"]))
    if known is not None:
        if known["code"] not in reasons:
            reasons.append(known["code"])
        updated["known_quarantine"] = known
    updated["reasons"] = reasons
    updated["warnings"] = warnings
    updated["eligible_tasks"] = []
    if semantic_pixels > 0:
        updated["eligible_tasks"].append("semantic")
    if regression_pixels > 0:
        updated["eligible_tasks"].append("height_regression")
    updated["status"] = "approved" if not reasons else "quarantined"
    return updated


def _sum_numeric(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    aggregate = Counter()
    class_counts: Counter[str] = Counter()
    minima: list[float] = []
    maxima: list[float] = []
    for record in records:
        numeric = record.get("numeric")
        if not numeric:
            continue
        for key in (
            "pixels",
            "height_finite_pixels",
            "height_valid_pixels",
            "height_negative_pixels",
            "height_above_limit_pixels",
            "height_declared_nodata_pixels",
            "class_finite_pixels",
            "class_noninteger_pixels",
            "class_out_of_range_pixels",
            "semantic_supervised_pixels",
            "regression_supervised_pixels",
        ):
            aggregate[key] += int(numeric[key])
        class_counts.update(numeric["class_counts"])
        if numeric["height_valid_min"] is not None:
            minima.append(float(numeric["height_valid_min"]))
        if numeric["height_valid_max"] is not None:
            maxima.append(float(numeric["height_valid_max"]))
    return {
        **dict(aggregate),
        "height_valid_min": min(minima) if minima else None,
        "height_valid_max": max(maxima) if maxima else None,
        "class_counts": {
            key: class_counts[key] for key in sorted(class_counts, key=int)
        },
    }


def _validate_global_inventory(
    discoveries: Mapping[str, dict[str, Any]],
    expected_counts: Mapping[str, int],
    *,
    require_expected_counts: bool,
) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    ids_by_split: dict[str, set[str]] = {}
    failures: list[str] = []
    for split in SPLIT_ORDER:
        discovered = discoveries[split]
        ids_by_split[split] = set(discovered["sample_ids"])
        role_counts = {
            role: len(discovered[role])
            for role in ("images", "heights", "classes")
        }
        duplicate_counts = {
            role: sum(len(paths) - 1 for paths in discovered[role].values())
            for role in ("images", "heights", "classes")
        }
        expected = int(expected_counts[split])
        inventory[split] = {
            "expected_official_tiles": expected,
            "unique_image_tile_ids": role_counts["images"],
            "unique_height_tile_ids": role_counts["heights"],
            "unique_class_tile_ids": role_counts["classes"],
            "union_tile_ids": len(discovered["sample_ids"]),
            "duplicate_files": duplicate_counts,
            "unrecognized_h5_files": [str(path) for path in discovered["unrecognized"]],
        }
        if require_expected_counts and role_counts["images"] != expected:
            failures.append(
                f"{split} image count {role_counts['images']} != expected {expected}"
            )
        if require_expected_counts and role_counts["heights"] != expected:
            failures.append(
                f"{split} height count {role_counts['heights']} != expected {expected}"
            )
        if require_expected_counts and role_counts["classes"] != expected:
            failures.append(
                f"{split} class count {role_counts['classes']} != expected {expected}"
            )
        if any(duplicate_counts.values()):
            failures.append(f"{split} contains duplicate role files")
        if discovered["unrecognized"]:
            failures.append(f"{split} contains unrecognized HDF5 filenames")

    overlaps: dict[str, list[str]] = {}
    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1 :]:
            overlap = sorted(ids_by_split[left] & ids_by_split[right])
            overlaps[f"{left}__{right}"] = overlap
            if overlap:
                failures.append(
                    f"tile leakage between {left} and {right}: {overlap[:5]}"
                )
    return {
        "splits": inventory,
        "split_overlaps": overlaps,
        "split_ids_disjoint": not any(overlaps.values()),
        "failures": failures,
    }


def build_quality_contract(
    root: Path,
    output_root: Path,
    *,
    expected_counts: Mapping[str, int] = GAMUS_OFFICIAL_SPLIT_COUNTS,
    require_expected_counts: bool = True,
    height_max_m: float = DEFAULT_HEIGHT_MAX_M,
    block_rows: int = 256,
    show_progress: bool = True,
    reuse_tile_audit: Path | None = None,
) -> dict[str, Any]:
    """Scan all GAMUS triplets and atomically publish a versioned contract."""

    root = root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"GAMUS root does not exist: {root}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite quality contract: {output_root}")
    if height_max_m <= 0 or block_rows <= 0:
        raise ValueError("height_max_m and block_rows must be positive")
    missing_expected = set(SPLIT_ORDER) - set(expected_counts)
    if missing_expected:
        raise ValueError(f"expected_counts missing splits: {sorted(missing_expected)}")

    discoveries = {split: discover_split(root, split) for split in SPLIT_ORDER}
    inventory = _validate_global_inventory(
        discoveries,
        expected_counts,
        require_expected_counts=require_expected_counts,
    )
    if inventory["failures"]:
        raise ValueError("GAMUS inventory failed: " + "; ".join(inventory["failures"]))

    fingerprint = inventory_fingerprint(root, discoveries)
    expected_keys = {
        (split, sample_id)
        for split in SPLIT_ORDER
        for sample_id in discoveries[split]["sample_ids"]
    }
    if reuse_tile_audit is not None:
        audit_path = reuse_tile_audit.expanduser().resolve()
        all_records = [
            apply_task_eligibility_policy(json.loads(line))
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        actual_keys = {(r["split"], r["sample_id"]) for r in all_records}
        if len(actual_keys) != len(all_records) or actual_keys != expected_keys:
            raise ValueError(
                "reused tile audit does not exactly match the current GAMUS inventory"
            )
    else:
        all_records = []
        approved_so_far = 0
        quarantined_so_far = 0
        progress = tqdm(
            total=len(expected_keys),
            desc="GAMUS full QC",
            unit="tile",
            disable=not show_progress,
        )
        try:
            for split in SPLIT_ORDER:
                discovered = discoveries[split]
                for sample_id in discovered["sample_ids"]:
                    record = apply_task_eligibility_policy(
                        audit_triplet(
                            split=split,
                            sample_id=sample_id,
                            image_paths=discovered["images"].get(sample_id, []),
                            height_paths=discovered["heights"].get(sample_id, []),
                            class_paths=discovered["classes"].get(sample_id, []),
                            height_max_m=height_max_m,
                            block_rows=block_rows,
                        )
                    )
                    all_records.append(record)
                    approved_so_far += int(record["status"] == "approved")
                    quarantined_so_far += int(record["status"] == "quarantined")
                    progress.update(1)
                    progress.set_postfix(
                        approved=approved_so_far,
                        quarantined=quarantined_so_far,
                        refresh=False,
                    )
        finally:
            progress.close()

    approved_by_split = {
        split: sorted(
            record["sample_id"]
            for record in all_records
            if record["split"] == split and record["status"] == "approved"
        )
        for split in SPLIT_ORDER
    }
    quarantined = [
        record for record in all_records if record["status"] == "quarantined"
    ]
    index = {
        "schema": APPROVED_INDEX_SCHEMA,
        "dataset": "GAMUS",
        "source_inventory_metadata_sha256": fingerprint,
        "source_inventory_fingerprint_scope": (
            "sorted relative HDF5 paths, roles, split IDs, and byte sizes; this is "
            "a deterministic metadata fingerprint, not a full 80 GB content hash"
        ),
        "height_unit_contract": UNIT_CONTRACT,
        "height_validity_policy": {
            "minimum": 0.0,
            "maximum": height_max_m,
            "unit": "metre_assumed",
            "negative_non_nodata_values": "masked",
            "nonfinite_values": "masked",
            "values_above_maximum": "masked",
        },
        "class_validity_policy": {
            "allowed_integer_ids": sorted(GAMUS_CLASSES),
            "class_zero": "valid_ignore_label",
            "invalid_pixels": "masked_as_ignore",
            "tile_without_any_usable_semantic_or_height_supervision": (
                "quarantine_tile"
            ),
        },
        "splits": {
            split: {
                "source_count": len(discoveries[split]["sample_ids"]),
                "expected_official_count": int(expected_counts[split]),
                "approved_count": len(approved_by_split[split]),
                "approved_sample_ids": approved_by_split[split],
                "semantic_eligible_sample_ids": sorted(
                    record["sample_id"]
                    for record in all_records
                    if record["split"] == split
                    and record["status"] == "approved"
                    and "semantic" in record.get("eligible_tasks", [])
                ),
                "height_regression_eligible_sample_ids": sorted(
                    record["sample_id"]
                    for record in all_records
                    if record["split"] == split
                    and record["status"] == "approved"
                    and "height_regression" in record.get("eligible_tasks", [])
                ),
            }
            for split in SPLIT_ORDER
        },
    }
    quarantine_contract = {
        "schema": QUARANTINE_SCHEMA,
        "mode": "logical_only_source_files_untouched",
        "source_inventory_metadata_sha256": fingerprint,
        "count": len(quarantined),
        "records": quarantined,
    }
    report_splits: dict[str, Any] = {}
    for split in SPLIT_ORDER:
        split_records = [r for r in all_records if r["split"] == split]
        split_quarantine = [r for r in split_records if r["status"] == "quarantined"]
        report_splits[split] = {
            **inventory["splits"][split],
            "audited_tiles": len(split_records),
            "approved_tiles": len(approved_by_split[split]),
            "semantic_eligible_tiles": sum(
                record["status"] == "approved"
                and "semantic" in record.get("eligible_tasks", [])
                for record in split_records
            ),
            "height_regression_eligible_tiles": sum(
                record["status"] == "approved"
                and "height_regression" in record.get("eligible_tasks", [])
                for record in split_records
            ),
            "quarantined_tiles": len(split_quarantine),
            "quarantined_sample_ids": sorted(r["sample_id"] for r in split_quarantine),
            "numeric_totals": _sum_numeric(split_records),
        }
    report = {
        "schema": AUDIT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "source_inventory_metadata_sha256": fingerprint,
        "source_files_modified": False,
        "source_files_deleted": False,
        "split_ids_disjoint": inventory["split_ids_disjoint"],
        "split_overlaps": inventory["split_overlaps"],
        "height_unit_contract": UNIT_CONTRACT,
        "policies": {
            "height_max_m": height_max_m,
            "block_rows": block_rows,
            "known_quarantine": [
                {"split": split, "sample_id": sample_id, **details}
                for (split, sample_id), details in sorted(KNOWN_QUARANTINE.items())
            ],
        },
        "splits": report_splits,
        "totals": {
            "audited_tiles": len(all_records),
            "approved_tiles": sum(len(ids) for ids in approved_by_split.values()),
            "quarantined_tiles": len(quarantined),
        },
        "approved_index_canonical_json_sha256": _canonical_sha256(index),
        "quarantine_contract_canonical_json_sha256": _canonical_sha256(
            quarantine_contract
        ),
        "supersedes_quality_contract_root": (
            str(reuse_tile_audit.expanduser().resolve().parent)
            if reuse_tile_audit is not None
            else None
        ),
        "remaining_uncertainty": [
            (
                "GAMUS numeric height units are treated as metres by explicit project "
                "assumption; the local HDF5 metadata does not independently declare them."
            ),
            (
                "This audit proves file structure, numeric admissibility, alignment, split "
                "identity, and the configured logical quarantine. It does not prove that "
                "every geographically aligned label is free of upstream annotation error."
            ),
        ],
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.pending-", dir=output_root.parent
        )
    )
    try:
        index_path = staging / "approved_samples.json"
        quarantine_path = staging / "quarantine.json"
        tile_audit_path = staging / "tile_audit.jsonl"
        _write_json(index_path, index)
        _write_json(quarantine_path, quarantine_contract)
        with tile_audit_path.open("w", encoding="utf-8") as handle:
            for record in sorted(
                all_records,
                key=lambda item: (SPLIT_ORDER.index(item["split"]), item["sample_id"]),
            ):
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        # These values intentionally hash the exact persisted bytes, including
        # whitespace and the final newline.  Earlier v1 reports accidentally
        # recorded canonical-object hashes under file-hash names.
        report["artifact_hash_scope"] = "sha256_of_exact_persisted_file_bytes"
        report["approved_index_sha256"] = _file_sha256(index_path)
        report["quarantine_contract_sha256"] = _file_sha256(quarantine_path)
        report["tile_audit_sha256"] = _file_sha256(tile_audit_path)
        report["artifact_sha256"] = {
            "approved_samples.json": report["approved_index_sha256"],
            "quarantine.json": report["quarantine_contract_sha256"],
            "tile_audit.jsonl": report["tile_audit_sha256"],
        }
        _write_json(staging / "qc_report.json", report)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--height-max-m", type=float, default=DEFAULT_HEIGHT_MAX_M)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument(
        "--allow-nonofficial-counts",
        action="store_true",
        help="For synthetic/development data only; production should keep this off.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--reuse-tile-audit",
        type=Path,
        help="Reuse a completed tile_audit.jsonl and only republish policy metadata.",
    )
    args = parser.parse_args()
    report = build_quality_contract(
        args.root,
        args.output_root,
        require_expected_counts=not args.allow_nonofficial_counts,
        height_max_m=args.height_max_m,
        block_rows=args.block_rows,
        show_progress=not args.no_progress,
        reuse_tile_audit=args.reuse_tile_audit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
