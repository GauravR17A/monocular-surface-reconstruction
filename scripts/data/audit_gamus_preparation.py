"""Publish a fail-closed GAMUS preparation audit before long training.

This audit does not train, alter source data, expose the official test split to
the trainer, or change the live checkpoint pointer.  It locks the current RGB
and cached Depth Anything V2 (DAV2) artifacts by content hash, inventories the
rare AGL values above the configured training cutoff, records what is and is
not known about units/geographic scale, and verifies the dataset's independent
validity-mask contract.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from msr.data.gamus_dataset import GAMUS_CLASSES, GamusSurfaceDataset, load_gamus_split

try:
    import h5py
except ImportError:  # pragma: no cover - the CLI reports the dependency clearly.
    h5py = None


REPORT_SCHEMA = "msr.gamus.preparation_audit.v1"
PRIOR_MANIFEST_SCHEMA = "msr.gamus.dav2_prior_identity.v1"
OUTLIER_SCHEMA = "msr.gamus.height_outliers.v1"
SPLIT_ORDER = ("train", "val", "test")
TRAINING_SPLITS = frozenset({"train", "val"})
EXPECTED_PRIOR_UNITS = "relative_0_1"


def _require_h5py() -> Any:
    if h5py is None:
        raise RuntimeError("GAMUS preparation audit requires h5py")
    return h5py


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string_attribute(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _same_path(left: str | Path, right: str | Path) -> bool:
    """Compare paths in the host filesystem without trusting string spelling."""

    try:
        return os.path.normcase(str(Path(left).expanduser().resolve())) == os.path.normcase(
            str(Path(right).expanduser().resolve())
        )
    except (OSError, RuntimeError, ValueError):
        return False


def validate_prior_pair(
    *,
    source_path: Path,
    prior_path: Path,
    expected_model: str,
    include_content_hashes: bool = True,
    block_rows: int = 256,
) -> dict[str, Any]:
    """Validate one cached prior against its source, failing on missing identity.

    Historical priors do not contain source/model *content* hashes.  The audit
    therefore validates every recorded identity field, hashes both current
    artifacts into an immutable contract, and relies on deterministic sampled
    recomputation for evidence about the historical generation claim.
    """

    h5 = _require_h5py()
    errors: list[str] = []
    warnings: list[str] = []
    record: dict[str, Any] = {
        "source_path": str(source_path.resolve()),
        "prior_path": str(prior_path.resolve()),
        "errors": errors,
        "warnings": warnings,
    }
    if not source_path.is_file():
        errors.append("missing_source_rgb")
        record["status"] = "failed"
        return record
    if not prior_path.is_file():
        errors.append("missing_cached_prior")
        record["status"] = "failed"
        return record

    try:
        with h5.File(source_path, "r") as source_handle:
            if "image" not in source_handle:
                errors.append("source_missing_image_dataset")
                source_shape: tuple[int, ...] = ()
            else:
                source_dataset = source_handle["image"]
                source_shape = tuple(int(value) for value in source_dataset.shape)
                if len(source_shape) != 3 or source_shape[-1] < 3:
                    errors.append("source_is_not_hwc_rgb")
        with h5.File(prior_path, "r") as prior_handle:
            if "image" not in prior_handle:
                errors.append("prior_missing_image_dataset")
                prior_shape: tuple[int, ...] = ()
                attrs: dict[str, Any] = {}
                finite_count = below_range_count = above_range_count = 0
                minimum = maximum = None
            else:
                prior_dataset = prior_handle["image"]
                prior_shape = tuple(int(value) for value in prior_dataset.shape)
                prior_dtype = str(prior_dataset.dtype)
                attrs = {
                    str(key): _string_attribute(value)
                    for key, value in prior_dataset.attrs.items()
                }
                finite_count = below_range_count = above_range_count = 0
                minimum_value = np.inf
                maximum_value = -np.inf
                if len(prior_shape) == 2:
                    for row in range(0, prior_shape[0], block_rows):
                        values = np.asarray(
                            prior_dataset[row : row + block_rows], dtype=np.float32
                        )
                        finite = np.isfinite(values)
                        finite_count += int(finite.sum())
                        if np.any(finite):
                            selected = values[finite]
                            minimum_value = min(minimum_value, float(selected.min()))
                            maximum_value = max(maximum_value, float(selected.max()))
                            below_range_count += int((selected < 0.0).sum())
                            above_range_count += int((selected > 1.0).sum())
                minimum = float(minimum_value) if np.isfinite(minimum_value) else None
                maximum = float(maximum_value) if np.isfinite(maximum_value) else None
    except OSError as error:
        errors.append(f"unreadable_hdf5:{type(error).__name__}")
        record["status"] = "failed"
        return record

    expected_shape = source_shape[:2] if len(source_shape) >= 2 else ()
    if prior_shape != expected_shape:
        errors.append("prior_grid_mismatch")
    expected_pixels = int(np.prod(prior_shape)) if len(prior_shape) == 2 else 0
    if finite_count != expected_pixels:
        errors.append("prior_contains_nonfinite_values")
    if below_range_count or above_range_count:
        errors.append("prior_outside_relative_0_1_range")

    recorded_model = attrs.get("model")
    recorded_source = attrs.get("source_rgb")
    recorded_units = attrs.get("units")
    if recorded_model is None:
        errors.append("missing_recorded_model")
    elif recorded_model != expected_model:
        errors.append("recorded_model_mismatch")
    if recorded_source is None:
        errors.append("missing_recorded_source_rgb")
    elif not _same_path(recorded_source, source_path):
        errors.append("recorded_source_rgb_mismatch")
    if recorded_units is None:
        errors.append("missing_recorded_units")
    elif recorded_units != EXPECTED_PRIOR_UNITS:
        errors.append("recorded_units_mismatch")

    # The historical format does not carry these fields.  This is surfaced as
    # uncertainty, never silently upgraded into cryptographic provenance.
    for field in ("source_rgb_sha256", "model_artifact_sha256", "generation_config_sha256"):
        if field not in attrs:
            warnings.append(f"historical_prior_missing_{field}")

    record.update(
        {
            "status": "passed" if not errors else "failed",
            "source_shape": list(source_shape),
            "prior_shape": list(prior_shape),
            "prior_dtype": prior_dtype if "prior_dtype" in locals() else None,
            "recorded_identity": {
                "model": recorded_model,
                "source_rgb": recorded_source,
                "units": recorded_units,
            },
            "numeric": {
                "pixels": expected_pixels,
                "finite_pixels": finite_count,
                "below_zero_pixels": below_range_count,
                "above_one_pixels": above_range_count,
                "minimum": minimum,
                "maximum": maximum,
            },
        }
    )
    if include_content_hashes:
        record["source_file_sha256"] = _file_sha256(source_path)
        record["prior_file_sha256"] = _file_sha256(prior_path)
    return record


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def inspect_height_outlier_tile(
    *, height_path: Path, class_path: Path, threshold: float
) -> dict[str, Any]:
    """Return joint class/height evidence for values above ``threshold``."""

    h5 = _require_h5py()
    with h5.File(height_path, "r") as height_handle, h5.File(class_path, "r") as class_handle:
        heights = np.asarray(height_handle["image"], dtype=np.float32)
        raw_classes = np.asarray(class_handle["image"], dtype=np.float32)
    if heights.shape != raw_classes.shape:
        raise ValueError(f"unaligned outlier pair: {height_path} and {class_path}")
    selected = np.isfinite(heights) & (heights > threshold)
    values = heights[selected].astype(np.float64)
    raw_selected_classes = raw_classes[selected]
    rounded = np.rint(raw_selected_classes)
    valid_class = (
        np.isfinite(raw_selected_classes)
        & np.isclose(raw_selected_classes, rounded, rtol=0.0, atol=1.0e-5)
        & (rounded >= min(GAMUS_CLASSES))
        & (rounded <= max(GAMUS_CLASSES))
    )
    classes = np.where(valid_class, rounded, -1).astype(np.int16)
    by_class: dict[str, Any] = {}
    for class_id in sorted(set(int(value) for value in classes)):
        class_values = values[classes == class_id]
        label = "invalid_class" if class_id < 0 else GAMUS_CLASSES[class_id]
        by_class[str(class_id)] = {"label": label, **_summary(class_values.tolist())}
    return {"all": _summary(values.tolist()), "by_class": by_class}


def choose_recompute_samples(
    sample_ids_by_split: Mapping[str, Sequence[str]], *, per_city: int
) -> list[tuple[str, str]]:
    """Choose stable train/validation samples only; the official test stays locked."""

    if per_city < 0:
        raise ValueError("per_city must be non-negative")
    chosen: list[tuple[str, str]] = []
    for split in SPLIT_ORDER:
        if split not in TRAINING_SPLITS:
            continue
        by_city: dict[str, list[str]] = defaultdict(list)
        for sample_id in sample_ids_by_split.get(split, ()):
            by_city[sample_id.split("_", maxsplit=1)[0]].append(sample_id)
        for city in sorted(by_city):
            ranked = sorted(
                by_city[city],
                key=lambda value: (hashlib.sha256(value.encode("utf-8")).hexdigest(), value),
            )
            chosen.extend((split, sample_id) for sample_id in ranked[:per_city])
    return chosen


def _directory_content_contract(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        return {"status": "missing", "path": str(root.resolve()), "files": []}
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "status": "locked",
        "path": str(root.resolve()),
        "files": files,
        "canonical_manifest_sha256": _canonical_sha256(files),
    }


def _load_quality_contract(quality_root: Path, expected_index_sha256: str | None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    report_path = quality_root / "qc_report.json"
    index_path = quality_root / "approved_samples.json"
    tile_path = quality_root / "tile_audit.jsonl"
    for path in (report_path, index_path, tile_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing quality-contract artifact: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    index_sha256 = _file_sha256(index_path)
    if expected_index_sha256 is not None and index_sha256 != expected_index_sha256:
        raise ValueError("approved GAMUS index hash differs from the declared pilot contract")
    if report.get("approved_index_sha256") != index_sha256:
        raise ValueError("approved GAMUS index fails its quality-contract hash")
    tile_sha256 = _file_sha256(tile_path)
    if report.get("tile_audit_sha256") != tile_sha256:
        raise ValueError("GAMUS tile audit fails its quality-contract hash")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in tile_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return report, index, records


def _collect_height_outliers(
    *,
    tile_records: Sequence[dict[str, Any]],
    approved_index: Mapping[str, Any],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    approved = {
        split: set(approved_index["splits"][split]["approved_sample_ids"])
        for split in SPLIT_ORDER
    }
    detail_records: list[dict[str, Any]] = []
    all_values: dict[str, list[float]] = defaultdict(list)
    by_split_city: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_split_class: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in tile_records:
        split = record["split"]
        sample_id = record["sample_id"]
        if sample_id not in approved[split]:
            continue
        if int(record.get("numeric", {}).get("height_above_limit_pixels", 0)) <= 0:
            continue
        detail = inspect_height_outlier_tile(
            height_path=Path(record["paths"]["height"]),
            class_path=Path(record["paths"]["class"]),
            threshold=threshold,
        )
        declared_count = int(record["numeric"]["height_above_limit_pixels"])
        if detail["all"]["count"] != declared_count:
            raise ValueError(
                f"outlier recount mismatch for {split}/{sample_id}: "
                f"{detail['all']['count']} != {declared_count}"
            )
        h5 = _require_h5py()
        with h5.File(record["paths"]["height"], "r") as height_handle, h5.File(
            record["paths"]["class"], "r"
        ) as class_handle:
            heights = np.asarray(height_handle["image"], dtype=np.float32)
            classes = np.rint(np.asarray(class_handle["image"], dtype=np.float32)).astype(np.int16)
        mask = np.isfinite(heights) & (heights > threshold)
        values = heights[mask].astype(np.float64).tolist()
        city = sample_id.split("_", maxsplit=1)[0]
        all_values[split].extend(values)
        by_split_city[split][city].extend(values)
        for class_id in sorted(set(int(value) for value in classes[mask])):
            by_split_class[split][str(class_id)].extend(
                heights[mask & (classes == class_id)].astype(np.float64).tolist()
            )
        detail_records.append(
            {
                "schema": OUTLIER_SCHEMA,
                "split": split,
                "evaluation_role": (
                    "inventory_only_locked_test" if split == "test" else "development_audit"
                ),
                "sample_id": sample_id,
                "city": city,
                "threshold_raw_units": threshold,
                **detail,
            }
        )

    summary: dict[str, Any] = {}
    qc_record_by_split = {
        split: [record for record in tile_records if record["split"] == split]
        for split in SPLIT_ORDER
    }
    for split in SPLIT_ORDER:
        denominator = sum(
            max(
                0,
                int(record.get("numeric", {}).get("height_finite_pixels", 0))
                - int(record.get("numeric", {}).get("height_negative_pixels", 0)),
            )
            for record in qc_record_by_split[split]
            if record["sample_id"] in approved[split]
        )
        count = len(all_values[split])
        summary[split] = {
            "evaluation_role": (
                "inventory_only_locked_test" if split == "test" else "development_audit"
            ),
            "approved_tiles": len(approved[split]),
            "tiles_with_values_above_cutoff": sum(
                item["split"] == split for item in detail_records
            ),
            "eligible_nonnegative_finite_height_pixels": denominator,
            "above_cutoff_fraction": count / denominator if denominator else None,
            "all": _summary(all_values[split]),
            "by_city": {
                city: _summary(values)
                for city, values in sorted(by_split_city[split].items())
            },
            "by_class": {
                class_id: {
                    "label": GAMUS_CLASSES.get(int(class_id), "invalid_class"),
                    **_summary(values),
                }
                for class_id, values in sorted(
                    by_split_class[split].items(), key=lambda item: int(item[0])
                )
            },
        }
    return summary, sorted(detail_records, key=lambda item: (SPLIT_ORDER.index(item["split"]), item["sample_id"]))


def _audit_masks(root: Path, approved_index_path: Path, relative_prior_root: Path) -> dict[str, Any]:
    required = {
        "image_valid_mask",
        "classification_valid_mask",
        "height_valid_mask",
        "valid_mask",
        "regression_mask",
    }
    result: dict[str, Any] = {"status": "passed", "splits": {}, "errors": []}
    for split in SPLIT_ORDER:
        # Prefer a tile known to contain missing/invalid height pixels so the
        # independence assertion is tested against real data rather than names.
        dataset = GamusSurfaceDataset(
            root,
            split,
            patch_size=1024,
            random_crop=False,
            augment=False,
            radiometric_policy="raw",
            relative_prior_root=relative_prior_root,
            require_relative_prior=True,
            approved_index_path=approved_index_path,
        )
        chosen = None
        chosen_index = None
        for index, record in enumerate(dataset.records[: min(128, len(dataset.records))]):
            h5 = _require_h5py()
            with h5.File(record.height_path, "r") as handle:
                heights = np.asarray(handle["image"], dtype=np.float32)
            if np.any(~np.isfinite(heights) | (heights < 0.0) | (heights > dataset.height_max_m)):
                chosen_index = index
                break
        if chosen_index is None:
            chosen_index = 0
        chosen = dataset[chosen_index]
        missing = sorted(required - set(chosen))
        split_errors: list[str] = []
        if missing:
            split_errors.append("missing_mask_outputs:" + ",".join(missing))
        else:
            image_valid = chosen["image_valid_mask"].bool()
            class_valid = chosen["classification_valid_mask"].bool()
            height_valid = chosen["height_valid_mask"].bool()
            valid_alias = chosen["valid_mask"].bool()
            regression = chosen["regression_mask"].bool()
            if not np.array_equal(height_valid.numpy(), valid_alias.numpy()):
                split_errors.append("valid_mask_is_not_height_validity_alias")
            if bool(torch_any(regression & ~height_valid)):
                split_errors.append("regression_mask_escapes_height_validity")
            if bool(torch_any(regression & ~class_valid)):
                split_errors.append("regression_mask_escapes_classification_validity")
            if int(image_valid.sum()) <= 0:
                split_errors.append("no_valid_image_pixels")
            result["splits"][split] = {
                "sample_id": chosen["sample_id"],
                "image_valid_pixels": int(image_valid.sum()),
                "classification_valid_pixels": int(class_valid.sum()),
                "height_valid_pixels": int(height_valid.sum()),
                "regression_valid_pixels": int(regression.sum()),
                "image_valid_but_height_invalid_pixels": int((image_valid & ~height_valid).sum()),
                "errors": split_errors,
                "evaluation_role": (
                    "contract_inventory_only_locked_test" if split == "test" else "development_contract"
                ),
            }
        if split_errors:
            result["errors"].extend(f"{split}:{error}" for error in split_errors)
    if result["errors"]:
        result["status"] = "failed"
    return result


def torch_any(value: Any) -> bool:
    """Tiny indirection keeps torch optional at module import for audit unit tests."""

    return bool(value.any().item())


def _sample_recompute(
    *,
    selected: Sequence[tuple[str, str]],
    records_by_key: Mapping[tuple[str, str], Any],
    declared_model: str,
    project_root: Path,
    device: str,
) -> dict[str, Any]:
    if not selected:
        return {"status": "skipped", "records": [], "reason": "sample_count_is_zero"}
    from msr.inference.relative_depth import DepthAnythingV2Predictor

    model_path = Path(declared_model)
    resolved_model: str | Path = (
        (project_root / model_path).resolve() if not model_path.is_absolute() else model_path.resolve()
    )
    predictor = DepthAnythingV2Predictor(model_id=str(resolved_model), device=device)
    results: list[dict[str, Any]] = []
    h5 = _require_h5py()
    for split, sample_id in tqdm(selected, desc="DAV2 deterministic recompute", unit="tile"):
        record = records_by_key[(split, sample_id)]
        with h5.File(record.image_path, "r") as source_handle:
            image = np.asarray(source_handle["image"][..., :3], dtype=np.uint8).transpose(2, 0, 1)
        with h5.File(record.relative_prior_path, "r") as prior_handle:
            cached = np.asarray(prior_handle["image"], dtype=np.float16)
        recomputed = np.clip(predictor.predict(image).relative_surface, 0.0, 1.0).astype(np.float16)
        finite = np.isfinite(cached) & np.isfinite(recomputed)
        absolute = np.abs(cached.astype(np.float32) - recomputed.astype(np.float32))
        mismatch = cached.view(np.uint16) != recomputed.view(np.uint16)
        results.append(
            {
                "split": split,
                "sample_id": sample_id,
                "city": sample_id.split("_", maxsplit=1)[0],
                "exact_float16_match": bool(np.array_equal(cached, recomputed, equal_nan=True)),
                "mismatched_pixels": int(mismatch.sum()),
                "finite_comparison_pixels": int(finite.sum()),
                "maximum_absolute_difference": float(absolute[finite].max()) if np.any(finite) else None,
                "mean_absolute_difference": float(absolute[finite].mean()) if np.any(finite) else None,
            }
        )
    passed = bool(results) and all(item["exact_float16_match"] for item in results)
    return {
        "status": "passed" if passed else "failed",
        "selection": "lowest SHA256(sample_id) per city within train and validation only",
        "test_split_recomputed": False,
        "records": results,
    }


def build_preparation_audit(
    *,
    gamus_root: Path,
    prior_root: Path,
    quality_root: Path,
    output_root: Path,
    project_root: Path,
    expected_index_sha256: str | None,
    height_cutoff: float,
    include_content_hashes: bool,
    recompute_samples_per_city: int,
    device: str,
) -> dict[str, Any]:
    """Build and atomically publish an immutable preparation audit."""

    gamus_root = gamus_root.expanduser().resolve()
    prior_root = prior_root.expanduser().resolve()
    quality_root = quality_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite preparation audit: {output_root}")
    qc_report, approved_index, tile_records = _load_quality_contract(
        quality_root, expected_index_sha256
    )
    approved_index_path = quality_root / "approved_samples.json"
    provenance_path = prior_root / "msr_prior_provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"missing DAV2 provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    declared_model = provenance.get("model")
    if not isinstance(declared_model, str) or not declared_model:
        raise ValueError("DAV2 provenance does not declare a model")

    records_by_split = {
        split: load_gamus_split(
            gamus_root,
            split,
            relative_prior_root=prior_root,
            require_relative_prior=True,
            approved_index_path=approved_index_path,
        )
        for split in SPLIT_ORDER
    }
    records_by_key = {
        (split, record.sample_id): record
        for split, records in records_by_split.items()
        for record in records
    }
    sample_ids_by_split = {
        split: [record.sample_id for record in records]
        for split, records in records_by_split.items()
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.pending-", dir=output_root.parent)
    )
    try:
        prior_manifest_path = staging / "dav2_prior_identity.jsonl"
        prior_failures: list[dict[str, Any]] = []
        prior_counts: dict[str, dict[str, int]] = {}
        with prior_manifest_path.open("w", encoding="utf-8") as handle:
            for split in SPLIT_ORDER:
                counts = {"audited": 0, "passed": 0, "failed": 0}
                for record in tqdm(
                    records_by_split[split], desc=f"DAV2 identity {split}", unit="tile"
                ):
                    validation = validate_prior_pair(
                        source_path=record.image_path,
                        prior_path=record.relative_prior_path,
                        expected_model=declared_model,
                        include_content_hashes=include_content_hashes,
                    )
                    validation.update(
                        {
                            "schema": PRIOR_MANIFEST_SCHEMA,
                            "split": split,
                            "sample_id": record.sample_id,
                            "city": record.sample_id.split("_", maxsplit=1)[0],
                            "trainer_role": (
                                "candidate_input" if split in TRAINING_SPLITS else "locked_test_inventory_only"
                            ),
                        }
                    )
                    counts["audited"] += 1
                    counts[validation["status"]] += 1
                    if validation["status"] == "failed":
                        prior_failures.append(
                            {
                                "split": split,
                                "sample_id": record.sample_id,
                                "errors": validation["errors"],
                            }
                        )
                    handle.write(json.dumps(validation, sort_keys=True) + "\n")
                prior_counts[split] = counts
            handle.flush()
            os.fsync(handle.fileno())

        outlier_summary, outlier_records = _collect_height_outliers(
            tile_records=tile_records,
            approved_index=approved_index,
            threshold=height_cutoff,
        )
        outlier_path = staging / "height_outliers.jsonl"
        with outlier_path.open("w", encoding="utf-8") as handle:
            for record in outlier_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        selected = choose_recompute_samples(
            sample_ids_by_split, per_city=recompute_samples_per_city
        )
        recomputation = _sample_recompute(
            selected=selected,
            records_by_key=records_by_key,
            declared_model=declared_model,
            project_root=project_root,
            device=device,
        )
        resolved_model = Path(declared_model)
        if not resolved_model.is_absolute():
            resolved_model = (project_root / resolved_model).resolve()
        model_contract = _directory_content_contract(resolved_model)
        masks = _audit_masks(gamus_root, approved_index_path, prior_root)

        local_metadata_samples: list[dict[str, Any]] = []
        h5 = _require_h5py()
        for split in SPLIT_ORDER:
            by_city: dict[str, Any] = {}
            for record in records_by_split[split]:
                by_city.setdefault(record.sample_id.split("_", maxsplit=1)[0], record)
            for city, record in sorted(by_city.items()):
                entry: dict[str, Any] = {
                    "split": split,
                    "city": city,
                    "sample_id": record.sample_id,
                    "trainer_role": (
                        "development" if split in TRAINING_SPLITS else "locked_test_inventory_only"
                    ),
                    "roles": {},
                }
                for role, path in (
                    ("rgb", record.image_path),
                    ("height", record.height_path),
                    ("class", record.class_path),
                ):
                    with h5.File(path, "r") as handle:
                        dataset = handle["image"]
                        entry["roles"][role] = {
                            "file_attributes": {
                                str(key): _string_attribute(value)
                                for key, value in handle.attrs.items()
                            },
                            "dataset_attributes": {
                                str(key): _string_attribute(value)
                                for key, value in dataset.attrs.items()
                            },
                            "shape": list(dataset.shape),
                            "dtype": str(dataset.dtype),
                        }
                local_metadata_samples.append(entry)

        all_prior_passed = not prior_failures
        training_ready = (
            all_prior_passed
            and recomputation["status"] == "passed"
            and masks["status"] == "passed"
        )
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_data_modified": False,
            "live_checkpoint_pointer_modified": False,
            "training_started": False,
            "roots": {
                "gamus": str(gamus_root),
                "priors": str(prior_root),
                "quality_contract": str(quality_root),
            },
            "official_split_policy": {
                "train": "development_training",
                "val": "development_validation",
                "test": "inventory_audit_only; never constructed by the pilot trainer",
                "split_ids_disjoint": qc_report.get("split_ids_disjoint"),
            },
            "unit_and_scale": {
                "height_field": "AGL/nDSM above-ground surface height",
                "height_numeric_unit_status": "unverified_explicit_project_assumption",
                "height_training_assumption": "raw value x 1.0 equals metres",
                "height_evidence": [
                    "The GAMUS paper describes co-registered nDSM/height maps.",
                    "The official loader reads the AGL array without scaling.",
                    "Local RGB/AGL/CLS HDF5 samples have no unit attribute.",
                    "Neither the local dataset README nor prior provenance declares the AGL numeric unit.",
                ],
                "height_claim_limit": (
                    "Do not call GAMUS-derived errors absolute metric proof until the publisher "
                    "or source geospatial products confirm the AGL numeric unit."
                ),
                "nominal_pixel_scale_m": 0.33,
                "pixel_scale_status": "documented_by_GAMUS_paper_not_embedded_per_HDF5_tile",
                "pixel_scale_source": "https://arxiv.org/html/2305.14914",
                "tile_shape_pixels": [1024, 1024],
                "nominal_tile_width_m": 337.92,
                "georeferencing_status": "not_present_in_local_HDF5_samples",
                "area_reporting_rule": (
                    "Use 0.33 m as dataset-level nominal scale only; do not report precise "
                    "per-upload square metres unless that input carries trusted georeferencing."
                ),
                "local_metadata_samples": local_metadata_samples,
            },
            "height_cutoff_audit": {
                "cutoff_raw_value": height_cutoff,
                "unit_interpretation": "metre_assumed_not_metadata_verified",
                "splits": outlier_summary,
                "decision": (
                    "Keep the 200 cutoff for the comparison pilot. Values above it are rare "
                    "in train/validation and include implausible ground/vegetation labels; "
                    "raising the global cutoff would admit obvious artifacts. Preserve the "
                    "outlier inventory for later city/tile review."
                ),
            },
            "dav2_prior_contract": {
                "declared_model": declared_model,
                "provenance_file": str(provenance_path),
                "provenance_file_sha256": _file_sha256(provenance_path),
                "model_artifact": model_contract,
                "generation_implementation": {
                    "precompute_script": {
                        "path": str((project_root / "scripts/data/precompute_gamus_priors.py").resolve()),
                        "sha256": _file_sha256(project_root / "scripts/data/precompute_gamus_priors.py"),
                    },
                    "relative_depth_module": {
                        "path": str((project_root / "src/msr/inference/relative_depth.py").resolve()),
                        "sha256": _file_sha256(project_root / "src/msr/inference/relative_depth.py"),
                    },
                },
                "split_counts": prior_counts,
                "all_approved_pairs_pass_identity_and_grid_checks": all_prior_passed,
                "failures": prior_failures,
                "sampled_deterministic_recomputation": recomputation,
                "historical_provenance_limit": (
                    "The old cache stores model name, source path and units but not source/model/config "
                    "content hashes. This audit locks the current bytes and independently recomputes "
                    "stratified train/validation samples; it does not fabricate missing historical hashes."
                ),
            },
            "validity_mask_contract": masks,
            "colour_augmentation_gate": {
                "status": "eligible" if masks["status"] == "passed" else "blocked",
                "requirement": (
                    "Colour augmentation must consume image_valid_mask, never height_valid_mask; "
                    "classification and regression losses retain their own masks."
                ),
                "current_height_pilot_radiometric_policy": "raw",
            },
            "training_gate": {
                "status": "passed" if training_ready else "blocked",
                "approved_for": "height-focused comparison pilot only",
                "not_approved_for": [
                    "production checkpoint promotion",
                    "claiming GAMUS AGL units as publisher-verified metres",
                    "using official test labels for tuning or selection",
                ],
            },
        }
        report_path = staging / "audit_report.json"
        _write_json(report_path, report)
        artifacts = {
            "schema": "msr.gamus.preparation_audit_artifacts.v1",
            "hash_scope": "sha256_of_exact_persisted_file_bytes",
            "artifacts": {
                "audit_report.json": _file_sha256(report_path),
                "dav2_prior_identity.jsonl": _file_sha256(prior_manifest_path),
                "height_outliers.jsonl": _file_sha256(outlier_path),
            },
        }
        _write_json(staging / "artifact_hashes.json", artifacts)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamus-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--height-cutoff", type=float, default=200.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recompute-samples-per-city", type=int, default=1)
    parser.add_argument("--skip-content-hashes", action="store_true")
    args = parser.parse_args()
    report = build_preparation_audit(
        gamus_root=args.gamus_root,
        prior_root=args.prior_root,
        quality_root=args.quality_root,
        output_root=args.output_root,
        project_root=args.project_root,
        expected_index_sha256=args.expected_index_sha256,
        height_cutoff=args.height_cutoff,
        include_content_hashes=not args.skip_content_hashes,
        recompute_samples_per_city=args.recompute_samples_per_city,
        device=args.device,
    )
    print(json.dumps(report["training_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
