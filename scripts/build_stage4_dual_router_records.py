"""Build authenticated, resumable Stage-4 dual-router rich records on CUDA.

The Stage-3 generator and its saved journal remain byte-for-byte frozen.  This
versioned path reuses its authenticated dataset plan and endpoint loader, takes
descriptors from the authenticated Stage-3 consolidated records, and recomputes
the richer per-endpoint component evidence required by Stage 4.

Official test splits are never configured or opened.  To prevent GAMUS city
leakage, controller fit keeps only NYC from the original training split;
original-train DC/PHL rows are recorded in a distinct policy-exclusion ledger,
while original-validation DC/PHL rows remain calibration-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import platform
import random
import re
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

# Importing these helpers preserves the exact Stage-3 data and endpoint
# contracts instead of copying them into a subtly divergent implementation.
import scripts.build_stage3_scene_router_records as stage3_generation
from msr.models.stage3_endpoints import load_stage3_endpoint_packs
from msr.training.scene_router import (
    SceneRouterRecord,
    SceneUtilityConfig,
    read_scene_router_records,
)
from msr.training.scene_router_record_store import (
    RECORD_STORE_SCHEMA as STAGE3_RECORD_STORE_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    validate_record_selection_provenance,
)
from msr.training.stage4_dual_router import (
    STAGE4_RECORD_SCHEMA,
    Stage4DualRouterRecord,
    build_stage4_dual_router_record,
)
from msr.training.stage4_dual_router_record_store import (
    AtomicStage4DualRouterRecordStore,
    build_stage4_record_store_provenance,
    validate_stage4_record_selection_provenance,
)


STAGE4_DATA_PROVENANCE_SCHEMA = "msr.stage4_dual_router_data.v1"
STAGE4_IMPLEMENTATION_SCHEMA = "msr.stage4_record_generation.v1"
OPEN_CANOPY_REGION_PATTERN = re.compile(r"^open_canopy_r\d+_\d+$")
SAFE_GAMUS_TRAIN_CITIES = ("NYC",)
SAFE_GAMUS_CALIBRATION_CITIES = ("DC", "PHL")
EXCLUDED_GAMUS_TRAIN_CITIES = ("DC", "PHL")
STAGE3_AGGREGATE_REL_TOL = 0.0
STAGE3_AGGREGATE_ABS_TOL = 1.0e-12

CRITICAL_IMPLEMENTATION_FILES = tuple(
    dict.fromkeys(
        (
            "scripts/build_stage4_dual_router_records.py",
            "src/msr/training/stage4_dual_router.py",
            "src/msr/training/stage4_dual_router_record_store.py",
            *stage3_generation.CRITICAL_IMPLEMENTATION_FILES,
        )
    )
)


@dataclass(frozen=True)
class Stage4DatasetPlan:
    source: str
    split: str
    partition: str
    dataset: Dataset
    stage3_scored_indices: tuple[int, ...]
    original_indices: tuple[int, ...]
    original_sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]

    @property
    def record_keys(self) -> tuple[str, ...]:
        return tuple(
            stage3_generation.scene_router_record_key(
                self.source, self.split, sample_id
            )
            for sample_id in self.original_sample_ids
        )


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_sha256(value: object, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return digest


def validate_generation_config(config: Mapping[str, Any]) -> None:
    """Validate the inherited Stage-3 contract and the stricter Stage-4 policy."""

    stage3_generation.validate_generation_config(config)
    if int(config["record_generation"]["batch_size"]) != 4:
        raise ValueError(
            "Stage-4 must replay the authenticated Stage-3 batch size of 4"
        )
    ancestry = _require_mapping(config.get("stage3_ancestry"), "stage3_ancestry")
    selection = _require_mapping(
        config.get("controller_selection"), "controller_selection"
    )
    safety = _require_mapping(config.get("safety"), "safety")
    required_ancestry = {
        "provenance_path",
        "provenance_sha256",
        "completion_path",
        "completion_sha256",
        "records_path",
        "records_sha256",
        "config_sha256",
        "record_count",
    }
    missing = sorted(required_ancestry - set(ancestry))
    if missing:
        raise ValueError(f"stage3_ancestry is missing fields: {missing}")
    for name in (
        "provenance_sha256",
        "completion_sha256",
        "records_sha256",
        "config_sha256",
    ):
        _require_sha256(ancestry[name], f"stage3_ancestry.{name}")
    if int(ancestry["record_count"]) != 7184:
        raise ValueError("Stage-4 requires the authenticated 7,184 Stage-3 records")

    required_selection = {
        "gamus_train_cities",
        "gamus_calibration_cities",
        "excluded_gamus_train_cities",
        "expected_excluded_group_policy_count",
        "expected_selected_record_count",
        "expected_selected_partition_counts",
        "expected_selected_source_split_counts",
    }
    missing = sorted(required_selection - set(selection))
    if missing:
        raise ValueError(f"controller_selection is missing fields: {missing}")

    def cities(name: str) -> tuple[str, ...]:
        value = selection[name]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"controller_selection.{name} must be a string list")
        return tuple(item.strip().upper() for item in value)

    if cities("gamus_train_cities") != SAFE_GAMUS_TRAIN_CITIES:
        raise ValueError("Stage-4 controller fit must use GAMUS NYC train only")
    if cities("gamus_calibration_cities") != SAFE_GAMUS_CALIBRATION_CITIES:
        raise ValueError("Stage-4 calibration must use GAMUS DC/PHL val only")
    if cities("excluded_gamus_train_cities") != EXCLUDED_GAMUS_TRAIN_CITIES:
        raise ValueError("Stage-4 must explicitly exclude GAMUS DC/PHL train")
    if int(selection["expected_excluded_group_policy_count"]) != 3837:
        raise ValueError("expected GAMUS group-policy exclusion count must be 3,837")
    if int(selection["expected_selected_record_count"]) != 3347:
        raise ValueError("expected Stage-4 rich-record count must be 3,347")
    if selection["expected_selected_partition_counts"] != {
        "train": 2208,
        "calibration": 1139,
    }:
        raise ValueError("expected Stage-4 partition counts have changed")
    if selection["expected_selected_source_split_counts"] != {
        "gamus/train": 1158,
        "gamus/val": 859,
        "legacy/train": 1050,
        "legacy/val": 280,
    }:
        raise ValueError("expected Stage-4 source/split counts have changed")

    if set(safety) != {
        "app_pointer",
        "app_pointer_sha256",
        "expected_protected_checkpoint",
    }:
        raise ValueError("safety must lock only the protected app pointer")
    _require_sha256(safety["app_pointer_sha256"], "safety.app_pointer_sha256")


def _read_json_mapping(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def authenticate_stage3_ancestry(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[SceneRouterRecord]]:
    """Authenticate the frozen Stage-3 proof and consolidated descriptor rows."""

    ancestry = config["stage3_ancestry"]
    paths = {
        name: _resolve(ancestry[f"{name}_path"])
        for name in ("provenance", "completion", "records")
    }
    for name, path in paths.items():
        expected = _require_sha256(
            ancestry[f"{name}_sha256"], f"stage3_ancestry.{name}_sha256"
        )
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Stage-3 {name} SHA-256 mismatch: expected {expected}, found {actual}"
            )

    provenance = _read_json_mapping(paths["provenance"], "Stage-3 provenance")
    completion = _read_json_mapping(paths["completion"], "Stage-3 completion")
    expected_config_hash = _require_sha256(
        ancestry["config_sha256"], "stage3_ancestry.config_sha256"
    )
    recomputed_config_hash = canonical_json_sha256(
        {
            "generation_config": provenance.get("generation_config"),
            "endpoints": provenance.get("endpoints"),
            "shared_model": provenance.get("shared_model"),
            "data": provenance.get("data"),
        }
    )
    if (
        provenance.get("schema") != STAGE3_RECORD_STORE_SCHEMA
        or provenance.get("record_schema") is not None
        or provenance.get("config_sha256") != expected_config_hash
        or recomputed_config_hash != expected_config_hash
        or provenance.get("test_splits_excluded") is not True
    ):
        raise ValueError("Stage-3 provenance identity is invalid")
    validate_record_selection_provenance(provenance.get("data"))
    expected_count = int(ancestry["record_count"])
    if (
        completion.get("schema") != STAGE3_RECORD_STORE_SCHEMA
        or completion.get("complete") is not True
        or completion.get("config_sha256") != expected_config_hash
        or completion.get("record_count") != expected_count
        or completion.get("records_sha256")
        != _require_sha256(ancestry["records_sha256"], "stage3 records hash")
        or Path(str(completion.get("records_path", ""))).resolve()
        != paths["records"]
    ):
        raise ValueError("Stage-3 completion proof is invalid")
    validate_record_selection_provenance(
        provenance["data"], completed_record_count=expected_count
    )

    recorded_implementation = provenance.get("generation_config", {}).get(
        "implementation", {}
    )
    recorded_files = recorded_implementation.get("files", {})
    if not isinstance(recorded_files, Mapping) or not recorded_files:
        raise ValueError("Stage-3 provenance has no implementation file hashes")
    for relative, expected in recorded_files.items():
        path = (PROJECT_ROOT / str(relative)).resolve(strict=True)
        if file_sha256(path) != expected:
            raise ValueError(
                f"frozen Stage-3 implementation changed: {relative}"
            )

    records = read_scene_router_records(paths["records"])
    if len(records) != expected_count:
        raise ValueError("Stage-3 consolidated record count changed")
    record_keys = [record.sample_id for record in records]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("Stage-3 consolidated records contain duplicate keys")
    authenticated = {
        "authenticated": True,
        "schema": STAGE3_RECORD_STORE_SCHEMA,
        "config_sha256": expected_config_hash,
        "provenance_path": str(paths["provenance"]),
        "provenance_sha256": file_sha256(paths["provenance"]),
        "provenance_canonical_sha256": canonical_json_sha256(provenance),
        "completion_path": str(paths["completion"]),
        "completion_sha256": file_sha256(paths["completion"]),
        "completion_canonical_sha256": canonical_json_sha256(completion),
        "records_path": str(paths["records"]),
        "records_sha256": file_sha256(paths["records"]),
        "record_count": len(records),
        "record_keys_sha256": canonical_json_sha256(record_keys),
        "record_key_set_sha256": canonical_json_sha256(sorted(record_keys)),
    }
    return authenticated, provenance, records


def _gamus_city(sample_id: str, region: str) -> str:
    components = sample_id.split("_", maxsplit=1)
    if len(components) != 2 or not all(components):
        raise ValueError(f"GAMUS sample has no canonical city prefix: {sample_id!r}")
    city = components[0].upper()
    if region.strip().upper() != f"GAMUS_{city}":
        raise ValueError(
            f"GAMUS region disagrees with sample city: {sample_id!r}, {region!r}"
        )
    return city


def controller_group_id(
    *, source: str, sample_id: str, region: str, landscape: str
) -> str:
    """Return the authenticated geographic group for one source record."""

    source = source.strip().lower()
    region = region.strip()
    landscape = landscape.strip().lower()
    if source == "gamus":
        return f"gamus:{_gamus_city(sample_id, region).lower()}"
    if source != "legacy":
        raise ValueError(f"unsupported Stage-4 source: {source!r}")
    normalized_region = region.lower()
    if OPEN_CANOPY_REGION_PATTERN.fullmatch(normalized_region):
        if landscape != "forest" or not sample_id.lower().startswith("oc_"):
            raise ValueError("OpenCanopy group metadata is inconsistent")
        # The manifest region is already the canonical 5 km EPSG:2154 block.
        return f"opencanopy:{normalized_region}"
    if landscape != "urban" or sample_id.lower().startswith("oc_"):
        raise ValueError(
            "legacy non-OpenCanopy records must identify an exact HighBuild city"
        )
    if not region:
        raise ValueError("HighBuild record has no exact city/region")
    return f"highbuild:{normalized_region}"


def _record_metadata(record: object) -> tuple[str, str, str]:
    try:
        return (
            str(getattr(record, "sample_id")),
            str(getattr(record, "region")),
            str(getattr(record, "landscape", "mixed")),
        )
    except (TypeError, ValueError) as error:  # pragma: no cover - defensive
        raise ValueError("dataset source record has invalid metadata") from error


def create_stage4_dataset_plans(
    config: Mapping[str, Any],
    stage3_plans: Sequence[stage3_generation.RouterDatasetPlan],
) -> tuple[tuple[Stage4DatasetPlan, ...], dict[str, Any]]:
    """Apply the explicit, leakage-safe controller selection to Stage-3 plans."""

    no_reference = set(stage3_generation.configured_no_reference_exclusions(config))
    seen_no_reference: list[str] = []
    group_policy_exclusions: list[str] = []
    plans: list[Stage4DatasetPlan] = []
    all_planned_keys: list[str] = []

    for original in stage3_plans:
        records = getattr(original.dataset, "records", None)
        if not isinstance(records, Sequence) or len(records) != len(original.dataset):
            raise ValueError("Stage-4 datasets must expose one source record per item")
        indices: list[int] = []
        stage3_scored_indices: list[int] = []
        sample_ids: list[str] = []
        group_ids: list[str] = []
        for index, source_record in enumerate(records):
            sample_id, region, landscape = _record_metadata(source_record)
            key = stage3_generation.scene_router_record_key(
                original.source, original.split, sample_id
            )
            all_planned_keys.append(key)
            group_id = controller_group_id(
                source=original.source,
                sample_id=sample_id,
                region=region,
                landscape=landscape,
            )
            if key in no_reference:
                seen_no_reference.append(key)
                continue
            stage3_scored_indices.append(index)
            include = True
            if original.source == "gamus":
                city = _gamus_city(sample_id, region)
                if original.split == "train":
                    if city in EXCLUDED_GAMUS_TRAIN_CITIES:
                        group_policy_exclusions.append(key)
                        include = False
                    elif city not in SAFE_GAMUS_TRAIN_CITIES:
                        raise ValueError(f"unreviewed GAMUS train city: {city}")
                elif original.split == "val":
                    if city not in SAFE_GAMUS_CALIBRATION_CITIES:
                        raise ValueError(f"unreviewed GAMUS calibration city: {city}")
                else:
                    raise ValueError("official GAMUS test data is forbidden")
            if include:
                indices.append(index)
                sample_ids.append(sample_id)
                group_ids.append(group_id)
        if indices:
            plans.append(
                Stage4DatasetPlan(
                    source=original.source,
                    split=original.split,
                    partition=original.partition,
                    dataset=original.dataset,
                    stage3_scored_indices=tuple(stage3_scored_indices),
                    original_indices=tuple(indices),
                    original_sample_ids=tuple(sample_ids),
                    group_ids=tuple(group_ids),
                )
            )

    if len(all_planned_keys) != len(set(all_planned_keys)):
        raise ValueError("Stage-4 inherited data plan contains duplicate keys")
    if set(seen_no_reference) != no_reference or len(seen_no_reference) != len(
        no_reference
    ):
        raise ValueError("Stage-4 no-reference exclusions do not match the data plan")
    selected_keys = [key for plan in plans for key in plan.record_keys]
    if set(selected_keys) & set(group_policy_exclusions):
        raise ValueError("Stage-4 selected and policy-excluded rows overlap")
    partition_counts = dict(Counter(plan.partition for plan in plans for _ in plan.record_keys))
    source_split_counts = {
        f"{source}/{split}": sum(
            len(plan.record_keys)
            for plan in plans
            if plan.source == source and plan.split == split
        )
        for source, split, _ in stage3_generation.ALLOWED_SOURCE_SPLITS
    }
    group_keys_by_partition = {
        partition: sorted(
            {
                f"{plan.source}/{group_id}"
                for plan in plans
                if plan.partition == partition
                for group_id in plan.group_ids
            }
        )
        for partition in ("train", "calibration")
    }
    overlap = set(group_keys_by_partition["train"]) & set(
        group_keys_by_partition["calibration"]
    )
    if overlap:
        raise ValueError(
            "Stage-4 geographic groups cross train/calibration: "
            + ", ".join(sorted(overlap)[:5])
        )

    expected = config["controller_selection"]
    checks = (
        len(group_policy_exclusions)
        == int(expected["expected_excluded_group_policy_count"]),
        len(selected_keys) == int(expected["expected_selected_record_count"]),
        partition_counts == expected["expected_selected_partition_counts"],
        source_split_counts == expected["expected_selected_source_split_counts"],
    )
    if not all(checks):
        raise ValueError(
            "Stage-4 controller selection counts differ from the reviewed config: "
            f"selected={len(selected_keys)}, policy_excluded="
            f"{len(group_policy_exclusions)}, partitions={partition_counts}, "
            f"source_splits={source_split_counts}"
        )
    selection = {
        "planned_record_count": len(all_planned_keys),
        "stage3_scored_record_count": len(all_planned_keys) - len(no_reference),
        "excluded_no_reference_count": len(no_reference),
        "excluded_no_reference_record_keys": sorted(no_reference),
        "excluded_group_policy_count": len(group_policy_exclusions),
        "excluded_group_policy_record_keys": group_policy_exclusions,
        "excluded_reason_counts": {
            "no_reference": len(no_reference),
            "gamus_train_city_group_overlap": len(group_policy_exclusions),
        },
        "selected_record_count": len(selected_keys),
        "selected_record_keys_sha256": canonical_json_sha256(selected_keys),
        "stage3_scored_record_key_set_sha256": canonical_json_sha256(
            sorted([*selected_keys, *group_policy_exclusions])
        ),
        "selected_partition_counts": partition_counts,
        "selected_source_split_counts": source_split_counts,
        "group_keys_by_partition": group_keys_by_partition,
        "policy": (
            "preserve original endpoint train/val roles; fit GAMUS NYC train only; "
            "exclude reference-bearing GAMUS DC/PHL train rows whose city groups "
            "occur in calibration; calibrate on GAMUS DC/PHL val; preserve legacy "
            "train/val with exact HighBuild city and OpenCanopy 5km region groups"
        ),
    }
    return tuple(plans), selection


def stage3_replay_batches(
    plan: Stage4DatasetPlan,
    remaining_keys: set[str],
    *,
    batch_size: int,
) -> list[tuple[tuple[int, ...], tuple[str, ...]]]:
    """Select complete original Stage-3 batches containing unfinished rows.

    CUDA kernels can choose a different deterministic path for a partial batch.
    Replaying the Stage-3 scored-index boundaries keeps the endpoint evidence
    byte-stable even though Stage 4 stores only the leakage-safe subset.
    """

    if batch_size <= 0:
        raise ValueError("Stage-3 replay batch size must be positive")
    selected_by_index = dict(zip(plan.original_indices, plan.record_keys, strict=True))
    if len(selected_by_index) != len(plan.original_indices):
        raise ValueError("Stage-4 plan contains duplicate source indices")
    batches: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    for start in range(0, len(plan.stage3_scored_indices), batch_size):
        indices = plan.stage3_scored_indices[start : start + batch_size]
        unfinished = tuple(
            selected_by_index[index]
            for index in indices
            if index in selected_by_index
            and selected_by_index[index] in remaining_keys
        )
        if unfinished:
            batches.append((indices, unfinished))
    covered = [key for _, keys in batches for key in keys]
    expected = [key for key in plan.record_keys if key in remaining_keys]
    if covered != expected:
        raise ValueError("Stage-3 replay batches do not cover unfinished Stage-4 rows")
    return batches


def build_data_provenance(
    config: Mapping[str, Any],
    stage3_plans: Sequence[stage3_generation.RouterDatasetPlan],
    selection: Mapping[str, Any],
    authenticated_stage3_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-fingerprint live data and require exact Stage-3 data ancestry."""

    current = stage3_generation.build_data_provenance(config, stage3_plans)
    inherited = authenticated_stage3_provenance.get("data")
    if canonical_json_sha256(current) != canonical_json_sha256(inherited):
        raise ValueError("live data provenance differs from authenticated Stage-3 data")
    value = {
        "schema": STAGE4_DATA_PROVENANCE_SCHEMA,
        "stage3_data_provenance": current,
        "controller_selection": json.loads(json.dumps(selection)),
    }
    validate_stage4_record_selection_provenance(value)
    return value


def implementation_fingerprint() -> dict[str, Any]:
    files = {
        relative: file_sha256((PROJECT_ROOT / relative).resolve(strict=True))
        for relative in CRITICAL_IMPLEMENTATION_FILES
    }
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": importlib_metadata.version("torchvision"),
        "timm": importlib_metadata.version("timm"),
        "numpy": str(np.__version__),
        "rasterio": importlib_metadata.version("rasterio"),
        "h5py": importlib_metadata.version("h5py"),
        "cuda_runtime": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
    }
    return {
        "schema": STAGE4_IMPLEMENTATION_SCHEMA,
        "files": files,
        "runtime": runtime,
        "combined_sha256": canonical_json_sha256(
            {"files": files, "runtime": runtime}
        ),
    }


def _app_pointer_snapshot(config: Mapping[str, Any]) -> dict[str, str]:
    safety = config["safety"]
    pointer = _resolve(safety["app_pointer"])
    actual_hash = file_sha256(pointer)
    expected_hash = _require_sha256(
        safety["app_pointer_sha256"], "safety.app_pointer_sha256"
    )
    if actual_hash != expected_hash:
        raise ValueError("protected app pointer SHA-256 changed")
    # The protected pointer is a PowerShell-authored UTF-8 file and may retain
    # a BOM; decode it without changing the byte-level SHA-256 lock above.
    content = pointer.read_text(encoding="utf-8-sig").strip()
    expected_checkpoint = _resolve(safety["expected_protected_checkpoint"])
    if Path(content).expanduser().resolve() != expected_checkpoint:
        raise ValueError("app pointer does not select the protected endpoint")
    if file_sha256(expected_checkpoint) != str(
        config["endpoints"]["protected_checkpoint_sha256"]
    ).lower():
        raise ValueError("app pointer target is not the authenticated protected endpoint")
    return {
        "path": str(pointer),
        "sha256": actual_hash,
        "content": content,
    }


def _normalised_generation_identity(
    config: Mapping[str, Any], *, app_pointer: Mapping[str, str]
) -> dict[str, Any]:
    generation = config["record_generation"]
    data = config["data"]
    utility = config["utility"]
    return {
        "seed": int(generation["seed"]),
        "patch_size": int(generation["patch_size"]),
        "crop": "center",
        "batch_size": int(generation["batch_size"]),
        "num_workers": int(generation["num_workers"]),
        "precision": "bf16",
        "deterministic_algorithms": True,
        "allow_tf32": False,
        "rgb_scale": float(data["rgb_scale"]),
        "height_max_m": float(data["height_max_m"]),
        "building_threshold_m": float(data["building_threshold_m"]),
        "radiometric_policy": "raw",
        "legacy_relative_prior_policy": "stored_01",
        "descriptor_mask": "full_center_crop_all_pixels_no_reference_mask",
        "descriptor_source": "authenticated_stage3_records.jsonl",
        "height_evidence": "float64_sse+count+rmse_on_dataset_regression_mask",
        "semantic_evidence": (
            "reference_row_prediction_column_3x3_confusion+count+macro_error"
        ),
        "group_id_policy": (
            "GAMUS city; HighBuild exact city; OpenCanopy existing EPSG:2154 5km region"
        ),
        "controller_split_policy": config["controller_selection"],
        "app_pointer": dict(app_pointer),
        "implementation": implementation_fingerprint(),
        "utility": {
            "height_rmse_weight": float(utility.get("height_rmse_weight", 1.0)),
            "semantic_error_weight": float(
                utility.get("semantic_error_weight", 1.0)
            ),
            "height_scale_m": float(utility.get("height_scale_m", 10.0)),
            "minimum_candidate_gain": float(
                utility.get("minimum_candidate_gain", 0.02)
            ),
        },
    }


def _cpu_endpoint_batch(
    endpoint: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "height": endpoint["height"].detach().float().cpu(),
        "domain_logits": endpoint["domain_logits"].detach().float().cpu(),
    }


def build_records_from_batch(
    *,
    batch: Mapping[str, Any],
    source: str,
    split: str,
    partition: str,
    shared_model: torch.nn.Module,
    protected_endpoint: torch.nn.Module,
    candidate_endpoint: torch.nn.Module,
    stage3_records: Mapping[str, SceneRouterRecord],
    utility_config: SceneUtilityConfig,
    device: torch.device,
    precision: str,
) -> list[Stage4DualRouterRecord]:
    """Evaluate one batch while inheriting authenticated Stage-3 descriptors."""

    if precision != "bf16":
        raise ValueError("Stage-4 record generation supports bf16 only")
    image = batch["image"].to(device, non_blocking=True)
    prior = batch["relative_prior"].to(device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        shared = shared_model(image, prior)
        required = (
            "adapter_features",
            "prior_domain_logits",
            "base_height",
            "protected_building_logits",
            "refinement_logits",
            "log_variance",
        )
        missing = [name for name in required if name not in shared]
        if missing:
            raise RuntimeError(f"shared model is missing routed tensors: {missing}")
        common = {
            "protected_building_logits": shared["protected_building_logits"],
            "refinement_logits": shared["refinement_logits"],
            "log_variance": shared["log_variance"],
        }
        protected = protected_endpoint.forward_fused(
            shared["adapter_features"],
            shared["prior_domain_logits"],
            shared["base_height"],
            **common,
        )
        candidate = candidate_endpoint.forward_fused(
            shared["adapter_features"],
            shared["prior_domain_logits"],
            shared["base_height"],
            **common,
        )
    for name in ("height", "domain_logits", "building_height", "canopy_height"):
        if not torch.equal(protected[name], shared[name]):
            maximum_difference = float(
                torch.max(torch.abs(protected[name].float() - shared[name].float()))
            )
            raise RuntimeError(
                f"protected fused endpoint drifted from shared model at {name}: "
                f"max_abs={maximum_difference}"
            )

    protected_cpu = _cpu_endpoint_batch(protected)
    candidate_cpu = _cpu_endpoint_batch(candidate)
    sample_ids = [str(value) for value in batch["sample_id"]]
    regions = [str(value) for value in batch["region"]]
    landscapes = [str(value) for value in batch["landscape"]]
    records: list[Stage4DualRouterRecord] = []
    for index, original_id in enumerate(sample_ids):
        key = stage3_generation.scene_router_record_key(source, split, original_id)
        if key not in stage3_records:
            raise ValueError(f"Stage-4 row has no authenticated Stage-3 record: {key}")
        inherited = stage3_records[key]
        group_id = controller_group_id(
            source=source,
            sample_id=original_id,
            region=regions[index],
            landscape=landscapes[index],
        )
        record = build_stage4_dual_router_record(
            sample_id=key,
            descriptor=inherited.descriptor,
            protected_endpoint={
                name: value[index] for name, value in protected_cpu.items()
            },
            candidate_endpoint={
                name: value[index] for name, value in candidate_cpu.items()
            },
            source=source,
            group_id=group_id,
            landscape=landscapes[index],
            partition=partition,
            height_target=batch["height"][index],
            valid_mask=batch["regression_mask"][index],
            domain_target=batch["domain_target"][index],
            domain_valid_mask=batch["domain_valid_mask"][index],
            utility_config=utility_config,
        )
        protected_delta = (
            record.protected.aggregate_utility - inherited.fallback_utility
        )
        candidate_delta = (
            record.candidate.aggregate_utility - inherited.candidate_utility
        )
        metadata_matches = (
            record.source == inherited.source
            and record.landscape == inherited.landscape
            and record.partition == inherited.partition
        )
        aggregates_match = math.isclose(
            record.protected.aggregate_utility,
            inherited.fallback_utility,
            rel_tol=STAGE3_AGGREGATE_REL_TOL,
            abs_tol=STAGE3_AGGREGATE_ABS_TOL,
        ) and math.isclose(
            record.candidate.aggregate_utility,
            inherited.candidate_utility,
            rel_tol=STAGE3_AGGREGATE_REL_TOL,
            abs_tol=STAGE3_AGGREGATE_ABS_TOL,
        )
        if not metadata_matches or not aggregates_match:
            raise ValueError(
                "Stage-4 rich evidence disagrees with authenticated Stage-3 row: "
                f"{key}; metadata_matches={metadata_matches}; "
                f"protected_stage3={inherited.fallback_utility:.17g}; "
                f"protected_stage4={record.protected.aggregate_utility:.17g}; "
                f"protected_delta={protected_delta:.17g}; "
                f"candidate_stage3={inherited.candidate_utility:.17g}; "
                f"candidate_stage4={record.candidate.aggregate_utility:.17g}; "
                f"candidate_delta={candidate_delta:.17g}"
            )
        records.append(record)
    return records


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = _resolve(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Stage-4 record config must be a mapping")
    validate_generation_config(config)
    app_pointer_before = _app_pointer_snapshot(config)
    seed = int(config["record_generation"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ancestry, stage3_provenance, inherited_records = authenticate_stage3_ancestry(
        config
    )
    inherited_by_key = {record.sample_id: record for record in inherited_records}
    endpoints_config = config["endpoints"]
    endpoint_bundle = load_stage3_endpoint_packs(
        _resolve(endpoints_config["protected_checkpoint"]),
        _resolve(endpoints_config["gamus_stage1_checkpoint"]),
        expected_protected_sha256=str(
            endpoints_config["protected_checkpoint_sha256"]
        ),
        expected_gamus_stage1_sha256=str(
            endpoints_config["gamus_stage1_checkpoint_sha256"]
        ),
        expected_base_sha256=str(endpoints_config["base_checkpoint_sha256"]),
        project_root=PROJECT_ROOT,
    )
    shared_model = endpoint_bundle.require_shared_model().eval()
    protected_endpoint = endpoint_bundle.protected.eval()
    candidate_endpoint = endpoint_bundle.gamus_stage1.eval()
    for module in (shared_model, protected_endpoint, candidate_endpoint):
        module.requires_grad_(False)

    stage3_plans = stage3_generation.create_dataset_plans(config)
    plans, selection = create_stage4_dataset_plans(config, stage3_plans)
    data_provenance = build_data_provenance(
        config, stage3_plans, selection, stage3_provenance
    )
    expected_stage3_keys = [
        key
        for plan in stage3_plans
        for key in plan.record_keys
        if key
        not in set(stage3_provenance["data"]["excluded_no_reference_record_keys"])
    ]
    if [record.sample_id for record in inherited_records] != expected_stage3_keys:
        raise ValueError("authenticated Stage-3 record order differs from live data plan")
    expected_keys = [key for plan in plans for key in plan.record_keys]
    if not set(expected_keys) <= set(inherited_by_key):
        raise ValueError("Stage-4 selection contains rows absent from Stage-3 records")

    diagnostics = endpoint_bundle.diagnostics.as_dict()
    endpoint_provenance = {
        "protected": diagnostics["protected"],
        "gamus_stage1": diagnostics["gamus_stage1"],
        "compatibility": diagnostics["compatibility"],
    }
    shared_provenance = {
        "base_checkpoint": diagnostics["base_checkpoint"],
        "shared_state_sha256": diagnostics["compatibility"][
            "shared_state_sha256"
        ],
    }
    if canonical_json_sha256(endpoint_provenance) != canonical_json_sha256(
        stage3_provenance["endpoints"]
    ) or canonical_json_sha256(shared_provenance) != canonical_json_sha256(
        stage3_provenance["shared_model"]
    ):
        raise ValueError("loaded endpoints differ from authenticated Stage-3 endpoints")
    provenance = build_stage4_record_store_provenance(
        generation_config=_normalised_generation_identity(
            config, app_pointer=app_pointer_before
        ),
        endpoints={
            **endpoint_provenance,
            "shared_model": shared_provenance,
        },
        data=data_provenance,
        stage3=ancestry,
    )

    utility_config = SceneUtilityConfig(
        height_rmse_weight=float(
            config["utility"].get("height_rmse_weight", 1.0)
        ),
        semantic_error_weight=float(
            config["utility"].get("semantic_error_weight", 1.0)
        ),
        height_scale_m=float(config["utility"].get("height_scale_m", 10.0)),
    )
    generation = config["record_generation"]
    device = torch.device("cuda")
    output_dir = _resolve(generation["output_dir"])
    with AtomicStage4DualRouterRecordStore(output_dir, provenance) as store:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real Stage-4 record generation")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the active CUDA GPU does not support bf16")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        shared_model.to(device)
        protected_endpoint.to(device)
        candidate_endpoint.to(device)
        remaining = set(store.validate_expected_keys(expected_keys))
        progress = tqdm(
            total=len(expected_keys),
            initial=store.record_count,
            desc="Stage-4 rich records",
            unit="scene",
            dynamic_ncols=True,
        )
        try:
            for plan in plans:
                batch_size = int(generation["batch_size"])
                replay_batches = stage3_replay_batches(
                    plan, remaining, batch_size=batch_size
                )
                if not replay_batches:
                    continue
                replay_indices = [
                    index for indices, _ in replay_batches for index in indices
                ]
                workers = int(generation["num_workers"])
                loader = DataLoader(
                    Subset(plan.dataset, replay_indices),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=True,
                    persistent_workers=workers > 0,
                    drop_last=False,
                )
                if len(loader) != len(replay_batches):
                    raise RuntimeError("Stage-3 replay batch boundaries collapsed")
                for batch, (_, unfinished_keys) in zip(
                    loader, replay_batches, strict=True
                ):
                    replayed_records = build_records_from_batch(
                        batch=batch,
                        source=plan.source,
                        split=plan.split,
                        partition=plan.partition,
                        shared_model=shared_model,
                        protected_endpoint=protected_endpoint,
                        candidate_endpoint=candidate_endpoint,
                        stage3_records=inherited_by_key,
                        utility_config=utility_config,
                        device=device,
                        precision="bf16",
                    )
                    replayed_by_key = {
                        record.sample_id: record for record in replayed_records
                    }
                    try:
                        records = [replayed_by_key[key] for key in unfinished_keys]
                    except KeyError as error:
                        raise RuntimeError(
                            "Stage-3 replay batch omitted an unfinished Stage-4 row"
                        ) from error
                    store.append(records)
                    remaining.difference_update(
                        record.sample_id for record in records
                    )
                    progress.update(len(records))
                    progress.set_postfix(
                        source=f"{plan.source}/{plan.split}",
                        saved=store.record_count,
                    )
            completion = store.finalize(expected_keys)
        finally:
            progress.close()

    validate_stage4_record_selection_provenance(
        data_provenance, completed_record_count=completion["record_count"]
    )
    app_pointer_after = _app_pointer_snapshot(config)
    if app_pointer_after != app_pointer_before:
        raise RuntimeError("protected app pointer changed during Stage-4 generation")
    summary = {
        "schema": provenance["schema"],
        "record_schema": STAGE4_RECORD_SCHEMA,
        "identity_sha256": provenance["identity_sha256"],
        "planned_record_count": selection["planned_record_count"],
        "excluded_no_reference_count": selection["excluded_no_reference_count"],
        "excluded_group_policy_count": selection["excluded_group_policy_count"],
        "selected_record_count": completion["record_count"],
        "selected_partition_counts": selection["selected_partition_counts"],
        "selected_source_split_counts": selection["selected_source_split_counts"],
        "records_path": completion["records_path"],
        "records_sha256": completion["records_sha256"],
        "part_count": completion["part_count"],
        "test_splits_excluded": True,
        "app_pointer_changed": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
