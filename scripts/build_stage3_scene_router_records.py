"""Build resumable, label-derived Stage-3 scene-router records on CUDA.

Only GAMUS train/validation and legacy train/validation are admitted.  Test
splits are deliberately neither configured nor opened.  Each source scene is
evaluated once at its deterministic 384px centre crop against both validated,
fully fused frozen endpoints.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msr.data.gamus_dataset import (
    GAMUS_OFFICIAL_SPLIT_COUNTS,
    GamusSurfaceDataset,
)
from msr.data.surface_dataset import (
    MultiDomainSurfaceDataset,
    SurfaceSampleRecord,
    assert_surface_regions_disjoint,
    load_surface_manifest,
)
from msr.models.routed_surface import ConservativeSceneRouter
from msr.models.stage3_endpoints import load_stage3_endpoint_packs
from msr.training.scene_router import (
    SceneRouterRecord,
    SceneUtilityConfig,
    build_scene_router_record,
)
from msr.training.scene_router_record_store import (
    AtomicSceneRouterRecordStore,
    build_record_store_provenance,
    canonical_json_sha256,
    file_sha256,
)


ALLOWED_SOURCE_SPLITS = (
    ("gamus", "train", "train"),
    ("gamus", "val", "calibration"),
    ("legacy", "train", "train"),
    ("legacy", "val", "calibration"),
)

NO_REFERENCE_EXCLUSION_FIELD = "excluded_no_reference_record_keys"

CRITICAL_IMPLEMENTATION_FILES = (
    "scripts/build_stage3_scene_router_records.py",
    "src/msr/data/gamus_dataset.py",
    "src/msr/data/radiometry.py",
    "src/msr/data/raster_dataset.py",
    "src/msr/data/surface_dataset.py",
    "src/msr/models/domain_surface_net.py",
    "src/msr/models/height_net.py",
    "src/msr/models/routed_surface.py",
    "src/msr/models/stage3_endpoints.py",
    "src/msr/training/scene_router.py",
    "src/msr/training/scene_router_record_store.py",
)


@dataclass(frozen=True)
class RouterDatasetPlan:
    source: str
    split: str
    partition: str
    dataset: Dataset
    original_sample_ids: tuple[str, ...]

    @property
    def record_keys(self) -> tuple[str, ...]:
        return tuple(
            scene_router_record_key(self.source, self.split, sample_id)
            for sample_id in self.original_sample_ids
        )


def scene_router_record_key(source: str, split: str, sample_id: str) -> str:
    """Namespace keys so source/split collisions can never overwrite a scene."""

    values = (source.strip().lower(), split.strip().lower(), sample_id.strip())
    if not all(values):
        raise ValueError("router record source, split, and sample_id cannot be empty")
    if any("/" in value or "\\" in value for value in values):
        raise ValueError("router record key components cannot contain path separators")
    return "/".join(values)


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def configured_no_reference_exclusions(
    config: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the explicit, canonical train-only no-reference exclusions."""

    data = _require_mapping(config.get("data"), "data")
    value = data.get(NO_REFERENCE_EXCLUSION_FIELD)
    if not isinstance(value, list):
        raise ValueError(f"data.{NO_REFERENCE_EXCLUSION_FIELD} must be a list")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(
            f"data.{NO_REFERENCE_EXCLUSION_FIELD} must contain only strings"
        )
    keys = tuple(value)
    if not keys:
        raise ValueError(f"data.{NO_REFERENCE_EXCLUSION_FIELD} cannot be empty")
    if len(keys) != len(set(keys)):
        raise ValueError(
            f"data.{NO_REFERENCE_EXCLUSION_FIELD} contains duplicate record keys"
        )
    allowed_partitions = {
        (source, split): partition
        for source, split, partition in ALLOWED_SOURCE_SPLITS
    }
    for key in keys:
        components = key.split("/")
        if len(components) != 3 or any(not component for component in components):
            raise ValueError(
                f"data.{NO_REFERENCE_EXCLUSION_FIELD} keys must be namespaced as "
                f"source/split/sample_id: {key!r}"
            )
        source, split, sample_id = components
        try:
            canonical = scene_router_record_key(source, split, sample_id)
        except ValueError as error:
            raise ValueError(
                f"data.{NO_REFERENCE_EXCLUSION_FIELD} has malformed key {key!r}"
            ) from error
        if canonical != key:
            raise ValueError(
                f"data.{NO_REFERENCE_EXCLUSION_FIELD} keys must use canonical "
                f"namespacing: {key!r}"
            )
        partition = allowed_partitions.get((source, split))
        if partition is None:
            raise ValueError(
                f"data.{NO_REFERENCE_EXCLUSION_FIELD} key is outside the allowed "
                f"source splits: {key!r}"
            )
        if partition != "train":
            raise ValueError(
                f"data.{NO_REFERENCE_EXCLUSION_FIELD} may exclude only train "
                f"records, not calibration: {key!r}"
            )
    return keys


def _no_reference_exclusion_locations(
    config: Mapping[str, Any], plans: Sequence[RouterDatasetPlan]
) -> tuple[tuple[str, ...], dict[str, tuple[RouterDatasetPlan, int]]]:
    """Resolve each configured exclusion to exactly one planned dataset item."""

    exclusions = configured_no_reference_exclusions(config)
    locations: dict[str, tuple[RouterDatasetPlan, int]] = {}
    for plan in plans:
        if len(plan.dataset) != len(plan.original_sample_ids):
            raise ValueError(
                f"{plan.source}/{plan.split} dataset length differs from its sample IDs"
            )
        for index, key in enumerate(plan.record_keys):
            if key in locations:
                raise ValueError(f"Stage-3 data plan contains duplicate key {key!r}")
            locations[key] = (plan, index)
    missing = sorted(set(exclusions) - set(locations))
    if missing:
        raise ValueError(
            "data.excluded_no_reference_record_keys contains keys outside the "
            f"Stage-3 data plan: {missing}"
        )
    return exclusions, locations


def validate_no_reference_exclusions_against_plan(
    config: Mapping[str, Any], plans: Sequence[RouterDatasetPlan]
) -> tuple[str, ...]:
    """Verify every exclusion exists and truly has no scoreable references."""

    exclusions, locations = _no_reference_exclusion_locations(config, plans)
    for key in exclusions:
        plan, index = locations[key]
        sample = plan.dataset[index]
        if not isinstance(sample, Mapping):
            raise ValueError(f"excluded no-reference dataset item is invalid: {key}")
        expected_sample_id = plan.original_sample_ids[index]
        if str(sample.get("sample_id", "")) != expected_sample_id:
            raise ValueError(
                f"excluded no-reference dataset item changed identity: {key}"
            )
        support_counts: dict[str, int] = {}
        for mask_name in ("regression_mask", "domain_valid_mask"):
            mask = sample.get(mask_name)
            if mask is None:
                raise ValueError(
                    f"excluded no-reference item has no {mask_name}: {key}"
                )
            support_counts[mask_name] = int(
                torch.count_nonzero(torch.as_tensor(mask)).item()
            )
        if any(support_counts.values()):
            raise ValueError(
                "configured no-reference exclusion has scoreable reference support: "
                f"{key} ({support_counts})"
            )
    return exclusions


def validate_generation_config(config: Mapping[str, Any]) -> None:
    generation = _require_mapping(config.get("record_generation"), "record_generation")
    endpoints = _require_mapping(config.get("endpoints"), "endpoints")
    data = _require_mapping(config.get("data"), "data")
    utility = _require_mapping(config.get("utility"), "utility")
    required_generation = {
        "output_dir",
        "seed",
        "patch_size",
        "batch_size",
        "num_workers",
        "precision",
        "crop",
    }
    required_endpoints = {
        "protected_checkpoint",
        "gamus_stage1_checkpoint",
        "base_checkpoint_sha256",
        "protected_checkpoint_sha256",
        "gamus_stage1_checkpoint_sha256",
    }
    required_data = {
        "gamus_root",
        "gamus_relative_prior_root",
        "legacy_train_manifest",
        "legacy_val_manifest",
        "legacy_train_manifest_sha256",
        "legacy_val_manifest_sha256",
        "legacy_train_expected_count",
        "legacy_val_expected_count",
        "gamus_train_inventory_sha256",
        "gamus_val_inventory_sha256",
        "legacy_train_inventory_sha256",
        "legacy_val_inventory_sha256",
        "rgb_scale",
        "height_max_m",
        "building_threshold_m",
        "radiometric_policy",
        "legacy_relative_prior_policy",
        NO_REFERENCE_EXCLUSION_FIELD,
    }
    for mapping, required, name in (
        (generation, required_generation, "record_generation"),
        (endpoints, required_endpoints, "endpoints"),
        (data, required_data, "data"),
    ):
        missing = sorted(required - set(mapping))
        if missing:
            raise ValueError(f"{name} is missing fields: {missing}")
    if int(generation["patch_size"]) != 384:
        raise ValueError("Stage-3 records require exactly one 384px crop")
    if str(generation["crop"]).strip().lower() != "center":
        raise ValueError("Stage-3 record generation must use deterministic center crops")
    if str(generation["precision"]).strip().lower() != "bf16":
        raise ValueError("Stage-3 GPU record generation requires bf16 precision")
    if int(generation["batch_size"]) <= 0 or int(generation["num_workers"]) < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if str(data["radiometric_policy"]).strip().lower() != "raw":
        raise ValueError("Both endpoint sources must use raw radiometry")
    if str(data["legacy_relative_prior_policy"]).strip().lower() != "stored_01":
        raise ValueError(
            "Stage-3 legacy records must preserve stored 0..1 priors without labels"
        )
    configured_no_reference_exclusions(config)
    forbidden = [key for key in data if "test" in str(key).lower()]
    if forbidden:
        raise ValueError(
            f"test data configuration is forbidden for router fitting: {forbidden}"
        )
    for name in (
        "base_checkpoint_sha256",
        "protected_checkpoint_sha256",
        "gamus_stage1_checkpoint_sha256",
    ):
        digest = str(endpoints[name]).strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"endpoints.{name} must be a SHA-256 digest")
    for name in (
        "legacy_train_manifest_sha256",
        "legacy_val_manifest_sha256",
        "gamus_train_inventory_sha256",
        "gamus_val_inventory_sha256",
        "legacy_train_inventory_sha256",
        "legacy_val_inventory_sha256",
    ):
        digest = str(data[name]).strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"data.{name} must be a SHA-256 digest")
    for name in ("legacy_train_expected_count", "legacy_val_expected_count"):
        if int(data[name]) <= 0:
            raise ValueError(f"data.{name} must be positive")
    SceneUtilityConfig(
        height_rmse_weight=float(utility.get("height_rmse_weight", 1.0)),
        semantic_error_weight=float(utility.get("semantic_error_weight", 1.0)),
        height_scale_m=float(utility.get("height_scale_m", 10.0)),
    )
    if float(utility.get("minimum_candidate_gain", 0.02)) < 0:
        raise ValueError("utility.minimum_candidate_gain cannot be negative")


def _legacy_dataset(
    records: list[SurfaceSampleRecord], data: Mapping[str, Any], patch_size: int
) -> MultiDomainSurfaceDataset:
    return MultiDomainSurfaceDataset(
        records,
        patch_size=patch_size,
        random_crop=False,
        augment=False,
        samples_per_epoch=None,
        rgb_scale=float(data["rgb_scale"]),
        height_max_m=float(data["height_max_m"]),
        building_threshold_m=float(data["building_threshold_m"]),
        radiometric_policy=str(data["radiometric_policy"]).lower(),
        relative_prior_policy=str(data["legacy_relative_prior_policy"]).lower(),
    )


def create_dataset_plans(config: Mapping[str, Any]) -> tuple[RouterDatasetPlan, ...]:
    """Open only the four allowed fitting/calibration datasets."""

    validate_generation_config(config)
    generation = config["record_generation"]
    data = config["data"]
    patch_size = int(generation["patch_size"])
    gamus_root = _resolve(data["gamus_root"])
    prior_root = _resolve(data["gamus_relative_prior_root"])
    gamus_datasets = {
        split: GamusSurfaceDataset(
            gamus_root,
            split,
            patch_size=patch_size,
            random_crop=False,
            augment=False,
            samples_per_epoch=None,
            rgb_scale=float(data["rgb_scale"]),
            height_max_m=float(data["height_max_m"]),
            radiometric_policy=str(data["radiometric_policy"]).lower(),
            relative_prior_root=prior_root,
            require_relative_prior=True,
        )
        for split in ("train", "val")
    }
    if bool(data.get("require_complete_official_gamus_splits", True)):
        for split in ("train", "val"):
            expected = GAMUS_OFFICIAL_SPLIT_COUNTS[split]
            found = len(gamus_datasets[split].records)
            if found != expected:
                raise ValueError(
                    f"incomplete official GAMUS {split}: expected {expected}, found {found}"
                )
    gamus_train_ids = {record.sample_id for record in gamus_datasets["train"].records}
    gamus_val_ids = {record.sample_id for record in gamus_datasets["val"].records}
    if gamus_train_ids & gamus_val_ids:
        raise ValueError("GAMUS train/validation sample IDs overlap")

    train_manifest = _resolve(data["legacy_train_manifest"])
    val_manifest = _resolve(data["legacy_val_manifest"])
    legacy_records = {
        "train": load_surface_manifest(train_manifest),
        "val": load_surface_manifest(val_manifest),
    }
    for split in ("train", "val"):
        expected = int(data[f"legacy_{split}_expected_count"])
        found = len(legacy_records[split])
        if found != expected:
            raise ValueError(
                f"legacy {split} manifest count changed: expected {expected}, "
                f"found {found}"
            )
    assert_surface_regions_disjoint(legacy_records["train"], legacy_records["val"])
    plans = (
        RouterDatasetPlan(
            "gamus",
            "train",
            "train",
            gamus_datasets["train"],
            tuple(record.sample_id for record in gamus_datasets["train"].records),
        ),
        RouterDatasetPlan(
            "gamus",
            "val",
            "calibration",
            gamus_datasets["val"],
            tuple(record.sample_id for record in gamus_datasets["val"].records),
        ),
        RouterDatasetPlan(
            "legacy",
            "train",
            "train",
            _legacy_dataset(legacy_records["train"], data, patch_size),
            tuple(record.sample_id for record in legacy_records["train"]),
        ),
        RouterDatasetPlan(
            "legacy",
            "val",
            "calibration",
            _legacy_dataset(legacy_records["val"], data, patch_size),
            tuple(record.sample_id for record in legacy_records["val"]),
        ),
    )
    observed = tuple((plan.source, plan.split, plan.partition) for plan in plans)
    if observed != ALLOWED_SOURCE_SPLITS:
        raise AssertionError("router data plan departed from its no-test contract")
    all_keys = [key for plan in plans for key in plan.record_keys]
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("namespaced Stage-3 data plan contains duplicate record keys")
    validate_no_reference_exclusions_against_plan(config, plans)
    return plans


def _path_metadata(path_like: Path) -> dict[str, object]:
    path = path_like.resolve(strict=True)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"data provenance path is not a regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def dataset_plan_fingerprint(plan: RouterDatasetPlan) -> dict[str, object]:
    """Hash the ordered, local data inventory without rereading every raster."""

    digest = hashlib.sha256()
    total_bytes = 0
    unique_paths: set[str] = set()
    records = getattr(plan.dataset, "records", None)
    if not isinstance(records, Sequence) or len(records) != len(plan.dataset):
        raise ValueError("router dataset must expose exactly one source record per item")
    for record in records:
        paths: list[Path] = []
        for field in (
            "image_path",
            "height_path",
            "class_path",
            "relative_prior_path",
            "rgb_path",
            "surface_path",
            "dtm_path",
            "building_mask_path",
            "vegetation_mask_path",
            "valid_mask_path",
        ):
            value = getattr(record, field, None)
            if value is not None:
                paths.append(Path(value))
        files = []
        for path in paths:
            metadata = _path_metadata(path)
            files.append(metadata)
            if str(metadata["path"]) not in unique_paths:
                unique_paths.add(str(metadata["path"]))
                total_bytes += int(metadata["size_bytes"])
        entry = {
            "sample_id": str(record.sample_id),
            "region": str(record.region),
            "landscape": str(getattr(record, "landscape", "mixed")),
            "files": files,
        }
        encoded = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return {
        "source": plan.source,
        "split": plan.split,
        "partition": plan.partition,
        "record_count": len(plan.dataset),
        "unique_file_count": len(unique_paths),
        "total_file_bytes": total_bytes,
        "fingerprint_method": "ordered resolved_path+size_bytes+mtime_ns",
        "inventory_sha256": digest.hexdigest(),
        "sample_ids_sha256": canonical_json_sha256(list(plan.original_sample_ids)),
    }


def record_selection_provenance(
    config: Mapping[str, Any], plans: Sequence[RouterDatasetPlan]
) -> dict[str, object]:
    """Describe planned sources separately from the records that can be scored."""

    planned_keys = [key for plan in plans for key in plan.record_keys]
    exclusions, _ = _no_reference_exclusion_locations(config, plans)
    excluded = set(exclusions)
    scored_keys = [key for key in planned_keys if key not in excluded]
    planned_partition_counts = {
        partition: sum(
            len(plan.dataset) for plan in plans if plan.partition == partition
        )
        for partition in ("train", "calibration")
    }
    return {
        "planned_record_count": len(planned_keys),
        "excluded_no_reference_count": len(exclusions),
        "excluded_no_reference_record_keys": list(exclusions),
        "scored_record_count": len(scored_keys),
        "planned_partition_counts": planned_partition_counts,
        "scored_partition_counts": {
            "train": planned_partition_counts["train"] - len(exclusions),
            "calibration": planned_partition_counts["calibration"],
        },
        "planned_source_split_counts": {
            f"{plan.source}/{plan.split}": len(plan.dataset) for plan in plans
        },
        "scored_source_split_counts": {
            f"{plan.source}/{plan.split}": sum(
                key not in excluded for key in plan.record_keys
            )
            for plan in plans
        },
    }


def build_data_provenance(
    config: Mapping[str, Any], plans: Sequence[RouterDatasetPlan]
) -> dict[str, object]:
    data = config["data"]
    manifests: dict[str, dict[str, object]] = {}
    for name, key, expected_key in (
        (
            "legacy_train",
            "legacy_train_manifest",
            "legacy_train_manifest_sha256",
        ),
        ("legacy_val", "legacy_val_manifest", "legacy_val_manifest_sha256"),
    ):
        path = _resolve(data[key])
        digest = file_sha256(path)
        expected = str(data[expected_key]).strip().lower()
        if digest != expected:
            raise ValueError(
                f"{name} manifest SHA-256 mismatch: expected {expected}, found {digest}"
            )
        manifests[name] = {"path": str(path), "sha256": digest}

    fingerprints = [dataset_plan_fingerprint(plan) for plan in plans]
    expected_inventory = {
        "gamus/train": str(data["gamus_train_inventory_sha256"]).lower(),
        "gamus/val": str(data["gamus_val_inventory_sha256"]).lower(),
        "legacy/train": str(data["legacy_train_inventory_sha256"]).lower(),
        "legacy/val": str(data["legacy_val_inventory_sha256"]).lower(),
    }
    for fingerprint in fingerprints:
        identity = f"{fingerprint['source']}/{fingerprint['split']}"
        expected = expected_inventory[identity]
        actual = str(fingerprint["inventory_sha256"])
        if actual != expected:
            raise ValueError(
                f"{identity} source inventory SHA-256 mismatch: "
                f"expected {expected}, found {actual}"
            )
    return {
        "included_source_splits": [
            f"{source}/{split}" for source, split, _ in ALLOWED_SOURCE_SPLITS
        ],
        "excluded_source_splits": ["gamus/test", "legacy/test"],
        "manifests": manifests,
        "datasets": fingerprints,
        **record_selection_provenance(config, plans),
    }


def implementation_fingerprint() -> dict[str, object]:
    """Bind a record journal to every implementation that shapes its values."""

    files: dict[str, str] = {}
    for relative in CRITICAL_IMPLEMENTATION_FILES:
        path = (PROJECT_ROOT / relative).resolve(strict=True)
        files[relative] = file_sha256(path)
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
        "schema": "msr.stage3_scene_router_implementation.v1",
        "files": files,
        "runtime": runtime,
        "combined_sha256": canonical_json_sha256(
            {"files": files, "runtime": runtime}
        ),
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
    descriptor_router: ConservativeSceneRouter,
    utility_config: SceneUtilityConfig,
    device: torch.device,
    precision: str,
) -> list[SceneRouterRecord]:
    """Evaluate one GPU batch and return detached, label-audited records."""

    image = batch["image"].to(device, non_blocking=True)
    prior = batch["relative_prior"].to(device, non_blocking=True)
    if precision != "bf16":
        raise ValueError("Stage-3 record generation supports bf16 only")
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
        # Deployment has RGB/prior only.  Pooling over a label-derived validity
        # mask here would leak reference availability and create train/serve skew.
        route = descriptor_router(shared["adapter_features"], None)
    # The loader proved that the shared model is the protected endpoint.  This
    # runtime equality check catches any future fusion drift before records are
    # committed to the resumable journal.
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
    descriptors = route["scene_descriptor"].detach().float().cpu()
    records: list[SceneRouterRecord] = []
    sample_ids = list(batch["sample_id"])
    landscapes = list(batch["landscape"])
    for index, original_id in enumerate(sample_ids):
        record, _ = build_scene_router_record(
            sample_id=scene_router_record_key(source, split, str(original_id)),
            descriptor=descriptors[index],
            fallback_endpoint={
                name: value[index] for name, value in protected_cpu.items()
            },
            candidate_endpoint={
                name: value[index] for name, value in candidate_cpu.items()
            },
            source=source,
            landscape=str(landscapes[index]),
            partition=partition,
            height_target=batch["height"][index],
            valid_mask=batch["regression_mask"][index],
            domain_target=batch["domain_target"][index],
            domain_valid_mask=batch["domain_valid_mask"][index],
            utility_config=utility_config,
        )
        records.append(record)
    return records


def _normalised_generation_identity(config: Mapping[str, Any]) -> dict[str, object]:
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
        "height_utility_mask": "dataset_regression_mask",
        "semantic_utility_mask": "dataset_domain_valid_mask",
        "implementation": implementation_fingerprint(),
        "utility": {
            "height_rmse_weight": float(utility.get("height_rmse_weight", 1.0)),
            "semantic_error_weight": float(utility.get("semantic_error_weight", 1.0)),
            "height_scale_m": float(utility.get("height_scale_m", 10.0)),
            "minimum_candidate_gain": float(
                utility.get("minimum_candidate_gain", 0.02)
            ),
        },
    }


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = _resolve(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("Stage-3 router record config must be a mapping")
    validate_generation_config(config)
    generation = config["record_generation"]
    endpoints_config = config["endpoints"]
    seed = int(generation["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda")

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
    # Keep endpoint reconstruction on CPU until the authoritative record-store
    # lock is held. Parallel accidental launches can hash/preflight together,
    # but only one process may allocate the GPU and publish records.
    shared_model = endpoint_bundle.require_shared_model().eval()
    protected_endpoint = endpoint_bundle.protected.eval()
    candidate_endpoint = endpoint_bundle.gamus_stage1.eval()
    for module in (shared_model, protected_endpoint, candidate_endpoint):
        module.requires_grad_(False)
    descriptor_router = ConservativeSceneRouter(
        endpoint_bundle.protected.feature_channels
    ).eval()
    descriptor_router.requires_grad_(False)

    plans = create_dataset_plans(config)
    data_provenance = build_data_provenance(config, plans)
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
    provenance = build_record_store_provenance(
        generation_config=_normalised_generation_identity(config),
        endpoints=endpoint_provenance,
        shared_model=shared_provenance,
        data=data_provenance,
    )
    output_dir = _resolve(generation["output_dir"])
    excluded_keys = set(data_provenance["excluded_no_reference_record_keys"])
    expected_keys = [
        key
        for plan in plans
        for key in plan.record_keys
        if key not in excluded_keys
    ]
    utility_config = SceneUtilityConfig(
        height_rmse_weight=float(config["utility"].get("height_rmse_weight", 1.0)),
        semantic_error_weight=float(
            config["utility"].get("semantic_error_weight", 1.0)
        ),
        height_scale_m=float(config["utility"].get("height_scale_m", 10.0)),
    )
    workers = int(generation["num_workers"])
    batch_size = int(generation["batch_size"])
    with AtomicSceneRouterRecordStore(output_dir, provenance) as store:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real Stage-3 record generation")
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
        descriptor_router.to(device)
        remaining = set(store.validate_expected_keys(expected_keys))
        progress = tqdm(
            total=len(expected_keys),
            initial=store.record_count,
            desc="Stage-3 router records",
            unit="scene",
            dynamic_ncols=True,
        )
        try:
            for plan in plans:
                missing_indices = [
                    index
                    for index, key in enumerate(plan.record_keys)
                    if key in remaining
                ]
                if not missing_indices:
                    continue
                loader = DataLoader(
                    Subset(plan.dataset, missing_indices),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=True,
                    persistent_workers=workers > 0,
                    drop_last=False,
                )
                for batch in loader:
                    records = build_records_from_batch(
                        batch=batch,
                        source=plan.source,
                        split=plan.split,
                        partition=plan.partition,
                        shared_model=shared_model,
                        protected_endpoint=protected_endpoint,
                        candidate_endpoint=candidate_endpoint,
                        descriptor_router=descriptor_router,
                        utility_config=utility_config,
                        device=device,
                        precision="bf16",
                    )
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
    summary = {
        "schema": provenance["schema"],
        "config_sha256": provenance["config_sha256"],
        "planned_record_count": data_provenance["planned_record_count"],
        "excluded_no_reference_count": data_provenance[
            "excluded_no_reference_count"
        ],
        "excluded_no_reference_record_keys": data_provenance[
            "excluded_no_reference_record_keys"
        ],
        "scored_record_count": completion["record_count"],
        "scored_partition_counts": data_provenance["scored_partition_counts"],
        "records_path": completion["records_path"],
        "records_sha256": completion["records_sha256"],
        "part_count": completion["part_count"],
        "planned_source_split_counts": data_provenance[
            "planned_source_split_counts"
        ],
        "scored_source_split_counts": data_provenance[
            "scored_source_split_counts"
        ],
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
