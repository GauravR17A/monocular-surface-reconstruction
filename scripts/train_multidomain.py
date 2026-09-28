"""Safe staged training for urban, canopy, plains, and terrain surface domains."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from msr.data.gamus_dataset import (
    GAMUS_OFFICIAL_SPLIT_COUNTS,
    GAMUS_SIX_CLASS_NAMES,
    GamusSurfaceDataset,
)
from msr.data.mixed_replay import (
    SourceRatioSampler,
    SourceTaggedDataset,
    assert_matching_sample_contract,
)
from msr.data.surface_dataset import (
    LANDSCAPE_CLASSES,
    MultiDomainSurfaceDataset,
    assert_surface_regions_disjoint,
    load_surface_manifest,
)
from msr.evaluation.classification_metrics import (
    StreamingClassBoundaryMetrics,
    StreamingMulticlassMetrics,
    StreamingWaterDarkPixelProxy,
    multiclass_metric_deltas,
)
from msr.evaluation.metrics import StreamingRegressionMetrics
from msr.inference.predict import load_predictor
from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.training.losses import (
    FrozenDomainHeadTeacher,
    MultiDomainSurfaceLoss,
    Stage2SemanticRepairLoss,
)


GAMUS_DEVELOPMENT_SPLITS = ("train", "val")
FUSION_EVALUATION_OVERRIDE_FIELDS = frozenset(
    {
        "fusion_mode",
        "building_protection_power",
        "building_fusion_min_height_m",
        "building_fusion_temperature_m",
        "building_fusion_score_threshold",
        "building_fusion_score_temperature",
        "building_fusion_strength",
        "vegetation_fusion_temperature",
        "vegetation_expert_fusion_threshold",
        "vegetation_expert_fusion_strength",
    }
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_surface_dataset(
    records,
    data_config: dict,
    *,
    training: bool,
    use_train_samples_per_epoch: bool = True,
) -> MultiDomainSurfaceDataset:
    patch_size = int(
        data_config["patch_size"]
        if training
        else data_config.get("validation_patch_size", data_config["patch_size"])
    )
    return MultiDomainSurfaceDataset(
        records,
        patch_size=patch_size,
        random_crop=training,
        augment=training,
        samples_per_epoch=(
            int(data_config.get("train_samples_per_epoch", len(records)))
            if training and use_train_samples_per_epoch
            else None
        ),
        rgb_scale=float(data_config.get("rgb_scale", 255.0)),
        height_max_m=float(data_config.get("height_max_m", 200.0)),
        building_threshold_m=float(data_config.get("building_threshold_m", 2.0)),
        radiometric_policy=str(
            data_config.get(
                "legacy_train_radiometric_policy",
                data_config.get(
                    "train_radiometric_policy"
                    if training
                    else "validation_radiometric_policy",
                    "raw",
                ),
            )
            if training
            else data_config.get("validation_radiometric_policy", "raw")
        ),
        supervised_crop_probability=(
            float(data_config.get("legacy_supervised_crop_probability", 0.0))
            if training
            else 0.0
        ),
        relative_prior_policy=str(
            data_config.get(
                "legacy_relative_prior_policy", "target_crop_normalized"
            )
        ),
    )


def make_loader(records, data_config: dict, training_config: dict, *, training: bool):
    dataset = make_surface_dataset(records, data_config, training=training)
    workers = int(data_config.get("num_workers", 0))
    sampler = None
    if training and bool(data_config.get("balanced_landscape_sampling", False)):
        landscape_counts: dict[str, int] = defaultdict(int)
        for record in records:
            landscape_counts[record.landscape] += 1
        weights = [
            1.0 / landscape_counts[records[index % len(records)].landscape]
            for index in range(len(dataset))
        ]
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]) if training else 1,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=training,
    )


def make_gamus_dataset(
    root: str | Path,
    split: str,
    data_config: dict,
    *,
    training: bool,
    use_train_samples_per_epoch: bool = True,
) -> GamusSurfaceDataset:
    """Build one official GAMUS split while keeping its native HDF5 files."""

    approved_index_path = data_config.get("approved_index_path")
    if bool(data_config.get("require_approved_index", False)) and not approved_index_path:
        raise ValueError(
            "data.require_approved_index is true but data.approved_index_path is missing"
        )
    expected_index_sha256 = data_config.get("approved_index_file_sha256")
    if bool(data_config.get("require_approved_index", False)) and not expected_index_sha256:
        raise ValueError(
            "data.require_approved_index is true but "
            "data.approved_index_file_sha256 is missing"
        )
    if expected_index_sha256:
        if not approved_index_path:
            raise ValueError(
                "data.approved_index_file_sha256 requires data.approved_index_path"
            )
        digest = hashlib.sha256()
        with Path(approved_index_path).expanduser().resolve().open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_index_sha256 = digest.hexdigest()
        if actual_index_sha256 != str(expected_index_sha256).strip().lower():
            raise ValueError(
                "GAMUS approved-index file SHA-256 mismatch: "
                f"expected {expected_index_sha256}, found {actual_index_sha256}"
            )
    patch_size = int(
        data_config["patch_size"]
        if training
        else data_config.get("validation_patch_size", data_config["patch_size"])
    )
    samples_per_epoch = None
    if (
        training
        and use_train_samples_per_epoch
        and "train_samples_per_epoch" in data_config
    ):
        samples_per_epoch = int(data_config["train_samples_per_epoch"])
    return GamusSurfaceDataset(
        root,
        split,
        patch_size=patch_size,
        random_crop=training,
        augment=training,
        samples_per_epoch=samples_per_epoch,
        rgb_scale=float(data_config.get("rgb_scale", 255.0)),
        height_max_m=float(data_config.get("height_max_m", 200.0)),
        dark_pixel_threshold=float(data_config.get("dark_pixel_threshold", 0.15)),
        radiometric_policy=str(
            data_config.get(
                "gamus_train_radiometric_policy",
                data_config.get("train_radiometric_policy", "raw"),
            )
            if training
            else data_config.get("validation_radiometric_policy", "raw")
        ),
        relative_prior_root=data_config.get("relative_prior_root"),
        require_relative_prior=bool(
            data_config.get("require_relative_priors", False)
        ),
        approved_index_path=approved_index_path,
    )


def make_gamus_development_datasets(
    root: str | Path,
    data_config: dict,
    *,
    use_train_samples_per_epoch: bool = True,
) -> dict[str, GamusSurfaceDataset]:
    """Construct train/validation datasets without resolving the test split."""

    return {
        split: make_gamus_dataset(
            root,
            split,
            data_config,
            training=split == "train",
            use_train_samples_per_epoch=use_train_samples_per_epoch,
        )
        for split in GAMUS_DEVELOPMENT_SPLITS
    }


def validate_gamus_splits(
    datasets: dict[str, GamusSurfaceDataset],
    *,
    require_complete_official_splits: bool,
    required_splits: tuple[str, ...] = GAMUS_DEVELOPMENT_SPLITS,
) -> None:
    """Validate only the explicitly loaded GAMUS development splits.

    The training path deliberately requires ``train`` and ``val`` but not
    ``test``.  Passing a test dataset remains supported for old offline audits;
    this validator never requires the trainer to construct one.
    """

    unknown_splits = set(datasets) - set(GAMUS_OFFICIAL_SPLIT_COUNTS)
    if unknown_splits:
        raise ValueError(f"Unknown GAMUS splits: {sorted(unknown_splits)}")
    missing_splits = set(required_splits) - set(datasets)
    if missing_splits:
        raise ValueError(f"Missing GAMUS splits: {sorted(missing_splits)}")
    sample_ids = {
        split: {record.sample_id for record in dataset.records}
        for split, dataset in datasets.items()
    }
    loaded_splits = tuple(sorted(sample_ids))
    for left_index, left in enumerate(loaded_splits):
        for right in loaded_splits[left_index + 1 :]:
            overlap = sample_ids[left] & sample_ids[right]
            if overlap:
                preview = sorted(overlap)[:5]
                raise ValueError(
                    f"GAMUS tile leakage between {left} and {right}: {preview}"
                )
    if require_complete_official_splits:
        def expected_record_count(
            split: str,
            dataset: GamusSurfaceDataset,
        ) -> int:
            approved_index_path = getattr(dataset, "approved_index_path", None)
            if approved_index_path is None:
                return GAMUS_OFFICIAL_SPLIT_COUNTS[split]
            payload = json.loads(
                Path(approved_index_path).read_text(encoding="utf-8")
            )
            split_payload = payload.get("splits", {}).get(split)
            if not isinstance(split_payload, dict) or "approved_count" not in split_payload:
                raise ValueError(
                    f"GAMUS approved index does not declare approved_count for {split}"
                )
            official_count = GAMUS_OFFICIAL_SPLIT_COUNTS[split]
            source_count = int(split_payload.get("source_count", -1))
            expected_official_count = int(
                split_payload.get("expected_official_count", -1)
            )
            if source_count != official_count or expected_official_count != official_count:
                raise ValueError(
                    f"GAMUS approved index does not authenticate the complete {split} "
                    f"source inventory: source={source_count}, "
                    f"expected={expected_official_count}, official={official_count}"
                )
            expected = int(split_payload["approved_count"])
            if not 0 < expected <= official_count:
                raise ValueError(
                    f"Invalid approved GAMUS count for {split}: {expected}"
                )
            return expected

        wrong_counts = {
            split: {
                "expected": expected_record_count(split, dataset),
                "found": len(dataset.records),
            }
            for split, dataset in datasets.items()
            if len(dataset.records) != expected_record_count(split, dataset)
        }
        if wrong_counts:
            raise ValueError(
                "Incomplete GAMUS official split counts: "
                + json.dumps(wrong_counts, sort_keys=True)
            )


def make_gamus_loader(
    dataset: GamusSurfaceDataset,
    data_config: dict,
    training_config: dict,
    *,
    training: bool,
) -> DataLoader:
    """Wrap a native GAMUS dataset without manifest-only landscape sampling."""

    workers = int(data_config.get("num_workers", 0))
    sampler = None
    if training and data_config.get("fine_class_sampling_index_path"):
        record_weights, diagnostics = gamus_fine_class_sampling_weights(
            dataset,
            data_config,
        )
        weights = [
            record_weights[index % len(dataset.records)]
            for index in range(len(dataset))
        ]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
        )
        print(
            "Enabled sealed GAMUS fine-class sampling: "
            + json.dumps(diagnostics, sort_keys=True)
        )
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]) if training else 1,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=training and bool(training_config.get("drop_last", True)),
    )


def gamus_fine_class_sampling_weights(
    dataset: GamusSurfaceDataset,
    data_config: dict,
) -> tuple[list[float], dict[str, object]]:
    """Load sealed water-hit estimates and build city-preserving tile weights.

    This changes which *training* tiles are drawn, never their labels.  The
    index is generated from the approved training class rasters only.  Exact ID
    coverage and a caller-supplied SHA-256 are mandatory so a partial or stale
    scan cannot silently influence an experiment.
    """

    raw_path = data_config.get("fine_class_sampling_index_path")
    expected_sha256 = str(
        data_config.get("fine_class_sampling_index_sha256", "")
    ).strip().lower()
    if not raw_path or not expected_sha256:
        raise ValueError(
            "fine_class_sampling_index_path requires "
            "fine_class_sampling_index_sha256"
        )
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GAMUS fine-class sampling index is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "GAMUS fine-class sampling index SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )

    indexed: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "msr.gamus.train_six_class_tile_index.v1":
                raise ValueError(
                    f"Unexpected GAMUS sampling schema on line {line_number}"
                )
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id or sample_id in indexed:
                raise ValueError(
                    f"Missing or duplicate GAMUS sample ID on line {line_number}"
                )
            indexed[sample_id] = row

    expected_ids = [record.sample_id for record in dataset.records]
    if set(indexed) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(indexed))
        extra = sorted(set(indexed) - set(expected_ids))
        raise ValueError(
            "GAMUS fine-class sampling index does not exactly match the loaded "
            f"training split: missing={missing[:5]}, extra={extra[:5]}"
        )
    boost = float(data_config.get("water_sampling_boost", 0.0))
    if not math.isfinite(boost) or boost < 0.0:
        raise ValueError("data.water_sampling_boost must be finite and nonnegative")

    raw_weights: list[float] = []
    probabilities: list[float] = []
    cities: list[str] = []
    for sample_id in expected_ids:
        row = indexed[sample_id]
        crop = row.get("uniform_random_crop_384")
        if not isinstance(crop, dict):
            raise ValueError(f"Sampling index lacks crop statistics for {sample_id}")
        probability = float(crop.get("water_hit_probability", -1.0))
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid water-hit probability for {sample_id}")
        city = str(row.get("city", "")).strip()
        if not city or not sample_id.startswith(f"{city}_"):
            raise ValueError(f"Invalid city identity for {sample_id}")
        probabilities.append(probability)
        cities.append(city)
        raw_weights.append(1.0 + boost * probability)

    weights = list(raw_weights)
    if bool(data_config.get("preserve_city_sampling_mass", True)):
        city_indices: dict[str, list[int]] = defaultdict(list)
        for index, city in enumerate(cities):
            city_indices[city].append(index)
        for indices in city_indices.values():
            mean_weight = sum(raw_weights[index] for index in indices) / len(indices)
            for index in indices:
                weights[index] = raw_weights[index] / mean_weight

    uniform_expected_hit = sum(probabilities) / len(probabilities)
    weighted_expected_hit = sum(
        weight * probability for weight, probability in zip(weights, probabilities)
    ) / sum(weights)
    diagnostics: dict[str, object] = {
        "index": str(path),
        "index_sha256": actual_sha256,
        "records": len(weights),
        "water_sampling_boost": boost,
        "preserve_city_sampling_mass": bool(
            data_config.get("preserve_city_sampling_mass", True)
        ),
        "uniform_expected_water_crop_hit_rate": uniform_expected_hit,
        "weighted_expected_water_crop_hit_rate": weighted_expected_hit,
        "minimum_weight": min(weights),
        "maximum_weight": max(weights),
    }
    return weights, diagnostics


def make_mixed_replay_loader(
    gamus_dataset: GamusSurfaceDataset,
    legacy_dataset: MultiDomainSurfaceDataset,
    data_config: dict,
    training_config: dict,
    *,
    seed: int,
) -> DataLoader:
    """Build exact-ratio mixed batches over contract-compatible datasets."""

    assert_matching_sample_contract(gamus_dataset, legacy_dataset)
    dataset = ConcatDataset(
        (
            SourceTaggedDataset(gamus_dataset, "gamus"),
            SourceTaggedDataset(legacy_dataset, "legacy"),
        )
    )
    sampler = SourceRatioSampler(
        len(gamus_dataset),
        len(legacy_dataset),
        num_samples=int(data_config["train_samples_per_epoch"]),
        batch_size=int(training_config["batch_size"]),
        gamus_fraction=float(data_config.get("gamus_train_fraction", 0.75)),
        seed=seed,
        legacy_landscapes=[record.landscape for record in legacy_dataset.records],
        balance_legacy_landscapes=bool(
            data_config.get("balanced_legacy_landscape_sampling", False)
        ),
    )
    workers = int(data_config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=True,
    )


def configure_explicit_parameter_groups(
    model: DomainGatedSurfaceNet,
    training_config: dict,
) -> list[dict[str, object]] | None:
    """Freeze everything except non-overlapping, explicitly named LR groups.

    Existing staged configurations continue to use the historical base/adapter
    controls.  A direct-height run can opt into this stricter path with
    ``training.parameter_groups`` so a typo cannot silently unfreeze the whole
    protected base model.
    """

    raw_groups = training_config.get("parameter_groups")
    if raw_groups is None:
        return None
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("training.parameter_groups must be a non-empty list")
    if training_config.get("trainable_adapter_prefixes"):
        raise ValueError(
            "training.parameter_groups cannot be combined with "
            "training.trainable_adapter_prefixes"
        )
    if int(training_config.get("freeze_base_epochs", 0)) != 0:
        raise ValueError(
            "training.parameter_groups requires freeze_base_epochs: 0; the "
            "explicit groups define the immutable trainable scope"
        )

    named_parameters = list(model.named_parameters())
    for _, parameter in named_parameters:
        parameter.requires_grad_(False)

    configured: list[dict[str, object]] = []
    assigned_parameters: dict[str, str] = {}
    group_names: set[str] = set()
    allowed_keys = {"name", "prefixes", "learning_rate", "weight_decay"}
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            raise ValueError(
                f"training.parameter_groups[{group_index}] must be a mapping"
            )
        unknown_keys = set(raw_group) - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"training.parameter_groups[{group_index}] has unsupported keys: "
                f"{sorted(unknown_keys)}"
            )
        name = str(raw_group.get("name", "")).strip()
        if not name or name in group_names:
            raise ValueError("training.parameter_groups names must be unique and non-empty")
        group_names.add(name)
        raw_prefixes = raw_group.get("prefixes")
        if not isinstance(raw_prefixes, list) or not raw_prefixes:
            raise ValueError(f"Parameter group {name!r} requires a non-empty prefixes list")
        prefixes = tuple(str(prefix).strip() for prefix in raw_prefixes)
        if any(not prefix for prefix in prefixes):
            raise ValueError(f"Parameter group {name!r} contains an empty prefix")
        learning_rate = float(raw_group.get("learning_rate", 0.0))
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(f"Parameter group {name!r} learning_rate must be positive")

        prefix_matches = {prefix: 0 for prefix in prefixes}
        matched: list[torch.nn.Parameter] = []
        matched_names: list[str] = []
        for parameter_name, parameter in named_parameters:
            matching = [prefix for prefix in prefixes if parameter_name.startswith(prefix)]
            if not matching:
                continue
            for prefix in matching:
                prefix_matches[prefix] += 1
            previous_group = assigned_parameters.get(parameter_name)
            if previous_group is not None:
                raise ValueError(
                    f"Model parameter {parameter_name!r} matches both "
                    f"{previous_group!r} and {name!r}"
                )
            assigned_parameters[parameter_name] = name
            parameter.requires_grad_(True)
            matched.append(parameter)
            matched_names.append(parameter_name)
        unmatched_prefixes = [
            prefix for prefix, count in prefix_matches.items() if count == 0
        ]
        if unmatched_prefixes:
            raise ValueError(
                f"Parameter group {name!r} matched no tensors for prefixes "
                f"{unmatched_prefixes}"
            )

        optimizer_group: dict[str, object] = {
            "params": matched,
            "lr": learning_rate,
            "group_name": name,
        }
        if "weight_decay" in raw_group:
            weight_decay = float(raw_group["weight_decay"])
            if not math.isfinite(weight_decay) or weight_decay < 0.0:
                raise ValueError(
                    f"Parameter group {name!r} weight_decay must be non-negative"
                )
            optimizer_group["weight_decay"] = weight_decay
        configured.append(optimizer_group)
        print(
            f"Trainable parameter group {name}: tensors={len(matched_names)}, "
            f"parameters={sum(parameter.numel() for parameter in matched)}, "
            f"prefixes={list(prefixes)}"
        )

    model.base_trainable = any(
        parameter.requires_grad
        for name, parameter in named_parameters
        if name.startswith("base_model.")
    )
    return configured


def make_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: dict,
    *,
    batches_per_epoch: int,
) -> tuple[LRScheduler, str]:
    """Build either the legacy epoch cosine or a warm-up step cosine."""

    schedule = str(training_config.get("learning_rate_schedule", "epoch_cosine"))
    epochs = int(training_config["epochs"])
    if schedule == "epoch_cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs), "epoch"
    if schedule != "warmup_cosine_steps":
        raise ValueError(
            "training.learning_rate_schedule must be 'epoch_cosine' or "
            "'warmup_cosine_steps'"
        )

    accumulation = int(training_config.get("gradient_accumulation", 1))
    if accumulation <= 0:
        raise ValueError("training.gradient_accumulation must be positive")
    steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(training_config.get("warmup_optimizer_steps", 0))
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "training.warmup_optimizer_steps must be non-negative and smaller "
            "than the total optimizer-step budget"
        )
    minimum_factor = float(training_config.get("minimum_learning_rate_factor", 0.1))
    if not 0.0 <= minimum_factor <= 1.0:
        raise ValueError("training.minimum_learning_rate_factor must be within [0, 1]")

    def learning_rate_factor(step_index: int) -> float:
        if warmup_steps and step_index < warmup_steps:
            return max((step_index + 1) / warmup_steps, 1.0 / warmup_steps)
        cosine_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((step_index - warmup_steps) / cosine_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return LambdaLR(optimizer, lr_lambda=learning_rate_factor), "optimizer_step"


def _batch_tensors(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    values = {
        name: batch[name].to(device, non_blocking=True)
        for name in (
            "image",
            "height",
            "regression_mask",
            "domain_target",
            "domain_valid_mask",
            "relative_prior",
        )
    }
    for name in (
        "image_valid_mask",
        "classification_valid_mask",
        "height_valid_mask",
        "fine_class_target",
        "dark_pixel_proxy_mask",
    ):
        if name in batch:
            values[name] = batch[name].to(device, non_blocking=True)
    return values


def _autocast(device: torch.device, precision: str):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=precision != "fp32",
    )


def gradient_accumulation_window_size(
    step: int,
    total_batches: int,
    accumulation: int,
) -> int:
    """Return the actual batch count in this step's accumulation window."""

    if accumulation <= 0 or total_batches <= 0 or not 1 <= step <= total_batches:
        raise ValueError("invalid gradient-accumulation window arguments")
    window_start = ((step - 1) // accumulation) * accumulation
    return min(accumulation, total_batches - window_start)


def train_epoch(
    model: DomainGatedSurfaceNet,
    loader: DataLoader,
    criterion: MultiDomainSurfaceLoss | Stage2SemanticRepairLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    precision: str,
    accumulation: int,
    max_gradient_norm: float,
    teacher: FrozenDomainHeadTeacher | None = None,
    scheduler: LRScheduler | None = None,
    scheduler_interval: str = "epoch",
) -> dict[str, float]:
    if accumulation <= 0:
        raise ValueError("gradient accumulation must be positive")
    if scheduler_interval not in {"epoch", "optimizer_step"}:
        raise ValueError("scheduler_interval must be 'epoch' or 'optimizer_step'")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    batches = 0
    total_batches = len(loader)
    progress = tqdm(loader, desc="multidomain train", leave=False)
    for step, batch in enumerate(progress, start=1):
        values = _batch_tensors(batch, device)
        with _autocast(device, precision):
            output = model(values["image"], values["relative_prior"])
            if isinstance(criterion, Stage2SemanticRepairLoss):
                if teacher is None:
                    raise ValueError("Stage-2 mixed replay requires a frozen teacher")
                teacher_logits = teacher(
                    output["adapter_features"],
                    output["prior_domain_logits"],
                )
                loss, components = criterion(
                    output,
                    values["domain_target"],
                    values["domain_valid_mask"],
                    batch["source"],
                    batch_landscapes=batch["landscape"],
                    teacher_domain_logits=teacher_logits,
                    target=values["height"],
                    regression_mask=values["regression_mask"],
                )
            else:
                loss, components = criterion(
                    output,
                    values["height"],
                    values["regression_mask"],
                    values["domain_target"],
                    values["domain_valid_mask"],
                    fine_class_target=values.get("fine_class_target"),
                    classification_valid_mask=values.get(
                        "classification_valid_mask"
                    ),
                    image_valid_mask=values.get("image_valid_mask"),
                )
            scaled_loss = loss / gradient_accumulation_window_size(
                step,
                total_batches,
                accumulation,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite multidomain loss at step {step}")
        scaler.scale(scaled_loss).backward()
        if step % accumulation == 0:
            if max_gradient_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and scheduler_interval == "optimizer_step":
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        for name, value in components.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach())
        batches += 1
        progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    if batches % accumulation:
        if max_gradient_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None and scheduler_interval == "optimizer_step":
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return {name: value / max(batches, 1) for name, value in sums.items()}


@torch.inference_mode()
def validate(
    model: DomainGatedSurfaceNet,
    loader: DataLoader,
    criterion: MultiDomainSurfaceLoss,
    device: torch.device,
    *,
    precision: str,
) -> tuple[dict[str, float], dict[str, object]]:
    model.eval()
    overall = StreamingRegressionMetrics()
    semantic_metrics = {
        name: StreamingRegressionMetrics() for name in LANDSCAPE_CLASSES
    }
    protected_base_semantic_metrics = {
        name: StreamingRegressionMetrics() for name in LANDSCAPE_CLASSES
    }
    building_expert_metrics = StreamingRegressionMetrics()
    canopy_expert_metrics = StreamingRegressionMetrics()
    tall_building_metrics = StreamingRegressionMetrics()
    tall_vegetation_metrics = StreamingRegressionMetrics()
    landscape_metrics: dict[str, StreamingRegressionMetrics] = defaultdict(
        StreamingRegressionMetrics
    )
    protected_base_landscape_metrics: dict[str, StreamingRegressionMetrics] = defaultdict(
        StreamingRegressionMetrics
    )
    region_metrics: dict[str, StreamingRegressionMetrics] = defaultdict(
        StreamingRegressionMetrics
    )
    loss_sums: dict[str, float] = {}
    router_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    semantic_class_names = tuple(
        name
        for name, _ in sorted(
            LANDSCAPE_CLASSES.items(), key=lambda item: item[1]
        )
    )
    semantic_identification = StreamingMulticlassMetrics(semantic_class_names)
    six_class_identification = StreamingMulticlassMetrics(GAMUS_SIX_CLASS_NAMES)
    road_boundary_quality = StreamingClassBoundaryMetrics(
        GAMUS_SIX_CLASS_NAMES.index("roads"), tolerance_pixels=2
    )
    water_shadow_proxy = StreamingWaterDarkPixelProxy(
        GAMUS_SIX_CLASS_NAMES.index("water")
    )
    batches = 0
    for batch in tqdm(loader, desc="multidomain validation", leave=False):
        values = _batch_tensors(batch, device)
        with _autocast(device, precision):
            output = model(values["image"], values["relative_prior"])
            _, components = criterion(
                output,
                values["height"],
                values["regression_mask"],
                values["domain_target"],
                values["domain_valid_mask"],
                fine_class_target=values.get("fine_class_target"),
                classification_valid_mask=values.get(
                    "classification_valid_mask"
                ),
                image_valid_mask=values.get("image_valid_mask"),
            )
        if not torch.isfinite(output["height"]).all():
            raise FloatingPointError("Non-finite surface prediction during validation")
        for name, value in components.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value)

        prediction = output["height"].float().cpu().numpy()
        protected_base_prediction = output["base_height"].float().cpu().numpy()
        building_expert_prediction = output["building_height"].float().cpu().numpy()
        canopy_expert_prediction = output["canopy_height"].float().cpu().numpy()
        target = values["height"].float().cpu().numpy()
        regression_mask = values["regression_mask"].cpu().numpy()
        domain_target = values["domain_target"].cpu().numpy()
        semantic_identification.update(
            output["domain_logits"].argmax(dim=1).cpu().numpy(),
            domain_target,
            values["domain_valid_mask"].cpu().numpy(),
        )
        if "fine_semantic_logits" in output:
            required_fine_fields = {
                "fine_class_target",
                "classification_valid_mask",
                "dark_pixel_proxy_mask",
            }
            missing_fine_fields = required_fine_fields - set(values)
            if missing_fine_fields:
                raise KeyError(
                    "Six-class validation is missing GAMUS fields: "
                    f"{sorted(missing_fine_fields)}"
                )
            fine_prediction = (
                output["fine_semantic_logits"].argmax(dim=1).cpu().numpy()
            )
            fine_target = values["fine_class_target"].cpu().numpy()
            fine_valid = (
                values["classification_valid_mask"].bool()
                & values["image_valid_mask"][:, 0].bool()
            ).cpu().numpy()
            six_class_identification.update(
                fine_prediction,
                fine_target,
                fine_valid,
            )
            road_boundary_quality.update(
                fine_prediction,
                fine_target,
                fine_valid,
            )
            water_shadow_proxy.update(
                fine_prediction,
                fine_target,
                values["dark_pixel_proxy_mask"].cpu().numpy(),
                fine_valid,
            )
        router_score = (
            output["protected_building_logits"].sigmoid()
            * output["domain_probabilities"][:, 1:2]
            * (1.0 - output["domain_probabilities"][:, 2:3])
        )
        router_predicted = router_score[:, 0] >= 0.5
        router_truth = values["domain_target"] == LANDSCAPE_CLASSES["building"]
        router_valid = values["domain_valid_mask"].bool()
        router_counts["tp"] += int(
            torch.count_nonzero(router_valid & router_truth & router_predicted)
        )
        router_counts["fp"] += int(
            torch.count_nonzero(router_valid & ~router_truth & router_predicted)
        )
        router_counts["fn"] += int(
            torch.count_nonzero(router_valid & router_truth & ~router_predicted)
        )
        router_counts["tn"] += int(
            torch.count_nonzero(router_valid & ~router_truth & ~router_predicted)
        )
        overall.update(prediction, target, regression_mask)
        building_expert_metrics.update(
            building_expert_prediction,
            target,
            regression_mask & (domain_target[:, None] == LANDSCAPE_CLASSES["building"]),
        )
        canopy_expert_metrics.update(
            canopy_expert_prediction,
            target,
            regression_mask & (domain_target[:, None] == LANDSCAPE_CLASSES["vegetation"]),
        )
        tall_building_metrics.update(
            prediction,
            target,
            regression_mask
            & (domain_target[:, None] == LANDSCAPE_CLASSES["building"])
            & (target >= criterion.tall_building_threshold_m),
        )
        tall_vegetation_metrics.update(
            prediction,
            target,
            regression_mask
            & (domain_target[:, None] == LANDSCAPE_CLASSES["vegetation"])
            & (target >= criterion.tall_canopy_threshold_m),
        )
        for name, code in LANDSCAPE_CLASSES.items():
            semantic_metrics[name].update(
                prediction,
                target,
                regression_mask & (domain_target[:, None] == code),
            )
            protected_base_semantic_metrics[name].update(
                protected_base_prediction,
                target,
                regression_mask & (domain_target[:, None] == code),
            )
        for index, landscape in enumerate(batch["landscape"]):
            landscape_metrics[str(landscape)].update(
                prediction[index : index + 1],
                target[index : index + 1],
                regression_mask[index : index + 1],
            )
            protected_base_landscape_metrics[str(landscape)].update(
                protected_base_prediction[index : index + 1],
                target[index : index + 1],
                regression_mask[index : index + 1],
            )
        for index, region in enumerate(batch["region"]):
            region_metrics[str(region)].update(
                prediction[index : index + 1],
                target[index : index + 1],
                regression_mask[index : index + 1],
            )
        batches += 1

    metrics: dict[str, object] = overall.compute()
    domains = {
        name: metric.compute() for name, metric in semantic_metrics.items() if metric.count
    }
    protected_base_domains = {
        name: metric.compute()
        for name, metric in protected_base_semantic_metrics.items()
        if metric.count
    }
    landscapes = {
        name: metric.compute()
        for name, metric in sorted(landscape_metrics.items())
        if metric.count
    }
    protected_base_landscapes = {
        name: metric.compute()
        for name, metric in sorted(protected_base_landscape_metrics.items())
        if metric.count
    }
    regions = {
        name: metric.compute()
        for name, metric in sorted(region_metrics.items())
        if metric.count
    }
    metrics["domains"] = domains
    metrics["protected_base_domains"] = protected_base_domains
    metrics["landscapes"] = landscapes
    metrics["protected_base_landscapes"] = protected_base_landscapes
    metrics["regions"] = regions
    # A source-specific suite may legitimately have no vegetation/building
    # labels. Missing reference support is unavailable, never a zero error.
    expert_accumulators = {
        "building": building_expert_metrics,
        "vegetation": canopy_expert_metrics,
    }
    metrics["experts"] = {
        name: accumulator.compute()
        for name, accumulator in expert_accumulators.items()
        if accumulator.count
    }
    metrics["unavailable_expert_metrics"] = {
        name: "no_valid_reference_pixels"
        for name, accumulator in expert_accumulators.items()
        if not accumulator.count
    }
    metrics["tall_objects"] = {
        **(
            {"building": tall_building_metrics.compute()}
            if tall_building_metrics.count
            else {}
        ),
        **(
            {"vegetation": tall_vegetation_metrics.compute()}
            if tall_vegetation_metrics.count
            else {}
        ),
    }
    for domain_name, scalar_name in (
        ("building", "building_expert_rmse_m"),
        ("vegetation", "canopy_expert_rmse_m"),
    ):
        if domain_name in metrics["experts"]:
            metrics[scalar_name] = float(metrics["experts"][domain_name]["rmse_m"])
    router_precision = router_counts["tp"] / max(
        router_counts["tp"] + router_counts["fp"], 1
    )
    router_recall = router_counts["tp"] / max(
        router_counts["tp"] + router_counts["fn"], 1
    )
    beta_squared = 0.25
    router_f05 = (
        (1.0 + beta_squared) * router_precision * router_recall
        / max(beta_squared * router_precision + router_recall, 1e-12)
    )
    metrics["building_router"] = {
        **router_counts,
        "score_threshold": 0.5,
        "precision": router_precision,
        "recall": router_recall,
        "f0_5": router_f05,
    }
    metrics["building_router_error"] = 1.0 - router_f05
    metrics["semantic_identification"] = semantic_identification.compute()
    if six_class_identification.count:
        six_class_metrics = six_class_identification.compute()
        per_class = six_class_metrics.get("per_class")
        if not isinstance(per_class, dict):
            raise ValueError("Six-class metrics are missing per-class results")
        low_vegetation = per_class.get("low_vegetation")
        trees = per_class.get("trees")
        if not isinstance(low_vegetation, dict) or not isinstance(trees, dict):
            raise ValueError(
                "Six-class metrics are missing low-vegetation/tree results"
            )
        six_class_metrics["vegetation_balance_f1"] = min(
            float(low_vegetation["f1"]),
            float(trees["f1"]),
        )
        metrics["six_class_identification"] = six_class_metrics
        metrics["road_boundary_quality"] = road_boundary_quality.compute()
        metrics["water_shadow_proxy"] = water_shadow_proxy.compute()
    if domains:
        metrics["domain_macro_rmse_m"] = float(
            np.mean([float(item["rmse_m"]) for item in domains.values()])
        )
        if {"building", "vegetation"} <= set(domains):
            metrics["object_domain_macro_rmse_m"] = float(
                np.mean(
                    [
                        float(domains["building"]["rmse_m"]),
                        float(domains["vegetation"]["rmse_m"]),
                    ]
                )
            )
        metrics["worst_domain"] = max(
            domains, key=lambda name: float(domains[name]["rmse_m"])
        )
        metrics["worst_domain_rmse_m"] = float(
            domains[str(metrics["worst_domain"])]["rmse_m"]
        )
    if landscapes:
        metrics["landscape_macro_rmse_m"] = float(
            np.mean([float(item["rmse_m"]) for item in landscapes.values()])
        )
        metrics["worst_landscape"] = max(
            landscapes, key=lambda name: float(landscapes[name]["rmse_m"])
        )
        metrics["worst_landscape_rmse_m"] = float(
            landscapes[str(metrics["worst_landscape"])]["rmse_m"]
        )
    if regions:
        metrics["region_macro_rmse_m"] = float(
            np.mean([float(item["rmse_m"]) for item in regions.values()])
        )
    if "urban" in landscapes and "urban" in protected_base_landscapes:
        metrics["urban_rmse_regression_m"] = float(
            landscapes["urban"]["rmse_m"]
            - protected_base_landscapes["urban"]["rmse_m"]
        )
    if "building" in domains and "building" in protected_base_domains:
        metrics["building_rmse_regression_m"] = float(
            domains["building"]["rmse_m"]
            - protected_base_domains["building"]["rmse_m"]
        )
    losses = {name: value / max(batches, 1) for name, value in loss_sums.items()}
    return losses, metrics


def validate_suites(
    model: DomainGatedSurfaceNet,
    validation_loaders: dict[str, DataLoader],
    criterion: MultiDomainSurfaceLoss,
    device: torch.device,
    *,
    precision: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, object]]]:
    """Evaluate independent validation suites without merging their statistics."""

    losses_by_suite: dict[str, dict[str, float]] = {}
    metrics_by_suite: dict[str, dict[str, object]] = {}
    for suite, loader in validation_loaders.items():
        losses, metrics = validate(
            model,
            loader,
            criterion,
            device,
            precision=precision,
        )
        losses_by_suite[suite] = losses
        metrics_by_suite[suite] = metrics
    return losses_by_suite, metrics_by_suite


@contextmanager
def temporary_fusion_evaluation_overrides(
    model: DomainGatedSurfaceNet,
    overrides: dict[str, object] | None,
):
    """Evaluate a warm start under its protected policy, then restore candidate policy."""

    resolved = overrides or {}
    if not isinstance(resolved, dict):
        raise ValueError("evaluation.initial_model_overrides must be a mapping")
    unknown = set(resolved) - FUSION_EVALUATION_OVERRIDE_FIELDS
    if unknown:
        raise ValueError(
            "evaluation.initial_model_overrides contains unsupported fields: "
            f"{sorted(unknown)}"
        )
    original = {name: getattr(model, name) for name in resolved}
    try:
        for name, value in resolved.items():
            current = original[name]
            setattr(model, name, str(value) if isinstance(current, str) else float(value))
        yield
    finally:
        for name, value in original.items():
            setattr(model, name, value)


def _metric_at_path(metrics: dict[str, object], path: str) -> float:
    value: object = metrics
    for part in path.split("."):
        if not part:
            raise ValueError(f"Invalid empty component in metric path {path!r}")
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Validation metric path {path!r} is missing at {part!r}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"Validation metric path {path!r} is not numeric")
    return float(value)


def validation_guard_eligibility(
    metrics_by_suite: dict[str, dict[str, object]],
    initial_metrics_by_suite: dict[str, dict[str, object]],
    evaluation_config: dict,
) -> tuple[dict[str, object], bool]:
    """Evaluate strict, namespaced promotion guards for dual validation suites.

    Each dotted metric path accepts exactly one of ``min``, ``max``,
    ``max_regression``, ``max_drop``, ``min_improvement``, or ``min_gain``.
    Relative guards must explicitly name ``baseline: initial`` and compare only
    within the same validation suite.
    """

    guard_config = evaluation_config.get("validation_guards", {})
    if not isinstance(guard_config, dict):
        raise ValueError("evaluation.validation_guards must be a mapping")
    supported_comparisons = {
        "min",
        "max",
        "max_regression",
        "max_drop",
        "min_improvement",
        "min_gain",
    }
    details: dict[str, object] = {}
    all_pass = True
    for suite, suite_config in guard_config.items():
        suite_name = str(suite)
        if suite_name not in metrics_by_suite:
            raise KeyError(f"Validation guard suite {suite_name!r} has no metrics")
        if not isinstance(suite_config, dict) or not suite_config:
            raise ValueError(
                f"Validation guard suite {suite_name!r} must be a non-empty mapping"
            )
        suite_details: dict[str, object] = {}
        suite_pass = True
        for metric_path, raw_spec in suite_config.items():
            path = str(metric_path)
            if not isinstance(raw_spec, dict):
                raise ValueError(
                    f"Validation guard {suite_name}.{path} must be a mapping"
                )
            comparisons = supported_comparisons & set(raw_spec)
            if len(comparisons) != 1:
                raise ValueError(
                    f"Validation guard {suite_name}.{path} must define exactly one "
                    f"of {sorted(supported_comparisons)}"
                )
            comparison = next(iter(comparisons))
            allowed_keys = {comparison}
            if comparison in {
                "max_regression",
                "max_drop",
                "min_improvement",
                "min_gain",
            }:
                allowed_keys.add("baseline")
                if raw_spec.get("baseline") != "initial":
                    raise ValueError(
                        f"Relative validation guard {suite_name}.{path} must set "
                        "baseline: initial"
                    )
            elif "baseline" in raw_spec:
                raise ValueError(
                    f"Absolute validation guard {suite_name}.{path} cannot define "
                    "a baseline"
                )
            unknown_keys = set(raw_spec) - allowed_keys
            if unknown_keys:
                raise ValueError(
                    f"Validation guard {suite_name}.{path} has unsupported keys: "
                    f"{sorted(unknown_keys)}"
                )
            threshold = float(raw_spec[comparison])
            current = _metric_at_path(metrics_by_suite[suite_name], path)
            reference: float | None = None
            delta: float | None = None
            if comparison == "min":
                passed = current >= threshold
            elif comparison == "max":
                passed = current <= threshold
            else:
                if suite_name not in initial_metrics_by_suite:
                    raise KeyError(
                        f"Relative validation guard suite {suite_name!r} has no "
                        "initial baseline"
                    )
                reference = _metric_at_path(
                    initial_metrics_by_suite[suite_name], path
                )
                delta = current - reference
                if comparison == "max_regression":
                    passed = delta <= threshold
                elif comparison == "max_drop":
                    passed = delta >= -threshold
                elif comparison == "min_improvement":
                    passed = delta <= -threshold
                else:
                    passed = delta >= threshold
            passed = bool(passed and np.isfinite(current))
            suite_details[path] = {
                "comparison": comparison,
                "threshold": threshold,
                "current": current,
                "reference": reference,
                "delta": delta,
                "passes": passed,
            }
            suite_pass &= passed
        details[suite_name] = {
            "passes": suite_pass,
            "guards": suite_details,
        }
        all_pass &= suite_pass
    return {
        "passes": all_pass,
        "suites": details,
    }, all_pass


def resolve_initial_checkpoint_metrics(
    model: DomainGatedSurfaceNet,
    validation_loader: DataLoader,
    criterion: MultiDomainSurfaceLoss,
    device: torch.device,
    *,
    precision: str,
    stored_metrics: dict[str, object] | None,
    recompute_on_current_validation: bool,
) -> tuple[dict[str, object] | None, str]:
    """Choose a validation baseline for guarding a warm-start checkpoint.

    Historical configurations keep using the metrics stored in their initial
    checkpoint.  A dataset-changing stage can explicitly request one read-only
    evaluation of the warm-start model on its own validation loader instead.
    """

    if not recompute_on_current_validation:
        return stored_metrics, "stored_checkpoint"
    _, current_validation_metrics = validate(
        model,
        validation_loader,
        criterion,
        device,
        precision=precision,
    )
    return current_validation_metrics, "current_validation"


def validate_initial_checkpoint_metric_strategy(
    *,
    dataset_kind: str,
    initial_checkpoint: str | Path | None,
    recompute_on_current_validation: bool,
) -> None:
    """Reject warm-start metric strategies that are unsafe across datasets."""

    if recompute_on_current_validation and not initial_checkpoint:
        raise ValueError(
            "evaluation.recompute_initial_metrics_on_current_validation requires "
            "model.initial_checkpoint"
        )
    if (
        dataset_kind in {"gamus", "mixed_replay"}
        and initial_checkpoint
        and not recompute_on_current_validation
    ):
        raise ValueError(
            "A GAMUS warm start (including mixed replay) must set "
            "evaluation.recompute_initial_metrics_on_current_validation: true "
            "so regression guards use a GAMUS validation baseline"
        )


def validate_mixed_replay_training_config(
    *,
    dataset_kind: str,
    initial_checkpoint: str | Path | None,
    training_config: dict,
) -> None:
    """Validate either the legacy semantic repair or safe height replay.

    ``semantic_repair`` is the historical Stage-2 path and remains the default
    so every existing experiment keeps identical behaviour.  The separate
    ``height_supervision`` path exists for a corrected multi-source experiment:
    GAMUS, strictly masked HighBuild, and OpenCanopy can all supervise the same
    final height heads while the shared representation and identification heads
    remain frozen.  Keeping this allow-list here prevents a configuration typo
    from silently turning the safety replay into broad fine-tuning.
    """

    if dataset_kind != "mixed_replay":
        return
    if not initial_checkpoint:
        raise ValueError("Mixed replay Stage 2 requires model.initial_checkpoint")
    objective = str(training_config.get("mixed_replay_objective", "semantic_repair"))
    if objective not in {"semantic_repair", "height_supervision"}:
        raise ValueError(
            "training.mixed_replay_objective must be 'semantic_repair' or "
            "'height_supervision'"
        )
    if objective == "height_supervision":
        raw_groups = training_config.get("parameter_groups")
        if not isinstance(raw_groups, list) or not raw_groups:
            raise ValueError(
                "Height-supervision mixed replay requires explicit parameter_groups"
            )
        allowed_prefixes = {
            "base_model.height_head.",
            "canopy_height_head.",
            "refinement_strength_head.",
        }
        configured_prefixes = {
            str(prefix)
            for group in raw_groups
            if isinstance(group, dict)
            for prefix in group.get("prefixes", [])
        }
        unsafe = configured_prefixes - allowed_prefixes
        if unsafe:
            raise ValueError(
                "Height-supervision mixed replay may only train the sealed final "
                f"height/fusion heads; unsafe prefixes: {sorted(unsafe)}"
            )
        if not {
            "base_model.height_head.",
            "canopy_height_head.",
        }.issubset(configured_prefixes):
            raise ValueError(
                "Height-supervision mixed replay requires both base_model.height_head. "
                "and canopy_height_head."
            )
        if training_config.get("trainable_adapter_prefixes"):
            raise ValueError(
                "Height-supervision mixed replay cannot use trainable_adapter_prefixes"
            )
        if int(training_config.get("freeze_base_epochs", 0)) != 0:
            raise ValueError(
                "Height-supervision mixed replay requires freeze_base_epochs: 0; "
                "parameter_groups define the frozen scope"
            )
        return
    prefixes = tuple(
        str(prefix) for prefix in training_config.get("trainable_adapter_prefixes", [])
    )
    if prefixes != ("domain_head.",):
        raise ValueError(
            "Mixed replay Stage 2 requires "
            "training.trainable_adapter_prefixes: [domain_head.]"
        )
    if float(training_config.get("base_learning_rate_multiplier", 0.0)) != 0.0:
        raise ValueError(
            "Mixed replay Stage 2 requires base_learning_rate_multiplier: 0.0"
        )
    if int(training_config.get("freeze_base_epochs", 0)) < int(
        training_config["epochs"]
    ):
        raise ValueError(
            "Mixed replay Stage 2 requires freeze_base_epochs >= epochs"
        )


def uses_stage2_semantic_repair(
    dataset_kind: str,
    training_config: Mapping[str, object],
) -> bool:
    """Return whether mixed replay should use its historical semantic loss."""

    return (
        dataset_kind == "mixed_replay"
        and str(training_config.get("mixed_replay_objective", "semantic_repair"))
        == "semantic_repair"
    )


def validate_six_class_training_config(
    *,
    dataset_kind: str,
    model_config: dict,
    training_config: dict,
) -> None:
    """Fail closed when the optional GAMUS classifier is partly configured."""

    class_count = int(model_config.get("fine_semantic_classes", 0))
    head_type = str(model_config.get("fine_semantic_head_type", "linear"))
    supported_head_types = {
        "linear",
        "spatial_refined",
        "hierarchical_vegetation",
    }
    if head_type not in supported_head_types:
        raise ValueError(
            "model.fine_semantic_head_type must be 'linear', "
            "'spatial_refined', or 'hierarchical_vegetation'"
        )
    loss_weight = float(training_config.get("fine_semantic_weight", 0.0))
    split_weight = float(
        training_config.get("fine_semantic_vegetation_split_weight", 0.0)
    )
    if not math.isfinite(split_weight) or split_weight < 0.0:
        raise ValueError(
            "training.fine_semantic_vegetation_split_weight must be finite "
            "and nonnegative"
        )
    if class_count == 0:
        if loss_weight != 0.0 or split_weight != 0.0:
            raise ValueError(
                "fine-semantic losses require model.fine_semantic_classes: 6"
            )
        return
    if class_count != len(GAMUS_SIX_CLASS_NAMES):
        raise ValueError("model.fine_semantic_classes must be 0 or 6")
    if dataset_kind != "gamus":
        raise ValueError(
            "Six-class training currently requires data.dataset: gamus so every "
            "sample has an authenticated fine-class mask"
        )
    if loss_weight <= 0.0:
        raise ValueError(
            "An enabled six-class head requires a positive fine_semantic_weight"
        )
    if head_type == "hierarchical_vegetation" and split_weight <= 0.0:
        raise ValueError(
            "hierarchical_vegetation requires a positive "
            "fine_semantic_vegetation_split_weight"
        )
    if head_type != "hierarchical_vegetation" and split_weight != 0.0:
        raise ValueError(
            "fine_semantic_vegetation_split_weight is only valid for the "
            "hierarchical_vegetation head"
        )
    weights = training_config.get("fine_semantic_class_weights")
    if weights is not None and (
        not isinstance(weights, list)
        or len(weights) != len(GAMUS_SIX_CLASS_NAMES)
        or any(float(value) <= 0.0 for value in weights)
    ):
        raise ValueError(
            "fine_semantic_class_weights must contain six positive values"
        )
    groups = training_config.get("parameter_groups")
    if isinstance(groups, list):
        trainable_prefixes = {
            str(prefix)
            for group in groups
            if isinstance(group, dict)
            for prefix in group.get("prefixes", [])
        }
        if not any(
            "fine_semantic_head.".startswith(prefix)
            or prefix.startswith("fine_semantic_head.")
            for prefix in trainable_prefixes
        ):
            raise ValueError(
                "Explicit parameter groups must include fine_semantic_head."
            )


def initial_checkpoint_regressions(
    current_metrics: dict[str, object],
    initial_metrics: dict[str, object],
    evaluation_config: dict,
) -> tuple[dict[str, float], bool]:
    """Compare like-for-like validation metrics for an initial-checkpoint guard."""

    for group_name in ("landscapes", "domains"):
        current_group = current_metrics.get(group_name)
        initial_group = initial_metrics.get(group_name)
        if not isinstance(current_group, dict) or not isinstance(initial_group, dict):
            raise ValueError(
                f"Initial checkpoint guard requires {group_name!r} metrics in both "
                "the current and baseline validation results"
            )
        current_names = {str(name) for name in current_group}
        initial_names = {str(name) for name in initial_group}
        if current_names != initial_names:
            raise ValueError(
                "Initial checkpoint validation metrics are incompatible with the "
                f"current validation dataset: {group_name} are "
                f"current={sorted(current_names)}, checkpoint={sorted(initial_names)}. "
                "For a dataset-changing warm start, set "
                "evaluation.recompute_initial_metrics_on_current_validation: true."
            )

    current_values = {"overall": float(current_metrics["rmse_m"])}
    reference_values = {"overall": float(initial_metrics["rmse_m"])}
    for group_name in ("landscapes", "domains"):
        current_group = current_metrics[group_name]
        initial_group = initial_metrics[group_name]
        assert isinstance(current_group, dict) and isinstance(initial_group, dict)
        for name in sorted(current_group):
            metric_name = str(name)
            if metric_name in current_values:
                raise ValueError(
                    f"Duplicate initial-checkpoint guard metric name {metric_name!r}"
                )
            current_item = current_group[name]
            initial_item = initial_group[name]
            if not isinstance(current_item, dict) or not isinstance(initial_item, dict):
                raise ValueError(
                    f"Initial checkpoint guard metric {metric_name!r} must be a mapping"
                )
            current_values[metric_name] = float(current_item["rmse_m"])
            reference_values[metric_name] = float(initial_item["rmse_m"])

    regressions: dict[str, float] = {}
    passes_guard = True
    for name, current_value in current_values.items():
        regressions[name] = current_value - reference_values[name]
        limit = float(
            evaluation_config.get(f"max_initial_{name}_rmse_regression_m", 0.10)
        )
        passes_guard &= regressions[name] <= limit

    specialist_guards = (
        (
            "building_expert",
            "building_expert_rmse_m",
            "initial_building_expert_rmse_m",
            "max_initial_building_expert_rmse_regression_m",
        ),
        (
            "canopy_expert",
            "canopy_expert_rmse_m",
            "initial_canopy_expert_rmse_m",
            "max_initial_canopy_expert_rmse_regression_m",
        ),
        (
            "building_router_error",
            "building_router_error",
            "initial_building_router_error",
            "max_initial_building_router_error_regression",
        ),
    )
    for regression_name, metric_name, reference_name, limit_name in specialist_guards:
        # Existing configurations opt into the building-expert guard with an
        # explicit reference. New dataset-changing stages can instead provide
        # only a regression limit and use the freshly recomputed baseline.
        if reference_name not in evaluation_config and limit_name not in evaluation_config:
            continue
        if metric_name not in current_metrics:
            raise ValueError(
                f"Initial checkpoint guard requires current metric {metric_name!r}"
            )
        if reference_name in evaluation_config:
            reference_value = float(evaluation_config[reference_name])
        else:
            if metric_name not in initial_metrics:
                raise ValueError(
                    f"Initial checkpoint guard requires baseline metric {metric_name!r}"
                )
            reference_value = float(initial_metrics[metric_name])
        regression = float(current_metrics[metric_name]) - reference_value
        regressions[regression_name] = regression
        passes_guard &= regression <= float(evaluation_config.get(limit_name, 0.0))
    return regressions, passes_guard


def selection_result(
    metrics: dict[str, object],
    evaluation_config: dict,
    *,
    metrics_by_suite: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Resolve one selection value while keeping a lower score always better."""

    primary_selection = evaluation_config.get("primary_selection")
    if primary_selection is not None:
        if not isinstance(primary_selection, dict):
            raise ValueError("evaluation.primary_selection must be a mapping")
        required = {"suite", "metric", "mode"}
        missing = required - set(primary_selection)
        unknown = set(primary_selection) - required
        if missing or unknown:
            raise ValueError(
                "evaluation.primary_selection requires exactly suite, metric, and "
                f"mode; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        suite = str(primary_selection["suite"]).strip()
        metric = str(primary_selection["metric"]).strip()
        mode = str(primary_selection["mode"]).strip().lower()
        if not suite or not metric:
            raise ValueError("primary_selection suite and metric must not be empty")
        if mode not in {"min", "max"}:
            raise ValueError("primary_selection mode must be 'min' or 'max'")
        if metrics_by_suite is None or suite not in metrics_by_suite:
            raise KeyError(
                f"Primary selection suite {suite!r} has no validation metrics"
            )
        value = _metric_at_path(metrics_by_suite[suite], metric)
        if not np.isfinite(value):
            raise ValueError(
                f"Primary selection metric {suite}.{metric} must be finite"
            )
        return {
            "suite": suite,
            "metric": metric,
            "mode": mode,
            "value": value,
            "score": value if mode == "min" else -value,
            "lower_score_is_better": True,
        }

    configured_primary_selection = evaluation_config.get("primary_selection")
    if isinstance(configured_primary_selection, dict):
        selection_metric = (
            f"{configured_primary_selection.get('suite')}."
            f"{configured_primary_selection.get('metric')}"
        )
        selection_mode = str(configured_primary_selection.get("mode", ""))
    else:
        selection_metric = str(
            evaluation_config.get("primary_metric", "landscape_macro_rmse_m")
        )
        selection_mode = "min"
    if selection_metric not in metrics:
        raise KeyError(
            f"Selection metric {selection_metric!r} is missing; "
            f"available: {sorted(metrics)}"
        )
    domains = metrics["domains"]
    experts = metrics["experts"]
    if not isinstance(domains, dict) or not isinstance(experts, dict):
        raise ValueError("Selection metrics require domain and expert mappings")
    score = (
        float(metrics[selection_metric])
        + float(evaluation_config.get("building_selection_weight", 0.0))
        * float(domains["building"]["rmse_m"])
        + float(evaluation_config.get("vegetation_selection_weight", 0.0))
        * float(domains["vegetation"]["rmse_m"])
        + float(evaluation_config.get("building_expert_selection_weight", 0.0))
        * float(experts["building"]["rmse_m"])
    )
    return {
        "suite": "flat",
        "metric": selection_metric,
        "mode": "min",
        "value": float(metrics[selection_metric]),
        "score": score,
        "lower_score_is_better": True,
    }


def selection_score(
    metrics: dict[str, object],
    evaluation_config: dict,
    *,
    metrics_by_suite: dict[str, dict[str, object]] | None = None,
) -> float:
    """Apply the configured checkpoint-selection formula to one validation result."""

    return float(
        selection_result(
            metrics,
            evaluation_config,
            metrics_by_suite=metrics_by_suite,
        )["score"]
    )


def resolve_initial_best_selection(
    initial_metrics: dict[str, object] | None,
    initial_metrics_source: str,
    evaluation_config: dict,
    *,
    initial_metrics_by_suite: dict[str, dict[str, object]] | None = None,
) -> float:
    """Seed selection safely without changing historical stored-metric behavior."""

    if "primary_selection" in evaluation_config:
        if (
            initial_metrics is None
            or initial_metrics_source != "current_validation"
            or not initial_metrics_by_suite
        ):
            raise ValueError(
                "evaluation.primary_selection requires a current-validation "
                "warm-start baseline; refusing an unguarded first-epoch selection"
            )
        return selection_score(
            initial_metrics,
            evaluation_config,
            metrics_by_suite=initial_metrics_by_suite,
        )
    if "initial_selection_value_m" in evaluation_config:
        return float(evaluation_config["initial_selection_value_m"])
    if initial_metrics is not None and initial_metrics_source == "current_validation":
        return selection_score(initial_metrics, evaluation_config)
    return float("inf")


def selection_improved(
    selection_value: float,
    best_selection: float,
    *,
    eligible: bool,
) -> bool:
    """Require both hard-guard eligibility and a strict baseline improvement."""

    return bool(eligible and selection_value < best_selection)


def _atomic_checkpoint(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    model: DomainGatedSurfaceNet,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    metrics: dict[str, object],
    config: dict,
    training_state: dict[str, object],
    validation_metrics_by_suite: dict[str, dict[str, object]] | None = None,
    validation_loss_by_suite: dict[str, dict[str, float]] | None = None,
) -> None:
    _atomic_checkpoint(
        {
            "model_type": (
                "domain_gated_surface_v4_six_class"
                if model.fine_semantic_head is not None
                else (
                    "domain_gated_surface_v3"
                    if model.fusion_mode == "calibrated_surface"
                    else (
                        "domain_gated_surface_v2"
                        if model.fusion_mode == "protected_vegetation"
                        else "domain_gated_surface_v1"
                    )
                )
            ),
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "metrics": metrics,
            "validation_metrics_by_suite": validation_metrics_by_suite,
            "validation_loss_by_suite": validation_loss_by_suite,
            "config": config,
            "training_state": training_state,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment_config = config["experiment"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    evaluation_config = config.get("evaluation", {})

    seed = int(experiment_config["seed"])
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multidomain training")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    dataset_kind = str(data_config.get("dataset", "manifest")).strip().lower()
    validate_six_class_training_config(
        dataset_kind=dataset_kind,
        model_config=model_config,
        training_config=training_config,
    )
    initial_checkpoint = model_config.get("initial_checkpoint")
    recompute_initial_metrics = bool(
        evaluation_config.get(
            "recompute_initial_metrics_on_current_validation", False
        )
    )
    initial_model_overrides = evaluation_config.get("initial_model_overrides")
    if initial_model_overrides and not initial_checkpoint:
        raise ValueError(
            "evaluation.initial_model_overrides requires model.initial_checkpoint"
        )
    validate_initial_checkpoint_metric_strategy(
        dataset_kind=dataset_kind,
        initial_checkpoint=initial_checkpoint,
        recompute_on_current_validation=recompute_initial_metrics,
    )
    validate_mixed_replay_training_config(
        dataset_kind=dataset_kind,
        initial_checkpoint=initial_checkpoint,
        training_config=training_config,
    )
    validation_loaders_by_suite: dict[str, DataLoader] | None = None
    if dataset_kind == "manifest":
        # This remains the original path for every existing configuration.
        train_records = load_surface_manifest(data_config["train_manifest"])
        validation_records = load_surface_manifest(data_config["val_manifest"])
        test_records = load_surface_manifest(data_config["test_manifest"])
        assert_surface_regions_disjoint(
            train_records, validation_records, test_records
        )
        training_landscapes = {
            str(name) for name in data_config.get("training_landscapes", [])
        }
        if training_landscapes:
            train_records = [
                record
                for record in train_records
                if record.landscape in training_landscapes
            ]
            if not train_records:
                raise ValueError(
                    f"No training records match landscapes {sorted(training_landscapes)}"
                )
            print(
                f"Restricted training to {len(train_records)} records from "
                f"{sorted(training_landscapes)}"
            )
        train_loader = make_loader(
            train_records, data_config, training_config, training=True
        )
        validation_loader = make_loader(
            validation_records, data_config, training_config, training=False
        )
    elif dataset_kind == "gamus":
        if data_config.get("training_landscapes"):
            raise ValueError(
                "data.training_landscapes is manifest-only; GAMUS uses exact "
                "pixel-level semantic classes"
            )
        if "root" not in data_config:
            raise ValueError("data.root is required when data.dataset is 'gamus'")
        gamus_root = Path(data_config["root"]).expanduser().resolve()
        gamus_datasets = make_gamus_development_datasets(
            gamus_root,
            data_config,
        )
        validate_gamus_splits(
            gamus_datasets,
            require_complete_official_splits=bool(
                data_config.get("require_complete_official_splits", True)
            ),
        )
        print(
            f"Loaded native GAMUS splits from {gamus_root}: "
            + ", ".join(
                f"{split}={len(gamus_datasets[split].records)}"
                for split in GAMUS_DEVELOPMENT_SPLITS
            )
        )
        train_loader = make_gamus_loader(
            gamus_datasets["train"], data_config, training_config, training=True
        )
        validation_loader = make_gamus_loader(
            gamus_datasets["val"], data_config, training_config, training=False
        )
        validation_loaders_by_suite = {"gamus": validation_loader}
    elif dataset_kind == "mixed_replay":
        required_data_keys = {
            "root",
            "train_manifest",
            "val_manifest",
            "train_samples_per_epoch",
        }
        missing_data_keys = required_data_keys - set(data_config)
        if missing_data_keys:
            raise ValueError(
                "Mixed replay is missing data settings: "
                f"{sorted(missing_data_keys)}"
            )
        gamus_root = Path(data_config["root"]).expanduser().resolve()
        gamus_datasets = make_gamus_development_datasets(
            gamus_root,
            data_config,
            use_train_samples_per_epoch=False,
        )
        validate_gamus_splits(
            gamus_datasets,
            require_complete_official_splits=bool(
                data_config.get("require_complete_official_splits", True)
            ),
        )
        legacy_train_records = load_surface_manifest(data_config["train_manifest"])
        legacy_validation_records = load_surface_manifest(data_config["val_manifest"])
        assert_surface_regions_disjoint(
            legacy_train_records,
            legacy_validation_records,
        )
        training_landscapes = {
            str(name) for name in data_config.get("training_landscapes", [])
        }
        if training_landscapes:
            legacy_train_records = [
                record
                for record in legacy_train_records
                if record.landscape in training_landscapes
            ]
            if not legacy_train_records:
                raise ValueError(
                    "No legacy mixed-replay records match landscapes "
                    f"{sorted(training_landscapes)}"
                )
        legacy_train_dataset = make_surface_dataset(
            legacy_train_records,
            data_config,
            training=True,
            use_train_samples_per_epoch=False,
        )
        train_loader = make_mixed_replay_loader(
            gamus_datasets["train"],
            legacy_train_dataset,
            data_config,
            training_config,
            seed=seed,
        )
        gamus_validation_loader = make_gamus_loader(
            gamus_datasets["val"], data_config, training_config, training=False
        )
        legacy_validation_loader = make_loader(
            legacy_validation_records,
            data_config,
            training_config,
            training=False,
        )
        validation_loaders_by_suite = {
            "gamus": gamus_validation_loader,
            "legacy": legacy_validation_loader,
        }
        # The flat compatibility view for mixed replay is always GAMUS.
        validation_loader = gamus_validation_loader
        sampler = train_loader.sampler
        assert isinstance(sampler, SourceRatioSampler)
        print(
            f"Loaded mixed replay: GAMUS train={len(gamus_datasets['train'])}, "
            f"legacy train={len(legacy_train_dataset)}, "
            f"epoch quotas={sampler.source_counts}"
        )
    else:
        raise ValueError(
            f"Unsupported data.dataset {dataset_kind!r}; expected 'manifest', "
            "'gamus', or 'mixed_replay'"
        )

    base_checkpoint = Path(model_config["base_checkpoint"])
    base_model, _ = load_predictor(base_checkpoint, device="cpu")
    model = DomainGatedSurfaceNet(
        base_model,
        hidden_channels=int(model_config.get("hidden_channels", 48)),
        maximum_building_residual_m=float(
            model_config.get("maximum_building_residual_m", 30.0)
        ),
        initial_canopy_height_m=float(model_config.get("initial_canopy_height_m", 8.0)),
        initial_refinement_strength=float(
            model_config.get("initial_refinement_strength", 0.08)
        ),
        fusion_mode=str(model_config.get("fusion_mode", "legacy")),
        building_protection_power=float(
            model_config.get("building_protection_power", 2.0)
        ),
        building_fusion_min_height_m=float(
            model_config.get("building_fusion_min_height_m", 10.0)
        ),
        building_fusion_temperature_m=float(
            model_config.get("building_fusion_temperature_m", 2.0)
        ),
        building_fusion_score_threshold=float(
            model_config.get("building_fusion_score_threshold", 0.5)
        ),
        building_fusion_score_temperature=float(
            model_config.get("building_fusion_score_temperature", 0.05)
        ),
        building_fusion_strength=float(
            model_config.get("building_fusion_strength", 1.0)
        ),
        vegetation_fusion_temperature=float(
            model_config.get("vegetation_fusion_temperature", 1.0)
        ),
        vegetation_expert_fusion_threshold=float(
            model_config.get("vegetation_expert_fusion_threshold", 1.0)
        ),
        vegetation_expert_fusion_strength=float(
            model_config.get("vegetation_expert_fusion_strength", 0.0)
        ),
        fine_semantic_classes=int(model_config.get("fine_semantic_classes", 0)),
        fine_semantic_head_type=str(
            model_config.get("fine_semantic_head_type", "linear")
        ),
        freeze_base=True,
    ).to(device)
    initial_metrics: dict[str, object] | None = None
    initial_metrics_by_suite: dict[str, dict[str, object]] = {}
    teacher: FrozenDomainHeadTeacher | None = None
    if initial_checkpoint:
        initial_payload = torch.load(
            Path(initial_checkpoint), map_location=device, weights_only=False
        )
        model.load_legacy_compatible_state_dict(initial_payload["model"])
        initial_metrics = initial_payload.get("metrics")
        print(f"Warm-started adapter from {Path(initial_checkpoint).resolve()}")
    if uses_stage2_semantic_repair(dataset_kind, training_config):
        teacher = FrozenDomainHeadTeacher(model.domain_head).to(device)
    explicit_parameter_groups = configure_explicit_parameter_groups(
        model, training_config
    )
    if explicit_parameter_groups is None:
        trainable_adapter_prefixes = tuple(
            str(prefix)
            for prefix in training_config.get("trainable_adapter_prefixes", [])
        )
        if trainable_adapter_prefixes:
            for name, parameter in model.named_parameters():
                if not name.startswith("base_model."):
                    parameter.requires_grad_(
                        any(
                            name.startswith(prefix)
                            for prefix in trainable_adapter_prefixes
                        )
                    )
            trainable_names = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            if not trainable_names:
                raise ValueError(
                    "trainable_adapter_prefixes selected no model parameters"
                )
            print(f"Trainable adapter tensors: {trainable_names}")
        learning_rate = float(training_config["learning_rate"])
        base_lr_multiplier = float(
            training_config.get("base_learning_rate_multiplier", 0.02)
        )
        optimizer_groups: list[dict[str, object]] = [
            {
                "params": model.base_model.parameters(),
                "lr": learning_rate * base_lr_multiplier,
                "group_name": "protected_base",
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith("base_model.") and parameter.requires_grad
                ],
                "lr": learning_rate,
                "group_name": "domain_adapter",
            },
        ]
    else:
        optimizer_groups = explicit_parameter_groups
    optimizer = AdamW(
        optimizer_groups,
        weight_decay=float(training_config.get("weight_decay", 0.02)),
    )
    epochs = int(training_config["epochs"])
    scheduler, scheduler_interval = make_learning_rate_scheduler(
        optimizer,
        training_config,
        batches_per_epoch=len(train_loader),
    )
    precision = str(training_config.get("precision", "bf16")).lower()
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("training.precision must be fp32, bf16, or fp16")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured GPU does not support bf16 training")
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    criterion = MultiDomainSurfaceLoss(
        huber_delta_m=float(training_config.get("huber_delta_m", 2.0)),
        height_weight=float(training_config.get("height_weight", 1.0)),
        semantic_weight=float(training_config.get("semantic_weight", 0.3)),
        fine_semantic_weight=float(
            training_config.get("fine_semantic_weight", 0.0)
        ),
        fine_semantic_vegetation_split_weight=float(
            training_config.get(
                "fine_semantic_vegetation_split_weight", 0.0
            )
        ),
        building_weight=float(training_config.get("building_weight", 1.0)),
        building_mse_weight=float(training_config.get("building_mse_weight", 0.0)),
        tall_building_weight=float(
            training_config.get("tall_building_weight", 0.0)
        ),
        tall_building_threshold_m=float(
            training_config.get("tall_building_threshold_m", 20.0)
        ),
        canopy_weight=float(training_config.get("canopy_weight", 1.0)),
        canopy_mse_weight=float(training_config.get("canopy_mse_weight", 0.0)),
        tall_canopy_weight=float(training_config.get("tall_canopy_weight", 0.0)),
        tall_canopy_threshold_m=float(
            training_config.get("tall_canopy_threshold_m", 15.0)
        ),
        fused_building_weight=float(
            training_config.get("fused_building_weight", 0.0)
        ),
        fused_building_mse_weight=float(
            training_config.get("fused_building_mse_weight", 0.0)
        ),
        fused_vegetation_weight=float(
            training_config.get("fused_vegetation_weight", 0.0)
        ),
        fused_tall_building_weight=float(
            training_config.get("fused_tall_building_weight", 0.0)
        ),
        fused_tall_vegetation_weight=float(
            training_config.get("fused_tall_vegetation_weight", 0.0)
        ),
        building_distillation_weight=float(
            training_config.get("building_distillation_weight", 0.0)
        ),
        ground_suppression_weight=float(
            training_config.get("ground_suppression_weight", 0.25)
        ),
        refinement_weight=float(training_config.get("refinement_weight", 0.0)),
        semantic_class_weights=training_config.get("semantic_class_weights"),
        fine_semantic_class_weights=training_config.get(
            "fine_semantic_class_weights"
        ),
        fine_semantic_focal_gamma=float(
            training_config.get("fine_semantic_focal_gamma", 0.0)
        ),
        uncertainty_weight=float(training_config.get("uncertainty_weight", 0.0)),
    ).to(device)
    training_criterion: MultiDomainSurfaceLoss | Stage2SemanticRepairLoss = criterion
    if uses_stage2_semantic_repair(dataset_kind, training_config):
        training_criterion = Stage2SemanticRepairLoss(
            semantic_weight=float(training_config.get("semantic_weight", 1.0)),
            gamus_semantic_weight=float(
                training_config.get("gamus_semantic_weight", 0.75)
            ),
            legacy_semantic_weight=float(
                training_config.get("legacy_semantic_weight", 0.25)
            ),
            gamus_class_weights=training_config.get(
                "gamus_class_weights",
                training_config.get(
                    "gamus_semantic_class_weights", [1.0, 1.5, 0.75]
                ),
            ),
            legacy_class_weights=training_config.get(
                "legacy_class_weights",
                training_config.get(
                    "legacy_semantic_class_weights", [1.0, 1.0, 1.0]
                ),
            ),
            legacy_urban_ground_pixel_weight=float(
                training_config.get("legacy_urban_ground_pixel_weight", 0.35)
            ),
            distillation_temperature=float(
                training_config.get("distillation_temperature", 2.0)
            ),
            gamus_distillation_weight=float(
                training_config.get("gamus_distillation_weight", 0.20)
            ),
            legacy_fused_height_weight=float(
                training_config.get("legacy_fused_height_weight", 0.025)
            ),
            huber_delta_m=float(training_config.get("huber_delta_m", 2.0)),
        ).to(device)
    initial_metrics_source = "none"
    if initial_checkpoint:
        if recompute_initial_metrics:
            print(
                "Evaluating the warm-start checkpoint once on the current "
                "validation dataset for like-for-like regression guards"
            )
        with temporary_fusion_evaluation_overrides(
            model,
            initial_model_overrides,
        ):
            if validation_loaders_by_suite is not None:
                if not recompute_initial_metrics:
                    raise AssertionError(
                        "Named validation suites require current-suite baselines"
                    )
                _, initial_metrics_by_suite = validate_suites(
                    model,
                    validation_loaders_by_suite,
                    criterion,
                    device,
                    precision=precision,
                )
                initial_metrics = initial_metrics_by_suite["gamus"]
                initial_metrics_source = "current_validation"
            else:
                initial_metrics, initial_metrics_source = (
                    resolve_initial_checkpoint_metrics(
                        model,
                        validation_loader,
                        criterion,
                        device,
                        precision=precision,
                        stored_metrics=initial_metrics,
                        recompute_on_current_validation=recompute_initial_metrics,
                    )
                )
        if initial_metrics is not None:
            print(f"Initial-checkpoint guard baseline: {initial_metrics_source}")

    selection_metric = str(
        evaluation_config.get("primary_metric", "landscape_macro_rmse_m")
    )
    maximum_urban_regression = float(
        evaluation_config.get("max_urban_rmse_regression_m", 0.25)
    )
    maximum_building_regression = float(
        evaluation_config.get("max_building_rmse_regression_m", 0.25)
    )
    best_selection = resolve_initial_best_selection(
        initial_metrics,
        initial_metrics_source,
        evaluation_config,
        initial_metrics_by_suite=initial_metrics_by_suite,
    )
    initial_selection_metadata: dict[str, object] | None = None
    if (
        "primary_selection" in evaluation_config
        and initial_metrics is not None
        and initial_metrics_by_suite
    ):
        initial_selection_metadata = selection_result(
            initial_metrics,
            evaluation_config,
            metrics_by_suite=initial_metrics_by_suite,
        )
    epochs_without_improvement = 0
    early_stopping_min_epochs = int(
        training_config.get("early_stopping_min_epochs", 1)
    )
    if early_stopping_min_epochs < 1:
        raise ValueError("training.early_stopping_min_epochs must be positive")
    start_epoch = 1
    if args.resume:
        resume_path = args.resume.resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("config") != config:
            raise ValueError("Resume config differs from the immutable checkpoint config")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint.get("training_state", {})
        best_selection = float(state.get("best_selection", float("inf")))
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        experiment_dir = resume_path.parent
        rng = checkpoint.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            torch.cuda.set_rng_state_all([item.cpu() for item in rng["cuda"]])
        print(f"Resuming {experiment_dir} at epoch {start_epoch}")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_dir = Path(experiment_config["output_root"]) / (
            f"{timestamp}_{experiment_config['name']}"
        )
        experiment_dir.mkdir(parents=True, exist_ok=False)
        (experiment_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
    if initial_metrics is not None and initial_metrics_source == "current_validation":
        baseline_path = experiment_dir / "initial_checkpoint_validation_metrics.json"
        if not baseline_path.exists():
            baseline_path.write_text(
                json.dumps(
                    {
                        "source": initial_metrics_source,
                        **(
                            {"validation_metrics_by_suite": initial_metrics_by_suite}
                            if initial_metrics_by_suite
                            else {"validation_metrics": initial_metrics}
                        ),
                        "selection": initial_selection_metadata,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    history_path = experiment_dir / "metrics.jsonl"
    freeze_base_epochs = int(training_config.get("freeze_base_epochs", 2))

    for epoch in range(start_epoch, epochs + 1):
        if explicit_parameter_groups is None:
            model.set_base_trainable(epoch > freeze_base_epochs)
        if isinstance(train_loader.sampler, SourceRatioSampler):
            train_loader.sampler.set_epoch(epoch)
        train_losses = train_epoch(
            model,
            train_loader,
            training_criterion,
            optimizer,
            scaler,
            device,
            precision=precision,
            accumulation=int(training_config.get("gradient_accumulation", 1)),
            max_gradient_norm=float(training_config.get("max_gradient_norm", 1.0)),
            teacher=teacher,
            scheduler=scheduler,
            scheduler_interval=scheduler_interval,
        )
        validation_loss_by_suite: dict[str, dict[str, float]] | None = None
        validation_metrics_by_suite: dict[str, dict[str, object]] | None = None
        if validation_loaders_by_suite is not None:
            validation_loss_by_suite, validation_metrics_by_suite = validate_suites(
                model,
                validation_loaders_by_suite,
                criterion,
                device,
                precision=precision,
            )
            # Keep the historical flat fields useful for existing analysis tools.
            validation_losses = validation_loss_by_suite["gamus"]
            metrics = validation_metrics_by_suite["gamus"]
        else:
            validation_losses, metrics = validate(
                model, validation_loader, criterion, device, precision=precision
            )
        if scheduler_interval == "epoch":
            scheduler.step()
        selection = selection_result(
            metrics,
            evaluation_config,
            metrics_by_suite=validation_metrics_by_suite,
        )
        selection_value = float(selection["score"])
        metrics["guarded_selection_score_m"] = selection_value
        urban_regression = float(metrics.get("urban_rmse_regression_m", 0.0))
        building_regression = float(metrics.get("building_rmse_regression_m", 0.0))
        passes_protected_base_guard = (
            urban_regression <= maximum_urban_regression
            and building_regression <= maximum_building_regression
        )
        initial_regressions: dict[str, float] = {}
        passes_initial_guard = True
        if initial_metrics:
            initial_regressions, passes_initial_guard = (
                initial_checkpoint_regressions(
                    metrics,
                    initial_metrics,
                    evaluation_config,
                )
            )
            current_identification = metrics.get("semantic_identification")
            initial_identification = initial_metrics.get("semantic_identification")
            if isinstance(current_identification, dict) and isinstance(
                initial_identification, dict
            ):
                metrics["initial_checkpoint_semantic_identification_deltas"] = (
                    multiclass_metric_deltas(
                        current_identification,
                        initial_identification,
                    )
                )
        validation_guard_details: dict[str, object] = {
            "passes": True,
            "suites": {},
        }
        passes_validation_guards = True
        if validation_metrics_by_suite is not None:
            validation_guard_details, passes_validation_guards = (
                validation_guard_eligibility(
                    validation_metrics_by_suite,
                    initial_metrics_by_suite,
                    evaluation_config,
                )
            )
        passes_urban_guard = (
            passes_protected_base_guard
            and passes_initial_guard
            and passes_validation_guards
        )
        metrics["initial_checkpoint_regressions_m"] = initial_regressions
        metrics["passes_initial_checkpoint_guard"] = passes_initial_guard
        metrics["initial_checkpoint_metrics_source"] = initial_metrics_source
        metrics["passes_protected_base_guard"] = passes_protected_base_guard
        metrics["validation_guard_eligibility"] = validation_guard_details
        metrics["passes_validation_guards"] = passes_validation_guards
        metrics["passes_urban_regression_guard"] = passes_urban_guard
        metrics["max_urban_rmse_regression_m"] = maximum_urban_regression
        metrics["max_building_rmse_regression_m"] = maximum_building_regression
        improved = selection_improved(
            selection_value,
            best_selection,
            eligible=passes_urban_guard,
        )
        if improved:
            best_selection = selection_value
            epochs_without_improvement = 0
        elif epoch >= early_stopping_min_epochs:
            epochs_without_improvement += 1
        selection_metadata = {
            **selection,
            "eligible": passes_urban_guard,
            "improved": improved,
            "best_score": best_selection,
        }
        metrics["selection"] = selection_metadata
        state: dict[str, object] = {
            "selection_metric": str(selection["metric"]),
            "selection_mode": str(selection["mode"]),
            "selection": selection_metadata,
            "initial_selection": initial_selection_metadata,
            "best_selection": best_selection,
            "epochs_without_improvement": epochs_without_improvement,
            "base_trainable": model.base_trainable,
            "precision": precision,
            "passes_urban_regression_guard": passes_urban_guard,
            "passes_validation_guards": passes_validation_guards,
        }
        record = {
            "epoch": epoch,
            "base_trainable": model.base_trainable,
            "learning_rates": scheduler.get_last_lr(),
            "train_loss": train_losses,
            "validation_loss": validation_losses,
            "validation_metrics": metrics,
        }
        if validation_loss_by_suite is not None:
            record["validation_loss_by_suite"] = validation_loss_by_suite
        if validation_metrics_by_suite is not None:
            record["validation_metrics_by_suite"] = validation_metrics_by_suite
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2))
        if improved:
            save_checkpoint(
                experiment_dir / "checkpoint_best_landscape.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                metrics=metrics,
                config=config,
                training_state=state,
                validation_metrics_by_suite=validation_metrics_by_suite,
                validation_loss_by_suite=validation_loss_by_suite,
            )
        save_checkpoint(
            experiment_dir / "checkpoint_latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            metrics=metrics,
            config=config,
            training_state=state,
            validation_metrics_by_suite=validation_metrics_by_suite,
            validation_loss_by_suite=validation_loss_by_suite,
        )
        if epochs_without_improvement >= int(training_config.get("early_stopping_patience", 5)):
            print(f"Early stopping after epoch {epoch}")
            break


if __name__ == "__main__":
    main()
