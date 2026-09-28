"""Train an isolated residual metric-height decoder over a frozen app model.

The script supports a required training-only overfit proof and a resumable
multi-source development run.  It never reads an official test or external
holdout, never writes the application pointer, and never updates any tensor in
the protected model or classification path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Sampler, default_collate
import yaml

from msr.data.surface_dataset import load_surface_manifest
from msr.data.mixed_replay import SourceTaggedDataset, assert_matching_sample_contract
from msr.inference.predict import load_predictor
from msr.models.residual_height import ResidualHeightSurfaceNet
from msr.training.balanced_height_loss import SourceBalancedHeightLoss
from msr.training.losses import MultiDomainSurfaceLoss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_PATH = PROJECT_ROOT / "scripts" / "train_multidomain.py"
SPEC = importlib.util.spec_from_file_location("residual_height_shared", SHARED_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import shared trainer: {SHARED_PATH}")
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)

OVERFIT_REPORT = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "residual_height_v1_overfit" / "report.json"
)
VALIDATION_SMOKE_ROOT = (
    PROJECT_ROOT / "outputs" / "diagnostics" / "residual_height_v1_validation_smoke"
)
SEMANTIC_METRIC_KEYS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_iou",
)
SEMANTIC_OUTPUT_KEYS = (
    "domain_logits",
    "domain_probabilities",
    "building_logits",
    "protected_building_logits",
    "vegetation_logits",
    "fine_semantic_logits",
    "fine_semantic_probabilities",
    "fine_semantic_coarse_logits",
    "fine_semantic_coarse_probabilities",
    "fine_semantic_vegetation_split_logits",
)
CRITICAL_SOURCE_PATHS = (
    "scripts/train_residual_height.py",
    "scripts/train_multidomain.py",
    "src/msr/models/residual_height.py",
    "src/msr/models/domain_surface_net.py",
    "src/msr/models/height_net.py",
    "src/msr/inference/predict.py",
    "src/msr/training/balanced_height_loss.py",
    "src/msr/training/losses.py",
    "src/msr/data/gamus_dataset.py",
    "src/msr/data/surface_dataset.py",
    "src/msr/data/mixed_replay.py",
)


def file_sha256(path: str | Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> None:
    path = path.resolve()
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    atomic_replace(temporary, path)


def atomic_replace(source: Path, destination: Path) -> None:
    """Retry Windows reader-lock collisions without weakening atomic writes."""

    for attempt in range(20):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.25))


def atomic_text(path: Path, value: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    atomic_replace(temporary, path)


def source_code_manifest() -> dict[str, str]:
    return {
        relative: file_sha256(PROJECT_ROOT / relative)
        for relative in CRITICAL_SOURCE_PATHS
    }


def snapshot_source_code(destination: Path, manifest: Mapping[str, str]) -> None:
    snapshot_root = destination.resolve() / "source_snapshot"
    for relative, expected_hash in manifest.items():
        source = (PROJECT_ROOT / relative).resolve()
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if file_sha256(target) != expected_hash:
            raise RuntimeError(f"Source snapshot hash mismatch for {relative}")
    atomic_json(destination / "source_code_manifest.json", dict(manifest))


def assert_source_code_identity(expected: Mapping[str, str]) -> None:
    current = source_code_manifest()
    if dict(expected) != current:
        changed = sorted(
            path
            for path in set(expected) | set(current)
            if expected.get(path) != current.get(path)
        )
        raise RuntimeError(f"Critical source-code identity changed: {changed}")


def atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    atomic_replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def load_and_validate_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Residual config must be a mapping")
    contracts = config.get("contracts", {})
    data = config.get("data", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    if contracts.get("official_test_used") is not False:
        raise ValueError("Official test must remain unused")
    if contracts.get("external_holdout_used_for_training_or_selection") is not False:
        raise ValueError("External holdout must remain sealed")
    if contracts.get("missing_height_policy") != "ignored_never_zero":
        raise ValueError("Missing height policy is not fail-closed")
    if tuple(contracts.get("gamus_regression_source_class_ids", ())) != (1, 2, 3, 6):
        raise ValueError("GAMUS height supervision must exclude water and roads")
    if tuple(contracts.get("gamus_excluded_regression_source_class_ids", ())) != (0, 4, 5):
        raise ValueError("GAMUS background/water/road exclusion contract changed")
    if data.get("dataset") != "mixed_replay":
        raise ValueError("Residual height requires mixed_replay data")
    if int(training.get("batch_size", 0)) != 4:
        raise ValueError("Residual height requires batch size 4")
    if not math.isclose(float(data.get("gamus_train_fraction", 0)), 0.5):
        raise ValueError("Each batch must contain exactly 2 GAMUS and 2 legacy samples")
    if not bool(data.get("balanced_legacy_landscape_sampling")):
        raise ValueError("Legacy urban/forest sampling must be balanced")
    if data.get("legacy_relative_prior_policy") != "stored_01":
        raise ValueError("Legacy replay must use target-independent stored_01 priors")
    if float(data.get("legacy_supervised_crop_probability", 0)) != 1.0:
        raise ValueError("Sparse HighBuild crops must retain measured support")
    if training.get("learning_rate_schedule") != "warmup_cosine_steps":
        raise ValueError("Residual training requires optimizer-step warmup cosine")
    if int(training.get("overfit_steps", 0)) != 48:
        raise ValueError("Residual feasibility proof is fixed at 48 training-only steps")
    if evaluation.get("promotion_eligible") is not False:
        raise ValueError("Training config may not auto-promote")
    forbidden = [key for key in data if "test" in str(key).lower()]
    if forbidden:
        raise ValueError(f"Test-related data keys are forbidden: {forbidden}")
    for key, hash_key in (
        ("approved_index_path", "approved_index_file_sha256"),
        ("train_manifest", "train_manifest_file_sha256"),
        ("val_manifest", "val_manifest_file_sha256"),
        ("highbuild_contract_report", "highbuild_contract_report_sha256"),
    ):
        target = resolve(data[key])
        if file_sha256(target) != str(data[hash_key]).lower():
            raise ValueError(f"Authenticated data hash mismatch for {key}")
    pointer = resolve(evaluation["protected_pointer"])
    if file_sha256(pointer) != str(evaluation["protected_pointer_sha256"]).lower():
        raise ValueError("Protected application pointer hash mismatch")
    checkpoint = resolve(config["model"]["protected_checkpoint"])
    if file_sha256(checkpoint) != str(
        config["model"]["protected_checkpoint_sha256"]
    ).lower():
        raise ValueError("Protected checkpoint hash mismatch")
    if resolve(pointer.read_text(encoding="utf-8-sig").strip()) != checkpoint:
        raise ValueError("Live pointer does not resolve to protected checkpoint")
    return config


def make_model(config: Mapping[str, Any], device: torch.device) -> ResidualHeightSurfaceNet:
    model_config = config["model"]
    protected, _ = load_predictor(resolve(model_config["protected_checkpoint"]), device="cpu")
    model = ResidualHeightSurfaceNet(
        protected,
        hidden_channels=int(model_config.get("hidden_channels", 32)),
        maximum_correction_m=float(model_config.get("maximum_correction_m", 80.0)),
    )
    model.to(device)
    model.protected.eval().requires_grad_(False)
    return model


def protected_state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def assert_protected_unchanged(model: ResidualHeightSurfaceNet, expected: str) -> None:
    actual = protected_state_digest(model.protected.state_dict())
    if actual != expected:
        raise RuntimeError("Frozen protected tensors changed in memory")
    if any(parameter.grad is not None for parameter in model.protected.parameters()):
        raise RuntimeError("Frozen protected tensors received gradients")


def make_optimizer(model: ResidualHeightSurfaceNet, config: Mapping[str, Any]) -> AdamW:
    training = config["training"]
    return AdamW(
        [
            {
                "params": list(model.deepest.parameters())
                + list(model.decoder_stages.parameters()),
                "lr": float(training["decoder_learning_rate"]),
                "group_name": "copied_multiscale_decoder",
            },
            {
                "params": list(model.spatial_correction.parameters())
                + list(model.correction_head.parameters()),
                "lr": float(training["learning_rate"]),
                "group_name": "spatial_residual",
            },
        ],
        weight_decay=float(training.get("weight_decay", 0.0)),
    )


def make_train_datasets(config: Mapping[str, Any]):
    data = dict(config["data"])
    gamus = shared.make_gamus_dataset(
        data["root"], "train", data, training=True, use_train_samples_per_epoch=False
    )
    records = load_surface_manifest(data["train_manifest"])
    if {record.landscape for record in records} != {"urban", "forest"}:
        raise ValueError("Legacy training manifest must contain urban and forest")
    urban = shared.make_surface_dataset(
        [record for record in records if record.landscape == "urban"],
        data,
        training=True,
        use_train_samples_per_epoch=False,
    )
    forest = shared.make_surface_dataset(
        [record for record in records if record.landscape == "forest"],
        data,
        training=True,
        use_train_samples_per_epoch=False,
    )
    assert_matching_sample_contract(gamus, urban, forest)
    return gamus, urban, forest


class ExactThreeSourceBatchSampler(Sampler[list[int]]):
    """Deterministically yield 2 GAMUS + 1 HighBuild + 1 OpenCanopy."""

    def __init__(self, sizes: tuple[int, int, int], *, batches: int, seed: int) -> None:
        if any(size <= 0 for size in sizes) or batches <= 0:
            raise ValueError("Three-source sizes and batch count must be positive")
        self.sizes = tuple(int(size) for size in sizes)
        self.batches = int(batches)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.batches

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    @staticmethod
    def _cycles(size: int, count: int, generator: torch.Generator) -> list[int]:
        values: list[int] = []
        while len(values) < count:
            values.extend(torch.randperm(size, generator=generator).tolist())
        return values[:count]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        gamus = self._cycles(self.sizes[0], self.batches * 2, generator)
        urban = self._cycles(self.sizes[1], self.batches, generator)
        forest = self._cycles(self.sizes[2], self.batches, generator)
        urban_offset = self.sizes[0]
        forest_offset = self.sizes[0] + self.sizes[1]
        for index in range(self.batches):
            batch = [
                gamus[2 * index],
                gamus[2 * index + 1],
                urban_offset + urban[index],
                forest_offset + forest[index],
            ]
            order = torch.randperm(4, generator=generator).tolist()
            yield [batch[position] for position in order]


def make_train_loader(config: Mapping[str, Any], *, seed: int):
    gamus, urban, forest = make_train_datasets(config)
    dataset = ConcatDataset(
        (
            SourceTaggedDataset(gamus, "gamus"),
            SourceTaggedDataset(urban, "legacy"),
            SourceTaggedDataset(forest, "legacy"),
        )
    )
    samples = int(config["data"]["train_samples_per_epoch"])
    if samples % 4:
        raise ValueError("train_samples_per_epoch must be divisible by four")
    sampler = ExactThreeSourceBatchSampler(
        (len(gamus), len(urban), len(forest)), batches=samples // 4, seed=seed
    )
    workers = int(config["data"].get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def make_validation_loaders(config: Mapping[str, Any]) -> dict[str, DataLoader]:
    data = dict(config["data"])
    training = dict(config["training"])
    gamus = shared.make_gamus_dataset(data["root"], "val", data, training=False)
    records = load_surface_manifest(data["val_manifest"])
    groups = {
        "highbuild": [record for record in records if record.landscape == "urban"],
        "open_canopy": [record for record in records if record.landscape == "forest"],
    }
    if not all(groups.values()):
        raise ValueError("Validation manifest must contain urban and forest")
    loaders = {
        "gamus": shared.make_gamus_loader(gamus, data, training, training=False)
    }
    for name, subset in groups.items():
        suite_data = dict(data)
        suite_data["validation_patch_size"] = int(
            data[f"{name}_validation_patch_size"]
        )
        dataset = shared.make_surface_dataset(subset, suite_data, training=False)
        loaders[name] = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(data.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=int(data.get("num_workers", 0)) > 0,
        )
    return loaders


def move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def gradient_consistency(output: Mapping[str, torch.Tensor], batch: Mapping[str, Any]) -> torch.Tensor:
    prediction = output["height"].float()
    target = batch["height"].float()
    valid = batch["regression_mask"].bool()
    domain = batch["domain_target"]
    sources = tuple(str(value).lower() for value in batch["source"])
    landscapes = tuple(str(value).lower() for value in batch["landscape"])
    groups = tuple(
        "gamus"
        if source == "gamus"
        else ("highbuild" if landscape == "urban" else "open_canopy")
        for source, landscape in zip(sources, landscapes)
    )
    source_terms = []
    for source_name in ("gamus", "highbuild", "open_canopy"):
        domain_terms = []
        for domain_id in range(3):
            sample_terms = []
            for sample_index, group in enumerate(groups):
                if group != source_name:
                    continue
                sample_valid = valid[sample_index : sample_index + 1] & (
                    domain[sample_index : sample_index + 1, None] == domain_id
                )
                axis_terms = []
                for axis in (-1, -2):
                    predicted_delta = torch.diff(
                        prediction[sample_index : sample_index + 1], dim=axis
                    )
                    target_delta = torch.diff(
                        target[sample_index : sample_index + 1], dim=axis
                    )
                    left = sample_valid.narrow(axis, 0, sample_valid.shape[axis] - 1)
                    right = sample_valid.narrow(axis, 1, sample_valid.shape[axis] - 1)
                    pair = left & right
                    if torch.any(pair):
                        axis_terms.append(
                            torch.abs(predicted_delta[pair] - target_delta[pair]).mean()
                        )
                if axis_terms:
                    sample_terms.append(torch.stack(axis_terms).mean())
            if sample_terms:
                domain_terms.append(torch.stack(sample_terms).mean())
        if domain_terms:
            source_terms.append(torch.stack(domain_terms).mean())
    return (
        torch.stack(source_terms).mean()
        if source_terms
        else prediction.reshape(-1)[:0].sum()
    )


def train_step_loss(
    model: ResidualHeightSurfaceNet,
    criterion: SourceBalancedHeightLoss,
    batch: Mapping[str, Any],
    gradient_weight: float,
    device: torch.device,
    precision: str,
):
    values = move_batch(batch, device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and precision == "bf16"
        else nullcontext()
    )
    with autocast:
        output = model(values["image"], values["relative_prior"])
        balanced, components = criterion(output, values)
        gradient = (
            gradient_consistency(output, values)
            if gradient_weight > 0.0
            else output["height"].reshape(-1)[:0].sum().float()
        )
        total = balanced + gradient_weight * gradient
    return total, {**components, "gradient": gradient, "total": total}


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    device,
    config,
    *,
    epoch: int,
    status_path: Path | None = None,
) -> dict[str, float]:
    model.train()
    loader.batch_sampler.set_epoch(epoch)
    training = config["training"]
    accumulation = int(training.get("gradient_accumulation", 1))
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = defaultdict(float)
    for step, batch in enumerate(loader, start=1):
        loss, components = train_step_loss(
            model,
            criterion,
            batch,
            float(training.get("gradient_weight", 0.0)),
            device,
            str(training.get("precision", "bf16")),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite residual loss at step {step}")
        (loss / accumulation).backward()
        if step % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(training.get("max_gradient_norm", 1.0)),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        for name, value in components.items():
            totals[name] += float(value.detach())
        if status_path is not None and (step == 1 or step % 25 == 0):
            atomic_json(
                status_path,
                {
                    "stage": "training",
                    "epoch": epoch,
                    "batch": step,
                    "total_batches": len(loader),
                    "loss": float(loss.detach()),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                replace=True,
            )
    if len(loader) % accumulation:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(training.get("max_gradient_norm", 1.0)),
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return {name: value / len(loader) for name, value in totals.items()}


def metric_criterion(config: Mapping[str, Any]) -> MultiDomainSurfaceLoss:
    training = config["training"]
    return MultiDomainSurfaceLoss(
        huber_delta_m=float(training.get("huber_delta_m", 2.0)),
        height_weight=1.0,
        semantic_weight=0.0,
        building_weight=0.0,
        canopy_weight=0.0,
        ground_suppression_weight=0.0,
    )


class StatusReportingLoader:
    """Expose bounded per-suite validation progress without changing samples."""

    def __init__(
        self,
        loader,
        *,
        status_path: Path | None,
        stage: str,
        suite: str,
        suite_index: int,
        total_suites: int,
        epoch: int | None,
        maximum_batches: int | None = None,
        update_every: int = 10,
    ) -> None:
        self.loader = loader
        self.status_path = status_path
        self.stage = stage
        self.suite = suite
        self.suite_index = suite_index
        self.total_suites = total_suites
        self.epoch = epoch
        self.maximum_batches = maximum_batches
        self.update_every = update_every

    def __len__(self) -> int:
        available = len(self.loader)
        return (
            min(available, self.maximum_batches)
            if self.maximum_batches is not None
            else available
        )

    def _write(self, completed: int) -> None:
        if self.status_path is None:
            return
        total = len(self)
        atomic_json(
            self.status_path,
            {
                "stage": self.stage,
                "epoch": self.epoch,
                "suite": self.suite,
                "suite_index": self.suite_index,
                "total_suites": self.total_suites,
                "completed_batches": completed,
                "total_batches": total,
                "progress_fraction": completed / total if total else 1.0,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            replace=True,
        )

    def __iter__(self):
        total = len(self)
        self._write(0)
        for index, batch in enumerate(self.loader, start=1):
            if index > total:
                break
            yield batch
            if index == 1 or index % self.update_every == 0 or index == total:
                self._write(index)


def evaluate_suites(
    model,
    loaders,
    criterion,
    device,
    precision,
    *,
    status_path: Path | None = None,
    stage: str = "validation",
    epoch: int | None = None,
    maximum_batches: int | None = None,
) -> dict[str, Any]:
    result = {}
    total_suites = len(loaders)
    for suite_index, (name, loader) in enumerate(loaders.items(), start=1):
        observable_loader = StatusReportingLoader(
            loader,
            status_path=status_path,
            stage=stage,
            suite=name,
            suite_index=suite_index,
            total_suites=total_suites,
            epoch=epoch,
            maximum_batches=maximum_batches,
        )
        losses, metrics = shared.validate(
            model, observable_loader, criterion, device, precision=precision
        )
        result[name] = {"loss": losses, "metrics": metrics}
    return result


def validate_smoke_domains(results: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "gamus": None,
        "highbuild": "building",
        "open_canopy": "vegetation",
    }
    if set(results) != set(expected):
        raise RuntimeError(f"Validation smoke suite mismatch: {sorted(results)}")
    observed: dict[str, Any] = {}
    for suite, required_domain in expected.items():
        metrics = results[suite]["metrics"]
        domains = metrics.get("domains")
        if not isinstance(domains, Mapping) or not domains:
            raise RuntimeError(f"Validation smoke has no valid height domain for {suite}")
        if int(metrics.get("pixel_count", 0)) <= 0:
            raise RuntimeError(f"Validation smoke has no valid height pixels for {suite}")
        if required_domain is not None and required_domain not in domains:
            raise RuntimeError(
                f"Validation smoke expected {required_domain} support in {suite}"
            )
        observed[suite] = {
            "pixel_count": int(metrics["pixel_count"]),
            "domains": {
                name: int(values["pixel_count"])
                for name, values in domains.items()
            },
        }
    return observed


def validation_smoke(config, config_path: Path, device: torch.device) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = VALIDATION_SMOKE_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "status.json"
    code_manifest = source_code_manifest()
    snapshot_source_code(output_dir, code_manifest)
    atomic_json(
        status_path,
        {
            "stage": "validation_smoke",
            "suite": "pending",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        replace=True,
    )
    try:
        model = make_model(config, device)
        expected_protected = protected_state_digest(model.protected.state_dict())
        loaders = make_validation_loaders(config)
        results = evaluate_suites(
            model,
            loaders,
            metric_criterion(config).to(device),
            device,
            str(config["training"]["precision"]),
            status_path=status_path,
            stage="validation_smoke",
            maximum_batches=2,
        )
        observed = validate_smoke_domains(results)
        assert_protected_unchanged(model, expected_protected)
        assert_source_code_identity(code_manifest)
        report = {
            "schema": "msr.residual_height_validation_smoke.v1",
            "status": "passed",
            "run_id": run_id,
            "config_sha256": file_sha256(config_path),
            "source_code_sha256": code_manifest,
            "maximum_batches_per_suite": 2,
            "observed_support": observed,
            "official_test_used": False,
            "external_holdout_used": False,
            "promotion_eligible": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(output_dir / "report.json", report)
        atomic_json(
            status_path,
            {**report, "stage": "complete"},
            replace=True,
        )
        return output_dir
    except BaseException as error:
        mark_terminal_failure(output_dir, error)
        raise


def preflight(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    gamus, urban, forest = make_train_datasets(config)
    samples = int(config["data"]["train_samples_per_epoch"])
    sampler = ExactThreeSourceBatchSampler(
        (len(gamus), len(urban), len(forest)),
        batches=samples // 4,
        seed=int(config["experiment"]["seed"]),
    )
    first = next(iter(sampler))
    offsets = (len(gamus), len(gamus) + len(urban))
    composition = {
        "gamus": sum(index < offsets[0] for index in first),
        "highbuild": sum(offsets[0] <= index < offsets[1] for index in first),
        "open_canopy": sum(index >= offsets[1] for index in first),
    }
    if composition != {"gamus": 2, "highbuild": 1, "open_canopy": 1}:
        raise RuntimeError(f"Exact three-source sampler failed: {composition}")
    records = load_surface_manifest(config["data"]["val_manifest"])
    report = {
        "schema": "msr.residual_height_preflight.v1",
        "status": "passed",
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "protected_checkpoint_sha256": file_sha256(
            resolve(config["model"]["protected_checkpoint"])
        ),
        "official_test_used": False,
        "external_holdout_used": False,
        "train_records": {
            "gamus": len(gamus),
            "highbuild": len(urban),
            "open_canopy": len(forest),
        },
        "validation_records": {
            "gamus": 859,
            "highbuild": sum(r.landscape == "urban" for r in records),
            "open_canopy": sum(r.landscape == "forest" for r in records),
        },
        "batches_per_epoch": len(sampler),
        "per_batch_composition": composition,
        "promotion_eligible": False,
    }
    return report


def _metric(metrics: Mapping[str, Any], path: str) -> float:
    value: Any = metrics
    for component in path.split("."):
        value = value[component]
    return float(value)


def evaluate_candidate_gate(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    required_suites = {"gamus", "highbuild", "open_canopy"}
    if set(baseline) != required_suites or set(candidate) != required_suites:
        return {
            "passes": False,
            "all_regression_guards_pass": False,
            "material_improvement": False,
            "selection_score": float("inf"),
            "reason": "suite_set_or_count_identity_failed",
            "checks": {
                "baseline_suites": sorted(baseline),
                "candidate_suites": sorted(candidate),
            },
        }
    max_rmse = float(evaluation["maximum_domain_rmse_regression_m"])
    max_mae = float(evaluation["maximum_mae_regression_m"])
    max_bias = float(evaluation["maximum_absolute_bias_regression_m"])
    minimum_m = float(evaluation["minimum_material_improvement_m"])
    minimum_fraction = float(evaluation["minimum_material_improvement_fraction"])
    checks: dict[str, Any] = {}
    passed = True
    for suite in sorted(required_suites):
        before = baseline[suite]["metrics"]
        after = candidate[suite]["metrics"]
        suite_checks: dict[str, Any] = {}
        count_identity = before.get("pixel_count") == after.get("pixel_count")
        suite_checks["pixel_count_identity"] = count_identity
        passed &= count_identity
        before_domains = before.get("domains")
        after_domains = after.get("domains")
        domain_identity = (
            isinstance(before_domains, Mapping)
            and isinstance(after_domains, Mapping)
            and set(before_domains) == set(after_domains)
        )
        suite_checks["domain_set_identity"] = domain_identity
        passed &= domain_identity

        metric_groups: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = [
            ("overall", before, after)
        ]
        if domain_identity:
            for domain in sorted(before_domains):
                metric_groups.append(
                    (f"domains.{domain}", before_domains[domain], after_domains[domain])
                )
        before_tall = before.get("tall_objects", {})
        after_tall = after.get("tall_objects", {})
        tall_identity = (
            isinstance(before_tall, Mapping)
            and isinstance(after_tall, Mapping)
            and set(before_tall) == set(after_tall)
        )
        suite_checks["tall_object_set_identity"] = tall_identity
        passed &= tall_identity
        if tall_identity:
            for domain in sorted(before_tall):
                if int(before_tall[domain].get("pixel_count", 0)) > 0:
                    metric_groups.append(
                        (f"tall_objects.{domain}", before_tall[domain], after_tall[domain])
                    )

        for prefix, old_metrics, new_metrics in metric_groups:
            group_checks: dict[str, Any] = {}
            group_count_identity = old_metrics.get("pixel_count") == new_metrics.get(
                "pixel_count"
            )
            group_checks["pixel_count_identity"] = group_count_identity
            passed &= group_count_identity
            for metric_name, limit in (("rmse_m", max_rmse), ("mae_m", max_mae)):
                old_value = float(old_metrics[metric_name])
                new_value = float(new_metrics[metric_name])
                finite = math.isfinite(old_value) and math.isfinite(new_value)
                delta = new_value - old_value if finite else None
                guard_pass = bool(finite and delta <= limit)
                group_checks[metric_name] = {
                    "baseline": old_value,
                    "candidate": new_value,
                    "delta": delta,
                    "limit": limit,
                    "passes": guard_pass,
                }
                passed &= guard_pass
            old_bias = float(old_metrics["bias_m"])
            new_bias = float(new_metrics["bias_m"])
            finite_bias = math.isfinite(old_bias) and math.isfinite(new_bias)
            bias_delta = abs(new_bias) - abs(old_bias) if finite_bias else None
            bias_pass = bool(finite_bias and bias_delta <= max_bias)
            group_checks["absolute_bias"] = {
                "baseline": old_bias,
                "candidate": new_bias,
                "delta": bias_delta,
                "limit": max_bias,
                "passes": bias_pass,
            }
            passed &= bias_pass
            for metric_name in ("correlation", "r2"):
                old_value = old_metrics.get(metric_name)
                new_value = new_metrics.get(metric_name)
                if old_value is None or not math.isfinite(float(old_value)):
                    correlation_pass = True
                    group_checks[metric_name] = {
                        "baseline": old_value,
                        "candidate": new_value,
                        "baseline_undefined": True,
                        "comparison": "not_comparable",
                        "passes": correlation_pass,
                    }
                else:
                    finite_new = new_value is not None and math.isfinite(float(new_value))
                    delta = float(new_value) - float(old_value) if finite_new else None
                    correlation_pass = bool(finite_new and delta >= -1.0e-6)
                    group_checks[metric_name] = {
                        "baseline": float(old_value),
                        "candidate": float(new_value) if finite_new else new_value,
                        "delta": delta,
                        "minimum_delta": -1.0e-6,
                        "passes": correlation_pass,
                    }
                passed &= correlation_pass
            suite_checks[prefix] = group_checks
        before_semantic = before.get("semantic_identification", {})
        after_semantic = after.get("semantic_identification", {})
        semantic_identity = (
            isinstance(before_semantic, Mapping)
            and isinstance(after_semantic, Mapping)
            and all(
                key in before_semantic
                and key in after_semantic
                and before_semantic[key] == after_semantic[key]
                for key in SEMANTIC_METRIC_KEYS
            )
        )
        suite_checks["classification_identity"] = semantic_identity
        passed &= semantic_identity
        checks[suite] = suite_checks

    priority_results: dict[str, Any] = {}
    for name, suite, domain in (
        ("highbuild_building", "highbuild", "building"),
        ("open_canopy_vegetation", "open_canopy", "vegetation"),
    ):
        try:
            old = _metric(baseline[suite]["metrics"], f"domains.{domain}.rmse_m")
            new = _metric(candidate[suite]["metrics"], f"domains.{domain}.rmse_m")
            improvement = old - new
            material = bool(
                math.isfinite(old)
                and math.isfinite(new)
                and (
                    improvement >= minimum_m
                    or (old > 0 and improvement / old >= minimum_fraction)
                )
            )
        except (KeyError, TypeError, ValueError):
            old = new = improvement = None
            material = False
        priority_results[name] = {
            "baseline": old,
            "candidate": new,
            "improvement_m": improvement,
            "material_improvement": material,
        }
    material = any(item["material_improvement"] for item in priority_results.values())
    gamus_domains = candidate["gamus"]["metrics"].get("domains", {})
    gamus_domain_rmse = [
        float(item["rmse_m"])
        for item in gamus_domains.values()
        if math.isfinite(float(item["rmse_m"]))
    ]
    selection_score = (
        (
            float(candidate["highbuild"]["metrics"]["domains"]["building"]["rmse_m"])
            + float(candidate["open_canopy"]["metrics"]["domains"]["vegetation"]["rmse_m"])
            + sum(gamus_domain_rmse) / len(gamus_domain_rmse)
        )
        / 3.0
        if gamus_domain_rmse
        else float("inf")
    )
    return {
        "passes": bool(passed and material),
        "all_regression_guards_pass": bool(passed),
        "material_improvement": material,
        "priority_improvements": priority_results,
        "selection_score": selection_score,
        "selection_formula": "mean(highbuild_building_rmse, open_canopy_vegetation_rmse, gamus_domain_macro_rmse)",
        "checks": checks,
    }


def semantic_snapshot(
    model: ResidualHeightSurfaceNet,
    batch: Mapping[str, Any],
    device: torch.device,
    precision: str,
) -> dict[str, torch.Tensor]:
    values = move_batch(batch, device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and precision == "bf16"
        else nullcontext()
    )
    model.eval()
    with torch.no_grad(), autocast:
        output = model(values["image"], values["relative_prior"])
    snapshot = {
        key: output[key].detach().cpu().clone()
        for key in SEMANTIC_OUTPUT_KEYS
        if key in output
    }
    if not snapshot:
        raise RuntimeError("Protected model exposed no semantic outputs")
    return snapshot


def overfit_proof(config, config_path, steps, device) -> dict[str, Any]:
    if OVERFIT_REPORT.is_file():
        previous = json.loads(OVERFIT_REPORT.read_text(encoding="utf-8"))
        previous_run_id = str(previous.get("run_id", "unknown_run"))
        previous_dir = OVERFIT_REPORT.parent / previous_run_id
        previous_dir.mkdir(parents=True, exist_ok=True)
        previous_archive = previous_dir / "report.json"
        if not previous_archive.exists():
            atomic_json(previous_archive, previous)
    code_manifest = source_code_manifest()
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    model = make_model(config, device)
    expected_protected = protected_state_digest(model.protected.state_dict())
    optimizer = make_optimizer(model, config)
    criterion = SourceBalancedHeightLoss(
        huber_delta_m=float(config["training"]["huber_delta_m"]),
        mse_weight=float(config["training"]["mse_weight"]),
    ).to(device)
    gamus, urban, forest = make_train_datasets(config)
    samples = [
        {**gamus[0], "source": "gamus"},
        {**gamus[len(gamus) - 1], "source": "gamus"},
        {**urban[0], "source": "legacy"},
        {**forest[0], "source": "legacy"},
    ]
    batch = default_collate(samples)
    losses = []
    source_history: dict[str, list[float]] = {
        name: [] for name in ("gamus", "highbuild", "open_canopy")
    }
    precision = str(config["training"]["precision"])
    semantic_before = semantic_snapshot(model, batch, device, precision)
    maximum_copied_decoder_gradient = 0.0
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, components = train_step_loss(
            model,
            criterion,
            batch,
            float(config["training"]["gradient_weight"]),
            device,
            precision,
        )
        loss.backward()
        copied_gradients = [
            parameter.grad.detach().abs().max()
            for module in (model.deepest, model.decoder_stages)
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if copied_gradients:
            maximum_copied_decoder_gradient = max(
                maximum_copied_decoder_gradient,
                max(float(value) for value in copied_gradients),
            )
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(config["training"]["max_gradient_norm"]))
        optimizer.step()
        losses.append(float(loss.detach()))
        for source_name in source_history:
            if source_name in components:
                source_history[source_name].append(
                    float(components[source_name].detach())
                )
    assert_protected_unchanged(model, expected_protected)
    semantic_after = semantic_snapshot(model, batch, device, precision)
    semantic_identity_by_key = {
        key: key in semantic_after and torch.equal(value, semantic_after[key])
        for key, value in semantic_before.items()
    }
    semantic_identity = (
        set(semantic_before) == set(semantic_after)
        and all(semantic_identity_by_key.values())
    )
    initial = sum(losses[: min(4, len(losses))]) / min(4, len(losses))
    final = sum(losses[-min(4, len(losses)):]) / min(4, len(losses))
    reduction = (initial - final) / initial if initial > 0 else 0.0
    threshold = float(config["training"]["overfit_minimum_relative_loss_reduction"])
    loss_passes = bool(
        math.isfinite(initial)
        and math.isfinite(final)
        and reduction >= threshold
    )
    copied_decoder_gradient_nonzero = bool(
        math.isfinite(maximum_copied_decoder_gradient)
        and maximum_copied_decoder_gradient > 0.0
    )
    all_sources_observed = all(len(values) == steps for values in source_history.values())
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    assert_source_code_identity(code_manifest)
    proof_dir = OVERFIT_REPORT.parent / run_id
    report = {
        "schema": "msr.residual_height_overfit.v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_only": True,
        "validation_used": False,
        "test_used": False,
        "external_holdout_used": False,
        "steps": steps,
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "source_landscape": [[str(sample["source"]), str(sample["landscape"])] for sample in samples],
        "initial_loss": initial,
        "final_loss": final,
        "relative_loss_reduction": reduction,
        "minimum_required": threshold,
        "loss_passes": loss_passes,
        "source_losses": {
            source_name: {
                "first": values[0] if values else None,
                "last": values[-1] if values else None,
                "first_four_mean": sum(values[:4]) / min(4, len(values)) if values else None,
                "last_four_mean": sum(values[-4:]) / min(4, len(values)) if values else None,
            }
            for source_name, values in source_history.items()
        },
        "all_sources_observed_every_step": all_sources_observed,
        "maximum_copied_decoder_gradient": maximum_copied_decoder_gradient,
        "copied_decoder_gradient_nonzero": copied_decoder_gradient_nonzero,
        "semantic_output_keys": sorted(semantic_before),
        "semantic_identity_by_key": semantic_identity_by_key,
        "classification_outputs_torch_equal": semantic_identity,
        "passes": bool(
            loss_passes
            and copied_decoder_gradient_nonzero
            and semantic_identity
            and all_sources_observed
        ),
        "config_sha256": file_sha256(config_path),
        "protected_checkpoint_sha256": file_sha256(resolve(config["model"]["protected_checkpoint"])),
        "source_code_sha256": code_manifest,
        "source_snapshot_dir": str(proof_dir / "source_snapshot"),
        "weights_reused_for_full_training": False,
    }
    proof_dir.mkdir(parents=True, exist_ok=False)
    snapshot_source_code(proof_dir, code_manifest)
    atomic_json(OVERFIT_REPORT, report, replace=True)
    return report


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    epoch,
    config,
    baseline,
    metrics,
    state,
    source_manifest,
    epoch_record,
):
    return {
        "model_type": model.model_type,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "config": config,
        "baseline_validation": baseline,
        "validation": metrics,
        "training_state": state,
        "source_code_sha256": dict(source_manifest),
        "epoch_record": epoch_record,
        "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }


def reconcile_history(history: Path, committed_epoch: int, record: Mapping[str, Any]) -> None:
    retained: list[dict[str, Any]] = []
    if history.is_file():
        for line in history.read_text(encoding="utf-8").splitlines():
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                break
            if int(parsed.get("epoch", -1)) <= committed_epoch:
                retained.append(parsed)
    by_epoch = {int(item["epoch"]): item for item in retained}
    by_epoch[committed_epoch] = dict(record)
    normalized = [by_epoch[index] for index in sorted(by_epoch) if index <= committed_epoch]
    atomic_text(
        history,
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in normalized),
    )


def mark_terminal_failure(experiment_dir: Path, error: BaseException) -> None:
    status_path = experiment_dir / "status.json"
    previous: dict[str, Any] = {}
    if status_path.is_file():
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except (json.JSONDecodeError, OSError):
            previous = {}
    summary = {
        "schema": "msr.residual_height_training_summary.v1",
        "stage": "failed",
        "failed_stage": previous.get("stage", "initialization"),
        "epoch": previous.get("epoch"),
        "suite": previous.get("suite"),
        "completed_batches": previous.get("completed_batches"),
        "total_batches": previous.get("total_batches"),
        "error_type": type(error).__name__,
        "error": str(error),
        "official_test_used": False,
        "external_holdout_used": False,
        "promotion_eligible": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(experiment_dir / "training_summary.json", summary, replace=True)
    atomic_json(status_path, summary, replace=True)


def _run_training(config, config_path, resume, device, run_context) -> Path:
    if resume is not None and resume.name != "checkpoint_latest.pt":
        raise ValueError("Resume is restricted to checkpoint_latest.pt")
    current_source_manifest = source_code_manifest()
    if resume is None:
        overfit = json.loads(OVERFIT_REPORT.read_text(encoding="utf-8")) if OVERFIT_REPORT.is_file() else None
        if (
            not overfit
            or not overfit.get("passes")
            or overfit.get("config_sha256") != file_sha256(config_path)
            or int(overfit.get("steps", -1)) != int(config["training"]["overfit_steps"])
            or overfit.get("protected_checkpoint_sha256")
            != file_sha256(resolve(config["model"]["protected_checkpoint"]))
            or overfit.get("source_code_sha256") != current_source_manifest
        ):
            raise RuntimeError("A matching passing 48-step training-only overfit proof is required")
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    model = make_model(config, device)
    expected_protected = protected_state_digest(model.protected.state_dict())
    optimizer = make_optimizer(model, config)
    train_loader = make_train_loader(config, seed=seed)
    assert_source_code_identity(current_source_manifest)
    scheduler, scheduler_interval = shared.make_learning_rate_scheduler(
        optimizer, config["training"], batches_per_epoch=len(train_loader)
    )
    if scheduler_interval != "optimizer_step":
        raise RuntimeError("Residual trainer requires an optimizer-step scheduler")
    criterion = SourceBalancedHeightLoss(huber_delta_m=float(config["training"]["huber_delta_m"]), mse_weight=float(config["training"]["mse_weight"])).to(device)
    metrics_criterion = metric_criterion(config).to(device)
    start_epoch = 1
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        if payload["config"] != config:
            raise ValueError("Resume config mismatch")
        if payload.get("source_code_sha256") != current_source_manifest:
            raise ValueError("Resume checkpoint source-code identity mismatch")
        model.load_state_dict(payload["model"], strict=True)
        assert_protected_unchanged(model, expected_protected)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        experiment_dir = resume.parent.resolve()
        run_context["experiment_dir"] = experiment_dir
        baseline = payload["baseline_validation"]
        state = payload["training_state"]
        best_score = float(state.get("best_score", float("inf")))
        rng = payload.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
        if state.get("protected_state_sha256") != expected_protected:
            raise ValueError("Resume checkpoint protected-state identity mismatch")
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        print(f"EXPERIMENT_DIR={experiment_dir}", flush=True)
        sealed_manifest = json.loads(
            (experiment_dir / "source_code_manifest.json").read_text(encoding="utf-8")
        )
        if sealed_manifest != current_source_manifest:
            raise ValueError("Experiment source snapshot identity mismatch")
        epoch_record = payload.get("epoch_record")
        if not isinstance(epoch_record, Mapping):
            raise ValueError("Resume checkpoint is missing its committed epoch record")
        reconcile_history(
            experiment_dir / "metrics.jsonl", int(payload["epoch"]), epoch_record
        )
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        experiment_dir = resolve(config["experiment"]["output_root"]) / f"{stamp}_{config['experiment']['name']}"
        experiment_dir.mkdir(parents=True, exist_ok=False)
        run_context["experiment_dir"] = experiment_dir
        print(f"EXPERIMENT_DIR={experiment_dir}", flush=True)
        atomic_text(
            experiment_dir / "config.yaml",
            config_path.read_text(encoding="utf-8"),
        )
        snapshot_source_code(experiment_dir, current_source_manifest)
        status_path = experiment_dir / "status.json"
        atomic_json(
            status_path,
            {
                "stage": "baseline_validation",
                "suite": "all_full_native",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            replace=True,
        )
        assert_protected_unchanged(model, expected_protected)
        assert_source_code_identity(current_source_manifest)
        loaders = make_validation_loaders(config)
        baseline = evaluate_suites(
            model,
            loaders,
            metrics_criterion,
            device,
            str(config["training"]["precision"]),
            status_path=status_path,
            stage="baseline_validation",
        )
        assert_source_code_identity(current_source_manifest)
        atomic_json(experiment_dir / "baseline_validation.json", baseline)
        best_score = float("inf")
        state = {}
        epochs_without_improvement = 0
    loaders = make_validation_loaders(config)
    history = experiment_dir / "metrics.jsonl"
    status_path = experiment_dir / "status.json"
    final_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
        assert_source_code_identity(current_source_manifest)
        assert_protected_unchanged(model, expected_protected)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            config,
            epoch=epoch,
            status_path=status_path,
        )
        assert_source_code_identity(current_source_manifest)
        atomic_json(
            status_path,
            {
                "stage": "validation",
                "epoch": epoch,
                "suite": "all_full_native",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            replace=True,
        )
        validation = evaluate_suites(
            model,
            loaders,
            metrics_criterion,
            device,
            str(config["training"]["precision"]),
            status_path=status_path,
            stage="validation",
            epoch=epoch,
        )
        gate = evaluate_candidate_gate(baseline, validation, config["evaluation"])
        score = float(gate["selection_score"])
        improved = bool(gate["passes"] and score < best_score)
        if improved:
            best_score = score
            epochs_without_improvement = 0
        elif epoch >= int(config["training"].get("early_stopping_min_epochs", 1)):
            epochs_without_improvement += 1
        state = {
            "best_score": best_score,
            "gate": gate,
            "protected_state_sha256": expected_protected,
            "epochs_without_improvement": epochs_without_improvement,
        }
        record = {"epoch": epoch, "train": train_metrics, "validation": validation, "gate": gate, "selection_score": score, "improved": improved}
        assert_source_code_identity(current_source_manifest)
        assert_protected_unchanged(model, expected_protected)
        payload = checkpoint_payload(
            model,
            optimizer,
            scheduler,
            epoch,
            config,
            baseline,
            validation,
            state,
            current_source_manifest,
            record,
        )
        atomic_torch_save(experiment_dir / f"checkpoint_epoch_{epoch:03d}.pt", payload)
        if improved:
            atomic_torch_save(experiment_dir / "checkpoint_best_guarded.pt", payload)
        # ``checkpoint_latest`` is the commit point. If interruption happens
        # before the JSONL append, resume reconstructs the row from the payload.
        atomic_torch_save(experiment_dir / "checkpoint_latest.pt", payload)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        atomic_json(
            status_path,
            {
                "stage": "epoch_complete",
                "epoch": epoch,
                "gate_passed": gate["passes"],
                "improved": improved,
                "selection_score": score,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            replace=True,
        )
        print(json.dumps({"epoch": epoch, "train": train_metrics, "gate": gate, "selection_score": score, "best_score": best_score}, indent=2))
        final_epoch = epoch
        if epochs_without_improvement >= int(config["training"].get("early_stopping_patience", 3)):
            stopped_early = True
            break
    terminal_stage = "early_stopped" if stopped_early else "complete"
    summary = {
        "schema": "msr.residual_height_training_summary.v1",
        "stage": terminal_stage,
        "experiment_dir": str(experiment_dir),
        "final_epoch": final_epoch,
        "best_guarded_checkpoint": str(experiment_dir / "checkpoint_best_guarded.pt")
        if (experiment_dir / "checkpoint_best_guarded.pt").is_file()
        else None,
        "best_score": best_score if math.isfinite(best_score) else None,
        "protected_checkpoint_unchanged": True,
        "official_test_used": False,
        "external_holdout_used": False,
        "promotion_eligible": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(experiment_dir / "training_summary.json", summary, replace=True)
    atomic_json(status_path, summary, replace=True)
    return experiment_dir


def run_training(config, config_path, resume, device) -> Path:
    run_context: dict[str, Path] = {}
    try:
        return _run_training(config, config_path, resume, device, run_context)
    except BaseException as error:
        experiment_dir = run_context.get("experiment_dir")
        if experiment_dir is not None:
            try:
                mark_terminal_failure(experiment_dir, error)
            except Exception:
                pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overfit-steps", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validation-smoke-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config.resolve()
    config = load_and_validate_config(config_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.preflight_only:
        if args.resume is not None or args.overfit_steps is not None or args.validation_smoke_only:
            raise ValueError("Preflight-only cannot train or resume")
        print(json.dumps(preflight(config, config_path), indent=2))
    elif args.validation_smoke_only:
        if args.resume is not None or args.overfit_steps is not None:
            raise ValueError("Validation smoke cannot train or resume")
        print(validation_smoke(config, config_path, device))
    elif args.overfit_steps is not None:
        if args.resume is not None or args.overfit_steps != int(config["training"]["overfit_steps"]):
            raise ValueError("Overfit proof must use the configured 48 steps and cannot resume")
        print(json.dumps(overfit_proof(config, config_path, args.overfit_steps, device), indent=2))
    else:
        print(run_training(config, config_path, args.resume.resolve() if args.resume else None, device))
