"""Train an isolated RGB SegFormer on the fixed GAMUS DC/PHL development split.

This experiment never loads a height model for inference, never changes an app
pointer, and never opens official test, NYC, or the sealed external geography.
Feasibility proofs and epoch checkpoints are bound to code/config/data identities.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msr.models.rgb_segmenter import RgbSegformer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_replace(source: Path, destination: Path) -> None:
    """Readers may briefly hold Windows files without FILE_SHARE_DELETE."""
    for attempt in range(20):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.25))


def atomic_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    atomic_replace(temporary, path)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_replace(temporary, path)


def write_history(path: Path, history: list[dict]) -> None:
    text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in history)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    atomic_replace(temporary, path)


def source_manifest() -> dict[str, str]:
    names = [
        "scripts/train_rgb_segmenter.py",
        "src/msr/models/rgb_segmenter.py",
        "src/msr/data/gamus_rgb_segmentation.py",
        "src/msr/evaluation/rgb_segmentation.py",
        "src/msr/evaluation/classification_metrics.py",
        "src/msr/data/gamus_dataset.py",
        "src/msr/data/raster_dataset.py",
    ]
    return {name: sha256(name) for name in names}


def assert_sources(expected: dict[str, str]) -> None:
    if source_manifest() != expected:
        raise RuntimeError("Sealed RGB training source changed; create a new validated recipe")


def assert_file_hash(path: str | Path, expected: str) -> None:
    if len(expected) != 64 or sha256(path) != expected:
        raise RuntimeError(f"Required file identity mismatch: {path}")


def verify_protected(config: dict) -> dict:
    protocol = config["protocol"]
    assert_file_hash(protocol["protected_checkpoint"], protocol["protected_checkpoint_sha256"])
    assert_file_hash(protocol["live_pointer_file"], protocol["live_pointer_file_sha256"])
    target = resolve(protocol["live_pointer_file"]).read_text(encoding="utf-8-sig").strip()
    if resolve(target).resolve() != resolve(protocol["protected_checkpoint"]).resolve():
        raise RuntimeError("Protected pointer resolves to a different checkpoint")
    return {"checkpoint_sha256": protocol["protected_checkpoint_sha256"],
            "pointer_sha256": protocol["live_pointer_file_sha256"]}


def verify_pretrained(config: dict) -> dict:
    model = config["model"]
    if len(model.get("pretrained_revision", "")) != 40:
        raise RuntimeError("An immutable pretrained revision is required")
    hashes = model.get("pretrained_files_sha256", {})
    if "config.json" not in hashes or not ({"pytorch_model.bin", "model.safetensors"} & set(hashes)):
        raise RuntimeError("Pretrained config and tensor file hashes are required")
    directory = resolve(model["pretrained_directory"])
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ValueError("Pretrained files must be immediate snapshot children")
        assert_file_hash(directory / name, expected)
    return {"revision": model["pretrained_revision"], "sha256": hashes}


def software_versions() -> dict:
    return {"python": sys.version.split()[0], **{
        package: importlib.metadata.version(package)
        for package in ("torch", "transformers", "numpy", "h5py", "scipy", "PyYAML")
    }}


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Expected a YAML mapping")
    protocol = config["protocol"]
    if protocol.get("auto_promotion") is not False or protocol.get("development_only") is not True:
        raise ValueError("Only an isolated development experiment is supported")
    for name in ("learning_cities", "development_validation_cities"):
        if protocol.get(name) != ["DC", "PHL"]:
            raise ValueError("This recipe only permits the fixed DC/PHL development split")
    training = config["training"]
    for name in ("epochs", "batch_size", "gradient_accumulation"):
        if not isinstance(training[name], int) or training[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if training["precision"] not in {"bf16", "fp32"}:
        raise ValueError("Only bf16/fp32 are supported; fp16 requires an independently tested scaler")
    if int(config["data"]["validation_patch_size"]) != 1024:
        raise ValueError("Development comparison requires native 1024-pixel validation")
    if config["data"]["train_radiometric_policy"] != "raw" or config["data"]["validation_radiometric_policy"] != "raw":
        raise ValueError("Preserve the fixed raw-radiometry comparison")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def precision_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


class SixClassFocalLoss(nn.Module):
    """Same weighted-pixel-mean focal objective as the fixed V3 comparator."""

    def __init__(self, weights: list[float], gamma: float) -> None:
        super().__init__()
        if len(weights) != 6 or any(not math.isfinite(x) or x <= 0 for x in weights):
            raise ValueError("Six finite, positive class weights are required")
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("Focal gamma must be finite and nonnegative")
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, image_valid: torch.Tensor,
                classification_valid: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 6 or labels.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError("Logits/labels must have matching [B,6,H,W]/[B,H,W] shapes")
        if image_valid.shape != labels.shape or classification_valid.shape != labels.shape:
            raise ValueError("Independent validity masks must match label shape")
        valid = image_valid.bool() & classification_valid.bool()
        if bool(((labels[valid] < 0) | (labels[valid] >= 6)).any()):
            raise ValueError("Classification-valid labels must be class indices 0..5")
        if not bool(valid.any()):
            return logits.reshape(-1)[:0].sum()
        selected = logits.permute(0, 2, 3, 1)[valid].float()
        targets = labels[valid].long()
        values = F.cross_entropy(selected, targets, weight=self.weights, reduction="none")
        probability = selected.softmax(dim=1).gather(1, targets[:, None])[:, 0]
        return (values * (1.0 - probability).pow(self.gamma)).mean()


def loss_for(model, batch, criterion, device, precision):
    image = batch["image"].to(device, non_blocking=True)
    with precision_context(device, precision):
        logits = model(image)["logits"]
    return criterion(logits, batch["labels"].to(device), batch["image_valid_mask"].to(device),
                     batch["classification_valid_mask"].to(device))


def learning_rate_factor(step: int, total_steps: int, warmup_steps: int, minimum: float) -> float:
    if total_steps <= 0 or warmup_steps < 0 or not 0 <= minimum <= 1:
        raise ValueError("Invalid learning-rate schedule")
    if warmup_steps and step < warmup_steps:
        return float(step + 1) / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def rng_state(generator: torch.Generator) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def status(path: Path, stage: str, **fields) -> None:
    atomic_json(path / "status.json", {"stage": stage, "updated_at_utc": utc_now(), **fields})


def save_proof(config: dict, key: str, report: dict) -> None:
    destination = resolve(config["proof"][key])
    versioned = destination.parent / report["run_id"] / destination.name
    atomic_json(versioned, report)
    atomic_json(destination, report)


def verify_proofs(config: dict, binding: dict) -> None:
    for key in ("overfit_report", "validation_smoke_report"):
        report = json.loads(resolve(config["proof"][key]).read_text(encoding="utf-8"))
        if report.get("passes") is not True or report.get("binding") != binding:
            raise RuntimeError(f"Missing, failed, or stale source-bound {key}; rerun the proof")
        if key == "overfit_report":
            if report.get("steps", 0) < config["proof"]["minimum_steps"] or report.get("loss_reduction_fraction", -1) < config["proof"]["minimum_loss_reduction_fraction"] or report.get("encoder_gradient_norm", 0) <= 0:
                raise RuntimeError("Training-only proof does not establish the configured learning criterion")
        elif report.get("native_size") != 1024 or report.get("device") != "cuda":
            raise RuntimeError("Native 1024-pixel CUDA validation smoke is required")


def dataset_tools():
    from msr.data.gamus_rgb_segmentation import make_datasets, build_train_sampler, authenticate_validation_content
    from msr.evaluation.rgb_segmentation import (
        evaluate_rgb_segmentation, classification_acceptance,
    )
    return make_datasets, build_train_sampler, evaluate_rgb_segmentation, classification_acceptance, authenticate_validation_content


def verify_data_unchanged(config: dict, train_data, val_data, binding: dict) -> None:
    from msr.data.gamus_rgb_segmentation import source_metadata_snapshot
    data = config["data"]
    assert_file_hash(data["approved_index_path"], data["approved_index_file_sha256"])
    assert_file_hash(data["fine_class_sampling_index_path"], data["fine_class_sampling_index_sha256"])
    current = {"train": source_metadata_snapshot(train_data), "val": source_metadata_snapshot(val_data)}
    if current != binding["data_contract"]["source_metadata"]:
        raise RuntimeError("RGB/class source metadata changed during the sealed experiment")


def preflight(config: dict, config_path: Path) -> tuple:
    make_datasets, _, _, _, authenticate = dataset_tools()
    protected = verify_protected(config)
    pretrained = verify_pretrained(config)
    protocol = config["protocol"]
    assert_file_hash(protocol["fixed_v3_independent_replay"], protocol["fixed_v3_independent_replay_sha256"])
    train_data, val_data, contract = make_datasets(config)
    assert_file_hash(config["data"]["fine_class_sampling_index_path"], config["data"]["fine_class_sampling_index_sha256"])
    validation_identity = authenticate(
        val_data, resolve(protocol["fixed_v3_independent_replay"]),
        protocol["fixed_v3_independent_replay_sha256"],
        resolve(config["proof"]["validation_content_cache"]),
    )
    binding = {"config_sha256": sha256(config_path), "source_code_sha256": source_manifest(),
               "protected": protected, "pretrained": pretrained, "software": software_versions(),
               "data_contract": contract}
    report = {"schema": "msr.rgb_segmenter_preflight.v1", "passes": True,
              "binding": binding, "validation_identity": validation_identity,
              "development_only": True, "created_at_utc": utc_now()}
    atomic_json(resolve("outputs/diagnostics/gamus_rgb_segmenter_v1/preflight_report.json"), report)
    return train_data, val_data, binding


def make_model(config: dict, device: torch.device) -> RgbSegformer:
    model = RgbSegformer.from_local_pretrained(resolve(config["model"]["pretrained_directory"]))
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Encoder and decoder must both be trainable")
    return model.to(device)


def overfit_proof(config: dict, data, binding: dict, device: torch.device, steps: int) -> dict:
    if steps < config["proof"]["minimum_steps"]:
        raise ValueError("Proof must run at least the configured minimum number of steps")
    seed_everything(config["experiment"]["seed"])
    model = make_model(config, device)
    train = config["training"]
    criterion = SixClassFocalLoss(train["fine_semantic_class_weights"], train["fine_semantic_focal_gamma"]).to(device)
    # Only training records are used, and one fixed batch is held constant.
    batch = next(iter(DataLoader(Subset(data, list(range(train["batch_size"]))), batch_size=train["batch_size"], num_workers=0)))
    if not bool((batch["image_valid_mask"] & batch["classification_valid_mask"]).any()):
        raise RuntimeError("The train-only proof batch contains no labelled pixels")
    optimizer = torch.optim.AdamW(model.parameter_groups(train["encoder_learning_rate"], train["decoder_learning_rate"], train["weight_decay"]))
    model.eval()
    with torch.no_grad():
        initial = float(loss_for(model, batch, criterion, device, train["precision"]))
    max_encoder_gradient = 0.0
    losses = []
    started = time.monotonic()
    for step in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, batch, criterion, device, train["precision"])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite train-only proof loss")
        loss.backward()
        encoder_norm = sum(float(parameter.grad.detach().float().square().sum()) for parameter in model.network.segformer.parameters() if parameter.grad is not None) ** 0.5
        max_encoder_gradient = max(max_encoder_gradient, encoder_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train["max_gradient_norm"])
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 12 == 0:
            print(f"Training-only feasibility {step + 1}/{steps}: loss={losses[-1]:.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        final = float(loss_for(model, batch, criterion, device, train["precision"]))
    reduction = (initial - final) / max(initial, 1e-12)
    assert_sources(binding["source_code_sha256"])
    verify_protected(config)
    report = {"schema": "msr.rgb_segmenter_train_only_proof.v1", "run_id": run_id(),
              "binding": binding, "steps": steps, "initial_loss": initial, "final_loss": final,
              "loss_reduction_fraction": reduction, "encoder_gradient_norm": max_encoder_gradient,
              "train_sample_ids": list(batch["sample_id"]), "losses": losses,
              "elapsed_seconds": time.monotonic() - started,
              "passes": bool(reduction >= config["proof"]["minimum_loss_reduction_fraction"] and max_encoder_gradient > 0),
              "interpretation": "Train-only optimization feasibility, not validation accuracy; proof weights are discarded."}
    save_proof(config, "overfit_report", report)
    return report


def validation_smoke(config: dict, data, binding: dict, device: torch.device) -> dict:
    if device.type != "cuda":
        raise ValueError("Actual CUDA memory compatibility is required for the validation smoke")
    _, _, evaluate, _, _ = dataset_tools()
    indices = []
    seen = set()
    for index, record in enumerate(data.records):
        if record.city not in seen:
            indices.append(index)
            seen.add(record.city)
    if seen != {"DC", "PHL"}:
        raise RuntimeError("Smoke requires one development sample from each expected city")
    seed_everything(config["experiment"]["seed"])
    model = make_model(config, device).eval()
    loader = DataLoader(Subset(data, indices), batch_size=1, num_workers=0)
    torch.cuda.reset_peak_memory_stats(device)
    evaluation = evaluate(model, loader, device, precision=config["training"]["precision"],
                          expected_ids=[data.records[index].sample_id for index in indices], native_size=1024)
    peak = torch.cuda.max_memory_allocated(device)
    assert_sources(binding["source_code_sha256"])
    verify_protected(config)
    report = {"schema": "msr.rgb_segmenter_validation_smoke.v1", "run_id": run_id(),
              "binding": binding, "passes": True, "device": "cuda", "native_size": 1024,
              "peak_allocated_bytes": peak, "sample_ids": [data.records[index].sample_id for index in indices],
              "evaluation": evaluation,
              "interpretation": "Fresh decoder shape/memory validation only, not a trained candidate comparison."}
    save_proof(config, "validation_smoke_report", report)
    return report


def snapshot_sources(experiment: Path, config_path: Path, binding: dict) -> None:
    shutil.copy2(config_path, experiment / "config.yaml")
    for name in binding["source_code_sha256"]:
        destination = experiment / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolve(name), destination)
    atomic_json(experiment / "binding.json", binding)
    config = load_config(config_path)
    for key in ("overfit_report", "validation_smoke_report"):
        shutil.copy2(resolve(config["proof"][key]), experiment / f"{key}.json")


def verify_resume(path: Path, binding: dict, config: dict) -> dict:
    if path.name != "checkpoint_latest.pt" or not path.parent.name.endswith("_gamus_rgb_segmenter_v1"):
        raise ValueError("Resume only this recipe's checkpoint_latest.pt at an epoch boundary")
    path.resolve().relative_to(resolve(config["experiment"]["output_root"]).resolve())
    # Only locally generated, identity-bound optimizer checkpoints are accepted.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_type") != RgbSegformer.model_type or payload.get("binding") != binding:
        raise RuntimeError("Resume checkpoint does not match the sealed recipe/code/data/software")
    if json.loads((path.parent / "binding.json").read_text(encoding="utf-8")) != binding:
        raise RuntimeError("Experiment binding differs from checkpoint binding")
    if sha256(path.parent / "config.yaml") != binding["config_sha256"]:
        raise RuntimeError("Experiment config snapshot changed")
    for name, expected in binding["source_code_sha256"].items():
        assert_file_hash(path.parent / "source_snapshot" / name, expected)
    epoch = payload.get("epoch")
    history = payload.get("history", [])
    if not isinstance(epoch, int) or epoch <= 0 or [row.get("epoch") for row in history] != list(range(1, epoch + 1)):
        raise RuntimeError("Resume checkpoint has inconsistent completed epoch history")
    return payload


def train(config: dict, config_path: Path, train_data, val_data, binding: dict,
          device: torch.device, resume: Path | None) -> Path:
    _, sampler_factory, evaluate, acceptance, _ = dataset_tools()
    verify_proofs(config, binding)
    seed_everything(config["experiment"]["seed"])
    settings = config["training"]
    generator = torch.Generator().manual_seed(config["experiment"]["seed"])
    sampler = sampler_factory(train_data, config, generator)
    loader = DataLoader(train_data, batch_size=settings["batch_size"], sampler=sampler,
                        num_workers=config["data"]["num_workers"], pin_memory=device.type == "cuda",
                        worker_init_fn=seed_worker, generator=generator, persistent_workers=False)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=config["data"]["num_workers"],
                            pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator)
    payload = verify_resume(resume, binding, config) if resume else None
    model = (RgbSegformer.from_architecture(payload["architecture"]).to(device)
             if payload else make_model(config, device))
    optimizer = torch.optim.AdamW(model.parameter_groups(settings["encoder_learning_rate"], settings["decoder_learning_rate"], settings["weight_decay"]))
    total_steps = settings["epochs"] * math.ceil(len(loader) / settings["gradient_accumulation"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate_factor(
        step, total_steps, settings["warmup_optimizer_steps"], settings["minimum_learning_rate_factor"]))
    criterion = SixClassFocalLoss(settings["fine_semantic_class_weights"], settings["fine_semantic_focal_gamma"]).to(device)
    if payload:
        experiment = resume.parent
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        restore_rng(payload["rng_state"], generator)
        history = payload["history"]
        # Latest checkpoint is authoritative; preserve any uncommitted log rows.
        if (experiment / "metrics.jsonl").exists():
            shutil.copy2(experiment / "metrics.jsonl", experiment / f"metrics_before_resume_{run_id()}.jsonl")
        write_history(experiment / "metrics.jsonl", history)
        best_development = payload["best_development"]
        best_guarded = payload["best_guarded"]
        stale_epochs = payload["stale_epochs"]
        first_epoch = payload["epoch"] + 1
    else:
        experiment = resolve(config["experiment"]["output_root"]) / f"{run_id()}_{config['experiment']['name']}"
        experiment.mkdir(parents=True, exist_ok=False)
        snapshot_sources(experiment, config_path, binding)
        history, best_development, best_guarded, stale_epochs, first_epoch = [], None, None, 0, 1
    atomic_json(resolve("outputs/orchestration/rgb_segmenter_v1_active.json"),
                {"experiment_path": str(experiment.resolve()), "experiment_dir": str(experiment.resolve()),
                 "updated_at_utc": utc_now()})
    status(experiment, "starting", epoch=first_epoch, resumed=bool(payload))
    baseline = json.loads(resolve(config["protocol"]["fixed_v3_independent_replay"]).read_text(encoding="utf-8"))
    reason = "completed"
    committed_epoch = first_epoch - 1
    try:
        for epoch in range(first_epoch, settings["epochs"] + 1):
            assert_file_hash(config_path, binding["config_sha256"])
            assert_sources(binding["source_code_sha256"])
            verify_protected(config)
            verify_pretrained(config)
            verify_data_unchanged(config, train_data, val_data, binding)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            started = time.monotonic()
            accumulation = settings["gradient_accumulation"]
            for batch_index, batch in enumerate(loader, start=1):
                loss = loss_for(model, batch, criterion, device, settings["precision"])
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"Nonfinite training loss in epoch {epoch}, batch {batch_index}")
                window_start = ((batch_index - 1) // accumulation) * accumulation
                window_size = min(accumulation, len(loader) - window_start)
                (loss / window_size).backward()
                if batch_index % accumulation == 0 or batch_index == len(loader):
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_gradient_norm"])
                    if not bool(torch.isfinite(norm)):
                        raise RuntimeError("Nonfinite gradient norm")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
                if batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader):
                    status(experiment, "training", epoch=epoch, batch=batch_index, total_batches=len(loader), loss=losses[-1])
                if batch_index % 100 == 0:
                    print(f"Epoch {epoch}/{settings['epochs']} | {batch_index}/{len(loader)} batches | loss {losses[-1]:.5f}", flush=True)
            def progress(update):
                done, count = update["completed_batches"], update["total_batches"]
                if done == 1 or done % 10 == 0 or done == count:
                    status(experiment, "validation", epoch=epoch, completed_batches=done, total_batches=count)
            status(experiment, "validation", epoch=epoch, completed_batches=0, total_batches=len(val_loader))
            validation = evaluate(model, val_loader, device, precision=settings["precision"],
                                  expected_ids=[record.sample_id for record in val_data.records], native_size=1024,
                                  progress_callback=progress)
            gate = acceptance(validation, baseline, config["protocol"]["classification_acceptance"],
                              config["protocol"]["classification_acceptance_by_city"])
            score = validation["overall"]["six_class_identification"]["macro_f1"]
            if not math.isfinite(score):
                raise RuntimeError("Nonfinite validation macro F1")
            improved = best_development is None or score > best_development["macro_f1"] + 1e-4
            eligible = gate["passes"]
            if improved:
                best_development = {"epoch": epoch, "macro_f1": score}
                stale_epochs = 0
            else:
                stale_epochs += 1
            guarded_improved = eligible and (best_guarded is None or score > best_guarded["macro_f1"])
            if guarded_improved:
                best_guarded = {"epoch": epoch, "macro_f1": score}
            row = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "validation": validation,
                   "gate": gate, "elapsed_seconds": time.monotonic() - started, "updated_at_utc": utc_now(),
                   "development_only": True, "promotion_eligible": False}
            history.append(row)
            assert_file_hash(config_path, binding["config_sha256"])
            assert_sources(binding["source_code_sha256"])
            verify_protected(config)
            verify_data_unchanged(config, train_data, val_data, binding)
            checkpoint = {"model_type": RgbSegformer.model_type, "architecture": model.architecture,
                          "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                          "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "binding": binding,
                          "history": history, "best_development": best_development, "best_guarded": best_guarded,
                          "stale_epochs": stale_epochs, "rng_state": rng_state(generator),
                          "promotion_eligible": False, "input_contract": "raw_rgb_01_imagenet_normalized_in_model"}
            atomic_checkpoint(experiment / f"checkpoint_epoch_{epoch:03d}.pt", checkpoint)
            atomic_checkpoint(experiment / "checkpoint_latest.pt", checkpoint)
            committed_epoch = epoch
            if improved:
                atomic_checkpoint(experiment / "checkpoint_best_development.pt", checkpoint)
            if guarded_improved:
                atomic_checkpoint(experiment / "checkpoint_best_guarded.pt", checkpoint)
            write_history(experiment / "metrics.jsonl", history)
            status(experiment, "epoch_complete", epoch=epoch, macro_f1=score, guards_pass=eligible)
            print(f"Epoch {epoch} validated | six-class F1 {score:.4f} | guards {'PASS' if eligible else 'FAIL'}", flush=True)
            if epoch >= settings["early_stopping_min_epochs"] and stale_epochs >= settings["early_stopping_patience"]:
                reason = "early_stopped"
                break
        summary = {"stage": reason, "completed_epochs": len(history), "best_development": best_development,
                   "best_guarded": best_guarded, "development_only": True, "promotion_performed": False,
                   "protected": verify_protected(config), "updated_at_utc": utc_now()}
        atomic_json(experiment / "training_summary.json", summary)
        status(experiment, reason, **{key: value for key, value in summary.items() if key != "stage"})
        return experiment
    except BaseException as exc:
        failure = {"stage": "failed", "error_type": type(exc).__name__, "error": str(exc),
                   "completed_epochs": committed_epoch, "evaluated_epochs": len(history),
                   "updated_at_utc": utc_now(), "promotion_performed": False}
        atomic_json(experiment / f"failure_{run_id()}.json", failure)
        atomic_json(experiment / "training_summary.json", failure)
        status(experiment, "failed", error=failure["error"], error_type=failure["error_type"],
               completed_epochs=committed_epoch)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gamus_rgb_segmenter_v1.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--overfit-steps", type=int)
    modes.add_argument("--validation-smoke-only", action="store_true")
    modes.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    config_path = resolve(args.config)
    config = load_config(config_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; do not silently train on CPU")
    torch.set_num_threads(4)
    train_data, val_data, binding = preflight(config, config_path)
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "train_samples": len(train_data), "validation_samples": len(val_data)}))
    elif args.overfit_steps is not None:
        report = overfit_proof(config, train_data, binding, device, args.overfit_steps)
        print(json.dumps({key: report[key] for key in ("passes", "initial_loss", "final_loss", "loss_reduction_fraction")}))
        return 0 if report["passes"] else 1
    elif args.validation_smoke_only:
        report = validation_smoke(config, val_data, binding, device)
        print(json.dumps({key: report[key] for key in ("passes", "native_size", "peak_allocated_bytes")}))
    else:
        print(f"Experiment saved: {train(config, config_path, train_data, val_data, binding, device, resolve(args.resume) if args.resume else None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
