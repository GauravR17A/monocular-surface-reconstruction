"""Reproducible single-GPU training entry point for Monocular Surface Reconstruction."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from msr.data.manifest import assert_disjoint_regions, load_manifest
from msr.data.raster_dataset import RasterHeightDataset
from msr.data.webdataset_dataset import make_webdataset
from msr.evaluation.metrics import StreamingRegressionMetrics
from msr.models.height_net import HeightNet
from msr.training.losses import CompositeHeightLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    records,
    data_config: dict,
    training_config: dict,
    *,
    training: bool,
) -> DataLoader:
    patch_size = (
        int(data_config["patch_size"])
        if training
        else int(data_config.get("validation_patch_size", data_config["patch_size"]))
    )
    dataset = RasterHeightDataset(
        records,
        patch_size=patch_size,
        random_crop=training,
        augment=training,
        samples_per_epoch=(
            int(data_config.get("train_samples_per_epoch", len(records))) if training else None
        ),
        height_min_m=float(data_config["height_min_m"]),
        height_max_m=float(data_config["height_max_m"]),
    )
    workers = int(data_config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]) if training else 1,
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=training,
    )


def make_web_loader(
    data_config: dict,
    training_config: dict,
    *,
    training: bool,
    seed: int,
):
    split = "train" if training else "validation"
    patch_size = int(
        data_config["patch_size"]
        if training
        else data_config.get("validation_patch_size", data_config["patch_size"])
    )
    dataset_options = {
        "patch_size": patch_size,
        "training": training,
        "height_min_m": float(data_config["height_min_m"]),
        "height_max_m": float(data_config["height_max_m"]),
        "seed": seed,
        "foreground_crop_probability": (
            float(data_config.get("foreground_crop_probability", 0.0))
            if training
            else 0.0
        ),
        "tall_crop_probability": (
            float(data_config.get("tall_crop_probability", 0.0)) if training else 0.0
        ),
        "tall_threshold_m": float(data_config.get("tall_threshold_m", 20.0)),
    }
    city_patterns = data_config.get("city_balanced_train_shards") if training else None
    if city_patterns:
        city_weights = data_config.get("city_sampling_weights", {})
        cities = list(city_patterns)
        probabilities = [float(city_weights.get(city, 1.0)) for city in cities]
        probability_sum = sum(probabilities)
        if probability_sum <= 0.0:
            raise ValueError("city sampling weights must sum to a positive value")
        probabilities = [value / probability_sum for value in probabilities]
        city_datasets = []
        for city in cities:
            shards = sorted(glob.glob(str(city_patterns[city])))
            if not shards:
                raise FileNotFoundError(f"No balanced training shards for {city}")
            city_datasets.append(make_webdataset(shards, **dataset_options))
        dataset = wds.RandomMix(city_datasets, probabilities)
        print(
            "Using city-balanced training streams: "
            + ", ".join(
                f"{city}={probability:.3f}"
                for city, probability in zip(cities, probabilities)
            )
        )
    else:
        shards = sorted(glob.glob(str(data_config[f"{split}_shards"])))
        dataset = make_webdataset(shards, **dataset_options)
    workers = int(data_config.get("num_workers", 0))
    batch_size = int(training_config["batch_size"]) if training else 1
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=training,
    )
    if training:
        batches = math.ceil(int(data_config["train_samples_per_epoch"]) / batch_size)
        loader = loader.with_epoch(nbatches=batches)
    return loader


def load_region_names(path: str | Path, column: str = "public_dir") -> set[str]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Manifest {path} lacks region column {column}")
        regions = {row[column].strip() for row in reader if row[column].strip()}
    if not regions:
        raise ValueError(f"Manifest {path} contains no regions")
    return regions


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CompositeHeightLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    precision: str,
    accumulation: int,
    max_gradient_norm: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    batches = 0
    progress = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(progress, start=1):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["height"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        building = batch["building_mask"].to(device, non_blocking=True)
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"
        ):
            output = model(image)
            loss, components = criterion(output, target, valid, building)
            scaled_loss = loss / accumulation
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}")
        scaler.scale(scaled_loss).backward()
        if step % accumulation == 0:
            if max_gradient_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            scaler.step(optimizer)
            scaler.update()
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
        optimizer.zero_grad(set_to_none=True)
    return {name: value / max(batches, 1) for name, value in sums.items()}


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CompositeHeightLoss,
    device: torch.device,
    *,
    precision: str,
) -> tuple[dict[str, float], dict[str, int | float | None]]:
    model.eval()
    metric = StreamingRegressionMetrics()
    building_metric = StreamingRegressionMetrics()
    region_metrics: dict[str, StreamingRegressionMetrics] = defaultdict(
        StreamingRegressionMetrics
    )
    region_building_metrics: dict[str, StreamingRegressionMetrics] = defaultdict(
        StreamingRegressionMetrics
    )
    loss_sums: dict[str, float] = {}
    batches = 0
    for batch in tqdm(loader, desc="validation", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["height"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        building = batch["building_mask"].to(device, non_blocking=True)
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=precision != "fp32"
        ):
            output = model(image)
            _, components = criterion(output, target, valid, building)
        if not torch.isfinite(output["height"]).all():
            raise FloatingPointError("Non-finite height values during validation")
        for name, value in components.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value)
        metric.update(
            output["height"].float().cpu().numpy(),
            target.float().cpu().numpy(),
            valid.cpu().numpy(),
        )
        building_metric.update(
            output["height"].float().cpu().numpy(),
            target.float().cpu().numpy(),
            (valid & building.bool()).cpu().numpy(),
        )
        regions = batch.get("region")
        if regions is not None:
            for index, region in enumerate(regions):
                region_name = str(region)
                region_metrics[region_name].update(
                    output["height"][index : index + 1].float().cpu().numpy(),
                    target[index : index + 1].float().cpu().numpy(),
                    valid[index : index + 1].cpu().numpy(),
                )
                region_building_metrics[region_name].update(
                    output["height"][index : index + 1].float().cpu().numpy(),
                    target[index : index + 1].float().cpu().numpy(),
                    (valid[index : index + 1] & building[index : index + 1].bool())
                    .cpu()
                    .numpy(),
                )
        batches += 1
    losses = {name: value / max(batches, 1) for name, value in loss_sums.items()}
    metrics = metric.compute()
    if building_metric.count:
        metrics.update(
            {f"building_{name}": value for name, value in building_metric.compute().items()}
        )
    if region_metrics:
        region_results = {
            region: metric.compute() for region, metric in sorted(region_metrics.items())
        }
        region_building_results = {
            region: metric.compute()
            for region, metric in sorted(region_building_metrics.items())
            if metric.count
        }
        metrics["region_macro_rmse_m"] = float(
            np.mean([result["rmse_m"] for result in region_results.values()])
        )
        metrics["worst_region_rmse_m"] = float(
            max(result["rmse_m"] for result in region_results.values())
        )
        if region_building_results:
            metrics["region_macro_building_rmse_m"] = float(
                np.mean(
                    [result["rmse_m"] for result in region_building_results.values()]
                )
            )
        metrics["regions"] = {
            region: {
                "all": result,
                "building": region_building_results.get(region),
            }
            for region, result in region_results.items()
        }
    return losses, metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    metrics: dict,
    config: dict,
    training_state: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "metrics": metrics,
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


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["cuda"]])


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

    seed = int(experiment_config["seed"])
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured training run")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    data_format = str(data_config.get("format", "raster")).lower()
    if data_format == "raster":
        train_records = load_manifest(data_config["train_manifest"])
        validation_records = load_manifest(data_config["val_manifest"])
        test_records = load_manifest(data_config["test_manifest"])
        assert_disjoint_regions(train_records, validation_records, test_records)
        train_loader = make_loader(
            train_records, data_config, training_config, training=True
        )
        validation_loader = make_loader(
            validation_records, data_config, training_config, training=False
        )
    elif data_format == "webdataset":
        region_sets = [
            load_region_names(data_config[name])
            for name in ("train_manifest", "val_manifest", "test_manifest")
        ]
        for left in range(len(region_sets)):
            for right in range(left + 1, len(region_sets)):
                overlap = region_sets[left] & region_sets[right]
                if overlap:
                    raise ValueError(f"Geographic leakage between splits: {sorted(overlap)}")
        train_loader = make_web_loader(
            data_config, training_config, training=True, seed=seed
        )
        validation_loader = make_web_loader(
            data_config, training_config, training=False, seed=seed
        )
    else:
        raise ValueError("data.format must be raster or webdataset")
    model = HeightNet(
        backbone=model_config["backbone"],
        pretrained=bool(model_config["pretrained"]),
        decoder_channels=model_config["decoder_channels"],
        auxiliary_building_head=bool(model_config["auxiliary_building_head"]),
    ).to(device)
    initial_checkpoint = training_config.get("initial_checkpoint")
    if initial_checkpoint and args.resume is None:
        initial_state = torch.load(
            Path(initial_checkpoint), map_location=device, weights_only=False
        )
        model.load_state_dict(initial_state["model"])
        print(f"Initialized model weights from {initial_checkpoint}")
    criterion = CompositeHeightLoss(
        huber_delta_m=float(training_config.get("huber_delta_m", 1.0)),
        gradient_weight=float(training_config["gradient_loss_weight"]),
        foreground_regression_weight=float(
            training_config.get("foreground_regression_weight", 1.0)
        ),
        foreground_mse_weight=float(training_config.get("foreground_mse_weight", 0.0)),
        tall_regression_weight=float(training_config.get("tall_regression_weight", 0.0)),
        tall_threshold_m=float(training_config.get("tall_threshold_m", 20.0)),
        building_weight=float(training_config["building_loss_weight"]),
    )
    learning_rate = float(training_config["learning_rate"])
    encoder_lr_multiplier = float(
        training_config.get("encoder_learning_rate_multiplier", 1.0)
    )
    encoder_parameters = list(model.encoder.parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_parameter_ids
    ]
    if encoder_lr_multiplier == 1.0:
        parameter_groups = model.parameters()
    else:
        parameter_groups = [
            {
                "params": encoder_parameters,
                "lr": learning_rate * encoder_lr_multiplier,
                "group_name": "encoder",
            },
            {
                "params": decoder_parameters,
                "lr": learning_rate,
                "group_name": "decoder",
            },
        ]
    optimizer = AdamW(
        parameter_groups,
        lr=learning_rate,
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = int(training_config["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    precision = str(
        training_config.get(
            "precision", "fp16" if bool(training_config.get("amp", True)) else "fp32"
        )
    ).lower()
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("training.precision must be fp32, bf16, or fp16")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured GPU does not support bf16 training")
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    best_rmse = float("inf")
    best_mae = float("inf")
    best_building_rmse = float("inf")
    best_region_macro_rmse = float("inf")
    epochs_without_improvement = 0
    start_epoch = 1
    if args.resume is not None:
        resume_path = args.resume.resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("config") != config:
            raise ValueError(
                "Resume config differs from the checkpoint config; use the immutable "
                "config.yaml stored in the experiment directory"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint.get("training_state", {})
        best_rmse = float(state.get("best_rmse", checkpoint["metrics"]["rmse_m"]))
        best_mae = float(state.get("best_mae", checkpoint["metrics"]["mae_m"]))
        best_building_rmse = float(
            state.get(
                "best_building_rmse",
                checkpoint["metrics"].get("building_rmse_m", float("inf")),
            )
        )
        best_region_macro_rmse = float(
            state.get(
                "best_region_macro_rmse",
                checkpoint["metrics"].get("region_macro_rmse_m", float("inf")),
            )
        )
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        experiment_dir = resume_path.parent
        restore_rng_state(checkpoint.get("rng_state"))
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
    history_path = experiment_dir / "metrics.jsonl"

    for epoch in range(start_epoch, epochs + 1):
        train_losses = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            precision=precision,
            accumulation=int(training_config["gradient_accumulation"]),
            max_gradient_norm=float(training_config.get("max_gradient_norm", 0.0)),
        )
        validation_losses, validation_metrics = validate(
            model, validation_loader, criterion, device, precision=precision
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": max(scheduler.get_last_lr()),
            "encoder_learning_rate": scheduler.get_last_lr()[0],
            "train_loss": train_losses,
            "validation_loss": validation_losses,
            "validation_metrics": validation_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2))
        rmse_improved = float(validation_metrics["rmse_m"]) < best_rmse
        mae_improved = float(validation_metrics["mae_m"]) < best_mae
        building_rmse_improved = (
            float(validation_metrics.get("building_rmse_m", float("inf")))
            < best_building_rmse
        )
        region_macro_improved = (
            float(validation_metrics.get("region_macro_rmse_m", float("inf")))
            < best_region_macro_rmse
        )
        if rmse_improved:
            best_rmse = float(validation_metrics["rmse_m"])
        if mae_improved:
            best_mae = float(validation_metrics["mae_m"])
        if building_rmse_improved:
            best_building_rmse = float(validation_metrics["building_rmse_m"])
        if region_macro_improved:
            best_region_macro_rmse = float(validation_metrics["region_macro_rmse_m"])
        epochs_without_improvement = (
            0 if rmse_improved else epochs_without_improvement + 1
        )
        state = {
            "best_rmse": best_rmse,
            "best_mae": best_mae,
            "best_building_rmse": best_building_rmse,
            "best_region_macro_rmse": best_region_macro_rmse,
            "epochs_without_improvement": epochs_without_improvement,
            "precision": precision,
        }
        if rmse_improved:
            save_checkpoint(
                experiment_dir / "checkpoint_best_rmse.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                validation_metrics,
                config,
                state,
            )
        if mae_improved:
            save_checkpoint(
                experiment_dir / "checkpoint_best_mae.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                validation_metrics,
                config,
                state,
            )
        if building_rmse_improved:
            save_checkpoint(
                experiment_dir / "checkpoint_best_building_rmse.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                validation_metrics,
                config,
                state,
            )
        if region_macro_improved:
            save_checkpoint(
                experiment_dir / "checkpoint_best_region_macro_rmse.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                validation_metrics,
                config,
                state,
            )
        save_checkpoint(
            experiment_dir / "checkpoint_latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            validation_metrics,
            config,
            state,
        )
        if epochs_without_improvement >= int(training_config["early_stopping_patience"]):
            print(f"Early stopping after epoch {epoch}")
            break


if __name__ == "__main__":
    main()
