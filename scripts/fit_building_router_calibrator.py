"""Fit and validate a lightweight building router without changing the neural model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from msr.data.surface_dataset import (
    LANDSCAPE_CLASSES,
    MultiDomainSurfaceDataset,
    load_surface_manifest,
)
from msr.evaluation.metrics import StreamingRegressionMetrics
from msr.inference.predict import load_predictor


FEATURE_NAMES = (
    "protected_logit",
    "learned_logit",
    "vegetation_logit",
    "protected_learned_product",
    "log_base_height",
    "building_residual_scaled",
    "relative_prior",
    "normalized_red",
    "normalized_green",
    "normalized_blue",
)
THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
BLENDS = (0.25, 0.50, 0.75, 1.00)


def _dataset(manifest: Path, data_config: dict) -> MultiDomainSurfaceDataset:
    return MultiDomainSurfaceDataset(
        load_surface_manifest(manifest),
        patch_size=int(data_config.get("validation_patch_size", 384)),
        random_crop=False,
        augment=False,
        rgb_scale=float(data_config.get("rgb_scale", 255.0)),
        height_max_m=float(data_config.get("height_max_m", 200.0)),
        building_threshold_m=float(data_config.get("building_threshold_m", 2.0)),
    )


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def _features(batch: dict, output: dict[str, torch.Tensor]) -> np.ndarray:
    protected = output["protected_building_logits"].sigmoid().float().cpu().numpy()
    probabilities = output["domain_probabilities"].float().cpu().numpy()
    learned = probabilities[:, 1:2]
    vegetation = probabilities[:, 2:3]
    base = output["base_height"].float().cpu().numpy()
    residual = (
        output["building_height"] - output["base_height"]
    ).float().cpu().numpy()
    prior = batch["relative_prior"].numpy()
    image = batch["image"].numpy()
    channels = [
        _logit(protected),
        _logit(learned),
        _logit(vegetation),
        protected * learned * (1.0 - vegetation),
        np.log1p(np.maximum(base, 0.0)) / 5.0,
        residual / 30.0,
        prior,
        image[:, 0:1],
        image[:, 1:2],
        image[:, 2:3],
    ]
    return np.concatenate(channels, axis=1).transpose(0, 2, 3, 1)


def _candidate_specs() -> list[tuple[str, float, float]]:
    specs = [("protected", 1.0, 0.0)]
    for threshold in THRESHOLDS:
        for blend in BLENDS:
            specs.append((f"cal_t{threshold:.2f}_b{blend:.2f}", threshold, blend))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-chip", type=int, default=1024)
    parser.add_argument("--max-guard-regression-m", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_config = payload["config"]["data"]
    device = torch.device(args.device)
    model, _ = load_predictor(args.checkpoint, device=device)
    model.eval()
    rng = np.random.default_rng(args.seed)
    sampled_features: list[np.ndarray] = []
    sampled_labels: list[np.ndarray] = []
    train_loader = DataLoader(
        _dataset(args.train_manifest, data_config),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    with torch.inference_mode():
        for batch in tqdm(train_loader, desc="sample router training pixels"):
            output = model(
                batch["image"].to(device, non_blocking=True),
                batch["relative_prior"].to(device, non_blocking=True),
            )
            features = _features(batch, output)[0]
            valid = batch["domain_valid_mask"].numpy()[0].astype(bool)
            labels = (
                batch["domain_target"].numpy()[0]
                == LANDSCAPE_CLASSES["building"]
            )
            indices = np.flatnonzero(valid)
            if indices.size > args.samples_per_chip:
                indices = rng.choice(
                    indices, size=args.samples_per_chip, replace=False
                )
            sampled_features.append(features.reshape(-1, features.shape[-1])[indices])
            sampled_labels.append(labels.reshape(-1)[indices])

    train_x = np.concatenate(sampled_features).astype(np.float32, copy=False)
    train_y = np.concatenate(sampled_labels).astype(np.uint8, copy=False)
    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=300,
        random_state=args.seed,
        solver="lbfgs",
    ).fit(scaler.transform(train_x), train_y)

    specs = _candidate_specs()
    overall = {name: StreamingRegressionMetrics() for name, _, _ in specs}
    domains = {
        name: {domain: StreamingRegressionMetrics() for domain in LANDSCAPE_CLASSES}
        for name, _, _ in specs
    }
    landscapes = {
        name: defaultdict(StreamingRegressionMetrics) for name, _, _ in specs
    }
    confusion = {
        threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for threshold in THRESHOLDS
    }
    validation_loader = DataLoader(
        _dataset(args.validation_manifest, data_config),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    with torch.inference_mode():
        for batch in tqdm(validation_loader, desc="validate calibrated router"):
            output = model(
                batch["image"].to(device, non_blocking=True),
                batch["relative_prior"].to(device, non_blocking=True),
            )
            features = _features(batch, output)
            flat = features.reshape(-1, features.shape[-1])
            probability = classifier.predict_proba(scaler.transform(flat))[:, 1]
            probability = probability.reshape(features.shape[:-1])[:, None]
            current = output["height"].float().cpu().numpy()
            residual = (
                output["building_height"] - output["base_height"]
            ).float().cpu().numpy()
            target = batch["height"].numpy()
            valid = batch["regression_mask"].numpy().astype(bool)
            domain = batch["domain_target"].numpy()

            for name, threshold, blend in specs:
                if name == "protected":
                    prediction = current
                else:
                    gate = probability >= threshold
                    prediction = np.maximum(current + blend * gate * residual, 0.0)
                overall[name].update(prediction, target, valid)
                for domain_name, code in LANDSCAPE_CLASSES.items():
                    domains[name][domain_name].update(
                        prediction,
                        target,
                        valid & (domain[:, None] == code),
                    )
                for index, landscape in enumerate(batch["landscape"]):
                    landscapes[name][str(landscape)].update(
                        prediction[index : index + 1],
                        target[index : index + 1],
                        valid[index : index + 1],
                    )

            semantic_valid = batch["domain_valid_mask"].numpy()[:, None].astype(bool)
            truth = domain[:, None] == LANDSCAPE_CLASSES["building"]
            for threshold in THRESHOLDS:
                predicted = probability >= threshold
                counts = confusion[threshold]
                counts["tp"] += int(np.count_nonzero(semantic_valid & truth & predicted))
                counts["fp"] += int(np.count_nonzero(semantic_valid & ~truth & predicted))
                counts["fn"] += int(np.count_nonzero(semantic_valid & truth & ~predicted))
                counts["tn"] += int(np.count_nonzero(semantic_valid & ~truth & ~predicted))

    results: dict[str, dict] = {}
    for name, threshold, blend in specs:
        results[name] = {
            "threshold": threshold,
            "blend": blend,
            "overall": overall[name].compute(),
            "domains": {
                domain: metric.compute()
                for domain, metric in domains[name].items()
                if metric.count
            },
            "landscapes": {
                landscape: metric.compute()
                for landscape, metric in sorted(landscapes[name].items())
                if metric.count
            },
        }
    baseline = results["protected"]
    guard = float(args.max_guard_regression_m)
    for result in results.values():
        regressions = {
            "overall": result["overall"]["rmse_m"] - baseline["overall"]["rmse_m"],
            "urban": result["landscapes"]["urban"]["rmse_m"]
            - baseline["landscapes"]["urban"]["rmse_m"],
            "forest": result["landscapes"]["forest"]["rmse_m"]
            - baseline["landscapes"]["forest"]["rmse_m"],
            "ground": result["domains"]["ground"]["rmse_m"]
            - baseline["domains"]["ground"]["rmse_m"],
            "vegetation": result["domains"]["vegetation"]["rmse_m"]
            - baseline["domains"]["vegetation"]["rmse_m"],
        }
        result["regressions_m"] = regressions
        result["building_improvement_m"] = (
            baseline["domains"]["building"]["rmse_m"]
            - result["domains"]["building"]["rmse_m"]
        )
        result["eligible"] = all(value <= guard for value in regressions.values())

    eligible = [
        (name, result)
        for name, result in results.items()
        if result["eligible"] and result["building_improvement_m"] > 0.0
    ]
    recommended = min(
        eligible,
        key=lambda item: item[1]["domains"]["building"]["rmse_m"],
        default=("protected", results["protected"]),
    )[0]
    router_metrics = {}
    for threshold, counts in confusion.items():
        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        f05 = 1.25 * precision * recall / max(0.25 * precision + recall, 1e-12)
        router_metrics[f"{threshold:.2f}"] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f0_5": f05,
        }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "train_manifest": str(args.train_manifest.resolve()),
        "validation_manifest": str(args.validation_manifest.resolve()),
        "training_pixel_count": int(train_y.size),
        "training_building_fraction": float(train_y.mean()),
        "feature_names": FEATURE_NAMES,
        "calibrator": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficient": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
        },
        "router_metrics": router_metrics,
        "max_guard_regression_m": guard,
        "recommended": recommended,
        "candidates": results,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({"recommended": recommended, "metrics": results[recommended]}, indent=2))


if __name__ == "__main__":
    main()
