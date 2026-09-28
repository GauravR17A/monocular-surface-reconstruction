"""Sweep conservative vegetation gates without retraining or changing a checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from msr.data.surface_dataset import (
    LANDSCAPE_CLASSES,
    MultiDomainSurfaceDataset,
    load_surface_manifest,
)
from msr.evaluation.metrics import StreamingRegressionMetrics
from msr.inference.predict import load_predictor


TRANSFORMS = (
    "identity",
    "temperature_0.80",
    "temperature_0.67",
    "temperature_0.50",
    "soft_0.10",
    "soft_0.20",
    "soft_0.30",
    "soft_0.40",
    "hard_0.30",
    "hard_0.40",
    "hard_0.50",
    "hard_0.60",
)


def _transform_probability(name: str, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1.0e-5, 1.0 - 1.0e-5)
    if name == "identity":
        return probability
    if name.startswith("temperature_"):
        temperature = float(name.split("_", maxsplit=1)[1])
        logit = np.log(probability / (1.0 - probability)) / temperature
        return 1.0 / (1.0 + np.exp(-logit))
    if name.startswith("soft_"):
        threshold = float(name.split("_", maxsplit=1)[1])
        return np.clip((probability - threshold) / (1.0 - threshold), 0.0, 1.0)
    if name.startswith("hard_"):
        threshold = float(name.split("_", maxsplit=1)[1])
        return probability * (probability >= threshold)
    raise ValueError(f"Unknown vegetation transform: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-guard-regression-m", type=float, default=0.02)
    parser.add_argument("--only-transform", choices=TRANSFORMS[1:])
    args = parser.parse_args()
    transforms = (
        ("identity", args.only_transform) if args.only_transform else TRANSFORMS
    )

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    data_config = config["data"]
    dataset = MultiDomainSurfaceDataset(
        load_surface_manifest(args.manifest),
        patch_size=int(data_config.get("validation_patch_size", 384)),
        random_crop=False,
        augment=False,
        rgb_scale=float(data_config.get("rgb_scale", 255.0)),
        height_max_m=float(data_config.get("height_max_m", 200.0)),
        building_threshold_m=float(data_config.get("building_threshold_m", 2.0)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model, _ = load_predictor(args.checkpoint, device=device)
    model.eval()

    overall = {name: StreamingRegressionMetrics() for name in transforms}
    domains = {
        name: {domain: StreamingRegressionMetrics() for domain in LANDSCAPE_CLASSES}
        for name in transforms
    }
    landscapes = {name: defaultdict(StreamingRegressionMetrics) for name in transforms}

    with torch.inference_mode():
        for batch in tqdm(loader, desc="vegetation fusion sweep"):
            image = batch["image"].to(device, non_blocking=True)
            prior = batch["relative_prior"].to(device, non_blocking=True)
            output = model(image, prior)

            target = batch["height"].numpy()
            valid = batch["regression_mask"].numpy()
            domain = batch["domain_target"].numpy()
            base = output["base_height"].float().cpu().numpy()
            canopy = output["canopy_height"].float().cpu().numpy()
            refinement = output["refinement_strength"].float().cpu().numpy()
            protected = (
                output["protected_building_logits"].sigmoid().float().cpu().numpy()
            )
            probabilities = output["domain_probabilities"].float().cpu().numpy()
            learned_building = probabilities[:, 1:2]
            vegetation = probabilities[:, 2:3]
            non_building = np.clip(
                (1.0 - protected) * (1.0 - learned_building), 0.0, 1.0
            )

            for name in transforms:
                if name == "identity":
                    prediction = output["height"].float().cpu().numpy()
                else:
                    adjusted = _transform_probability(name, vegetation)
                    building_protection = (
                        1.0
                        - np.power(non_building, float(model.building_protection_power))
                    ) * (1.0 - adjusted)
                    effective = refinement * (1.0 - building_protection)
                    gated = learned_building * base + adjusted * canopy
                    prediction = np.maximum(base + effective * (gated - base), 0.0)

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

    results: dict[str, dict] = {}
    for name in transforms:
        results[name] = {
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

    baseline = results["identity"]
    guard = float(args.max_guard_regression_m)
    guarded_keys = ("overall", "urban", "ground", "building")
    for name, result in results.items():
        regressions = {
            "overall": result["overall"]["rmse_m"] - baseline["overall"]["rmse_m"],
            "urban": result["landscapes"]["urban"]["rmse_m"]
            - baseline["landscapes"]["urban"]["rmse_m"],
            "forest": result["landscapes"]["forest"]["rmse_m"]
            - baseline["landscapes"]["forest"]["rmse_m"],
            "ground": result["domains"]["ground"]["rmse_m"]
            - baseline["domains"]["ground"]["rmse_m"],
            "building": result["domains"]["building"]["rmse_m"]
            - baseline["domains"]["building"]["rmse_m"],
            "vegetation": result["domains"]["vegetation"]["rmse_m"]
            - baseline["domains"]["vegetation"]["rmse_m"],
        }
        result["regressions_m"] = regressions
        result["eligible"] = (
            all(regressions[key] <= guard for key in guarded_keys)
            and regressions["forest"] <= 0.0
            and regressions["vegetation"] < 0.0
        )

    eligible = [(name, result) for name, result in results.items() if result["eligible"]]
    recommended = min(
        eligible,
        key=lambda item: (
            item[1]["domains"]["vegetation"]["rmse_m"],
            item[1]["overall"]["rmse_m"],
        ),
        default=("identity", results["identity"]),
    )[0]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
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
