"""Sweep guarded canopy-expert fusion using learned and RGB vegetation gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from msr.data.raster_dataset import IMAGENET_MEAN, IMAGENET_STD
from msr.data.surface_dataset import (
    LANDSCAPE_CLASSES,
    MultiDomainSurfaceDataset,
    load_surface_manifest,
)
from msr.evaluation.metrics import StreamingRegressionMetrics
from msr.inference.predict import load_predictor
from msr.inference.viewer_mesh import vegetation_mask_from_rgb


THRESHOLDS = (0.85, 0.90, 0.93, 0.95, 0.97)
BLENDS = (0.10, 0.25, 0.50)
MODES = ("model", "rgb_and_model")


def _specs() -> list[tuple[str, str, float, float]]:
    specs = [("protected", "none", 0.0, 0.0)]
    for mode in MODES:
        for threshold in THRESHOLDS:
            for blend in BLENDS:
                name = f"{mode}_t{threshold:.2f}_b{blend:.2f}"
                specs.append((name, mode, threshold, blend))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-guard-regression-m", type=float, default=0.05)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_config = payload["config"]["data"]
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
    specs = _specs()

    overall = {name: StreamingRegressionMetrics() for name, *_ in specs}
    domains = {
        name: {domain: StreamingRegressionMetrics() for domain in LANDSCAPE_CLASSES}
        for name, *_ in specs
    }
    landscapes = {name: defaultdict(StreamingRegressionMetrics) for name, *_ in specs}

    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="vegetation expert fusion sweep"):
            image = batch["image"].to(device, non_blocking=True)
            prior = batch["relative_prior"].to(device, non_blocking=True)
            output = model(image, prior)

            target = batch["height"].numpy()
            valid = batch["regression_mask"].numpy()
            domain = batch["domain_target"].numpy()
            current = output["height"].float().cpu().numpy()
            canopy = output["canopy_height"].float().cpu().numpy()
            vegetation = output["domain_probabilities"][:, 2:3].float().cpu().numpy()
            rgb = batch["image"].numpy()[0] * std + mean
            rgb_mask = vegetation_mask_from_rgb(rgb)[None, None]

            for name, mode, threshold, blend in specs:
                if mode == "none":
                    prediction = current
                else:
                    gate = vegetation >= threshold
                    if mode == "rgb_and_model":
                        gate &= rgb_mask
                    prediction = np.maximum(
                        current + blend * gate * (canopy - current),
                        0.0,
                    )
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
    for name, mode, threshold, blend in specs:
        results[name] = {
            "mode": mode,
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
            "building": result["domains"]["building"]["rmse_m"]
            - baseline["domains"]["building"]["rmse_m"],
            "vegetation": result["domains"]["vegetation"]["rmse_m"]
            - baseline["domains"]["vegetation"]["rmse_m"],
        }
        result["regressions_m"] = regressions
        result["eligible"] = (
            all(
                regressions[key] <= guard
                for key in ("overall", "urban", "ground", "building")
            )
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
        default=("protected", baseline),
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
