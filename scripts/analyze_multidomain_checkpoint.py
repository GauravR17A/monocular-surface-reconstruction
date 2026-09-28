"""Inspect expert quality, semantic routing, and fusion for a multidomain checkpoint."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
from msr.evaluation.error_analysis import BinarySums
from msr.evaluation.metrics import StreamingRegressionMetrics
from msr.inference.predict import load_predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fusion-mode-override",
        choices=("legacy", "protected_vegetation"),
        help="Evaluate a fusion rule without mutating the saved checkpoint.",
    )
    parser.add_argument("--building-protection-power", type=float)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    data_config = config["data"]
    manifest = args.manifest or Path(data_config["val_manifest"])
    records = load_surface_manifest(manifest)
    dataset = MultiDomainSurfaceDataset(
        records,
        patch_size=int(data_config.get("validation_patch_size", data_config["patch_size"])),
        random_crop=False,
        augment=False,
        rgb_scale=float(data_config.get("rgb_scale", 255.0)),
        height_max_m=float(data_config.get("height_max_m", 200.0)),
        building_threshold_m=float(data_config.get("building_threshold_m", 2.0)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model, _ = load_predictor(args.checkpoint, device=device)
    if args.fusion_mode_override:
        model.fusion_mode = args.fusion_mode_override
    if args.building_protection_power is not None:
        if args.building_protection_power <= 0:
            raise ValueError("building-protection-power must be positive")
        model.building_protection_power = args.building_protection_power
    model.eval()

    regression = {
        name: StreamingRegressionMetrics()
        for name in ("base_height", "height", "gated_height", "building_height", "canopy_height")
    }
    domain_regression: dict[str, dict[str, StreamingRegressionMetrics]] = {
        output: {name: StreamingRegressionMetrics() for name in LANDSCAPE_CLASSES}
        for output in ("base_height", "height", "gated_height")
    }
    landscape_regression: dict[str, dict[str, StreamingRegressionMetrics]] = {
        output: defaultdict(StreamingRegressionMetrics)
        for output in ("base_height", "height")
    }
    strength_sum: dict[str, float] = defaultdict(float)
    effective_strength_sum: dict[str, float] = defaultdict(float)
    protection_sum: dict[str, float] = defaultdict(float)
    strength_count: dict[str, int] = defaultdict(int)
    confusion = np.zeros((len(LANDSCAPE_CLASSES), len(LANDSCAPE_CLASSES)), dtype=np.int64)
    protected_building = BinarySums()

    with torch.inference_mode():
        for batch in tqdm(loader, desc="checkpoint analysis"):
            image = batch["image"].to(device, non_blocking=True)
            prior = batch["relative_prior"].to(device, non_blocking=True)
            output = model(image, prior)
            target = batch["height"].numpy()
            valid = batch["regression_mask"].numpy()
            domain = batch["domain_target"].numpy()
            domain_valid = batch["domain_valid_mask"].numpy()

            arrays = {
                name: output[name].float().cpu().numpy()
                for name in regression
            }
            for name, metric in regression.items():
                mask = valid
                if name == "building_height":
                    mask = valid & (domain[:, None] == LANDSCAPE_CLASSES["building"])
                elif name == "canopy_height":
                    mask = valid & (domain[:, None] == LANDSCAPE_CLASSES["vegetation"])
                metric.update(arrays[name], target, mask)

            for index, landscape in enumerate(batch["landscape"]):
                for output_name in landscape_regression:
                    landscape_regression[output_name][str(landscape)].update(
                        arrays[output_name][index : index + 1],
                        target[index : index + 1],
                        valid[index : index + 1],
                    )

            strength = output["refinement_strength"].float().cpu().numpy()
            effective_strength = output.get(
                "effective_refinement_strength", output["refinement_strength"]
            ).float().cpu().numpy()
            protection = output.get(
                "building_protection", torch.zeros_like(output["refinement_strength"])
            ).float().cpu().numpy()
            for domain_name, code in LANDSCAPE_CLASSES.items():
                mask = valid & (domain[:, None] == code)
                for output_name in domain_regression:
                    domain_regression[output_name][domain_name].update(
                        arrays[output_name], target, mask
                    )
                strength_sum[domain_name] += float(strength[mask].sum())
                effective_strength_sum[domain_name] += float(
                    effective_strength[mask].sum()
                )
                protection_sum[domain_name] += float(protection[mask].sum())
                strength_count[domain_name] += int(mask.sum())

            prediction = output["domain_logits"].argmax(dim=1).cpu().numpy()
            protected_building.update(
                output["protected_building_logits"].sigmoid().float().cpu().numpy()[:, 0],
                domain == LANDSCAPE_CLASSES["building"],
                domain_valid,
                threshold=0.5,
            )
            truth = domain
            semantic_valid = domain_valid
            for true_code in LANDSCAPE_CLASSES.values():
                true_mask = semantic_valid & (truth == true_code)
                for predicted_code in LANDSCAPE_CLASSES.values():
                    confusion[true_code, predicted_code] += int(
                        np.count_nonzero(true_mask & (prediction == predicted_code))
                    )

    expert_metrics = {name: metric.compute() for name, metric in regression.items()}
    per_domain = {
        output_name: {
            domain_name: metric.compute()
            for domain_name, metric in metrics.items()
            if metric.count
        }
        for output_name, metrics in domain_regression.items()
    }
    per_landscape = {
        output_name: {
            landscape: metric.compute()
            for landscape, metric in sorted(metrics.items())
            if metric.count
        }
        for output_name, metrics in landscape_regression.items()
    }
    semantic = {}
    for name, code in LANDSCAPE_CLASSES.items():
        true_positive = int(confusion[code, code])
        predicted = int(confusion[:, code].sum())
        actual = int(confusion[code, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        semantic[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }

    evaluation_config = config.get("evaluation", {})
    urban_delta = None
    if "urban" in per_landscape["height"] and "urban" in per_landscape["base_height"]:
        urban_delta = float(
            per_landscape["height"]["urban"]["rmse_m"]
            - per_landscape["base_height"]["urban"]["rmse_m"]
        )
    building_delta = None
    if (
        "building" in per_domain["height"]
        and "building" in per_domain["base_height"]
    ):
        building_delta = float(
            per_domain["height"]["building"]["rmse_m"]
            - per_domain["base_height"]["building"]["rmse_m"]
        )
    landscape_rmse = [
        float(metrics["rmse_m"]) for metrics in per_landscape["height"].values()
    ]
    maximum_urban_regression = float(
        evaluation_config.get("max_urban_rmse_regression_m", 0.25)
    )
    maximum_building_regression = float(
        evaluation_config.get("max_building_rmse_regression_m", 0.25)
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(manifest.resolve()),
        "evaluated_fusion_mode": model.fusion_mode,
        "evaluated_building_protection_power": model.building_protection_power,
        "expert_metrics": expert_metrics,
        "per_domain_metrics": per_domain,
        "per_landscape_metrics": per_landscape,
        "landscape_macro_rmse_m": (
            float(np.mean(landscape_rmse)) if landscape_rmse else None
        ),
        "urban_rmse_regression_m": urban_delta,
        "building_rmse_regression_m": building_delta,
        "passes_regression_guards": (
            (urban_delta is None or urban_delta <= maximum_urban_regression)
            and (
                building_delta is None
                or building_delta <= maximum_building_regression
            )
        ),
        "mean_refinement_strength": {
            name: strength_sum[name] / max(strength_count[name], 1)
            for name in LANDSCAPE_CLASSES
        },
        "mean_effective_refinement_strength": {
            name: effective_strength_sum[name] / max(strength_count[name], 1)
            for name in LANDSCAPE_CLASSES
        },
        "mean_building_protection": {
            name: protection_sum[name] / max(strength_count[name], 1)
            for name in LANDSCAPE_CLASSES
        },
        "semantic_metrics": semantic,
        "protected_building_metrics": protected_building.metrics(),
        "semantic_confusion": confusion.tolist(),
    }
    serialized = json.dumps(result, indent=2)
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(serialized + "\n", encoding="utf-8")
        temporary.replace(output_path)
    print(serialized)


if __name__ == "__main__":
    main()
