"""Diagnose building-height and routing errors on a held-out manifest."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from msr.data.surface_dataset import MultiDomainSurfaceDataset, load_surface_manifest
from msr.evaluation.metrics import StreamingRegressionMetrics
from msr.inference.predict import load_predictor


HEIGHT_BINS = (
    ("2_to_5m", 2.0, 5.0),
    ("5_to_10m", 5.0, 10.0),
    ("10_to_20m", 10.0, 20.0),
    ("20_to_40m", 20.0, 40.0),
    ("40m_plus", 40.0, float("inf")),
)
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)


def _metric_group() -> dict[str, StreamingRegressionMetrics]:
    return {
        "protected_final": StreamingRegressionMetrics(),
        "protected_base": StreamingRegressionMetrics(),
        "building_specialist": StreamingRegressionMetrics(),
    }


def _computed(group: dict[str, StreamingRegressionMetrics]) -> dict[str, dict]:
    return {name: metric.compute() for name, metric in group.items() if metric.count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
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

    all_buildings = _metric_group()
    bins = {name: _metric_group() for name, _, _ in HEIGHT_BINS}
    regions: dict[str, dict[str, StreamingRegressionMetrics]] = defaultdict(_metric_group)
    confusion = {
        detector: {
            threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for threshold in THRESHOLDS
        }
        for detector in ("protected", "learned", "product")
    }

    with torch.inference_mode():
        for batch in tqdm(loader, desc="building error analysis"):
            output = model(
                batch["image"].to(device, non_blocking=True),
                batch["relative_prior"].to(device, non_blocking=True),
            )
            target = batch["height"].numpy()
            valid = batch["regression_mask"].numpy().astype(bool)
            domain = batch["domain_target"].numpy()
            building = valid & (domain[:, None] == 1)
            predictions = {
                "protected_final": output["height"].float().cpu().numpy(),
                "protected_base": output["base_height"].float().cpu().numpy(),
                "building_specialist": output["building_height"].float().cpu().numpy(),
            }
            for name, prediction in predictions.items():
                all_buildings[name].update(prediction, target, building)
                regions[str(batch["region"][0])][name].update(prediction, target, building)
                for bin_name, lower, upper in HEIGHT_BINS:
                    bins[bin_name][name].update(
                        prediction,
                        target,
                        building & (target >= lower) & (target < upper),
                    )

            protected = output["protected_building_logits"].sigmoid().float().cpu().numpy()
            learned = output["domain_probabilities"][:, 1:2].float().cpu().numpy()
            vegetation = output["domain_probabilities"][:, 2:3].float().cpu().numpy()
            scores = {
                "protected": protected,
                "learned": learned,
                "product": protected * learned * (1.0 - vegetation),
            }
            semantic_valid = batch["domain_valid_mask"].numpy()[:, None].astype(bool)
            truth = domain[:, None] == 1
            for detector, score in scores.items():
                for threshold in THRESHOLDS:
                    predicted = score >= threshold
                    counts = confusion[detector][threshold]
                    counts["tp"] += int(np.count_nonzero(semantic_valid & truth & predicted))
                    counts["fp"] += int(np.count_nonzero(semantic_valid & ~truth & predicted))
                    counts["fn"] += int(np.count_nonzero(semantic_valid & truth & ~predicted))
                    counts["tn"] += int(np.count_nonzero(semantic_valid & ~truth & ~predicted))

    detector_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for detector, thresholds in confusion.items():
        detector_metrics[detector] = {}
        for threshold, counts in thresholds.items():
            precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
            recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            detector_metrics[detector][f"{threshold:.3f}"] = {
                **counts,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "building_metrics": _computed(all_buildings),
        "height_bins": {
            name: _computed(group) for name, group in bins.items()
        },
        "regions": {
            name: _computed(group) for name, group in sorted(regions.items())
        },
        "detectors": detector_metrics,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(report["building_metrics"], indent=2))


if __name__ == "__main__":
    main()
