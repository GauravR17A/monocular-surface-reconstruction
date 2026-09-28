"""Sweep guarded building-expert fusion on validation data without retraining."""

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


BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)
GATES = ("intersection", "minimum", "protection", "consensus")
STRICT_THRESHOLDS = (0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99)
TALL_BASE_THRESHOLDS_M = (8.0, 10.0, 12.0, 15.0, 20.0)
TALL_PRODUCT_THRESHOLDS = (0.30, 0.50, 0.70)


def _candidate_specs(sweep_mode: str) -> list[tuple[str, str, float]]:
    specs = [("protected", "none", 0.0)]
    if sweep_mode == "high_precision":
        for threshold in (0.85, 0.875, 0.90, 0.925, 0.95, 0.975):
            gate = f"product_{threshold:.3f}"
            for blend in BLENDS[1:]:
                specs.append((f"{gate}_b{blend:.2f}", gate, blend))
        return specs
    if sweep_mode == "tall":
        for height_m in TALL_BASE_THRESHOLDS_M:
            for product_threshold in TALL_PRODUCT_THRESHOLDS:
                for positive_only in (False, True):
                    prefix = "tallpositive" if positive_only else "tallproduct"
                    gate = f"{prefix}_{height_m:.1f}_{product_threshold:.2f}"
                    for blend in (0.5, 1.0):
                        specs.append((f"{gate}_b{blend:.2f}", gate, blend))
        return specs
    for gate in GATES:
        for blend in BLENDS[1:]:
            specs.append((f"{gate}_b{blend:.2f}", gate, blend))
    for threshold in STRICT_THRESHOLDS:
        for gate_prefix in ("strict", "product"):
            gate = f"{gate_prefix}_{threshold:.3f}"
            for blend in BLENDS[1:]:
                specs.append((f"{gate}_b{blend:.2f}", gate, blend))
    return specs


def _gate(
    gate_name: str,
    *,
    protected: np.ndarray,
    learned: np.ndarray,
    vegetation: np.ndarray,
    building_protection: np.ndarray,
    base_height: np.ndarray,
    residual: np.ndarray,
) -> np.ndarray:
    non_vegetation = np.clip(1.0 - vegetation, 0.0, 1.0)
    if gate_name == "intersection":
        return protected * learned * non_vegetation
    if gate_name == "minimum":
        return np.minimum(protected, learned) * non_vegetation
    if gate_name == "protection":
        return building_protection * learned
    if gate_name == "consensus":
        return (
            (protected >= 0.5)
            & (learned >= 0.5)
            & (vegetation < 0.5)
        ).astype(np.float32)
    if gate_name.startswith("strict_"):
        threshold = float(gate_name.split("_", maxsplit=1)[1])
        return (
            (protected >= threshold)
            & (learned >= threshold)
            & (vegetation <= 1.0 - threshold)
        ).astype(np.float32)
    if gate_name.startswith("product_"):
        threshold = float(gate_name.split("_", maxsplit=1)[1])
        score = protected * learned * non_vegetation
        return (score >= threshold).astype(np.float32)
    if gate_name.startswith(("tallproduct_", "tallpositive_")):
        _, height_text, threshold_text = gate_name.split("_")
        score = protected * learned * non_vegetation
        selected = (base_height >= float(height_text)) & (
            score >= float(threshold_text)
        )
        if gate_name.startswith("tallpositive_"):
            selected &= residual > 0.0
        return selected.astype(np.float32)
    if gate_name == "none":
        return np.zeros_like(protected)
    raise ValueError(f"Unknown gate: {gate_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-guard-regression-m", type=float, default=0.05)
    parser.add_argument(
        "--sweep-mode",
        choices=("broad", "tall", "high_precision"),
        default="broad",
    )
    args = parser.parse_args()

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

    specs = _candidate_specs(args.sweep_mode)
    overall = {name: StreamingRegressionMetrics() for name, _, _ in specs}
    domains = {
        name: {domain: StreamingRegressionMetrics() for domain in LANDSCAPE_CLASSES}
        for name, _, _ in specs
    }
    landscapes = {
        name: defaultdict(StreamingRegressionMetrics) for name, _, _ in specs
    }

    with torch.inference_mode():
        for batch in tqdm(loader, desc="building fusion sweep"):
            image = batch["image"].to(device, non_blocking=True)
            prior = batch["relative_prior"].to(device, non_blocking=True)
            output = model(image, prior)

            target = batch["height"].numpy()
            valid = batch["regression_mask"].numpy()
            domain = batch["domain_target"].numpy()
            current = output["height"].float().cpu().numpy()
            residual = (
                output["building_height"] - output["base_height"]
            ).float().cpu().numpy()
            base_height = output["base_height"].float().cpu().numpy()
            protected = output["protected_building_logits"].sigmoid().float().cpu().numpy()
            probabilities = output["domain_probabilities"].float().cpu().numpy()
            learned = probabilities[:, 1:2]
            vegetation = probabilities[:, 2:3]
            building_protection = output["building_protection"].float().cpu().numpy()

            gates = {
                gate_name: _gate(
                    gate_name,
                    protected=protected,
                    learned=learned,
                    vegetation=vegetation,
                    building_protection=building_protection,
                    base_height=base_height,
                    residual=residual,
                )
                for gate_name in {gate_name for _, gate_name, _ in specs}
            }

            for name, gate_name, blend in specs:
                prediction = np.maximum(current + blend * gates[gate_name] * residual, 0.0)
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
    for name, gate_name, blend in specs:
        results[name] = {
            "gate": gate_name,
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
    for name, result in results.items():
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
        if result["eligible"] and result["building_improvement_m"] > 0
    ]
    recommended = min(
        eligible,
        key=lambda item: item[1]["domains"]["building"]["rmse_m"],
        default=("protected", results["protected"]),
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
