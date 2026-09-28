"""Create a versioned checkpoint with validated conservative canopy fusion."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or not 0 <= args.strength <= 1:
        raise ValueError("threshold and strength must be within [0, 1]")

    source = args.source.resolve()
    payload = deepcopy(torch.load(source, map_location="cpu", weights_only=False))
    model_config = payload["config"]["model"]
    model_config["vegetation_expert_fusion_threshold"] = float(args.threshold)
    model_config["vegetation_expert_fusion_strength"] = float(args.strength)
    validation_report = json.loads(args.validation_report.read_text(encoding="utf-8"))
    recommended = validation_report["recommended"]
    candidate = validation_report["candidates"][recommended]
    metrics = payload.setdefault("metrics", {})
    metrics.update(candidate["overall"])
    metrics["domains"] = candidate["domains"]
    metrics["landscapes"] = candidate["landscapes"]
    domain_rmse = {
        name: values["rmse_m"] for name, values in candidate["domains"].items()
    }
    landscape_rmse = {
        name: values["rmse_m"] for name, values in candidate["landscapes"].items()
    }
    metrics["domain_macro_rmse_m"] = sum(domain_rmse.values()) / len(domain_rmse)
    metrics["worst_domain"] = max(domain_rmse, key=domain_rmse.get)
    metrics["worst_domain_rmse_m"] = domain_rmse[metrics["worst_domain"]]
    metrics["landscape_macro_rmse_m"] = sum(landscape_rmse.values()) / len(
        landscape_rmse
    )
    metrics["worst_landscape"] = max(landscape_rmse, key=landscape_rmse.get)
    metrics["worst_landscape_rmse_m"] = landscape_rmse[metrics["worst_landscape"]]
    metrics["postprocessing_validation"] = {
        "candidate": recommended,
        "regressions_m": candidate["regressions_m"],
        "eligible": candidate["eligible"],
    }
    previous_calibration = payload.get("calibration")
    payload["calibration"] = {
        "source_checkpoint": str(source),
        "method": "high_confidence_canopy_expert_fusion",
        "threshold": float(args.threshold),
        "strength": float(args.strength),
        "selected_on": "validation_only",
        "validation_report": str(args.validation_report.resolve()),
        "confirmed_on": "untouched_test_once",
        "test_report": str(args.test_report.resolve()),
        "previous_calibration": previous_calibration,
    }

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "checkpoint_best_guarded_vegetation.pt"
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
