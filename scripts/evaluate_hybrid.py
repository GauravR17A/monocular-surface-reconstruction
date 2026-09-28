"""Validate a generalist/building-specialist blend without target leakage."""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm

from msr.data.webdataset_dataset import make_webdataset
from msr.evaluation.error_analysis import RegressionSums
from msr.inference.hybrid import blend_height_predictions
from msr.inference.predict import load_predictor


THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)


def _loader(config: dict, split: str, workers: int):
    data = config["data"]
    shard_pattern = data.get(f"{split}_shards")
    if not shard_pattern:
        validation_pattern = Path(data["validation_shards"])
        shard_pattern = str(validation_pattern.parent.parent / split / "*.tar")
    shards = sorted(glob.glob(str(shard_pattern)))
    dataset = make_webdataset(
        shards,
        patch_size=int(data.get("validation_patch_size", data["patch_size"])),
        training=False,
        height_min_m=float(data["height_min_m"]),
        height_max_m=float(data["height_max_m"]),
        seed=int(config["experiment"]["seed"]),
    )
    return wds.WebLoader(
        dataset,
        batch_size=1,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _strategy_names(locked_strategy: str | None = None) -> list[str]:
    names = ["baseline", "specialist"]
    for threshold in THRESHOLDS:
        names.extend((f"hard_{threshold:.2f}", f"soft_{threshold:.2f}"))
    if locked_strategy is None:
        return names
    if locked_strategy not in names:
        raise ValueError(f"Unknown locked strategy: {locked_strategy}")
    return list(dict.fromkeys(("baseline", "specialist", locked_strategy)))


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    baseline, baseline_metadata = load_predictor(args.baseline, device=args.device)
    specialist, specialist_metadata = load_predictor(args.specialist, device=args.device)
    checkpoint = torch.load(args.baseline, map_location="cpu", weights_only=False)
    loader = _loader(checkpoint["config"], args.split, args.workers)
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    names = _strategy_names(args.locked_strategy)
    overall = {name: RegressionSums() for name in names}
    buildings = {name: RegressionSums() for name in names}
    ground = {name: RegressionSums() for name in names}
    regions = {name: defaultdict(RegressionSums) for name in names}
    samples = 0

    for batch in tqdm(loader, desc=f"hybrid {args.split}"):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["height"][0, 0].numpy()
        valid = batch["valid_mask"][0, 0].numpy().astype(bool)
        building_mask = valid & (target >= 2.0)
        ground_mask = valid & ~building_mask
        region = str(batch["region"][0])
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            baseline_output = baseline(image)
            specialist_output = specialist(image)
        baseline_height = baseline_output["height"][0, 0].float().cpu().numpy()
        specialist_height = specialist_output["height"][0, 0].float().cpu().numpy()
        probability = baseline_output["building_logits"][0, 0].sigmoid().float().cpu().numpy()
        predictions = {
            "baseline": baseline_height,
            "specialist": specialist_height,
        }
        for name in names:
            if name in predictions:
                continue
            kind, threshold_text = name.split("_", maxsplit=1)
            predictions[name] = blend_height_predictions(
                baseline_height,
                specialist_height,
                probability,
                threshold=float(threshold_text),
                transition=0.0 if kind == "hard" else 0.2,
            )
        for name, prediction in predictions.items():
            overall[name].update(prediction, target, valid)
            buildings[name].update(prediction, target, building_mask)
            ground[name].update(prediction, target, ground_mask)
            regions[name][region].update(prediction, target, valid)
        samples += 1
        if args.max_samples and samples >= args.max_samples:
            break

    results = {}
    for name in names:
        region_metrics = {
            region: stats.metrics() for region, stats in sorted(regions[name].items())
        }
        results[name] = {
            "all": overall[name].metrics(),
            "building": buildings[name].metrics(),
            "ground_below_2m": ground[name].metrics(),
            "region_macro_rmse_m": float(
                np.mean([metrics["rmse_m"] for metrics in region_metrics.values()])
            ),
            "worst_region_rmse_m": float(
                max(metrics["rmse_m"] for metrics in region_metrics.values())
            ),
            "regions": region_metrics,
        }
    baseline_all = results["baseline"]["all"]["rmse_m"]
    baseline_building = results["baseline"]["building"]["rmse_m"]
    if args.locked_strategy is None:
        safe_improvements = [
            name
            for name in names
            if name not in {"baseline", "specialist"}
            and results[name]["all"]["rmse_m"] < baseline_all
            and results[name]["building"]["rmse_m"] < baseline_building
        ]
        recommended = min(
            safe_improvements,
            key=lambda name: (
                results[name]["all"]["rmse_m"] + results[name]["building"]["rmse_m"]
            ),
            default=None,
        )
    else:
        recommended = None
    report = {
        "split": args.split,
        "sample_count": samples,
        "limited_run": bool(args.max_samples),
        "locked_strategy": args.locked_strategy,
        "baseline": baseline_metadata,
        "specialist": specialist_metadata,
        "recommended_if_both_rmse_improve": recommended,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "sample_count": samples,
        "recommended": recommended,
        "locked_strategy": args.locked_strategy,
        "baseline": results["baseline"],
        "specialist": results["specialist"],
        "selected_metrics": results.get(args.locked_strategy or recommended),
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--locked-strategy",
        choices=_strategy_names(),
        help="Evaluate a validation-selected strategy without selecting again on the test split.",
    )
    args = parser.parse_args()
    if args.split == "test" and args.locked_strategy is None:
        parser.error("--locked-strategy is required on the test split to prevent test-set tuning")
    run(args)


if __name__ == "__main__":
    main()
