"""Run validation-only failure analysis for a trained HeightNet checkpoint."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm

from msr.data.raster_dataset import IMAGENET_MEAN, IMAGENET_STD
from msr.data.webdataset_dataset import make_webdataset
from msr.evaluation.error_analysis import BinarySums, RegressionSums
from msr.inference.predict import load_predictor


HEIGHT_BINS = (
    (0.0, 2.0, "00_ground_0_2m"),
    (2.0, 5.0, "01_low_2_5m"),
    (5.0, 10.0, "02_5_10m"),
    (10.0, 20.0, "03_10_20m"),
    (20.0, 40.0, "04_20_40m"),
    (40.0, 80.0, "05_40_80m"),
    (80.0, math.inf, "06_80m_plus"),
)
BUILDING_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def load_manifest_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {row["webdataset_key"]: row for row in rows}


def sample_fold(sample_id: str) -> int:
    """Stable two-way split used only for cross-fitted calibration diagnostics."""

    return hashlib.sha1(sample_id.encode("utf-8")).digest()[0] & 1


def describe(stats: RegressionSums) -> dict[str, float]:
    count = float(stats.count)
    prediction_variance = max(
        stats.sum_prediction_squared / count - (stats.sum_prediction / count) ** 2,
        0.0,
    )
    target_variance = max(
        stats.sum_target_squared / count - (stats.sum_target / count) ** 2,
        0.0,
    )
    return {
        "prediction_mean_m": stats.sum_prediction / count,
        "target_mean_m": stats.sum_target / count,
        "prediction_std_m": math.sqrt(prediction_variance),
        "target_std_m": math.sqrt(target_variance),
    }


def point_metrics(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | None]:
    valid = mask & np.isfinite(prediction) & np.isfinite(target)
    if not np.any(valid):
        return {"pixel_count": 0, "rmse_m": None, "mae_m": None, "bias_m": None}
    error = prediction[valid].astype(np.float64) - target[valid].astype(np.float64)
    return {
        "pixel_count": int(error.size),
        "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "mae_m": float(np.mean(np.abs(error))),
        "bias_m": float(np.mean(error)),
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_worst_cases(output_dir: Path, entries: list[tuple]) -> None:
    visual_dir = output_dir / "worst_cases"
    visual_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, _, payload) in enumerate(sorted(entries, reverse=True), start=1):
        target = payload["target"].astype(np.float32)
        prediction = payload["prediction"].astype(np.float32)
        probability = payload["building_probability"].astype(np.float32)
        valid = payload["valid"]
        error = np.abs(prediction - target)
        valid_target = target[valid]
        height_vmax = float(np.percentile(valid_target, 99)) if valid_target.size else 10.0
        height_vmax = float(np.clip(height_vmax, 10.0, 80.0))
        valid_error = error[valid]
        error_vmax = float(np.percentile(valid_error, 95)) if valid_error.size else 5.0
        error_vmax = float(np.clip(error_vmax, 5.0, 30.0))

        figure, axes = plt.subplots(1, 5, figsize=(20, 4.3), constrained_layout=True)
        axes[0].imshow(payload["rgb"])
        axes[0].set_title("RGB")
        target_image = axes[1].imshow(target, cmap="viridis", vmin=0.0, vmax=height_vmax)
        axes[1].set_title("Reference height (m)")
        axes[2].imshow(prediction, cmap="viridis", vmin=0.0, vmax=height_vmax)
        axes[2].set_title("Predicted height (m)")
        error_image = axes[3].imshow(error, cmap="magma", vmin=0.0, vmax=error_vmax)
        axes[3].set_title("Absolute error (m)")
        axes[4].imshow(probability, cmap="gray", vmin=0.0, vmax=1.0)
        axes[4].set_title("Predicted building probability")
        for axis in axes:
            axis.axis("off")
        figure.colorbar(target_image, ax=axes[1:3], fraction=0.025, pad=0.02)
        figure.colorbar(error_image, ax=axes[3], fraction=0.045, pad=0.02)
        figure.suptitle(
            f"#{rank} {payload['city']} | {payload['sample_id']} | "
            f"building RMSE {payload['score']:.2f} m",
            fontsize=10,
        )
        figure.savefig(
            visual_dir / f"{rank:02d}_{safe_name(payload['sample_id'])}.png",
            dpi=130,
        )
        plt.close(figure)


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def cross_fitted_affine(folds: list[RegressionSums]) -> dict:
    if len(folds) != 2 or any(fold.count == 0 for fold in folds):
        raise ValueError("Two non-empty folds are required")
    scale_0, offset_0 = folds[0].affine_fit()
    scale_1, offset_1 = folds[1].affine_fit()
    # Each half is evaluated using parameters learned from the opposite half.
    evaluated_0 = folds[0].transformed(scale_1, offset_1)
    evaluated_1 = folds[1].transformed(scale_0, offset_0)
    combined = RegressionSums().merge(evaluated_0).merge(evaluated_1)
    return {
        "fold_0_fit": {"scale": scale_0, "offset_m": offset_0},
        "fold_1_fit": {"scale": scale_1, "offset_m": offset_1},
        "cross_fitted_metrics": combined.metrics(),
        "note": "Diagnostic only: each validation half used calibration fitted on the other half.",
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    data_config = config["data"]
    seed = int(config["experiment"]["seed"])
    validation_manifest = Path(data_config["val_manifest"])
    metadata = load_manifest_metadata(validation_manifest)
    shards = sorted(glob.glob(str(data_config["validation_shards"])))
    dataset = make_webdataset(
        shards,
        patch_size=int(data_config.get("validation_patch_size", data_config["patch_size"])),
        training=False,
        height_min_m=float(data_config["height_min_m"]),
        height_max_m=float(data_config["height_max_m"]),
        seed=seed,
    )
    loader = wds.WebLoader(
        dataset,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    model, model_metadata = load_predictor(checkpoint_path, device=args.device)
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    overall = RegressionSums()
    building = RegressionSums()
    ground = RegressionSums()
    cities: dict[str, RegressionSums] = defaultdict(RegressionSums)
    city_buildings: dict[str, RegressionSums] = defaultdict(RegressionSums)
    height_bins = {label: RegressionSums() for _, _, label in HEIGHT_BINS}
    raw_folds = [RegressionSums(), RegressionSums()]
    building_folds = [RegressionSums(), RegressionSums()]
    classification = {threshold: BinarySums() for threshold in BUILDING_THRESHOLDS}
    gated_all = {f"hard_{threshold:.1f}": RegressionSums() for threshold in BUILDING_THRESHOLDS}
    gated_building = {
        f"hard_{threshold:.1f}": RegressionSums() for threshold in BUILDING_THRESHOLDS
    }
    gated_ground = {
        f"hard_{threshold:.1f}": RegressionSums() for threshold in BUILDING_THRESHOLDS
    }
    gated_all["soft"] = RegressionSums()
    gated_building["soft"] = RegressionSums()
    gated_ground["soft"] = RegressionSums()
    city_tile_counts: Counter[str] = Counter()
    sample_rows: list[dict] = []
    worst_heap: list[tuple] = []
    heap_counter = 0

    for batch in tqdm(loader, desc="validation error analysis"):
        sample_id = str(batch["sample_id"][0])
        row = metadata.get(sample_id, {})
        city = row.get("city") or str(batch["region"][0])
        city_tile_counts[city] += 1
        image = batch["image"].to(device, non_blocking=True)
        target_tensor = batch["height"].to(device, non_blocking=True)
        valid_tensor = batch["valid_mask"].to(device, non_blocking=True).bool()
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            output = model(image)
        prediction = output["height"][0, 0].float().cpu().numpy()
        target = target_tensor[0, 0].float().cpu().numpy()
        valid = valid_tensor[0, 0].cpu().numpy().astype(bool)
        building_mask = valid & (target >= 2.0)
        ground_mask = valid & ~building_mask
        probability = torch.sigmoid(output["building_logits"])[0, 0].float().cpu().numpy()

        overall.update(prediction, target, valid)
        building.update(prediction, target, building_mask)
        ground.update(prediction, target, ground_mask)
        cities[city].update(prediction, target, valid)
        city_buildings[city].update(prediction, target, building_mask)
        fold = sample_fold(sample_id)
        raw_folds[fold].update(prediction, target, valid)
        building_folds[fold].update(prediction, target, building_mask)
        for low, high, label in HEIGHT_BINS:
            height_bins[label].update(
                prediction,
                target,
                valid & (target >= low) & (target < high),
            )
        for threshold, stats in classification.items():
            stats.update(probability, building_mask, valid, threshold=threshold)
            gated_prediction = prediction * (probability >= threshold)
            name = f"hard_{threshold:.1f}"
            gated_all[name].update(gated_prediction, target, valid)
            gated_building[name].update(gated_prediction, target, building_mask)
            gated_ground[name].update(gated_prediction, target, ground_mask)
        soft_prediction = prediction * probability
        gated_all["soft"].update(soft_prediction, target, valid)
        gated_building["soft"].update(soft_prediction, target, building_mask)
        gated_ground["soft"].update(soft_prediction, target, ground_mask)

        sample_all = point_metrics(prediction, target, valid)
        sample_building = point_metrics(prediction, target, building_mask)
        sample_rows.append(
            {
                "sample_id": sample_id,
                "continent": row.get("continent", ""),
                "country": row.get("country", ""),
                "city": city,
                **{f"all_{key}": value for key, value in sample_all.items()},
                **{f"building_{key}": value for key, value in sample_building.items()},
                "reference_max_height_m": float(target[valid].max()) if np.any(valid) else None,
                "building_fraction": float(building_mask.sum() / max(valid.sum(), 1)),
            }
        )

        score_value = sample_building["rmse_m"]
        if score_value is not None and args.max_visuals > 0:
            denormalized = (
                image[0].float().cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN
            )
            rgb = np.clip(np.moveaxis(denormalized, 0, -1) * 255.0, 0, 255).astype(np.uint8)
            payload = {
                "sample_id": sample_id,
                "city": city,
                "score": float(score_value),
                "rgb": rgb,
                "target": target.astype(np.float16),
                "prediction": prediction.astype(np.float16),
                "building_probability": probability.astype(np.float16),
                "valid": valid,
            }
            entry = (float(score_value), heap_counter, payload)
            heap_counter += 1
            if len(worst_heap) < args.max_visuals:
                heapq.heappush(worst_heap, entry)
            elif entry[0] > worst_heap[0][0]:
                heapq.heapreplace(worst_heap, entry)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rows.sort(
        key=lambda item: item["building_rmse_m"] if item["building_rmse_m"] is not None else -1,
        reverse=True,
    )
    write_csv(output_dir / "per_sample.csv", sample_rows)

    city_rows = []
    for city in sorted(cities):
        city_rows.append(
            {
                "city": city,
                "tile_count": city_tile_counts[city],
                **{f"all_{key}": value for key, value in cities[city].metrics().items()},
                **{
                    f"building_{key}": value
                    for key, value in city_buildings[city].metrics().items()
                },
            }
        )
    write_csv(output_dir / "per_city.csv", city_rows)

    bin_rows = []
    for low, high, label in HEIGHT_BINS:
        stats = height_bins[label]
        if stats.count:
            bin_rows.append(
                {
                    "height_bin": label,
                    "minimum_m": low,
                    "maximum_m": None if math.isinf(high) else high,
                    **stats.metrics(),
                    **describe(stats),
                }
            )
    write_csv(output_dir / "height_bins.csv", bin_rows)

    gating_rows = []
    for name in gated_all:
        gate_metrics = gated_all[name].metrics()
        gating_rows.append(
            {
                "gating": name,
                **{f"all_{key}": value for key, value in gate_metrics.items()},
                **{
                    f"building_{key}": value
                    for key, value in gated_building[name].metrics().items()
                },
                **{
                    f"ground_{key}": value
                    for key, value in gated_ground[name].metrics().items()
                },
            }
        )
    write_csv(output_dir / "gating.csv", gating_rows)

    full_scale, full_offset = overall.affine_fit()
    building_scale, building_offset = building.affine_fit()
    summary = {
        "checkpoint": model_metadata,
        "validation_manifest": str(validation_manifest.resolve()),
        "tile_count": sum(city_tile_counts.values()),
        "city_tile_counts": dict(sorted(city_tile_counts.items())),
        "raw": {
            "all": {**overall.metrics(), **describe(overall)},
            "building": {**building.metrics(), **describe(building)},
            "ground_below_2m": {**ground.metrics(), **describe(ground)},
        },
        "height_bins": {label: stats.metrics() for label, stats in height_bins.items() if stats.count},
        "building_classification": {
            f"threshold_{threshold:.1f}": stats.metrics()
            for threshold, stats in classification.items()
        },
        "gating": {
            name: {
                "all": gated_all[name].metrics(),
                "building": gated_building[name].metrics(),
                "ground_below_2m": gated_ground[name].metrics(),
            }
            for name in gated_all
        },
        "affine_calibration": {
            "full_validation_optimistic_diagnostic": {
                "scale": full_scale,
                "offset_m": full_offset,
                "metrics": overall.transformed(full_scale, full_offset).metrics(),
                "note": "Not an honest validation score because fit and evaluation use the same pixels.",
            },
            "two_fold_cross_fitted_all": cross_fitted_affine(raw_folds),
            "building_only_optimistic_diagnostic": {
                "scale": building_scale,
                "offset_m": building_offset,
                "metrics": building.transformed(building_scale, building_offset).metrics(),
            },
            "two_fold_cross_fitted_building": cross_fitted_affine(building_folds),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(clean_json(summary), indent=2) + "\n", encoding="utf-8"
    )
    render_worst_cases(output_dir, worst_heap)
    print(json.dumps(clean_json(summary["raw"]), indent=2))
    print(f"Saved validation analysis to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-visuals", type=int, default=12)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
