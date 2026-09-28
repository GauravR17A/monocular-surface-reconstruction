"""CPU-only DC/PHL TRAINING crop exposure audit and authenticated V2 index.

Existing sealed fine-class index is authenticated and summarized first. Only
approved training RGB/CLS files are decoded. No model or height inputs are used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msr.data.gamus_dataset import GAMUS_SIX_CLASS_NAMES
from msr.data.gamus_rgb_segmentation import (
    GamusRgbSegmentationDataset, build_train_sampler, canonical_sha256,
    file_sha256, source_metadata_snapshot, authenticated_json,
)
from msr.data.gamus_rgb_targeted_v2 import (
    TARGET_INDEX_SCHEMA, TARGET_NAMES, TARGET_CLASS_IDS, candidate_origins,
    read_rgb_class_window, training_tile_statistics, target_candidate_eligibility,
)


def make_training_dataset(config):
    data = config["data"]
    if data.get("train_radiometric_policy") != "raw" or int(data.get("patch_size")) != 384:
        raise ValueError("Expected sealed raw 384px training contract")
    return GamusRgbSegmentationDataset(data["root"], "train", patch_size=384, random_crop=True,
        approved_index_path=data["approved_index_path"], approved_index_sha256=data["approved_index_file_sha256"],
        rgb_scale=float(data.get("rgb_scale", 255)), dark_pixel_threshold=float(data.get("dark_pixel_threshold", .15)))


def existing_index_summary(config, dataset, weights):
    path = Path(config["data"]["fine_class_sampling_index_path"])
    if file_sha256(path) != config["data"]["fine_class_sampling_index_sha256"]:
        raise ValueError("Existing fine sampling index SHA mismatch")
    rows = {row["sample_id"]: row for row in [json.loads(line) for line in path.read_text().splitlines() if line.strip()]}
    result = {}
    for city in ("DC", "PHL"):
        selected = [(i, rows[r.sample_id]) for i, r in enumerate(dataset.records) if r.city == city]
        mass = sum(weights[i] for i, _ in selected)
        result[city] = {"training_tiles": len(selected), "tile_class_presence": {
            name: sum(row["class_counts"][name] > 0 for _, row in selected) for name in GAMUS_SIX_CLASS_NAMES},
            "ordinary_expected_class_pixels_class_valid_only": {
                name: sum(weights[i] * row["uniform_random_crop_384"]["expected_class_pixels"][name] for i, row in selected) / mass
                for name in GAMUS_SIX_CLASS_NAMES}}
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "by_city": result,
        "limitation": "Existing class-only index has no RGB validity or ground/low-vegetation hit/support probabilities."}


def expected_crop_exposure(tiles, weights, patch_size, min_target_pixels, target_probability=.5):
    """Analytic expectations over identical V1 image weights; no Monte Carlo."""
    area, result = patch_size ** 2, {}
    for city in ("DC", "PHL", "pooled"):
        selected = [(i, row) for i, row in enumerate(tiles) if city == "pooled" or row["city"] == city]
        if not selected:
            continue
        mass = sum(float(weights[i]) for i, _ in selected)
        values = {arm: {kind: np.zeros(6) for kind in ("pixels", "hit", "support")}
                  for arm in ("control", "targeted")}
        eligible_tiles = {name: 0 for name in TARGET_NAMES}
        target_branch_mass = 0.
        for i, tile in selected:
            weight = float(weights[i]) / mass
            ordinary = tile["ordinary_uniform_crop"]
            control = {"pixels": np.array([ordinary["expected_valid_class_pixels"][name] for name in GAMUS_SIX_CLASS_NAMES]),
                       "hit": np.array([ordinary["class_hit_probability"][name] for name in GAMUS_SIX_CLASS_NAMES]),
                       "support": np.array([ordinary["meaningful_support_probability"][name] for name in GAMUS_SIX_CLASS_NAMES])}
            candidates = np.asarray(tile["candidate_valid_class_counts"])
            sets = []
            for name in TARGET_NAMES:
                indices = tile["eligible_candidate_indices"][name]
                if len(indices):
                    eligible_tiles[name] += 1
                    sets.append(candidates[indices])
            if sets:
                target_branch_mass += weight * target_probability
                pure = {"pixels": np.mean([part.mean(0) for part in sets], axis=0),
                        "hit": np.mean([(part > 0).mean(0) for part in sets], axis=0),
                        "support": np.mean([(part >= min_target_pixels).mean(0) for part in sets], axis=0)}
                mixture = {kind: (1 - target_probability) * control[kind] + target_probability * pure[kind] for kind in control}
            else:
                mixture = control
            for kind in control:
                values["control"][kind] += weight * control[kind]
                values["targeted"][kind] += weight * mixture[kind]
        arms = {}
        for arm, data in values.items():
            arms[arm] = {"expected_valid_class_pixels": dict(zip(GAMUS_SIX_CLASS_NAMES, data["pixels"].tolist())),
                "expected_valid_class_fraction_of_crop": dict(zip(GAMUS_SIX_CLASS_NAMES, (data["pixels"] / area).tolist())),
                "class_hit_probability": dict(zip(GAMUS_SIX_CLASS_NAMES, data["hit"].tolist())),
                "meaningful_support_probability": dict(zip(GAMUS_SIX_CLASS_NAMES, data["support"].tolist()))}
        result[city] = {"training_tiles": len(selected), "image_sampler_mass": mass,
            "city_probability": mass / float(sum(weights)), "target_eligible_tiles": eligible_tiles,
            "effective_targeted_crop_probability": target_branch_mass, "arms": arms,
            "expected_valid_class_pixel_relative_change": {
                name: (values["targeted"]["pixels"][cid] / values["control"]["pixels"][cid] - 1)
                    if values["control"]["pixels"][cid] else None
                for cid, name in enumerate(GAMUS_SIX_CLASS_NAMES)},
            "joint_valid_class_pixels_in_all_training_tiles": {
                name: sum(row["joint_valid_class_pixels"][name] for _, row in selected) for name in GAMUS_SIX_CLASS_NAMES}}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gamus_rgb_segmenter_v1.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/data_audits/rgb_targeted_v2_training")
    parser.add_argument("--mode", choices=("summary", "benchmark", "build", "refine"), default="benchmark")
    parser.add_argument("--source-index", type=Path)
    parser.add_argument("--source-index-sha256")
    parser.add_argument("--benchmark-tiles", type=int, default=8)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-target-fraction", type=float, default=.01)
    parser.add_argument("--target-quantile", type=float, default=.75)
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_target_fraction) or not 0 < args.min_target_fraction <= 1:
        raise ValueError("Meaningful target fraction must lie in (0,1]")
    if not math.isfinite(args.target_quantile) or not 0 <= args.target_quantile <= 1:
        raise ValueError("Target quantile must lie in [0,1]")
    if args.benchmark_tiles < 1:
        raise ValueError("Benchmark tile count must be positive")
    if not 1 <= args.workers <= 4:
        raise ValueError("CPU audit workers must be between1 and4")
    torch.set_num_threads(1)
    config = yaml.safe_load(args.config.read_text())
    dataset = make_training_dataset(config)
    weights = build_train_sampler(dataset, config, torch.Generator().manual_seed(0)).weights.numpy()
    prior = existing_index_summary(config, dataset, weights)
    print(json.dumps({"existing_index": prior}, sort_keys=True), flush=True)
    if args.mode == "summary":
        return 0
    origins = candidate_origins(dataset.native_shape, dataset.patch_size, args.stride)
    minimum = math.ceil(dataset.patch_size ** 2 * args.min_target_fraction)
    snapshot = source_metadata_snapshot(dataset)
    if args.mode in ("build", "refine"):
        output = args.output_dir.resolve()
        if output.is_relative_to(dataset.root):
            raise ValueError("Audit outputs cannot be written into source data root")
        output.mkdir(parents=True, exist_ok=True)
        index_path, report_path = output / "targeted_crop_index_v2.json", output / "training_crop_exposure_report_v2.json"
        for path in (index_path, report_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite an existing audit artifact: {path}")
        records = dataset.records
    else:
        # Spread benchmark over both approved cities and the manifest order.
        positions = np.linspace(0, len(dataset) - 1, min(args.benchmark_tiles, len(dataset)), dtype=int)
        records = [dataset.records[int(i)] for i in positions]
    def scan(record):
        arrays = read_rgb_class_window(record, native_shape=dataset.native_shape, patch_size=dataset.native_shape,
            row=0, col=0, rgb_scale=dataset.rgb_scale, dark_pixel_threshold=dataset.dark_pixel_threshold)
        return {"sample_id": record.sample_id, "city": record.city,
            **training_tile_statistics(arrays, dataset.patch_size, origins, minimum, args.target_quantile)}
    started, tiles = time.monotonic(), []
    if args.mode == "refine":
        if not args.source_index or not args.source_index_sha256:
            raise ValueError("Refinement requires source index and exact SHA256")
        previous = authenticated_json(args.source_index, args.source_index_sha256)
        for key, value in {"schema": TARGET_INDEX_SCHEMA, "split": "train",
                "approved_index_sha256": dataset.approved_index_sha256,
                "train_ids_sha256": canonical_sha256(list(dataset.sample_ids)),
                "source_metadata": snapshot, "native_shape": dataset.native_shape,
                "patch_size": dataset.patch_size, "rgb_scale": dataset.rgb_scale,
                "min_target_pixels": minimum, "candidate_origins": origins,
                "class_names": list(GAMUS_SIX_CLASS_NAMES)}.items():
            if previous.get(key) != value:
                raise ValueError(f"Cached training index differs for {key}")
        tiles = previous["tiles"]
        if [row["sample_id"] for row in tiles] != list(dataset.sample_ids):
            raise ValueError("Cached index does not match ordered training IDs")
        for record, row in zip(dataset.records, tiles):
            if row["city"] != record.city:
                raise ValueError("Cached index city differs")
            counts = np.asarray(row["candidate_valid_class_counts"])
            if counts.shape != (len(origins), 6) or counts.dtype.kind not in "iu" or np.any(counts < 0) or np.any(counts > dataset.patch_size ** 2) or np.any(counts.sum(1) > dataset.patch_size ** 2):
                raise ValueError("Cached candidate counts are invalid")
            row["target_minimum_valid_pixels"], row["eligible_candidate_indices"] = target_candidate_eligibility(
                counts, minimum, args.target_quantile)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for n, row in enumerate(executor.map(scan, records), 1):
                tiles.append(row)
                if n % 64 == 0 or n == len(records):
                    elapsed = time.monotonic() - started
                    print(json.dumps({"processed_training_tiles": n, "elapsed_seconds": round(elapsed, 3),
                        "estimated_full_scan_seconds": round(elapsed / n * len(dataset), 1)}), flush=True)
    elapsed = time.monotonic() - started
    if source_metadata_snapshot(dataset) != snapshot:
        raise RuntimeError("Training source metadata changed during audit")
    if args.mode == "benchmark":
        print(json.dumps({"benchmark_only": True, "decoded_training_tiles": len(tiles),
            "candidate_windows_per_tile": len(origins), "min_target_pixels": minimum,
            "elapsed_seconds": elapsed, "estimated_full_scan_seconds": elapsed / len(tiles) * len(dataset)}, sort_keys=True), flush=True)
        return 0
    payload = {"schema": TARGET_INDEX_SCHEMA, "split": "train", "cities": ["DC", "PHL"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "approved_index_sha256": dataset.approved_index_sha256,
        "train_ids_sha256": canonical_sha256(list(dataset.sample_ids)), "train_count": len(dataset),
        "native_shape": dataset.native_shape, "patch_size": dataset.patch_size, "rgb_scale": dataset.rgb_scale,
        "class_names": list(GAMUS_SIX_CLASS_NAMES), "target_names": list(TARGET_NAMES),
        "candidate_stride": args.stride, "candidate_origins": origins, "min_target_pixels": minimum,
        "target_quantile": args.target_quantile,
        "target_support_rule": "max(min_target_pixels, ceil(linear quantile of same-tile candidate valid class counts))",
        "derived_from_index_sha256": args.source_index_sha256 if args.mode == "refine" else None,
        "minimum_target_fraction_of_crop": args.min_target_fraction,
        "validity": "independent image_valid AND classification_valid, never height",
        "source_metadata": snapshot, "existing_fine_index_sha256": prior["sha256"], "tiles": tiles}
    index_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    report = {"schema": "msr.rgb_training_crop_exposure_audit.v2", "passes": True,
        "index_path": str(index_path), "index_sha256": file_sha256(index_path),
        "config_sha256": file_sha256(args.config), "approved_index_sha256": dataset.approved_index_sha256,
        "implementation_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in (
            Path(__file__).resolve(), ROOT / "src/msr/data/gamus_rgb_targeted_v2.py")},
        "train_ids_sha256": payload["train_ids_sha256"], "existing_fine_index": prior,
        "indexed_training_tiles": len(tiles), "decoded_training_tiles": 0 if args.mode == "refine" else len(tiles),
        "reused_training_index_sha256": args.source_index_sha256 if args.mode == "refine" else None,
        "validation_or_test_decoded": False,
        "nyc_or_christchurch_decoded": False, "height_or_prior_decoded": False,
        "source_metadata_before": snapshot, "source_metadata_after": source_metadata_snapshot(dataset),
        "elapsed_seconds": elapsed, "cpu_only": True, "cpu_workers": args.workers,
        "sampling": {"image_sampler": "identical V1 weights: 1+2*water-hit, normalized within each city",
            "target_probability": .5, "targets": list(TARGET_NAMES), "candidate_windows_per_tile": len(origins),
            "candidate_stride": args.stride, "min_target_pixels": minimum,
            "target_quantile": args.target_quantile, "target_support_rule": payload["target_support_rule"],
            "minimum_target_fraction_of_crop": args.min_target_fraction,
            "target_selection": "uniform among targets with >=1 eligible window in selected tile",
            "window_selection": "uniform among candidate windows meeting chosen target support",
            "fallback": "ordinary uniform crop if no target has eligible windows"},
        "exposure": expected_crop_exposure(tiles, weights, dataset.patch_size, minimum),
        "method": "Exact analytic expectation: all integer ordinary origins; deterministic target grid; V1 image weights.",
        "interpretation_limit": "Exposure measures training crop support, not label errors or model performance; no validation measurements inform this audit."}
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"index": str(index_path), "index_sha256": report["index_sha256"], "report": str(report_path)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
