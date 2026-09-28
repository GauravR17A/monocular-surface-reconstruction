"""Run an immutable, app-equivalent full-raster Monocular Surface Reconstruction benchmark.

Unlike the training-time evaluator, this command never centre-crops a scene.
It follows the upload API path: bounded raster read, a fresh Depth Anything V2
prior for each RGB representation, and overlapping 512 px height tiles.  It
only reads checkpoints and writes one atomic JSON report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from msr.api.app import MAX_INFERENCE_DIMENSION, MAX_INFERENCE_PIXELS
from msr.data.surface_dataset import SurfaceSampleRecord, load_surface_manifest
from msr.evaluation.locked_benchmark import (
    RADIOMETRIC_VARIANT_DESCRIPTIONS,
    RADIOMETRIC_VARIANTS,
    benchmark_dataset_digest,
    load_benchmark_reference,
    make_radiometric_variant,
    sha256_file,
    sha256_path_tree,
)
from msr.evaluation.metrics import (
    StreamingRegressionMetrics,
    compute_height_metrics,
)
from msr.inference.predict import load_predictor, predict_height
from msr.inference.relative_depth import DepthAnythingV2Predictor
from msr.io.raster import read_rgb_raster


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELATIVE_MODEL = PROJECT_ROOT / "models/foundation/depth-anything-v2-small-hf"
METRIC_KEYS = ("pixel_count", "rmse_m", "mae_m", "bias_m", "correlation", "r2")


class MetricBook:
    """Streaming micro metrics for overall, landscape, and surface scopes."""

    def __init__(self) -> None:
        self.overall = StreamingRegressionMetrics()
        self.landscape: dict[str, StreamingRegressionMetrics] = defaultdict(
            StreamingRegressionMetrics
        )
        self.surface: dict[str, StreamingRegressionMetrics] = defaultdict(
            StreamingRegressionMetrics
        )

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        record: SurfaceSampleRecord,
        regression_mask: np.ndarray,
        surface_masks: dict[str, np.ndarray],
    ) -> None:
        self.overall.update(prediction, target, regression_mask)
        self.landscape[record.landscape].update(prediction, target, regression_mask)
        for name, mask in surface_masks.items():
            selected = regression_mask & mask
            if np.any(selected):
                self.surface[name].update(prediction, target, selected)

    def compute(self) -> dict[str, object]:
        return {
            "overall": _compact_streaming(self.overall),
            "per_landscape": {
                name: _compact_streaming(metric)
                for name, metric in sorted(self.landscape.items())
                if metric.count
            },
            "per_surface": {
                name: _compact_streaming(metric)
                for name, metric in sorted(self.surface.items())
                if metric.count
            },
        }


def _compact_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {name: metrics.get(name) for name in METRIC_KEYS}


def _compact_streaming(metrics: StreamingRegressionMetrics) -> dict[str, object]:
    if not metrics.count:
        raise ValueError("No valid benchmark pixels were accumulated")
    return _compact_metrics(metrics.compute())


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _compact_model_metadata(metadata: dict[str, object]) -> dict[str, object]:
    keys = ("epoch", "model_type", "backbone", "task", "base_checkpoint")
    return {key: metadata.get(key) for key in keys}


def _check_expected(name: str, actual: str, expected: str | None) -> None:
    if expected is not None and actual.lower() != expected.lower():
        raise ValueError(f"{name} hash mismatch: expected {expected}, found {actual}")


def _record_identity(record: SurfaceSampleRecord) -> tuple[str, str]:
    return record.landscape, record.region


def _audit_denied_manifests(
    records: list[SurfaceSampleRecord],
    rgb_hashes: set[str],
    denied_paths: Iterable[Path],
) -> list[dict[str, object]]:
    """Reject identity, regional, path, or byte-identical RGB leakage."""

    sample_ids = {record.sample_id for record in records}
    regions = {_record_identity(record) for record in records}
    rgb_paths = {record.rgb_path.resolve() for record in records}
    reports: list[dict[str, object]] = []
    for path in denied_paths:
        denied = load_surface_manifest(path)
        denied_ids = {record.sample_id for record in denied}
        denied_regions = {_record_identity(record) for record in denied}
        denied_paths_set = {record.rgb_path.resolve() for record in denied}
        denied_rgb_hashes = {sha256_file(record.rgb_path) for record in denied}
        overlaps = {
            "sample_ids": sorted(sample_ids & denied_ids),
            "landscape_regions": sorted(
                f"{landscape}:{region}"
                for landscape, region in regions & denied_regions
            ),
            "rgb_paths": sorted(str(item) for item in rgb_paths & denied_paths_set),
            "rgb_content_hashes": sorted(rgb_hashes & denied_rgb_hashes),
        }
        reports.append(
            {
                "manifest": str(path.resolve()),
                "manifest_sha256": sha256_file(path),
                "sample_count": len(denied),
                "overlaps": {name: len(values) for name, values in overlaps.items()},
            }
        )
        if any(overlaps.values()):
            details = "; ".join(
                f"{name}={values[:5]}" for name, values in overlaps.items() if values
            )
            raise ValueError(f"Benchmark leakage against {path}: {details}")
    return reports


def _licensing_metadata(records: list[SurfaceSampleRecord]) -> dict[str, object]:
    """Attach conservative source notices without pretending to give legal advice."""

    text = "\n".join(
        str(path).lower()
        for record in records
        for path in (record.rgb_path, record.surface_path)
    )
    detected: list[dict[str, str]] = []
    if "open_canopy" in text:
        detected.append(
            {
                "source": "AI4Forest Open-Canopy",
                "status": "Etalab Open Licence 2.0; attribution and frozen revision required",
                "local_evidence": "data/open_canopy_v2/subset_summary.json",
                "caveat": (
                    "Local source URLs use mutable resolve/main links and the existing test "
                    "split has already been repeatedly evaluated."
                ),
            }
        )
    if "multidomain_urban" in text or "highbuild" in text:
        detected.append(
            {
                "source": "HighBuild-1M-derived urban controls",
                "status": "internal evaluation pending complete height-label provenance",
                "local_evidence": "data/highbuild_full/LICENSES.md",
                "caveat": (
                    "The imagery inventory is source-specific, but local files do not yet "
                    "freeze a separate licence/provenance chain for height labels."
                ),
            }
        )
    if "dfc19" in text or "us3d" in text:
        detected.append(
            {
                "source": "US3D / IEEE GRSS DFC19",
                "status": "internal-only until written redistribution permission is confirmed",
                "local_evidence": "scripts/package_reviewer_lidar_cases.py",
                "caveat": (
                    "A secondary CC-attribution claim conflicts with restrictive official "
                    "DFC19 redistribution terms; the secondary mirror cannot resolve that."
                ),
            }
        )
    return {
        "legal_advice": False,
        "benchmark_code_grants_redistribution_rights": False,
        "detected_sources": detected,
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deny-manifest", type=Path, action="append", default=[])
    parser.add_argument("--benchmark-id", default="msr-app-equivalent-v1")
    parser.add_argument(
        "--benchmark-status",
        choices=("development", "legacy_regression", "locked_release"),
        default="development",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=RADIOMETRIC_VARIANTS,
        default=list(RADIOMETRIC_VARIANTS),
    )
    parser.add_argument("--relative-model", default=str(DEFAULT_RELATIVE_MODEL))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-relative-model-sha256")
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="Hash and leakage-check inputs without loading either neural network.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision, as the app does by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    denied_paths = [path.expanduser().resolve() for path in args.deny_manifest]
    if output.suffix.lower() != ".json":
        raise ValueError("Benchmark output must be a .json file")
    if output == checkpoint:
        raise ValueError("Benchmark output may not overwrite the checkpoint")
    if args.tile_size <= 0 or args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("Require tile_size > overlap >= 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")
    if args.benchmark_status == "locked_release" and args.max_samples is not None:
        raise ValueError("A locked release may not use --max-samples")
    if args.benchmark_status == "locked_release" and args.integrity_only:
        raise ValueError("Use development status for an integrity-only freeze pass")

    variants = ["raw", *(name for name in args.variants if name != "raw")]
    checkpoint_hash_before = sha256_file(checkpoint)
    manifest_hash = sha256_file(manifest)
    relative_model_path = Path(args.relative_model).expanduser()
    relative_model_hash = (
        sha256_path_tree(relative_model_path)
        if relative_model_path.exists()
        else None
    )
    _check_expected("checkpoint", checkpoint_hash_before, args.expected_checkpoint_sha256)
    _check_expected("manifest", manifest_hash, args.expected_manifest_sha256)
    if args.expected_relative_model_sha256 is not None:
        if relative_model_hash is None:
            raise ValueError(
                "Cannot verify a remote model identifier; use a frozen local "
                "--relative-model directory for locked evaluation"
            )
        _check_expected(
            "relative model", relative_model_hash, args.expected_relative_model_sha256
        )

    records = load_surface_manifest(manifest)
    full_manifest_count = len(records)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    dataset_hash, sample_hashes = benchmark_dataset_digest(records)
    _check_expected("dataset", dataset_hash, args.expected_dataset_sha256)
    rgb_hashes = {
        hashes["rgb_path"]
        for hashes in sample_hashes.values()
        if "rgb_path" in hashes
    }
    deny_reports = _audit_denied_manifests(records, rgb_hashes, denied_paths)

    warnings: list[str] = []
    lock_fully_specified = all(
        (
            args.expected_manifest_sha256,
            args.expected_dataset_sha256,
            args.expected_checkpoint_sha256,
            args.expected_relative_model_sha256,
        )
    )
    if not lock_fully_specified:
        warnings.append(
            "Hashes are reported but not all were supplied as expectations; freeze them "
            "(including the relative model) before calling this a locked release."
        )
    if not denied_paths:
        warnings.append(
            "No --deny-manifest was supplied, so train/validation leakage was not checked."
        )
    if args.max_samples is not None:
        warnings.append(
            f"Development subset only: first {len(records)}/{full_manifest_count} rows."
        )
    if args.benchmark_status == "locked_release" and not lock_fully_specified:
        raise ValueError(
            "locked_release requires expected manifest, dataset, checkpoint, and "
            "relative-model hashes"
        )
    if args.benchmark_status == "locked_release" and not denied_paths:
        raise ValueError("locked_release requires at least one --deny-manifest")

    licensing = _licensing_metadata(records)
    if args.integrity_only:
        checkpoint_hash_after = sha256_file(checkpoint)
        if checkpoint_hash_after != checkpoint_hash_before:
            raise RuntimeError("Checkpoint bytes changed during integrity audit")
        integrity_report: dict[str, object] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "integrity_only",
            "benchmark": {
                "id": args.benchmark_id,
                "manifest": str(manifest),
                "manifest_sha256": manifest_hash,
                "dataset_sha256": dataset_hash,
                "sample_count": len(records),
                "full_manifest_count": full_manifest_count,
                "landscape_counts": dict(
                    sorted(Counter(record.landscape for record in records).items())
                ),
                "warnings": warnings,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_hash_before,
                "unchanged": True,
            },
            "relative_model": {
                "path_or_id": str(args.relative_model),
                "sha256": relative_model_hash,
            },
            "integrity": {
                "deny_manifests": deny_reports,
                "checkpoint_write_operations": 0,
            },
            "licensing": licensing,
        }
        _write_json_atomic(output, integrity_report)
        print(json.dumps(integrity_report, indent=2))
        return

    # Read configuration without changing it, then release the temporary payload
    # before loading the live predictor.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data_config = payload.get("config", {}).get("data", {})
    height_max_m = float(data_config.get("height_max_m", 200.0))
    building_threshold_m = float(data_config.get("building_threshold_m", 2.0))
    del payload

    model, model_metadata = load_predictor(checkpoint, device=args.device)
    relative_predictor = DepthAnythingV2Predictor(
        model_id=args.relative_model, device=args.device
    )
    books = {variant: MetricBook() for variant in variants}
    robustness_books = {variant: MetricBook() for variant in variants if variant != "raw"}
    scene_rows: list[dict[str, object]] = []
    alignment_counts: Counter[str] = Counter()

    progress = tqdm(total=len(records) * len(variants), unit="prediction", desc="locked eval")
    try:
        for record in records:
            source = read_rgb_raster(
                record.rgb_path,
                max_pixels=MAX_INFERENCE_PIXELS,
                max_dimension=MAX_INFERENCE_DIMENSION,
            )
            reference = load_benchmark_reference(
                record,
                source,
                height_max_m=height_max_m,
                building_threshold_m=building_threshold_m,
            )
            alignment_counts.update(reference.alignments.values())
            ground_mask = (
                reference.valid_mask
                & ~reference.building_mask
                & ~reference.vegetation_mask
            )
            surface_masks = {
                "ground": ground_mask,
                "building": reference.building_mask,
                "vegetation": reference.vegetation_mask,
            }
            raw_prediction: np.ndarray | None = None
            raw_mask: np.ndarray | None = None
            scene_metrics: dict[str, dict[str, object]] = {}
            scene_robustness: dict[str, dict[str, object]] = {}

            for variant in variants:
                rgb = make_radiometric_variant(source.rgb, source.valid_mask, variant)
                relative = relative_predictor.predict(rgb, valid_mask=source.valid_mask)
                prediction = predict_height(
                    rgb,
                    model=model,
                    device=args.device,
                    tile_size=args.tile_size,
                    overlap=args.overlap,
                    amp=args.amp,
                    model_metadata=model_metadata,
                    relative_prior=relative.relative_surface,
                    valid_mask=source.valid_mask,
                )
                metric_mask = reference.regression_mask & prediction.valid_mask
                if not np.any(metric_mask):
                    raise ValueError(f"No valid evaluation pixels for {record.sample_id}")
                metrics = compute_height_metrics(
                    prediction.height_map, reference.target_m, metric_mask
                ).to_dict()
                scene_metrics[variant] = _compact_metrics(metrics)
                books[variant].update(
                    prediction.height_map,
                    reference.target_m,
                    record,
                    metric_mask,
                    surface_masks,
                )
                if variant == "raw":
                    raw_prediction = prediction.height_map.copy()
                    raw_mask = metric_mask.copy()
                else:
                    assert raw_prediction is not None and raw_mask is not None
                    comparison_mask = metric_mask & raw_mask
                    delta_metrics = compute_height_metrics(
                        prediction.height_map, raw_prediction, comparison_mask
                    ).to_dict()
                    scene_robustness[variant] = _compact_metrics(delta_metrics)
                    robustness_books[variant].update(
                        prediction.height_map,
                        raw_prediction,
                        record,
                        comparison_mask,
                        surface_masks,
                    )
                progress.update(1)
                progress.set_postfix(scene=record.sample_id, variant=variant)

            scene_rows.append(
                {
                    "sample_id": record.sample_id,
                    "region": record.region,
                    "landscape": record.landscape,
                    "target_kind": record.target_kind,
                    "processing_shape": [
                        int(source.profile["height"]),
                        int(source.profile["width"]),
                    ],
                    "input_resampled_by_app": source.resampled,
                    "source_sha256": sample_hashes[record.sample_id]["rgb_path"],
                    "reference_sha256": sample_hashes[record.sample_id]["surface_path"],
                    "metrics": scene_metrics,
                    "prediction_vs_raw": scene_robustness,
                }
            )
    finally:
        progress.close()

    result_metrics = {variant: book.compute() for variant, book in books.items()}
    raw_overall = result_metrics["raw"]["overall"]
    robustness: dict[str, object] = {}
    for variant, book in robustness_books.items():
        variant_overall = result_metrics[variant]["overall"]
        robustness[variant] = {
            "prediction_vs_raw": book.compute(),
            "task_metric_delta_vs_raw": {
                key: (
                    float(variant_overall[key]) - float(raw_overall[key])
                    if variant_overall.get(key) is not None
                    and raw_overall.get(key) is not None
                    and key != "pixel_count"
                    else None
                )
                for key in METRIC_KEYS
                if key != "pixel_count"
            },
        }

    checkpoint_hash_after = sha256_file(checkpoint)
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError(
            "Checkpoint bytes changed during evaluation; refusing to publish a report"
        )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "id": args.benchmark_id,
            "status": args.benchmark_status,
            "manifest": str(manifest),
            "manifest_sha256": manifest_hash,
            "dataset_sha256": dataset_hash,
            "sample_count": len(records),
            "full_manifest_count": full_manifest_count,
            "landscape_counts": dict(
                sorted(Counter(record.landscape for record in records).items())
            ),
            "variants": variants,
            "reference_used_for_inference": False,
            "warnings": warnings,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256_before": checkpoint_hash_before,
            "sha256_after": checkpoint_hash_after,
            "unchanged": checkpoint_hash_before == checkpoint_hash_after,
            "model_metadata": _compact_model_metadata(model_metadata),
        },
        "reproducibility": {
            "git_commit": _git_commit(),
            "git_worktree_dirty": _git_dirty(),
            "relative_model": str(args.relative_model),
            "relative_model_sha256": relative_model_hash,
            "device": str(args.device),
            "amp": bool(args.amp),
            "app_read_limits": {
                "max_pixels": MAX_INFERENCE_PIXELS,
                "max_dimension": MAX_INFERENCE_DIMENSION,
            },
            "height_tiling": {
                "tile_size": args.tile_size,
                "overlap": args.overlap,
            },
            "radiometric_variants": {
                name: RADIOMETRIC_VARIANT_DESCRIPTIONS[name] for name in variants
            },
        },
        "integrity": {
            "lock_fully_specified": lock_fully_specified,
            "deny_manifests": deny_reports,
            "alignment_counts": dict(sorted(alignment_counts.items())),
            "checkpoint_write_operations": 0,
        },
        "licensing": licensing,
        "metrics": result_metrics,
        "radiometric_robustness": robustness,
        "per_scene": scene_rows,
    }
    _write_json_atomic(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "checkpoint_unchanged": True,
                "manifest_sha256": manifest_hash,
                "dataset_sha256": dataset_hash,
                "metrics": result_metrics,
                "radiometric_robustness": robustness,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
