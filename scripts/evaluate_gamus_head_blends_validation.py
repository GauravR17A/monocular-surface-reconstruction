"""Evaluate versioned GAMUS head blends on authenticated validation data only.

This command has no test-split switch and never writes a model or application
pointer.  It authenticates the protected/candidate sources and every derived
blend against the immutable sweep manifest, then evaluates the complete GAMUS
and legacy validation suites with one checkpoint resident on the GPU at a time.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import scripts.create_gamus_head_blend_sweep as blend_builder
import scripts.evaluate_routed_surface as routed_evaluator
from msr.evaluation.classification_metrics import multiclass_metric_deltas
from msr.evaluation.routed_validation import evaluate_dense_surface_model
from msr.inference.predict import load_predictor
from msr.training.scene_router_record_store import canonical_json_sha256


REPORT_SCHEMA = "msr.gamus_head_blend_validation.v1"
DEFAULT_EXPECTED_ALPHAS = (0.25, 0.5, 0.75)
SEMANTIC_GUARD_PATHS = (
    "accuracy",
    "macro_f1",
    "per_class.ground.f1",
    "per_class.building.f1",
    "per_class.vegetation.f1",
)
COMMON_HEIGHT_GUARD_PATHS = (
    "rmse_m",
    "domains.ground.rmse_m",
    "domains.building.rmse_m",
    "domains.vegetation.rmse_m",
)
SUITE_HEIGHT_GUARD_PATHS = {
    "gamus": (*COMMON_HEIGHT_GUARD_PATHS, "landscapes.mixed.rmse_m"),
    "legacy": (
        *COMMON_HEIGHT_GUARD_PATHS,
        "landscapes.urban.rmse_m",
        "landscapes.forest.rmse_m",
    ),
}


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pointer_snapshot(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "target": None}
    contents = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(contents).hexdigest(),
        "target": contents.decode("utf-8-sig").strip(),
    }


def _mapping(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    return value


def load_config(path_like: str | Path) -> dict[str, Any]:
    path = _resolve(path_like)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = dict(_mapping(value, "blend validation config"))
    config["config_path"] = str(path)
    return config


def _metric(result: Mapping[str, Any], path: str) -> float:
    value: object = result
    for part in path.split("."):
        value = _mapping(value, f"metric parent for {path}").get(part)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metric {path!r} is missing or non-numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metric {path!r} is non-finite")
    return number


def compare_suite_metrics(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], *, suite: str
) -> dict[str, Any]:
    """Return exact baseline-relative semantic and height guard evidence."""

    if suite not in SUITE_HEIGHT_GUARD_PATHS:
        raise ValueError(f"unsupported validation suite {suite!r}")
    semantic_baseline = _mapping(
        baseline.get("semantic_identification"), "baseline semantic metrics"
    )
    semantic_candidate = _mapping(
        candidate.get("semantic_identification"), "candidate semantic metrics"
    )
    semantic_deltas = {
        path: _metric(semantic_candidate, path) - _metric(semantic_baseline, path)
        for path in SEMANTIC_GUARD_PATHS
    }
    height_deltas = {
        path: _metric(candidate, path) - _metric(baseline, path)
        for path in SUITE_HEIGHT_GUARD_PATHS[suite]
    }
    semantic_guards = {path: delta >= 0.0 for path, delta in semantic_deltas.items()}
    height_guards = {path: delta <= 0.0 for path, delta in height_deltas.items()}
    return {
        "semantic": {
            "candidate_minus_protected": semantic_deltas,
            "higher_is_better": True,
            "non_regression": semantic_guards,
            "all_non_regression": all(semantic_guards.values()),
            "standard_deltas": multiclass_metric_deltas(
                semantic_candidate, semantic_baseline
            ),
        },
        "height": {
            "candidate_minus_protected_m": height_deltas,
            "lower_is_better": True,
            "non_regression": height_guards,
            "all_non_regression": all(height_guards.values()),
        },
    }


def blend_decision(
    comparisons: Mapping[str, Mapping[str, Any]],
    configured_guards: Mapping[str, Any],
) -> dict[str, object]:
    gamus = comparisons["gamus"]
    legacy = comparisons["legacy"]
    gamus_semantic = gamus["semantic"]
    gamus_height = gamus["height"]
    legacy_semantic = legacy["semantic"]
    legacy_height = legacy["height"]
    macro_gain = float(gamus_semantic["candidate_minus_protected"]["macro_f1"])
    accuracy_gain = float(gamus_semantic["candidate_minus_protected"]["accuracy"])
    gamus_improved = macro_gain > 0.0 and accuracy_gain > 0.0
    legacy_zero_regression = bool(
        legacy_semantic["all_non_regression"] and legacy_height["all_non_regression"]
    )
    height_maintained_both = bool(
        gamus_height["all_non_regression"] and legacy_height["all_non_regression"]
    )
    configured_guard_pass = bool(configured_guards.get("passes") is True)
    return {
        "gamus_semantic_improved": gamus_improved,
        "configured_stage3_guards_pass": configured_guard_pass,
        "legacy_has_zero_listed_metric_regression": legacy_zero_regression,
        "height_maintained_on_both_suites": height_maintained_both,
        "passes_development_guard": bool(
            gamus_improved and configured_guard_pass
        ),
        "passes_strict_zero_regression_audit": bool(
            gamus_improved and legacy_zero_regression and height_maintained_both
        ),
        "guard_policy": (
            "GAMUS accuracy and macro-F1 must both improve and every authenticated "
            "Stage-3 validation guard must pass. A stricter zero-regression audit "
            "is reported separately for every listed legacy semantic metric and "
            "all listed GAMUS/legacy height RMSE values. Per-class GAMUS changes "
            "are reported but are not required to all improve."
        ),
    }


def authenticate_sweep(
    manifest_path: Path, *, expected_alphas: Sequence[float]
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = dict(_mapping(manifest, "blend sweep manifest"))
    baseline_info = _mapping(manifest.get("baseline"), "blend baseline")
    candidate_info = _mapping(manifest.get("candidate"), "blend candidate")
    baseline_path = _resolve(str(baseline_info.get("checkpoint", "")))
    candidate_path = _resolve(str(candidate_info.get("checkpoint", "")))
    for role, path, info in (
        ("baseline", baseline_path, baseline_info),
        ("candidate", candidate_path, candidate_info),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if file_sha256(path) != str(info.get("checkpoint_sha256", "")):
            raise ValueError(f"{role} checkpoint hash differs from blend manifest")
    baseline_payload = torch.load(baseline_path, map_location="cpu", weights_only=False)
    candidate_payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    source_verification = blend_builder.validate_blend_sources(
        baseline_payload, candidate_payload
    )
    if canonical_json_sha256(source_verification) != canonical_json_sha256(
        _mapping(manifest.get("source_verification"), "source verification")
    ):
        raise ValueError("live blend-source verification differs from the manifest")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("blend manifest outputs must be a list")
    expected_alphas = tuple(float(alpha) for alpha in expected_alphas)
    if not expected_alphas or any(
        not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0
        for alpha in expected_alphas
    ):
        raise ValueError("configured expected_alphas must be finite values within [0, 1]")
    if len(set(expected_alphas)) != len(expected_alphas):
        raise ValueError("configured expected_alphas must be unique")
    observed_alphas = tuple(
        float(_mapping(item, "blend output")["alpha"]) for item in outputs
    )
    if observed_alphas != expected_alphas:
        raise ValueError(
            "blend manifest alphas differ from the exact configured sequence; "
            f"expected {expected_alphas}, found {observed_alphas}"
        )
    selected_names = set(source_verification["blend_tensor_names"])
    authenticated_outputs: list[dict[str, object]] = []
    for item_value in outputs:
        item = _mapping(item_value, "blend output")
        alpha = float(item["alpha"])
        checkpoint = (manifest_path.parent / str(item["checkpoint"])).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        digest = file_sha256(checkpoint)
        if digest != str(item.get("checkpoint_sha256", "")):
            raise ValueError(f"blend alpha={alpha} hash differs from manifest")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_role") != "evaluation_only_derived_head_blend":
            raise ValueError(f"blend alpha={alpha} is not evaluation-only")
        derivation = _mapping(payload.get("derivation"), "blend derivation")
        if float(derivation.get("alpha", float("nan"))) != alpha:
            raise ValueError(f"blend alpha={alpha} derivation metadata disagrees")
        blend_builder._verify_derived_payload(
            payload,
            baseline_payload,
            candidate_payload,
            alpha=alpha,
            selected_names=selected_names,
        )
        authenticated_outputs.append(
            {"alpha": alpha, "checkpoint": str(checkpoint), "checkpoint_sha256": digest}
        )
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "baseline": {
            "checkpoint": str(baseline_path),
            "checkpoint_sha256": file_sha256(baseline_path),
        },
        "candidate_source": {
            "checkpoint": str(candidate_path),
            "checkpoint_sha256": file_sha256(candidate_path),
        },
        "source_verification": source_verification,
        "outputs": authenticated_outputs,
    }


def _loader(dataset: Dataset, *, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=False,
    )


def evaluate_checkpoint(
    checkpoint: Path,
    datasets: Mapping[str, Dataset],
    *,
    device: str,
    precision: str,
    batch_size: int,
    workers: int,
    label: str,
) -> dict[str, Any]:
    """Evaluate both suites with one checkpoint resident on the GPU."""

    model, metadata = load_predictor(checkpoint, device=torch.device(device))
    model.eval()
    results: dict[str, Any] = {}
    try:
        for suite in ("gamus", "legacy"):
            dataset = datasets[suite]
            print(f"[{label}] {suite}: {len(dataset)} validation scenes", flush=True)
            results[suite] = evaluate_dense_surface_model(
                model,
                _loader(dataset, batch_size=batch_size, workers=workers),
                device=device,
                precision=precision,
            )
    finally:
        del model
        if str(device).lower().startswith("cuda"):
            torch.cuda.empty_cache()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_epoch": metadata.get("epoch"),
        "model_type": metadata.get("model_type"),
        "suites": results,
    }


def run(config_path: str | Path) -> Path:
    config = load_config(config_path)
    evaluation = _mapping(config.get("evaluation"), "evaluation config")
    output_dir = _resolve(str(evaluation["output_dir"]))
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite validation report: {output_dir}")
    batch_size = int(evaluation.get("batch_size", 4))
    workers = int(evaluation.get("num_workers", 0))
    if batch_size <= 0 or workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    device = str(evaluation.get("device", "cuda"))
    precision = str(evaluation.get("precision", "bf16"))
    if device.lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this validation configuration")

    manifest_path = _resolve(str(evaluation["blend_manifest"]))
    expected_alphas_value = evaluation.get(
        "expected_alphas", list(DEFAULT_EXPECTED_ALPHAS)
    )
    if not isinstance(expected_alphas_value, list):
        raise ValueError("evaluation.expected_alphas must be a list")
    sweep = authenticate_sweep(
        manifest_path, expected_alphas=expected_alphas_value
    )
    pointer_path = _resolve(str(evaluation["app_pointer"]))
    pointer_before = pointer_snapshot(pointer_path)
    if not pointer_before["exists"]:
        raise ValueError("protected application pointer is missing")
    if _resolve(str(pointer_before["target"])) != Path(
        str(sweep["baseline"]["checkpoint"])
    ):
        raise ValueError("application pointer no longer names the protected blend baseline")

    provenance_path = _resolve(str(evaluation["stage3_provenance"]))
    expected_provenance_sha = str(evaluation["stage3_provenance_sha256"])
    if file_sha256(provenance_path) != expected_provenance_sha:
        raise ValueError("Stage-3 validation provenance hash changed")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    guard_path = _resolve(str(evaluation["guard_config"]))
    guard_config = yaml.safe_load(guard_path.read_text(encoding="utf-8"))
    datasets, forbidden_tests, guards, data_evidence, _ = routed_evaluator._validation_plan(
        provenance, guard_config, include_test=False
    )
    if forbidden_tests or data_evidence.get("test_inventory"):
        raise RuntimeError("validation-only planner exposed a test dataset")
    if set(datasets) != {"gamus", "legacy"}:
        raise ValueError("validation planner did not return exactly GAMUS and legacy")

    protected_paths = [
        Path(str(sweep["baseline"]["checkpoint"])),
        Path(str(sweep["candidate_source"]["checkpoint"])),
        *(Path(str(item["checkpoint"])) for item in sweep["outputs"]),
        manifest_path,
        pointer_path,
        provenance_path,
        guard_path,
        Path(str(config["config_path"])),
    ]
    identity_before = {str(path): file_sha256(path) for path in protected_paths}
    print("[baseline] authenticated protected endpoint", flush=True)
    baseline = evaluate_checkpoint(
        Path(str(sweep["baseline"]["checkpoint"])),
        datasets,
        device=device,
        precision=precision,
        batch_size=batch_size,
        workers=workers,
        label="protected",
    )
    blends: list[dict[str, Any]] = []
    for item in sweep["outputs"]:
        alpha = float(item["alpha"])
        label = f"alpha={alpha:g}"
        evaluated = evaluate_checkpoint(
            Path(str(item["checkpoint"])),
            datasets,
            device=device,
            precision=precision,
            batch_size=batch_size,
            workers=workers,
            label=label,
        )
        comparisons = {
            suite: compare_suite_metrics(
                baseline["suites"][suite], evaluated["suites"][suite], suite=suite
            )
            for suite in ("gamus", "legacy")
        }
        configured_guards, _ = routed_evaluator.evaluate_validation_guards(
            evaluated["suites"], baseline["suites"], guards
        )
        decision = blend_decision(comparisons, configured_guards)
        blends.append(
            {
                "alpha": alpha,
                **evaluated,
                "comparisons": comparisons,
                "configured_stage3_guards": configured_guards,
                "decision": decision,
            }
        )
        gamus_semantic = comparisons["gamus"]["semantic"][
            "candidate_minus_protected"
        ]
        legacy_semantic = comparisons["legacy"]["semantic"][
            "candidate_minus_protected"
        ]
        gamus_height = comparisons["gamus"]["height"][
            "candidate_minus_protected_m"
        ]
        legacy_height = comparisons["legacy"]["height"][
            "candidate_minus_protected_m"
        ]
        print(
            (
                f"[{label}] complete; "
                f"GAMUS macro-F1 delta={gamus_semantic['macro_f1']:+.6f}, "
                f"accuracy delta={gamus_semantic['accuracy']:+.6f}, "
                f"RMSE delta={gamus_height['rmse_m']:+.6f} m; "
                f"legacy macro-F1 delta={legacy_semantic['macro_f1']:+.6f}, "
                f"RMSE delta={legacy_height['rmse_m']:+.6f} m; "
                f"configured_guards={decision['configured_stage3_guards_pass']}, "
                f"development_guard={decision['passes_development_guard']}"
            ),
            flush=True,
        )

    current_validation_inventory = {
        suite: routed_evaluator.dataset_inventory_fingerprint(
            dataset, source=suite, split="val", partition="calibration"
        )
        for suite, dataset in datasets.items()
    }
    if canonical_json_sha256(current_validation_inventory) != canonical_json_sha256(
        data_evidence["authenticated_validation_inventory"]
    ):
        raise RuntimeError("validation data changed during evaluation")
    identity_after = {str(path): file_sha256(path) for path in protected_paths}
    pointer_after = pointer_snapshot(pointer_path)
    if identity_after != identity_before or pointer_after != pointer_before:
        raise RuntimeError("a protected source, config, or app pointer changed during evaluation")
    passing = [
        float(item["alpha"])
        for item in blends
        if item["decision"]["passes_development_guard"]
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_validation_only": True,
        "test_splits_evaluated": False,
        "test_paths_opened": False,
        "promotion_performed": False,
        "live_application_pointer_changed": False,
        "sweep_authentication": sweep,
        "data_evidence": data_evidence,
        "runtime": {
            "device": device,
            "precision": precision,
            "batch_size": batch_size,
            "num_workers": workers,
            "one_checkpoint_on_gpu_at_a_time": True,
        },
        "protected": baseline,
        "blends": blends,
        "passing_alphas": passing,
        "conclusion": (
            "one or more blends improved GAMUS while maintaining strict legacy/height guards"
            if passing
            else "no blend improved GAMUS while maintaining every strict legacy/height guard"
        ),
        "identity_before": identity_before,
        "identity_after": identity_after,
        "pointer_before": pointer_before,
        "pointer_after": pointer_after,
        "config": {
            "path": str(config["config_path"]),
            "sha256": file_sha256(Path(str(config["config_path"]))),
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.pending-", dir=output_dir.parent)
    )
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output_dir)
    except Exception:
        # Leave the versioned staging directory for failure diagnosis; never
        # overwrite or remove user data during an evaluation failure.
        raise
    print(f"Validation report: {output_dir / 'report.json'}", flush=True)
    return output_dir / "report.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gamus_head_blend_validation_v1.yaml",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
