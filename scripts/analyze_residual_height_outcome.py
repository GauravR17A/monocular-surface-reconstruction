"""Summarize the completed residual pilot from saved reports only (no inference)."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/20260915T124056Z_residual_height_v1"
OUTPUT = ROOT / "outputs/diagnostics/residual_height_v1_outcome/report.json"
KEYS = ("rmse_m", "mae_m", "bias_m", "correlation", "r2", "pixel_count")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def extract_metrics(metrics: dict) -> dict:
    return {key: metrics[key] for key in KEYS}


def metric_pair(before: dict, after: dict) -> dict:
    same_count = before["pixel_count"] == after["pixel_count"]
    if not same_count:
        raise ValueError("Cannot make a paired summary: support count changed")
    deltas = {}
    for key in KEYS[:-1]:
        a, b = before[key], after[key]
        deltas[key] = b - a if a is not None and b is not None else None
    return {
        "baseline": extract_metrics(before),
        "candidate": extract_metrics(after),
        "candidate_minus_baseline": deltas,
        "pixel_count_identity": same_count,
        "rmse_reduction_percent": 100 * (before["rmse_m"] - after["rmse_m"]) / before["rmse_m"],
    }


def summarize_suite(before: dict, after: dict) -> dict:
    result = {"overall": metric_pair(before, after)}
    for group in ("domains", "tall_objects", "regions", "landscapes"):
        if set(before[group]) != set(after[group]):
            raise ValueError(f"Support keys changed: {group}")
        result[group] = {
            name: metric_pair(before[group][name], after[group][name])
            for name in sorted(before[group])
        }
    for name in ("domain_macro_rmse_m", "region_macro_rmse_m"):
        result[name] = {
            "baseline": before[name],
            "candidate": after[name],
            "candidate_minus_baseline": after[name] - before[name],
        }
    result["classification_metrics_identical"] = (
        before["semantic_identification"] == after["semantic_identification"]
    )
    if not result["classification_metrics_identical"]:
        raise ValueError("Protected classification metrics changed")
    return result


def failed_checks(node: dict, prefix: str = "") -> dict:
    failures = {}
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if value.get("passes") is False:
                failures[path] = value
            else:
                failures.update(failed_checks(value, path))
        elif value is False:
            failures[path] = value
    return failures


def region_distribution(paired_regions: dict) -> dict:
    rows = [
        {"name": name, **pair}
        for name, pair in paired_regions.items()
    ]
    rows.sort(key=lambda row: row["candidate_minus_baseline"]["rmse_m"])
    changes = [row["candidate_minus_baseline"]["rmse_m"] for row in rows]
    return {
        "region_group_count": len(rows),
        "improved_rmse_count": sum(delta < -1e-9 for delta in changes),
        "worsened_rmse_count": sum(delta > 1e-9 for delta in changes),
        "unchanged_rmse_count": sum(abs(delta) <= 1e-9 for delta in changes),
        "unweighted_mean_rmse_change_m": sum(changes) / len(changes),
        "largest_improvements": [row for row in rows if row["candidate_minus_baseline"]["rmse_m"] < -1e-9][:3],
        "largest_regressions": [row for row in reversed(rows) if row["candidate_minus_baseline"]["rmse_m"] > 1e-9][:3],
        "note": "Region-group aggregates, not independent scene or per-building counts.",
    }


def main() -> None:
    baseline = load_json(RUN / "baseline_validation.json")
    epochs = [json.loads(line) for line in (RUN / "metrics.jsonl").read_text().splitlines() if line.strip()]
    summary = load_json(RUN / "training_summary.json")
    config = yaml.safe_load((RUN / "config.yaml").read_text())
    assert [row["epoch"] for row in epochs] == [1, 2, 3, 4, 5]
    assert summary["stage"] == "early_stopped"
    assert summary["best_guarded_checkpoint"] is None

    protected = {}
    for label, section, field in (
        ("checkpoint", "model", "protected_checkpoint"),
        ("pointer", "evaluation", "protected_pointer"),
    ):
        path = ROOT / config[section][field]
        expected = config[section][field + "_sha256"]
        actual = sha256(path)
        protected[label] = {"path": str(path), "expected_sha256": expected, "actual_sha256": actual, "matches": actual == expected}
        if actual != expected:
            raise ValueError(f"Protected {label} hash changed")

    rows = []
    for epoch in epochs:
        validation = epoch["validation"]
        assert set(validation) == set(baseline)
        paired = {
            source: summarize_suite(baseline[source]["metrics"], validation[source]["metrics"])
            for source in baseline
        }
        rows.append({
            "epoch": epoch["epoch"],
            "material_improvement": epoch["gate"]["material_improvement"],
            "all_regression_guards_pass": epoch["gate"]["all_regression_guards_pass"],
            "eligible": epoch["gate"]["passes"],
            "selection_score": epoch["selection_score"],
            "failed_checks": failed_checks(epoch["gate"]["checks"]),
            "paired_validation": paired,
            "region_distributions": {source: region_distribution(value["regions"]) for source, value in paired.items()},
        })
    assert not any(row["eligible"] for row in rows)

    final = rows[-1]
    building_pair = final["paired_validation"]["highbuild"]["domains"]["building"]
    decomposition = {}
    for name in ("baseline", "candidate"):
        values = building_pair[name]
        mse = values["rmse_m"] ** 2
        bias_sq = values["bias_m"] ** 2
        decomposition[name] = {
            "mse_m2": mse,
            "squared_bias_m2": bias_sq,
            "centered_error_variance_m2": max(0.0, mse - bias_sq),
            "centered_error_rmse_m": math.sqrt(max(0.0, mse - bias_sq)),
        }

    tall_pair = final["paired_validation"]["highbuild"]["tall_objects"]["building"]
    tall_contribution = {}
    for side in ("baseline", "candidate"):
        total = building_pair[side]
        tall = tall_pair[side]
        tall_contribution[side] = {
            "share_of_measured_building_pixels": tall["pixel_count"] / total["pixel_count"],
            "share_of_measured_building_squared_error": (tall["rmse_m"] ** 2 * tall["pixel_count"]) / (total["rmse_m"] ** 2 * total["pixel_count"]),
        }

    report = {
        "schema": "msr.residual_height_v1_outcome.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": str(RUN),
        "evidence_type": "saved_paired_development_validation_reports_only",
        "model_inference_performed": False,
        "new_dataset_access_performed": False,
        "official_test_or_external_holdout_access_performed": False,
        "source_artifact_sha256": {name: sha256(RUN / name) for name in ("baseline_validation.json", "metrics.jsonl", "training_summary.json", "config.yaml", "source_code_manifest.json")},
        "training_summary": summary,
        "protected_artifacts": protected,
        "support_statement": "Source/group keys and each saved pixel count match baseline in all five epochs. Matching counts alone do not independently prove spatial-mask identity; this analysis did not reread masks or inputs.",
        "epochs": rows,
        "best_eligible_epoch": None,
        "lowest_rejected_selection_score_epoch": min(rows, key=lambda item: item["selection_score"])["epoch"],
        "highbuild_final_error_decomposition": decomposition,
        "highbuild_tall_building_error_contribution": tall_contribution,
        "tall_subset_reference_thresholds_m": {"building": 20.0, "vegetation": 15.0},
        "interpretation_limits": [
            "HighBuild measured COCO-intersection pixels only; these are not per-building instance metrics.",
            "GAMUS height metres remain assumed by this run's contract, not newly publisher-verified here.",
            "OpenCanopy validation overlaps source mosaics with training; development evidence is not untouched-geography evidence.",
            "Absolute bias reduction and centered error variance are descriptive, not proof of what features the network learned.",
            "No residual candidate passed all registered guards; no replacement or final scorecard is claimed.",
        ],
        "next_train_only_diagnostics": [
            "On measured HighBuild training annotations, tabulate building-instance height bands, city, label provenance and roof support before any new recipe; never count unknown pixels as zero.",
            "Inspect fixed train-only examples stratified by measured height and dataset: reference alignment, input resolution, image/prior identity and residual changes on correctly labelled ground versus objects.",
            "Test whether independent per-building supervision and tall-building exposure change the fixed train-only optimization proof before scheduling a full run; keep classification untouched.",
        ],
        "matched_source_ablation_proposal": {
            "status": "proposal_not_run",
            "hypothesis": "GAMUS height supervision may trade off against verified legacy height supervision; current aggregate scores do not establish causation.",
            "control": "A fresh matched three-source residual recipe, not only the historical pilot if exposure or update budgets differ.",
            "candidate": "Same protected start, architecture, initialization, legacy examples/crops, legacy loss coefficients and optimizer steps; omit GAMUS height-loss contribution only, leaving its identification task separate.",
            "controls": "Match per-source HighBuild/OpenCanopy exposures and their numerical loss weights. Do not silently replace removed GAMUS batches with more legacy training. Predeclare total updates, schedule, seeds, masks and validation protocol.",
            "decision": "Evaluate all existing source/domain and tall-object guards under the same native development protocol; preserve GAMUS unit caveat. After a reproducible benefit, test one per-building/tall-sampling change separately. No tuning on the sealed external holdout.",
        },
        "saved_epoch_checkpoints": [
            {"path": str(path), "size_bytes": path.stat().st_size}
            for path in sorted(RUN.glob("checkpoint_epoch_*.pt"))
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(OUTPUT),
        "epochs": len(rows),
        "best_eligible_epoch": None,
        "final_failed_checks": final["failed_checks"],
        "final_region_counts": {source: {key: value for key, value in distribution.items() if key.endswith("count")} for source, distribution in final["region_distributions"].items()},
        "highbuild_error_decomposition": decomposition,
        "tall_contribution": tall_contribution,
        "protected_hashes_match": all(item["matches"] for item in protected.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
