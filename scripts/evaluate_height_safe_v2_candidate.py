"""Evaluate the protected model and one height-safe v2 candidate identically.

This is a narrow, versioned adapter around ``evaluate_corrected_legacy_triplet``.
It deliberately reuses that runner's authenticated data loading, tensor hashing,
full-scene inference, HighBuild instance scoring, resumable partials, and pointer
protection.  The frozen three-model evaluator is not edited.  This adapter only
changes the model roster to ``protected`` plus ``height_safe_v2`` and emits the
same report schema required by the immutable height scorecard.

No official test data is read and this script never promotes a checkpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RUNNER = PROJECT_ROOT / "scripts" / "evaluate_corrected_legacy_triplet.py"
SPEC = importlib.util.spec_from_file_location(
    "msr_corrected_full_scene_runner", LEGACY_RUNNER
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - impossible in repo
    raise RuntimeError(f"Cannot import corrected evaluator: {LEGACY_RUNNER}")
legacy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = legacy
SPEC.loader.exec_module(legacy)


MODEL_NAMES = ("protected", "height_safe_v2")
legacy.MODEL_NAMES = MODEL_NAMES


def build_report(
    *,
    plan: Mapping[str, Any],
    plan_identity: Mapping[str, Any],
    model_auth: Mapping[str, Any],
    data_auth: Mapping[str, Any],
    partials: Mapping[str, Mapping[str, Any]],
    artifacts_before: Mapping[str, Any],
    artifacts_after: Mapping[str, Any],
    pointer_before: Mapping[str, Any],
    pointer_after: Mapping[str, Any],
    complete_manifest: bool,
) -> dict[str, Any]:
    """Build a two-model report with the frozen report contract."""

    input_identities = [partials[name]["input_identity"] for name in MODEL_NAMES]
    if any(identity != input_identities[0] for identity in input_identities[1:]):
        raise RuntimeError("Models did not consume identical inputs and supervision")

    primary = str(plan["protocol"]["primary_height_protocol"])
    models: dict[str, Any] = {}
    for name in MODEL_NAMES:
        partial = partials[name]
        models[name] = {
            "checkpoint": partial["checkpoint"],
            "loaded_model": partial["loaded_model"],
            "metrics": partial["protocols"][primary]["combined"],
            "protocol_metrics": partial["protocols"],
            "open_canopy_forest_metrics": partial["open_canopy_forest_metrics"],
        }

    gates = legacy._mapping(plan["gates"], "gates")
    gate = legacy._mapping(
        gates["candidate_vs_protected_max_regression"],
        "candidate_vs_protected_max_regression",
    )
    comparisons: dict[str, Any] = {}
    all_pass = True
    for protocol_name in data_auth["protocol_records"]:
        result = legacy.compare_candidate(
            partials["height_safe_v2"]["protocols"][protocol_name]["combined"],
            partials["protected"]["protocols"][protocol_name]["combined"],
            gate,
        )
        comparisons[protocol_name] = {"candidate_vs_protected": result}
        all_pass &= bool(result["passes"])

    protocols = [
        *data_auth["protocol_records"].keys(),
        data_auth["open_canopy_protocol"],
    ]
    return {
        "schema": (
            legacy.REPORT_SCHEMA if complete_manifest else legacy.SUBSET_REPORT_SCHEMA
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "passed"
            if complete_manifest and all_pass
            else ("failed" if complete_manifest else "development_subset")
        ),
        "complete_manifest": complete_manifest,
        "identical_protocol_for_all_models": True,
        "identical_input_identity": input_identities[0],
        "official_test_used": False,
        "split": "validation",
        "protocols": protocols,
        "primary_height_protocol": primary,
        "primary_metric_interpretation": (
            "Strict measured HighBuild pixels pooled once with OpenCanopy; the "
            "inclusive HighBuild protocol is independently scored."
        ),
        "evaluation_contract": {
            "highbuild_grid": "full native 1024x1024 scenes",
            "open_canopy_grid": "full native 384x384 scenes",
            "rgb": "raw",
            "relative_prior": "target-independent cached stored_01 for both suites",
            "outside_highbuild_coco_footprints": "unknown_not_ground",
            "highbuild_classification_support": "positive_annotations_only",
            "highbuild_detection_precision_and_count": "unavailable",
            "open_canopy_role": "corrected forest validation",
        },
        "plan": dict(plan_identity),
        "authenticated_artifacts_before": dict(artifacts_before),
        "authenticated_artifacts_after": dict(artifacts_after),
        "read_only_artifacts_unchanged": artifacts_before == artifacts_after,
        "live_application_before": dict(pointer_before),
        "live_application_after": dict(pointer_after),
        "live_application_pointer_changed": False,
        "models": models,
        "per_protocol_comparisons": comparisons,
        "passes_predeclared_gates": bool(complete_manifest and all_pass),
        "promotion_performed": False,
        "active_model_changed": False,
        "decision": "development evidence only; no activation or promotion",
    }


legacy.build_report = build_report


if __name__ == "__main__":
    legacy.run(legacy.parse_args())
