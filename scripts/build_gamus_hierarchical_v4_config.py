"""Build the sealed hierarchical V4 config from the fair DC+PHL V3 result.

The output is derived rather than hand-edited: all data, preprocessing, sampling,
optimizer, schedule, and height settings remain identical to the comparator.
Only the isolated head type and its vegetation-subtype auxiliary loss are added.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml

from msr.evaluation.classification_metrics import compute_multiclass_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = (
    "ground",
    "buildings",
    "water",
    "roads",
    "low_vegetation",
    "trees",
)
PROTOCOL_SCHEMA = "msr.gamus_six_class_hierarchical_dcphl.v4"
BASELINE_AUDIT_SCHEMA = (
    "msr.gamus_spatial_refined_geographic_checkpoint_audit.v1"
)
HEIGHT_OUTPUT_CONTRACT_SHA256 = (
    "a02164cb32d1fab97280d5c1766ff2e36c09fa752355fb7a73e3ea873ab644ed"
)
INDEPENDENT_REPLAY_SCHEMA = "msr.gamus_dcphl_paired_independent_replay.v1"
REPLAY_EVALUATOR_PATH = PROJECT_ROOT / "scripts" / "evaluate_gamus_dcphl_paired_replay.py"


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _mapping(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def _load_v4_auditor():
    path = PROJECT_ROOT / "scripts" / "audit_gamus_hierarchical_v4_checkpoint.py"
    spec = importlib.util.spec_from_file_location("_msr_v4_builder_auditor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load V4 auditor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recompute_baseline(metrics: Mapping[str, Any]) -> dict[str, Any]:
    stored = _mapping(
        metrics.get("six_class_identification"), "six-class identification"
    )
    if tuple(stored.get("class_names", ())) != CLASS_NAMES:
        raise ValueError("baseline uses the wrong six-class order")
    matrix = np.asarray(stored.get("confusion_matrix"))
    if matrix.shape != (6, 6) or not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError("baseline confusion matrix must be an integer 6x6 matrix")
    computed = compute_multiclass_metrics(matrix.astype(np.int64), CLASS_NAMES)
    for name in CLASS_NAMES:
        expected = float(computed["per_class"][name]["f1"])
        actual = float(_mapping(stored.get("per_class"), "per-class")[name]["f1"])
        if abs(actual - expected) > 1.0e-12:
            raise ValueError(f"stored baseline {name} F1 differs from its matrix")
    if abs(float(stored.get("macro_f1")) - float(computed["macro_f1"])) > 1.0e-12:
        raise ValueError("stored baseline macro F1 is not the six-class matrix mean")
    return computed


def derive_thresholds(
    baseline_metrics: Mapping[str, Any],
) -> dict[str, float]:
    return _load_v4_auditor().derive_fixed_gates(baseline_metrics)


def build_config(
    comparator: Mapping[str, Any],
    baseline_audit: Mapping[str, Any],
    baseline_replay: Mapping[str, Any],
    *,
    comparator_path: Path,
    baseline_audit_path: Path,
    baseline_replay_path: Path,
) -> dict[str, Any]:
    comparator_protocol = _mapping(comparator.get("protocol"), "V3 protocol")
    if comparator_protocol.get("learning_cities") != ["DC", "PHL"]:
        raise ValueError("source comparator is not the fixed DC+PHL control")
    if comparator_protocol.get("locked_holdout_interpretation") != (
        "excluded_from_this_classifier_training_not_system_unseen"
    ):
        raise ValueError("source comparator does not state NYC exposure accurately")
    if baseline_audit.get("schema") != BASELINE_AUDIT_SCHEMA:
        raise ValueError("fixed V3 baseline uses an unexpected audit schema")
    for section_name in ("recipe_audit", "state_audit", "six_class_metric_audit"):
        if _mapping(baseline_audit.get(section_name), section_name).get("passes") is not True:
            raise ValueError(f"fixed V3 baseline section failed: {section_name}")
    if baseline_audit.get("nyc_evaluation_performed") is not False:
        raise ValueError("fixed V3 baseline must not consume the NYC classifier holdout")
    if baseline_audit.get("official_test_used_by_this_comparator") is not False:
        raise ValueError("fixed V3 comparator must not reuse the official GAMUS test")
    if baseline_audit.get("official_test_global_status") != (
        "previously_consumed_forbidden_for_reuse"
    ):
        raise ValueError("fixed V3 audit must record prior official-test exposure")
    if baseline_audit.get("promotion_performed") is not False:
        raise ValueError("fixed V3 audit unexpectedly reports promotion")
    if _mapping(
        baseline_audit.get("app_pointer_verification"), "baseline app pointer"
    ).get("passes") is not True:
        raise ValueError("fixed V3 audit did not verify the protected app pointer")
    audited_config = _mapping(
        baseline_audit.get("candidate_config"), "baseline candidate config"
    )
    if audited_config.get("sha256") != file_sha256(comparator_path):
        raise ValueError("fixed V3 audit is not bound to the comparator config")
    audited_candidate = _mapping(
        baseline_audit.get("candidate"), "baseline candidate checkpoint"
    )
    audited_candidate_path = _resolve(str(audited_candidate.get("path")))
    if not audited_candidate_path.is_file():
        raise FileNotFoundError("fixed V3 audited checkpoint no longer exists")
    if audited_candidate.get("sha256") != file_sha256(audited_candidate_path):
        raise ValueError("fixed V3 audited checkpoint hash no longer matches")

    auditor = _load_v4_auditor()
    auditor.validate_fixed_v3_independent_replay(
        baseline_replay,
        comparator_config_path=comparator_path,
        baseline_audit=baseline_audit,
    )
    replay_v3 = _mapping(baseline_replay.get("v3"), "independent replay V3")
    baseline_metrics = _mapping(replay_v3.get("overall"), "V3 replay overall metrics")
    baseline_by_city = _mapping(replay_v3.get("by_city"), "V3 replay per-city metrics")
    thresholds = derive_thresholds(baseline_metrics)
    per_city_thresholds = auditor.derive_per_city_gates(baseline_by_city)

    result = deepcopy(dict(comparator))
    result["experiment"] = {
        **dict(_mapping(result.get("experiment"), "experiment")),
        "name": "multidomain_surface_gamus_six_class_hierarchical_v4",
    }
    result["protocol"] = {
        "schema": PROTOCOL_SCHEMA,
        "purpose": "paired_hierarchical_head_architecture_test_on_fixed_dc_phl",
        "source_comparator_config": str(comparator_path),
        "source_comparator_config_sha256": file_sha256(comparator_path),
        "fixed_v3_baseline_audit": str(baseline_audit_path),
        "fixed_v3_baseline_audit_sha256": file_sha256(baseline_audit_path),
        "fixed_v3_independent_replay": str(baseline_replay_path),
        "fixed_v3_independent_replay_sha256": file_sha256(baseline_replay_path),
        "independent_replay_schema": INDEPENDENT_REPLAY_SCHEMA,
        "independent_replay_evaluator_sha256": file_sha256(REPLAY_EVALUATOR_PATH),
        "independent_replay_validation_contract_sha256": _mapping(
            baseline_replay.get("validation_contract"), "replay validation contract"
        )["contract_sha256"],
        "independent_replay_source_content_manifest_sha256": _mapping(
            baseline_replay.get("source_content_binding"), "replay source binding"
        )["combined_content_manifest_sha256"],
        "paired_independent_replay_required_before_holdout": True,
        "protected_checkpoint_sha256": comparator_protocol[
            "protected_checkpoint_sha256"
        ],
        "live_pointer_file": comparator_protocol["live_pointer_file"],
        "live_pointer_file_sha256": comparator_protocol["live_pointer_file_sha256"],
        "learning_cities": ["DC", "PHL"],
        "development_validation_cities": ["DC", "PHL"],
        "locked_classifier_holdout_city": "NYC",
        "nyc_interpretation": (
            "excluded_from_this_classifier_training_not_system_unseen"
        ),
        "external_system_unseen_geography": "pending_separate_dataset",
        "official_test_policy": "previously_consumed_forbidden_for_reuse",
        "holdout_policy": "one_post_freeze_pass_no_tuning_or_reselection",
        "height_output_contract_sha256": HEIGHT_OUTPUT_CONTRACT_SHA256,
        "height_output_identity_required_before_holdout": True,
        "auto_promotion": False,
        "classification_acceptance": thresholds,
        "classification_acceptance_by_city": per_city_thresholds,
    }

    model = dict(_mapping(result.get("model"), "model"))
    if model.get("fine_semantic_head_type") != "spatial_refined":
        raise ValueError("source comparator does not use the spatial V3 head")
    model["fine_semantic_head_type"] = "hierarchical_vegetation"
    result["model"] = model

    training = dict(_mapping(result.get("training"), "training"))
    training["fine_semantic_vegetation_split_weight"] = 0.5
    result["training"] = training

    evaluation = dict(_mapping(result.get("evaluation"), "evaluation"))
    if evaluation.get("primary_selection") != {
        "suite": "gamus",
        "metric": "six_class_identification.macro_f1",
        "mode": "max",
    }:
        raise ValueError("source comparator does not select by six-class macro F1")
    # Keep V3's height-only training guards byte-for-byte equivalent in value.
    # Fixed classification gates live in protocol.classification_acceptance and
    # are enforced by the post-training auditor, so they cannot distort early
    # stopping or prevent capture of the best comparable macro-F1 checkpoint.
    evaluation["promotion_eligible"] = False
    evaluation["promotion_blockers"] = [
        "all paired DC+PHL class and safety gates must pass",
        "bit-exact height-output identity audit must pass",
        "external system-unseen geography remains pending",
        "explicit human approval is required",
    ]
    result["evaluation"] = evaluation
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-comparator-config", required=True)
    parser.add_argument("--fixed-v3-baseline-audit", required=True)
    parser.add_argument("--fixed-v3-independent-replay", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparator_path = _resolve(args.source_comparator_config)
    baseline_path = _resolve(args.fixed_v3_baseline_audit)
    baseline_replay_path = _resolve(args.fixed_v3_independent_replay)
    for path, role in (
        (comparator_path, "comparator"),
        (baseline_path, "baseline"),
        (baseline_replay_path, "independent replay"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{role} file does not exist: {path}")
    config = build_config(
        _load_yaml(comparator_path),
        _load_json(baseline_path),
        _load_json(baseline_replay_path),
        comparator_path=comparator_path,
        baseline_audit_path=baseline_path,
        baseline_replay_path=baseline_replay_path,
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(config["protocol"]["classification_acceptance"], indent=2))
    print(f"Saved sealed V4 config: {output}")


if __name__ == "__main__":
    main()
