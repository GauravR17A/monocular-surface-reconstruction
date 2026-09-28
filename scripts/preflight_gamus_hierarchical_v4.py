"""Fail-closed preflight for the paired DC+PHL hierarchical GAMUS V4 run.

The V4 configuration is admitted only after the spatial-V3 comparator has a
sealed audit.  The two runs must use identical data, sampling, optimization,
epoch selection, and height behavior.  V4 may change only the classifier-head
type and add the fixed vegetation-split auxiliary loss.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "msr.gamus_hierarchical_v4_preflight.v1"
EXPECTED_EXPERIMENT = "multidomain_surface_gamus_six_class_hierarchical_v4"
EXPECTED_PROTOCOL = "msr.gamus_six_class_hierarchical_dcphl.v4"
HEIGHT_OUTPUT_CONTRACT_SHA256 = (
    "a02164cb32d1fab97280d5c1766ff2e36c09fa752355fb7a73e3ea873ab644ed"
)
BASELINE_AUDIT_SCHEMA = (
    "msr.gamus_spatial_refined_geographic_checkpoint_audit.v1"
)
CLASS_NAMES = (
    "ground",
    "buildings",
    "water",
    "roads",
    "low_vegetation",
    "trees",
)


def _load_v4_auditor():
    path = PROJECT_ROOT / "scripts" / "audit_gamus_hierarchical_v4_checkpoint.py"
    spec = importlib.util.spec_from_file_location("_msr_v4_auditor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load V4 auditor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V4_AUDITOR = _load_v4_auditor()


def _mapping(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


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


def _metric_at_path(metrics: Mapping[str, Any], path: str) -> float:
    value: object = metrics
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"metric path {path!r} is missing at {part!r}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric path {path!r} is not numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"metric path {path!r} is outside [0, 1]")
    return result


def derive_fixed_gates(baseline_metrics: Mapping[str, Any]) -> dict[str, float]:
    """Derive the immutable paired gates before V4 can be trained."""
    return V4_AUDITOR.derive_fixed_gates(baseline_metrics)


def validate_fixed_baseline(
    audit: Mapping[str, Any], *, comparator_sha256: str
) -> Mapping[str, Any]:
    if audit.get("schema") != BASELINE_AUDIT_SCHEMA:
        raise ValueError("paired V3 baseline uses an unexpected audit schema")
    for key in ("recipe_audit", "state_audit", "six_class_metric_audit"):
        if _mapping(audit.get(key), f"V3 {key}").get("passes") is not True:
            raise ValueError(f"paired V3 baseline {key} did not pass")
    if audit.get("nyc_evaluation_performed") is not False:
        raise ValueError("paired V3 baseline must precede NYC evaluation")
    if audit.get("official_test_used_by_this_comparator") is not False:
        raise ValueError("paired V3 comparator must not reuse official test data")
    if audit.get("official_test_global_status") != (
        "previously_consumed_forbidden_for_reuse"
    ):
        raise ValueError("paired V3 audit must record the truthful global test status")
    if audit.get("promotion_performed") is not False:
        raise ValueError("paired V3 baseline must not promote the app model")
    config_identity = _mapping(audit.get("candidate_config"), "V3 config identity")
    if config_identity.get("sha256") != comparator_sha256:
        raise ValueError("paired V3 audit is not bound to the comparator config")
    metrics = _mapping(audit.get("candidate_metrics"), "paired V3 candidate metrics")
    V4_AUDITOR.recompute_six_class_metrics(metrics)
    _metric_at_path(metrics, "road_boundary_quality.f1")
    _metric_at_path(
        metrics, "water_shadow_proxy.false_water_rate_on_dark_non_water"
    )
    return metrics


def _assert_exact_float_mapping(
    actual: Mapping[str, Any], expected: Mapping[str, float], role: str
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"{role} keys are incomplete or ambiguous")
    for key, expected_value in expected.items():
        value = actual.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or abs(float(value) - expected_value) > 1.0e-12
        ):
            raise ValueError(
                f"{role}.{key} must be frozen at {expected_value!r}, found {value!r}"
            )


def validate_v4_config(
    candidate: Mapping[str, Any],
    comparator: Mapping[str, Any],
    baseline_audit: Mapping[str, Any],
    baseline_replay: Mapping[str, Any],
    *,
    comparator_path: Path,
    baseline_replay_path: Path,
    comparator_sha256: str,
    baseline_sha256: str,
    baseline_replay_sha256: str,
) -> dict[str, Any]:
    validate_fixed_baseline(
        baseline_audit, comparator_sha256=comparator_sha256
    )
    replay_seal = V4_AUDITOR.validate_fixed_v3_independent_replay(
        baseline_replay,
        comparator_config_path=comparator_path,
        baseline_audit=baseline_audit,
    )
    replay_v3 = _mapping(baseline_replay.get("v3"), "independent replay V3")
    baseline_metrics = _mapping(replay_v3.get("overall"), "independent V3 overall")
    baseline_by_city = _mapping(replay_v3.get("by_city"), "independent V3 by city")
    fixed_gates = derive_fixed_gates(baseline_metrics)
    fixed_city_gates = V4_AUDITOR.derive_per_city_gates(baseline_by_city)

    experiment = _mapping(candidate.get("experiment"), "V4 experiment")
    comparator_experiment = _mapping(
        comparator.get("experiment"), "V3 comparator experiment"
    )
    if experiment.get("name") != EXPECTED_EXPERIMENT:
        raise ValueError("V4 experiment has an unexpected name")
    for key in ("seed", "output_root"):
        if experiment.get(key) != comparator_experiment.get(key):
            raise ValueError(f"V4 experiment.{key} must match paired V3 exactly")

    protocol = _mapping(candidate.get("protocol"), "V4 protocol")
    if protocol.get("schema") != EXPECTED_PROTOCOL:
        raise ValueError("V4 protocol schema is incorrect")
    if protocol.get("source_comparator_config_sha256") != comparator_sha256:
        raise ValueError("V4 does not pin the paired V3 config")
    if protocol.get("fixed_v3_baseline_audit_sha256") != baseline_sha256:
        raise ValueError("V4 does not pin the paired V3 audit")
    if _resolve(str(protocol.get("fixed_v3_independent_replay", ""))) != baseline_replay_path:
        raise ValueError("V4 does not name the fixed independent replay")
    if protocol.get("fixed_v3_independent_replay_sha256") != baseline_replay_sha256:
        raise ValueError("V4 does not pin the fixed independent replay")
    if protocol.get("independent_replay_schema") != V4_AUDITOR.INDEPENDENT_REPLAY_SCHEMA:
        raise ValueError("V4 does not pin the independent replay schema")
    if protocol.get("independent_replay_evaluator_sha256") != V4_AUDITOR.file_sha256(
        V4_AUDITOR.REPLAY_EVALUATOR_PATH
    ):
        raise ValueError("V4 does not pin the independent replay evaluator")
    if protocol.get("paired_independent_replay_required_before_holdout") is not True:
        raise ValueError("V4 must require paired replay before holdout")
    if protocol.get("nyc_interpretation") != (
        "excluded_from_this_classifier_training_not_system_unseen"
    ):
        raise ValueError("V4 must describe NYC exposure accurately")
    if protocol.get("external_system_unseen_geography") != "pending_separate_dataset":
        raise ValueError("V4 must preserve the external-geography blocker")
    if protocol.get("official_test_policy") != (
        "previously_consumed_forbidden_for_reuse"
    ):
        raise ValueError("V4 must record and forbid reuse of the consumed official test")
    if protocol.get("auto_promotion") is not False:
        raise ValueError("V4 must disable automatic promotion")
    if protocol.get("height_output_contract_sha256") != HEIGHT_OUTPUT_CONTRACT_SHA256:
        raise ValueError("V4 must pin the canonical height-output contract")
    if protocol.get("height_output_identity_required_before_holdout") is not True:
        raise ValueError("V4 must require output-level height identity")
    if protocol.get("protected_checkpoint_sha256") != _mapping(
        comparator.get("protocol"), "V3 protocol"
    ).get("protected_checkpoint_sha256"):
        raise ValueError("V4 protected-checkpoint identity differs from paired V3")
    _assert_exact_float_mapping(
        _mapping(protocol.get("classification_acceptance"), "V4 fixed gates"),
        fixed_gates,
        "classification_acceptance",
    )
    city_gates = _mapping(
        protocol.get("classification_acceptance_by_city"), "V4 per-city gates"
    )
    if set(city_gates) != set(V4_AUDITOR.ALLOWED_CITIES):
        raise ValueError("V4 per-city gates must contain exactly DC and PHL")
    for city in V4_AUDITOR.ALLOWED_CITIES:
        _assert_exact_float_mapping(
            _mapping(city_gates.get(city), f"V4 {city} gates"),
            fixed_city_gates[city],
            f"classification_acceptance_by_city.{city}",
        )

    if candidate.get("data") != comparator.get("data"):
        raise ValueError("V4 and V3 must use byte-identical DC+PHL data settings")
    candidate_model = deepcopy(dict(_mapping(candidate.get("model"), "V4 model")))
    comparator_model = deepcopy(
        dict(_mapping(comparator.get("model"), "V3 comparator model"))
    )
    if candidate_model.pop("fine_semantic_head_type", None) != (
        "hierarchical_vegetation"
    ):
        raise ValueError("V4 must use the hierarchical vegetation head")
    if comparator_model.pop("fine_semantic_head_type", None) != "spatial_refined":
        raise ValueError("paired comparator is not spatial V3")
    if candidate_model != comparator_model:
        raise ValueError("V4 changes model behavior outside the classifier head")

    candidate_training = deepcopy(
        dict(_mapping(candidate.get("training"), "V4 training"))
    )
    comparator_training = deepcopy(
        dict(_mapping(comparator.get("training"), "V3 comparator training"))
    )
    split_weight = candidate_training.pop(
        "fine_semantic_vegetation_split_weight", None
    )
    if split_weight != 0.5:
        raise ValueError("V4 vegetation-split auxiliary weight must equal 0.5")
    if candidate_training != comparator_training:
        raise ValueError(
            "V4 training schedule/optimizer/losses differ from paired V3"
        )

    candidate_evaluation = deepcopy(
        dict(_mapping(candidate.get("evaluation"), "V4 evaluation"))
    )
    comparator_evaluation = deepcopy(
        dict(_mapping(comparator.get("evaluation"), "V3 evaluation"))
    )
    candidate_blockers = candidate_evaluation.pop("promotion_blockers", None)
    comparator_evaluation.pop("promotion_blockers", None)
    candidate_guards = _mapping(
        candidate_evaluation.pop("validation_guards", None), "V4 guards"
    )
    comparator_guards = _mapping(
        comparator_evaluation.pop("validation_guards", None), "V3 guards"
    )
    if candidate_evaluation != comparator_evaluation:
        raise ValueError("V4 evaluation differs beyond fixed validation gates")
    if candidate_evaluation.get("promotion_eligible") is not False:
        raise ValueError("V4 must remain ineligible for automatic promotion")
    if not isinstance(candidate_blockers, list) or not candidate_blockers:
        raise ValueError("V4 must state its unresolved promotion blockers")

    # Final class gates must not participate in early stopping.  Keeping the
    # paired V3 height-only guards here ensures the best macro-F1 checkpoint is
    # captured honestly; protocol.classification_acceptance is enforced only
    # by the post-training auditor.
    if candidate_guards != comparator_guards:
        raise ValueError("V4 must retain V3's exact training-time height guards")

    return {
        "passes": True,
        "paired_v3_comparator_sha256": comparator_sha256,
        "paired_v3_baseline_audit_sha256": baseline_sha256,
        "fixed_v3_independent_replay_sha256": baseline_replay_sha256,
        "independent_replay_validation": replay_seal,
        "fixed_classification_gates": fixed_gates,
        "fixed_classification_gates_by_city": fixed_city_gates,
        "same_data": True,
        "same_seed_schedule_optimizer_and_selection": True,
        "only_model_change": "hierarchical_vegetation classifier head",
        "only_training_change": "vegetation split auxiliary weight 0.5",
        "height_output_identity_required_before_holdout": True,
        "nyc_interpretation": protocol["nyc_interpretation"],
        "nyc_evaluated": False,
        "official_test_used_by_v4": False,
        "official_test_global_status": "previously_consumed_forbidden_for_reuse",
        "auto_promotion": False,
    }


def build_report(
    config_path: Path,
    comparator_path: Path,
    baseline_path: Path,
    baseline_replay_path: Path,
) -> dict[str, Any]:
    candidate = _load_yaml(config_path)
    comparator = _load_yaml(comparator_path)
    baseline = _load_json(baseline_path)
    baseline_replay = _load_json(baseline_replay_path)
    validation = validate_v4_config(
        candidate,
        comparator,
        baseline,
        baseline_replay,
        comparator_path=comparator_path,
        baseline_replay_path=baseline_replay_path,
        comparator_sha256=file_sha256(comparator_path),
        baseline_sha256=file_sha256(baseline_path),
        baseline_replay_sha256=file_sha256(baseline_replay_path),
    )
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "paired_v3_config": {
            "path": str(comparator_path),
            "sha256": file_sha256(comparator_path),
        },
        "paired_v3_baseline_audit": {
            "path": str(baseline_path),
            "sha256": file_sha256(baseline_path),
        },
        "fixed_v3_independent_replay": {
            "path": str(baseline_replay_path),
            "sha256": file_sha256(baseline_replay_path),
        },
        "validation": validation,
        "training_started": False,
        "nyc_evaluated": False,
        "promotion_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-comparator-config", required=True)
    parser.add_argument("--fixed-v3-baseline-audit", required=True)
    parser.add_argument("--fixed-v3-independent-replay", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    comparator_path = _resolve(args.source_comparator_config)
    baseline_path = _resolve(args.fixed_v3_baseline_audit)
    baseline_replay_path = _resolve(args.fixed_v3_independent_replay)
    for path, role in (
        (config_path, "V4 config"),
        (comparator_path, "paired V3 config"),
        (baseline_path, "paired V3 baseline audit"),
        (baseline_replay_path, "fixed V3 independent replay"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{role} does not exist: {path}")
    report = build_report(
        config_path, comparator_path, baseline_path, baseline_replay_path
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report["validation"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
