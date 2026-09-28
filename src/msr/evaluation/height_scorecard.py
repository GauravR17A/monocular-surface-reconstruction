"""Fail-closed authentication and comparison for metric-height candidates.

This module never loads model tensors, changes a checkpoint, or updates the
application pointer.  It only authenticates an already-completed full-scene
evaluation report against a versioned scorecard and returns an eligibility
decision.  Production promotion remains a separate, explicit human action.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


SCORECARD_SCHEMA = "msr.height_scorecard.v1"
REPORT_SCHEMA = "msr.corrected_legacy_triplet.v1"
DECISION_SCHEMA = "msr.height_candidate_decision.v1"


class HeightScorecardError(ValueError):
    """Raised when scorecard or report evidence cannot be authenticated."""


@dataclass(frozen=True)
class AuthenticatedScorecard:
    """A scorecard plus its byte identity and authenticated baseline report."""

    config: dict[str, Any]
    path: Path
    sha256: str
    baseline_report: dict[str, Any]


def file_sha256(path: str | Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).resolve().open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise HeightScorecardError(f"{role} must be a mapping")
    return dict(value)


def _resolve(raw: str | Path, project_root: Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HeightScorecardError(f"Cannot read JSON evidence {path}: {error}") from error
    return _mapping(value, str(path))


def _metric_at(metrics: Mapping[str, Any], path: str) -> float:
    value: Any = metrics
    for component in path.split("."):
        value = _mapping(value, f"metric parent for {path}").get(component)
        if value is None:
            raise HeightScorecardError(f"Required metric {path!r} is missing")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HeightScorecardError(f"Metric {path!r} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HeightScorecardError(f"Metric {path!r} is not finite")
    return result


def _assert_exact_metrics(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, tolerance: float = 1e-12
) -> None:
    for path, expected_value in expected.items():
        actual_value = _metric_at(actual, str(path))
        if not math.isclose(
            actual_value,
            float(expected_value),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise HeightScorecardError(
                f"Frozen baseline metric {path!r} changed: expected "
                f"{expected_value}, found {actual_value}"
            )


def _report_model_protocol(report: Mapping[str, Any], model_name: str, protocol: str) -> dict[str, Any]:
    models = _mapping(report.get("models"), "report models")
    model = _mapping(models.get(model_name), f"report model {model_name}")
    protocols = _mapping(model.get("protocol_metrics"), f"{model_name} protocol metrics")
    return _mapping(protocols.get(protocol), f"{model_name} protocol {protocol}")


def _report_model(report: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    return _mapping(
        _mapping(report.get("models"), "report models").get(model_name),
        f"report model {model_name}",
    )


def _artifact_hashes(report: Mapping[str, Any], when: str) -> dict[str, str]:
    artifacts = _mapping(
        report.get(f"authenticated_artifacts_{when}"),
        f"authenticated artifacts {when}",
    )
    result: dict[str, str] = {}
    for name, raw in artifacts.items():
        identity = _mapping(raw, f"artifact {name}")
        digest = identity.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise HeightScorecardError(f"Artifact {name!r} has no valid SHA-256")
        result[str(name)] = digest.lower()
    return result


def _pointer_target(report: Mapping[str, Any], when: str) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _mapping(report.get(f"live_application_{when}"), f"live application {when}")
    return (
        _mapping(identity.get("pointer"), f"live pointer {when}"),
        _mapping(identity.get("target"), f"live target {when}"),
    )


def _assert_common_report_contract(
    report: Mapping[str, Any], scorecard: Mapping[str, Any]
) -> None:
    protocol = _mapping(scorecard.get("protocol"), "scorecard protocol")
    if report.get("schema") != REPORT_SCHEMA:
        raise HeightScorecardError("Candidate is not a complete corrected full-scene report")
    required_flags = {
        "complete_manifest": True,
        "identical_protocol_for_all_models": True,
        "official_test_used": False,
        "read_only_artifacts_unchanged": True,
        "live_application_pointer_changed": False,
        "promotion_performed": False,
        "active_model_changed": False,
    }
    for name, expected in required_flags.items():
        if report.get(name) is not expected:
            raise HeightScorecardError(f"Report safety flag {name!r} must be {expected}")
    if report.get("split") != protocol.get("split"):
        raise HeightScorecardError("Candidate report uses a different split")
    if report.get("primary_height_protocol") != protocol.get("primary_height_protocol"):
        raise HeightScorecardError("Candidate report uses a different height protocol")
    if list(report.get("protocols", [])) != list(protocol.get("protocols", [])):
        raise HeightScorecardError("Candidate report protocol order/content changed")
    if report.get("evaluation_contract") != protocol.get("evaluation_contract"):
        raise HeightScorecardError("Candidate report changed the full-scene evaluation contract")


def _assert_live_pointer(
    report: Mapping[str, Any], scorecard: Mapping[str, Any], project_root: Path
) -> None:
    expected = _mapping(scorecard.get("protected_application"), "protected application")
    before_pointer, before_target = _pointer_target(report, "before")
    after_pointer, after_target = _pointer_target(report, "after")
    if (before_pointer, before_target) != (after_pointer, after_target):
        raise HeightScorecardError("Live application pointer changed during evaluation")
    if before_pointer.get("sha256", "").lower() != str(expected["pointer_sha256"]).lower():
        raise HeightScorecardError("Live application pointer bytes differ from the frozen pointer")
    if before_target.get("sha256", "").lower() != str(expected["checkpoint_sha256"]).lower():
        raise HeightScorecardError("Live application target differs from the protected checkpoint")

    pointer_path = _resolve(expected["pointer"], project_root)
    checkpoint_path = _resolve(expected["checkpoint"], project_root)
    if not pointer_path.is_file() or not checkpoint_path.is_file():
        raise HeightScorecardError("Protected application pointer/checkpoint is missing")
    if file_sha256(pointer_path) != str(expected["pointer_sha256"]).lower():
        raise HeightScorecardError("Current live pointer hash does not match the scorecard")
    if file_sha256(checkpoint_path) != str(expected["checkpoint_sha256"]).lower():
        raise HeightScorecardError("Current protected checkpoint hash does not match the scorecard")
    pointer_value = pointer_path.read_text(encoding="utf-8-sig").strip()
    if _resolve(pointer_value, project_root) != checkpoint_path:
        raise HeightScorecardError("Current live pointer does not resolve to the protected checkpoint")


def _assert_data_artifacts(
    report: Mapping[str, Any], scorecard: Mapping[str, Any], project_root: Path
) -> None:
    expected = {
        str(name): str(value).lower()
        for name, value in _mapping(
            scorecard.get("data_artifact_sha256"), "data artifact hashes"
        ).items()
    }
    before = _artifact_hashes(report, "before")
    after = _artifact_hashes(report, "after")
    if before != after:
        raise HeightScorecardError("Authenticated artifacts changed during evaluation")
    for name, digest in expected.items():
        if before.get(name) != digest:
            raise HeightScorecardError(f"Data artifact {name!r} differs from the scorecard")
        raw_identity = _mapping(
            _mapping(
                report.get("authenticated_artifacts_before"),
                "authenticated artifacts before",
            ).get(name),
            f"artifact {name}",
        )
        artifact_path = _resolve(raw_identity.get("path", ""), project_root)
        if not artifact_path.is_file() or file_sha256(artifact_path) != digest:
            raise HeightScorecardError(
                f"Current bytes for data artifact {name!r} do not match the scorecard"
            )


def _assert_input_identity(report: Mapping[str, Any], scorecard: Mapping[str, Any]) -> None:
    expected = _mapping(scorecard.get("input_identity"), "input identity")
    actual = _mapping(report.get("identical_input_identity"), "report input identity")
    if actual != expected:
        raise HeightScorecardError(
            "Candidate did not consume the exact frozen inputs and supervision tensors"
        )


def _assert_protected_model(report: Mapping[str, Any], scorecard: Mapping[str, Any]) -> None:
    baseline = _mapping(scorecard.get("baseline"), "baseline")
    model_name = str(baseline["model_key"])
    model = _report_model(report, model_name)
    checkpoint = _mapping(model.get("checkpoint"), "protected checkpoint identity")
    if checkpoint.get("sha256", "").lower() != str(baseline["checkpoint_sha256"]).lower():
        raise HeightScorecardError("Candidate report did not replay the protected checkpoint")
    metrics = _report_model_protocol(
        report, model_name, str(scorecard["protocol"]["primary_height_protocol"])
    )
    _assert_exact_metrics(metrics, _mapping(baseline.get("metrics"), "baseline metrics"))


def _assert_supporting_audits(scorecard: Mapping[str, Any], project_root: Path) -> None:
    audits = _mapping(scorecard.get("supporting_audits"), "supporting audits")
    for name, raw in audits.items():
        identity = _mapping(raw, f"supporting audit {name}")
        path = _resolve(identity["path"], project_root)
        if not path.is_file() or file_sha256(path) != str(identity["sha256"]).lower():
            raise HeightScorecardError(f"Supporting audit {name!r} hash changed")
    gamus = _mapping(audits.get("gamus_preparation"), "GAMUS preparation audit")
    audit_path = _resolve(gamus["path"], project_root)
    audit = _load_json(audit_path)
    masks = _mapping(audit.get("validity_mask_contract"), "GAMUS validity-mask contract")
    if masks.get("status") != "passed":
        raise HeightScorecardError("GAMUS independent validity-mask audit did not pass")
    unit = _mapping(audit.get("unit_and_scale"), "GAMUS unit/scale audit")
    if unit.get("height_numeric_unit_status") != "unverified_explicit_project_assumption":
        raise HeightScorecardError("GAMUS unit status changed; review is required")
    split_policy = _mapping(audit.get("official_split_policy"), "GAMUS split policy")
    if split_policy.get("test") != "inventory_audit_only; never constructed by the pilot trainer":
        raise HeightScorecardError("GAMUS official-test exclusion is not authenticated")


def authenticate_scorecard(
    scorecard_path: str | Path, *, project_root: str | Path
) -> AuthenticatedScorecard:
    root = Path(project_root).resolve()
    path = _resolve(scorecard_path, root)
    scorecard_hash = file_sha256(path)
    seal_path = path.with_suffix(".sha256")
    if seal_path.is_file():
        declared = seal_path.read_text(encoding="ascii").strip().split()[0].lower()
        if declared != scorecard_hash:
            raise HeightScorecardError("Height scorecard does not match its SHA-256 seal")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise HeightScorecardError(f"Cannot read scorecard {path}: {error}") from error
    scorecard = _mapping(config, "height scorecard")
    if scorecard.get("schema") != SCORECARD_SCHEMA:
        raise HeightScorecardError("Unsupported height scorecard schema")

    baseline = _mapping(scorecard.get("baseline"), "baseline")
    report_path = _resolve(baseline["report"], root)
    report_hash = file_sha256(report_path)
    if report_hash != str(baseline["report_sha256"]).lower():
        raise HeightScorecardError("Frozen baseline report hash changed")
    report = _load_json(report_path)
    _assert_common_report_contract(report, scorecard)
    _assert_data_artifacts(report, scorecard, root)
    _assert_input_identity(report, scorecard)
    _assert_live_pointer(report, scorecard, root)
    _assert_protected_model(report, scorecard)
    _assert_supporting_audits(scorecard, root)
    return AuthenticatedScorecard(
        config=scorecard,
        path=path,
        sha256=scorecard_hash,
        baseline_report=report,
    )


def _compare_no_regression(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], limits: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    checks: dict[str, Any] = {}
    passes = True
    for path, raw_limit in limits.items():
        candidate_value = _metric_at(candidate, str(path))
        baseline_value = _metric_at(baseline, str(path))
        delta = candidate_value - baseline_value
        passed = delta <= float(raw_limit) + 1e-12
        checks[str(path)] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "candidate_minus_baseline": delta,
            "max_regression": float(raw_limit),
            "passes": passed,
        }
        passes &= passed
    return checks, passes


def _compare_floor(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], paths: list[str]
) -> tuple[dict[str, Any], bool]:
    checks: dict[str, Any] = {}
    passes = True
    for path in paths:
        candidate_value = _metric_at(candidate, path)
        baseline_value = _metric_at(baseline, path)
        passed = candidate_value + 1e-12 >= baseline_value
        checks[path] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "candidate_minus_baseline": candidate_value - baseline_value,
            "passes": passed,
        }
        passes &= passed
    return checks, passes


def _compare_absolute_no_regression(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], limits: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    checks: dict[str, Any] = {}
    passes = True
    for path, raw_limit in limits.items():
        candidate_value = _metric_at(candidate, str(path))
        baseline_value = _metric_at(baseline, str(path))
        delta = abs(candidate_value) - abs(baseline_value)
        passed = delta <= float(raw_limit) + 1e-12
        checks[str(path)] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "absolute_candidate_minus_absolute_baseline": delta,
            "max_absolute_regression": float(raw_limit),
            "passes": passed,
        }
        passes &= passed
    return checks, passes


def _material_improvements(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    paths = list(config.get("paths", []))
    if not paths:
        raise HeightScorecardError("Material-improvement paths are empty")
    minimum_m = float(config["minimum_absolute_m"])
    minimum_fraction = float(config["minimum_relative_fraction"])
    required = int(config.get("minimum_passing_paths", 1))
    checks: dict[str, Any] = {}
    count = 0
    for path in paths:
        baseline_value = _metric_at(baseline, str(path))
        candidate_value = _metric_at(candidate, str(path))
        improvement_m = baseline_value - candidate_value
        improvement_fraction = improvement_m / baseline_value if baseline_value > 0 else 0.0
        passed = improvement_m + 1e-12 >= minimum_m or improvement_fraction + 1e-12 >= minimum_fraction
        count += int(passed)
        checks[str(path)] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "improvement_m": improvement_m,
            "improvement_fraction": improvement_fraction,
            "passes": passed,
        }
    return checks, count >= required


def assess_candidate(
    authenticated: AuthenticatedScorecard,
    candidate_report_path: str | Path,
    *,
    candidate_model_key: str,
    project_root: str | Path,
) -> dict[str, Any]:
    """Authenticate and score one candidate report without promoting anything."""

    root = Path(project_root).resolve()
    scorecard = authenticated.config
    candidate_path = _resolve(candidate_report_path, root)
    candidate_report = _load_json(candidate_path)
    _assert_common_report_contract(candidate_report, scorecard)
    _assert_data_artifacts(candidate_report, scorecard, root)
    _assert_input_identity(candidate_report, scorecard)
    _assert_live_pointer(candidate_report, scorecard, root)
    _assert_protected_model(candidate_report, scorecard)

    candidate_model = _report_model(candidate_report, candidate_model_key)
    checkpoint = _mapping(candidate_model.get("checkpoint"), "candidate checkpoint")
    checkpoint_path = _resolve(checkpoint["path"], root)
    checkpoint_hash = str(checkpoint.get("sha256", "")).lower()
    if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != checkpoint_hash:
        raise HeightScorecardError("Candidate checkpoint bytes do not match its report")
    protected_hash = str(scorecard["baseline"]["checkpoint_sha256"]).lower()
    if checkpoint_hash == protected_hash:
        distinct_candidate = False
    else:
        distinct_candidate = True

    protocol = str(scorecard["protocol"]["primary_height_protocol"])
    candidate_metrics = _report_model_protocol(candidate_report, candidate_model_key, protocol)
    baseline_metrics = _report_model_protocol(
        authenticated.baseline_report,
        str(scorecard["baseline"]["model_key"]),
        protocol,
    )
    gates = _mapping(scorecard.get("gates"), "gates")
    regression_checks, no_regression = _compare_no_regression(
        candidate_metrics,
        baseline_metrics,
        _mapping(gates.get("maximum_regression"), "maximum-regression gates"),
    )
    floor_checks, floors_pass = _compare_floor(
        candidate_metrics,
        baseline_metrics,
        [str(path) for path in gates.get("nondecreasing_metrics", [])],
    )
    absolute_checks, absolute_pass = _compare_absolute_no_regression(
        candidate_metrics,
        baseline_metrics,
        _mapping(
            gates.get("maximum_absolute_bias_regression"),
            "maximum absolute-bias regression gates",
        ),
    )
    improvement_checks, material_improvement = _material_improvements(
        candidate_metrics,
        baseline_metrics,
        _mapping(gates.get("material_improvement"), "material-improvement gate"),
    )
    development_passed = bool(
        distinct_candidate
        and no_regression
        and floors_pass
        and absolute_pass
        and material_improvement
    )

    external = _mapping(scorecard.get("external_height_evidence"), "external height evidence")
    external_passed = external.get("status") == "authenticated_passed"
    promotion_eligible = bool(development_passed and external_passed)
    reasons: list[str] = []
    if not distinct_candidate:
        reasons.append("candidate_is_the_protected_baseline")
    if not no_regression:
        reasons.append("same_protocol_regression_gate_failed")
    if not floors_pass:
        reasons.append("correlation_or_r2_decreased")
    if not absolute_pass:
        reasons.append("absolute_bias_regressed")
    if not material_improvement:
        reasons.append("no_material_primary_height_improvement")
    if not external_passed:
        reasons.append("genuinely_external_metric_height_evidence_missing")

    return {
        "schema": DECISION_SCHEMA,
        "scorecard": {
            "path": str(authenticated.path),
            "sha256": authenticated.sha256,
            "baseline_report_sha256": file_sha256(
                _resolve(scorecard["baseline"]["report"], root)
            ),
        },
        "candidate_report": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
            "model_key": candidate_model_key,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
        },
        "official_test_used": False,
        "same_full_scene_protocol_authenticated": True,
        "development_gate_passed": development_passed,
        "external_height_gate_passed": external_passed,
        "promotion_eligible": promotion_eligible,
        "promotion_performed": False,
        "live_application_changed": False,
        "reasons": reasons,
        "checks": {
            "distinct_candidate": distinct_candidate,
            "maximum_regression": regression_checks,
            "nondecreasing_metrics": floor_checks,
            "maximum_absolute_bias_regression": absolute_checks,
            "material_improvement": improvement_checks,
        },
    }


__all__ = [
    "AuthenticatedScorecard",
    "DECISION_SCHEMA",
    "HeightScorecardError",
    "REPORT_SCHEMA",
    "SCORECARD_SCHEMA",
    "assess_candidate",
    "authenticate_scorecard",
    "file_sha256",
]
