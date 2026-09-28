"""Fail-closed audit for the isolated hierarchical GAMUS V4 classifier.

The candidate may add only ``fine_semantic_head.*`` to the protected model.
All inherited tensors must be bit-identical, and the candidate is compared to
the frozen DC+Philadelphia spatial-V3 control recorded in its protocol.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "msr.gamus_hierarchical_v4_checkpoint_audit.v1"
PROTOCOL_SCHEMA = "msr.gamus_six_class_hierarchical_dcphl.v4"
HEIGHT_IDENTITY_SCHEMA = "msr.height_output_identity_audit.v1"
BASELINE_AUDIT_SCHEMA = (
    "msr.gamus_spatial_refined_geographic_checkpoint_audit.v1"
)
INDEPENDENT_REPLAY_SCHEMA = "msr.gamus_dcphl_paired_independent_replay.v1"
REPLAY_EVALUATOR_PATH = PROJECT_ROOT / "scripts" / "evaluate_gamus_dcphl_paired_replay.py"
ALLOWED_CITIES = ("DC", "PHL")
REPLAY_STORED_CONSISTENCY_TOLERANCE = 5.0e-5
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
HEAD_PREFIX = "fine_semantic_head."
EXPECTED_HEAD_KEYS = frozenset(
    {
        "fine_semantic_head.spatial.weight",
        "fine_semantic_head.normalization.weight",
        "fine_semantic_head.normalization.bias",
        "fine_semantic_head.coarse_classifier.weight",
        "fine_semantic_head.coarse_classifier.bias",
        "fine_semantic_head.vegetation_split.weight",
        "fine_semantic_head.vegetation_split.bias",
    }
)
CLASS_NAMES = (
    "ground",
    "buildings",
    "water",
    "roads",
    "low_vegetation",
    "trees",
)


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


def _load_replay_evaluator():
    spec = importlib.util.spec_from_file_location(
        "_msr_gamus_dcphl_replay", REPLAY_EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load paired replay evaluator: {REPLAY_EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_sha256(value: object, role: str) -> str:
    normalized = str(value).strip().lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{role} must be exactly 64 lowercase hexadecimal digits")
    return normalized


def _validate_file_identity(
    record: Mapping[str, Any], actual_path: Path, role: str
) -> dict[str, Any]:
    actual_path = actual_path.resolve()
    if not actual_path.is_file():
        raise FileNotFoundError(f"{role} does not exist: {actual_path}")
    if _resolve(str(record.get("path", ""))) != actual_path:
        raise ValueError(f"{role} path is not bound to the expected file")
    actual_sha256 = file_sha256(actual_path)
    if _require_sha256(record.get("sha256"), f"{role} SHA-256") != actual_sha256:
        raise ValueError(f"{role} SHA-256 differs from the current file")
    if "size_bytes" in record and int(record.get("size_bytes", -1)) != actual_path.stat().st_size:
        raise ValueError(f"{role} byte count differs from the current file")
    return {
        "path": str(actual_path),
        "sha256": actual_sha256,
        "size_bytes": actual_path.stat().st_size,
    }


def _metric_at_path(metrics: Mapping[str, Any], path: str) -> float:
    value: object = metrics
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"metric path {path!r} is missing at {part!r}")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric path {path!r} is not numeric")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"metric path {path!r} is non-finite")
    return result


def recompute_six_class_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all six-class scores from the 6x6 confusion matrix.

    This prevents a five-group or hand-edited macro score from satisfying the
    V4 contract without six-class predictions actually improving.
    """

    report = _mapping(
        metrics.get("six_class_identification"), "six-class identification"
    )
    expected_names = CLASS_NAMES
    if tuple(report.get("class_names", ())) != expected_names:
        raise ValueError("six-class report does not use the required class order")
    matrix = report.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 6
        or any(not isinstance(row, list) or len(row) != 6 for row in matrix)
    ):
        raise ValueError("six-class confusion matrix must be exactly 6x6")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for row in matrix
        for value in row
    ):
        raise ValueError("six-class confusion matrix entries must be nonnegative ints")

    stored_classes = _mapping(report.get("per_class"), "stored per-class scores")
    recomputed_classes: dict[str, dict[str, float]] = {}
    for index, name in enumerate(expected_names):
        true_positive = int(matrix[index][index])
        support = sum(int(value) for value in matrix[index])
        predicted = sum(int(matrix[row][index]) for row in range(6))
        false_positive = predicted - true_positive
        false_negative = support - true_positive
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        iou_denominator = true_positive + false_positive + false_negative
        iou = true_positive / iou_denominator if iou_denominator else 0.0
        recomputed = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }
        stored = _mapping(stored_classes.get(name), f"stored class {name}")
        for metric_name, expected in recomputed.items():
            actual = float(stored.get(metric_name, float("nan")))
            if abs(actual - expected) > 1.0e-12:
                raise ValueError(
                    f"stored {name} {metric_name} does not match the 6x6 matrix"
                )
        recomputed_classes[name] = recomputed

    macro = {
        metric_name: sum(
            values[metric_name] for values in recomputed_classes.values()
        )
        / 6.0
        for metric_name in ("precision", "recall", "f1", "iou")
    }
    stored_macro = _mapping(report.get("macro"), "stored six-class macro")
    for metric_name, expected in macro.items():
        candidates = [stored_macro.get(metric_name), report.get(f"macro_{metric_name}")]
        for actual in candidates:
            if actual is None or abs(float(actual) - expected) > 1.0e-12:
                raise ValueError(
                    f"stored six-class macro {metric_name} is not the mean of six classes"
                )
    return {"macro": macro, "per_class": recomputed_classes}


def _dark_false_water_rate(metrics: Mapping[str, Any]) -> float:
    for section_name in ("water_dark_pixel_proxy", "water_shadow_proxy"):
        section = metrics.get(section_name)
        if isinstance(section, Mapping):
            value = _metric_at_path(
                metrics,
                f"{section_name}.false_water_rate_on_dark_non_water",
            )
            if not 0.0 <= value <= 1.0:
                raise ValueError("dark false-water rate is outside [0, 1]")
            return value
    raise KeyError("metrics contain no water-dark/shadow diagnostic")


def derive_fixed_gates(baseline_metrics: Mapping[str, Any]) -> dict[str, float]:
    """Re-derive global gates from the independently replayed V3 metrics."""

    recompute_six_class_metrics(baseline_metrics)
    stored_six = _mapping(
        baseline_metrics.get("six_class_identification"), "verified six-class metrics"
    )
    per_class = _mapping(stored_six.get("per_class"), "verified class metrics")
    boundary_f1 = _metric_at_path(baseline_metrics, "road_boundary_quality.f1")
    dark_false_water = _dark_false_water_rate(baseline_metrics)
    for value, role in (
        (boundary_f1, "baseline road-boundary F1"),
        (dark_false_water, "baseline dark false-water rate"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{role} is outside [0, 1]")
    f1 = {
        name: float(_mapping(per_class.get(name), f"class {name}").get("f1"))
        for name in CLASS_NAMES
    }
    return {
        "six_class_macro_f1_min": max(0.60, float(stored_six["macro_f1"])),
        "ground_f1_min": max(0.40, f1["ground"] - 0.005),
        "buildings_f1_min": max(0.82, f1["buildings"] - 0.005),
        "water_f1_min": max(0.45, f1["water"] - 0.005),
        "roads_f1_min": max(0.65, f1["roads"] - 0.005),
        "low_vegetation_f1_min": max(0.58, f1["low_vegetation"] + 0.010),
        "trees_f1_min": max(0.62, f1["trees"] + 0.010),
        "road_boundary_f1_min": max(0.0, boundary_f1 - 0.005),
        "dark_non_water_false_water_rate_max": min(
            1.0, dark_false_water + 0.010
        ),
    }


def derive_per_city_gates(
    replay_by_city: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Freeze non-regression gates for each independently replayed city."""

    if set(replay_by_city) != set(ALLOWED_CITIES):
        raise ValueError("per-city replay must contain exactly DC and PHL")
    result: dict[str, dict[str, float]] = {}
    for city in ALLOWED_CITIES:
        metrics = _mapping(replay_by_city.get(city), f"{city} replay metrics")
        recompute_six_class_metrics(metrics)
        stored_six = _mapping(
            metrics.get("six_class_identification"), f"{city} six-class metrics"
        )
        macro = float(stored_six["macro_f1"])
        per_class = _mapping(stored_six.get("per_class"), f"{city} classes")
        boundary = _metric_at_path(metrics, "road_boundary_quality.f1")
        dark_false_water = _dark_false_water_rate(metrics)
        result[city] = {
            "six_class_macro_f1_min": max(0.0, macro - 0.005),
            **{
                f"{name}_f1_min": max(
                    0.0,
                    float(_mapping(per_class.get(name), f"{city} {name}")["f1"])
                    - 0.005,
                )
                for name in CLASS_NAMES
            },
            "road_boundary_f1_min": max(0.0, boundary - 0.005),
            "dark_non_water_false_water_rate_max": min(
                1.0, dark_false_water + 0.010
            ),
        }
    return result


def replay_vs_stored_consistency(
    replay_metrics: Mapping[str, Any], stored_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Diagnostic only: compare independent replay with checkpoint-stored metrics."""

    paths = {
        "six_class_macro_f1": "macro.f1",
        **{
            f"{name}_f1": f"per_class.{name}.f1" for name in CLASS_NAMES
        },
    }
    result: dict[str, Any] = {
        "tolerance": REPLAY_STORED_CONSISTENCY_TOLERANCE,
        "gating_role": "diagnostic_only",
    }
    try:
        replay_six = recompute_six_class_metrics(replay_metrics)
        stored_six = recompute_six_class_metrics(stored_metrics)
        checks: dict[str, Any] = {}
        for name, path in paths.items():
            replay_value = _metric_at_path(replay_six, path)
            stored_value = _metric_at_path(stored_six, path)
            delta = replay_value - stored_value
            checks[name] = {
                "independent_replay": replay_value,
                "checkpoint_stored": stored_value,
                "delta": delta,
                "within_tolerance": abs(delta)
                <= REPLAY_STORED_CONSISTENCY_TOLERANCE,
            }
        for name, replay_value, stored_value in (
            (
                "road_boundary_f1",
                _metric_at_path(replay_metrics, "road_boundary_quality.f1"),
                _metric_at_path(stored_metrics, "road_boundary_quality.f1"),
            ),
            (
                "dark_non_water_false_water_rate",
                _dark_false_water_rate(replay_metrics),
                _dark_false_water_rate(stored_metrics),
            ),
        ):
            delta = replay_value - stored_value
            checks[name] = {
                "independent_replay": replay_value,
                "checkpoint_stored": stored_value,
                "delta": delta,
                "within_tolerance": abs(delta)
                <= REPLAY_STORED_CONSISTENCY_TOLERANCE,
            }
        result["passes"] = all(
            bool(item["within_tolerance"]) for item in checks.values()
        )
        result["checks"] = checks
    except (KeyError, TypeError, ValueError) as error:
        result.update({"passes": False, "error": str(error), "checks": {}})
    return result


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
            or not torch.isfinite(torch.tensor(float(value)))
            or abs(float(value) - expected_value) > 1.0e-12
        ):
            raise ValueError(
                f"{role}.{key} must be fixed at {expected_value!r}, found {value!r}"
            )


def validate_candidate_config_identity(
    candidate_config_path: Path,
    *,
    expected_sha256: str,
    embedded_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Bind checkpoint metadata to the exact human-reviewed V4 YAML bytes."""

    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("expected candidate-config SHA-256 must be 64 hex digits")
    actual = file_sha256(candidate_config_path)
    if actual != expected:
        raise ValueError(
            "sealed V4 config SHA-256 mismatch: "
            f"expected {expected}, found {actual}"
        )
    sealed_config = _load_yaml(candidate_config_path)
    if dict(embedded_config) != sealed_config:
        raise ValueError(
            "checkpoint embedded config differs from the reviewed sealed V4 config"
        )
    return sealed_config, {"path": str(candidate_config_path), "sha256": actual}


def load_model_state(
    path: Path,
) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload = _mapping(payload, f"checkpoint {path}")
    raw_state = _mapping(payload.get("model"), f"model state in {path}")
    state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"invalid model-state entry {name!r} in {path}")
        state[name] = value
    if not state:
        raise ValueError(f"checkpoint contains an empty model state: {path}")
    return state, payload


def audit_states(
    base_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Prove that V4 is the protected model plus exactly one isolated head."""

    base_keys = set(base_state)
    candidate_keys = set(candidate_state)
    if any(name.startswith(HEAD_PREFIX) for name in base_keys):
        raise ValueError("protected checkpoint unexpectedly contains a fine head")
    missing = sorted(base_keys - candidate_keys)
    added = candidate_keys - base_keys
    if missing:
        raise ValueError(f"candidate is missing inherited tensors: {missing[:20]}")
    if added != EXPECTED_HEAD_KEYS:
        raise ValueError(
            "candidate must add exactly the hierarchical V4 head tensors; "
            f"found {sorted(added)}"
        )

    changed: list[str] = []
    inherited_elements = 0
    for name in sorted(base_keys):
        base = base_state[name]
        candidate = candidate_state[name]
        if base.shape != candidate.shape or base.dtype != candidate.dtype:
            changed.append(name)
            continue
        inherited_elements += base.numel()
        if not torch.equal(base, candidate):
            changed.append(name)
    if changed:
        raise ValueError(
            "protected height/shared tensors changed: " + ", ".join(changed[:20])
        )

    head: dict[str, Any] = {}
    for name in sorted(EXPECTED_HEAD_KEYS):
        tensor = candidate_state[name]
        if not torch.isfinite(tensor).all():
            raise ValueError(f"candidate head tensor is non-finite: {name}")
        head[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "l2_norm": float(torch.linalg.vector_norm(tensor.float()).item()),
        }
    expected_parameter_count = 1094
    parameter_count = sum(candidate_state[name].numel() for name in EXPECTED_HEAD_KEYS)
    if parameter_count != expected_parameter_count:
        raise ValueError(
            f"hierarchical V4 head has {parameter_count} parameters; "
            f"expected {expected_parameter_count}"
        )
    return {
        "passes": True,
        "inherited_tensor_count": len(base_keys),
        "inherited_element_count": inherited_elements,
        "inherited_equality": "torch.equal",
        "changed_inherited_tensors": [],
        "allowed_trainable_prefix": HEAD_PREFIX,
        "head_parameter_count": parameter_count,
        "head_tensors": head,
    }


def validate_recipe_contract(
    comparator: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_replay: Mapping[str, Any],
    *,
    comparator_sha256: str,
    baseline_audit_sha256: str,
    baseline_replay_sha256: str,
) -> dict[str, Any]:
    protocol = _mapping(candidate.get("protocol"), "V4 protocol")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unexpected V4 protocol schema")
    if protocol.get("source_comparator_config_sha256") != comparator_sha256:
        raise ValueError("V4 source-comparator hash is not pinned correctly")
    if protocol.get("fixed_v3_baseline_audit_sha256") != baseline_audit_sha256:
        raise ValueError("V4 fixed-baseline audit hash is not pinned correctly")
    if protocol.get("fixed_v3_independent_replay_sha256") != baseline_replay_sha256:
        raise ValueError("V4 fixed independent-replay hash is not pinned correctly")
    if protocol.get("independent_replay_schema") != INDEPENDENT_REPLAY_SCHEMA:
        raise ValueError("V4 does not pin the independent-replay schema")
    if protocol.get("independent_replay_evaluator_sha256") != file_sha256(
        REPLAY_EVALUATOR_PATH
    ):
        raise ValueError("V4 does not pin the independent-replay evaluator")
    baseline_contract = _mapping(
        baseline_replay.get("validation_contract"), "fixed replay contract"
    )
    if protocol.get("independent_replay_validation_contract_sha256") != baseline_contract.get(
        "contract_sha256"
    ):
        raise ValueError("V4 does not pin the independent validation contract")
    baseline_source = _mapping(
        baseline_replay.get("source_content_binding"), "fixed replay source binding"
    )
    if protocol.get("independent_replay_source_content_manifest_sha256") != baseline_source.get(
        "combined_content_manifest_sha256"
    ):
        raise ValueError("V4 does not pin the independent replay source content")
    if protocol.get("paired_independent_replay_required_before_holdout") is not True:
        raise ValueError("V4 must require paired independent replay before holdout")
    if protocol.get("official_test_policy") != (
        "previously_consumed_forbidden_for_reuse"
    ):
        raise ValueError(
            "V4 must record prior official-test exposure and forbid its reuse"
        )
    if protocol.get("auto_promotion") is not False:
        raise ValueError("V4 must disable automatic promotion")
    if protocol.get("nyc_interpretation") != (
        "excluded_from_this_classifier_training_not_system_unseen"
    ):
        raise ValueError("V4 must describe NYC exposure accurately")
    if protocol.get("external_system_unseen_geography") != "pending_separate_dataset":
        raise ValueError("V4 must retain an explicit external-geography blocker")
    height_contract_sha256 = protocol.get("height_output_contract_sha256")
    if (
        not isinstance(height_contract_sha256, str)
        or len(height_contract_sha256) != 64
        or any(character not in "0123456789abcdef" for character in height_contract_sha256)
    ):
        raise ValueError("V4 must pin the canonical height-output contract SHA-256")
    if protocol.get("height_output_identity_required_before_holdout") is not True:
        raise ValueError("V4 must require height-output identity before any holdout")

    if candidate.get("data") != comparator.get("data"):
        raise ValueError("V4 and spatial V3 must use the exact same DC+PHL data")
    comparator_model = deepcopy(dict(_mapping(comparator.get("model"), "V3 model")))
    candidate_model = deepcopy(dict(_mapping(candidate.get("model"), "V4 model")))
    if comparator_model.pop("fine_semantic_head_type", None) != "spatial_refined":
        raise ValueError("source comparator is not spatial-refined V3")
    if candidate_model.pop("fine_semantic_head_type", None) != "hierarchical_vegetation":
        raise ValueError("candidate is not hierarchical-vegetation V4")
    if candidate_model != comparator_model:
        raise ValueError("V4 changes model settings outside the isolated head type")

    comparator_training = deepcopy(
        dict(_mapping(comparator.get("training"), "V3 training"))
    )
    candidate_training = deepcopy(
        dict(_mapping(candidate.get("training"), "V4 training"))
    )
    # A fair architecture comparison keeps the complete V3 schedule fixed.
    # The auxiliary subtype loss is the only permitted training-field addition.
    allowed_training_changes = {"fine_semantic_vegetation_split_weight"}
    for key in allowed_training_changes:
        comparator_training.pop(key, None)
        candidate_training.pop(key, None)
    if candidate_training != comparator_training:
        raise ValueError("V4 training differs from V3 outside approved fields")
    if float(
        _mapping(candidate.get("training"), "V4 training").get(
            "fine_semantic_vegetation_split_weight", 0.0
        )
    ) != 0.5:
        raise ValueError("V4 vegetation-split auxiliary weight must be exactly 0.5")
    for schedule_field in (
        "epochs",
        "early_stopping_patience",
        "early_stopping_min_epochs",
    ):
        if candidate["training"].get(schedule_field) != comparator["training"].get(
            schedule_field
        ):
            raise ValueError(
                f"V4 must retain V3's {schedule_field} for a fair comparison"
            )

    groups = _mapping(candidate.get("training"), "V4 training").get(
        "parameter_groups"
    )
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("V4 requires exactly one optimizer parameter group")
    group = _mapping(groups[0], "V4 parameter group")
    if group.get("prefixes") != [HEAD_PREFIX]:
        raise ValueError("only fine_semantic_head may be trainable")

    baseline_v3 = _mapping(baseline_replay.get("v3"), "fixed replay V3")
    baseline_metrics = _mapping(baseline_v3.get("overall"), "fixed replay overall")
    baseline_by_city = _mapping(baseline_v3.get("by_city"), "fixed replay by city")
    acceptance = _mapping(protocol.get("classification_acceptance"), "V4 gates")
    _assert_exact_float_mapping(
        acceptance,
        derive_fixed_gates(baseline_metrics),
        "classification_acceptance",
    )
    city_acceptance = _mapping(
        protocol.get("classification_acceptance_by_city"), "V4 per-city gates"
    )
    expected_city_acceptance = derive_per_city_gates(baseline_by_city)
    if set(city_acceptance) != set(ALLOWED_CITIES):
        raise ValueError("V4 per-city gates must contain exactly DC and PHL")
    for city in ALLOWED_CITIES:
        _assert_exact_float_mapping(
            _mapping(city_acceptance.get(city), f"V4 {city} gates"),
            expected_city_acceptance[city],
            f"classification_acceptance_by_city.{city}",
        )
    evaluation = _mapping(candidate.get("evaluation"), "V4 evaluation")
    comparator_evaluation = _mapping(comparator.get("evaluation"), "V3 evaluation")
    selection = _mapping(evaluation.get("primary_selection"), "V4 selection")
    if dict(selection) != {
        "suite": "gamus",
        "metric": "six_class_identification.macro_f1",
        "mode": "max",
    }:
        raise ValueError("V4 must retain V3's six-class macro-F1 selection")
    # Training-time guards must remain identical to paired V3 so they do not
    # alter checkpoint selection or early stopping. The fixed class gates are
    # enforced afterward by this auditor and cannot authorize app promotion.
    if evaluation.get("validation_guards") != comparator_evaluation.get(
        "validation_guards"
    ):
        raise ValueError("V4 must retain V3's exact training-time height guards")
    return {
        "passes": True,
        "same_dc_phl_data_as_v3": True,
        "only_model_difference": (
            "fine_semantic_head_type=hierarchical_vegetation"
        ),
        "only_trainable_prefix": HEAD_PREFIX,
        "six_class_macro_definition": "final six classes; no merged vegetation",
        "classification_acceptance": "external_fail_closed_post_training_audit",
        "per_city_classification_acceptance": "DC_and_PHL_non_regression",
        "fixed_v3_independent_replay_sha256": baseline_replay_sha256,
        "nyc_interpretation": protocol["nyc_interpretation"],
        "external_system_unseen_geography": protocol[
            "external_system_unseen_geography"
        ],
        "height_output_contract_sha256": height_contract_sha256,
        "official_test_used_by_v4": False,
        "official_test_global_status": "previously_consumed",
        "auto_promotion": False,
    }


def validate_height_output_identity_report(
    report: Mapping[str, Any],
    *,
    base_path: Path,
    base_sha256: str,
    candidate_path: Path,
    candidate_sha256: str,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Bind the external replay report to this exact base and candidate."""

    if report.get("schema") != HEIGHT_IDENTITY_SCHEMA:
        raise ValueError("unexpected height-output identity schema")
    if report.get("passes") is not True:
        raise ValueError("height-output identity audit did not pass")

    protected = _mapping(report.get("protected"), "identity protected checkpoint")
    candidate = _mapping(report.get("candidate"), "identity candidate checkpoint")
    expected_pairs = (
        (protected.get("sha256_before"), base_sha256, "protected before hash"),
        (protected.get("sha256_after"), base_sha256, "protected after hash"),
        (candidate.get("sha256_before"), candidate_sha256, "candidate before hash"),
        (candidate.get("sha256_after"), candidate_sha256, "candidate after hash"),
    )
    for actual, expected, role in expected_pairs:
        if actual != expected:
            raise ValueError(f"height-output identity {role} is not bound to this run")
    if _resolve(str(protected.get("path"))) != base_path:
        raise ValueError("height-output identity report names a different protected file")
    if _resolve(str(candidate.get("path"))) != candidate_path:
        raise ValueError("height-output identity report names a different candidate file")

    required_pass_sections = (
        "provenance_audit",
        "state_audit",
        "pipeline_contract_audit",
        "loaded_model_audit",
        "direct_output_audit",
        "app_output_audit",
    )
    for section_name in required_pass_sections:
        section = _mapping(report.get(section_name), section_name)
        if section.get("passes") is not True:
            raise ValueError(f"height-output identity section failed: {section_name}")
    pipeline = _mapping(report.get("pipeline_contract_audit"), "pipeline contract")
    if pipeline.get("full_contract_sha256") != expected_contract_sha256:
        raise ValueError("height-output identity used an unpinned pipeline contract")
    application = _mapping(report.get("app_output_audit"), "app output audit")
    app_cases = application.get("cases")
    if not isinstance(app_cases, list) or len(app_cases) < 3:
        raise ValueError("height-output identity must cover at least three app cases")
    if report.get("checkpoint_files_unchanged_during_audit") is not True:
        raise ValueError("identity audit did not prove checkpoint files stayed unchanged")
    if report.get("promotion_performed") is not False:
        raise ValueError("identity audit unexpectedly reports promotion")
    return {
        "passes": True,
        "schema": HEIGHT_IDENTITY_SCHEMA,
        "protected_sha256": base_sha256,
        "candidate_sha256": candidate_sha256,
        "pipeline_contract_sha256": expected_contract_sha256,
        "app_case_count": len(app_cases),
    }


def validate_fixed_v3_baseline_audit(
    report: Mapping[str, Any], *, comparator_config_path: Path
) -> dict[str, Any]:
    """Reject an unsealed, failed, or holdout-consuming comparator report."""

    if report.get("schema") != BASELINE_AUDIT_SCHEMA:
        raise ValueError("unexpected fixed V3 baseline audit schema")
    for section_name in ("recipe_audit", "state_audit", "six_class_metric_audit"):
        section = _mapping(report.get(section_name), section_name)
        if section.get("passes") is not True:
            raise ValueError(f"fixed V3 baseline section failed: {section_name}")
    config_record = _mapping(report.get("candidate_config"), "baseline config")
    comparator_sha256 = file_sha256(comparator_config_path)
    if config_record.get("sha256") != comparator_sha256:
        raise ValueError("fixed V3 baseline is not bound to the comparator config")
    candidate_record = _mapping(
        report.get("candidate"), "baseline candidate checkpoint"
    )
    candidate_path = _resolve(str(candidate_record.get("path")))
    if not candidate_path.is_file():
        raise FileNotFoundError("fixed V3 baseline checkpoint no longer exists")
    candidate_sha256 = file_sha256(candidate_path)
    if candidate_record.get("sha256") != candidate_sha256:
        raise ValueError("fixed V3 baseline checkpoint hash no longer matches")
    if report.get("nyc_evaluation_performed") is not False:
        raise ValueError("fixed V3 baseline consumed the NYC classifier holdout")
    if report.get("official_test_used_by_this_comparator") is not False:
        raise ValueError("fixed V3 baseline reused the official GAMUS test")
    if report.get("official_test_global_status") != (
        "previously_consumed_forbidden_for_reuse"
    ):
        raise ValueError("fixed V3 baseline hides prior official-test exposure")
    if report.get("promotion_performed") is not False:
        raise ValueError("fixed V3 baseline unexpectedly reports promotion")
    pointer_verification = _mapping(
        report.get("app_pointer_verification"), "baseline app pointer"
    )
    if pointer_verification.get("passes") is not True:
        raise ValueError("fixed V3 baseline did not verify the protected app pointer")
    return {
        "passes": True,
        "schema": BASELINE_AUDIT_SCHEMA,
        "comparator_config_sha256": comparator_sha256,
        "checkpoint_path": str(candidate_path),
        "checkpoint_sha256": candidate_sha256,
        "nyc_evaluation_performed": False,
        "official_test_used_by_this_comparator": False,
        "official_test_global_status": "previously_consumed_forbidden_for_reuse",
    }


def _validate_replay_report_common(
    report: Mapping[str, Any], *, expected_mode: str
) -> Any:
    if report.get("schema") != INDEPENDENT_REPLAY_SCHEMA:
        raise ValueError("unexpected independent replay schema")
    if report.get("mode") != expected_mode:
        raise ValueError(f"independent replay mode must be {expected_mode}")
    if report.get("passes") is not True:
        raise ValueError("independent replay did not pass")
    if report.get("interpretation") != (
        "Independent pixel replay on the fixed DC+Philadelphia development "
        "validation partition; this is not NYC, official GAMUS test, or a "
        "genuinely system-unseen geography."
    ):
        raise ValueError("independent replay interpretation is missing or misleading")
    access = _mapping(report.get("access_policy"), "replay access policy")
    if access.get("split_constructed") != "val":
        raise ValueError("independent replay must use only the validation split")
    if list(access.get("cities_opened", ())) != list(ALLOWED_CITIES):
        raise ValueError("independent replay must open exactly DC and PHL")
    for key in ("nyc_opened", "official_test_opened_or_reused", "promotion_performed"):
        if access.get(key) is not False:
            raise ValueError(f"independent replay access policy violates {key}")

    implementation = _mapping(
        report.get("implementation_sha256"), "replay implementation hashes"
    )
    before = _mapping(implementation.get("before"), "implementation before hashes")
    after = _mapping(implementation.get("after"), "implementation after hashes")
    if implementation.get("unchanged_during_replay") is not True or before != after:
        raise ValueError("replay implementation was not immutable")
    evaluator_sha256 = file_sha256(REPLAY_EVALUATOR_PATH)
    if _require_sha256(before.get("evaluator"), "replay evaluator SHA-256") != evaluator_sha256:
        raise ValueError("replay report is not bound to the current evaluator")
    for role, value in before.items():
        _require_sha256(value, f"replay implementation {role} SHA-256")

    pointer = _mapping(report.get("protected_pointer"), "replay protected pointer")
    if pointer.get("unchanged") is not True:
        raise ValueError("protected application pointer changed during replay")
    if _mapping(pointer.get("before"), "pointer before") != _mapping(
        pointer.get("after"), "pointer after"
    ):
        raise ValueError("protected pointer before/after records differ")

    runtime = _mapping(report.get("runtime"), "replay runtime")
    if runtime.get("device") != "cuda":
        raise ValueError("independent paired replay must run on CUDA")
    if runtime.get("precision") not in {"bf16", "fp16", "fp32"}:
        raise ValueError("independent replay precision is invalid")
    if int(runtime.get("batch_size", 0)) <= 0 or int(runtime.get("num_workers", -1)) < 0:
        raise ValueError("independent replay runtime counts are invalid")

    source = _mapping(report.get("source_content_binding"), "source binding")
    if source.get("unchanged_during_replay") is not True:
        raise ValueError("replay source content was not immutable")
    sample_count = int(source.get("sample_count", -1))
    total_file_count = int(source.get("total_file_count", -1))
    total_bytes = int(source.get("total_bytes", -1))
    if sample_count <= 0 or total_file_count <= 0 or total_bytes <= 0:
        raise ValueError("replay source counts must be positive")
    per_role = _mapping(source.get("per_role"), "source per-role binding")
    expected_roles = {"image", "height", "class", "relative_prior"}
    if set(per_role) != expected_roles:
        raise ValueError("source binding must contain exactly four input roles")
    role_files = 0
    role_bytes = 0
    for role in sorted(expected_roles):
        item = _mapping(per_role.get(role), f"source role {role}")
        count = int(item.get("file_count", -1))
        size = int(item.get("total_bytes", -1))
        if count != sample_count or size <= 0:
            raise ValueError(f"source role {role} has inconsistent counts")
        _require_sha256(item.get("content_manifest_sha256"), f"{role} content hash")
        role_files += count
        role_bytes += size
    if total_file_count != role_files or total_bytes != role_bytes:
        raise ValueError("source total counts do not equal the per-role sums")
    if total_file_count != sample_count * len(expected_roles):
        raise ValueError("source file count does not equal four files per sample")
    _require_sha256(
        source.get("combined_content_manifest_sha256"), "combined source hash"
    )
    metadata_before = _mapping(source.get("metadata_before"), "source metadata before")
    metadata_after = _mapping(source.get("metadata_after"), "source metadata after")
    if metadata_before != metadata_after:
        raise ValueError("source metadata changed during replay")
    if int(metadata_before.get("file_count", -1)) != total_file_count:
        raise ValueError("source metadata file count is inconsistent")
    _require_sha256(
        metadata_before.get("metadata_manifest_sha256"), "source metadata hash"
    )

    evaluator = _load_replay_evaluator()
    return evaluator


def _validate_replay_result_binding(
    result: Mapping[str, Any], contract: Mapping[str, Any], evaluator: Any, role: str
) -> None:
    evaluator.validate_replay_result(result)
    scopes = [
        ("overall", _mapping(result.get("overall"), f"{role} overall")),
        *[
            (
                city,
                _mapping(
                    _mapping(result.get("by_city"), f"{role} by city").get(city),
                    f"{role} {city}",
                ),
            )
            for city in ALLOWED_CITIES
        ],
    ]
    for scope_name, metrics in scopes:
        per_class = _mapping(
            _mapping(
                metrics.get("six_class_identification"), f"{role} {scope_name} six-class"
            ).get("per_class"),
            f"{role} {scope_name} classes",
        )
        for class_name in CLASS_NAMES:
            if int(
                _mapping(per_class.get(class_name), f"{role} {scope_name} {class_name}").get(
                    "support_pixels", 0
                )
            ) <= 0:
                raise ValueError(
                    f"{role} {scope_name} has no support for gated class {class_name}"
                )
    validation_count = int(contract.get("validation_count", -1))
    if int(result.get("evaluated_sample_count", -1)) != validation_count:
        raise ValueError(f"{role} replay count differs from the validation contract")
    if result.get("evaluated_ids_sha256") != contract.get("validation_ids_sha256"):
        raise ValueError(f"{role} replay IDs differ from the validation contract")
    if dict(_mapping(result.get("evaluated_samples_by_city"), f"{role} city counts")) != dict(
        _mapping(contract.get("city_counts"), "validation city counts")
    ):
        raise ValueError(f"{role} replay city counts differ from the validation contract")


def validate_fixed_v3_independent_replay(
    report: Mapping[str, Any],
    *,
    comparator_config_path: Path,
    baseline_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the independent V3 DC+PHL replay used to freeze V4 gates."""

    evaluator = _validate_replay_report_common(report, expected_mode="baseline_v3_only")
    if "v4" in report or "v4_minus_v3" in report:
        raise ValueError("fixed V3 replay must not contain V4 results")
    comparator_config_path = comparator_config_path.resolve()
    comparator_sha256 = file_sha256(comparator_config_path)
    pointer_record = _mapping(
        _mapping(report.get("protected_pointer"), "replay pointer").get("before"),
        "replay pointer before",
    )
    baseline_pointer = _mapping(
        baseline_audit.get("app_pointer_verification"), "baseline pointer verification"
    )
    if (
        _resolve(str(pointer_record.get("path", "")))
        != _resolve(str(baseline_pointer.get("path", "")))
        or pointer_record.get("sha256") != baseline_pointer.get("sha256")
        or _resolve(str(pointer_record.get("target", "")))
        != _resolve(str(baseline_pointer.get("target", "")))
    ):
        raise ValueError("independent replay pointer differs from the sealed V3 audit")
    pointer_path = _resolve(str(pointer_record.get("path", "")))
    _validate_file_identity(pointer_record, pointer_path, "replay protected pointer")
    base_record = _mapping(baseline_audit.get("base"), "baseline protected model")
    if pointer_record.get("target_sha256") != base_record.get("sha256"):
        raise ValueError("independent replay pointer target hash is not the protected model")
    config_auth = evaluator.authenticate_replay_config(
        comparator_config_path,
        expected_sha256=comparator_sha256,
        expected_protocol=evaluator.V3_PROTOCOL,
        expected_head_type="spatial_refined",
    )
    sealed = _mapping(report.get("sealed_artifacts"), "replay sealed artifacts")
    if sealed.get("unchanged") is not True:
        raise ValueError("V3 replay sealed artifacts changed")
    _validate_file_identity(
        _mapping(sealed.get("v3_config"), "replay V3 config"),
        comparator_config_path,
        "replay V3 config",
    )
    baseline_candidate = _mapping(
        baseline_audit.get("candidate"), "baseline-audit V3 checkpoint"
    )
    baseline_checkpoint = _resolve(str(baseline_candidate.get("path", "")))
    if _require_sha256(
        baseline_candidate.get("sha256"), "baseline-audit checkpoint SHA-256"
    ) != file_sha256(baseline_checkpoint):
        raise ValueError("baseline-audit V3 checkpoint hash no longer matches")
    checkpoint_identity = _validate_file_identity(
        _mapping(sealed.get("v3_checkpoint"), "replay V3 checkpoint"),
        baseline_checkpoint,
        "replay V3 checkpoint",
    )
    authenticated_checkpoint = evaluator.authenticate_checkpoint(
        baseline_checkpoint,
        expected_sha256=checkpoint_identity["sha256"],
        external_config=config_auth["config"],
    )
    checkpoint_record = _mapping(sealed.get("v3_checkpoint"), "replay V3 checkpoint")
    for key in ("epoch", "model_type", "embedded_config_sha256"):
        if checkpoint_record.get(key) != authenticated_checkpoint.get(key):
            raise ValueError(f"V3 replay checkpoint {key} is not bound to its payload")
    after_sha = _mapping(sealed.get("after_sha256"), "replay after hashes")
    expected_after = {
        "approved_index_sha256": config_auth["approved_index_identity"]["sha256"],
        "v3_checkpoint_sha256": checkpoint_identity["sha256"],
        "v3_config_sha256": comparator_sha256,
    }
    if dict(after_sha) != expected_after:
        raise ValueError("V3 replay after-hash seal is incomplete or inconsistent")

    approved_index_path = _resolve(
        str(_mapping(config_auth["config"].get("data"), "V3 data").get("approved_index_path"))
    )
    _validate_file_identity(
        _mapping(report.get("approved_index"), "replay approved index"),
        approved_index_path,
        "replay approved index",
    )
    contract = _mapping(report.get("validation_contract"), "replay validation contract")
    if dict(contract) != dict(config_auth["validation_contract"]):
        raise ValueError("V3 replay validation contract differs from the sealed config")
    v3 = _mapping(report.get("v3"), "independently replayed V3 result")
    _validate_replay_result_binding(v3, contract, evaluator, "V3")
    source = _mapping(report.get("source_content_binding"), "source binding")
    if int(source.get("sample_count", -1)) != int(v3["evaluated_sample_count"]):
        raise ValueError("V3 replay source sample count is inconsistent")
    if v3.get("fine_semantic_head_type") != "spatial_refined":
        raise ValueError("fixed independent replay is not the spatial V3 head")
    if int(v3.get("checkpoint_epoch", -1)) != int(
        _mapping(sealed.get("v3_checkpoint"), "V3 checkpoint").get("epoch", -2)
    ):
        raise ValueError("V3 replay epoch differs from its checkpoint record")

    stored_metrics = _mapping(
        baseline_audit.get("candidate_metrics"), "baseline-audit stored metrics"
    )
    consistency = replay_vs_stored_consistency(
        _mapping(v3.get("overall"), "V3 replay overall metrics"), stored_metrics
    )
    if consistency.get("passes") is not True:
        raise ValueError("independent V3 replay is inconsistent with the baseline audit")
    return {
        "passes": True,
        "schema": INDEPENDENT_REPLAY_SCHEMA,
        "mode": "baseline_v3_only",
        "v3_checkpoint": checkpoint_identity,
        "v3_config_sha256": comparator_sha256,
        "approved_index_sha256": config_auth["approved_index_identity"]["sha256"],
        "validation_contract": dict(contract),
        "stored_metric_consistency": consistency,
    }


def validate_paired_independent_replay(
    report: Mapping[str, Any],
    baseline_report: Mapping[str, Any],
    *,
    report_path: Path,
    expected_report_sha256: str,
    comparator_config_path: Path,
    candidate_config_path: Path,
    baseline_checkpoint_path: Path,
    candidate_checkpoint_path: Path,
) -> dict[str, Any]:
    """Authenticate the post-V4 replay and return its independent metrics."""

    actual_report_sha256 = file_sha256(report_path)
    if actual_report_sha256 != _require_sha256(
        expected_report_sha256, "expected paired-replay SHA-256"
    ):
        raise ValueError("paired independent replay SHA-256 mismatch")
    evaluator = _validate_replay_report_common(report, expected_mode="paired_v3_v4")
    paired_implementation = _mapping(
        _mapping(report.get("implementation_sha256"), "paired implementation").get(
            "before"
        ),
        "paired implementation before",
    )
    baseline_implementation = _mapping(
        _mapping(
            baseline_report.get("implementation_sha256"), "baseline implementation"
        ).get("before"),
        "baseline implementation before",
    )
    if paired_implementation != baseline_implementation:
        raise ValueError("paired replay implementation differs from fixed V3 replay")
    if _mapping(report.get("runtime"), "paired replay runtime") != _mapping(
        baseline_report.get("runtime"), "fixed replay runtime"
    ):
        raise ValueError("paired replay runtime differs from the fixed V3 replay")
    comparator_config_path = comparator_config_path.resolve()
    candidate_config_path = candidate_config_path.resolve()
    v3_auth = evaluator.authenticate_replay_config(
        comparator_config_path,
        expected_sha256=file_sha256(comparator_config_path),
        expected_protocol=evaluator.V3_PROTOCOL,
        expected_head_type="spatial_refined",
    )
    v4_auth = evaluator.authenticate_replay_config(
        candidate_config_path,
        expected_sha256=file_sha256(candidate_config_path),
        expected_protocol=evaluator.V4_PROTOCOL,
        expected_head_type="hierarchical_vegetation",
    )
    evaluator.assert_paired_contracts(v3_auth, v4_auth)
    contract = _mapping(report.get("validation_contract"), "paired validation contract")
    if dict(contract) != dict(v3_auth["validation_contract"]):
        raise ValueError("paired replay validation contract differs from sealed configs")
    baseline_contract = _mapping(
        baseline_report.get("validation_contract"), "baseline replay contract"
    )
    if dict(contract) != dict(baseline_contract):
        raise ValueError("paired replay data contract differs from the fixed V3 replay")

    sealed = _mapping(report.get("sealed_artifacts"), "paired sealed artifacts")
    if sealed.get("unchanged") is not True:
        raise ValueError("paired replay sealed artifacts changed")
    identities = {
        "v3_config": _validate_file_identity(
            _mapping(sealed.get("v3_config"), "paired V3 config"),
            comparator_config_path,
            "paired V3 config",
        ),
        "v3_checkpoint": _validate_file_identity(
            _mapping(sealed.get("v3_checkpoint"), "paired V3 checkpoint"),
            baseline_checkpoint_path,
            "paired V3 checkpoint",
        ),
        "v4_config": _validate_file_identity(
            _mapping(sealed.get("v4_config"), "paired V4 config"),
            candidate_config_path,
            "paired V4 config",
        ),
        "v4_checkpoint": _validate_file_identity(
            _mapping(sealed.get("v4_checkpoint"), "paired V4 checkpoint"),
            candidate_checkpoint_path,
            "paired V4 checkpoint",
        ),
    }
    authenticated_v3_checkpoint = evaluator.authenticate_checkpoint(
        baseline_checkpoint_path,
        expected_sha256=identities["v3_checkpoint"]["sha256"],
        external_config=v3_auth["config"],
    )
    authenticated_v4_checkpoint = evaluator.authenticate_checkpoint(
        candidate_checkpoint_path,
        expected_sha256=identities["v4_checkpoint"]["sha256"],
        external_config=v4_auth["config"],
    )
    for name, authenticated in (
        ("v3_checkpoint", authenticated_v3_checkpoint),
        ("v4_checkpoint", authenticated_v4_checkpoint),
    ):
        record = _mapping(sealed.get(name), f"paired {name}")
        for key in ("epoch", "model_type", "embedded_config_sha256"):
            if record.get(key) != authenticated.get(key):
                raise ValueError(f"paired {name} {key} is not bound to its checkpoint")
    expected_after = {
        "approved_index_sha256": v3_auth["approved_index_identity"]["sha256"],
        **{f"{name}_sha256": identity["sha256"] for name, identity in identities.items()},
    }
    if dict(_mapping(sealed.get("after_sha256"), "paired after hashes")) != expected_after:
        raise ValueError("paired replay after-hash seal is incomplete or inconsistent")
    approved_index_path = _resolve(
        str(_mapping(v3_auth["config"].get("data"), "V3 data").get("approved_index_path"))
    )
    _validate_file_identity(
        _mapping(report.get("approved_index"), "paired approved index"),
        approved_index_path,
        "paired approved index",
    )
    if _mapping(report.get("approved_index"), "paired approved index") != _mapping(
        baseline_report.get("approved_index"), "baseline approved index"
    ):
        raise ValueError("paired replay approved-index identity differs from baseline")
    paired_source = _mapping(report.get("source_content_binding"), "paired source binding")
    baseline_source = _mapping(
        baseline_report.get("source_content_binding"), "baseline source binding"
    )
    stable_source_fields = (
        "sample_count",
        "total_file_count",
        "total_bytes",
        "combined_content_manifest_sha256",
        "per_role",
    )
    if any(paired_source.get(key) != baseline_source.get(key) for key in stable_source_fields):
        raise ValueError("paired replay source-content binding differs from baseline")

    v3 = _mapping(report.get("v3"), "paired replay V3 result")
    v4 = _mapping(report.get("v4"), "paired replay V4 result")
    _validate_replay_result_binding(v3, contract, evaluator, "V3")
    _validate_replay_result_binding(v4, contract, evaluator, "V4")
    if dict(v3) != dict(_mapping(baseline_report.get("v3"), "fixed replay V3 result")):
        raise ValueError("paired replay V3 result differs from the fixed V3 replay")
    if v4.get("fine_semantic_head_type") != "hierarchical_vegetation":
        raise ValueError("paired replay V4 result is not hierarchical")
    for role, result, authenticated, expected_head in (
        ("V3", v3, authenticated_v3_checkpoint, "spatial_refined"),
        ("V4", v4, authenticated_v4_checkpoint, "hierarchical_vegetation"),
    ):
        if result.get("fine_semantic_head_type") != expected_head:
            raise ValueError(f"paired replay {role} head type is incorrect")
        if result.get("checkpoint_epoch") != authenticated.get("epoch"):
            raise ValueError(f"paired replay {role} epoch differs from its checkpoint")
        if result.get("model_type") != authenticated.get("model_type"):
            raise ValueError(f"paired replay {role} model type differs from its checkpoint")
    if v3.get("evaluated_ids_sha256") != v4.get("evaluated_ids_sha256") or _mapping(
        v3.get("evaluated_ids_by_city_sha256"), "V3 city ID hashes"
    ) != _mapping(v4.get("evaluated_ids_by_city_sha256"), "V4 city ID hashes"):
        raise ValueError("paired V3/V4 replay IDs differ")
    expected_deltas = evaluator.paired_deltas(v3, v4)
    if _mapping(report.get("v4_minus_v3"), "paired replay deltas") != expected_deltas:
        raise ValueError("paired replay delta formulas are inconsistent")
    return {
        "passes": True,
        "path": str(report_path),
        "sha256": actual_report_sha256,
        "validation_contract_sha256": contract.get("contract_sha256"),
        "evaluated_sample_count": int(v4["evaluated_sample_count"]),
        "evaluated_ids_sha256": v4.get("evaluated_ids_sha256"),
        "v3": v3,
        "v4": v4,
        "identities": identities,
    }


def verify_protected_application_state(
    protocol: Mapping[str, Any], *, base_path: Path
) -> dict[str, Any]:
    """Verify both the immutable production checkpoint and live pointer."""

    base_sha256 = file_sha256(base_path)
    if protocol.get("protected_checkpoint_sha256") != base_sha256:
        raise ValueError("V4 protocol does not pin the supplied protected checkpoint")
    pointer_path = _resolve(str(protocol.get("live_pointer_file")))
    if not pointer_path.is_file():
        raise FileNotFoundError("live application pointer does not exist")
    pointer_sha256 = file_sha256(pointer_path)
    if protocol.get("live_pointer_file_sha256") != pointer_sha256:
        raise ValueError("live application pointer hash changed")
    pointer_target = _resolve(pointer_path.read_text(encoding="utf-8").lstrip("\ufeff").strip())
    if pointer_target != base_path:
        raise ValueError("live application pointer no longer targets the protected model")
    return {
        "passes": True,
        "protected_checkpoint_path": str(base_path),
        "protected_checkpoint_sha256": base_sha256,
        "live_pointer_path": str(pointer_path),
        "live_pointer_sha256": pointer_sha256,
        "live_pointer_target": str(pointer_target),
    }


def classification_acceptance(
    metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    current_six = recompute_six_class_metrics(metrics)
    baseline_six = recompute_six_class_metrics(baseline_metrics)
    checks = {
        "six_class_macro_f1": (
            "macro.f1",
            "six_class_macro_f1_min",
            "min",
        ),
        "ground_f1": (
            "per_class.ground.f1",
            "ground_f1_min",
            "min",
        ),
        "buildings_f1": (
            "per_class.buildings.f1",
            "buildings_f1_min",
            "min",
        ),
        "water_f1": (
            "per_class.water.f1",
            "water_f1_min",
            "min",
        ),
        "roads_f1": (
            "per_class.roads.f1",
            "roads_f1_min",
            "min",
        ),
        "low_vegetation_f1": (
            "per_class.low_vegetation.f1",
            "low_vegetation_f1_min",
            "min",
        ),
        "trees_f1": (
            "per_class.trees.f1",
            "trees_f1_min",
            "min",
        ),
        "road_boundary_f1": (
            "road_boundary_quality.f1",
            "road_boundary_f1_min",
            "min",
        ),
        "dark_non_water_false_water_rate": (
            "__dark_false_water__",
            "dark_non_water_false_water_rate_max",
            "max",
        ),
    }
    results: dict[str, Any] = {}
    for name, (path, threshold_name, comparison) in checks.items():
        metric_source = (
            metrics if name in {"road_boundary_f1", "dark_non_water_false_water_rate"}
            else current_six
        )
        baseline_source = (
            baseline_metrics
            if name in {"road_boundary_f1", "dark_non_water_false_water_rate"}
            else baseline_six
        )
        value = (
            _dark_false_water_rate(metric_source)
            if path == "__dark_false_water__"
            else _metric_at_path(metric_source, path)
        )
        threshold = float(thresholds[threshold_name])
        baseline = (
            _dark_false_water_rate(baseline_source)
            if path == "__dark_false_water__"
            else _metric_at_path(baseline_source, path)
        )
        passed = value >= threshold if comparison == "min" else value <= threshold
        results[name] = {
            "value": value,
            "v3_dc_phl_value": baseline,
            "delta": value - baseline,
            comparison: threshold,
            "passes": bool(passed),
        }
    return {
        "passes": all(bool(item["passes"]) for item in results.values()),
        "checks": results,
    }


def per_city_classification_acceptance(
    current_by_city: Mapping[str, Any],
    baseline_by_city: Mapping[str, Any],
    thresholds_by_city: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        set(current_by_city) != set(ALLOWED_CITIES)
        or set(baseline_by_city) != set(ALLOWED_CITIES)
        or set(thresholds_by_city) != set(ALLOWED_CITIES)
    ):
        raise ValueError("per-city acceptance requires exactly DC and PHL")
    cities = {
        city: classification_acceptance(
            _mapping(current_by_city.get(city), f"current {city} metrics"),
            _mapping(baseline_by_city.get(city), f"baseline {city} metrics"),
            _mapping(thresholds_by_city.get(city), f"{city} thresholds"),
        )
        for city in ALLOWED_CITIES
    }
    return {
        "passes": all(bool(result["passes"]) for result in cities.values()),
        "cities": cities,
    }


def build_report(
    base_path: Path,
    candidate_path: Path,
    candidate_config_path: Path,
    comparator_config_path: Path,
    baseline_audit_path: Path,
    baseline_replay_path: Path,
    paired_replay_path: Path,
    output_identity_path: Path | None = None,
    *,
    expected_candidate_config_sha256: str,
    expected_paired_replay_sha256: str,
) -> dict[str, Any]:
    candidate_state, candidate_payload = load_model_state(candidate_path)
    base_state, base_payload = load_model_state(base_path)
    embedded_config = _mapping(
        candidate_payload.get("config"), "candidate embedded config"
    )
    candidate_config, candidate_config_identity = validate_candidate_config_identity(
        candidate_config_path,
        expected_sha256=expected_candidate_config_sha256,
        embedded_config=embedded_config,
    )
    comparator_config = _load_yaml(comparator_config_path)
    baseline_audit = _load_json(baseline_audit_path)
    baseline_replay = _load_json(baseline_replay_path)
    paired_replay = _load_json(paired_replay_path)
    baseline_seal = validate_fixed_v3_baseline_audit(
        baseline_audit, comparator_config_path=comparator_config_path
    )
    baseline_replay_seal = validate_fixed_v3_independent_replay(
        baseline_replay,
        comparator_config_path=comparator_config_path,
        baseline_audit=baseline_audit,
    )
    candidate_metrics = _mapping(
        candidate_payload.get("metrics"), "candidate checkpoint metrics"
    )
    protocol = _mapping(candidate_config.get("protocol"), "candidate protocol")
    if _resolve(str(protocol.get("fixed_v3_independent_replay", ""))) != baseline_replay_path:
        raise ValueError("V4 protocol names a different fixed independent replay")
    if protocol.get("fixed_v3_independent_replay_sha256") != file_sha256(
        baseline_replay_path
    ):
        raise ValueError("V4 protocol fixed independent-replay hash differs")
    protected_application_state = verify_protected_application_state(
        protocol, base_path=base_path
    )
    state_audit = audit_states(base_state, candidate_state)
    recipe_audit = validate_recipe_contract(
        comparator_config,
        candidate_config,
        baseline_replay,
        comparator_sha256=file_sha256(comparator_config_path),
        baseline_audit_sha256=file_sha256(baseline_audit_path),
        baseline_replay_sha256=file_sha256(baseline_replay_path),
    )
    baseline_checkpoint_path = _resolve(
        str(_mapping(baseline_audit.get("candidate"), "V3 checkpoint").get("path"))
    )
    paired_replay_seal = validate_paired_independent_replay(
        paired_replay,
        baseline_replay,
        report_path=paired_replay_path,
        expected_report_sha256=expected_paired_replay_sha256,
        comparator_config_path=comparator_config_path,
        candidate_config_path=candidate_config_path,
        baseline_checkpoint_path=baseline_checkpoint_path,
        candidate_checkpoint_path=candidate_path,
    )
    replayed_v3 = _mapping(paired_replay_seal.get("v3"), "paired replay V3")
    replayed_v4 = _mapping(paired_replay_seal.get("v4"), "paired replay V4")
    baseline_metrics = _mapping(replayed_v3.get("overall"), "replayed V3 overall")
    independent_candidate_metrics = _mapping(
        replayed_v4.get("overall"), "replayed V4 overall"
    )
    acceptance = classification_acceptance(
        independent_candidate_metrics,
        baseline_metrics,
        _mapping(protocol.get("classification_acceptance"), "V4 gates"),
    )
    per_city_acceptance = per_city_classification_acceptance(
        _mapping(replayed_v4.get("by_city"), "replayed V4 by city"),
        _mapping(replayed_v3.get("by_city"), "replayed V3 by city"),
        _mapping(
            protocol.get("classification_acceptance_by_city"), "V4 per-city gates"
        ),
    )
    stored_metric_diagnostic = replay_vs_stored_consistency(
        independent_candidate_metrics, candidate_metrics
    )
    output_identity: dict[str, Any] = {
        "status": "required_external_audit",
        "passes": False,
    }
    if output_identity_path is not None:
        loaded_identity = _load_json(output_identity_path)
        base_sha256 = file_sha256(base_path)
        candidate_sha256 = file_sha256(candidate_path)
        identity_validation = validate_height_output_identity_report(
            loaded_identity,
            base_path=base_path,
            base_sha256=base_sha256,
            candidate_path=candidate_path,
            candidate_sha256=candidate_sha256,
            expected_contract_sha256=str(
                protocol["height_output_contract_sha256"]
            ),
        )
        output_identity = {
            "path": str(output_identity_path),
            "sha256": file_sha256(output_identity_path),
            **identity_validation,
        }
    fully_eligible = bool(
        state_audit["passes"]
        and recipe_audit["passes"]
        and paired_replay_seal["passes"]
        and acceptance["passes"]
        and per_city_acceptance["passes"]
        and output_identity.get("passes") is True
    )
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base": {
            "path": str(base_path),
            "sha256": file_sha256(base_path),
            "epoch": base_payload.get("epoch"),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
            "epoch": candidate_payload.get("epoch"),
        },
        "candidate_config": candidate_config_identity,
        "fixed_v3_dc_phl_baseline": {
            "audit_path": str(baseline_audit_path),
            "audit_sha256": file_sha256(baseline_audit_path),
            "checkpoint": baseline_audit.get("candidate"),
            "seal": baseline_seal,
        },
        "fixed_v3_independent_replay": {
            "path": str(baseline_replay_path),
            "sha256": file_sha256(baseline_replay_path),
            "seal": baseline_replay_seal,
        },
        "paired_independent_replay": {
            "path": str(paired_replay_path),
            "sha256": file_sha256(paired_replay_path),
            "seal": {
                key: value
                for key, value in paired_replay_seal.items()
                if key not in {"v3", "v4"}
            },
        },
        "recipe_audit": recipe_audit,
        "state_audit": state_audit,
        "protected_application_state": protected_application_state,
        "height_output_identity": output_identity,
        "classification_acceptance": acceptance,
        "classification_acceptance_by_city": per_city_acceptance,
        "independent_replay_candidate_metrics": independent_candidate_metrics,
        "checkpoint_stored_metrics": candidate_metrics,
        "checkpoint_stored_vs_independent_replay_diagnostic": stored_metric_diagnostic,
        "acceptance_metric_authority": "paired_independent_replay",
        "fully_eligible_for_locked_classifier_excluded_city_evaluation": fully_eligible,
        "application_model_changed": False,
        "promotion_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--expected-candidate-config-sha256", required=True)
    parser.add_argument("--source-comparator-config", required=True)
    parser.add_argument("--fixed-v3-baseline-audit", required=True)
    parser.add_argument("--fixed-v3-independent-replay", required=True)
    parser.add_argument("--paired-independent-replay", required=True)
    parser.add_argument("--expected-paired-independent-replay-sha256", required=True)
    parser.add_argument("--height-output-identity")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = _resolve(args.base_checkpoint)
    candidate = _resolve(args.candidate_checkpoint)
    candidate_config = _resolve(args.candidate_config)
    comparator = _resolve(args.source_comparator_config)
    baseline = _resolve(args.fixed_v3_baseline_audit)
    baseline_replay = _resolve(args.fixed_v3_independent_replay)
    paired_replay = _resolve(args.paired_independent_replay)
    output_identity = (
        _resolve(args.height_output_identity) if args.height_output_identity else None
    )
    for path, role in (
        (base, "base checkpoint"),
        (candidate, "candidate checkpoint"),
        (candidate_config, "sealed candidate config"),
        (comparator, "source comparator config"),
        (baseline, "fixed V3 baseline audit"),
        (baseline_replay, "fixed V3 independent replay"),
        (paired_replay, "paired V3/V4 independent replay"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{role} does not exist: {path}")
    if output_identity is not None and not output_identity.is_file():
        raise FileNotFoundError(
            f"height-output identity report does not exist: {output_identity}"
        )
    report = build_report(
        base,
        candidate,
        candidate_config,
        comparator,
        baseline,
        baseline_replay,
        paired_replay,
        output_identity,
        expected_candidate_config_sha256=args.expected_candidate_config_sha256,
        expected_paired_replay_sha256=args.expected_paired_independent_replay_sha256,
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report["recipe_audit"], indent=2))
    print(json.dumps(report["state_audit"], indent=2))
    print(json.dumps(report["classification_acceptance"], indent=2))
    print(
        "Fully eligible: "
        + str(
            report[
                "fully_eligible_for_locked_classifier_excluded_city_evaluation"
            ]
        )
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
