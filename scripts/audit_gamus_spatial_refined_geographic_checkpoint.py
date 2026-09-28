"""Audit the fixed DC+Philadelphia spatial-V3 classifier comparator.

This auditor exists because the original spatial-V3 auditor intentionally
accepts only the original all-city V3 recipe.  It authenticates the exact
geographic recipe, proves that only the five spatial-head tensors differ from
the protected production checkpoint, and independently recomputes every
six-class score from the stored 6x6 confusion matrix.

It never constructs the NYC image dataset, never discovers the official GAMUS
test split, and never modifies the application checkpoint pointer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msr.data.gamus_geographic_candidate import (  # noqa: E402
    EXPECTED_SPATIAL_REFINED_EXPERIMENT_NAME,
    GAMUS_SPATIAL_REFINED_GEOGRAPHIC_CANDIDATE_SCHEMA,
    resolve_project_path,
    validate_geographic_candidate_config,
)


def _load_spatial_state_auditor():
    path = SCRIPTS_ROOT / "audit_gamus_spatial_refined_head_checkpoint.py"
    spec = importlib.util.spec_from_file_location(
        "_msr_spatial_v3_state_audit", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load spatial-V3 state auditor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SPATIAL_STATE_AUDIT = _load_spatial_state_auditor()
SCHEMA = "msr.gamus_spatial_refined_geographic_checkpoint_audit.v1"
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


def recompute_six_class_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Verify stored values and return metrics derived only from the 6x6 matrix."""

    report = _mapping(
        metrics.get("six_class_identification"), "six-class identification"
    )
    if tuple(report.get("class_names", ())) != CLASS_NAMES:
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
        raise ValueError("six-class confusion entries must be nonnegative integers")

    stored_classes = _mapping(report.get("per_class"), "stored per-class metrics")
    per_class: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(CLASS_NAMES):
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
        denominator = true_positive + false_positive + false_negative
        iou = true_positive / denominator if denominator else 0.0
        computed: dict[str, float | int] = {
            "support": support,
            "predicted_pixels": predicted,
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }
        stored = _mapping(stored_classes.get(name), f"stored class {name}")
        for metric_name in ("precision", "recall", "f1", "iou"):
            actual = stored.get(metric_name)
            expected = float(computed[metric_name])
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or abs(float(actual) - expected) > 1.0e-12
            ):
                raise ValueError(
                    f"stored {name} {metric_name} does not match the 6x6 matrix"
                )
        per_class[name] = computed

    macro = {
        metric_name: sum(float(values[metric_name]) for values in per_class.values())
        / 6.0
        for metric_name in ("precision", "recall", "f1", "iou")
    }
    stored_macro = _mapping(report.get("macro"), "stored six-class macro")
    for metric_name, expected in macro.items():
        for actual in (
            stored_macro.get(metric_name),
            report.get(f"macro_{metric_name}"),
        ):
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or abs(float(actual) - expected) > 1.0e-12
            ):
                raise ValueError(
                    f"stored six-class macro {metric_name} is not the mean of six classes"
                )
    return {
        "passes": True,
        "definition": "arithmetic mean across the final six classes",
        "confusion_matrix": matrix,
        "per_class": per_class,
        "macro": macro,
    }


def validate_geographic_comparator_recipe(
    config: Mapping[str, Any], *, project_root: Path
) -> dict[str, Any]:
    """Apply the sealed geographic preflight plus stricter truth wording."""

    experiment = _mapping(config.get("experiment"), "candidate experiment")
    protocol = _mapping(config.get("protocol"), "candidate protocol")
    if experiment.get("name") != EXPECTED_SPATIAL_REFINED_EXPERIMENT_NAME:
        raise ValueError("not the exact spatial-refined geographic comparator")
    if protocol.get("schema") != GAMUS_SPATIAL_REFINED_GEOGRAPHIC_CANDIDATE_SCHEMA:
        raise ValueError("unexpected spatial geographic protocol schema")
    if protocol.get("purpose") != (
        "fair_dc_phl_spatial_classifier_baseline_with_classifier_excluded_nyc"
    ):
        raise ValueError("geographic comparator purpose is not the corrected policy")
    if protocol.get("locked_holdout_interpretation") != (
        "excluded_from_this_classifier_training_not_system_unseen"
    ):
        raise ValueError("NYC must not be described as system-unseen")
    if protocol.get("external_system_unseen_geography") != "pending_separate_dataset":
        raise ValueError("external system-unseen geography must remain pending")
    # This legacy per-run field means this comparator did not construct or
    # discover the official split.  It must not be inflated into a global
    # claim: an older Stage-3 experiment already consumed that benchmark.
    if protocol.get("official_test_policy") != "never_constructed_or_discovered":
        raise ValueError("this comparator must forbid official GAMUS test access")
    if protocol.get("auto_promotion") is not False:
        raise ValueError("automatic app promotion must remain disabled")

    preflight = validate_geographic_candidate_config(
        config, project_root=project_root
    )
    learning = _mapping(preflight.get("learning"), "learning preflight")
    if int(learning.get("tiles", -1)) != 3837 or learning.get("cities") != [
        "DC",
        "PHL",
    ]:
        raise ValueError("learning partition is not the fixed 3,837-tile DC+PHL set")
    if preflight["epoch_selection"]["tiles"] != 859:
        raise ValueError("development partition is not the fixed 859-tile set")
    if preflight["locked_post_freeze_evaluation"]["tiles"] != 1164:
        raise ValueError("locked NYC inventory is not the fixed 1,164-tile set")
    return {
        "passes": True,
        "geographic_preflight": preflight,
        "classifier_training_cities": ["DC", "PHL"],
        "development_selection_cities": ["DC", "PHL"],
        "nyc_interpretation": (
            "excluded_from_this_classifier_training_not_system_unseen"
        ),
        "nyc_imagery_opened": False,
        "official_test_used_by_this_comparator": False,
        "official_test_global_status": "previously_consumed_forbidden_for_reuse",
        "external_system_unseen_geography": "pending_separate_dataset",
        "promotion_performed": False,
    }


def build_report(
    base_path: Path,
    candidate_path: Path,
    candidate_config_path: Path,
) -> dict[str, Any]:
    config = _load_yaml(candidate_config_path)
    recipe_audit = validate_geographic_comparator_recipe(
        config, project_root=PROJECT_ROOT
    )
    base_state, base_payload = SPATIAL_STATE_AUDIT.load_model_state(base_path)
    candidate_state, candidate_payload = SPATIAL_STATE_AUDIT.load_model_state(
        candidate_path
    )
    embedded_config = _mapping(candidate_payload.get("config"), "embedded config")
    if embedded_config != config:
        raise ValueError("checkpoint embedded config differs from authenticated config")
    state_audit = SPATIAL_STATE_AUDIT.audit_states(base_state, candidate_state)
    metrics = _mapping(candidate_payload.get("metrics"), "candidate metrics")
    six_class_audit = recompute_six_class_metrics(metrics)

    road_boundary = _mapping(
        metrics.get("road_boundary_quality"), "road-boundary metrics"
    )
    water_shadow = _mapping(
        metrics.get("water_shadow_proxy"), "water-shadow proxy metrics"
    )
    for value, role in (
        (road_boundary.get("f1"), "road-boundary F1"),
        (
            water_shadow.get("false_water_rate_on_dark_non_water"),
            "false-water-on-dark-non-water rate",
        ),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not torch.isfinite(torch.tensor(float(value)))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{role} is missing, non-finite, or outside [0, 1]")

    protocol = _mapping(config.get("protocol"), "candidate protocol")
    pointer_path = resolve_project_path(
        PROJECT_ROOT, protocol.get("live_pointer_file")
    )
    protected_path = resolve_project_path(
        PROJECT_ROOT, _mapping(config.get("model"), "candidate model").get(
            "initial_checkpoint"
        )
    )
    if protected_path != base_path:
        raise ValueError("audited base is not the config's protected checkpoint")
    pointer_value = pointer_path.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    if resolve_project_path(PROJECT_ROOT, pointer_value) != base_path:
        raise ValueError("live app pointer no longer targets the protected checkpoint")

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
        "candidate_config": {
            "path": str(candidate_config_path),
            "sha256": file_sha256(candidate_config_path),
        },
        "recipe_audit": recipe_audit,
        "state_audit": state_audit,
        "six_class_metric_audit": six_class_audit,
        "candidate_metrics": dict(metrics),
        "app_pointer_verification": {
            "passes": True,
            "path": str(pointer_path),
            "sha256": file_sha256(pointer_path),
            "target": str(base_path),
        },
        "nyc_evaluation_performed": False,
        "official_test_used_by_this_comparator": False,
        "official_test_global_status": "previously_consumed_forbidden_for_reuse",
        "promotion_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_checkpoint).expanduser().resolve()
    candidate_path = Path(args.candidate_checkpoint).expanduser().resolve()
    config_path = Path(args.candidate_config).expanduser().resolve()
    for path, role in (
        (base_path, "protected checkpoint"),
        (candidate_path, "candidate checkpoint"),
        (config_path, "candidate config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{role} does not exist: {path}")

    pointer_path = resolve_project_path(
        PROJECT_ROOT,
        _mapping(_load_yaml(config_path).get("protocol"), "protocol").get(
            "live_pointer_file"
        ),
    )
    pointer_before = file_sha256(pointer_path)
    protected_before = file_sha256(base_path)
    report = build_report(base_path, candidate_path, config_path)
    if file_sha256(pointer_path) != pointer_before:
        raise RuntimeError("live application pointer changed during read-only audit")
    if file_sha256(base_path) != protected_before:
        raise RuntimeError("protected checkpoint changed during read-only audit")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(report["six_class_metric_audit"]["macro"], indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
