import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from msr.evaluation.classification_metrics import compute_multiclass_metrics


ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_gamus_hierarchical_v4_config.py"
SPEC = importlib.util.spec_from_file_location("build_v4_config_test", SCRIPT_PATH)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
COMPARATOR_PATH = ROOT / "configs" / "multidomain_surface_gamus_six_class_spatial_refined_geographic_v1.yaml"
BASELINE_AUDIT_PATH = ROOT / "outputs" / "evaluation" / "gamus_spatial_refined_geographic_v1" / "checkpoint_best_landscape_audit.json"
BASELINE_REPLAY_PATH = ROOT / "outputs" / "evaluation" / "gamus_dcphl_independent_replay_v1" / "v3_baseline_report.json"


def _baseline_metrics() -> dict:
    matrix = np.array(
        [
            [80, 2, 3, 10, 4, 1],
            [2, 88, 0, 4, 3, 3],
            [8, 0, 72, 5, 5, 10],
            [12, 4, 2, 76, 4, 2],
            [7, 2, 3, 5, 65, 18],
            [3, 3, 5, 1, 22, 66],
        ],
        dtype=np.int64,
    )
    return {
        "six_class_identification": compute_multiclass_metrics(
            matrix, BUILDER.CLASS_NAMES
        ),
        "road_boundary_quality": {"f1": 0.61},
        "water_shadow_proxy": {
            "false_water_rate_on_dark_non_water": 0.08,
        },
    }


def _comparator() -> dict:
    return {
        "experiment": {"name": "spatial-v3", "seed": 7},
        "protocol": {
            "learning_cities": ["DC", "PHL"],
            "locked_holdout_interpretation": (
                "excluded_from_this_classifier_training_not_system_unseen"
            ),
            "protected_checkpoint_sha256": "a" * 64,
            "live_pointer_file": "outputs/runtime/showcase_checkpoint.txt",
            "live_pointer_file_sha256": "b" * 64,
        },
        "data": {"approved_index_path": "fixed.json", "rgb_scale": 255.0},
        "model": {
            "initial_checkpoint": "protected.pt",
            "hidden_channels": 64,
            "fine_semantic_head_type": "spatial_refined",
        },
        "training": {
            "epochs": 8,
            "early_stopping_patience": 3,
            "early_stopping_min_epochs": 4,
            "fine_semantic_weight": 1.0,
            "parameter_groups": [{"prefixes": ["fine_semantic_head."]}],
        },
        "evaluation": {
            "primary_selection": {
                "suite": "gamus",
                "metric": "six_class_identification.macro_f1",
                "mode": "max",
            },
            "validation_guards": {
                "gamus": {"rmse_m": {"baseline": "initial", "max_regression": 0.0001}}
            },
        },
    }


def test_builder_changes_only_the_head_and_auxiliary_loss(tmp_path: Path) -> None:
    comparator = yaml.safe_load(COMPARATOR_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_AUDIT_PATH.read_text(encoding="utf-8"))
    replay = json.loads(BASELINE_REPLAY_PATH.read_text(encoding="utf-8"))

    candidate = BUILDER.build_config(
        comparator,
        baseline,
        replay,
        comparator_path=COMPARATOR_PATH,
        baseline_audit_path=BASELINE_AUDIT_PATH,
        baseline_replay_path=BASELINE_REPLAY_PATH,
    )

    assert candidate["data"] == comparator["data"]
    assert candidate["model"]["fine_semantic_head_type"] == "hierarchical_vegetation"
    assert candidate["training"]["epochs"] == comparator["training"]["epochs"]
    assert candidate["training"]["early_stopping_patience"] == 3
    assert candidate["training"]["fine_semantic_vegetation_split_weight"] == 0.5
    assert candidate["protocol"]["nyc_interpretation"].endswith(
        "not_system_unseen"
    )
    assert candidate["protocol"]["external_system_unseen_geography"].startswith(
        "pending"
    )
    assert candidate["protocol"]["official_test_policy"] == (
        "previously_consumed_forbidden_for_reuse"
    )
    assert candidate["protocol"]["height_output_identity_required_before_holdout"] is True
    assert candidate["protocol"]["fixed_v3_independent_replay_sha256"] == BUILDER.file_sha256(BASELINE_REPLAY_PATH)
    gates = candidate["protocol"]["classification_acceptance"]
    assert gates["six_class_macro_f1_min"] >= 0.60
    assert gates["buildings_f1_min"] >= 0.82
    assert gates["low_vegetation_f1_min"] >= 0.58
    assert gates["trees_f1_min"] >= 0.62
    assert set(candidate["protocol"]["classification_acceptance_by_city"]) == {"DC", "PHL"}
    assert (
        candidate["evaluation"]["validation_guards"]
        == comparator["evaluation"]["validation_guards"]
    )
    assert "six_class_identification.macro_f1" not in candidate["evaluation"][
        "validation_guards"
    ]["gamus"]


def test_builder_rejects_a_spoofed_stored_macro() -> None:
    metrics = _baseline_metrics()
    metrics["six_class_identification"]["macro_f1"] += 0.1

    with pytest.raises(ValueError, match="mean of six classes"):
        BUILDER.derive_thresholds(metrics)
