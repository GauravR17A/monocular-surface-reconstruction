import importlib.util
import hashlib
import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from msr.evaluation.classification_metrics import compute_multiclass_metrics


ROOT = Path(__file__).parents[1]
AUDITOR_PATH = ROOT / "scripts" / "audit_gamus_hierarchical_v4_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("hierarchical_v4_audit_test", AUDITOR_PATH)
assert SPEC and SPEC.loader
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)
BUILDER_PATH = ROOT / "scripts" / "build_gamus_hierarchical_v4_config.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("v4_builder_for_audit_test", BUILDER_PATH)
assert BUILDER_SPEC and BUILDER_SPEC.loader
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)

CLASS_NAMES = (
    "ground",
    "buildings",
    "water",
    "roads",
    "low_vegetation",
    "trees",
)


def _classification_metrics(diagonal: int) -> dict:
    matrix = np.full((6, 6), (100 - diagonal) // 5, dtype=np.int64)
    np.fill_diagonal(matrix, diagonal)
    return compute_multiclass_metrics(matrix, CLASS_NAMES)


def _full_metrics(diagonal: int) -> dict:
    return {
        "six_class_identification": _classification_metrics(diagonal),
        "road_boundary_quality": {"f1": diagonal / 100.0},
        "water_shadow_proxy": {
            "false_water_rate_on_dark_non_water": (100 - diagonal) / 100.0
        },
    }


def _base_state() -> dict[str, torch.Tensor]:
    return {
        "adapter.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "domain_head.bias": torch.tensor([1.0, 2.0, 3.0]),
    }


def _candidate_state() -> dict[str, torch.Tensor]:
    channels = 64
    return {
        **_base_state(),
        "fine_semantic_head.spatial.weight": torch.ones(channels, 1, 3, 3),
        "fine_semantic_head.normalization.weight": torch.ones(channels),
        "fine_semantic_head.normalization.bias": torch.zeros(channels),
        "fine_semantic_head.coarse_classifier.weight": torch.ones(
            5, channels, 1, 1
        ),
        "fine_semantic_head.coarse_classifier.bias": torch.zeros(5),
        "fine_semantic_head.vegetation_split.weight": torch.ones(
            1, channels, 1, 1
        ),
        "fine_semantic_head.vegetation_split.bias": torch.zeros(1),
    }


def _recipe_configs() -> tuple[dict, dict, dict]:
    baseline_metrics = _full_metrics(50)
    baseline_replay = {
        "validation_contract": {"contract_sha256": "d" * 64},
        "source_content_binding": {"combined_content_manifest_sha256": "e" * 64},
        "v3": {
            "overall": baseline_metrics,
            "by_city": {
                "DC": deepcopy(baseline_metrics),
                "PHL": deepcopy(baseline_metrics),
            },
        },
    }
    thresholds = AUDITOR.derive_fixed_gates(baseline_metrics)
    comparator = {
        "data": {"approved_index_path": "fixed-dc-phl.json"},
        "model": {"hidden_channels": 64, "fine_semantic_head_type": "spatial_refined"},
        "training": {
            "epochs": 8,
            "early_stopping_patience": 3,
            "early_stopping_min_epochs": 4,
            "fine_semantic_weight": 1.0,
            "parameter_groups": [
                {"prefixes": ["fine_semantic_head."], "learning_rate": 0.001}
            ],
        },
        "evaluation": {
            "primary_selection": {
                "suite": "gamus",
                "metric": "six_class_identification.macro_f1",
                "mode": "max",
            },
            "validation_guards": {
                "gamus": {
                    "rmse_m": {"baseline": "initial", "max_regression": 0.0001}
                }
            },
        },
    }
    candidate = deepcopy(comparator)
    candidate["protocol"] = {
        "schema": AUDITOR.PROTOCOL_SCHEMA,
        "source_comparator_config_sha256": "a" * 64,
        "fixed_v3_baseline_audit_sha256": "b" * 64,
        "fixed_v3_independent_replay_sha256": "f" * 64,
        "independent_replay_schema": AUDITOR.INDEPENDENT_REPLAY_SCHEMA,
        "independent_replay_evaluator_sha256": AUDITOR.file_sha256(
            AUDITOR.REPLAY_EVALUATOR_PATH
        ),
        "independent_replay_validation_contract_sha256": "d" * 64,
        "independent_replay_source_content_manifest_sha256": "e" * 64,
        "paired_independent_replay_required_before_holdout": True,
        "official_test_policy": "previously_consumed_forbidden_for_reuse",
        "auto_promotion": False,
        "nyc_interpretation": "excluded_from_this_classifier_training_not_system_unseen",
        "external_system_unseen_geography": "pending_separate_dataset",
        "height_output_contract_sha256": "c" * 64,
        "height_output_identity_required_before_holdout": True,
        "classification_acceptance": thresholds,
        "classification_acceptance_by_city": AUDITOR.derive_per_city_gates(
            baseline_replay["v3"]["by_city"]
        ),
    }
    candidate["model"]["fine_semantic_head_type"] = "hierarchical_vegetation"
    candidate["training"]["fine_semantic_vegetation_split_weight"] = 0.5
    return comparator, candidate, baseline_replay


def test_v4_state_audit_allows_only_the_1094_parameter_head() -> None:
    result = AUDITOR.audit_states(_base_state(), _candidate_state())

    assert result["passes"] is True
    assert result["head_parameter_count"] == 1094
    assert result["changed_inherited_tensors"] == []
    assert set(result["head_tensors"]) == AUDITOR.EXPECTED_HEAD_KEYS


def test_v4_state_audit_rejects_any_inherited_change() -> None:
    candidate = _candidate_state()
    candidate["adapter.weight"][0, 0] += 1

    with pytest.raises(ValueError, match="protected height/shared tensors changed"):
        AUDITOR.audit_states(_base_state(), candidate)


def test_six_class_macro_is_recomputed_and_cannot_be_replaced_by_five_group() -> None:
    metrics = _full_metrics(55)
    actual = AUDITOR.recompute_six_class_metrics(metrics)
    assert actual["macro"]["f1"] == pytest.approx(0.55)

    metrics["six_class_identification"]["macro"]["f1"] = 0.61
    metrics["six_class_identification"]["macro_f1"] = 0.61
    with pytest.raises(ValueError, match="mean of six classes"):
        AUDITOR.recompute_six_class_metrics(metrics)


def test_classification_acceptance_uses_all_six_classes() -> None:
    baseline = _full_metrics(50)
    candidate = _full_metrics(55)
    thresholds = {
        "six_class_macro_f1_min": 0.60,
        "ground_f1_min": 0.40,
        "buildings_f1_min": 0.40,
        "water_f1_min": 0.40,
        "roads_f1_min": 0.40,
        "low_vegetation_f1_min": 0.40,
        "trees_f1_min": 0.40,
        "road_boundary_f1_min": 0.40,
        "dark_non_water_false_water_rate_max": 0.50,
    }

    result = AUDITOR.classification_acceptance(candidate, baseline, thresholds)

    assert result["passes"] is False
    assert result["checks"]["six_class_macro_f1"]["value"] == pytest.approx(0.55)
    assert result["checks"]["six_class_macro_f1"]["passes"] is False


def test_v4_recipe_keeps_the_exact_v3_training_schedule() -> None:
    comparator, candidate, baseline_replay = _recipe_configs()

    result = AUDITOR.validate_recipe_contract(
        comparator,
        candidate,
        baseline_replay,
        comparator_sha256="a" * 64,
        baseline_audit_sha256="b" * 64,
        baseline_replay_sha256="f" * 64,
    )
    assert result["passes"] is True

    candidate["training"]["epochs"] = 10
    with pytest.raises(ValueError, match="outside approved fields|retain V3's epochs"):
        AUDITOR.validate_recipe_contract(
            comparator,
            candidate,
            baseline_replay,
            comparator_sha256="a" * 64,
            baseline_audit_sha256="b" * 64,
            baseline_replay_sha256="f" * 64,
        )


def test_v4_recipe_rederives_and_rejects_lowered_fixed_gates() -> None:
    comparator, candidate, baseline_replay = _recipe_configs()
    candidate["protocol"]["classification_acceptance"][
        "trees_f1_min"
    ] -= 0.01

    with pytest.raises(ValueError, match="classification_acceptance.trees_f1_min"):
        AUDITOR.validate_recipe_contract(
            comparator,
            candidate,
            baseline_replay,
            comparator_sha256="a" * 64,
            baseline_audit_sha256="b" * 64,
            baseline_replay_sha256="f" * 64,
        )


def test_candidate_config_identity_requires_reviewed_bytes_and_exact_embedding(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "sealed_v4.yaml"
    config = {"experiment": {"name": "v4"}, "training": {"epochs": 8}}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    expected_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    loaded, identity = AUDITOR.validate_candidate_config_identity(
        config_path,
        expected_sha256=expected_sha256,
        embedded_config=deepcopy(config),
    )
    assert loaded == config
    assert identity == {"path": str(config_path), "sha256": expected_sha256}

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        AUDITOR.validate_candidate_config_identity(
            config_path,
            expected_sha256="0" * 64,
            embedded_config=config,
        )

    changed = deepcopy(config)
    changed["training"]["epochs"] = 9
    with pytest.raises(ValueError, match="embedded config differs"):
        AUDITOR.validate_candidate_config_identity(
            config_path,
            expected_sha256=expected_sha256,
            embedded_config=changed,
        )


def test_height_output_identity_report_is_bound_to_exact_candidate(tmp_path: Path) -> None:
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    base.write_bytes(b"base")
    candidate.write_bytes(b"candidate")
    base_sha = AUDITOR.file_sha256(base)
    candidate_sha = AUDITOR.file_sha256(candidate)
    report = {
        "schema": AUDITOR.HEIGHT_IDENTITY_SCHEMA,
        "passes": True,
        "protected": {
            "path": str(base),
            "sha256_before": base_sha,
            "sha256_after": base_sha,
        },
        "candidate": {
            "path": str(candidate),
            "sha256_before": candidate_sha,
            "sha256_after": candidate_sha,
        },
        "provenance_audit": {"passes": True},
        "state_audit": {"passes": True},
        "pipeline_contract_audit": {
            "passes": True,
            "full_contract_sha256": "c" * 64,
        },
        "loaded_model_audit": {"passes": True},
        "direct_output_audit": {"passes": True},
        "app_output_audit": {"passes": True, "cases": [{}, {}, {}]},
        "checkpoint_files_unchanged_during_audit": True,
        "promotion_performed": False,
        "app_pointer_verification": {"passes": True},
    }

    result = AUDITOR.validate_height_output_identity_report(
        report,
        base_path=base.resolve(),
        base_sha256=base_sha,
        candidate_path=candidate.resolve(),
        candidate_sha256=candidate_sha,
        expected_contract_sha256="c" * 64,
    )
    assert result["passes"] is True
    assert result["app_case_count"] == 3

    report["candidate"]["sha256_after"] = "d" * 64
    with pytest.raises(ValueError, match="candidate after hash"):
        AUDITOR.validate_height_output_identity_report(
            report,
            base_path=base.resolve(),
            base_sha256=base_sha,
            candidate_path=candidate.resolve(),
            candidate_sha256=candidate_sha,
            expected_contract_sha256="c" * 64,
        )


def test_fixed_v3_baseline_must_be_sealed_and_holdout_safe(tmp_path: Path) -> None:
    config = tmp_path / "v3.yaml"
    checkpoint = tmp_path / "v3_best.pt"
    config.write_text("sealed: true\n", encoding="utf-8")
    checkpoint.write_bytes(b"v3 checkpoint")
    report = {
        "schema": AUDITOR.BASELINE_AUDIT_SCHEMA,
        "recipe_audit": {"passes": True},
        "state_audit": {"passes": True},
        "six_class_metric_audit": {"passes": True},
        "candidate_config": {"sha256": AUDITOR.file_sha256(config)},
        "candidate": {
            "path": str(checkpoint),
            "sha256": AUDITOR.file_sha256(checkpoint),
        },
        "nyc_evaluation_performed": False,
        "official_test_used_by_this_comparator": False,
        "official_test_global_status": "previously_consumed_forbidden_for_reuse",
        "promotion_performed": False,
        "app_pointer_verification": {"passes": True},
    }

    result = AUDITOR.validate_fixed_v3_baseline_audit(
        report, comparator_config_path=config
    )
    assert result["passes"] is True

    report["nyc_evaluation_performed"] = True
    with pytest.raises(ValueError, match="consumed the NYC"):
        AUDITOR.validate_fixed_v3_baseline_audit(
            report, comparator_config_path=config
        )


def test_protected_application_state_requires_exact_pointer(tmp_path: Path) -> None:
    protected = tmp_path / "protected.pt"
    pointer = tmp_path / "pointer.txt"
    protected.write_bytes(b"production model")
    pointer.write_text(str(protected), encoding="utf-8")
    protocol = {
        "protected_checkpoint_sha256": AUDITOR.file_sha256(protected),
        "live_pointer_file": str(pointer),
        "live_pointer_file_sha256": AUDITOR.file_sha256(pointer),
    }

    result = AUDITOR.verify_protected_application_state(
        protocol, base_path=protected.resolve()
    )
    assert result["passes"] is True

    pointer.write_text(str(tmp_path / "other.pt"), encoding="utf-8")
    with pytest.raises(ValueError, match="pointer hash changed"):
        AUDITOR.verify_protected_application_state(
            protocol, base_path=protected.resolve()
        )


def _paired_replay_fixture(tmp_path: Path) -> tuple[dict, dict, Path, Path, Path, Path]:
    comparator_path = ROOT / "configs" / "multidomain_surface_gamus_six_class_spatial_refined_geographic_v1.yaml"
    audit_path = ROOT / "outputs" / "evaluation" / "gamus_spatial_refined_geographic_v1" / "checkpoint_best_landscape_audit.json"
    baseline_replay_path = ROOT / "outputs" / "evaluation" / "gamus_dcphl_independent_replay_v1" / "v3_baseline_report.json"
    comparator = yaml.safe_load(comparator_path.read_text(encoding="utf-8"))
    baseline_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    baseline_replay = json.loads(baseline_replay_path.read_text(encoding="utf-8"))
    candidate_config = BUILDER.build_config(
        comparator,
        baseline_audit,
        baseline_replay,
        comparator_path=comparator_path,
        baseline_audit_path=audit_path,
        baseline_replay_path=baseline_replay_path,
    )
    candidate_config_path = tmp_path / "v4.yaml"
    candidate_config_path.write_text(
        yaml.safe_dump(candidate_config, sort_keys=False), encoding="utf-8"
    )
    sealed_candidate_config = yaml.safe_load(
        candidate_config_path.read_text(encoding="utf-8")
    )
    candidate_checkpoint_path = tmp_path / "v4.pt"
    torch.save(
        {
            "config": sealed_candidate_config,
            "model": {},
            "epoch": 3,
            "model_type": baseline_replay["v3"]["model_type"],
        },
        candidate_checkpoint_path,
    )
    evaluator = AUDITOR._load_replay_evaluator()
    candidate_checkpoint_identity = evaluator.authenticate_checkpoint(
        candidate_checkpoint_path,
        expected_sha256=AUDITOR.file_sha256(candidate_checkpoint_path),
        external_config=sealed_candidate_config,
    )
    paired = deepcopy(baseline_replay)
    paired["mode"] = "paired_v3_v4"
    paired["v4"] = deepcopy(paired["v3"])
    paired["v4"]["fine_semantic_head_type"] = "hierarchical_vegetation"
    paired["v4_minus_v3"] = evaluator.paired_deltas(paired["v3"], paired["v4"])
    paired["sealed_artifacts"]["v4_config"] = {
        "path": str(candidate_config_path.resolve()),
        "sha256": AUDITOR.file_sha256(candidate_config_path),
        "size_bytes": candidate_config_path.stat().st_size,
    }
    paired["sealed_artifacts"]["v4_checkpoint"] = candidate_checkpoint_identity
    paired["sealed_artifacts"]["after_sha256"].update(
        {
            "v4_config_sha256": AUDITOR.file_sha256(candidate_config_path),
            "v4_checkpoint_sha256": AUDITOR.file_sha256(candidate_checkpoint_path),
        }
    )
    paired_path = tmp_path / "paired.json"
    paired_path.write_text(json.dumps(paired, indent=2), encoding="utf-8")
    baseline_checkpoint_path = Path(
        baseline_replay["sealed_artifacts"]["v3_checkpoint"]["path"]
    )
    return (
        paired,
        baseline_replay,
        paired_path,
        comparator_path,
        candidate_config_path,
        baseline_checkpoint_path,
        candidate_checkpoint_path,
    )


def test_paired_independent_replay_is_exactly_bound_and_formula_checked(
    tmp_path: Path,
) -> None:
    (
        paired,
        baseline,
        paired_path,
        comparator_path,
        candidate_config_path,
        baseline_checkpoint_path,
        candidate_checkpoint_path,
    ) = _paired_replay_fixture(tmp_path)
    result = AUDITOR.validate_paired_independent_replay(
        paired,
        baseline,
        report_path=paired_path,
        expected_report_sha256=AUDITOR.file_sha256(paired_path),
        comparator_config_path=comparator_path,
        candidate_config_path=candidate_config_path,
        baseline_checkpoint_path=baseline_checkpoint_path,
        candidate_checkpoint_path=candidate_checkpoint_path,
    )
    assert result["passes"] is True
    assert result["evaluated_sample_count"] == 859

    bad = deepcopy(paired)
    bad["v4"]["overall"]["six_class_identification"]["macro_f1"] = 0.99
    bad_path = tmp_path / "bad-paired.json"
    bad_path.write_text(json.dumps(bad, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent"):
        AUDITOR.validate_paired_independent_replay(
            bad,
            baseline,
            report_path=bad_path,
            expected_report_sha256=AUDITOR.file_sha256(bad_path),
            comparator_config_path=comparator_path,
            candidate_config_path=candidate_config_path,
            baseline_checkpoint_path=baseline_checkpoint_path,
            candidate_checkpoint_path=candidate_checkpoint_path,
        )


def test_per_city_gates_cover_macro_all_classes_boundary_and_dark_water() -> None:
    baseline = json.loads(
        (
            ROOT
            / "outputs"
            / "evaluation"
            / "gamus_dcphl_independent_replay_v1"
            / "v3_baseline_report.json"
        ).read_text(encoding="utf-8")
    )
    gates = AUDITOR.derive_per_city_gates(baseline["v3"]["by_city"])
    expected_keys = {
        "six_class_macro_f1_min",
        *(f"{name}_f1_min" for name in CLASS_NAMES),
        "road_boundary_f1_min",
        "dark_non_water_false_water_rate_max",
    }
    assert set(gates) == {"DC", "PHL"}
    assert all(set(city_gates) == expected_keys for city_gates in gates.values())
