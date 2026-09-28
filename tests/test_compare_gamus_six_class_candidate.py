import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "compare_gamus_six_class_candidate.py"
HEIGHT_CONFIG = ROOT / "configs" / "multidomain_surface_gamus_direct_height_pilot.yaml"
EXPANDED_CONFIG = (
    ROOT / "configs" / "multidomain_surface_gamus_six_class_candidate_v1.yaml"
)
PROTOCOL_CONFIG = ROOT / "configs" / "gamus_six_class_comparison_v1.yaml"
SPEC = importlib.util.spec_from_file_location("compare_gamus_six_class_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _regression_metrics(value: float = 5.0) -> dict:
    return {
        "rmse_m": value,
        "mae_m": value - 1.0,
        "correlation": 0.7,
        "r2": 0.45,
        "region_macro_rmse_m": value,
        "object_domain_macro_rmse_m": value,
        "regions": {
            "GAMUS_DC": {"rmse_m": value},
            "GAMUS_PHL": {"rmse_m": value},
        },
        "domains": {
            "ground": {"rmse_m": value},
            "building": {"rmse_m": value},
            "vegetation": {"rmse_m": value},
        },
        "tall_objects": {
            "building": {"rmse_m": value},
            "vegetation": {"rmse_m": value},
        },
    }


def _six_class_metrics() -> dict:
    names = [
        "ground",
        "buildings",
        "water",
        "roads",
        "low_vegetation",
        "trees",
    ]
    return {
        "six_class_identification": {
            "class_names": names,
            "confusion_matrix": [
                [10 if row == column else 0 for column in range(6)]
                for row in range(6)
            ],
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_iou": 1.0,
            "macro_f1": 1.0,
            "per_class": {
                name: {
                    "precision": 1.0,
                    "recall": 1.0,
                    "iou": 1.0,
                    "f1": 1.0,
                    "support_pixels": 10,
                }
                for name in names
            },
        },
        "road_boundary_quality": {
            "tolerance_pixels": 2,
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.847,
            "dilated_boundary_iou": 0.75,
            "reference_boundary_pixels": 100,
        },
        "water_shadow_proxy": {
            "is_shadow_proxy_not_ground_truth": True,
            "dark_non_water_pixels": 100,
            "false_water_rate_on_dark_non_water": 0.04,
            "dark_share_of_all_water_false_positives": 0.2,
        },
    }


def test_expanded_config_is_only_six_class_ablation_and_test_free() -> None:
    height = yaml.safe_load(HEIGHT_CONFIG.read_text(encoding="utf-8"))
    expanded = yaml.safe_load(EXPANDED_CONFIG.read_text(encoding="utf-8"))

    MODULE.validate_fair_source_configs(height, expanded)
    assert expanded["model"]["fine_semantic_classes"] == 6
    assert expanded["training"]["fine_semantic_weight"] > 0
    assert expanded["training"]["height_weight"] == height["training"]["height_weight"]
    assert expanded["experiment"]["seed"] == height["experiment"]["seed"]
    assert expanded["data"]["patch_size"] == height["data"]["patch_size"]
    assert expanded["data"]["validation_patch_size"] == 1024
    assert not any("test" in key.lower() for key in expanded["data"])
    assert expanded["evaluation"]["promotion_eligible"] is False


def test_protocol_pins_data_preparation_and_live_artifact_hashes() -> None:
    protocol = yaml.safe_load(PROTOCOL_CONFIG.read_text(encoding="utf-8"))["protocol"]
    for field in (
        "approved_index_sha256",
        "quality_report_sha256",
        "preparation_artifact_index_sha256",
        "preparation_report_sha256",
        "dav2_identity_manifest_sha256",
        "height_outlier_manifest_sha256",
        "height_pilot_config_sha256",
        "expanded_candidate_config_sha256",
        "live_pointer_file_sha256",
        "protected_checkpoint_sha256",
    ):
        assert len(protocol[field]) == 64
    assert MODULE.file_sha256(HEIGHT_CONFIG) == protocol[
        "height_pilot_config_sha256"
    ]
    assert MODULE.file_sha256(EXPANDED_CONFIG) == protocol[
        "expanded_candidate_config_sha256"
    ]
    assert protocol["approved_train_count"] == 5001
    assert protocol["approved_validation_count"] == 859
    assert protocol["official_test_constructed"] is False
    assert protocol["promotion_permitted"] is False


def test_height_gate_directions_are_enforced() -> None:
    baseline = _regression_metrics(5.0)
    improved = _regression_metrics(4.75)
    config = {
        "rmse_m": {"direction": "min_improvement", "limit": 0.20},
        "correlation": {"direction": "max_drop", "limit": 0.01},
        "domains.building.rmse_m": {
            "direction": "max_regression",
            "limit": 0.10,
        },
    }
    result = MODULE.compare_gates(improved, baseline, config)
    assert result["passes"] is True

    regressed = _regression_metrics(5.11)
    result = MODULE.compare_gates(
        regressed,
        baseline,
        {
            "domains.building.rmse_m": {
                "direction": "max_regression",
                "limit": 0.10,
            }
        },
    )
    assert result["passes"] is False


def test_recomputed_baselines_allow_only_recorded_cuda_noise() -> None:
    baseline = _regression_metrics(5.0)
    numerically_equivalent = _regression_metrics(5.0 + 9.9e-6)

    result = MODULE.compare_recomputed_baselines(
        baseline,
        numerically_equivalent,
        {"rmse_m", "domains.building.rmse_m"},
    )
    assert result["rmse_m"]["equal_within_tolerance"] is True
    assert result["rmse_m"]["absolute_tolerance"] == pytest.approx(1.0e-4)

    materially_different = _regression_metrics(5.0 + 1.1e-4)
    with pytest.raises(ValueError, match="baseline differs"):
        MODULE.compare_recomputed_baselines(
            baseline,
            materially_different,
            {"rmse_m"},
        )


def test_six_class_completeness_requires_every_class_and_diagnostics() -> None:
    protocol = yaml.safe_load(PROTOCOL_CONFIG.read_text(encoding="utf-8"))["six_class"]
    result = MODULE.validate_six_class_metrics(_six_class_metrics(), protocol)
    assert result["passes_completeness"] is True
    assert set(result["per_class"]) == set(protocol["class_names"])
    assert result["road_boundary"]["tolerance_pixels"] == 2

    incomplete = _six_class_metrics()
    del incomplete["six_class_identification"]["per_class"]["water"]["iou"]
    with pytest.raises(ValueError, match="iou"):
        MODULE.validate_six_class_metrics(incomplete, protocol)

    mislabelled = _six_class_metrics()
    mislabelled["water_shadow_proxy"]["is_shadow_proxy_not_ground_truth"] = False
    with pytest.raises(ValueError, match="proxy"):
        MODULE.validate_six_class_metrics(mislabelled, protocol)


def test_corrected_legacy_report_is_external_and_fail_closed(tmp_path: Path) -> None:
    protocol = yaml.safe_load(PROTOCOL_CONFIG.read_text(encoding="utf-8"))[
        "corrected_legacy"
    ]

    def legacy(value: float) -> dict:
        return {
            "rmse_m": value,
            "mae_m": value - 1.0,
            "landscapes": {
                "urban": {"rmse_m": value},
                "forest": {"rmse_m": value},
            },
            "domains": {
                "building": {"rmse_m": value},
                "vegetation": {"rmse_m": value},
            },
        }

    report = {
        "schema": "msr.corrected_legacy_triplet.v1",
        "identical_protocol_for_all_models": True,
        "official_test_used": False,
        "protocols": protocol["required_protocols"],
        "models": {
            "protected": legacy(5.0),
            "height_pilot": legacy(4.9),
            "expanded_candidate": legacy(5.0),
        },
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    result = MODULE.validate_corrected_legacy_report(path, protocol)
    assert result["status"] == "passed"

    report["models"]["expanded_candidate"] = legacy(5.2)
    path.write_text(json.dumps(report), encoding="utf-8")
    result = MODULE.validate_corrected_legacy_report(path, protocol)
    assert result["status"] == "failed"

    del report["models"]["expanded_candidate"]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        MODULE.validate_corrected_legacy_report(path, protocol)


def test_height_result_is_sealed_into_expanded_launch_config(tmp_path: Path) -> None:
    height_source = yaml.safe_load(HEIGHT_CONFIG.read_text(encoding="utf-8"))
    expanded_source = yaml.safe_load(EXPANDED_CONFIG.read_text(encoding="utf-8"))
    height_dir = tmp_path / "height"
    height_dir.mkdir()
    (height_dir / "config.yaml").write_text(
        yaml.safe_dump(height_source, sort_keys=False), encoding="utf-8"
    )
    selected = _regression_metrics(4.7)
    selected.update(
        {
            "passes_validation_guards": True,
            "passes_initial_checkpoint_guard": True,
            "passes_protected_base_guard": True,
        }
    )
    record = {
        "epoch": 1,
        "validation_metrics": selected,
        "validation_metrics_by_suite": {"gamus": selected},
    }
    (height_dir / "metrics.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    (height_dir / "checkpoint_latest.pt").write_bytes(b"latest")
    (height_dir / "checkpoint_best_landscape.pt").write_bytes(b"best")
    baseline = _regression_metrics(5.0)
    (height_dir / "initial_checkpoint_validation_metrics.json").write_text(
        json.dumps(
            {
                "source": "current_validation",
                "validation_metrics_by_suite": {"gamus": baseline},
            }
        ),
        encoding="utf-8",
    )
    height = MODULE.load_experiment(
        height_dir,
        height_source,
        expected_name="multidomain_surface_gamus_direct_height_pilot",
    )
    baseline_artifact = MODULE.load_protected_baseline(height_dir)
    lock_path = tmp_path / "height_lock.json"
    launch_config_path = tmp_path / "launch.yaml"
    sealed = MODULE.seal_height_pilot_comparison(
        {"expanded_source_config": expanded_source},
        height,
        baseline_artifact,
        lock_path=lock_path,
        launch_config_path=launch_config_path,
    )
    assert MODULE.file_sha256(lock_path) == sealed[
        "height_pilot_comparison_artifact_sha256"
    ]
    launch = yaml.safe_load(launch_config_path.read_text(encoding="utf-8"))
    assert launch["protocol"]["height_pilot_metrics_sha256"] == height[
        "metrics_sha256"
    ]
    assert launch["protocol"]["height_pilot_selected_checkpoint_sha256"] == height[
        "selected_checkpoint_sha256"
    ]
    assert launch["protocol"]["height_pilot_latest_checkpoint_sha256"] == height[
        "latest_checkpoint_sha256"
    ]

    expanded = {
        "saved_protocol": launch["protocol"],
    }
    evidence = MODULE.validate_embedded_height_pilot_lock(expanded, height)
    assert evidence["status"] == "passed"
    lock_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.validate_embedded_height_pilot_lock(expanded, height)


def test_raw_best_metrics_never_claim_a_different_latest_checkpoint(tmp_path: Path) -> None:
    height_source = yaml.safe_load(HEIGHT_CONFIG.read_text(encoding="utf-8"))
    experiment = tmp_path / "height"
    experiment.mkdir()
    (experiment / "config.yaml").write_text(
        yaml.safe_dump(height_source, sort_keys=False), encoding="utf-8"
    )
    records = []
    for epoch, rmse in ((1, 4.0), (2, 5.0)):
        metrics = _regression_metrics(rmse)
        metrics.update(
            {
                "passes_validation_guards": False,
                "passes_initial_checkpoint_guard": False,
                "passes_protected_base_guard": True,
            }
        )
        records.append(
            {
                "epoch": epoch,
                "validation_metrics": metrics,
                "validation_metrics_by_suite": {"gamus": metrics},
            }
        )
    (experiment / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (experiment / "checkpoint_latest.pt").write_bytes(b"epoch-two")

    result = MODULE.load_experiment(
        experiment,
        height_source,
        expected_name="multidomain_surface_gamus_direct_height_pilot",
    )

    assert result["best_raw_epoch"] == 1
    assert result["latest_epoch"] == 2
    assert result["selected_epoch"] == 1
    assert result["selected_checkpoint"] is None
    assert result["selected_checkpoint_sha256"] is None
    assert result["selected_checkpoint_available"] is False
    assert result["latest_checkpoint_sha256"] == MODULE.file_sha256(
        experiment / "checkpoint_latest.pt"
    )


def test_launcher_and_reporter_have_no_auto_promotion_surface() -> None:
    launcher = (ROOT / "scripts" / "run_gamus_six_class_candidate_v1.ps1").read_text(
        encoding="utf-8"
    )
    reporter = SCRIPT.read_text(encoding="utf-8")
    watcher = (
        ROOT / "scripts" / "watch_gamus_six_class_candidate_v1.ps1"
    ).read_text(encoding="utf-8")
    assert "COMPLETE offline GAMUS direct-height pilot" in launcher
    assert "showcase_checkpoint.txt" in launcher
    assert "select_showcase_checkpoint" not in launcher
    assert '"promotion_performed": False' in reporter
    assert '"active_model_changed": False' in reporter
    assert "test" not in EXPANDED_CONFIG.name.lower()
    assert "six_class_identification" in watcher
