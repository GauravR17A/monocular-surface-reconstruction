import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "watch_gamus_six_class_spatial_refined_v3.py"
SPEC = importlib.util.spec_from_file_location("spatial_refined_v3_watcher_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record() -> dict:
    classes = {
        "ground": {"precision": 0.51, "recall": 0.49, "f1": 0.50},
        "buildings": {"precision": 0.82, "recall": 0.78, "f1": 0.80},
        "water": {"precision": 0.40, "recall": 0.30, "f1": 0.34},
        "roads": {"precision": 0.70, "recall": 0.60, "f1": 0.65},
        "low_vegetation": {"precision": 0.62, "recall": 0.58, "f1": 0.60},
        "trees": {"precision": 0.71, "recall": 0.69, "f1": 0.70},
    }
    return {
        "epoch": 3,
        "validation_metrics": {
            "six_class_identification": {
                "macro_f1": 0.598,
                "per_class": classes,
            },
            "passes_validation_guards": True,
            "passes_initial_checkpoint_guard": True,
            "passes_protected_base_guard": True,
        },
    }


def _comparison_report() -> dict:
    return {
        "expanded_classification": {
            "per_class": {
                "ground": {"f1": 0.40},
                "buildings": {"f1": 0.69},
                "low_vegetation": {"f1": 0.57},
                "trees": {"f1": 0.62},
            }
        }
    }


def _config(report_path: Path) -> dict:
    return {
        "protocol": {
            "classification_acceptance": {
                "comparison_report": str(report_path),
                "macro_f1_min": 0.4723,
                "water_recall_min": 0.10,
                "water_f1_min": 0.10,
                "road_f1_min": 0.522,
                "max_f1_drop_other_classes": 0.005,
            }
        }
    }


def test_v3_profile_uses_v3_inputs_and_real_v1_baseline() -> None:
    assert MODULE.DEFAULT_CONFIG.name == (
        "multidomain_surface_gamus_six_class_spatial_refined_v3.yaml"
    )
    assert MODULE.EXPERIMENT_GLOB == (
        "*_multidomain_surface_gamus_six_class_spatial_refined_v3"
    )
    targets, baseline = MODULE.load_acceptance_inputs(MODULE.DEFAULT_CONFIG)
    assert targets["macro_f1_min"] == 0.4723
    assert baseline["water"] if "water" in baseline else True
    assert set(baseline) == {"ground", "buildings", "low_vegetation", "trees"}


def test_newest_experiment_filters_out_other_recipes(tmp_path: Path) -> None:
    (tmp_path / "99999999_multidomain_surface_gamus_six_class_head_only_v2").mkdir()
    older = tmp_path / "20260913T010000Z_multidomain_surface_gamus_six_class_spatial_refined_v3"
    newer = tmp_path / "20260913T020000Z_multidomain_surface_gamus_six_class_spatial_refined_v3"
    older.mkdir()
    newer.mkdir()
    assert MODULE.newest_experiment(tmp_path) == newer


def test_v3_terminal_markers_are_scoped_to_latest_session(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        "[x] COMPLETE spatial-refined v3\n"
        "[y] START protected-base spatial-refined six-class-head-only experiment\n",
        encoding="utf-8",
    )
    assert MODULE._run_has_ended(log) is False
    with log.open("a", encoding="utf-8") as handle:
        handle.write("[z] COMPLETE spatial-refined v3 and exact inherited-tensor audits\n")
    assert MODULE._run_has_ended(log) is True


def test_once_cli_prints_exactly_one_compact_v3_line(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_comparison_report()), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_config(report)), encoding="utf-8")
    experiment = tmp_path / "run_multidomain_surface_gamus_six_class_spatial_refined_v3"
    experiment.mkdir()
    (experiment / "metrics.jsonl").write_text(
        json.dumps(_record()) + "\n", encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--experiment",
            str(experiment),
            "--once",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stderr == ""
    assert completed.stdout.splitlines() == [
        "E03 mF1=0.598 | W(P/R/F1)=0.400/0.300/0.340 R=0.650 | "
        "G=0.500 B=0.800 LV=0.600 T=0.700 | H=PASS TARGET=PASS"
    ]
