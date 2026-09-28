import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "watch_gamus_six_class_head_only_v2.py"
SPEC = importlib.util.spec_from_file_location("head_only_watcher_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _targets() -> dict[str, float]:
    return {
        "macro_f1_min": 0.4723,
        "water_recall_min": 0.10,
        "water_f1_min": 0.10,
        "road_f1_min": 0.522,
        "max_f1_drop_other_classes": 0.005,
    }


def _baseline() -> dict[str, float]:
    return {
        "ground": 0.40,
        "buildings": 0.69,
        "low_vegetation": 0.57,
        "trees": 0.62,
    }


def _record() -> dict:
    classes = {
        "ground": {"precision": 0.5, "recall": 0.5, "f1": 0.50},
        "buildings": {"precision": 0.8, "recall": 0.8, "f1": 0.80},
        "water": {"precision": 0.4, "recall": 0.3, "f1": 0.34},
        "roads": {"precision": 0.7, "recall": 0.6, "f1": 0.65},
        "low_vegetation": {"precision": 0.7, "recall": 0.6, "f1": 0.60},
        "trees": {"precision": 0.7, "recall": 0.7, "f1": 0.70},
    }
    return {
        "epoch": 2,
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


def test_completed_epoch_formats_as_one_compact_line() -> None:
    summary = MODULE.summarize_epoch(_record(), _targets(), _baseline())
    line = MODULE.format_epoch_line(summary)

    assert line == (
        "E02 mF1=0.598 | W(P/R/F1)=0.400/0.300/0.340 R=0.650 | "
        "G=0.500 B=0.800 LV=0.600 T=0.700 | H=PASS TARGET=PASS"
    )
    assert "\n" not in line


def test_target_reports_classification_and_height_failures() -> None:
    record = _record()
    metrics = record["validation_metrics"]
    metrics["six_class_identification"]["macro_f1"] = 0.45
    metrics["six_class_identification"]["per_class"]["water"]["recall"] = 0.05
    metrics["six_class_identification"]["per_class"]["trees"]["f1"] = 0.60
    metrics["passes_protected_base_guard"] = False

    summary = MODULE.summarize_epoch(record, _targets(), _baseline())
    assert summary["target_passes"] is False
    assert summary["failed_targets"] == ["macro", "waterR", "T", "height"]
    assert "H=FAIL TARGET=FAIL[macro,waterR,T,height]" in MODULE.format_epoch_line(
        summary
    )


def test_missing_class_metric_fails_closed() -> None:
    record = _record()
    del record["validation_metrics"]["six_class_identification"]["per_class"][
        "roads"
    ]["f1"]
    with pytest.raises(MODULE.MetricFormatError, match="roads F1"):
        MODULE.summarize_epoch(record, _targets(), _baseline())


def test_jsonl_reader_ignores_only_an_incomplete_final_write(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(_record()) + "\n{\"epoch\":", encoding="utf-8")
    assert len(MODULE.read_completed_records(path)) == 1

    path.write_text(json.dumps(_record()) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(MODULE.MetricFormatError, match="invalid completed JSON"):
        MODULE.read_completed_records(path)
