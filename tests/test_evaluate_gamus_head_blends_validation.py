from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_gamus_head_blends_validation.py"
SPEC = importlib.util.spec_from_file_location("blend_validation_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _suite(*, accuracy: float, macro_f1: float, class_f1: float, rmse: float):
    semantic = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "error_rate": 1.0 - accuracy,
        "macro_precision": macro_f1,
        "macro_recall": macro_f1,
        "macro_iou": macro_f1,
        "macro": {
            "precision": macro_f1,
            "recall": macro_f1,
            "f1": macro_f1,
            "iou": macro_f1,
        },
        "per_class": {
            name: {
                "precision": class_f1,
                "recall": class_f1,
                "f1": class_f1,
                "iou": class_f1,
            }
            for name in ("ground", "building", "vegetation")
        },
    }
    return {
        "rmse_m": rmse,
        "domains": {
            name: {"rmse_m": rmse} for name in ("ground", "building", "vegetation")
        },
        "landscapes": {
            name: {"rmse_m": rmse} for name in ("mixed", "urban", "forest")
        },
        "semantic_identification": semantic,
    }


def test_exact_comparison_uses_zero_regression_allowance() -> None:
    baseline = _suite(accuracy=0.7, macro_f1=0.6, class_f1=0.5, rmse=5.0)
    improved = _suite(accuracy=0.71, macro_f1=0.61, class_f1=0.5, rmse=4.99)
    result = MODULE.compare_suite_metrics(baseline, improved, suite="legacy")
    assert result["semantic"]["all_non_regression"] is True
    assert result["height"]["all_non_regression"] is True

    regressed = _suite(accuracy=0.71, macro_f1=0.61, class_f1=0.499, rmse=5.001)
    result = MODULE.compare_suite_metrics(baseline, regressed, suite="legacy")
    assert result["semantic"]["all_non_regression"] is False
    assert result["height"]["all_non_regression"] is False


def test_decision_requires_gamus_improvement_and_legacy_height_safety() -> None:
    baseline = _suite(accuracy=0.7, macro_f1=0.6, class_f1=0.5, rmse=5.0)
    gamus = _suite(accuracy=0.72, macro_f1=0.62, class_f1=0.49, rmse=4.9)
    legacy = _suite(accuracy=0.7, macro_f1=0.6, class_f1=0.5, rmse=5.0)
    comparisons = {
        "gamus": MODULE.compare_suite_metrics(baseline, gamus, suite="gamus"),
        "legacy": MODULE.compare_suite_metrics(baseline, legacy, suite="legacy"),
    }
    configured = {"passes": True}
    assert MODULE.blend_decision(comparisons, configured)["passes_development_guard"]
    assert MODULE.blend_decision(comparisons, configured)[
        "passes_strict_zero_regression_audit"
    ]

    legacy["rmse_m"] = 5.00001
    comparisons["legacy"] = MODULE.compare_suite_metrics(
        baseline, legacy, suite="legacy"
    )
    # The authenticated configured guard is the deployment decision; strict
    # zero-regression remains a separately visible diagnostic.
    assert MODULE.blend_decision(comparisons, configured)["passes_development_guard"]
    assert not MODULE.blend_decision(comparisons, configured)[
        "passes_strict_zero_regression_audit"
    ]
    assert not MODULE.blend_decision(comparisons, {"passes": False})[
        "passes_development_guard"
    ]


def test_cli_and_config_have_no_test_or_promotion_surface() -> None:
    args = MODULE.parse_args([])
    assert not hasattr(args, "include_test")
    source = SCRIPT.read_text(encoding="utf-8")
    config = (
        Path(__file__).parents[1]
        / "configs"
        / "gamus_head_blend_validation_v1.yaml"
    ).read_text(encoding="utf-8")
    assert "include_test=False" in source
    assert "test_manifest" not in config
    assert "promotion_performed" in source
    assert "showcase_checkpoint.txt" not in source


def test_pointer_snapshot_accepts_utf8_bom(tmp_path: Path) -> None:
    pointer = tmp_path / "pointer.txt"
    pointer.write_text("checkpoint.pt\n", encoding="utf-8-sig")
    assert MODULE.pointer_snapshot(pointer)["target"] == "checkpoint.pt"


def test_unsupported_suite_fails_closed() -> None:
    metrics = _suite(accuracy=0.7, macro_f1=0.6, class_f1=0.5, rmse=5.0)
    with pytest.raises(ValueError, match="unsupported validation suite"):
        MODULE.compare_suite_metrics(metrics, metrics, suite="test")


def test_manifest_alpha_sequence_is_config_bound(tmp_path: Path) -> None:
    # Fail before any checkpoint is loaded when the manifest's sweep does not
    # match the versioned evaluator configuration.
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"baseline": {}, "candidate": {}, "outputs": []}', encoding="utf-8"
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        MODULE.authenticate_sweep(manifest, expected_alphas=[0.05, 0.10])
