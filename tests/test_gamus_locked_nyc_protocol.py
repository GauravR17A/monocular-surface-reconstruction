from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_gamus_locked_nyc.py"
SPEC = importlib.util.spec_from_file_location("evaluate_gamus_locked_nyc_test", SCRIPT)
assert SPEC and SPEC.loader
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _metrics(*, water_f1: float = 0.30) -> dict[str, object]:
    names = [
        "ground",
        "buildings",
        "water",
        "roads",
        "low_vegetation",
        "trees",
    ]
    f1 = {
        "ground": 0.60,
        "buildings": 0.70,
        "water": water_f1,
        "roads": 0.60,
        "low_vegetation": 0.65,
        "trees": 0.70,
    }
    return {
        "rmse_m": 5.0,
        "mae_m": 2.0,
        "bias_m": -0.5,
        "correlation": 0.6,
        "r2": 0.3,
        "six_class_identification": {
            "class_names": names,
            "macro_f1": sum(f1.values()) / 6,
            "macro_iou": 0.40,
            "per_class": {
                name: {
                    "precision": f1[name],
                    "recall": f1[name],
                    "iou": 0.40,
                    "f1": f1[name],
                    "support_pixels": 100,
                }
                for name in names
            },
        },
        "road_boundary_quality": {"f1": 0.30},
        "water_shadow_proxy": {
            "is_shadow_proxy_not_ground_truth": True,
            "false_water_rate_on_dark_non_water": 0.01,
        },
    }


def test_predeclared_locked_readiness_gates_report_pass_and_failure() -> None:
    _, protocol = EVALUATOR.load_protocol(
        ROOT / "configs" / "gamus_locked_nyc_evaluation_v1.yaml"
    )
    passing = EVALUATOR.evaluate_readiness(_metrics(), protocol)
    assert passing["passes_all_predeclared_readiness_gates"] is True

    failing = EVALUATOR.evaluate_readiness(_metrics(water_f1=0.01), protocol)
    assert failing["passes_all_predeclared_readiness_gates"] is False
    assert failing["checks"]["water_f1"]["passes"] is False


def test_locked_readiness_requires_every_class_and_nonzero_support() -> None:
    _, protocol = EVALUATOR.load_protocol(
        ROOT / "configs" / "gamus_locked_nyc_evaluation_v1.yaml"
    )
    metrics = _metrics()
    del metrics["six_class_identification"]["per_class"]["roads"]
    with pytest.raises(ValueError, match="all six classes"):
        EVALUATOR.evaluate_readiness(metrics, protocol)

    metrics = _metrics()
    metrics["six_class_identification"]["per_class"]["water"]["support_pixels"] = 0
    with pytest.raises(ValueError, match="no reference support"):
        EVALUATOR.evaluate_readiness(metrics, protocol)


def test_consumption_requires_confirmation_before_any_holdout_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_dataset(*args, **kwargs):
        raise AssertionError("holdout dataset must not be constructed")

    monkeypatch.setattr(EVALUATOR, "GamusSurfaceDataset", forbidden_dataset)
    with pytest.raises(ValueError, match="confirmation phrase"):
        EVALUATOR.consume_and_evaluate(
            protocol_path=ROOT / "configs" / "gamus_locked_nyc_evaluation_v1.yaml",
            candidate_checkpoint=tmp_path / "checkpoint_best_landscape.pt",
            output_root=tmp_path / "eval",
            confirmation="",
        )


def test_existing_consumption_marker_refuses_second_pass_before_data_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "eval"
    output.mkdir()
    (output / "freeze_manifest.json").write_text("{}", encoding="utf-8")
    (output / "holdout_consumed.json").write_text("{}", encoding="utf-8")

    def forbidden_dataset(*args, **kwargs):
        raise AssertionError("holdout dataset must not be constructed")

    monkeypatch.setattr(EVALUATOR, "GamusSurfaceDataset", forbidden_dataset)
    with pytest.raises(FileExistsError, match="already consumed"):
        EVALUATOR.consume_and_evaluate(
            protocol_path=ROOT / "configs" / "gamus_locked_nyc_evaluation_v1.yaml",
            candidate_checkpoint=tmp_path / "checkpoint_best_landscape.pt",
            output_root=output,
            confirmation="CONSUME_LOCKED_NYC_HOLDOUT_ONCE",
        )


def test_locked_artifacts_are_write_once(tmp_path: Path) -> None:
    path = tmp_path / "marker.json"
    EVALUATOR._write_once(path, {"value": 1}, accept_identical=True)
    EVALUATOR._write_once(path, {"value": 1}, accept_identical=True)
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
    with pytest.raises(FileExistsError, match="already exists"):
        EVALUATOR._write_once(path, {"value": 2}, accept_identical=True)


def test_locked_evaluator_selects_the_declared_isolated_head_auditor() -> None:
    assert EVALUATOR._head_state_auditor({"model": {}}) is (
        EVALUATOR.audit_linear_head_states
    )
    assert EVALUATOR._head_state_auditor(
        {"model": {"fine_semantic_head_type": "spatial_refined"}}
    ) is EVALUATOR.audit_spatial_refined_head_states
    with pytest.raises(ValueError, match="unsupported locked-evaluation"):
        EVALUATOR._head_state_auditor(
            {"model": {"fine_semantic_head_type": "mystery"}}
        )
