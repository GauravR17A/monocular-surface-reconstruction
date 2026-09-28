from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "train_residual_height.py"
SPEC = importlib.util.spec_from_file_location("train_residual_height_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _regression(rmse: float, *, count: int = 100) -> dict:
    return {
        "pixel_count": count,
        "rmse_m": rmse,
        "mae_m": 2.0,
        "bias_m": -1.0,
        "correlation": 0.5,
        "r2": 0.2,
    }


def _suite(rmse: float, *, domain: float = 8.0) -> dict:
    return {
        "loss": {},
        "metrics": {
            **_regression(rmse, count=300),
            "domains": {
                name: _regression(domain) for name in ("ground", "building", "vegetation")
            },
            "tall_objects": {},
            "semantic_identification": {
                "accuracy": 0.5,
                "macro_precision": 0.5,
                "macro_recall": 0.5,
                "macro_f1": 0.5,
                "macro_iou": 0.4,
            },
        },
    }


def _three_suites() -> dict:
    return {name: _suite(7.0) for name in ("gamus", "highbuild", "open_canopy")}


EVALUATION = {
    "maximum_domain_rmse_regression_m": 0.15,
    "maximum_mae_regression_m": 0.10,
    "maximum_absolute_bias_regression_m": 0.10,
    "minimum_material_improvement_m": 0.50,
    "minimum_material_improvement_fraction": 0.05,
}


def test_gate_requires_material_improvement_and_no_regression() -> None:
    baseline = _three_suites()
    tiny = deepcopy(baseline)
    tiny["highbuild"]["metrics"]["domains"]["building"]["rmse_m"] = 7.99
    assert not MODULE.evaluate_candidate_gate(baseline, tiny, EVALUATION)["passes"]
    win = deepcopy(baseline)
    win["highbuild"]["metrics"]["domains"]["building"]["rmse_m"] = 7.5
    result = MODULE.evaluate_candidate_gate(baseline, win, EVALUATION)
    assert result["passes"]
    assert result["priority_improvements"]["highbuild_building"]["material_improvement"]
    assert result["selection_score"] == pytest.approx((8.0 + 7.5 + 8.0) / 3.0)
    regression = deepcopy(win)
    regression["gamus"]["metrics"]["rmse_m"] = 7.2
    assert not MODULE.evaluate_candidate_gate(baseline, regression, EVALUATION)["passes"]


def test_gate_fails_closed_on_support_semantic_or_correlation_changes() -> None:
    baseline = _three_suites()
    candidate = deepcopy(baseline)
    candidate["open_canopy"]["metrics"]["domains"]["vegetation"]["rmse_m"] = 7.5
    assert MODULE.evaluate_candidate_gate(baseline, candidate, EVALUATION)["passes"]

    missing_suite = deepcopy(candidate)
    missing_suite.pop("gamus")
    assert not MODULE.evaluate_candidate_gate(baseline, missing_suite, EVALUATION)["passes"]

    changed_count = deepcopy(candidate)
    changed_count["gamus"]["metrics"]["domains"]["ground"]["pixel_count"] += 1
    assert not MODULE.evaluate_candidate_gate(baseline, changed_count, EVALUATION)["passes"]

    missing_semantic = deepcopy(candidate)
    missing_semantic["gamus"]["metrics"]["semantic_identification"].pop("macro_f1")
    assert not MODULE.evaluate_candidate_gate(baseline, missing_semantic, EVALUATION)["passes"]

    correlation_drop = deepcopy(candidate)
    correlation_drop["gamus"]["metrics"]["domains"]["ground"]["correlation"] -= 0.01
    assert not MODULE.evaluate_candidate_gate(baseline, correlation_drop, EVALUATION)["passes"]


def test_config_is_authenticated_validation_only() -> None:
    config = MODULE.load_and_validate_config(ROOT / "configs" / "residual_height_v1.yaml")
    assert config["contracts"]["official_test_used"] is False
    assert config["contracts"]["external_holdout_used_for_training_or_selection"] is False
    assert config["evaluation"]["promotion_eligible"] is False


def test_gradient_consistency_has_zero_for_identical_maps() -> None:
    import torch

    prediction = torch.rand(2, 1, 8, 8)
    batch = {
        "height": prediction.clone(),
        "regression_mask": torch.ones_like(prediction, dtype=torch.bool),
        "domain_target": torch.zeros(2, 8, 8, dtype=torch.long),
        "source": ["gamus", "legacy"],
        "landscape": ["mixed", "urban"],
    }
    loss = MODULE.gradient_consistency({"height": prediction}, batch)
    assert float(loss) == 0.0


def test_exact_sampler_yields_two_one_one_every_batch() -> None:
    sampler = MODULE.ExactThreeSourceBatchSampler((7, 5, 3), batches=23, seed=11)
    for batch in sampler:
        assert len(batch) == 4
        assert sum(index < 7 for index in batch) == 2
        assert sum(7 <= index < 12 for index in batch) == 1
        assert sum(index >= 12 for index in batch) == 1


def test_status_reporting_loader_exposes_suite_progress(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    wrapped = MODULE.StatusReportingLoader(
        ["first", "second", "third"],
        status_path=status_path,
        stage="baseline_validation",
        suite="highbuild",
        suite_index=2,
        total_suites=3,
        epoch=None,
        maximum_batches=2,
        update_every=1,
    )
    assert list(wrapped) == ["first", "second"]
    import json

    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["suite"] == "highbuild"
    assert status["completed_batches"] == 2
    assert status["total_batches"] == 2
    assert status["progress_fraction"] == 1.0


def test_validation_smoke_requires_expected_domain_only() -> None:
    results = _three_suites()
    observed = MODULE.validate_smoke_domains(results)
    assert observed["highbuild"]["domains"]["building"] == 100
    assert observed["open_canopy"]["domains"]["vegetation"] == 100

    missing_building = deepcopy(results)
    missing_building["highbuild"]["metrics"]["domains"].pop("building")
    with pytest.raises(RuntimeError, match="building support"):
        MODULE.validate_smoke_domains(missing_building)


def test_terminal_failure_replaces_stale_status(tmp_path: Path) -> None:
    MODULE.atomic_json(
        tmp_path / "status.json",
        {
            "stage": "baseline_validation",
            "suite": "highbuild",
            "completed_batches": 2,
            "total_batches": 2,
        },
    )
    MODULE.mark_terminal_failure(tmp_path, RuntimeError("empty expert"))
    import json

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (tmp_path / "training_summary.json").read_text(encoding="utf-8")
    )
    assert status["stage"] == "failed"
    assert status["failed_stage"] == "baseline_validation"
    assert status["suite"] == "highbuild"
    assert status["error_type"] == "RuntimeError"
    assert summary == status


@pytest.mark.parametrize("winerror", (5, 32, 33))
def test_atomic_replace_retries_transient_windows_reader_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, winerror: int
) -> None:
    source = tmp_path / "value.tmp"
    destination = tmp_path / "value.json"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    real_replace = MODULE.os.replace
    calls = 0

    def flaky_replace(first, second):
        nonlocal calls
        calls += 1
        if calls == 1:
            error = PermissionError(13, "reader lock")
            error.winerror = winerror
            raise error
        return real_replace(first, second)

    monkeypatch.setattr(MODULE.os, "replace", flaky_replace)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    MODULE.atomic_replace(source, destination)
    assert calls == 2
    assert destination.read_text(encoding="utf-8") == "new"


def test_atomic_replace_does_not_retry_permanent_permission_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    error = PermissionError(13, "permanent")
    error.winerror = 999
    calls = 0

    def always_fails(first, second):
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(MODULE.os, "replace", always_fails)
    with pytest.raises(PermissionError, match="permanent"):
        MODULE.atomic_replace(tmp_path / "source", tmp_path / "destination")
    assert calls == 1
