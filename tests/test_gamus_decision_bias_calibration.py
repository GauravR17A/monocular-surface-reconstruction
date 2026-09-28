from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import re

import numpy as np
import pytest
import yaml

from msr.data.gamus_dataset import GAMUS_SIX_CLASS_NAMES
from msr.evaluation.decision_bias_calibration import (
    DeterministicBoundedClassSampler,
    centered_biases,
    confusion_from_logits,
    decision_gate_report,
    fit_additive_decision_biases,
)


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "calibrate_gamus_six_class_decisions.py"
SPEC = importlib.util.spec_from_file_location("gamus_decision_bias_test", SCRIPT)
assert SPEC and SPEC.loader
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _gates() -> dict[str, object]:
    return {
        "objective": "macro_f1",
        "minimum_objective_gain": 0.01,
        "absolute_minimums": {
            "macro_f1": 0.95,
            "macro_iou": 0.90,
            "per_class": {
                "water": {"precision": 0.90, "recall": 0.90, "f1": 0.90}
            },
        },
        "maximum_drops_from_uncalibrated": {
            "macro_f1": 0.0,
            "per_class": {
                name: {"f1": 0.0}
                for name in GAMUS_SIX_CLASS_NAMES
                if name != "water"
            },
        },
    }


def _recoverable_water_logits() -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for class_index in range(6):
        for _ in range(40):
            logits = np.full(6, -2.0, dtype=np.float32)
            logits[class_index] = 2.0
            if class_index == 2:
                logits[2] = 1.0
                logits[4] = 1.4
            rows.append(logits)
            labels.append(class_index)
    return np.stack(rows), np.asarray(labels, dtype=np.int64)


def test_fit_recovers_water_with_external_mean_zero_decision_biases() -> None:
    logits, target = _recoverable_water_logits()
    original = logits.copy()
    result = fit_additive_decision_biases(
        logits,
        target,
        GAMUS_SIX_CLASS_NAMES,
        gate_config=_gates(),
        maximum_absolute_bias=2.0,
        step_schedule=(0.5, 0.25, 0.1, 0.05),
        maximum_sweeps_per_step=4,
    )

    assert result["gate_report"]["passes_all"] is True
    assert result["calibrated_metrics"]["per_class"]["water"]["f1"] == 1.0
    assert abs(sum(result["biases"])) < 1.0e-12
    assert result["changes_model_or_height_tensors"] is False
    assert result["probability_confidence_calibration_performed"] is False
    np.testing.assert_array_equal(logits, original)


def test_weighted_confusion_preserves_original_class_prior_estimate() -> None:
    logits = np.array(
        [
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
        ]
    )
    target = np.array([0, 0, 1, 1])
    weights = np.array([50.0, 50.0, 5.0, 5.0])
    confusion = confusion_from_logits(logits, target, (0.0, 0.0), weights)
    np.testing.assert_array_equal(confusion, np.array([[100, 0], [5, 5]]))


def test_bounded_sampler_is_deterministic_and_hard_capped() -> None:
    labels = np.arange(36, dtype=np.int64).reshape(6, 6) % 6
    logits = np.stack(
        [np.full(labels.shape, value, dtype=np.float32) for value in range(6)]
    )
    valid = np.ones_like(labels, dtype=bool)

    def run():
        sampler = DeterministicBoundedClassSampler(
            GAMUS_SIX_CLASS_NAMES,
            seed=91,
            per_tile_class_cap=4,
            maximum_pixels_per_class=5,
        )
        sampler.update(logits, labels, valid, sample_id="DC_a")
        sampler.update(logits + 0.25, labels, valid, sample_id="PHL_b")
        return sampler.finalize()

    logits_a, target_a, weight_a, provenance_a = run()
    logits_b, target_b, weight_b, provenance_b = run()
    assert logits_a.shape == (30, 6)
    assert provenance_a["retained_pixels"] == 30
    assert provenance_a["hard_maximum_cached_pixels"] == 30
    assert provenance_a == provenance_b
    np.testing.assert_array_equal(logits_a, logits_b)
    np.testing.assert_array_equal(target_a, target_b)
    np.testing.assert_array_equal(weight_a, weight_b)


def test_gate_report_fails_closed_on_class_regression() -> None:
    logits, target = _recoverable_water_logits()
    baseline_confusion = confusion_from_logits(logits, target, np.zeros(6))
    from msr.evaluation.classification_metrics import compute_multiclass_metrics

    baseline = compute_multiclass_metrics(baseline_confusion, GAMUS_SIX_CLASS_NAMES)
    degraded = dict(baseline)
    degraded["per_class"] = {
        name: dict(values) for name, values in baseline["per_class"].items()
    }
    degraded["macro_f1"] = baseline["macro_f1"] + 0.02
    degraded["macro_iou"] = 0.95
    degraded["per_class"]["water"].update(
        {"precision": 0.95, "recall": 0.95, "f1": 0.95}
    )
    degraded["per_class"]["buildings"]["f1"] -= 0.02
    report = decision_gate_report(
        degraded, baseline, _gates(), GAMUS_SIX_CLASS_NAMES
    )
    assert report["passes_all"] is False
    assert report["checks"]["buildings_f1_retention"]["passes"] is False


def test_centered_biases_reject_invalid_or_out_of_bound_values() -> None:
    centered = centered_biases([1.0, 2.0], class_count=2, maximum_absolute_bias=1.0)
    np.testing.assert_allclose(centered, [-0.5, 0.5])
    with pytest.raises(ValueError, match="exceed"):
        centered_biases([0.0, 4.0], class_count=2, maximum_absolute_bias=1.0)
    with pytest.raises(ValueError, match="finite"):
        centered_biases([0.0, np.nan], class_count=2, maximum_absolute_bias=1.0)


def test_protocol_is_dc_phl_only_and_explicitly_not_confidence_calibration() -> None:
    _, protocol = TOOL.validate_protocol(
        ROOT / "configs" / "gamus_six_class_decision_bias_v1.yaml"
    )
    assert protocol["data"]["source_split"] == "val"
    assert protocol["data"]["allowed_cities"] == ["DC", "PHL"]
    assert "NYC" in protocol["data"]["forbidden_cities"]
    assert protocol["safety"]["official_test_policy"] == "never_constructed_or_discovered"
    assert protocol["safety"]["probability_confidence_calibration"] is False
    assert protocol["safety"]["application_model_change"] == "forbidden"


def test_spatial_v3_calibration_protocol_is_valid_and_never_promotes() -> None:
    _, protocol = TOOL.validate_protocol(
        ROOT
        / "configs"
        / "gamus_six_class_spatial_refined_v3_decision_bias_v1.yaml"
    )
    assert protocol["candidate"]["required_tensor_audit_schema"] == (
        "msr.gamus_spatial_refined_head_checkpoint_audit.v1"
    )
    assert protocol["safety"]["application_model_change"] == "forbidden"
    assert protocol["safety"]["auto_promotion"] is False
    assert protocol["data"]["forbidden_cities"] == ["NYC"]


def test_calibrator_supports_both_isolated_six_class_head_architectures() -> None:
    linear_config = yaml.safe_load(
        (ROOT / "configs" / "multidomain_surface_gamus_six_class_head_only_v2.yaml")
        .read_text(encoding="utf-8")
    )
    spatial_config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "multidomain_surface_gamus_six_class_spatial_refined_v3.yaml"
        ).read_text(encoding="utf-8")
    )

    linear = TOOL._head_contract(linear_config)
    spatial = TOOL._head_contract(spatial_config)

    assert linear["head_type"] == "linear"
    assert linear["allowed_trained_tensors"] == TOOL.LINEAR_HEAD_KEYS
    assert linear["audit_schema"] == "msr.gamus_head_only_checkpoint_audit.v1"
    assert spatial["head_type"] == "spatial_refined"
    assert spatial["allowed_trained_tensors"] == TOOL.SPATIAL_REFINED_HEAD_KEYS
    assert (
        spatial["audit_schema"]
        == "msr.gamus_spatial_refined_head_checkpoint_audit.v1"
    )


def test_calibrator_rejects_unknown_isolated_head_architecture() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "multidomain_surface_gamus_six_class_head_only_v2.yaml")
        .read_text(encoding="utf-8")
    )
    config["model"]["fine_semantic_head_type"] = "mystery"
    with pytest.raises(ValueError, match="unsupported isolated"):
        TOOL._head_contract(config)


def test_spatial_v3_calibration_runner_pins_protocol_and_calibrator() -> None:
    runner = (
        ROOT / "scripts" / "run_gamus_spatial_v3_decision_bias.ps1"
    ).read_text(encoding="utf-8")
    expected = {
        "expectedCalibratorSha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        "expectedProtocolSha256": hashlib.sha256(
            (
                ROOT
                / "configs"
                / "gamus_six_class_spatial_refined_v3_decision_bias_v1.yaml"
            ).read_bytes()
        ).hexdigest(),
    }
    for variable, digest in expected.items():
        match = re.search(rf'\${variable} = "([0-9a-f]{{64}})"', runner)
        assert match is not None
        assert match.group(1) == digest
    assert "-RunFullValidation" in runner
    assert "NYC and official test remain unopened" in runner
    assert "showcase_checkpoint.txt" in runner


def test_protocol_rejects_nyc_as_an_allowed_calibration_city(tmp_path: Path) -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs" / "gamus_six_class_decision_bias_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    protocol["data"]["allowed_cities"] = ["DC", "PHL", "NYC"]
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=r"allow only DC\+PHL"):
        TOOL.validate_protocol(path)


def test_full_evaluation_requires_confirmation_before_any_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path, protocol = TOOL.validate_protocol(
        ROOT / "configs" / "gamus_six_class_decision_bias_v1.yaml"
    )

    def forbidden_dataset(*args, **kwargs):
        raise AssertionError("dataset must not be constructed before confirmation")

    monkeypatch.setattr(TOOL, "_make_dataset", forbidden_dataset)
    with pytest.raises(ValueError, match="confirmation phrase"):
        TOOL.evaluate_full_once(
            protocol_path,
            protocol,
            ROOT / "missing_checkpoint_best_landscape.pt",
            ROOT / "missing_fit.json",
            "",
        )
