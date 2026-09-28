import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "audit_gamus_head_only_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("head_only_audit_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _base_state() -> dict[str, torch.Tensor]:
    return {
        "base.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "domain_head.bias": torch.tensor([1.0, 2.0, 3.0]),
    }


def _candidate_state() -> dict[str, torch.Tensor]:
    return {
        **_base_state(),
        "fine_semantic_head.weight": torch.ones(6, 2, 1, 1),
        "fine_semantic_head.bias": torch.arange(6, dtype=torch.float32),
    }


def test_head_only_audit_accepts_exact_frozen_inherited_tensors() -> None:
    result = MODULE.audit_states(_base_state(), _candidate_state())
    assert result["passes"] is True
    assert result["changed_inherited_tensors"] == []
    assert set(result["allowed_trained_tensors"]) == MODULE.ALLOWED_NEW_KEYS


def test_head_only_audit_rejects_changed_inherited_tensor() -> None:
    candidate = _candidate_state()
    candidate["base.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="height/shared tensors changed"):
        MODULE.audit_states(_base_state(), candidate)


def test_head_only_audit_rejects_extra_or_untrained_head() -> None:
    extra = _candidate_state()
    extra["mystery.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="unexpected new tensors"):
        MODULE.audit_states(_base_state(), extra)

    zero_head = _candidate_state()
    zero_head["fine_semantic_head.weight"].zero_()
    with pytest.raises(ValueError, match="was not trained"):
        MODULE.audit_states(_base_state(), zero_head)


def test_build_report_does_not_claim_an_unobserved_pointer_result(tmp_path: Path) -> None:
    base_path = tmp_path / "base.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save({"model": _base_state(), "epoch": 1}, base_path)
    torch.save(
        {
            "model": _candidate_state(),
            "epoch": 2,
            "metrics": {},
            "config": {
                "model": {"fine_semantic_classes": 6},
                "training": {
                    "parameter_groups": [
                        {"prefixes": ["fine_semantic_head."]}
                    ]
                },
            },
        },
        candidate_path,
    )

    report = MODULE.build_report(base_path, candidate_path)

    assert "app_pointer_changed" not in report
    assert report["app_pointer_verification"] == {
        "status": "external_wrapper_required",
        "performed_by_this_auditor": False,
    }


def test_classification_acceptance_checks_recovery_and_retention() -> None:
    names = ["ground", "buildings", "water", "roads", "low_vegetation", "trees"]
    current = {
        "six_class_identification": {
            "macro_f1": 0.50,
            "per_class": {
                name: {"f1": 0.60, "recall": 0.60} for name in names
            },
        }
    }
    baseline = {
        "expanded_classification": {
            "macro": {"f1": 0.467},
            "per_class": {name: {"f1": 0.595} for name in names},
        }
    }
    thresholds = {
        "macro_f1_min": 0.4723,
        "water_recall_min": 0.10,
        "water_f1_min": 0.10,
        "road_f1_min": 0.522,
        "max_f1_drop_other_classes": 0.005,
    }
    result = MODULE.classification_acceptance(current, baseline, thresholds)
    assert result["passes"] is True

    current["six_class_identification"]["per_class"]["water"]["recall"] = 0.09
    result = MODULE.classification_acceptance(current, baseline, thresholds)
    assert result["passes"] is False
    assert result["gates"]["water_recall"]["passes"] is False
