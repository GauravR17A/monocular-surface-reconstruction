from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_stage4_dual_router.py"
RUNNER = Path(__file__).parents[1] / "scripts" / "run_stage4_dual_router_training.ps1"
SPEC = importlib.util.spec_from_file_location("train_stage4_dual_router_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_checked_in_config_locks_stage4_safety_guards() -> None:
    path = Path(__file__).parents[1] / "configs" / "stage4_dual_router_training.yaml"
    config = MODULE.load_training_config(path)
    MODULE.validate_training_config(config)
    assert config["require_group_disjoint"] is True
    assert config["height"]["minimum_candidate_gain"] == pytest.approx(0.25)
    assert config["semantic"]["minimum_candidate_gain"] == pytest.approx(0.01)
    for component in ("height", "semantic"):
        head = config[component]
        assert head["minimum_precision"] >= 0.98
        assert head["minimum_candidate_selections"] >= 32
        assert head["minimum_stratum_support"] >= 32
        assert head["maximum_mean_utility_regression"] == 0.0
        assert head["maximum_stratum_utility_regression"] == 0.0


@pytest.mark.parametrize(
    ("component", "field", "value", "message"),
    [
        ("height", "minimum_candidate_gain", 0.0, "must remain 0.25"),
        ("semantic", "minimum_precision", 0.90, "at least 0.98"),
        ("height", "minimum_candidate_selections", 1, "at least 32"),
        ("semantic", "maximum_stratum_utility_regression", 0.01, "remain zero"),
    ],
)
def test_config_rejects_relaxed_fail_closed_guards(
    component: str, field: str, value: float, message: str
) -> None:
    path = Path(__file__).parents[1] / "configs" / "stage4_dual_router_training.yaml"
    config = MODULE.load_training_config(path)
    modified = copy.deepcopy(config)
    modified[component][field] = value
    with pytest.raises(ValueError, match=message):
        MODULE.validate_training_config(modified)


def test_record_evidence_rejects_test_or_descriptor_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [SimpleNamespace(descriptor=torch.zeros(128))]
    provenance = {
        "schema": MODULE.STAGE4_RECORD_STORE_SCHEMA,
        "record_schema": MODULE.STAGE4_RECORD_SCHEMA,
        "test_splits_excluded": True,
        "app_pointer_changed": False,
        "identity_sha256": "a" * 64,
        "stage3": {"authenticated": True},
        "endpoints": {
            "protected": {},
            "gamus_stage1": {},
            "compatibility": {},
        },
        "generation_config": {
            "descriptor_mask": MODULE.DESCRIPTOR_MASK_CONTRACT,
            "legacy_relative_prior_policy": MODULE.LEGACY_PRIOR_CONTRACT,
            "descriptor_source": "authenticated_stage3_records.jsonl",
            "height_evidence": (
                "float64_sse+count+rmse_on_dataset_regression_mask"
            ),
            "semantic_evidence": (
                "reference_row_prediction_column_3x3_confusion+count+macro_error"
            ),
        },
        "data": {
            "stage3_data_provenance": {
                "included_source_splits": [
                    "gamus/train",
                    "gamus/val",
                    "legacy/train",
                    "legacy/val",
                ],
                "excluded_source_splits": ["gamus/test", "legacy/test"],
            },
        },
    }
    completion = {
        "schema": MODULE.STAGE4_RECORD_STORE_SCHEMA,
        "record_schema": MODULE.STAGE4_RECORD_SCHEMA,
        "identity_sha256": "a" * 64,
        "complete": True,
        "record_count": 1,
        "descriptor_size": 128,
        "test_splits_excluded": True,
        "app_pointer_changed": False,
    }
    monkeypatch.setattr(
        MODULE, "validate_stage4_record_selection_provenance", lambda *args, **kwargs: None
    )
    MODULE.validate_record_evidence(records, provenance, completion)

    tampered = copy.deepcopy(provenance)
    tampered["data"]["stage3_data_provenance"][
        "included_source_splits"
    ].append("gamus/test")
    with pytest.raises(ValueError, match="not test-safe"):
        MODULE.validate_record_evidence(records, tampered, completion)

    tampered = copy.deepcopy(provenance)
    tampered["generation_config"]["descriptor_mask"] = "reference-valid-only"
    with pytest.raises(ValueError, match="all-pixel"):
        MODULE.validate_record_evidence(records, tampered, completion)


def test_cli_has_no_test_or_pointer_mutation_option() -> None:
    args = MODULE.parse_args([])
    assert not hasattr(args, "include_test")
    source = SCRIPT.read_text(encoding="utf-8")
    assert "showcase_checkpoint.txt" not in source
    assert "live_application_pointer_changed\": False" in source
    assert "architecture_sha256" in source
    assert "os.replace(staging_dir, output_dir)" in source


def test_training_runner_uses_the_configured_v3_record_store() -> None:
    path = Path(__file__).parents[1] / "configs" / "stage4_dual_router_training.yaml"
    config = MODULE.load_training_config(path)
    configured_name = Path(config["record_store"]).name
    runner_source = RUNNER.read_text(encoding="utf-8")
    assert configured_name == "stage4_dual_router_records_v3"
    assert f'"outputs\\{configured_name}"' in runner_source
