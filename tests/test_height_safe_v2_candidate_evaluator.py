from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "evaluate_height_safe_v2_candidate.py"
)
SPEC = importlib.util.spec_from_file_location("height_safe_v2_evaluator_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_adapter_does_not_mutate_frozen_runner_source() -> None:
    assert MODULE.MODEL_NAMES == ("protected", "height_safe_v2")
    source = SCRIPT.with_name("evaluate_corrected_legacy_triplet.py").read_text(
        encoding="utf-8"
    )
    assert 'MODEL_NAMES = ("protected", "height_pilot", "expanded_candidate")' in source


def test_build_report_keeps_scorecard_safety_contract() -> None:
    protocol = "strict"
    identity = {"same": "hash"}
    partials = {
        name: {
            "input_identity": identity,
            "checkpoint": {"sha256": name},
            "loaded_model": {},
            "open_canopy_forest_metrics": {},
            "protocols": {protocol: {"combined": {"rmse_m": 1.0}}},
        }
        for name in MODULE.MODEL_NAMES
    }
    plan = {
        "protocol": {"primary_height_protocol": protocol},
        "gates": {"candidate_vs_protected_max_regression": {"rmse_m": 0.15}},
    }
    report = MODULE.build_report(
        plan=plan,
        plan_identity={"sha256": "plan"},
        model_auth={},
        data_auth={
            "protocol_records": {protocol: []},
            "open_canopy_protocol": "open_canopy_v2",
        },
        partials=partials,
        artifacts_before={},
        artifacts_after={},
        pointer_before={},
        pointer_after={},
        complete_manifest=True,
    )
    assert report["schema"] == "msr.corrected_legacy_triplet.v1"
    assert report["official_test_used"] is False
    assert report["promotion_performed"] is False
    assert report["active_model_changed"] is False
    assert set(report["models"]) == set(MODULE.MODEL_NAMES)
