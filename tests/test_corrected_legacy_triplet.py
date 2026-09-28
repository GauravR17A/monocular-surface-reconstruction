from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "evaluate_corrected_legacy_triplet.py"
)
SPEC = importlib.util.spec_from_file_location("corrected_legacy_triplet_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)

PLAN_PATH = Path(__file__).parents[1] / "configs" / "corrected_legacy_triplet_v1.yaml"


def _metric(value: float) -> dict[str, object]:
    return {
        "pixel_count": 10,
        "rmse_m": value,
        "mae_m": value - 1.0,
        "bias_m": 0.0,
        "correlation": 0.5,
        "r2": 0.2,
        "landscapes": {
            "urban": {"rmse_m": value},
            "forest": {"rmse_m": value},
        },
        "domains": {
            "building": {"rmse_m": value},
            "vegetation": {"rmse_m": value},
        },
        "regions": {},
    }


def test_frozen_plan_forbids_official_test_and_uses_exact_model_order(
    tmp_path: Path,
) -> None:
    plan = SCRIPT.load_plan(PLAN_PATH)
    assert tuple(plan["models"]) == SCRIPT.MODEL_NAMES
    assert plan["protocol"]["official_test_used"] is False
    assert plan["data"]["relative_prior_policy"] == "stored_01"

    unsafe = copy.deepcopy(plan)
    unsafe["protocol"]["official_test_used"] = True
    path = tmp_path / "unsafe.yaml"
    path.write_text(yaml.safe_dump(unsafe, sort_keys=False), encoding="utf-8")
    with pytest.raises(SCRIPT.CorrectedTripletError, match="Official test"):
        SCRIPT.load_plan(path)


def test_metric_scopes_use_one_mask_for_all_requested_slices() -> None:
    book = SCRIPT.MetricScopes()
    prediction = np.array([[1.0, 4.0], [8.0, 10.0]], dtype=np.float32)
    target = np.array([[1.0, 2.0], [6.0, 20.0]], dtype=np.float32)
    mask = np.array([[True, True], [True, False]])
    domain = np.array(
        [
            [SCRIPT.LANDSCAPE_CLASSES["ground"], SCRIPT.LANDSCAPE_CLASSES["building"]],
            [SCRIPT.LANDSCAPE_CLASSES["vegetation"], SCRIPT.LANDSCAPE_CLASSES["ground"]],
        ]
    )
    book.update(
        prediction,
        target,
        mask,
        landscape="urban",
        region="one",
        domain_target=domain,
    )
    result = book.compute()

    assert result["pixel_count"] == 3
    assert result["rmse_m"] == pytest.approx(np.sqrt(8.0 / 3.0))
    assert result["landscapes"]["urban"]["pixel_count"] == 3
    assert result["domains"]["ground"]["pixel_count"] == 1
    assert result["domains"]["building"]["rmse_m"] == pytest.approx(2.0)
    assert result["domains"]["vegetation"]["rmse_m"] == pytest.approx(2.0)


def _sample() -> dict[str, torch.Tensor]:
    return {
        "image": torch.zeros((3, 2, 2)),
        "image_valid_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        "building_mask": torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
        "vegetation_mask": torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]),
        "domain_target": torch.tensor([[0, 1], [2, 0]]),
        "domain_valid_mask": torch.ones((2, 2), dtype=torch.bool),
        "relative_prior": torch.zeros((1, 2, 2)),
        "height": torch.tensor([[[0.0, 3.0], [2.0, 0.0]]]),
        "regression_mask": torch.tensor([[[False, True], [True, False]]]),
    }


def test_highbuild_protocol_views_share_model_inputs_and_nest_height_support() -> None:
    inclusive = _sample()
    strict = {name: value.clone() for name, value in inclusive.items()}
    strict["regression_mask"][0, 1, 0] = False
    strict["height"][0, 1, 0] = 0.0
    SCRIPT._assert_shared_highbuild_samples(inclusive, strict, "tile")

    changed_prior = {name: value.clone() for name, value in strict.items()}
    changed_prior["relative_prior"][0, 0, 0] = 1.0
    with pytest.raises(SCRIPT.CorrectedTripletError, match="relative_prior"):
        SCRIPT._assert_shared_highbuild_samples(inclusive, changed_prior, "tile")

    non_nested = {name: value.clone() for name, value in strict.items()}
    non_nested["regression_mask"][0, 0, 0] = True
    with pytest.raises(SCRIPT.CorrectedTripletError, match="not a subset"):
        SCRIPT._assert_shared_highbuild_samples(inclusive, non_nested, "tile")


def test_candidate_gate_comparison_is_fail_closed() -> None:
    gates = {
        "rmse_m": 0.15,
        "mae_m": 0.10,
        "landscapes.forest.rmse_m": 0.15,
    }
    passing = SCRIPT.compare_candidate(_metric(5.1), _metric(5.0), gates)
    assert passing["passes"] is True

    failing = SCRIPT.compare_candidate(_metric(5.2), _metric(5.0), gates)
    assert failing["passes"] is False
    assert failing["checks"]["rmse_m"]["candidate_minus_reference"] == pytest.approx(
        0.2
    )


def test_checkpoint_authentication_requires_exact_embedded_model_config(
    tmp_path: Path,
) -> None:
    models: dict[str, dict[str, object]] = {}
    comparison: dict[str, object] = {}
    for index, name in enumerate(SCRIPT.MODEL_NAMES, start=1):
        expected_model = {
            "base_checkpoint": "base.pt",
            "fusion_mode": "protected_vegetation" if name == "protected" else "legacy",
            "fine_semantic_classes": 6 if name == "expanded_candidate" else 0,
        }
        checkpoint = tmp_path / f"{name}.pt"
        torch.save(
            {
                "model_type": f"type-{name}",
                "epoch": index,
                "model": {},
                "config": {"model": expected_model},
            },
            checkpoint,
        )
        digest = SCRIPT.file_sha256(checkpoint)
        section = f"section-{name}"
        models[name] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "epoch": index,
            "model_type": f"type-{name}",
            "comparison_section": section,
            "expected_model_config": expected_model,
        }
        comparison[section] = {
            "selected_checkpoint_sha256": digest,
            "selected_epoch": index,
        }
    plan = {"models": models}

    result = SCRIPT.authenticate_models(plan, comparison)
    assert result["expanded_candidate"]["model_config"]["fine_semantic_classes"] == 6

    bad = copy.deepcopy(plan)
    bad["models"]["height_pilot"]["expected_model_config"]["fusion_mode"] = (
        "protected_vegetation"
    )
    with pytest.raises(SCRIPT.CorrectedTripletError, match="exact config"):
        SCRIPT.authenticate_models(bad, comparison)


def test_reusable_partial_requires_complete_run_identity(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    identity = {
        "plan_sha256": "a",
        "model_name": "protected",
        "checkpoint_sha256": "b",
        "device": "cuda",
        "precision": "bf16",
        "highbuild_limit": None,
        "open_canopy_limit": None,
    }
    path.write_text(
        json.dumps({"schema": SCRIPT.PARTIAL_SCHEMA, "run_identity": identity}),
        encoding="utf-8",
    )
    assert SCRIPT._load_reusable_partial(path, identity)["run_identity"] == identity

    changed = {**identity, "precision": "fp32"}
    with pytest.raises(SCRIPT.CorrectedTripletError, match="do not reuse"):
        SCRIPT._load_reusable_partial(path, changed)
