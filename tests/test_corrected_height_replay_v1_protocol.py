from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "multidomain_surface_corrected_height_replay_v1.yaml"
PREFLIGHT = ROOT / "scripts" / "preflight_corrected_height_replay_v1.py"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_protocol_is_non_promotable_and_never_uses_official_test() -> None:
    config = _config()
    assert config["contracts"]["official_test_used"] is False
    assert config["evaluation"]["promotion_eligible"] is False
    assert config["data"]["dataset"] == "mixed_replay"
    assert not any("test" in str(key).lower() for key in config["data"])


def test_protocol_keeps_validity_and_prior_contracts_explicit() -> None:
    config = _config()
    assert config["contracts"]["missing_height_policy"] == "ignored_never_zero"
    assert set(config["contracts"]["gamus_regression_source_class_ids"]) == {
        1,
        2,
        3,
        6,
    }
    assert set(config["contracts"]["gamus_excluded_regression_source_class_ids"]) == {
        0,
        4,
        5,
    }
    assert config["data"]["legacy_relative_prior_policy"] == "stored_01"
    assert config["data"]["legacy_supervised_crop_probability"] == 1.0


def test_protocol_trains_only_small_final_height_and_fusion_heads() -> None:
    config = _config()
    prefixes = {
        prefix
        for group in config["training"]["parameter_groups"]
        for prefix in group["prefixes"]
    }
    assert prefixes == {
        "base_model.height_head.",
        "canopy_height_head.",
        "refinement_strength_head.",
    }
    checkpoint = Path(config["model"]["initial_checkpoint"])
    payload = torch.load(ROOT / checkpoint, map_location="cpu", weights_only=False)
    trainable = {
        name
        for name in payload["model"]
        if any(name.startswith(prefix) for prefix in prefixes)
    }
    assert trainable
    assert all(
        any(name.startswith(prefix) for prefix in prefixes) for name in trainable
    )


def test_preflight_module_imports_without_side_effects() -> None:
    spec = importlib.util.spec_from_file_location("corrected_replay_preflight_test", PREFLIGHT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.EXPECTED_PREFIXES == {
        "base_model.height_head.",
        "canopy_height_head.",
        "refinement_strength_head.",
    }
