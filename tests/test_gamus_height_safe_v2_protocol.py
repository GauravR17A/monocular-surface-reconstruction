from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "multidomain_surface_gamus_height_safe_v2.yaml"
PREFLIGHT = ROOT / "scripts" / "preflight_gamus_height_safe_v2.py"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_height_safe_v2_is_development_only_and_non_promotable() -> None:
    config = load_config()
    assert config["contracts"]["official_test_used"] is False
    assert config["evaluation"]["promotion_eligible"] is False
    assert not any("test" in str(key).lower() for key in config["data"])
    assert config["data"]["dataset"] == "gamus"
    assert "train_samples_per_epoch" not in config["data"]
    assert config["data"]["validation_patch_size"] == 1024


def test_height_safe_v2_freezes_shared_features_and_dav2_prior() -> None:
    config = load_config()
    prefixes = {
        prefix
        for group in config["training"]["parameter_groups"]
        for prefix in group["prefixes"]
    }
    assert prefixes == {"base_model.height_head.", "canopy_height_head."}
    assert config["contracts"]["depth_anything_prior_policy"] == (
        "fixed_cached_stored_01"
    )
    assert config["data"]["require_relative_priors"] is True
    assert config["model"]["fusion_mode"] == "protected_vegetation"


def test_height_safe_v2_declares_independent_mask_and_unit_contract() -> None:
    config = load_config()
    assert config["contracts"]["height_unit_status"] == "metre_assumed"
    assert config["contracts"]["raw_height_scale"] == 1.0
    assert set(config["contracts"]["regression_source_class_ids"]) == {1, 2, 3, 6}
    assert set(config["contracts"]["excluded_regression_source_class_ids"]) == {
        0,
        4,
        5,
    }
    assert config["data"]["height_max_m"] == 200.0
    assert config["data"]["train_radiometric_policy"] == "raw"


def test_preflight_module_imports_without_side_effects() -> None:
    spec = importlib.util.spec_from_file_location("height_safe_v2_preflight_test", PREFLIGHT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ALLOWED_TRAINABLE_PREFIXES == {
        "base_model.height_head.",
        "canopy_height_head.",
    }
