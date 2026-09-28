from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
import yaml

from msr.models.routed_surface import ConservativeSceneRouter
from msr.training.scene_router import SceneUtilityConfig


EXPECTED_NO_REFERENCE_KEYS = (
    "gamus/train/NYC_23371",
    "gamus/train/NYC_26939",
    "gamus/train/NYC_28048",
    "gamus/train/NYC_28140",
    "gamus/train/NYC_28325",
    "gamus/train/NYC_28971",
    "gamus/train/NYC_30468",
    "gamus/train/NYC_30653",
    "gamus/train/NYC_31575",
)


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "build_stage3_scene_router_records.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_stage3_scene_router_records_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config() -> dict:
    return {
        "record_generation": {
            "output_dir": "outputs/router",
            "seed": 7,
            "patch_size": 384,
            "crop": "center",
            "batch_size": 4,
            "num_workers": 0,
            "precision": "bf16",
        },
        "endpoints": {
            "protected_checkpoint": "protected.pt",
            "protected_checkpoint_sha256": "a" * 64,
            "gamus_stage1_checkpoint": "candidate.pt",
            "gamus_stage1_checkpoint_sha256": "b" * 64,
            "base_checkpoint_sha256": "c" * 64,
        },
        "data": {
            "gamus_root": "GAMUS",
            "gamus_relative_prior_root": "GAMUS-priors",
            "legacy_train_manifest": "train.csv",
            "legacy_val_manifest": "val.csv",
            "legacy_train_manifest_sha256": "d" * 64,
            "legacy_val_manifest_sha256": "e" * 64,
            "legacy_train_expected_count": 1050,
            "legacy_val_expected_count": 280,
            "gamus_train_inventory_sha256": "1" * 64,
            "gamus_val_inventory_sha256": "2" * 64,
            "legacy_train_inventory_sha256": "3" * 64,
            "legacy_val_inventory_sha256": "4" * 64,
            "rgb_scale": 255.0,
            "height_max_m": 200.0,
            "building_threshold_m": 2.0,
            "radiometric_policy": "raw",
            "legacy_relative_prior_policy": "stored_01",
            "excluded_no_reference_record_keys": ["gamus/train/a"],
        },
        "utility": {
            "height_rmse_weight": 1.0,
            "semantic_error_weight": 1.0,
            "height_scale_m": 10.0,
            "minimum_candidate_gain": 0.02,
        },
    }


def test_config_hard_locks_center_384_bf16_raw_and_forbids_test_data() -> None:
    MODULE.validate_generation_config(_config())
    for section, key, value, message in (
        ("record_generation", "patch_size", 256, "384px"),
        ("record_generation", "crop", "random", "center crops"),
        ("record_generation", "precision", "fp32", "bf16"),
        ("data", "radiometric_policy", "percentile", "raw radiometry"),
    ):
        config = _config()
        config[section][key] = value
        with pytest.raises(ValueError, match=message):
            MODULE.validate_generation_config(config)
    config = _config()
    config["data"]["legacy_test_manifest"] = "forbidden.csv"
    with pytest.raises(ValueError, match="test data configuration is forbidden"):
        MODULE.validate_generation_config(config)


def test_record_keys_are_namespaced_and_data_contract_lists_no_test_split() -> None:
    assert MODULE.scene_router_record_key("gamus", "train", "scene-1") == (
        "gamus/train/scene-1"
    )
    assert MODULE.ALLOWED_SOURCE_SPLITS == (
        ("gamus", "train", "train"),
        ("gamus", "val", "calibration"),
        ("legacy", "train", "train"),
        ("legacy", "val", "calibration"),
    )
    with pytest.raises(ValueError, match="path separators"):
        MODULE.scene_router_record_key("gamus", "test", "bad/id")


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("gamus/train/a", "must be a list"),
        ([], "cannot be empty"),
        (["gamus/train/a", "gamus/train/a"], "duplicate"),
        (["NYC_23371"], "namespaced"),
        (["GAMUS/train/NYC_23371"], "canonical"),
        (["gamus/val/NYC_23371"], "only train"),
        (["gamus/test/NYC_23371"], "outside the allowed source splits"),
    ),
)
def test_no_reference_exclusion_list_fails_closed(value: object, message: str) -> None:
    config = _config()
    config["data"]["excluded_no_reference_record_keys"] = value
    with pytest.raises(ValueError, match=message):
        MODULE.validate_generation_config(config)


def test_no_reference_exclusions_must_be_a_subset_of_the_plan() -> None:
    config = _config()
    dataset = [
        {
            "sample_id": "a",
            "regression_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
            "domain_valid_mask": torch.zeros(2, 2, dtype=torch.bool),
        }
    ]
    plans = (
        MODULE.RouterDatasetPlan(
            "gamus", "train", "train", dataset, ("a",)
        ),
    )
    assert MODULE.validate_no_reference_exclusions_against_plan(config, plans) == (
        "gamus/train/a",
    )

    config["data"]["excluded_no_reference_record_keys"] = [
        "gamus/train/not-in-plan"
    ]
    with pytest.raises(ValueError, match="outside the Stage-3 data plan"):
        MODULE.validate_no_reference_exclusions_against_plan(config, plans)


@pytest.mark.parametrize("mask_name", ("regression_mask", "domain_valid_mask"))
def test_no_reference_exclusion_must_have_zero_support(mask_name: str) -> None:
    config = _config()
    sample = {
        "sample_id": "a",
        "regression_mask": torch.zeros(1, 2, 2, dtype=torch.bool),
        "domain_valid_mask": torch.zeros(2, 2, dtype=torch.bool),
    }
    sample[mask_name] = torch.ones_like(sample[mask_name])
    plans = (
        MODULE.RouterDatasetPlan("gamus", "train", "train", [sample], ("a",)),
    )
    with pytest.raises(ValueError, match="scoreable reference support"):
        MODULE.validate_no_reference_exclusions_against_plan(config, plans)


def test_checked_in_exclusions_and_scored_counts_are_exact() -> None:
    config_path = (
        Path(__file__).parents[1] / "configs" / "stage3_scene_router_records.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert tuple(config["data"]["excluded_no_reference_record_keys"]) == (
        EXPECTED_NO_REFERENCE_KEYS
    )
    gamus_train_ids = tuple(key.rsplit("/", 1)[1] for key in EXPECTED_NO_REFERENCE_KEYS)
    gamus_train_ids += tuple(f"synthetic-gamus-train-{index}" for index in range(4995))
    plans = (
        MODULE.RouterDatasetPlan(
            "gamus", "train", "train", range(5004), gamus_train_ids
        ),
        MODULE.RouterDatasetPlan(
            "gamus",
            "val",
            "calibration",
            range(859),
            tuple(f"synthetic-gamus-val-{index}" for index in range(859)),
        ),
        MODULE.RouterDatasetPlan(
            "legacy",
            "train",
            "train",
            range(1050),
            tuple(f"synthetic-legacy-train-{index}" for index in range(1050)),
        ),
        MODULE.RouterDatasetPlan(
            "legacy",
            "val",
            "calibration",
            range(280),
            tuple(f"synthetic-legacy-val-{index}" for index in range(280)),
        ),
    )
    selection = MODULE.record_selection_provenance(config, plans)
    assert selection["planned_record_count"] == 7193
    assert selection["excluded_no_reference_count"] == 9
    assert selection["scored_record_count"] == 7184
    assert selection["scored_partition_counts"] == {
        "train": 6045,
        "calibration": 1139,
    }


class _Shared(nn.Module):
    def forward(
        self, image: torch.Tensor, prior: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch, _, height, width = image.shape
        features = image[:, :1]
        base_height = torch.ones(
            batch, 1, height, width, device=image.device, dtype=image.dtype
        )
        prior_logits = torch.zeros(
            batch, 3, height, width, device=image.device, dtype=image.dtype
        )
        zeros = torch.zeros_like(base_height)
        return {
            "adapter_features": features,
            "prior_domain_logits": prior_logits,
            "base_height": base_height,
            "protected_building_logits": zeros,
            "refinement_logits": zeros,
            "log_variance": zeros,
            "height": base_height,
            "domain_logits": prior_logits,
            "building_height": base_height,
            "canopy_height": base_height,
        }


class _Endpoint(nn.Module):
    def __init__(self, height_offset: float) -> None:
        super().__init__()
        self.height_offset = height_offset

    def forward_fused(
        self,
        features: torch.Tensor,
        prior_logits: torch.Tensor,
        base_height: torch.Tensor,
        **_: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        height = (base_height + self.height_offset).clamp_min(0)
        return {
            "height": height,
            "domain_logits": prior_logits,
            "building_height": base_height,
            "canopy_height": base_height,
        }


def _batch() -> dict:
    return {
        "image": torch.tensor(
            [
                [[[1.0, 2.0], [3.0, 4.0]]] * 3,
                [[[4.0, 3.0], [2.0, 1.0]]] * 3,
            ]
        ),
        "relative_prior": torch.zeros(2, 1, 2, 2),
        "height": torch.zeros(2, 1, 2, 2),
        "valid_mask": torch.ones(2, 1, 2, 2, dtype=torch.bool),
        "regression_mask": torch.ones(2, 1, 2, 2, dtype=torch.bool),
        "domain_target": torch.zeros(2, 2, 2, dtype=torch.long),
        "domain_valid_mask": torch.ones(2, 2, 2, dtype=torch.bool),
        "sample_id": ["a", "b"],
        "landscape": ["urban", "forest"],
    }


def test_gpu_batch_builder_uses_exact_fused_endpoints_and_reference_utility() -> None:
    records = MODULE.build_records_from_batch(
        batch=_batch(),
        source="legacy",
        split="train",
        partition="train",
        shared_model=_Shared(),
        protected_endpoint=_Endpoint(0.0),
        candidate_endpoint=_Endpoint(-1.0),
        descriptor_router=ConservativeSceneRouter(1),
        utility_config=SceneUtilityConfig(),
        device=torch.device("cpu"),
        precision="bf16",
    )

    assert [record.sample_id for record in records] == [
        "legacy/train/a",
        "legacy/train/b",
    ]
    assert [record.landscape for record in records] == ["urban", "forest"]
    assert all(record.descriptor.shape == (2,) for record in records)
    assert all(record.fallback_utility > record.candidate_utility for record in records)
    assert all(record.decision(0.02).candidate_target for record in records)


def test_batch_builder_fails_before_commit_if_protected_fusion_drifts() -> None:
    with pytest.raises(RuntimeError, match="protected fused endpoint drifted"):
        MODULE.build_records_from_batch(
            batch=_batch(),
            source="gamus",
            split="val",
            partition="calibration",
            shared_model=_Shared(),
            protected_endpoint=_Endpoint(0.25),
            candidate_endpoint=_Endpoint(-1.0),
            descriptor_router=ConservativeSceneRouter(1),
            utility_config=SceneUtilityConfig(),
            device=torch.device("cpu"),
            precision="bf16",
        )


def test_scene_descriptor_never_depends_on_reference_validity_mask() -> None:
    baseline = _batch()
    altered = _batch()
    altered["valid_mask"] = torch.tensor(
        [
            [[[True, False], [False, False]]],
            [[[False, False], [False, True]]],
        ]
    )
    common = {
        "source": "gamus",
        "split": "train",
        "partition": "train",
        "shared_model": _Shared(),
        "protected_endpoint": _Endpoint(0.0),
        "candidate_endpoint": _Endpoint(-1.0),
        "descriptor_router": ConservativeSceneRouter(1),
        "utility_config": SceneUtilityConfig(),
        "device": torch.device("cpu"),
        "precision": "bf16",
    }
    left = MODULE.build_records_from_batch(batch=baseline, **common)
    right = MODULE.build_records_from_batch(batch=altered, **common)
    assert all(
        torch.equal(a.descriptor, b.descriptor) for a, b in zip(left, right)
    )


def test_both_stage3_runners_configure_deterministic_cublas() -> None:
    project_root = Path(__file__).parents[1]
    for name in (
        "run_stage3_scene_router_records.ps1",
        "run_stage3_scene_router_training.ps1",
    ):
        content = (project_root / "scripts" / name).read_text(encoding="utf-8")
        assert '$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"' in content
