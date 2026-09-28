from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
import yaml

from msr.training.scene_router import SceneRouterRecord, SceneUtilityConfig


PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_stage4_dual_router_records.py"
SPEC = importlib.util.spec_from_file_location(
    "build_stage4_dual_router_records_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _config() -> dict:
    return yaml.safe_load(
        (PROJECT_ROOT / "configs" / "stage4_dual_router_records.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_config_locks_reviewed_city_policy_counts_and_forbids_test() -> None:
    config = _config()
    MODULE.validate_generation_config(config)
    config["controller_selection"]["gamus_train_cities"] = ["NYC", "DC"]
    with pytest.raises(ValueError, match="NYC train only"):
        MODULE.validate_generation_config(config)

    config = _config()
    config["record_generation"]["batch_size"] = 8
    with pytest.raises(ValueError, match="batch size of 4"):
        MODULE.validate_generation_config(config)

    config = _config()
    config["data"]["gamus_test_root"] = "forbidden"
    with pytest.raises(ValueError, match="test data configuration is forbidden"):
        MODULE.validate_generation_config(config)


@pytest.mark.parametrize(
    ("source", "sample_id", "region", "landscape", "expected"),
    (
        ("gamus", "NYC_12", "GAMUS_NYC", "mixed", "gamus:nyc"),
        (
            "legacy",
            "Europe_Denmark_Aarhus__grid_1",
            "Europe_Denmark_Aarhus",
            "urban",
            "highbuild:europe_denmark_aarhus",
        ),
        (
            "legacy",
            "oc_2023_771_6893",
            "open_canopy_r154_1378",
            "forest",
            "opencanopy:open_canopy_r154_1378",
        ),
    ),
)
def test_controller_group_ids_use_city_or_existing_5km_region(
    source: str,
    sample_id: str,
    region: str,
    landscape: str,
    expected: str,
) -> None:
    assert MODULE.controller_group_id(
        source=source,
        sample_id=sample_id,
        region=region,
        landscape=landscape,
    ) == expected


class _MetadataDataset(Dataset):
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SimpleNamespace:
        return self.records[index]


def _plan(
    source: str,
    split: str,
    partition: str,
    records: list[SimpleNamespace],
):
    return MODULE.stage3_generation.RouterDatasetPlan(
        source,
        split,
        partition,
        _MetadataDataset(records),
        tuple(record.sample_id for record in records),
    )


def _gamus_records(city: str, count: int, *, ids: list[str] | None = None):
    sample_ids = ids or [f"{city}_synthetic_{index:05d}" for index in range(count)]
    return [
        SimpleNamespace(sample_id=sample_id, region=f"GAMUS_{city}")
        for sample_id in sample_ids
    ]


def _legacy_records(prefix: str, count: int):
    return [
        SimpleNamespace(
            sample_id=f"{prefix}_sample_{index:05d}",
            region=f"{prefix}_city_{index:05d}",
            landscape="urban",
        )
        for index in range(count)
    ]


def test_selection_has_exact_leakage_safe_counts_and_separate_exclusion_ledgers() -> None:
    config = _config()
    no_reference_ids = [
        key.rsplit("/", maxsplit=1)[1]
        for key in config["data"]["excluded_no_reference_record_keys"]
    ]
    nyc_ids = no_reference_ids + [
        f"NYC_synthetic_{index:05d}" for index in range(1158)
    ]
    plans = (
        _plan(
            "gamus",
            "train",
            "train",
            _gamus_records("DC", 1439)
            + _gamus_records("NYC", 1167, ids=nyc_ids)
            + _gamus_records("PHL", 2398),
        ),
        _plan(
            "gamus",
            "val",
            "calibration",
            _gamus_records("DC", 359) + _gamus_records("PHL", 500),
        ),
        _plan(
            "legacy", "train", "train", _legacy_records("train", 1050)
        ),
        _plan(
            "legacy",
            "val",
            "calibration",
            _legacy_records("calibration", 280),
        ),
    )
    selected_plans, selection = MODULE.create_stage4_dataset_plans(config, plans)
    selected_keys = [key for plan in selected_plans for key in plan.record_keys]

    assert len(selected_keys) == 3347
    assert selection["selected_partition_counts"] == {
        "train": 2208,
        "calibration": 1139,
    }
    assert selection["excluded_no_reference_count"] == 9
    assert selection["excluded_group_policy_count"] == 3837
    assert all(
        key.startswith(("gamus/train/DC_", "gamus/train/PHL_"))
        for key in selection["excluded_group_policy_record_keys"]
    )
    assert not (
        set(selection["group_keys_by_partition"]["train"])
        & set(selection["group_keys_by_partition"]["calibration"])
    )
    assert "gamus/gamus:nyc" in selection["group_keys_by_partition"]["train"]
    assert "gamus/gamus:nyc" not in selection["group_keys_by_partition"][
        "calibration"
    ]


def test_replay_keeps_original_stage3_scored_batch_boundaries() -> None:
    records = [
        SimpleNamespace(
            sample_id=f"NYC_{index}", region="GAMUS_NYC", landscape="mixed"
        )
        for index in range(8)
    ]
    plan = MODULE.Stage4DatasetPlan(
        source="gamus",
        split="train",
        partition="train",
        dataset=_MetadataDataset(records),
        stage3_scored_indices=tuple(range(8)),
        # These are analogous to NYC_33930 at the end of one original batch
        # and NYC_33931 at the beginning of the next one.
        original_indices=(3, 4),
        original_sample_ids=("NYC_3", "NYC_4"),
        group_ids=("gamus:nyc", "gamus:nyc"),
    )
    remaining = set(plan.record_keys)

    replay = MODULE.stage3_replay_batches(plan, remaining, batch_size=4)

    assert replay == [
        ((0, 1, 2, 3), ("gamus/train/NYC_3",)),
        ((4, 5, 6, 7), ("gamus/train/NYC_4",)),
    ]
    assert [index for indices, _ in replay for index in indices] != [3, 4]


def test_replay_chunks_after_removing_only_stage3_no_reference_rows() -> None:
    plan = MODULE.Stage4DatasetPlan(
        source="gamus",
        split="train",
        partition="train",
        dataset=_MetadataDataset([]),
        # Original index 1 represents a no-reference row already absent from
        # Stage-3 scoring. It must be removed before the groups of four form.
        stage3_scored_indices=(0, 2, 3, 4, 5, 6),
        original_indices=(4, 5),
        original_sample_ids=("NYC_4", "NYC_5"),
        group_ids=("gamus:nyc", "gamus:nyc"),
    )
    replay = MODULE.stage3_replay_batches(
        plan, set(plan.record_keys), batch_size=4
    )
    assert replay == [
        ((0, 2, 3, 4), ("gamus/train/NYC_4",)),
        ((5, 6), ("gamus/train/NYC_5",)),
    ]


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
        "image": torch.ones(1, 3, 2, 2),
        "relative_prior": torch.zeros(1, 1, 2, 2),
        "height": torch.zeros(1, 1, 2, 2),
        "regression_mask": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        "domain_target": torch.zeros(1, 2, 2, dtype=torch.long),
        "domain_valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "sample_id": ["city_a"],
        "region": ["Europe_Test_City"],
        "landscape": ["urban"],
    }


def test_batch_builder_inherits_descriptor_and_saves_exact_endpoint_evidence() -> None:
    inherited = SceneRouterRecord(
        sample_id="legacy/train/city_a",
        descriptor=torch.tensor([9.0, 8.0]),
        fallback_utility=0.05,
        candidate_utility=0.0,
        source="legacy",
        landscape="urban",
        partition="train",
    )
    records = MODULE.build_records_from_batch(
        batch=_batch(),
        source="legacy",
        split="train",
        partition="train",
        shared_model=_Shared(),
        protected_endpoint=_Endpoint(0.0),
        candidate_endpoint=_Endpoint(-1.0),
        stage3_records={inherited.sample_id: inherited},
        utility_config=SceneUtilityConfig(),
        device=torch.device("cpu"),
        precision="bf16",
    )
    record = records[0]
    assert torch.equal(record.descriptor, inherited.descriptor)
    assert record.group_id == "highbuild:europe_test_city"
    assert record.protected.height_sse == pytest.approx(4.0)
    assert record.protected.height_rmse_m == pytest.approx(1.0)
    assert record.protected.valid_height_pixels == 4
    assert record.protected.semantic_confusion_3x3 == (
        (4, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    )
    assert record.protected.valid_semantic_pixels == 4
    assert record.protected.observed_semantic_classes == 1
    assert record.protected.semantic_balanced_error == pytest.approx(0.0)
    assert record.candidate.height_sse == pytest.approx(0.0)


def test_visible_runner_is_deterministic_and_does_not_promote() -> None:
    content = (PROJECT_ROOT / "scripts" / "run_stage4_dual_router_records.ps1").read_text(
        encoding="utf-8"
    )
    assert '$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"' in content
    assert "build_stage4_dual_router_records.py" in content
    assert "showcase_checkpoint.txt" not in content
    assert "select_showcase_checkpoint" not in content

    training_config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "stage4_dual_router_training.yaml").read_text(
            encoding="utf-8"
        )
    )
    training_runner = (
        PROJECT_ROOT / "scripts" / "run_stage4_dual_router_training.ps1"
    ).read_text(encoding="utf-8")
    expected_store = "outputs/stage4_dual_router_records_v3"
    assert training_config["stage4_dual_router_training"]["record_store"] == (
        expected_store
    )
    assert "outputs\\stage4_dual_router_records_v3" in training_runner
    assert "stage4_dual_router_records_v2" not in training_runner
