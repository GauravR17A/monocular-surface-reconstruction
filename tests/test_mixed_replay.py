import importlib.util
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from msr.data.mixed_replay import (
    SourceRatioSampler,
    SourceTaggedDataset,
    assert_matching_sample_contract,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_multidomain.py"
SPEC = importlib.util.spec_from_file_location("train_multidomain_mixed_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class _ToySurfaceDataset(Dataset):
    def __init__(self, size: int, *, dtype: torch.dtype = torch.float32) -> None:
        self.size = size
        self.dtype = dtype

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return {
            "image": torch.full((3, 2, 2), float(index), dtype=self.dtype),
            "height": torch.zeros((1, 2, 2)),
            "sample_id": f"sample-{index}",
            "landscape": "mixed",
        }


def test_source_ratio_sampler_enforces_stage2_quotas_and_balance() -> None:
    legacy_landscapes = ["urban"] * 450 + ["forest"] * 600
    sampler = SourceRatioSampler(
        5004,
        len(legacy_landscapes),
        num_samples=6672,
        batch_size=4,
        gamus_fraction=0.75,
        seed=20260912,
        legacy_landscapes=legacy_landscapes,
        balance_legacy_landscapes=True,
    )

    indices = list(sampler)

    assert len(indices) == 6672
    assert sampler.source_counts == {"gamus": 5004, "legacy": 1668}
    for start in range(0, len(indices), 4):
        batch = indices[start : start + 4]
        assert sum(index < 5004 for index in batch) == 3
        assert sum(index >= 5004 for index in batch) == 1
    legacy_draws = [index - 5004 for index in indices if index >= 5004]
    counts = Counter(legacy_landscapes[index] for index in legacy_draws)
    assert counts == {"urban": 834, "forest": 834}


def test_source_ratio_sampler_is_epoch_deterministic_and_cycles_without_replacement() -> None:
    settings = {
        "gamus_size": 5,
        "legacy_size": 4,
        "num_samples": 16,
        "batch_size": 4,
        "gamus_fraction": 0.75,
        "seed": 91,
        "legacy_landscapes": ["urban", "urban", "forest", "forest"],
        "balance_legacy_landscapes": True,
    }
    first = SourceRatioSampler(**settings)
    second = SourceRatioSampler(**settings)
    epoch_zero = list(first)
    assert epoch_zero == list(second)
    gamus_draws = [index for index in epoch_zero if index < 5]
    assert len(set(gamus_draws[:5])) == 5

    first.set_epoch(3)
    second.set_epoch(3)
    assert list(first) == list(second)
    assert list(first) != epoch_zero


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_samples": 15}, "divisible"),
        ({"gamus_fraction": 0.7}, "must be an integer"),
    ],
)
def test_source_ratio_sampler_rejects_inexact_batch_plans(kwargs, message) -> None:
    settings = {
        "gamus_size": 5,
        "legacy_size": 4,
        "num_samples": 16,
        "batch_size": 4,
        "gamus_fraction": 0.75,
    }
    settings.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SourceRatioSampler(**settings)


def test_concat_dataset_has_one_collatable_source_tagged_contract() -> None:
    gamus = _ToySurfaceDataset(3)
    legacy = _ToySurfaceDataset(2)
    assert assert_matching_sample_contract(gamus, legacy) == frozenset(
        {"image", "height", "sample_id", "landscape"}
    )
    combined = ConcatDataset(
        [SourceTaggedDataset(gamus, "gamus"), SourceTaggedDataset(legacy, "legacy")]
    )
    sampler = SourceRatioSampler(
        3,
        2,
        num_samples=4,
        batch_size=4,
        gamus_fraction=0.75,
        seed=1,
    )
    batch = next(iter(DataLoader(combined, batch_size=4, sampler=sampler)))

    assert set(batch) == {"image", "height", "sample_id", "landscape", "source"}
    assert Counter(batch["source"]) == {"gamus": 3, "legacy": 1}
    assert batch["image"].shape == (4, 3, 2, 2)

    with pytest.raises(ValueError, match="contracts differ"):
        assert_matching_sample_contract(gamus, _ToySurfaceDataset(2, dtype=torch.float64))


def _suite_metrics(*, f1: float, rmse: float) -> dict[str, object]:
    return {
        "rmse_m": rmse,
        "semantic_identification": {
            "macro_f1": f1,
            "accuracy": f1 + 0.01,
            "per_class": {
                "ground": {"f1": f1},
                "building": {"f1": f1},
                "vegetation": {"f1": f1},
            },
        },
        "domains": {
            "ground": {"rmse_m": rmse},
            "building": {"rmse_m": rmse},
            "vegetation": {"rmse_m": rmse},
        },
    }


def test_dual_validation_guards_require_every_suite_to_pass() -> None:
    initial = {
        "gamus": _suite_metrics(f1=0.73, rmse=5.0),
        "legacy": _suite_metrics(f1=0.63, rmse=6.0),
    }
    current = {
        "gamus": _suite_metrics(f1=0.74, rmse=5.015),
        "legacy": _suite_metrics(f1=0.621, rmse=6.1),
    }
    config = {
        "validation_guards": {
            "gamus": {
                "semantic_identification.macro_f1": {"min": 0.740},
                "rmse_m": {"max_regression": 0.02, "baseline": "initial"},
                "semantic_identification.accuracy": {
                    "max_drop": 0.01,
                    "baseline": "initial",
                },
            },
            "legacy": {
                "semantic_identification.macro_f1": {"min": 0.622},
                "rmse_m": {"max": 6.12},
            },
        }
    }

    details, eligible = MODULE.validation_guard_eligibility(
        current, initial, config
    )

    assert eligible is False
    assert details["suites"]["gamus"]["passes"] is True
    assert details["suites"]["legacy"]["passes"] is False
    rmse_guard = details["suites"]["gamus"]["guards"]["rmse_m"]
    assert rmse_guard["reference"] == pytest.approx(5.0)
    assert rmse_guard["delta"] == pytest.approx(0.015)


def test_validation_guards_reject_ambiguous_and_cross_suite_baselines() -> None:
    metrics = {"gamus": _suite_metrics(f1=0.74, rmse=5.0)}
    with pytest.raises(ValueError, match="exactly one"):
        MODULE.validation_guard_eligibility(
            metrics,
            metrics,
            {
                "validation_guards": {
                    "gamus": {"rmse_m": {"min": 4.0, "max": 6.0}}
                }
            },
        )
    with pytest.raises(KeyError, match="initial baseline"):
        MODULE.validation_guard_eligibility(
            metrics,
            {},
            {
                "validation_guards": {
                    "gamus": {
                        "rmse_m": {
                            "max_regression": 0.02,
                            "baseline": "initial",
                        }
                    }
                }
            },
        )


def test_mixed_replay_config_locks_training_to_domain_head() -> None:
    valid = {
        "epochs": 5,
        "freeze_base_epochs": 5,
        "base_learning_rate_multiplier": 0.0,
        "trainable_adapter_prefixes": ["domain_head."],
    }
    MODULE.validate_mixed_replay_training_config(
        dataset_kind="mixed_replay",
        initial_checkpoint="stage1.pt",
        training_config=valid,
    )
    with pytest.raises(ValueError, match="domain_head"):
        MODULE.validate_mixed_replay_training_config(
            dataset_kind="mixed_replay",
            initial_checkpoint="stage1.pt",
            training_config={**valid, "trainable_adapter_prefixes": ["adapter."]},
        )
    # Existing dataset modes remain outside the Stage-2-only validation path.
    MODULE.validate_mixed_replay_training_config(
        dataset_kind="manifest",
        initial_checkpoint=None,
        training_config={},
    )


def test_height_mixed_replay_uses_sealed_final_heads() -> None:
    valid = {
        "epochs": 4,
        "freeze_base_epochs": 0,
        "mixed_replay_objective": "height_supervision",
        "parameter_groups": [
            {
                "name": "metric_height",
                "prefixes": ["base_model.height_head."],
                "learning_rate": 2.0e-6,
            },
            {
                "name": "canopy_height",
                "prefixes": [
                    "canopy_height_head.",
                    "refinement_strength_head.",
                ],
                "learning_rate": 3.0e-5,
            },
        ],
    }
    MODULE.validate_mixed_replay_training_config(
        dataset_kind="mixed_replay",
        initial_checkpoint="protected.pt",
        training_config=valid,
    )
    assert MODULE.uses_stage2_semantic_repair("mixed_replay", valid) is False
    assert MODULE.uses_stage2_semantic_repair(
        "mixed_replay", {"epochs": 4}
    ) is True

    unsafe = {
        **valid,
        "parameter_groups": [
            {
                "name": "unsafe_shared_encoder",
                "prefixes": [
                    "base_model.height_head.",
                    "canopy_height_head.",
                    "base_model.encoder.stages_3.",
                ],
                "learning_rate": 2.0e-6,
            }
        ],
    }
    with pytest.raises(ValueError, match="unsafe prefixes"):
        MODULE.validate_mixed_replay_training_config(
            dataset_kind="mixed_replay",
            initial_checkpoint="protected.pt",
            training_config=unsafe,
        )

    missing_canopy = {
        **valid,
        "parameter_groups": [
            {
                "name": "metric_height",
                "prefixes": ["base_model.height_head."],
                "learning_rate": 2.0e-6,
            }
        ],
    }
    with pytest.raises(ValueError, match="requires both"):
        MODULE.validate_mixed_replay_training_config(
            dataset_kind="mixed_replay",
            initial_checkpoint="protected.pt",
            training_config=missing_canopy,
        )


def test_height_mixed_replay_rejects_unknown_objective() -> None:
    with pytest.raises(ValueError, match="mixed_replay_objective"):
        MODULE.validate_mixed_replay_training_config(
            dataset_kind="mixed_replay",
            initial_checkpoint="protected.pt",
            training_config={
                "epochs": 1,
                "mixed_replay_objective": "magic",
            },
        )


def test_primary_selection_can_maximize_legacy_macro_f1() -> None:
    metrics_by_suite = {
        "gamus": _suite_metrics(f1=0.745, rmse=5.0),
        "legacy": _suite_metrics(f1=0.631, rmse=6.0),
    }
    config = {
        "primary_selection": {
            "suite": "legacy",
            "metric": "semantic_identification.macro_f1",
            "mode": "max",
        }
    }

    result = MODULE.selection_result(
        metrics_by_suite["gamus"],
        config,
        metrics_by_suite=metrics_by_suite,
    )

    assert result == {
        "suite": "legacy",
        "metric": "semantic_identification.macro_f1",
        "mode": "max",
        "value": pytest.approx(0.631),
        "score": pytest.approx(-0.631),
        "lower_score_is_better": True,
    }


def test_primary_selection_seeds_baseline_and_prevents_first_epoch_loophole() -> None:
    initial_by_suite = {
        "gamus": _suite_metrics(f1=0.75, rmse=5.0),
        "legacy": _suite_metrics(f1=0.6321, rmse=6.0),
    }
    config = {
        "primary_selection": {
            "suite": "legacy",
            "metric": "semantic_identification.macro_f1",
            "mode": "max",
        }
    }
    baseline_score = MODULE.resolve_initial_best_selection(
        initial_by_suite["gamus"],
        "current_validation",
        config,
        initial_metrics_by_suite=initial_by_suite,
    )

    assert baseline_score == pytest.approx(-0.6321)
    # Passing hard guards is necessary but cannot promote a candidate that is
    # worse than the warm-start checkpoint on the configured objective.
    assert (
        MODULE.selection_improved(-0.6300, baseline_score, eligible=True) is False
    )
    assert MODULE.selection_improved(-0.6400, baseline_score, eligible=False) is False
    assert MODULE.selection_improved(-0.6400, baseline_score, eligible=True) is True

    with pytest.raises(ValueError, match="unguarded first-epoch"):
        MODULE.resolve_initial_best_selection(
            initial_by_suite["gamus"],
            "stored_checkpoint",
            config,
            initial_metrics_by_suite={},
        )
