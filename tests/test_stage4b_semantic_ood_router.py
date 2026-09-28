from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from msr.training.scene_router import SceneRouterRecord
from msr.training.stage4_dual_router import (
    EndpointComponentUtilities,
    Stage4DualRouterRecord,
)
from msr.training.stage4b_semantic_ood_router import (
    DevelopmentGateConfig,
    FitNormalizedLinearClassifier,
    LinearTrainingConfig,
    Stage4bDomainExample,
    Stage4bSemanticOODRouter,
    build_stage4b_training_examples,
    fit_ood_guard,
    fit_shrinkage_mahalanobis,
    make_group_held_out_folds,
    select_development_operating_point,
    train_linear_member,
)


def _stage3(
    sample_id: str,
    *,
    source: str,
    descriptor: tuple[float, ...],
    partition: str = "train",
    landscape: str = "mixed",
) -> SceneRouterRecord:
    return SceneRouterRecord(
        sample_id=sample_id,
        descriptor=torch.tensor(descriptor),
        fallback_utility=0.5,
        candidate_utility=0.4,
        source=source,
        landscape=landscape,
        partition=partition,
    )


def _endpoint(confusion: tuple[tuple[int, int, int], ...]) -> EndpointComponentUtilities:
    recalls = [confusion[index][index] / sum(confusion[index]) for index in range(3)]
    error = 1.0 - sum(recalls) / 3
    return EndpointComponentUtilities(
        aggregate_utility=error,
        height_sse=None,
        height_rmse_m=None,
        semantic_confusion_3x3=confusion,
        semantic_balanced_error=error,
        valid_height_pixels=0,
        valid_semantic_pixels=sum(sum(row) for row in confusion),
        observed_semantic_classes=3,
    )


PROTECTED = _endpoint(((70, 30, 0), (30, 70, 0), (30, 0, 70)))
CANDIDATE = _endpoint(((90, 10, 0), (10, 90, 0), (10, 0, 90)))
SAME = _endpoint(((80, 20, 0), (20, 80, 0), (20, 0, 80)))


def _rich(index: int, *, source: str) -> Stage4DualRouterRecord:
    gamus = source == "gamus"
    return Stage4DualRouterRecord(
        sample_id=f"{source}/val/{index}",
        descriptor=torch.tensor([float(gamus), float(index % 7)]),
        protected=PROTECTED if gamus else SAME,
        candidate=CANDIDATE if gamus else SAME,
        source=source,
        group_id="gamus:dc" if gamus else "highbuild:heldout",
        landscape="mixed" if gamus else "urban",
        partition="calibration",
    )


def test_build_examples_uses_only_train_and_requires_authenticated_legacy_groups() -> None:
    records = [
        _stage3("gamus/train/DC_1", source="gamus", descriptor=(1, 0)),
        _stage3("gamus/train/NYC_1", source="gamus", descriptor=(1, 1)),
        _stage3("gamus/train/PHL_1", source="gamus", descriptor=(1, 2)),
        _stage3(
            "legacy/train/city_1",
            source="legacy",
            descriptor=(0, 1),
            landscape="urban",
        ),
        _stage3(
            "legacy/val/city_2",
            source="legacy",
            descriptor=(0, 2),
            partition="calibration",
            landscape="urban",
        ),
    ]
    examples, audit = build_stage4b_training_examples(
        records,
        legacy_group_ids={"legacy/train/city_1": "highbuild:city"},
    )
    assert len(examples) == 4
    assert audit["stage3_partitions_consumed"] == ["train"]
    assert audit["official_test_used"] is False
    assert {item.group_id for item in examples if item.source == "gamus"} == {
        "gamus:dc",
        "gamus:nyc",
        "gamus:phl",
    }
    with pytest.raises(ValueError, match="missing"):
        build_stage4b_training_examples(records, legacy_group_ids={})


def test_group_folds_hold_every_city_and_legacy_region_once() -> None:
    examples: list[Stage4bDomainExample] = []
    for city in ("dc", "nyc", "phl"):
        for index in range(4):
            examples.append(
                Stage4bDomainExample(
                    sample_id=f"gamus/train/{city.upper()}_{index}",
                    descriptor=torch.tensor([1.0, float(index)]),
                    candidate_domain=True,
                    source="gamus",
                    landscape="mixed",
                    group_id=f"gamus:{city}",
                )
            )
    for group, landscape in (
        ("highbuild:a", "urban"),
        ("highbuild:b", "urban"),
        ("highbuild:c", "urban"),
        ("opencanopy:a", "forest"),
        ("opencanopy:b", "forest"),
        ("opencanopy:c", "forest"),
    ):
        for index in range(2):
            examples.append(
                Stage4bDomainExample(
                    sample_id=f"legacy/train/{group}_{index}",
                    descriptor=torch.tensor([0.0, float(index)]),
                    candidate_domain=False,
                    source="legacy",
                    landscape=landscape,
                    group_id=group,
                )
            )
    folds, audit = make_group_held_out_folds(examples, folds=3, seed=9)
    assert len(folds) == 3
    assert audit["zero_group_overlap_every_fold"] is True
    validation_groups = [group for fold in folds for group in fold.validation_groups]
    assert len(validation_groups) == len(set(validation_groups)) == 9
    assert all(not (set(fold.train_groups) & set(fold.validation_groups)) for fold in folds)


def test_member_normalization_is_fit_only_and_training_is_finite() -> None:
    descriptors = torch.tensor([[0.0, 1.0], [0.1, 2.0], [4.0, 5.0], [4.1, 6.0]])
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
    member, history = train_linear_member(
        descriptors,
        targets,
        seed=3,
        config=LinearTrainingConfig(epochs=3, learning_rate=0.02, weight_decay=0.01, batch_size=2),
    )
    assert torch.allclose(member.feature_mean, descriptors.mean(0))
    assert len(history) == 3
    assert np.isfinite(history[-1]["train_bce"])


def test_ood_fit_is_finite_and_uses_worst_fold_thresholds() -> None:
    examples: list[Stage4bDomainExample] = []
    generator = torch.Generator().manual_seed(4)
    for city, offset in (("dc", 2.0), ("nyc", 2.5), ("phl", 3.0)):
        for index in range(12):
            examples.append(
                Stage4bDomainExample(
                    sample_id=f"gamus/train/{city.upper()}_{index}",
                    descriptor=torch.randn(4, generator=generator) * 0.1 + offset,
                    candidate_domain=True,
                    source="gamus",
                    landscape="mixed",
                    group_id=f"gamus:{city}",
                )
            )
    for group_index in range(6):
        for index in range(6):
            examples.append(
                Stage4bDomainExample(
                    sample_id=f"legacy/train/{group_index}_{index}",
                    descriptor=torch.randn(4, generator=generator) * 0.1 - 2.0,
                    candidate_domain=False,
                    source="legacy",
                    landscape="urban" if group_index < 3 else "forest",
                    group_id=(
                        f"highbuild:{group_index}"
                        if group_index < 3
                        else f"opencanopy:{group_index}"
                    ),
                )
            )
    folds, _ = make_group_held_out_folds(examples, folds=3, seed=1)
    x = torch.stack([item.descriptor for item in examples])
    y = torch.tensor([item.candidate_domain for item in examples])
    fit, cutoff, margin, audit = fit_ood_guard(
        x, y, folds, shrinkage=0.2, quantile=0.95
    )
    assert cutoff > 0 and np.isfinite(margin)
    assert torch.all(torch.isfinite(fit.precision))
    assert cutoff == max(item["own_distance_quantile"] for item in audit["folds"])


def test_router_can_never_select_height() -> None:
    x = torch.tensor([[-2.0, -2.0], [-1.5, -2.0], [2.0, 2.0], [1.5, 2.0]])
    y = torch.tensor([False, False, True, True])
    ood = fit_shrinkage_mahalanobis(x, y, shrinkage=0.5)
    members = [FitNormalizedLinearClassifier(torch.zeros(2), torch.ones(2)) for _ in range(3)]
    for member in members:
        member.linear.weight.data.zero_()
        member.linear.bias.data.fill_(10.0)
    router = Stage4bSemanticOODRouter(
        members,
        ood,
        ensemble_std_multiplier=2.0,
        ood_max_distance=100.0,
        minimum_distance_margin=-100.0,
    )
    router.set_semantic_operating_point(threshold=0.0, enabled=True)
    result = router.forward_descriptor(x)
    assert router.height_route_enabled is False
    assert not bool(torch.any(result["height_candidate_selected"]))


def test_development_gate_passes_only_domain_safe_semantic_routing() -> None:
    records = [_rich(index, source="gamus") for index in range(500)] + [
        _rich(index, source="legacy") for index in range(80)
    ]
    lower = np.concatenate((np.full(450, 0.9), np.full(50, 0.1), np.full(80, 0.1)))
    members = np.repeat(lower[:, None], 5, axis=1)
    result = select_development_operating_point(
        records,
        lower_probabilities=lower,
        member_probabilities=members,
        ood_accepted=np.ones(len(records), dtype=bool),
        margin_accepted=np.ones(len(records), dtype=bool),
        std_multiplier=2.0,
        config=DevelopmentGateConfig(),
    )
    assert result.eligible is True
    assert result.metrics["legacy_selected"] == 0
    assert result.metrics["gamus_coverage"] == pytest.approx(0.9)
    assert result.metrics["routed_gain_vs_protected"] >= 0.05
    assert result.metrics["gamus_exact_gain"] >= 0.08
    assert result.metrics["fresh_holdout_required"] is True


def test_development_gate_fails_closed_when_legacy_scores_higher() -> None:
    records = [_rich(index, source="gamus") for index in range(500)] + [
        _rich(index, source="legacy") for index in range(80)
    ]
    lower = np.concatenate((np.full(500, 0.8), np.full(80, 0.9)))
    result = select_development_operating_point(
        records,
        lower_probabilities=lower,
        member_probabilities=np.repeat(lower[:, None], 5, axis=1),
        ood_accepted=np.ones(len(records), dtype=bool),
        margin_accepted=np.ones(len(records), dtype=bool),
        std_multiplier=2.0,
        config=DevelopmentGateConfig(),
    )
    assert result.eligible is False
    assert result.metrics["gamus_selected"] == 0
    assert result.metrics["legacy_selected"] == 0


def test_stage4b_trainer_contract_never_routes_height_or_uses_test() -> None:
    source = (Path(__file__).parents[1] / "scripts" / "train_stage4b_semantic_ood_router.py").read_text()
    assert '"height_route_enabled": False' in source
    assert '"test_splits_used": False' in source
    assert "file_sha256(pointer)" in source
    assert "pointer.read_text" not in source
