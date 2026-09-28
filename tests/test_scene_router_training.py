import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from msr.models.routed_surface import ConservativeSceneRouter
from msr.training.scene_router import (
    BalancedSourceLandscapeSampler,
    SceneRouterDataset,
    SceneRouterObjective,
    SceneRouterRecord,
    SceneRouterTrainingConfig,
    SceneUtilityConfig,
    build_scene_router_record,
    compute_scene_endpoint_utility,
    compute_scene_router_metrics,
    derive_scene_router_target,
    expected_calibration_error,
    read_scene_router_records,
    select_conservative_threshold,
    train_scene_router,
    write_scene_router_records,
)


def test_endpoint_utility_and_route_target_come_from_reference_labels() -> None:
    height_target = torch.zeros(2, 2)
    domain_target = torch.tensor([[0, 0], [1, 1]])
    fallback = {
        "height": torch.ones(1, 2, 2),
        "domain_logits": torch.tensor(
            [
                [[5.0, 5.0], [5.0, 5.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
    }
    candidate = {
        "height": torch.zeros(1, 2, 2),
        "domain_logits": torch.tensor(
            [
                [[5.0, 5.0], [0.0, 0.0]],
                [[0.0, 0.0], [5.0, 5.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ),
    }
    config = SceneUtilityConfig(height_scale_m=2.0)
    fallback_utility = compute_scene_endpoint_utility(
        fallback,
        height_target=height_target,
        valid_mask=torch.ones_like(height_target, dtype=torch.bool),
        domain_target=domain_target,
        domain_valid_mask=torch.ones_like(domain_target, dtype=torch.bool),
        config=config,
    )
    candidate_utility = compute_scene_endpoint_utility(
        candidate,
        height_target=height_target,
        valid_mask=torch.ones_like(height_target, dtype=torch.bool),
        domain_target=domain_target,
        domain_valid_mask=torch.ones_like(domain_target, dtype=torch.bool),
        config=config,
    )
    decision = derive_scene_router_target(
        fallback_utility, candidate_utility, minimum_candidate_gain=0.05
    )

    assert fallback_utility.height_rmse_m == pytest.approx(1.0)
    assert fallback_utility.semantic_balanced_error == pytest.approx(0.5)
    assert fallback_utility.utility == pytest.approx(0.5)
    assert candidate_utility.utility == pytest.approx(0.0)
    assert decision.candidate_target
    assert decision.candidate_gain == pytest.approx(0.5)

    record, audit = build_scene_router_record(
        sample_id="held-out-scene",
        descriptor=torch.zeros(1, 6),
        fallback_endpoint=fallback,
        candidate_endpoint=candidate,
        source="gamus",
        landscape="mixed",
        partition="train",
        height_target=height_target,
        valid_mask=torch.ones_like(height_target, dtype=torch.bool),
        domain_target=domain_target,
        domain_valid_mask=torch.ones_like(domain_target, dtype=torch.bool),
        utility_config=config,
    )
    assert record.decision(minimum_candidate_gain=0.05).candidate_target
    assert audit["fallback"]["height_rmse_m"] == pytest.approx(1.0)
    assert audit["candidate"]["semantic_balanced_error"] == pytest.approx(0.0)


def test_ties_and_submargin_gains_stay_on_protected_fallback() -> None:
    assert not derive_scene_router_target(0.5, 0.5).candidate_target
    assert not derive_scene_router_target(
        0.5, 0.49, minimum_candidate_gain=0.02
    ).candidate_target
    assert derive_scene_router_target(
        0.5, 0.47, minimum_candidate_gain=0.02
    ).candidate_target


@pytest.mark.parametrize("corrupt", ["height", "domain_logits"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_endpoint_utility_rejects_nonfinite_output_on_reference_support(
    corrupt: str,
    bad_value: float,
) -> None:
    endpoint = {
        "height": torch.zeros(1, 2, 2),
        "domain_logits": torch.zeros(3, 2, 2),
    }
    endpoint[corrupt].reshape(-1)[0] = bad_value
    with pytest.raises(ValueError, match="non-finite values on valid reference"):
        compute_scene_endpoint_utility(
            endpoint,
            height_target=torch.zeros(2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
            domain_target=torch.zeros(2, 2, dtype=torch.long),
            domain_valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_record_builder_rejects_nonfinite_candidate_instead_of_shrinking_support(
    bad_value: float,
) -> None:
    fallback = {
        "height": torch.zeros(1, 2, 2),
        "domain_logits": torch.zeros(3, 2, 2),
    }
    candidate = {name: value.clone() for name, value in fallback.items()}
    candidate["height"][0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="non-finite values on valid reference"):
        build_scene_router_record(
            sample_id="candidate-corrupt",
            descriptor=torch.zeros(6),
            fallback_endpoint=fallback,
            candidate_endpoint=candidate,
            source="gamus",
            landscape="mixed",
            partition="train",
            height_target=torch.zeros(2, 2),
            valid_mask=torch.ones(2, 2, dtype=torch.bool),
            domain_target=torch.zeros(2, 2, dtype=torch.long),
            domain_valid_mask=torch.ones(2, 2, dtype=torch.bool),
        )


def test_record_io_ignores_supplied_target_and_dataset_recomputes_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "records.jsonl"
    payloads = [
        {
            "sample_id": "gamus-but-fallback-wins",
            "descriptor": [1.0, 2.0],
            "fallback_utility": 0.1,
            "candidate_utility": 0.9,
            "source": "gamus",
            "landscape": "forest",
            "candidate_target": True,
        },
        {
            "sample_id": "legacy-but-candidate-wins",
            "descriptor": [3.0, 4.0],
            "fallback_utility": 0.9,
            "candidate_utility": 0.1,
            "source": "legacy",
            "landscape": "urban",
            "candidate_target": False,
        },
    ]
    path.write_text("\n".join(json.dumps(value) for value in payloads), encoding="utf-8")
    records = read_scene_router_records(path)
    dataset = SceneRouterDataset(records, minimum_candidate_gain=0.05)

    assert dataset[0]["candidate_target"].item() == 0.0
    assert dataset[1]["candidate_target"].item() == 1.0
    round_trip_path = tmp_path / "round-trip.jsonl"
    write_scene_router_records(records, round_trip_path)
    assert [record.sample_id for record in read_scene_router_records(round_trip_path)] == [
        "gamus-but-fallback-wins",
        "legacy-but-candidate-wins",
    ]


def _record(
    sample_id: str,
    source: str,
    landscape: str,
    *,
    candidate_wins: bool,
    descriptor: tuple[float, float] = (0.0, 0.0),
    partition: str = "train",
) -> SceneRouterRecord:
    return SceneRouterRecord(
        sample_id=sample_id,
        descriptor=torch.tensor(descriptor),
        fallback_utility=1.0,
        candidate_utility=0.5 if candidate_wins else 1.5,
        source=source,
        landscape=landscape,
        partition=partition,
    )


def test_sampler_exactly_balances_observed_source_landscape_strata() -> None:
    records = [
        _record("a", "gamus", "forest", candidate_wins=True),
        _record("b", "gamus", "forest", candidate_wins=False),
        _record("c", "gamus", "urban", candidate_wins=True),
        _record("d", "legacy", "forest", candidate_wins=False),
        _record("e", "legacy", "urban", candidate_wins=False),
    ]
    sampler = BalancedSourceLandscapeSampler(records, num_samples=12, seed=4)
    first = list(sampler)
    counts: dict[tuple[str, str], int] = {}
    for index in first:
        key = (records[index].source, records[index].landscape)
        counts[key] = counts.get(key, 0) + 1

    assert set(counts.values()) == {3}
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_asymmetric_bce_penalizes_false_candidate_more() -> None:
    logits = torch.tensor([[2.0], [2.0]])
    targets = torch.tensor([[0.0], [1.0]])
    symmetric = SceneRouterObjective()(logits, targets)["loss"]
    conservative = SceneRouterObjective(fallback_weight=4.0)(logits, targets)["loss"]
    assert conservative > symmetric


def test_router_metrics_include_calibration_and_endpoint_utility() -> None:
    probabilities = np.array([0.9, 0.8, 0.2, 0.1])
    targets = np.array([1, 1, 0, 0])
    fallback = np.ones(4)
    candidate = np.array([0.5, 0.6, 1.5, 2.0])
    metrics = compute_scene_router_metrics(
        probabilities,
        targets,
        threshold=0.75,
        fallback_utilities=fallback,
        candidate_utilities=candidate,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.025)
    assert metrics["expected_calibration_error"] == pytest.approx(0.15)
    assert metrics["routed_mean_utility"] == pytest.approx(0.775)
    assert metrics["routed_gain_vs_fallback"] == pytest.approx(0.225)
    assert expected_calibration_error(probabilities, targets) == pytest.approx(0.15)


def test_conservative_threshold_rejects_false_positive_even_if_net_gain_is_high() -> None:
    result = select_conservative_threshold(
        probabilities=np.array([0.95, 0.8, 0.7]),
        targets=np.array([1, 0, 1]),
        fallback_utilities=np.ones(3),
        candidate_utilities=np.array([0.5, 2.0, 0.6]),
        minimum_threshold=0.5,
        minimum_precision=1.0,
    )
    assert result.eligible
    assert result.threshold == pytest.approx(0.95)
    assert result.metrics["candidate_selected"] == 1
    assert result.metrics["precision"] == pytest.approx(1.0)

    rejected = select_conservative_threshold(
        probabilities=np.array([0.9, 0.8]),
        targets=np.array([0, 0]),
        fallback_utilities=np.ones(2),
        candidate_utilities=np.array([2.0, 3.0]),
        minimum_precision=1.0,
    )
    assert not rejected.eligible
    assert "fallback" in rejected.reason


def test_training_changes_only_router_and_leaves_frozen_model_untouched() -> None:
    train_records = [
        _record(
            f"train-{index}",
            "gamus" if index % 2 else "legacy",
            "forest" if (index // 2) % 2 else "urban",
            candidate_wins=index >= 4,
            descriptor=((2.0 if index >= 4 else -2.0), float(index % 2)),
        )
        for index in range(8)
    ]
    validation_records = [
        _record(
            f"validation-{index}",
            "gamus" if index % 2 else "legacy",
            "forest" if index % 2 else "urban",
            candidate_wins=index >= 2,
            descriptor=((2.0 if index >= 2 else -2.0), float(index % 2)),
            partition="calibration",
        )
        for index in range(4)
    ]
    train_dataset = SceneRouterDataset(train_records)
    validation_dataset = SceneRouterDataset(validation_records)
    sampler = BalancedSourceLandscapeSampler(train_records, num_samples=8, seed=9)
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)
    validation_loader = DataLoader(validation_dataset, batch_size=4)
    router = ConservativeSceneRouter(
        1,
        hidden_features=4,
        decision_threshold=0.9,
        initial_candidate_probability=0.1,
    )
    initial_router = {
        name: value.clone() for name, value in router.state_dict().items()
    }
    frozen_model = nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 1))
    initial_frozen = {
        name: value.clone() for name, value in frozen_model.state_dict().items()
    }
    wrapped_model = nn.Module()
    wrapped_model.shared_height_model = frozen_model
    wrapped_model.router = router
    result = train_scene_router(
        wrapped_model,
        train_loader,
        validation_loader,
        config=SceneRouterTrainingConfig(
            epochs=5,
            learning_rate=0.05,
            weight_decay=0.0,
            minimum_threshold=0.5,
            minimum_precision=0.0,
        ),
    )

    assert len(result.history) == 5
    assert any(
        not torch.equal(value, initial_router[name])
        for name, value in router.state_dict().items()
        if name != "decision_threshold"
    )
    for name, value in frozen_model.state_dict().items():
        torch.testing.assert_close(value, initial_frozen[name], rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in frozen_model.parameters())
    assert not router.training
