from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from msr.training.stage4_dual_router import (
    EndpointComponentUtilities,
    EndpointGainRouter,
    GainRouterTrainingConfig,
    Stage4DualRouterRecord,
    Stage4GainDataset,
    build_stage4_dual_router_record,
    compute_gain_route_metrics,
    partition_group_held_out_records,
    read_stage4_dual_router_records,
    select_group_guarded_gain_threshold,
    train_gain_router,
    write_stage4_dual_router_records,
)


def _balanced_error(confusion: tuple[tuple[int, int, int], ...]) -> float | None:
    recalls = [
        row[index] / sum(row)
        for index, row in enumerate(confusion)
        if sum(row)
    ]
    return None if not recalls else 1.0 - sum(recalls) / len(recalls)


def _utility(
    *,
    height_rmse: float | None = 1.0,
    confusion: tuple[tuple[int, int, int], ...] = (
        (2, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ),
) -> EndpointComponentUtilities:
    count = sum(sum(row) for row in confusion)
    height_count = 4 if height_rmse is not None else 0
    return EndpointComponentUtilities(
        aggregate_utility=0.1,
        height_sse=(None if height_rmse is None else height_rmse**2 * height_count),
        height_rmse_m=height_rmse,
        semantic_confusion_3x3=confusion,
        semantic_balanced_error=_balanced_error(confusion),
        valid_height_pixels=height_count,
        valid_semantic_pixels=count,
        observed_semantic_classes=sum(int(sum(row) > 0) for row in confusion),
    )


def _record(
    sample_id: str,
    *,
    partition: str,
    group_id: str,
    source: str = "gamus",
    landscape: str = "urban",
    protected_height: float = 1.0,
    candidate_height: float = 0.5,
) -> Stage4DualRouterRecord:
    return Stage4DualRouterRecord(
        sample_id=sample_id,
        descriptor=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        protected=_utility(height_rmse=protected_height),
        candidate=_utility(height_rmse=candidate_height),
        source=source,
        group_id=group_id,
        landscape=landscape,
        partition=partition,
    )


def test_rich_record_builder_retains_exact_endpoint_evidence(tmp_path: Path) -> None:
    target = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    domain = torch.tensor([[0, 1], [2, 0]])
    protected = {
        "height": torch.zeros(1, 1, 2, 2),
        "domain_logits": torch.nn.functional.one_hot(domain, 3)
        .permute(2, 0, 1)
        .float(),
    }
    candidate = {
        "height": target[None, None],
        "domain_logits": protected["domain_logits"].clone(),
    }
    record = build_stage4_dual_router_record(
        sample_id="gamus/train/DC_1",
        descriptor=torch.arange(4),
        protected_endpoint=protected,
        candidate_endpoint=candidate,
        source="gamus",
        group_id="dc",
        landscape="urban",
        partition="train",
        height_target=target,
        domain_target=domain,
    )
    assert record.protected.height_sse == pytest.approx(14.0)
    assert record.candidate.height_sse == pytest.approx(0.0)
    assert sum(sum(row) for row in record.protected.semantic_confusion_3x3) == 4

    path = tmp_path / "rich.jsonl"
    write_stage4_dual_router_records([record], path)
    loaded = read_stage4_dual_router_records(path)
    assert loaded[0].to_json_dict() == record.to_json_dict()

    old = tmp_path / "stage3.jsonl"
    old.write_text(json.dumps({"sample_id": "old"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate Stage-3 records cannot"):
        read_stage4_dual_router_records(old)


def test_source_group_holdout_rejects_city_leakage() -> None:
    train = _record("train", partition="train", group_id="dc")
    leaked = _record("cal", partition="calibration", group_id="dc")
    with pytest.raises(ValueError, match="cross train and calibration"):
        partition_group_held_out_records([train, leaked])

    held_out = _record("held", partition="calibration", group_id="phl")
    _, _, evidence = partition_group_held_out_records([train, held_out])
    assert evidence["overlap"] == []
    assert evidence["group_key"] == "source/group_id"


def test_threshold_uses_fp32_and_exact_per_stratum_non_regression() -> None:
    scenes = 64
    predicted = np.ones(scenes, dtype=np.float32)
    protected_rmse = np.full(scenes, 10.0)
    candidate_rmse = np.zeros(scenes)
    candidate_rmse[0] = 60.0
    actual = protected_rmse - candidate_rmse
    protected_sse = np.square(protected_rmse)
    candidate_sse = np.square(candidate_rmse)
    counts = np.ones(scenes, dtype=np.int64)
    strata = {
        "source": ["gamus"] * 32 + ["legacy"] * 32,
        "landscape": ["urban"] * scenes,
        "source_landscape": ["gamus/urban"] * 32 + ["legacy/urban"] * 32,
    }
    result = select_group_guarded_gain_threshold(
        predicted,
        actual,
        protected_rmse,
        candidate_rmse,
        protected_sse,
        candidate_sse,
        counts,
        strata,
        component="height",
        minimum_candidate_gain=0.25,
        minimum_precision=0.98,
        minimum_candidate_selections=32,
        minimum_stratum_support=32,
    )
    # 63/64 selections are beneficial and the pooled global RMSE improves, but
    # GAMUS alone regresses.  The per-stratum guard must keep this head closed.
    assert not result.eligible
    assert result.metrics["candidate_selected"] == 0
    assert result.threshold == float(np.float32(result.threshold))


def test_safe_threshold_and_training_keep_heads_independent() -> None:
    scenes = 32
    predicted = np.full(scenes, 0.5, dtype=np.float32)
    protected = np.ones(scenes)
    candidate = np.full(scenes, 0.5)
    exact_protected = np.ones(scenes)
    exact_candidate = np.full(scenes, 0.25)
    counts = np.ones(scenes, dtype=np.int64)
    strata = {
        "source": ["gamus"] * scenes,
        "landscape": ["forest"] * scenes,
        "source_landscape": ["gamus/forest"] * scenes,
    }
    threshold = select_group_guarded_gain_threshold(
        predicted,
        protected - candidate,
        protected,
        candidate,
        exact_protected,
        exact_candidate,
        counts,
        strata,
        component="height",
        minimum_candidate_gain=0.25,
    )
    assert threshold.eligible
    assert threshold.metrics["precision"] == pytest.approx(1.0)
    assert threshold.metrics["routed_gain_vs_protected"] == pytest.approx(0.5)

    records = [
        _record(
            f"r{index}",
            partition="train" if index < 8 else "calibration",
            group_id="nyc" if index < 8 else "dc",
            protected_height=1.0,
            candidate_height=0.5,
        )
        for index in range(12)
    ]
    train, validation, _ = partition_group_held_out_records(records)
    train_dataset = Stage4GainDataset(train, component="height")
    validation_dataset = Stage4GainDataset(validation, component="height")
    router = EndpointGainRouter(4, hidden_features=4, target_scale=10.0)
    class CountingLoader:
        def __init__(self, loader: DataLoader) -> None:
            self.loader = loader
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(self.loader)

    calibration_loader = CountingLoader(
        DataLoader(validation_dataset, batch_size=4, shuffle=False)
    )
    result = train_gain_router(
        router,
        DataLoader(train_dataset, batch_size=4, shuffle=False),
        calibration_loader,  # type: ignore[arg-type]
        component="height",
        config=GainRouterTrainingConfig(
            epochs=2,
            minimum_candidate_gain=0.25,
            minimum_precision=0.0,
            minimum_candidate_selections=1,
            minimum_stratum_support=1,
        ),
    )
    assert len(result.history) == 2
    assert calibration_loader.iterations == 1
    assert all(set(epoch) == {"epoch", "train_huber"} for epoch in result.history)
    assert all(name.startswith("network.") for name in router.state_dict() if name.startswith("network."))


def test_ineligible_gain_router_is_hard_disabled() -> None:
    router = EndpointGainRouter(2, hidden_features=2)
    with torch.no_grad():
        final = router.network[-1]
        assert isinstance(final, torch.nn.Linear)
        final.bias.fill_(100.0)
    router.set_operating_point(threshold=-100.0, eligible=False)
    output = router.forward_descriptor(torch.ones(1, 2))
    assert not bool(output["candidate_selected"].item())


def test_close_endpoint_utilities_keep_exact_float64_gain_identity() -> None:
    record = _record(
        "close",
        partition="calibration",
        group_id="dc",
        protected_height=100.00001,
        candidate_height=100.0,
    )
    item = Stage4GainDataset([record], component="height")[0]
    assert item["target_gain"].dtype == torch.float64
    assert item["protected_utility"].dtype == torch.float64
    assert item["candidate_utility"].dtype == torch.float64
    actual = item["target_gain"].numpy()
    protected = item["protected_utility"].numpy()
    candidate = item["candidate_utility"].numpy()
    # This strict audit check failed when the three tensors were rounded to
    # float32 independently. It must pass without relaxing its tolerance.
    metrics = compute_gain_route_metrics(
        actual,
        actual,
        protected,
        candidate,
        threshold=0.0,
        minimum_candidate_gain=0.0,
    )
    assert metrics["scene_count"] == 1
