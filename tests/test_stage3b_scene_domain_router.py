from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from msr.training.scene_router import SceneRouterRecord
from msr.training.scene_router_record_store import (
    AtomicSceneRouterRecordStore,
    build_record_store_provenance,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_scene_router.py"
SPEC = importlib.util.spec_from_file_location("stage3b_domain_router_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(
    source: str,
    split: str,
    sample: str,
    *,
    partition: str,
    fallback: float = 1.0,
    candidate: float = 0.5,
) -> SceneRouterRecord:
    return SceneRouterRecord(
        sample_id=f"{source}/{split}/{sample}",
        descriptor=torch.linspace(0.0, 1.0, 128),
        fallback_utility=fallback,
        candidate_utility=candidate,
        source=source,
        landscape="mixed",
        partition=partition,
    )


def _evidence() -> tuple[dict[str, object], dict[str, object]]:
    provenance = {
        "data": {
            "included_source_splits": [
                "gamus/train",
                "gamus/val",
                "legacy/train",
                "legacy/val",
            ],
            "excluded_source_splits": ["gamus/test", "legacy/test"],
        }
    }
    completion = {"descriptor_size": 128, "record_count": 4}
    return provenance, completion


def test_source_targets_ignore_utility_winner_and_use_descriptors_as_input() -> None:
    records = [
        _record(
            "gamus",
            "train",
            "g",
            partition="train",
            fallback=0.1,
            candidate=0.9,
        ),
        _record(
            "legacy",
            "train",
            "l",
            partition="train",
            fallback=0.9,
            candidate=0.1,
        ),
    ]
    dataset = MODULE.SourceGamusRouterDataset(records)
    assert dataset[0]["candidate_target"].item() == 1.0
    assert dataset[1]["candidate_target"].item() == 0.0
    assert torch.equal(dataset[0]["descriptor"], records[0].descriptor)
    assert set(dataset[0]) >= {
        "descriptor",
        "candidate_target",
        "fallback_utility",
        "candidate_utility",
    }


def test_source_sampler_exactly_balances_gamus_and_legacy() -> None:
    records = [
        *[
            _record("gamus", "train", f"g-{index}", partition="train")
            for index in range(5)
        ],
        *[
            _record("legacy", "train", f"l-{index}", partition="train")
            for index in range(2)
        ],
    ]
    sampler = MODULE.BalancedSourceSampler(records, seed=11)
    first = list(sampler)
    counts = Counter(records[index].source for index in first)
    assert counts == {"gamus": 5, "legacy": 5}
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_source_threshold_prefers_coverage_before_utility_gain() -> None:
    result = MODULE.select_source_gamus_threshold(
        np.array([0.95, 0.85, 0.60, 0.40, 0.10]),
        np.array([1, 1, 1, 0, 0]),
        np.ones(5),
        np.array([0.9, 0.9, 1.1, 0.1, 0.1]),
        minimum_threshold=0.5,
        minimum_source_precision=0.99,
        minimum_gamus_coverage=0.6,
        minimum_gamus_selections=2,
    )
    assert result.eligible
    assert result.threshold == pytest.approx(0.60)
    assert result.metrics["source_precision"] == pytest.approx(1.0)
    assert result.metrics["gamus_coverage"] == pytest.approx(1.0)
    assert result.metrics["selected_gamus"] == 3
    assert result.metrics["routed_utility_gain_vs_protected"] >= 0.0


def test_source_threshold_rejects_utility_regression_and_low_precision() -> None:
    regressing = MODULE.select_source_gamus_threshold(
        np.array([0.95, 0.90, 0.10]),
        np.array([1, 1, 0]),
        np.ones(3),
        np.array([2.0, 2.0, 0.0]),
        minimum_threshold=0.5,
        minimum_source_precision=0.99,
        minimum_gamus_coverage=0.5,
        minimum_gamus_selections=1,
    )
    assert not regressing.eligible
    assert regressing.metrics["candidate_selected"] == 0
    assert regressing.threshold > 1.0

    precise = MODULE.select_source_gamus_threshold(
        np.array([0.90, 0.80, 0.85, 0.10]),
        np.array([1, 1, 0, 0]),
        np.ones(4),
        np.zeros(4),
        minimum_threshold=0.5,
        minimum_source_precision=0.99,
        minimum_gamus_coverage=0.5,
        minimum_gamus_selections=1,
    )
    assert precise.eligible
    assert precise.threshold == pytest.approx(0.90)
    assert precise.metrics["source_precision"] == pytest.approx(1.0)
    assert precise.metrics["gamus_coverage"] == pytest.approx(0.5)

    saturated = MODULE.select_source_gamus_threshold(
        np.array([1.0, 1.0, 0.0]),
        np.array([1, 1, 0]),
        np.ones(3),
        np.array([2.0, 2.0, 0.0]),
        minimum_threshold=0.5,
        minimum_source_precision=0.99,
        minimum_gamus_coverage=0.5,
        minimum_gamus_selections=1,
    )
    assert not saturated.eligible
    assert saturated.threshold > 1.0
    assert saturated.metrics["candidate_selected"] == 0


def test_source_record_contract_rejects_wrong_width_or_split() -> None:
    train = [
        _record("gamus", "train", "g", partition="train"),
        _record("legacy", "train", "l", partition="train"),
    ]
    calibration = [
        _record("gamus", "val", "g", partition="calibration"),
        _record("legacy", "val", "l", partition="calibration"),
    ]
    provenance, completion = _evidence()
    counts = MODULE.validate_source_gamus_records(
        train,
        calibration,
        record_provenance=provenance,
        record_completion=completion,
    )
    assert counts == {
        "train": {"gamus": 1, "legacy": 1},
        "calibration": {"gamus": 1, "legacy": 1},
    }

    bad_split = list(train)
    bad_split[0] = _record("gamus", "test", "g", partition="train")
    with pytest.raises(ValueError, match="source/split"):
        MODULE.validate_source_gamus_records(
            bad_split,
            calibration,
            record_provenance=provenance,
            record_completion=completion,
        )
    bad_completion = {"descriptor_size": 64, "record_count": 4}
    with pytest.raises(ValueError, match="128-D"):
        MODULE.validate_source_gamus_records(
            train,
            calibration,
            record_provenance=provenance,
            record_completion=bad_completion,
        )


def test_checked_in_stage3b_config_and_runner_are_versioned_and_strict() -> None:
    project_root = Path(__file__).parents[1]
    config = project_root / "configs" / "stage3b_scene_domain_router_training.yaml"
    args = MODULE.parse_args(["--config", str(config)])
    assert args.target_mode == "source_gamus"
    assert args.minimum_precision == pytest.approx(0.99)
    assert args.minimum_gamus_coverage == pytest.approx(0.80)
    assert args.minimum_gamus_selections == 400
    assert args.maximum_mean_utility_regression == pytest.approx(0.0)
    assert args.output_dir.name == "stage3_scene_domain_router_guarded"

    runner = (
        project_root / "scripts" / "run_stage3b_scene_domain_router_training.ps1"
    ).read_text(encoding="utf-8")
    assert "stage3b_scene_domain_router_training.yaml" in runner
    assert "scene_domain_router.pt" in runner
    assert "stage3_scene_router_guarded" not in runner
    assert '$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"' in runner


def test_source_mode_writes_distinct_honest_artifact_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        _record("gamus", "train", "g", partition="train"),
        _record("legacy", "train", "l", partition="train"),
        _record("gamus", "val", "g", partition="calibration"),
        _record("legacy", "val", "l", partition="calibration"),
    ]
    data = {
        "included_source_splits": [
            "gamus/train",
            "gamus/val",
            "legacy/train",
            "legacy/val",
        ],
        "excluded_source_splits": ["gamus/test", "legacy/test"],
        "planned_record_count": 5,
        "excluded_no_reference_count": 1,
        "excluded_no_reference_record_keys": ["gamus/train/excluded"],
        "scored_record_count": 4,
        "planned_partition_counts": {"train": 3, "calibration": 2},
        "scored_partition_counts": {"train": 2, "calibration": 2},
    }
    provenance = build_record_store_provenance(
        generation_config={"utility": {"minimum_candidate_gain": 0.02}},
        endpoints={"protected": {}, "gamus_stage1": {}},
        shared_model={"base_checkpoint": {}},
        data=data,
    )
    store = AtomicSceneRouterRecordStore(tmp_path / "records", provenance)
    store.append(records)
    store.finalize([record.sample_id for record in records])
    record_paths = store.records_path, store.provenance_path, store.completion_path
    store.close()
    output = tmp_path / "domain-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--records",
            str(record_paths[0]),
            "--record-provenance",
            str(record_paths[1]),
            "--record-completion",
            str(record_paths[2]),
            "--output-dir",
            str(output),
            "--target-mode",
            "source_gamus",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-features",
            "4",
            "--minimum-threshold",
            "0.5",
            "--minimum-precision",
            "0.99",
            "--minimum-gamus-coverage",
            "0.5",
            "--minimum-gamus-selections",
            "1",
            "--device",
            "cpu",
        ],
    )
    MODULE.main()

    artifact_path = output / "scene_domain_router.pt"
    report_path = output / "scene_domain_router_report.json"
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert artifact["artifact_schema"] == "msr.stage3b_scene_domain_router.v1"
    assert report["artifact_schema"] == (
        "msr.stage3b_scene_domain_router_report.v1"
    )
    assert artifact["target_mode"] == "source_gamus"
    assert "record.source is GAMUS" in artifact["target_provenance"]
    assert report["source_threshold_guards"]["minimum_source_precision"] == 0.99
    assert not (output / "scene_router.pt").exists()
