from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from msr.training.scene_router import SceneRouterRecord
from msr.training.scene_router_record_store import (
    AtomicSceneRouterRecordStore,
    build_record_store_provenance,
)


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_scene_router.py"
SPEC = importlib.util.spec_from_file_location("train_scene_router_provenance_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _complete_store(
    tmp_path: Path, *, scored_record_count: int = 1
) -> tuple[Path, Path, Path]:
    provenance = build_record_store_provenance(
        generation_config={
            "utility": {"minimum_candidate_gain": 0.02},
            "patch_size": 384,
        },
        endpoints={
            "protected": {"sha256": "a" * 64},
            "gamus_stage1": {"sha256": "b" * 64},
            "compatibility": {"shared_state_sha256": "c" * 64},
        },
        shared_model={"base_checkpoint": {"sha256": "d" * 64}},
        data={
            "excluded_source_splits": ["gamus/test", "legacy/test"],
            "planned_record_count": scored_record_count + 1,
            "excluded_no_reference_count": 1,
            "excluded_no_reference_record_keys": ["gamus/train/excluded"],
            "scored_record_count": scored_record_count,
            "planned_partition_counts": {
                "train": scored_record_count + 1,
                "calibration": 0,
            },
            "scored_partition_counts": {
                "train": scored_record_count,
                "calibration": 0,
            },
        },
    )
    store = AtomicSceneRouterRecordStore(tmp_path, provenance)
    record = SceneRouterRecord(
        sample_id="gamus/train/a",
        descriptor=torch.tensor([1.0, 2.0]),
        fallback_utility=1.0,
        candidate_utility=0.5,
        source="gamus",
        landscape="mixed",
        partition="train",
    )
    store.append([record])
    store.finalize([record.sample_id])
    paths = store.records_path, store.provenance_path, store.completion_path
    store.close()
    return paths


def test_training_evidence_binds_completed_records_to_provenance(tmp_path: Path) -> None:
    records, provenance, completion = _complete_store(tmp_path)
    loaded_provenance, loaded_completion = MODULE.load_record_store_evidence(
        records,
        provenance_path=provenance,
        completion_path=completion,
        allow_unprovenanced=False,
    )
    assert loaded_provenance["test_splits_excluded"] is True
    assert loaded_completion["complete"] is True

    records.write_text(records.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSONL hash differs"):
        MODULE.load_record_store_evidence(
            records,
            provenance_path=provenance,
            completion_path=completion,
            allow_unprovenanced=False,
        )


def test_training_requires_stage3_evidence_unless_explicitly_development_only(
    tmp_path: Path,
) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires record provenance"):
        MODULE.load_record_store_evidence(
            records,
            provenance_path=None,
            completion_path=None,
            allow_unprovenanced=False,
        )
    assert MODULE.load_record_store_evidence(
        records,
        provenance_path=None,
        completion_path=None,
        allow_unprovenanced=True,
    ) == (None, None)


def test_training_rejects_completion_count_that_disagrees_with_provenance(
    tmp_path: Path,
) -> None:
    records, provenance, completion = _complete_store(
        tmp_path, scored_record_count=2
    )
    with pytest.raises(ValueError, match="provenance scored_record_count"):
        MODULE.load_record_store_evidence(
            records,
            provenance_path=provenance,
            completion_path=completion,
            allow_unprovenanced=False,
        )


def test_checked_in_training_config_uses_conservative_stage3_guards() -> None:
    config = Path(__file__).parents[1] / "configs" / "stage3_scene_router_training.yaml"
    args = MODULE.parse_args(["--config", str(config)])
    assert args.target_mode == "utility"
    assert args.records.is_absolute()
    assert args.output_dir.is_absolute()
    assert args.minimum_threshold == pytest.approx(0.90)
    assert args.minimum_precision == pytest.approx(0.98)
    assert args.maximum_mean_utility_regression == pytest.approx(0.0)
    assert args.minimum_candidate_gain == pytest.approx(0.02)
    assert args.fallback_weight > args.candidate_weight

    overridden = MODULE.parse_args(
        ["--config", str(config), "--batch-size", "12"]
    )
    assert overridden.batch_size == 12


def test_training_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "router.yaml"
    config.write_text(
        "router_training:\n"
        "  records: records.jsonl\n"
        "  output_dir: output\n"
        "  surprise_typo: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown router training config fields"):
        MODULE.parse_args(["--config", str(config)])
