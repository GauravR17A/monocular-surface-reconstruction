from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from msr.training.scene_router_record_store import (
    canonical_json_sha256,
    file_sha256,
)
from msr.training.stage4_dual_router import (
    EndpointComponentUtilities,
    STAGE4_RECORD_SCHEMA,
    STAGE4_RECORD_STORE_SCHEMA,
    Stage4DualRouterRecord,
)
from msr.training.stage4_dual_router_record_store import (
    AtomicStage4DualRouterRecordStore,
    build_stage4_record_store_provenance,
    load_completed_stage4_record_store,
    validate_stage4_record_selection_provenance,
)


KEYS = ("gamus/train/NYC_a", "legacy/val/city_b")


def _endpoint(value: float) -> EndpointComponentUtilities:
    return EndpointComponentUtilities(
        aggregate_utility=value,
        height_sse=value * value,
        height_rmse_m=value,
        semantic_confusion_3x3=((2, 0, 0), (0, 0, 0), (0, 0, 0)),
        semantic_balanced_error=0.0,
        valid_height_pixels=1,
        valid_semantic_pixels=2,
        observed_semantic_classes=1,
    )


def _record(key: str) -> Stage4DualRouterRecord:
    calibration = "/val/" in key
    return Stage4DualRouterRecord(
        sample_id=key,
        descriptor=torch.tensor([1.0, 2.0]),
        protected=_endpoint(1.0),
        candidate=_endpoint(0.5),
        source=key.split("/", 1)[0],
        group_id="highbuild:heldout" if calibration else "gamus:nyc",
        landscape="urban" if calibration else "mixed",
        partition="calibration" if calibration else "train",
    )


def _data_provenance(*, selected_hash: str | None = None) -> dict:
    stage3 = {
        "planned_record_count": 3,
        "excluded_no_reference_count": 1,
        "excluded_no_reference_record_keys": ["gamus/train/NYC_missing"],
        "scored_record_count": 2,
        "planned_partition_counts": {"train": 2, "calibration": 1},
        "scored_partition_counts": {"train": 1, "calibration": 1},
        "included_source_splits": [
            "gamus/train",
            "gamus/val",
            "legacy/train",
            "legacy/val",
        ],
        "excluded_source_splits": ["gamus/test", "legacy/test"],
    }
    selection = {
        "planned_record_count": 3,
        "stage3_scored_record_count": 2,
        "excluded_no_reference_count": 1,
        "excluded_no_reference_record_keys": ["gamus/train/NYC_missing"],
        "excluded_group_policy_count": 0,
        "excluded_group_policy_record_keys": [],
        "excluded_reason_counts": {
            "no_reference": 1,
            "gamus_train_city_group_overlap": 0,
        },
        "selected_record_count": 2,
        "selected_record_keys_sha256": selected_hash
        or canonical_json_sha256(list(KEYS)),
        "stage3_scored_record_key_set_sha256": canonical_json_sha256(sorted(KEYS)),
        "selected_partition_counts": {"train": 1, "calibration": 1},
        "selected_source_split_counts": {
            "gamus/train": 1,
            "gamus/val": 0,
            "legacy/train": 0,
            "legacy/val": 1,
        },
        "group_keys_by_partition": {
            "train": ["gamus/gamus:nyc"],
            "calibration": ["legacy/highbuild:heldout"],
        },
        "policy": "synthetic leakage-safe test policy",
    }
    return {
        "schema": "msr.stage4_dual_router_data.v1",
        "stage3_data_provenance": stage3,
        "controller_selection": selection,
    }


def _provenance(*, stage3_hash: str = "d" * 64) -> dict:
    return build_stage4_record_store_provenance(
        generation_config={
            "patch_size": 384,
            "crop": "center",
            "precision": "bf16",
        },
        endpoints={"protected": {"sha256": "a" * 64}},
        data=_data_provenance(),
        stage3={
            "authenticated": True,
            "provenance_sha256": stage3_hash,
            "record_key_set_sha256": canonical_json_sha256(sorted(KEYS)),
        },
    )


def test_atomic_resume_recovery_finalize_and_authenticated_load(tmp_path: Path) -> None:
    root = tmp_path / "rich-records"
    store = AtomicStage4DualRouterRecordStore(root, _provenance())
    first_part = store.append([_record(KEYS[0])])
    first_manifest = json.loads(
        (root / "parts_manifest.json").read_text(encoding="utf-8")
    )
    store.append([_record(KEYS[1])])
    store.close()

    # Simulate the exact crash point after the immutable part rename and before
    # publishing its ledger entry. One content-hashed tail is safely recovered.
    (root / "parts_manifest.json").write_text(
        json.dumps(first_manifest), encoding="utf-8"
    )
    resumed = AtomicStage4DualRouterRecordStore(root, _provenance())
    assert resumed.completed_keys == set(KEYS)
    completion = resumed.finalize(KEYS)
    resumed.close()

    records, provenance, loaded_completion = load_completed_stage4_record_store(root)
    assert [record.sample_id for record in records] == list(KEYS)
    assert completion == loaded_completion
    assert provenance["record_schema"] == STAGE4_RECORD_SCHEMA
    assert completion["schema"] == STAGE4_RECORD_STORE_SCHEMA
    assert completion["records_sha256"] == file_sha256(root / "records.jsonl")
    assert first_part.is_file()


def test_resume_rejects_changed_ancestry_and_corrupt_part(tmp_path: Path) -> None:
    root = tmp_path / "rich-records"
    store = AtomicStage4DualRouterRecordStore(root, _provenance())
    part = store.append([_record(KEYS[0])])
    store.close()
    with pytest.raises(ValueError, match="provenance mismatch"):
        AtomicStage4DualRouterRecordStore(root, _provenance(stage3_hash="e" * 64))

    part.write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid Stage-4 record|SHA-256 mismatch"):
        AtomicStage4DualRouterRecordStore(root, _provenance())


def test_finalize_binds_actual_keys_counts_and_groups_to_provenance(
    tmp_path: Path,
) -> None:
    bad_hash_data = _data_provenance(selected_hash="f" * 64)
    bad_hash_provenance = build_stage4_record_store_provenance(
        generation_config={},
        endpoints={},
        data=bad_hash_data,
        stage3={
            "authenticated": True,
            "record_key_set_sha256": canonical_json_sha256(sorted(KEYS)),
        },
    )
    with AtomicStage4DualRouterRecordStore(
        tmp_path / "bad-hash", bad_hash_provenance
    ) as store:
        with pytest.raises(ValueError, match="selection hash"):
            store.validate_expected_keys(KEYS)

    wrong_group = _data_provenance()
    wrong_group["controller_selection"]["group_keys_by_partition"]["train"] = [
        "gamus/gamus:not-nyc"
    ]
    provenance = build_stage4_record_store_provenance(
        generation_config={},
        endpoints={},
        data=wrong_group,
        stage3={
            "authenticated": True,
            "record_key_set_sha256": canonical_json_sha256(sorted(KEYS)),
        },
    )
    with AtomicStage4DualRouterRecordStore(
        tmp_path / "bad-group", provenance
    ) as store:
        store.append([_record(KEYS[0]), _record(KEYS[1])])
        with pytest.raises(ValueError, match="group keys disagree"):
            store.finalize(KEYS)


def test_stage4_selection_keeps_group_exclusions_distinct_from_no_reference() -> None:
    data = _data_provenance()
    selection = data["controller_selection"]
    selection["planned_record_count"] = 4
    selection["stage3_scored_record_count"] = 3
    selection["excluded_group_policy_count"] = 1
    selection["excluded_group_policy_record_keys"] = ["gamus/train/DC_a"]
    selection["excluded_reason_counts"]["gamus_train_city_group_overlap"] = 1
    selection["stage3_scored_record_key_set_sha256"] = canonical_json_sha256(
        sorted([*KEYS, "gamus/train/DC_a"])
    )
    data["stage3_data_provenance"]["planned_record_count"] = 4
    data["stage3_data_provenance"]["scored_record_count"] = 3
    data["stage3_data_provenance"]["planned_partition_counts"]["train"] = 3
    data["stage3_data_provenance"]["scored_partition_counts"]["train"] = 2
    validate_stage4_record_selection_provenance(data, completed_record_count=2)

    selection["excluded_group_policy_record_keys"] = [
        "gamus/train/NYC_not-a-policy-exclusion"
    ]
    with pytest.raises(ValueError, match="GAMUS DC/PHL"):
        validate_stage4_record_selection_provenance(data)
