import json
from pathlib import Path
import re

import pytest
import torch

import msr.training.scene_router_record_store as record_store_module
from msr.training.scene_router import SceneRouterRecord, read_scene_router_records
from msr.training.scene_router_record_store import (
    AtomicSceneRouterRecordStore,
    RECORD_STORE_SCHEMA,
    build_record_store_provenance,
    canonical_json_sha256,
    file_sha256,
    validate_record_selection_provenance,
)


def test_atomic_json_retries_a_transient_windows_sharing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "parts_manifest.json"
    destination.write_text('{"state": "old"}\n', encoding="utf-8")
    real_replace = record_store_module.os.replace
    replace_calls: list[tuple[Path, Path]] = []
    sleeps: list[float] = []

    def transient_replace(source: Path, target: Path) -> None:
        replace_calls.append((source, target))
        if len(replace_calls) < 3:
            raise PermissionError(13, "simulated Windows sharing violation")
        real_replace(source, target)

    monkeypatch.setattr(record_store_module.os, "replace", transient_replace)
    monkeypatch.setattr(record_store_module.time, "sleep", sleeps.append)

    record_store_module._atomic_json(destination, {"state": "new"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"state": "new"}
    assert len(replace_calls) == 3
    assert sleeps == pytest.approx([0.025, 0.05])
    assert not destination.with_suffix(".json.tmp").exists()


def test_atomic_json_persistent_sharing_lock_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "parts_manifest.json"
    original = '{"state": "old"}\n'
    destination.write_text(original, encoding="utf-8")
    replace_calls = 0
    sleeps: list[float] = []

    def persistently_locked(_source: Path, _target: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        raise PermissionError(13, "simulated persistent Windows sharing violation")

    monkeypatch.setattr(record_store_module.os, "replace", persistently_locked)
    monkeypatch.setattr(record_store_module.time, "sleep", sleeps.append)

    with pytest.raises(PermissionError, match="persistent Windows sharing"):
        record_store_module._atomic_json(destination, {"state": "new"})

    assert destination.read_text(encoding="utf-8") == original
    assert json.loads(
        destination.with_suffix(".json.tmp").read_text(encoding="utf-8")
    ) == {"state": "new"}
    assert replace_calls == record_store_module._ATOMIC_JSON_REPLACE_ATTEMPTS
    assert len(sleeps) == record_store_module._ATOMIC_JSON_REPLACE_ATTEMPTS - 1


def _provenance(*, candidate_hash: str = "b" * 64) -> dict:
    return build_record_store_provenance(
        generation_config={
            "patch_size": 384,
            "crop": "center",
            "precision": "bf16",
        },
        endpoints={
            "protected": {"sha256": "a" * 64},
            "gamus_stage1": {"sha256": candidate_hash},
        },
        shared_model={"sha256": "c" * 64},
        data={"included": ["gamus/train", "gamus/val", "legacy/train", "legacy/val"]},
    )


def _record(key: str, partition: str = "train") -> SceneRouterRecord:
    return SceneRouterRecord(
        sample_id=key,
        descriptor=torch.tensor([1.0, 2.0]),
        fallback_utility=1.0,
        candidate_utility=0.5,
        source=key.split("/", 1)[0],
        landscape="mixed",
        partition=partition,
    )


def test_atomic_parts_resume_and_finalize_in_planned_order(tmp_path: Path) -> None:
    root = tmp_path / "records"
    store = AtomicSceneRouterRecordStore(root, _provenance())
    first = _record("gamus/train/a")
    second = _record("legacy/val/b", "calibration")
    part = store.append([first])

    assert re.fullmatch(r"part-00000000-[0-9a-f]{64}\.jsonl", part.name)
    assert not (root / "records.jsonl").exists()
    store.close()
    resumed = AtomicSceneRouterRecordStore(root, _provenance())
    assert resumed.completed_keys == {"gamus/train/a"}
    assert resumed.validate_expected_keys(
        ["gamus/train/a", "legacy/val/b"]
    ) == ["legacy/val/b"]
    resumed.append([second])
    completion = resumed.finalize(["legacy/val/b", "gamus/train/a"])

    assert completion["complete"]
    assert completion["record_count"] == 2
    assert completion["records_sha256"] == file_sha256(root / "records.jsonl")
    records = read_scene_router_records(root / "records.jsonl")
    assert [record.sample_id for record in records] == [
        "legacy/val/b",
        "gamus/train/a",
    ]
    resumed.close()


def test_store_refuses_provenance_change_duplicate_or_unknown_key(tmp_path: Path) -> None:
    root = tmp_path / "records"
    store = AtomicSceneRouterRecordStore(root, _provenance())
    store.append([_record("gamus/train/a")])

    with pytest.raises(ValueError, match="already exist"):
        store.append([_record("gamus/train/a")])
    with pytest.raises(ValueError, match="duplicate keys"):
        store.append([_record("legacy/train/x"), _record("legacy/train/x")])
    with pytest.raises(ValueError, match="outside the current data plan"):
        store.validate_expected_keys(["legacy/train/x"])
    store.close()

    with pytest.raises(ValueError, match="provenance/config mismatch"):
        AtomicSceneRouterRecordStore(
            root, _provenance(candidate_hash="d" * 64)
        )
    # A constructor failure must release the OS lock.
    reopened = AtomicSceneRouterRecordStore(root, _provenance())
    reopened.close()


def test_store_rejects_orphaned_parts_and_corrupt_resume(tmp_path: Path) -> None:
    orphan = tmp_path / "orphan"
    (orphan / "parts").mkdir(parents=True)
    (orphan / "parts" / "part-00000000.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="without provenance"):
        AtomicSceneRouterRecordStore(orphan, _provenance())

    corrupt = tmp_path / "corrupt"
    store = AtomicSceneRouterRecordStore(corrupt, _provenance())
    part = store.append([_record("gamus/train/a")])
    store.close()
    part.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid router record|SHA-256 mismatch"):
        AtomicSceneRouterRecordStore(corrupt, _provenance())


def test_provenance_schema_and_hash_are_deterministic(tmp_path: Path) -> None:
    left = _provenance()
    right = _provenance()
    assert left == right
    assert left["schema"] == RECORD_STORE_SCHEMA
    assert len(left["config_sha256"]) == 64
    assert left["test_splits_excluded"] is True
    assert "source/split/landscape are metadata only" in left["target_rule"]

    tampered = json.loads(json.dumps(left))
    tampered["generation_config"]["patch_size"] = 256
    with pytest.raises(ValueError, match="does not match its contents"):
        AtomicSceneRouterRecordStore(tmp_path / "tampered", tampered)


def test_record_selection_provenance_fails_closed() -> None:
    valid = {
        "planned_record_count": 5,
        "excluded_no_reference_count": 1,
        "excluded_no_reference_record_keys": ["gamus/train/no-reference"],
        "scored_record_count": 4,
        "planned_partition_counts": {"train": 4, "calibration": 1},
        "scored_partition_counts": {"train": 3, "calibration": 1},
    }
    validate_record_selection_provenance(valid, completed_record_count=4)

    malformed: list[tuple[dict, str]] = []
    missing = json.loads(json.dumps(valid))
    missing.pop("excluded_no_reference_record_keys")
    malformed.append((missing, "missing fields"))
    empty = json.loads(json.dumps(valid))
    empty["excluded_no_reference_record_keys"] = []
    empty["excluded_no_reference_count"] = 0
    malformed.append((empty, "non-empty string list"))
    duplicate = json.loads(json.dumps(valid))
    duplicate["excluded_no_reference_record_keys"] *= 2
    duplicate["excluded_no_reference_count"] = 2
    duplicate["planned_record_count"] = 6
    duplicate["planned_partition_counts"]["train"] = 5
    malformed.append((duplicate, "duplicates"))
    calibration = json.loads(json.dumps(valid))
    calibration["excluded_no_reference_record_keys"] = ["gamus/val/no-reference"]
    malformed.append((calibration, "canonical train keys"))
    inconsistent = json.loads(json.dumps(valid))
    inconsistent["scored_record_count"] = 3
    malformed.append((inconsistent, "planned/excluded/scored"))
    partition = json.loads(json.dumps(valid))
    partition["scored_partition_counts"] = {"train": 2, "calibration": 2}
    malformed.append((partition, "only the train partition"))

    for value, message in malformed:
        with pytest.raises(ValueError, match=message):
            validate_record_selection_provenance(value, completed_record_count=4)
    with pytest.raises(ValueError, match="completion count"):
        validate_record_selection_provenance(valid, completed_record_count=3)


def test_store_lock_is_exclusive_and_released_by_context(tmp_path: Path) -> None:
    root = tmp_path / "records"
    with AtomicSceneRouterRecordStore(root, _provenance()) as store:
        store.append([_record("gamus/train/a")])
        with pytest.raises(RuntimeError, match="already locked"):
            AtomicSceneRouterRecordStore(root, _provenance())

    resumed = AtomicSceneRouterRecordStore(root, _provenance())
    assert resumed.completed_keys == {"gamus/train/a"}
    resumed.close()


def test_store_recovers_hashed_orphan_tail_and_ignores_stale_pending(
    tmp_path: Path,
) -> None:
    root = tmp_path / "records"
    store = AtomicSceneRouterRecordStore(root, _provenance())
    store.append([_record("gamus/train/a")])
    first_manifest = json.loads(
        (root / "parts_manifest.json").read_text(encoding="utf-8")
    )
    store.append([_record("legacy/train/b")])
    store.close()

    # Crash point: the second immutable part was renamed, but its new ledger
    # was not published. Its content-addressed name makes recovery auditable.
    (root / "parts_manifest.json").write_text(
        json.dumps(first_manifest), encoding="utf-8"
    )
    (root / "parts" / "part-00000002.pending").write_text(
        "truncated scratch data", encoding="utf-8"
    )
    resumed = AtomicSceneRouterRecordStore(root, _provenance())
    assert resumed.completed_keys == {"gamus/train/a", "legacy/train/b"}
    resumed.append([_record("legacy/train/c")])
    resumed.close()


def test_store_rejects_cross_part_descriptor_size_drift(tmp_path: Path) -> None:
    root = tmp_path / "records"
    store = AtomicSceneRouterRecordStore(root, _provenance())
    store.append([_record("gamus/train/a")])
    mismatched = SceneRouterRecord(
        sample_id="legacy/train/b",
        descriptor=torch.tensor([1.0, 2.0, 3.0]),
        fallback_utility=1.0,
        candidate_utility=0.5,
        source="legacy",
        landscape="mixed",
    )
    with pytest.raises(ValueError, match="descriptor size differs"):
        store.append([mismatched])
    store.close()
    with pytest.raises(RuntimeError, match="closed and no longer locked"):
        store.append([_record("legacy/train/c")])
    with pytest.raises(RuntimeError, match="closed and no longer locked"):
        store.finalize(["gamus/train/a"])


def test_store_rejects_more_than_one_unledgered_part(tmp_path: Path) -> None:
    root = tmp_path / "records"
    store = AtomicSceneRouterRecordStore(root, _provenance())
    store.append([_record("gamus/train/a")])
    store.append([_record("legacy/train/b")])
    store.append([_record("legacy/train/c")])
    store.close()
    manifest_path = root / "parts_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parts"] = manifest["parts"][:1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="more than one unledgered part"):
        AtomicSceneRouterRecordStore(root, _provenance())


def test_completed_store_is_anchored_to_its_integrity_proof(tmp_path: Path) -> None:
    root = tmp_path / "records"
    store = AtomicSceneRouterRecordStore(root, _provenance())
    original_path = store.append([_record("gamus/train/a")])
    store.finalize(["gamus/train/a"])
    store.close()

    # Simulate a coordinated part+ledger edit. The prior completion proof must
    # still prevent the altered journal from being adopted or re-finalized.
    altered = _record("gamus/train/altered")
    payload = json.dumps(altered.to_json_dict(), sort_keys=True) + "\n"
    scratch = original_path.with_suffix(".replacement")
    scratch.write_text(payload, encoding="utf-8")
    digest = file_sha256(scratch)
    replacement = original_path.parent / f"part-00000000-{digest}.jsonl"
    original_path.unlink()
    scratch.rename(replacement)
    manifest_path = root / "parts_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["parts"][0]
    entry.update(
        {
            "filename": replacement.name,
            "sha256": digest,
            "keys_sha256": canonical_json_sha256([altered.sample_id]),
            "first_key": altered.sample_id,
            "last_key": altered.sample_id,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="completion proof does not match"):
        AtomicSceneRouterRecordStore(root, _provenance())
