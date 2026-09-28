"""Crash-safe, provenance-locked storage for Stage-4 rich router records.

This is intentionally a new journal format.  Stage-3 record files omit the
component statistics needed by the two Stage-4 routing heads and must never be
silently treated as rich records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from .scene_router_record_store import (
    canonical_json_sha256,
    file_sha256,
    validate_record_selection_provenance,
)
from .stage4_dual_router import (
    STAGE4_RECORD_SCHEMA,
    STAGE4_RECORD_STORE_SCHEMA,
    Stage4DualRouterRecord,
    read_stage4_dual_router_records,
)


_PART_PATTERN = re.compile(r"^part-(\d{8})-([0-9a-f]{64})\.jsonl$")
_ATOMIC_REPLACE_ATTEMPTS = 8
_ATOMIC_REPLACE_INITIAL_DELAY_SECONDS = 0.025
_ATOMIC_REPLACE_MAX_DELAY_SECONDS = 0.2


def _json_clone(value: object) -> Any:
    """Return a JSON-only deep copy or fail before writing provenance."""

    return json.loads(json.dumps(value, sort_keys=True))


def _replace_with_retry(source: Path, destination: Path) -> None:
    delay = _ATOMIC_REPLACE_INITIAL_DELAY_SECONDS
    for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == _ATOMIC_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, _ATOMIC_REPLACE_MAX_DELAY_SECONDS)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(temporary, path)


def _atomic_records(
    path: Path, records: Sequence[Stage4DualRouterRecord]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(temporary, path)


def validate_stage4_record_selection_provenance(
    data: object, *, completed_record_count: object | None = None
) -> None:
    """Validate distinct no-reference and group-safety exclusion ledgers."""

    if not isinstance(data, Mapping):
        raise ValueError("Stage-4 data provenance must be a mapping")
    stage3_data = data.get("stage3_data_provenance")
    selection = data.get("controller_selection")
    if not isinstance(stage3_data, Mapping) or not isinstance(selection, Mapping):
        raise ValueError(
            "Stage-4 data provenance requires Stage-3 data and controller selection"
        )
    validate_record_selection_provenance(stage3_data)
    if stage3_data.get("excluded_source_splits") != [
        "gamus/test",
        "legacy/test",
    ]:
        raise ValueError("Stage-4 ancestry must exclude both official test splits")

    required = {
        "planned_record_count",
        "stage3_scored_record_count",
        "excluded_no_reference_count",
        "excluded_no_reference_record_keys",
        "excluded_group_policy_count",
        "excluded_group_policy_record_keys",
        "excluded_reason_counts",
        "selected_record_count",
        "selected_record_keys_sha256",
        "stage3_scored_record_key_set_sha256",
        "selected_partition_counts",
        "selected_source_split_counts",
        "group_keys_by_partition",
        "policy",
    }
    missing = sorted(required - set(selection))
    if missing:
        raise ValueError(f"Stage-4 controller selection is missing fields: {missing}")

    def count(name: str, *, allow_zero: bool = False) -> int:
        value = selection.get(name)
        minimum = 0 if allow_zero else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Stage-4 selection {name} has an invalid count")
        return value

    planned = count("planned_record_count")
    stage3_scored = count("stage3_scored_record_count")
    no_reference_count = count("excluded_no_reference_count", allow_zero=True)
    group_policy_count = count("excluded_group_policy_count", allow_zero=True)
    selected_count = count("selected_record_count")

    def keys(name: str, *, allowed_cities: set[str] | None = None) -> list[str]:
        value = selection.get(name)
        if not isinstance(value, list) or any(not isinstance(key, str) for key in value):
            raise ValueError(f"Stage-4 selection {name} must be a string list")
        if len(value) != len(set(value)):
            raise ValueError(f"Stage-4 selection {name} contains duplicates")
        for key in value:
            components = key.split("/")
            if (
                len(components) != 3
                or components[0] not in {"gamus", "legacy"}
                or components[1] != "train"
                or not components[2]
                or key.strip() != key
                or "\\" in key
            ):
                raise ValueError(
                    f"Stage-4 exclusion is not a canonical original-train key: {key!r}"
                )
            if allowed_cities is not None:
                city = components[2].split("_", maxsplit=1)[0].upper()
                if components[0] != "gamus" or city not in allowed_cities:
                    raise ValueError(
                        "Stage-4 group-policy exclusions must be GAMUS DC/PHL train rows"
                    )
        return value

    no_reference_keys = keys("excluded_no_reference_record_keys")
    group_policy_keys = keys(
        "excluded_group_policy_record_keys", allowed_cities={"DC", "PHL"}
    )
    if len(no_reference_keys) != no_reference_count:
        raise ValueError("Stage-4 no-reference exclusion count disagrees with ledger")
    if len(group_policy_keys) != group_policy_count:
        raise ValueError("Stage-4 group-policy exclusion count disagrees with ledger")
    if set(no_reference_keys) & set(group_policy_keys):
        raise ValueError("Stage-4 exclusion ledgers overlap")
    if no_reference_keys != stage3_data.get("excluded_no_reference_record_keys"):
        raise ValueError("Stage-4 no-reference ledger differs from Stage-3 ancestry")
    if no_reference_count != stage3_data.get("excluded_no_reference_count"):
        raise ValueError("Stage-4 no-reference count differs from Stage-3 ancestry")
    if planned != stage3_data.get("planned_record_count"):
        raise ValueError("Stage-4 planned count differs from Stage-3 ancestry")
    if stage3_scored != stage3_data.get("scored_record_count"):
        raise ValueError("Stage-4 scored ancestry count differs from Stage-3")
    if stage3_scored != planned - no_reference_count:
        raise ValueError("Stage-4 Stage-3 scored count is internally inconsistent")
    if planned != no_reference_count + group_policy_count + selected_count:
        raise ValueError(
            "Stage-4 planned count must equal selected plus distinct exclusion ledgers"
        )

    reason_counts = selection["excluded_reason_counts"]
    if reason_counts != {
        "no_reference": no_reference_count,
        "gamus_train_city_group_overlap": group_policy_count,
    }:
        raise ValueError("Stage-4 exclusion reason counts disagree with ledgers")
    partition_counts = selection["selected_partition_counts"]
    if (
        not isinstance(partition_counts, Mapping)
        or set(partition_counts) != {"train", "calibration"}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in partition_counts.values()
        )
        or sum(partition_counts.values()) != selected_count
    ):
        raise ValueError("Stage-4 selected partition counts are invalid")
    source_split_counts = selection["selected_source_split_counts"]
    allowed_source_splits = {
        "gamus/train",
        "gamus/val",
        "legacy/train",
        "legacy/val",
    }
    if (
        not isinstance(source_split_counts, Mapping)
        or set(source_split_counts) != allowed_source_splits
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in source_split_counts.values()
        )
        or sum(source_split_counts.values()) != selected_count
    ):
        raise ValueError("Stage-4 selected source/split counts are invalid")
    for name in (
        "selected_record_keys_sha256",
        "stage3_scored_record_key_set_sha256",
    ):
        digest = selection[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"Stage-4 selection {name} is invalid")

    groups = selection["group_keys_by_partition"]
    if not isinstance(groups, Mapping) or set(groups) != {"train", "calibration"}:
        raise ValueError("Stage-4 group ledger must contain train and calibration")
    parsed_groups: dict[str, list[str]] = {}
    for partition in ("train", "calibration"):
        value = groups[partition]
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(group, str) or not group.strip() for group in value)
            or len(value) != len(set(value))
            or value != sorted(value)
        ):
            raise ValueError(f"Stage-4 {partition} group ledger is invalid")
        parsed_groups[partition] = value
    if set(parsed_groups["train"]) & set(parsed_groups["calibration"]):
        raise ValueError("Stage-4 train/calibration group IDs overlap")
    if not isinstance(selection["policy"], str) or not selection["policy"].strip():
        raise ValueError("Stage-4 controller selection policy is missing")
    if completed_record_count is not None:
        if isinstance(completed_record_count, bool) or not isinstance(
            completed_record_count, int
        ):
            raise ValueError("Stage-4 completion count must be an integer")
        if completed_record_count != selected_count:
            raise ValueError("Stage-4 completion count differs from selected count")


def build_stage4_record_store_provenance(
    *,
    generation_config: Mapping[str, Any],
    endpoints: Mapping[str, Any],
    data: Mapping[str, Any],
    stage3: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable identity for one Stage-4 record journal.

    ``stage3`` contains hashes and parsed identities for the authenticated
    Stage-3 provenance, completion proof, and consolidated records.  Keeping it
    inside the identity makes an apparently identical endpoint/data run refuse
    to resume if its Stage-3 ancestry changes.
    """

    generation_value = _json_clone(generation_config)
    endpoint_value = _json_clone(endpoints)
    data_value = _json_clone(data)
    stage3_value = _json_clone(stage3)
    validate_stage4_record_selection_provenance(data_value)
    if stage3_value.get("authenticated") is not True:
        raise ValueError("Stage-4 provenance requires authenticated Stage-3 ancestry")
    selection = data_value["controller_selection"]
    if selection["stage3_scored_record_key_set_sha256"] != stage3_value.get(
        "record_key_set_sha256"
    ):
        raise ValueError(
            "Stage-4 selection does not cover the authenticated Stage-3 key set"
        )
    identity_inputs = {
        "generation_config": generation_value,
        "endpoints": endpoint_value,
        "data": data_value,
        "stage3": stage3_value,
    }
    return {
        "schema": STAGE4_RECORD_STORE_SCHEMA,
        "record_schema": STAGE4_RECORD_SCHEMA,
        **identity_inputs,
        "identity_sha256": canonical_json_sha256(identity_inputs),
        "test_splits_excluded": True,
        "app_pointer_changed": False,
    }


class AtomicStage4DualRouterRecordStore:
    """Content-addressed append journal with strict, resumable integrity.

    A batch first lands in a narrow ``.pending`` scratch file, is flushed, and
    is renamed to a filename containing its SHA-256.  The manifest is updated
    atomically only after the immutable part has been re-read and validated.
    At most one unledgered tail part is recoverable after a crash.
    """

    def __init__(self, root: str | Path, provenance: Mapping[str, Any]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.parts_dir = self.root / "parts"
        self.provenance_path = self.root / "provenance.json"
        self.parts_manifest_path = self.root / "parts_manifest.json"
        self.records_path = self.root / "records.jsonl"
        self.completion_path = self.root / "completion.json"
        self.lock_path = self.root / ".record_store.lock"
        self._lock_handle: Any | None = None

        supplied = _json_clone(provenance)
        self._validate_provenance(supplied)
        self.root.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        try:
            preexisting_parts = list((self.root / "parts").glob("part-*.jsonl"))
            if self.provenance_path.exists():
                existing = json.loads(
                    self.provenance_path.read_text(encoding="utf-8")
                )
                if canonical_json_sha256(existing) != canonical_json_sha256(supplied):
                    raise ValueError(
                        "Stage-4 record-store provenance mismatch; refusing to mix "
                        "records from different endpoints, data, or Stage-3 ancestry"
                    )
            else:
                if preexisting_parts or self.records_path.exists():
                    raise ValueError(
                        "Stage-4 record data exists without provenance; refusing unsafe resume"
                    )
                _atomic_json(self.provenance_path, supplied)
            self.provenance = supplied
            self.parts_dir.mkdir(parents=True, exist_ok=True)
            self._records_by_key: dict[str, Stage4DualRouterRecord] = {}
            self._part_entries: list[dict[str, Any]] = []
            self._part_paths: list[Path] = []
            self._descriptor_size: int | None = None
            self._completion: dict[str, Any] | None = None
            self._load_parts()
            if self.completion_path.exists():
                self._completion = self._validate_existing_completion()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _validate_provenance(value: Mapping[str, Any]) -> None:
        if value.get("schema") != STAGE4_RECORD_STORE_SCHEMA:
            raise ValueError(
                f"Stage-4 record provenance schema must be "
                f"{STAGE4_RECORD_STORE_SCHEMA!r}"
            )
        if value.get("record_schema") != STAGE4_RECORD_SCHEMA:
            raise ValueError("Stage-4 record provenance has the wrong rich-record schema")
        if value.get("test_splits_excluded") is not True:
            raise ValueError("Stage-4 provenance must explicitly exclude test splits")
        if value.get("app_pointer_changed") is not False:
            raise ValueError("Stage-4 record generation may not change the app pointer")
        identity = {
            name: value.get(name)
            for name in ("generation_config", "endpoints", "data", "stage3")
        }
        expected_hash = canonical_json_sha256(identity)
        if value.get("identity_sha256") != expected_hash:
            raise ValueError("Stage-4 provenance identity SHA-256 does not match contents")
        if not isinstance(identity["data"], Mapping):
            raise ValueError("Stage-4 data provenance must be a mapping")
        validate_stage4_record_selection_provenance(identity["data"])
        if not isinstance(identity["stage3"], Mapping) or identity["stage3"].get(
            "authenticated"
        ) is not True:
            raise ValueError("Stage-4 provenance has no authenticated Stage-3 ancestry")

    def _acquire_lock(self) -> None:
        handle = self.lock_path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
                os.fsync(handle.fileno())
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - Windows is the production host.
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            handle.close()
            raise RuntimeError(
                f"Stage-4 record store is already locked: {self.root}"
            ) from error
        self._lock_handle = handle

    def close(self) -> None:
        handle = self._lock_handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._lock_handle = None

    def __enter__(self) -> "AtomicStage4DualRouterRecordStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        if self._lock_handle is None:
            raise RuntimeError("Stage-4 record store is closed and no longer locked")

    def _manifest_value(self) -> dict[str, Any]:
        return {
            "schema": STAGE4_RECORD_STORE_SCHEMA,
            "record_schema": STAGE4_RECORD_SCHEMA,
            "identity_sha256": self.provenance["identity_sha256"],
            "parts": self._part_entries,
        }

    def _part_entry(
        self, path: Path, records: Sequence[Stage4DualRouterRecord]
    ) -> dict[str, Any]:
        if not records:
            raise ValueError("Stage-4 part cannot be empty")
        match = _PART_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected Stage-4 part filename: {path.name}")
        actual_hash = file_sha256(path)
        if actual_hash != match.group(2):
            raise ValueError(f"Stage-4 part SHA-256 mismatch: {path.name}")
        descriptor_sizes = {int(record.descriptor.numel()) for record in records}
        if len(descriptor_sizes) != 1:
            raise ValueError("Stage-4 part mixes descriptor sizes")
        keys = [record.sample_id for record in records]
        group_ids = [record.group_id for record in records]
        return {
            "index": int(match.group(1)),
            "filename": path.name,
            "sha256": actual_hash,
            "record_count": len(records),
            "descriptor_size": descriptor_sizes.pop(),
            "keys_sha256": canonical_json_sha256(keys),
            "group_ids_sha256": canonical_json_sha256(group_ids),
            "first_key": keys[0],
            "last_key": keys[-1],
        }

    def _write_manifest(self) -> None:
        self._ensure_open()
        _atomic_json(self.parts_manifest_path, self._manifest_value())

    def _load_parts(self) -> None:
        paths = sorted(self.parts_dir.glob("part-*.jsonl"))
        matches = [_PART_PATTERN.fullmatch(path.name) for path in paths]
        if any(match is None for match in matches):
            raise ValueError("Stage-4 part sequence contains an unexpected filename")
        indices = [int(match.group(1)) for match in matches if match is not None]
        if indices != list(range(len(paths))):
            raise ValueError("Stage-4 part sequence contains a gap")

        if self.parts_manifest_path.exists():
            manifest = json.loads(
                self.parts_manifest_path.read_text(encoding="utf-8")
            )
            if (
                manifest.get("schema") != STAGE4_RECORD_STORE_SCHEMA
                or manifest.get("record_schema") != STAGE4_RECORD_SCHEMA
                or manifest.get("identity_sha256")
                != self.provenance["identity_sha256"]
                or not isinstance(manifest.get("parts"), list)
            ):
                raise ValueError("Stage-4 parts manifest identity is invalid")
            recorded_entries = manifest["parts"]
            if len(recorded_entries) > len(paths):
                raise ValueError("Stage-4 parts manifest references a missing part")
        else:
            recorded_entries = []
        if len(paths) - len(recorded_entries) > 1:
            raise ValueError("Stage-4 store has more than one unledgered part")

        for path in paths:
            records = read_stage4_dual_router_records(path)
            entry = self._part_entry(path, records)
            descriptor_size = int(entry["descriptor_size"])
            if self._descriptor_size is None:
                self._descriptor_size = descriptor_size
            elif descriptor_size != self._descriptor_size:
                raise ValueError("Stage-4 parts contain inconsistent descriptor sizes")
            if entry["index"] < len(recorded_entries):
                if recorded_entries[entry["index"]] != entry:
                    raise ValueError(f"Stage-4 manifest mismatch for {path.name}")
            self._part_paths.append(path)
            self._part_entries.append(entry)
            for record in records:
                if record.sample_id in self._records_by_key:
                    raise ValueError(
                        f"duplicate Stage-4 record key on resume: {record.sample_id}"
                    )
                self._records_by_key[record.sample_id] = record
        if recorded_entries != self._part_entries:
            if recorded_entries != self._part_entries[: len(recorded_entries)]:
                raise ValueError("Stage-4 parts manifest is not a valid prefix")
            self._write_manifest()
        elif not self.parts_manifest_path.exists():
            self._write_manifest()

    def _validate_existing_completion(self) -> dict[str, Any]:
        try:
            completion = json.loads(
                self.completion_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Stage-4 completion proof is unreadable") from error
        checks = (
            completion.get("schema") == STAGE4_RECORD_STORE_SCHEMA,
            completion.get("record_schema") == STAGE4_RECORD_SCHEMA,
            completion.get("identity_sha256") == self.provenance["identity_sha256"],
            completion.get("complete") is True,
            completion.get("record_count") == len(self._records_by_key),
            completion.get("part_count") == len(self._part_paths),
            completion.get("descriptor_size") == self._descriptor_size,
            completion.get("parts_manifest_canonical_sha256")
            == canonical_json_sha256(self._manifest_value()),
            completion.get("records_path") == str(self.records_path),
        )
        if not all(checks) or not self.records_path.is_file():
            raise ValueError("Stage-4 completion proof does not match the record store")
        if completion.get("records_sha256") != file_sha256(self.records_path):
            raise ValueError("completed Stage-4 records SHA-256 mismatch")
        records = read_stage4_dual_router_records(self.records_path)
        keys = [record.sample_id for record in records]
        if len(keys) != len(set(keys)) or set(keys) != set(self._records_by_key):
            raise ValueError("completed Stage-4 records do not match immutable parts")
        self._validate_records_against_provenance(records, expected_keys=keys)
        return completion

    def _validate_records_against_provenance(
        self,
        records: Sequence[Stage4DualRouterRecord],
        *,
        expected_keys: Sequence[str],
    ) -> None:
        """Bind the declared controller selection to the records themselves."""

        selection = self.provenance["data"]["controller_selection"]
        actual_keys = [record.sample_id for record in records]
        if actual_keys != list(expected_keys):
            raise ValueError("Stage-4 record order differs from expected keys")
        if canonical_json_sha256(actual_keys) != selection[
            "selected_record_keys_sha256"
        ]:
            raise ValueError(
                "Stage-4 selected-record key hash disagrees with actual records"
            )
        inherited_key_set = sorted(
            [
                *actual_keys,
                *selection["excluded_group_policy_record_keys"],
            ]
        )
        if canonical_json_sha256(inherited_key_set) != selection[
            "stage3_scored_record_key_set_sha256"
        ]:
            raise ValueError(
                "Stage-4 selected and group-policy ledgers do not cover the "
                "authenticated Stage-3 scored key set"
            )
        partition_counts = {"train": 0, "calibration": 0}
        source_split_counts = {
            "gamus/train": 0,
            "gamus/val": 0,
            "legacy/train": 0,
            "legacy/val": 0,
        }
        groups: dict[str, set[str]] = {"train": set(), "calibration": set()}
        for record in records:
            if record.partition not in partition_counts:
                raise ValueError(
                    f"Stage-4 generated record has forbidden partition: "
                    f"{record.partition!r}"
                )
            components = record.sample_id.split("/")
            if len(components) != 3:
                raise ValueError(
                    f"Stage-4 record key is not source/split/sample: {record.sample_id!r}"
                )
            source, split, _ = components
            if source != record.source:
                raise ValueError("Stage-4 record source disagrees with its sample key")
            source_split = f"{source}/{split}"
            if source_split not in source_split_counts:
                raise ValueError(
                    f"Stage-4 record uses a forbidden source split: {source_split!r}"
                )
            partition_counts[record.partition] += 1
            source_split_counts[source_split] += 1
            groups[record.partition].add(record.group_key)
        if partition_counts != selection["selected_partition_counts"]:
            raise ValueError(
                "Stage-4 actual partition counts disagree with provenance"
            )
        if source_split_counts != selection["selected_source_split_counts"]:
            raise ValueError(
                "Stage-4 actual source/split counts disagree with provenance"
            )
        actual_groups = {
            partition: sorted(values) for partition, values in groups.items()
        }
        if actual_groups != selection["group_keys_by_partition"]:
            raise ValueError("Stage-4 actual canonical group keys disagree with provenance")
        if groups["train"] & groups["calibration"]:
            raise ValueError("Stage-4 actual train/calibration groups overlap")

    @property
    def completed_keys(self) -> frozenset[str]:
        return frozenset(self._records_by_key)

    @property
    def record_count(self) -> int:
        return len(self._records_by_key)

    @property
    def next_part_index(self) -> int:
        return len(self._part_paths)

    @property
    def completion(self) -> dict[str, Any] | None:
        """Return a copy of the authenticated completion proof, if finalized."""

        return None if self._completion is None else dict(self._completion)

    def append(self, records: Sequence[Stage4DualRouterRecord]) -> Path:
        self._ensure_open()
        if self._completion is not None:
            raise ValueError("cannot append to a completed Stage-4 record store")
        if not records:
            raise ValueError("cannot append an empty Stage-4 batch")
        keys = [record.sample_id for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError("incoming Stage-4 batch contains duplicate keys")
        duplicates = sorted(set(keys) & self.completed_keys)
        if duplicates:
            raise ValueError(
                f"Stage-4 record keys already exist; resume must skip them: "
                f"{duplicates[:5]}"
            )
        descriptor_sizes = {int(record.descriptor.numel()) for record in records}
        if len(descriptor_sizes) != 1:
            raise ValueError("incoming Stage-4 batch mixes descriptor sizes")
        descriptor_size = descriptor_sizes.pop()
        if self._descriptor_size is not None and descriptor_size != self._descriptor_size:
            raise ValueError("incoming Stage-4 descriptor size differs from prior parts")

        pending = self.parts_dir / f"part-{self.next_part_index:08d}.pending"
        with pending.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        digest = file_sha256(pending)
        path = self.parts_dir / f"part-{self.next_part_index:08d}-{digest}.jsonl"
        _replace_with_retry(pending, path)
        published = read_stage4_dual_router_records(path)
        if [record.sample_id for record in published] != keys:
            raise RuntimeError("published Stage-4 part failed verification")
        entry = self._part_entry(path, published)
        self._part_paths.append(path)
        self._part_entries.append(entry)
        self._records_by_key.update(
            (record.sample_id, record) for record in published
        )
        self._descriptor_size = descriptor_size
        self._write_manifest()
        return path

    def validate_expected_keys(self, expected_keys: Sequence[str]) -> list[str]:
        if not expected_keys:
            raise ValueError("Stage-4 data plan cannot be empty")
        if len(expected_keys) != len(set(expected_keys)):
            raise ValueError("planned Stage-4 record keys are not unique")
        declared_hash = self.provenance["data"]["controller_selection"][
            "selected_record_keys_sha256"
        ]
        if canonical_json_sha256(list(expected_keys)) != declared_hash:
            raise ValueError(
                "planned Stage-4 record keys disagree with provenance selection hash"
            )
        expected = set(expected_keys)
        unexpected = sorted(self.completed_keys - expected)
        if unexpected:
            raise ValueError(
                f"Stage-4 store contains keys outside the current data plan: "
                f"{unexpected[:5]}"
            )
        return [key for key in expected_keys if key not in self.completed_keys]

    def finalize(self, expected_keys: Sequence[str]) -> dict[str, Any]:
        self._ensure_open()
        remaining = self.validate_expected_keys(expected_keys)
        if remaining:
            raise ValueError(
                f"cannot finalize Stage-4 records; {len(remaining)} scenes remain"
            )
        if self._completion is not None:
            records = read_stage4_dual_router_records(self.records_path)
            if [record.sample_id for record in records] != list(expected_keys):
                raise ValueError("completed Stage-4 record order differs from data plan")
            return dict(self._completion)

        for path, expected_entry in zip(self._part_paths, self._part_entries):
            records = read_stage4_dual_router_records(path)
            if self._part_entry(path, records) != expected_entry:
                raise ValueError(f"Stage-4 part changed before finalize: {path.name}")
        ordered = [self._records_by_key[key] for key in expected_keys]
        self._validate_records_against_provenance(
            ordered, expected_keys=expected_keys
        )
        _atomic_records(self.records_path, ordered)
        completion = {
            "schema": STAGE4_RECORD_STORE_SCHEMA,
            "record_schema": STAGE4_RECORD_SCHEMA,
            "identity_sha256": self.provenance["identity_sha256"],
            "record_count": len(ordered),
            "part_count": len(self._part_paths),
            "descriptor_size": self._descriptor_size,
            "parts_manifest_canonical_sha256": canonical_json_sha256(
                self._manifest_value()
            ),
            "records_sha256": file_sha256(self.records_path),
            "records_path": str(self.records_path),
            "complete": True,
            "test_splits_excluded": True,
            "app_pointer_changed": False,
        }
        _atomic_json(self.completion_path, completion)
        self._completion = completion
        return completion


def load_completed_stage4_record_store(
    root: str | Path,
) -> tuple[list[Stage4DualRouterRecord], dict[str, Any], dict[str, Any]]:
    """Reauthenticate a finalized journal and return records plus both proofs."""

    resolved = Path(root).expanduser().resolve()
    provenance_path = resolved / "provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Stage-4 record-store provenance is unreadable") from error
    if not isinstance(provenance, Mapping):
        raise ValueError("Stage-4 record-store provenance must be a mapping")
    with AtomicStage4DualRouterRecordStore(resolved, provenance) as store:
        completion = store.completion
        if completion is None:
            raise ValueError("Stage-4 record store is not finalized")
        records = read_stage4_dual_router_records(store.records_path)
        return records, _json_clone(store.provenance), completion


__all__ = [
    "AtomicStage4DualRouterRecordStore",
    "build_stage4_record_store_provenance",
    "load_completed_stage4_record_store",
    "validate_stage4_record_selection_provenance",
]
