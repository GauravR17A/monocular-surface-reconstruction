"""Crash-safe, provenance-locked storage for Stage-3 router records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from .scene_router import SceneRouterRecord, read_scene_router_records


RECORD_STORE_SCHEMA = "msr.stage3_scene_router_records.v1"
_PART_PATTERN = re.compile(r"^part-(\d{8})-([0-9a-f]{64})\.jsonl$")
_ATOMIC_JSON_REPLACE_ATTEMPTS = 8
_ATOMIC_JSON_REPLACE_INITIAL_DELAY_SECONDS = 0.025
_ATOMIC_JSON_REPLACE_MAX_DELAY_SECONDS = 0.2


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_record_selection_provenance(
    data: object, *, completed_record_count: object | None = None
) -> None:
    """Validate planned, explicitly excluded, and scored record evidence."""

    if not isinstance(data, Mapping):
        raise ValueError("record data provenance must be a mapping")
    required = {
        "planned_record_count",
        "excluded_no_reference_count",
        "excluded_no_reference_record_keys",
        "scored_record_count",
        "planned_partition_counts",
        "scored_partition_counts",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"record selection provenance is missing fields: {missing}")

    def count(name: str, *, allow_zero: bool) -> int:
        value = data.get(name)
        minimum = 0 if allow_zero else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"record provenance {name} must be a {qualifier} integer")
        return value

    planned = count("planned_record_count", allow_zero=False)
    excluded_count = count("excluded_no_reference_count", allow_zero=True)
    scored = count("scored_record_count", allow_zero=False)
    keys = data.get("excluded_no_reference_record_keys")
    if not isinstance(keys, list) or not keys or any(
        not isinstance(key, str) for key in keys
    ):
        raise ValueError(
            "record provenance excluded_no_reference_record_keys must be a "
            "non-empty string list"
        )
    if len(keys) != len(set(keys)):
        raise ValueError(
            "record provenance excluded_no_reference_record_keys contains duplicates"
        )
    for key in keys:
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
                "record provenance no-reference exclusions must be canonical "
                f"train keys: {key!r}"
            )
    if excluded_count != len(keys) or planned != excluded_count + scored:
        raise ValueError("record provenance planned/excluded/scored counts disagree")

    partition_counts: dict[str, dict[str, int]] = {}
    for field in ("planned_partition_counts", "scored_partition_counts"):
        value = data.get(field)
        if not isinstance(value, Mapping) or set(value) != {"train", "calibration"}:
            raise ValueError(
                f"record provenance {field} must contain train and calibration"
            )
        parsed: dict[str, int] = {}
        for partition in ("train", "calibration"):
            item = value[partition]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(
                    f"record provenance {field}.{partition} must be non-negative"
                )
            parsed[partition] = item
        partition_counts[field] = parsed
    planned_partitions = partition_counts["planned_partition_counts"]
    scored_partitions = partition_counts["scored_partition_counts"]
    if sum(planned_partitions.values()) != planned or sum(
        scored_partitions.values()
    ) != scored:
        raise ValueError("record provenance partition counts disagree with totals")
    if (
        planned_partitions["train"] - excluded_count != scored_partitions["train"]
        or planned_partitions["calibration"] != scored_partitions["calibration"]
    ):
        raise ValueError(
            "record provenance exclusions must affect only the train partition"
        )
    if completed_record_count is not None:
        if isinstance(completed_record_count, bool) or not isinstance(
            completed_record_count, int
        ):
            raise ValueError("record completion count must be an integer")
        if completed_record_count != scored:
            raise ValueError(
                "record completion count differs from provenance scored_record_count"
            )


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    delay = _ATOMIC_JSON_REPLACE_INITIAL_DELAY_SECONDS
    for attempt in range(_ATOMIC_JSON_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == _ATOMIC_JSON_REPLACE_ATTEMPTS - 1:
                # Keep the fully flushed temporary file and the prior destination
                # untouched, matching the existing fail-closed crash semantics.
                raise
            time.sleep(delay)
            delay = min(delay * 2, _ATOMIC_JSON_REPLACE_MAX_DELAY_SECONDS)


def _atomic_records(path: Path, records: Sequence[SceneRouterRecord]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_record_store_provenance(
    *,
    generation_config: Mapping[str, Any],
    endpoints: Mapping[str, Any],
    shared_model: Mapping[str, Any],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical, immutable record-store identity."""

    config = json.loads(json.dumps(generation_config, sort_keys=True))
    endpoint_value = json.loads(json.dumps(endpoints, sort_keys=True))
    shared_value = json.loads(json.dumps(shared_model, sort_keys=True))
    data_value = json.loads(json.dumps(data, sort_keys=True))
    identity = {
        "schema": RECORD_STORE_SCHEMA,
        "generation_config": config,
        "endpoints": endpoint_value,
        "shared_model": shared_value,
        "data": data_value,
        "target_rule": (
            "candidate iff held-out fallback utility minus candidate utility "
            "exceeds minimum_candidate_gain; source/split/landscape are metadata only"
        ),
        "test_splits_excluded": True,
    }
    identity["config_sha256"] = canonical_json_sha256(
        {
            "generation_config": config,
            "endpoints": endpoint_value,
            "shared_model": shared_value,
            "data": data_value,
        }
    )
    return identity


class AtomicSceneRouterRecordStore:
    """Immutable-part journal that resumes only under identical provenance.

    Each appended batch is written to a temporary file, flushed, and atomically
    renamed into ``parts``.  A crash can therefore lose at most the unfinished
    temporary batch; it cannot leave a valid-looking partial JSON line.  The
    consolidated ``records.jsonl`` is published atomically only after every
    planned key is present exactly once.
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
        supplied = json.loads(json.dumps(provenance, sort_keys=True))
        if supplied.get("schema") != RECORD_STORE_SCHEMA:
            raise ValueError(
                f"router record provenance schema must be {RECORD_STORE_SCHEMA!r}"
            )
        config_hash = supplied.get("config_sha256")
        if not isinstance(config_hash, str) or len(config_hash) != 64:
            raise ValueError("router record provenance has no valid config_sha256")
        recomputed_config_hash = canonical_json_sha256(
            {
                "generation_config": supplied.get("generation_config"),
                "endpoints": supplied.get("endpoints"),
                "shared_model": supplied.get("shared_model"),
                "data": supplied.get("data"),
            }
        )
        if config_hash != recomputed_config_hash:
            raise ValueError(
                "router record provenance config_sha256 does not match its contents"
            )

        self.root.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        try:
            existing_parts = list((self.root / "parts").glob("part-*.jsonl"))
            if self.provenance_path.exists():
                existing = json.loads(self.provenance_path.read_text(encoding="utf-8"))
                if canonical_json_sha256(existing) != canonical_json_sha256(supplied):
                    raise ValueError(
                        "router record-store provenance/config mismatch; refusing to "
                        "mix endpoints or data"
                    )
            else:
                if existing_parts or self.records_path.exists():
                    raise ValueError(
                        "router record data exists without provenance; refusing unsafe resume"
                    )
                _atomic_json(self.provenance_path, supplied)
            self.provenance = supplied
            self.parts_dir.mkdir(parents=True, exist_ok=True)
            self._records_by_key: dict[str, SceneRouterRecord] = {}
            self._part_entries: list[dict[str, Any]] = []
            self._descriptor_size: int | None = None
            self._completion: dict[str, Any] | None = None
            self._part_paths = self._load_parts()
            if self.completion_path.exists():
                self._completion = self._validate_existing_completion()
        except BaseException:
            self.close()
            raise

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
                f"router record store is already locked by another process: {self.root}"
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
            else:  # pragma: no cover - Windows is the production host.
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._lock_handle = None

    def __enter__(self) -> "AtomicSceneRouterRecordStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - explicit/context close is preferred.
        try:
            self.close()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        if self._lock_handle is None:
            raise RuntimeError("router record store is closed and no longer locked")

    def _parts_manifest_value(self) -> dict[str, Any]:
        return {
            "schema": RECORD_STORE_SCHEMA,
            "config_sha256": self.provenance["config_sha256"],
            "parts": self._part_entries,
        }

    def _part_entry(
        self, path: Path, records: Sequence[SceneRouterRecord]
    ) -> dict[str, Any]:
        keys = [record.sample_id for record in records]
        descriptor_sizes = {int(record.descriptor.numel()) for record in records}
        if len(descriptor_sizes) != 1:
            raise ValueError("router record part mixes descriptor sizes")
        descriptor_size = descriptor_sizes.pop()
        match = _PART_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected router part filename: {path.name}")
        index = int(match.group(1))
        expected_hash = match.group(2)
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"router record part SHA-256 mismatch: {path.name}"
            )
        return {
            "index": index,
            "filename": path.name,
            "sha256": actual_hash,
            "record_count": len(records),
            "descriptor_size": descriptor_size,
            "keys_sha256": canonical_json_sha256(keys),
            "first_key": keys[0],
            "last_key": keys[-1],
        }

    def _write_parts_manifest(self) -> None:
        self._ensure_open()
        _atomic_json(self.parts_manifest_path, self._parts_manifest_value())

    def _load_parts(self) -> list[Path]:
        paths = sorted(self.parts_dir.glob("part-*.jsonl"))
        parsed = [_PART_PATTERN.fullmatch(path.name) for path in paths]
        if any(match is None for match in parsed):
            raise ValueError(
                "router record part sequence contains an unexpected filename"
            )
        indices = [int(match.group(1)) for match in parsed if match is not None]
        if indices != list(range(len(paths))):
            raise ValueError("router record part sequence contains a gap")
        if self.parts_manifest_path.exists():
            manifest = json.loads(
                self.parts_manifest_path.read_text(encoding="utf-8")
            )
            if (
                manifest.get("schema") != RECORD_STORE_SCHEMA
                or manifest.get("config_sha256")
                != self.provenance["config_sha256"]
                or not isinstance(manifest.get("parts"), list)
            ):
                raise ValueError("router parts manifest identity is invalid")
            recorded_entries = manifest["parts"]
            if len(recorded_entries) > len(paths):
                raise ValueError("router parts manifest references a missing part")
        else:
            if paths:
                # Hashed filenames permit safe recovery of a manifest lost
                # before its first atomic publication.
                recorded_entries = []
            else:
                recorded_entries = []
        orphan_tail_count = len(paths) - len(recorded_entries)
        if orphan_tail_count > 1:
            raise ValueError(
                "router record store has more than one unledgered part; "
                "this cannot result from one atomic append"
            )
        for path in paths:
            records = read_scene_router_records(path)
            entry = self._part_entry(path, records)
            descriptor_size = int(entry["descriptor_size"])
            if self._descriptor_size is None:
                self._descriptor_size = descriptor_size
            elif descriptor_size != self._descriptor_size:
                raise ValueError(
                    "router record parts contain inconsistent descriptor sizes"
                )
            if entry["index"] < len(recorded_entries):
                if recorded_entries[entry["index"]] != entry:
                    raise ValueError(
                        f"router parts manifest mismatch for {path.name}"
                    )
            self._part_entries.append(entry)
            for record in records:
                if record.sample_id in self._records_by_key:
                    raise ValueError(
                        f"duplicate router record key on resume: {record.sample_id}"
                    )
                self._records_by_key[record.sample_id] = record
        if recorded_entries != self._part_entries:
            # Recover only a contiguous, content-hashed tail that was renamed
            # just before a crash prevented the ledger update.
            if recorded_entries != self._part_entries[: len(recorded_entries)]:
                raise ValueError("router parts manifest is not a valid prefix")
            self._write_parts_manifest()
        elif not self.parts_manifest_path.exists():
            self._write_parts_manifest()
        return paths

    def _validate_existing_completion(self) -> dict[str, Any]:
        """Anchor a completed store to its published integrity proof."""

        try:
            completion = json.loads(
                self.completion_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("router completion proof is unreadable") from error
        expected_manifest_hash = canonical_json_sha256(
            self._parts_manifest_value()
        )
        checks = (
            completion.get("schema") == RECORD_STORE_SCHEMA,
            completion.get("config_sha256") == self.provenance["config_sha256"],
            completion.get("complete") is True,
            completion.get("record_count") == len(self._records_by_key),
            completion.get("part_count") == len(self._part_paths),
            completion.get("descriptor_size") == self._descriptor_size,
            completion.get("parts_manifest_canonical_sha256")
            == expected_manifest_hash,
            completion.get("records_path") == str(self.records_path),
        )
        if not all(checks) or not self.records_path.is_file():
            raise ValueError("router completion proof does not match the record store")
        if completion.get("records_sha256") != file_sha256(self.records_path):
            raise ValueError("completed router records SHA-256 mismatch")
        consolidated = read_scene_router_records(self.records_path)
        keys = [record.sample_id for record in consolidated]
        if len(keys) != len(set(keys)) or set(keys) != set(self._records_by_key):
            raise ValueError("completed router records do not match immutable parts")
        return completion

    @property
    def completed_keys(self) -> frozenset[str]:
        return frozenset(self._records_by_key)

    @property
    def record_count(self) -> int:
        return len(self._records_by_key)

    @property
    def next_part_index(self) -> int:
        return len(self._part_paths)

    def append(self, records: Sequence[SceneRouterRecord]) -> Path:
        self._ensure_open()
        if self._completion is not None:
            raise ValueError("cannot append to an already completed router record store")
        if not records:
            raise ValueError("cannot append an empty router record batch")
        keys = [record.sample_id for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError("incoming router record batch contains duplicate keys")
        duplicate = sorted(set(keys) & self.completed_keys)
        if duplicate:
            raise ValueError(
                f"router record keys already exist; resume must skip them: {duplicate[:5]}"
            )
        descriptor_sizes = {int(record.descriptor.numel()) for record in records}
        if len(descriptor_sizes) != 1:
            raise ValueError("incoming router record batch mixes descriptor sizes")
        descriptor_size = descriptor_sizes.pop()
        if (
            self._descriptor_size is not None
            and descriptor_size != self._descriptor_size
        ):
            raise ValueError(
                "incoming router record descriptor size differs from existing parts"
            )
        pending = self.parts_dir / f"part-{self.next_part_index:08d}.pending"
        # A stale pending file is never considered completed; this write
        # replaces only that same, narrowly scoped next-part scratch file.
        with pending.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record.to_json_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        digest = file_sha256(pending)
        path = self.parts_dir / (
            f"part-{self.next_part_index:08d}-{digest}.jsonl"
        )
        os.replace(pending, path)
        # Re-read the published part before accepting it into resume state.
        published = read_scene_router_records(path)
        if [record.sample_id for record in published] != keys:
            raise RuntimeError("atomically published router part failed verification")
        entry = self._part_entry(path, published)
        self._part_paths.append(path)
        self._part_entries.append(entry)
        self._records_by_key.update(
            (record.sample_id, record) for record in published
        )
        self._descriptor_size = descriptor_size
        self._write_parts_manifest()
        return path

    def validate_expected_keys(self, expected_keys: Sequence[str]) -> list[str]:
        if len(expected_keys) != len(set(expected_keys)):
            raise ValueError("planned router record keys are not unique")
        expected = set(expected_keys)
        unexpected = sorted(self.completed_keys - expected)
        if unexpected:
            raise ValueError(
                f"record store contains keys outside the current data plan: {unexpected[:5]}"
            )
        return [key for key in expected_keys if key not in self.completed_keys]

    def finalize(self, expected_keys: Sequence[str]) -> dict[str, Any]:
        self._ensure_open()
        remaining = self.validate_expected_keys(expected_keys)
        if remaining:
            raise ValueError(
                f"cannot finalize router records; {len(remaining)} planned scenes remain"
            )
        if self._completion is not None:
            consolidated = read_scene_router_records(self.records_path)
            if [record.sample_id for record in consolidated] != list(expected_keys):
                raise ValueError(
                    "completed router record order differs from the current plan"
                )
            return dict(self._completion)
        # Reauthenticate all immutable parts immediately before publishing the
        # consolidated file and its completion proof.
        for path, expected_entry in zip(self._part_paths, self._part_entries):
            records = read_scene_router_records(path)
            if self._part_entry(path, records) != expected_entry:
                raise ValueError(f"router record part changed before finalize: {path.name}")
        ordered_records = [self._records_by_key[key] for key in expected_keys]
        _atomic_records(self.records_path, ordered_records)
        records_sha256 = file_sha256(self.records_path)
        completion = {
            "schema": RECORD_STORE_SCHEMA,
            "config_sha256": self.provenance["config_sha256"],
            "record_count": len(ordered_records),
            "part_count": len(self._part_paths),
            "descriptor_size": self._descriptor_size,
            "parts_manifest_canonical_sha256": canonical_json_sha256(
                self._parts_manifest_value()
            ),
            "records_sha256": records_sha256,
            "records_path": str(self.records_path),
            "complete": True,
        }
        _atomic_json(self.completion_path, completion)
        self._completion = completion
        return completion


__all__ = [
    "AtomicSceneRouterRecordStore",
    "RECORD_STORE_SCHEMA",
    "build_record_store_provenance",
    "canonical_json_sha256",
    "file_sha256",
    "validate_record_selection_provenance",
]
