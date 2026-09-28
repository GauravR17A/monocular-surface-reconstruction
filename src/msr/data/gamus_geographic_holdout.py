"""Fail-closed city-disjoint development holdouts for the GAMUS dataset.

The released GAMUS HDF5 files do not carry coordinates or acquisition IDs.  A
tile name does, however, carry a city prefix.  The strongest geographic split
that can be proven from the local development data is therefore a whole-city
holdout.  This module deliberately does not discover or load the official test
split.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


GAMUS_APPROVED_INDEX_SCHEMA = "msr.gamus.approved_samples.v1"
GAMUS_GEOGRAPHIC_HOLDOUT_SCHEMA = "msr.gamus.geographic_holdout.v1"
SUPPORTED_DEVELOPMENT_SPLITS = ("train", "val")
SUPPORTED_CITIES = frozenset({"DC", "NYC", "PHL"})

_DC_PATTERN = re.compile(r"^(DC)_(\d+)_(\d+)$")
_SCALAR_PATTERN = re.compile(r"^(NYC|PHL)_(\d+)$")


@dataclass(frozen=True)
class GamusTileIdentity:
    """Geographic identity that is provable from one released tile name."""

    sample_id: str
    city: str
    coordinate_kind: str
    row: int | None = None
    column: int | None = None
    scalar_index: int | None = None


def file_sha256(path: str | Path) -> str:
    """Return the content identity of a local artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash JSON data independently from indentation and mapping insertion order."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_gamus_tile_id(sample_id: str) -> GamusTileIdentity:
    """Parse only spatial facts encoded unambiguously in a GAMUS tile ID.

    DC publishes a two-index name that can support grid-neighbour audits.  NYC
    and PHL publish a scalar index; treating that number as a row/column or a
    metric coordinate would be an unsupported assumption, so it remains an
    opaque scalar here.
    """

    normalized = str(sample_id).strip()
    dc_match = _DC_PATTERN.fullmatch(normalized)
    if dc_match:
        return GamusTileIdentity(
            sample_id=normalized,
            city="DC",
            coordinate_kind="released_row_column_index",
            row=int(dc_match.group(2)),
            column=int(dc_match.group(3)),
        )
    scalar_match = _SCALAR_PATTERN.fullmatch(normalized)
    if scalar_match:
        return GamusTileIdentity(
            sample_id=normalized,
            city=scalar_match.group(1),
            coordinate_kind="opaque_released_scalar_index",
            scalar_index=int(scalar_match.group(2)),
        )
    raise ValueError(f"unsupported or malformed GAMUS development tile ID: {sample_id!r}")


def _validated_ids(split_payload: Mapping[str, Any], split: str) -> list[str]:
    values = split_payload.get("approved_sample_ids")
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError(f"GAMUS {split} approved_sample_ids is missing or malformed")
    normalized = [value.strip() for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"GAMUS {split} approved_sample_ids contains duplicates")
    declared_count = split_payload.get("approved_count")
    if declared_count is None or int(declared_count) != len(normalized):
        raise ValueError(
            f"GAMUS {split} approved_count does not match approved_sample_ids"
        )
    for sample_id in normalized:
        parse_gamus_tile_id(sample_id)
    return sorted(normalized)


def load_development_index(path: str | Path) -> dict[str, Any]:
    """Load and validate train/validation identity only.

    The JSON document can contain an official-test inventory, but this function
    never reads that split and never discovers or opens a test HDF5 file.
    """

    index_path = Path(path).expanduser().resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema") != GAMUS_APPROVED_INDEX_SCHEMA:
        raise ValueError(
            f"unsupported GAMUS approved-index schema: {payload.get('schema')!r}"
        )
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("GAMUS approved index has no splits mapping")
    development: dict[str, Any] = {}
    for split in SUPPORTED_DEVELOPMENT_SPLITS:
        split_payload = splits.get(split)
        if not isinstance(split_payload, Mapping):
            raise ValueError(f"GAMUS approved index has no {split!r} split")
        development[split] = {
            **dict(split_payload),
            "approved_sample_ids": _validated_ids(split_payload, split),
        }
    train_ids = set(development["train"]["approved_sample_ids"])
    val_ids = set(development["val"]["approved_sample_ids"])
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise ValueError(f"GAMUS train/validation ID leakage: {overlap[:5]}")
    return {
        "schema": payload["schema"],
        "dataset": payload.get("dataset"),
        "source_path": str(index_path),
        "source_sha256": file_sha256(index_path),
        "source_payload": payload,
        "splits": development,
    }


def build_city_holdout_contract(
    development_index: Mapping[str, Any],
    *,
    holdout_city: str = "NYC",
    seed: str = "msr-gamus-city-holdout-v1",
) -> dict[str, Any]:
    """Create a deterministic whole-city holdout from approved train IDs.

    Selection uses tile IDs and the predeclared city only.  It never consults
    imagery, semantic labels, height labels, cached priors, or model results.
    """

    city = str(holdout_city).strip().upper()
    if city not in SUPPORTED_CITIES:
        raise ValueError(f"unsupported GAMUS holdout city: {holdout_city!r}")
    splits = development_index.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("development index has no validated splits")
    train_payload = splits.get("train")
    val_payload = splits.get("val")
    if not isinstance(train_payload, Mapping) or not isinstance(val_payload, Mapping):
        raise ValueError("development index requires validated train and val splits")
    train_ids = _validated_ids(train_payload, "train")
    val_ids = _validated_ids(val_payload, "val")
    train_identity = {sample_id: parse_gamus_tile_id(sample_id) for sample_id in train_ids}
    val_identity = {sample_id: parse_gamus_tile_id(sample_id) for sample_id in val_ids}

    holdout_ids = sorted(
        sample_id for sample_id, identity in train_identity.items() if identity.city == city
    )
    learning_ids = sorted(set(train_ids) - set(holdout_ids))
    if not holdout_ids:
        raise ValueError(f"holdout city {city} has no approved training tiles")
    if not learning_ids:
        raise ValueError("whole-city holdout would leave no learning tiles")
    val_holdout_city_ids = sorted(
        sample_id for sample_id, identity in val_identity.items() if identity.city == city
    )
    if val_holdout_city_ids:
        raise ValueError(
            f"holdout city {city} also occurs in development validation; using it for "
            "checkpoint selection would contaminate the geographic holdout"
        )

    learning_cities = sorted({train_identity[sample_id].city for sample_id in learning_ids})
    validation_cities = sorted({identity.city for identity in val_identity.values()})
    if len(learning_cities) < 2:
        raise ValueError("city holdout must retain at least two learning cities")
    if not set(validation_cities).issubset(learning_cities):
        raise ValueError(
            "development validation contains a city absent from the learning partition"
        )

    # A whole city is excluded, so there is no same-city boundary on which a
    # tile buffer could be computed.  An empty buffer is intentional, not a
    # missing value.  If a future within-city plan is used, this contract must
    # be replaced and a coordinate-backed buffer becomes mandatory.
    buffer_ids: list[str] = []
    holdout_set = set(holdout_ids)
    assignments = [
        {
            "sample_id": sample_id,
            "source_split": "train",
            "city": train_identity[sample_id].city,
            "role": "geographic_holdout" if sample_id in holdout_set else "learning",
        }
        for sample_id in train_ids
    ]
    assignments.extend(
        {
            "sample_id": sample_id,
            "source_split": "val",
            "city": val_identity[sample_id].city,
            "role": "development_validation",
        }
        for sample_id in val_ids
    )

    counts_by_role = {
        role: sum(row["role"] == role for row in assignments)
        for role in ("learning", "buffer", "development_validation", "geographic_holdout")
    }
    counts_by_city_and_role: dict[str, dict[str, int]] = {}
    for row in assignments:
        counts_by_city_and_role.setdefault(row["city"], {}).setdefault(row["role"], 0)
        counts_by_city_and_role[row["city"]][row["role"]] += 1

    assigned_train_ids = [
        row["sample_id"] for row in assignments if row["source_split"] == "train"
    ]
    assigned_val_ids = [
        row["sample_id"] for row in assignments if row["source_split"] == "val"
    ]
    contract: dict[str, Any] = {
        "schema": GAMUS_GEOGRAPHIC_HOLDOUT_SCHEMA,
        "contract_id": "gamus-development-city-nyc-v1" if city == "NYC" else f"gamus-development-city-{city.lower()}-v1",
        "seed": str(seed),
        "source_approved_index": str(development_index.get("source_path", "")),
        "source_approved_index_sha256": str(development_index.get("source_sha256", "")),
        "selection": {
            "strategy": "whole_city_prefix",
            "holdout_city": city,
            "selection_uses_labels_or_pixels": False,
            "selection_basis": "predeclared city prefix encoded in approved official-train tile IDs",
            "buffer_policy": "none_required_for_whole_city_separation",
            "buffer_tiles": buffer_ids,
        },
        "counts": {
            "approved_train_total": len(train_ids),
            "approved_validation_total": len(val_ids),
            "by_role": counts_by_role,
            "by_city_and_role": counts_by_city_and_role,
        },
        "city_sets": {
            "learning": learning_cities,
            "development_validation": validation_cities,
            "geographic_holdout": [city],
        },
        "official_test_policy": {
            "allowed": False,
            "labels_opened_by_planner": False,
            "files_discovered_by_planner": False,
            "ids_used_for_selection": False,
        },
        "use_policy": {
            "learning": "model fitting and learning-only statistics",
            "development_validation": "epoch choice, tuning, thresholds and error analysis",
            "geographic_holdout": "one evaluation after model, thresholds and guards are frozen",
            "forbidden_for_learning": ["buffer", "development_validation", "geographic_holdout"],
            "existing_full-train_candidates_are_eligible": False,
        },
        "verification": {
            "all_approved_train_ids_partitioned_once": sorted(assigned_train_ids)
            == train_ids
            and len(assigned_train_ids) == len(set(assigned_train_ids)),
            "all_approved_validation_ids_partitioned_once": sorted(assigned_val_ids)
            == val_ids
            and len(assigned_val_ids) == len(set(assigned_val_ids)),
            "learning_holdout_disjoint": not (set(learning_ids) & set(holdout_ids)),
            "learning_buffer_disjoint": not (set(learning_ids) & set(buffer_ids)),
            "holdout_buffer_disjoint": not (set(holdout_ids) & set(buffer_ids)),
            "holdout_city_absent_from_learning": city not in learning_cities,
            "holdout_city_absent_from_development_validation": city not in validation_cities,
            "development_validation_cities_seen_in_learning": set(validation_cities).issubset(
                learning_cities
            ),
        },
        "assignments": sorted(
            assignments,
            key=lambda row: (row["source_split"], row["sample_id"]),
        ),
        "limitations": [
            "The local GAMUS HDF5 files provide no per-tile CRS, transform, coordinates, or acquisition identity; city prefix is the strongest locally provable geographic grouping.",
            "NYC and PHL scalar tile numbers are opaque release indices and are not treated as metric coordinates or row/column locations.",
            "This is a development holdout from the approved official training split, not the untouched official GAMUS test benchmark.",
            "Previously trained height-focused and six-class checkpoints already saw all approved official-train tiles, including NYC, so they cannot claim this holdout as unseen.",
            "One held-out US city does not prove transfer to a new country, sensor, season, resolution, or terrain distribution.",
        ],
    }
    if not all(contract["verification"].values()):
        raise AssertionError("internal error: GAMUS geographic holdout verification failed")
    contract["assignment_sha256"] = canonical_sha256(contract["assignments"])
    return contract


def role_sample_ids(contract: Mapping[str, Any], role: str) -> tuple[str, ...]:
    """Return sorted IDs for one role after validating the contract schema."""

    rows = validate_holdout_contract(contract)
    ids = [
        str(row.get("sample_id", "")).strip()
        for row in rows
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    if any(not sample_id for sample_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"malformed or duplicate {role!r} assignments")
    return tuple(sorted(ids))


def validate_holdout_contract(
    contract: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Validate a sealed contract before it controls a training inventory."""

    if contract.get("schema") != GAMUS_GEOGRAPHIC_HOLDOUT_SCHEMA:
        raise ValueError("unsupported GAMUS geographic holdout contract schema")
    rows = contract.get("assignments")
    if not isinstance(rows, list):
        raise ValueError("GAMUS geographic holdout contract has no assignments")
    allowed_roles = {
        "learning",
        "buffer",
        "development_validation",
        "geographic_holdout",
    }
    normalized_rows: list[Mapping[str, Any]] = []
    identities: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("GAMUS geographic holdout assignment is not a mapping")
        sample_id = str(row.get("sample_id", "")).strip()
        source_split = str(row.get("source_split", "")).strip()
        city = str(row.get("city", "")).strip()
        role = str(row.get("role", "")).strip()
        if not sample_id or not source_split or not city or role not in allowed_roles:
            raise ValueError("GAMUS geographic holdout assignment is malformed")
        parsed = parse_gamus_tile_id(sample_id)
        if parsed.city != city:
            raise ValueError(f"assignment city mismatch for {sample_id}")
        expected_source_split = (
            "val" if role == "development_validation" else "train"
        )
        if source_split != expected_source_split:
            raise ValueError(
                f"assignment source split mismatch for {sample_id}: "
                f"role {role!r} requires {expected_source_split!r}"
            )
        identities.append((source_split, sample_id))
        normalized_rows.append(row)
    if len(identities) != len(set(identities)):
        raise ValueError("GAMUS geographic holdout assignments contain duplicates")
    expected_hash = str(contract.get("assignment_sha256", ""))
    actual_hash = canonical_sha256(rows)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError("GAMUS geographic holdout assignment SHA-256 mismatch")
    declared_counts = contract.get("counts", {}).get("by_role", {})
    if not isinstance(declared_counts, Mapping):
        raise ValueError("GAMUS geographic holdout contract has no role counts")
    for role in allowed_roles:
        found = sum(row["role"] == role for row in rows)
        if int(declared_counts.get(role, -1)) != found:
            raise ValueError(f"GAMUS geographic holdout {role} count mismatch")
    verification = contract.get("verification")
    if not isinstance(verification, Mapping) or not verification or not all(
        value is True for value in verification.values()
    ):
        raise ValueError("GAMUS geographic holdout verification is incomplete or failed")
    official_test_policy = contract.get("official_test_policy")
    if not isinstance(official_test_policy, Mapping) or official_test_policy.get("allowed") is not False:
        raise ValueError("GAMUS geographic holdout must forbid the official test split")
    selection = contract.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("GAMUS geographic holdout has no selection policy")
    holdout_city = str(selection.get("holdout_city", "")).strip()
    if holdout_city not in SUPPORTED_CITIES:
        raise ValueError("GAMUS geographic holdout has an invalid holdout city")
    if selection.get("selection_uses_labels_or_pixels") is not False:
        raise ValueError("GAMUS geographic holdout selection must be label-blind")
    holdout_ids = {
        str(row["sample_id"])
        for row in rows
        if row["role"] == "geographic_holdout"
    }
    if not holdout_ids or any(
        parse_gamus_tile_id(sample_id).city != holdout_city
        for sample_id in holdout_ids
    ):
        raise ValueError("GAMUS geographic holdout city membership is inconsistent")
    learning_cities = {
        str(row["city"]) for row in rows if row["role"] == "learning"
    }
    validation_cities = {
        str(row["city"])
        for row in rows
        if row["role"] == "development_validation"
    }
    if holdout_city in learning_cities or holdout_city in validation_cities:
        raise ValueError("GAMUS geographic holdout city leaks into development")
    if not validation_cities.issubset(learning_cities):
        raise ValueError("GAMUS validation contains a city unseen during learning")
    return normalized_rows


def load_sealed_holdout_contract(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load a holdout contract only when its exact persisted bytes are trusted."""

    contract_path = Path(path).expanduser().resolve()
    expected = str(expected_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected holdout-contract SHA-256 must contain 64 hex digits")
    if not contract_path.is_file():
        raise FileNotFoundError(f"GAMUS holdout contract does not exist: {contract_path}")
    actual = file_sha256(contract_path)
    if actual != expected:
        raise ValueError(
            f"GAMUS holdout-contract SHA-256 mismatch: expected {expected}, found {actual}"
        )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("GAMUS geographic holdout contract must be a JSON object")
    validate_holdout_contract(payload)
    return payload


def assert_training_ids_respect_contract(
    training_ids: Iterable[str],
    contract: Mapping[str, Any],
    *,
    require_complete_learning_partition: bool = True,
) -> None:
    """Fail when a training inventory touches any protected role."""

    values = [str(value).strip() for value in training_ids]
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError("training inventory contains blank or duplicate sample IDs")
    actual = set(values)
    learning = set(role_sample_ids(contract, "learning"))
    forbidden = set().union(
        *(set(role_sample_ids(contract, role)) for role in ("buffer", "development_validation", "geographic_holdout"))
    )
    leaked = sorted(actual & forbidden)
    if leaked:
        raise ValueError(f"training inventory leaks protected GAMUS IDs: {leaked[:5]}")
    unknown = sorted(actual - learning)
    if unknown:
        raise ValueError(f"training inventory contains IDs outside the learning role: {unknown[:5]}")
    missing = sorted(learning - actual)
    if require_complete_learning_partition and missing:
        raise ValueError(f"training inventory omits approved learning IDs: {missing[:5]}")


def build_learning_approved_index(
    development_index: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a loader-compatible train/val index without an official-test split."""

    source_payload = development_index.get("source_payload")
    if not isinstance(source_payload, Mapping):
        raise ValueError("development index does not preserve its source payload")
    if contract.get("source_approved_index_sha256") != development_index.get("source_sha256"):
        raise ValueError("holdout contract belongs to a different approved index")
    learning = set(role_sample_ids(contract, "learning"))
    train_source = development_index["splits"]["train"]
    val_source = development_index["splits"]["val"]

    def filter_train_list(name: str) -> list[str]:
        values = train_source.get(name)
        if values is None:
            return []
        if not isinstance(values, list):
            raise ValueError(f"source train field {name!r} must be a list")
        return sorted(value for value in values if value in learning)

    train = deepcopy(dict(train_source))
    train["approved_sample_ids"] = sorted(learning)
    train["approved_count"] = len(learning)
    for field in ("semantic_eligible_sample_ids", "height_regression_eligible_sample_ids"):
        if field in train_source:
            train[field] = filter_train_list(field)

    val = deepcopy(dict(val_source))
    output = {
        key: deepcopy(value)
        for key, value in source_payload.items()
        if key not in {"splits"}
    }
    output["derivation"] = {
        "schema": GAMUS_GEOGRAPHIC_HOLDOUT_SCHEMA,
        "contract_id": contract.get("contract_id"),
        "contract_assignment_sha256": contract.get("assignment_sha256"),
        "source_approved_index_sha256": development_index.get("source_sha256"),
        "official_test_split_copied": False,
    }
    output["splits"] = {"train": train, "val": val}
    assert_training_ids_respect_contract(train["approved_sample_ids"], contract)
    return output


def build_holdout_approved_index(
    development_index: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a loader-compatible index containing only locked holdout tiles.

    The held-out NYC tiles originate in GAMUS's official *training* directory,
    so the output intentionally exposes them under the loader's ``train`` key.
    It never copies or mentions the official GAMUS test inventory.
    """

    source_payload = development_index.get("source_payload")
    if not isinstance(source_payload, Mapping):
        raise ValueError("development index does not preserve its source payload")
    if contract.get("source_approved_index_sha256") != development_index.get("source_sha256"):
        raise ValueError("holdout contract belongs to a different approved index")
    holdout = set(role_sample_ids(contract, "geographic_holdout"))
    if not holdout:
        raise ValueError("geographic holdout contains no samples")
    rows = validate_holdout_contract(contract)
    source_by_id = {
        str(row["sample_id"]): str(row["source_split"])
        for row in rows
        if row["role"] == "geographic_holdout"
    }
    if set(source_by_id) != holdout or set(source_by_id.values()) != {"train"}:
        raise ValueError("geographic holdout must come only from the approved train split")

    train_source = development_index["splits"]["train"]

    def filter_train_list(name: str) -> list[str]:
        values = train_source.get(name)
        if values is None:
            return []
        if not isinstance(values, list):
            raise ValueError(f"source train field {name!r} must be a list")
        return sorted(value for value in values if value in holdout)

    train = deepcopy(dict(train_source))
    train["approved_sample_ids"] = sorted(holdout)
    train["approved_count"] = len(holdout)
    for field in ("semantic_eligible_sample_ids", "height_regression_eligible_sample_ids"):
        if field in train_source:
            train[field] = filter_train_list(field)
    if set(train.get("semantic_eligible_sample_ids", train["approved_sample_ids"])) != holdout:
        raise ValueError("not every geographic holdout tile is semantic-eligible")

    output = {
        key: deepcopy(value)
        for key, value in source_payload.items()
        if key != "splits"
    }
    output["derivation"] = {
        "schema": GAMUS_GEOGRAPHIC_HOLDOUT_SCHEMA,
        "contract_id": contract.get("contract_id"),
        "contract_assignment_sha256": contract.get("assignment_sha256"),
        "source_approved_index_sha256": development_index.get("source_sha256"),
        "role": "geographic_holdout",
        "official_test_split_copied": False,
    }
    output["splits"] = {"train": train}
    if set(train["approved_sample_ids"]) != holdout:
        raise AssertionError("internal error: holdout approved-index coverage mismatch")
    return output


__all__ = [
    "GAMUS_GEOGRAPHIC_HOLDOUT_SCHEMA",
    "GamusTileIdentity",
    "assert_training_ids_respect_contract",
    "build_city_holdout_contract",
    "build_holdout_approved_index",
    "build_learning_approved_index",
    "canonical_sha256",
    "file_sha256",
    "load_development_index",
    "load_sealed_holdout_contract",
    "parse_gamus_tile_id",
    "role_sample_ids",
    "validate_holdout_contract",
]
