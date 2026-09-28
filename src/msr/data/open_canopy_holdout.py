"""Deterministic, leakage-aware grouping for Open-Canopy holdout plans."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED_ISOLATION_FIELDS = frozenset(
    {
        "region",
        "image_name",
        "imagery_acquisition_date",
        "lidar_acquisition_date",
    }
)


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _normalized_fields(fields: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(field).strip() for field in fields))
    if not normalized:
        raise ValueError("at least one isolation field is required")
    unsupported = sorted(set(normalized) - SUPPORTED_ISOLATION_FIELDS)
    if unsupported:
        raise ValueError(f"unsupported Open-Canopy isolation fields: {unsupported}")
    return normalized


def connected_components(
    records: Sequence[Mapping[str, Any]],
    isolation_fields: Sequence[str],
) -> list[list[str]]:
    """Group samples transitively when they share any protected identity.

    Missing isolation metadata fails closed.  Transitive grouping is important:
    a source mosaic can join two regions, and either region can in turn join a
    second mosaic.  Splitting that connected component would leak information.
    """

    fields = _normalized_fields(isolation_fields)
    sample_ids = [str(record.get("sample_id", "")).strip() for record in records]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError("every Open-Canopy record must have a sample_id")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Open-Canopy holdout input contains duplicate sample IDs")
    groups = _DisjointSet(sample_ids)
    first_by_identity: dict[tuple[str, str], str] = {}
    for record, sample_id in zip(records, sample_ids, strict=True):
        for field in fields:
            raw_value = record.get(field)
            value = "" if raw_value is None else str(raw_value).strip()
            if not value:
                raise ValueError(
                    f"sample {sample_id} is missing required isolation field {field}"
                )
            identity = (field, value)
            first = first_by_identity.setdefault(identity, sample_id)
            groups.union(first, sample_id)

    members: dict[str, list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        members[groups.find(sample_id)].append(sample_id)
    return sorted((sorted(values) for values in members.values()), key=lambda x: x[0])


def split_overlap_audit(
    records: Sequence[Mapping[str, Any]],
    isolation_fields: Sequence[str],
    *,
    split_field: str = "split",
) -> dict[str, Any]:
    """Report protected identities that currently occur in multiple splits."""

    fields = _normalized_fields(isolation_fields)
    split_values: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in fields
    }
    for record in records:
        sample_id = str(record.get("sample_id", "")).strip()
        split = str(record.get(split_field, "")).strip()
        if not sample_id or not split:
            raise ValueError("every audit record needs sample_id and split")
        for field in fields:
            raw_value = record.get(field)
            value = "" if raw_value is None else str(raw_value).strip()
            if not value:
                raise ValueError(
                    f"sample {sample_id} is missing required isolation field {field}"
                )
            split_values[field][split].add(value)

    splits = sorted({str(record[split_field]) for record in records})
    pairs: dict[str, dict[str, list[str]]] = {}
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            key = f"{left}__{right}"
            pairs[key] = {
                field: sorted(
                    split_values[field][left] & split_values[field][right]
                )
                for field in fields
            }
    return {
        "passes": all(
            not overlap
            for pair in pairs.values()
            for overlap in pair.values()
        ),
        "isolation_fields": list(fields),
        "pairs": pairs,
    }


def plan_holdout(
    records: Sequence[Mapping[str, Any]],
    isolation_fields: Sequence[str],
    *,
    holdout_fraction: float,
    seed: str,
) -> dict[str, Any]:
    """Assign complete connected components to learning or held-out roles."""

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between zero and one")
    fields = _normalized_fields(isolation_fields)
    components = connected_components(records, fields)
    if len(components) < 2:
        raise ValueError(
            "requested isolation constraints collapse all samples into one component"
        )

    def order_key(component: list[str]) -> str:
        identity = "|".join(component)
        return hashlib.sha256(f"{seed}|{identity}".encode("utf-8")).hexdigest()

    target = round(len(records) * holdout_fraction)
    held_out: list[list[str]] = []
    learning: list[list[str]] = []
    held_count = 0
    ordered = sorted(components, key=order_key)
    for component in ordered:
        size = len(component)
        take_distance = abs(target - (held_count + size))
        skip_distance = abs(target - held_count)
        if take_distance < skip_distance:
            held_out.append(component)
            held_count += size
        else:
            learning.append(component)
    if not held_out:
        closest = min(
            learning,
            key=lambda values: (abs(target - len(values)), len(values), values[0]),
        )
        learning.remove(closest)
        held_out.append(closest)
    if not learning:
        smallest = min(held_out, key=lambda values: (len(values), values[0]))
        held_out.remove(smallest)
        learning.append(smallest)

    assignments: dict[str, str] = {}
    for component in learning:
        assignments.update({sample_id: "learning" for sample_id in component})
    for component in held_out:
        assignments.update({sample_id: "geographic_holdout" for sample_id in component})

    annotated = [
        {**dict(record), "planned_role": assignments[str(record["sample_id"])]}
        for record in records
    ]
    verification = split_overlap_audit(
        annotated, fields, split_field="planned_role"
    )
    if not verification["passes"]:
        raise AssertionError("internal error: planned holdout is not group-disjoint")
    component_sizes = sorted((len(component) for component in components), reverse=True)
    return {
        "isolation_fields": list(fields),
        "seed": str(seed),
        "requested_holdout_fraction": float(holdout_fraction),
        "sample_count": len(records),
        "component_count": len(components),
        "largest_component_samples": component_sizes[0],
        "learning_samples": sum(role == "learning" for role in assignments.values()),
        "holdout_samples": sum(
            role == "geographic_holdout" for role in assignments.values()
        ),
        "achieved_holdout_fraction": (
            sum(role == "geographic_holdout" for role in assignments.values())
            / len(records)
        ),
        "verification": verification,
        "assignments": [
            {"sample_id": sample_id, "planned_role": assignments[sample_id]}
            for sample_id in sorted(assignments)
        ],
    }
