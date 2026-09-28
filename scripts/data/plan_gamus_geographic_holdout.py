"""Seal a deterministic, city-disjoint GAMUS development holdout.

Only the approved ``train`` and ``val`` inventories and their RGB container
metadata are inspected.  The official GAMUS test directory and test labels are
never discovered or opened by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from msr.data.gamus_geographic_holdout import (
    assert_training_ids_respect_contract,
    build_city_holdout_contract,
    build_learning_approved_index,
    canonical_sha256,
    file_sha256,
    load_development_index,
    parse_gamus_tile_id,
    role_sample_ids,
)


REPORT_SCHEMA = "msr.gamus.geographic_holdout_audit.v1"
ARTIFACT_INDEX_SCHEMA = "msr.gamus.geographic_holdout_artifacts.v1"
_GEOSPATIAL_ATTR_TOKENS = (
    "crs",
    "epsg",
    "geo_transform",
    "geotransform",
    "projection",
    "latitude",
    "longitude",
    "pixel_size",
    "resolution",
    "spatial_reference",
    "transform",
    "x_origin",
    "y_origin",
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _immutable_write(path: Path, content: bytes) -> None:
    """Write a versioned artifact once; a differing rerun fails closed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"refusing to overwrite differing sealed holdout artifact: {path}"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _csv_bytes(rows: list[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot write an empty GAMUS holdout assignment table")
    output = io.StringIO(newline="")
    fieldnames = ["sample_id", "source_split", "city", "role"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows({name: row[name] for name in fieldnames} for row in rows)
    return output.getvalue().encode("utf-8")


def _text_ids(ids: Iterable[str]) -> bytes:
    return ("\n".join(sorted(ids)) + "\n").encode("utf-8")


def _tile_id_audit(development_index: Mapping[str, Any]) -> dict[str, Any]:
    by_split_city: dict[str, dict[str, int]] = {}
    coordinate_kinds: Counter[str] = Counter()
    dc_positions: dict[str, set[tuple[int, int]]] = defaultdict(set)
    scalar_ranges: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for split in ("train", "val"):
        ids = development_index["splits"][split]["approved_sample_ids"]
        identities = [parse_gamus_tile_id(sample_id) for sample_id in ids]
        by_split_city[split] = dict(sorted(Counter(item.city for item in identities).items()))
        coordinate_kinds.update(item.coordinate_kind for item in identities)
        for identity in identities:
            if identity.city == "DC":
                if identity.row is None or identity.column is None:
                    raise AssertionError("DC identity unexpectedly lacks row/column")
                dc_positions[split].add((identity.row, identity.column))
        for city in ("NYC", "PHL"):
            values = [
                item.scalar_index
                for item in identities
                if item.city == city and item.scalar_index is not None
            ]
            if values:
                scalar_ranges[split][city] = {
                    "count": len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }

    train_dc = dc_positions.get("train", set())
    val_dc = dc_positions.get("val", set())

    def touches_training(position: tuple[int, int], *, diagonal: bool) -> bool:
        row, column = position
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        return any((row + dr, column + dc) in train_dc for dr, dc in offsets)

    four_neighbour = sum(touches_training(position, diagonal=False) for position in val_dc)
    eight_neighbour = sum(touches_training(position, diagonal=True) for position in val_dc)
    return {
        "approved_counts_by_split_and_city": by_split_city,
        "coordinate_kinds": dict(sorted(coordinate_kinds.items())),
        "dc_released_grid_adjacency": {
            "validation_tiles": len(val_dc),
            "validation_tiles_touching_train_4_neighbour": four_neighbour,
            "validation_tiles_touching_train_8_neighbour": eight_neighbour,
            "metric_distance_claimed": False,
        },
        "opaque_scalar_index_ranges": dict(scalar_ranges),
        "interpretation": {
            "DC": "released row/column indices support neighbourhood checks but have no embedded CRS or metric transform",
            "NYC_PHL": "released scalar indices are treated as opaque and never converted to coordinates",
        },
    }


def _rgb_path(dataset_root: Path, split: str, sample_id: str) -> Path:
    suffix = "IMG" if sample_id.startswith("NYC_") else "RGB"
    return dataset_root / "images" / split / f"{sample_id}_{suffix}.h5"


def _attribute_names(attributes: Any) -> list[str]:
    return sorted(str(name) for name in attributes.keys())


def _looks_geospatial(attribute_name: str) -> bool:
    normalized = attribute_name.strip().lower()
    return any(token in normalized for token in _GEOSPATIAL_ATTR_TOKENS)


def _audit_rgb_spatial_metadata(
    development_index: Mapping[str, Any], dataset_root: str | Path
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - train extra is installed in production.
        raise RuntimeError("RGB metadata audit requires h5py") from error

    root = Path(dataset_root).expanduser().resolve()
    root_attr_names: set[str] = set()
    image_attr_names: set[str] = set()
    files_with_attributes: list[dict[str, Any]] = []
    files_with_geospatial_attributes: list[dict[str, Any]] = []
    scanned = 0
    missing: list[str] = []
    unexpected_shapes: list[dict[str, Any]] = []
    for split in ("train", "val"):
        for sample_id in development_index["splits"][split]["approved_sample_ids"]:
            path = _rgb_path(root, split, sample_id)
            if not path.is_file():
                missing.append(str(path))
                continue
            with h5py.File(path, "r") as handle:
                if "image" not in handle:
                    raise ValueError(f"approved GAMUS RGB has no image dataset: {path}")
                dataset = handle["image"]
                root_names = _attribute_names(handle.attrs)
                image_names = _attribute_names(dataset.attrs)
                root_attr_names.update(root_names)
                image_attr_names.update(image_names)
                if root_names or image_names:
                    row = {
                        "sample_id": sample_id,
                        "source_split": split,
                        "root_attributes": root_names,
                        "image_attributes": image_names,
                    }
                    files_with_attributes.append(row)
                    if any(_looks_geospatial(name) for name in root_names + image_names):
                        files_with_geospatial_attributes.append(row)
                shape = tuple(int(value) for value in dataset.shape)
                if shape != (1024, 1024, 3):
                    unexpected_shapes.append(
                        {"sample_id": sample_id, "source_split": split, "shape": list(shape)}
                    )
            scanned += 1
    if missing:
        raise FileNotFoundError(
            f"approved GAMUS development RGB files are missing: {missing[:5]}"
        )
    if unexpected_shapes:
        raise ValueError(
            f"approved GAMUS development RGB shapes changed: {unexpected_shapes[:5]}"
        )
    return {
        "scope": "approved train and val RGB HDF5 containers only; no official-test directory or file is enumerated",
        "dataset_root": str(root),
        "files_scanned": scanned,
        "expected_files": sum(
            len(development_index["splits"][split]["approved_sample_ids"])
            for split in ("train", "val")
        ),
        "root_attribute_names": sorted(root_attr_names),
        "image_dataset_attribute_names": sorted(image_attr_names),
        "files_with_any_attributes": len(files_with_attributes),
        "files_with_geospatial_attributes": len(files_with_geospatial_attributes),
        "attribute_examples": files_with_attributes[:20],
        "geospatial_attribute_examples": files_with_geospatial_attributes[:20],
        "per_tile_geospatial_proof_available": bool(files_with_geospatial_attributes),
    }


def build_report(
    approved_index: str | Path,
    dataset_root: str | Path,
    *,
    expected_approved_index_sha256: str,
    holdout_city: str,
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    development = load_development_index(approved_index)
    actual_sha = development["source_sha256"]
    if actual_sha != expected_approved_index_sha256:
        raise ValueError(
            "GAMUS approved-index SHA-256 mismatch: "
            f"expected {expected_approved_index_sha256}, found {actual_sha}"
        )
    contract = build_city_holdout_contract(
        development,
        holdout_city=holdout_city,
        seed=seed,
    )
    learning_index = build_learning_approved_index(development, contract)
    assert_training_ids_respect_contract(
        learning_index["splits"]["train"]["approved_sample_ids"], contract
    )
    report = {
        "schema": REPORT_SCHEMA,
        "contract_id": contract["contract_id"],
        "holdout_contract_schema": contract["schema"],
        "source_approved_index": contract["source_approved_index"],
        "source_approved_index_sha256": contract["source_approved_index_sha256"],
        "selection": contract["selection"],
        "counts": contract["counts"],
        "city_sets": contract["city_sets"],
        "official_test_policy": contract["official_test_policy"],
        "use_policy": contract["use_policy"],
        "verification": contract["verification"],
        "limitations": contract["limitations"],
        "tile_id_audit": _tile_id_audit(development),
        "rgb_spatial_metadata_audit": _audit_rgb_spatial_metadata(
            development, dataset_root
        ),
        "future_training_contract": {
            "approved_index_path": "learning_approved_samples.json",
            "approved_train_count": len(
                learning_index["splits"]["train"]["approved_sample_ids"]
            ),
            "development_validation_count": len(
                learning_index["splits"]["val"]["approved_sample_ids"]
            ),
            "regenerate_learning_only_class_sampling_index": True,
            "do_not_compute_class_weights_or_normalization_from": [
                "development_validation",
                "geographic_holdout",
            ],
            "holdout_evaluation_timing": "only after checkpoint, thresholds, post-processing and guards are frozen",
            "promotion_permitted_by_this_artifact_alone": False,
        },
    }
    report["contract_payload_sha256"] = canonical_sha256(contract)
    return report, learning_index, contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("approved_index", type=Path)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-approved-index-sha256", required=True)
    parser.add_argument("--holdout-city", default="NYC", choices=("DC", "NYC", "PHL"))
    parser.add_argument("--seed", default="msr-gamus-city-holdout-v1")
    args = parser.parse_args()

    report, learning_index, contract = build_report(
        args.approved_index,
        args.dataset_root,
        expected_approved_index_sha256=args.expected_approved_index_sha256,
        holdout_city=args.holdout_city,
        seed=args.seed,
    )
    output_dir = args.output_dir.expanduser().resolve()
    assignments = contract["assignments"]
    artifact_payloads = {
        "report.json": _json_bytes(report),
        "contract.json": _json_bytes(contract),
        "assignments.csv": _csv_bytes(assignments),
        "learning_approved_samples.json": _json_bytes(learning_index),
        "learning_sample_ids.txt": _text_ids(role_sample_ids(contract, "learning")),
        "geographic_holdout_sample_ids.txt": _text_ids(
            role_sample_ids(contract, "geographic_holdout")
        ),
        "buffer_sample_ids.txt": _text_ids(role_sample_ids(contract, "buffer")),
    }
    for name, content in artifact_payloads.items():
        _immutable_write(output_dir / name, content)

    hashes = {
        "schema": ARTIFACT_INDEX_SCHEMA,
        "contract_id": report["contract_id"],
        "source_approved_index_sha256": report["source_approved_index_sha256"],
        "artifacts": {
            name: {
                "sha256": file_sha256(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in sorted(artifact_payloads)
        },
    }
    _immutable_write(output_dir / "artifact_hashes.json", _json_bytes(hashes))
    print(
        json.dumps(
            {
                "status": "sealed",
                "contract_id": report["contract_id"],
                "learning": report["counts"]["by_role"]["learning"],
                "development_validation": report["counts"]["by_role"][
                    "development_validation"
                ],
                "geographic_holdout": report["counts"]["by_role"][
                    "geographic_holdout"
                ],
                "buffer": report["counts"]["by_role"]["buffer"],
                "holdout_city": report["selection"]["holdout_city"],
                "rgb_metadata_files_scanned": report["rgb_spatial_metadata_audit"][
                    "files_scanned"
                ],
                "per_tile_geospatial_proof_available": report[
                    "rgb_spatial_metadata_audit"
                ]["per_tile_geospatial_proof_available"],
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
