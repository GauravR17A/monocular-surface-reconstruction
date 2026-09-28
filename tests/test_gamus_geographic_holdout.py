from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from msr.data.gamus_geographic_holdout import (
    GAMUS_GEOGRAPHIC_HOLDOUT_SCHEMA,
    assert_training_ids_respect_contract,
    build_city_holdout_contract,
    build_learning_approved_index,
    canonical_sha256,
    file_sha256,
    load_development_index,
    parse_gamus_tile_id,
    role_sample_ids,
)


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "data" / "plan_gamus_geographic_holdout.py"
SPEC = importlib.util.spec_from_file_location("plan_gamus_geographic_holdout_test", SCRIPT)
assert SPEC and SPEC.loader
PLANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANNER)


def _split(ids: list[str]) -> dict[str, object]:
    return {
        "approved_count": len(ids),
        "approved_sample_ids": ids,
        "semantic_eligible_sample_ids": ids,
        "height_regression_eligible_sample_ids": ids,
        "source_count": len(ids),
        "expected_official_count": len(ids),
    }


def _payload(*, nyc_in_val: bool = False) -> dict[str, object]:
    train = [
        "DC_01_01",
        "DC_01_02",
        "NYC_00001",
        "NYC_00002",
        "PHL_0001",
        "PHL_0002",
    ]
    val = ["DC_02_01", "PHL_0003"]
    if nyc_in_val:
        val.append("NYC_00003")
    return {
        "schema": "msr.gamus.approved_samples.v1",
        "dataset": "GAMUS",
        "class_validity_policy": "synthetic",
        "splits": {
            "train": _split(train),
            "val": _split(val),
            # A sentinel official-test inventory proves the derived index omits it.
            "test": _split(["NYC_99999"]),
        },
    }


def _write_index(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    path = tmp_path / "approved_samples.json"
    path.write_text(json.dumps(payload or _payload(), sort_keys=True), encoding="utf-8")
    return path


def _development(tmp_path: Path, payload: dict[str, object] | None = None) -> dict:
    return load_development_index(_write_index(tmp_path, payload))


def test_tile_id_parser_does_not_invent_coordinates() -> None:
    dc = parse_gamus_tile_id("DC_07_21")
    assert (dc.city, dc.row, dc.column) == ("DC", 7, 21)
    assert dc.coordinate_kind == "released_row_column_index"

    nyc = parse_gamus_tile_id("NYC_22835")
    assert nyc.city == "NYC"
    assert nyc.scalar_index == 22835
    assert nyc.row is None and nyc.column is None
    assert nyc.coordinate_kind == "opaque_released_scalar_index"

    with pytest.raises(ValueError, match="malformed"):
        parse_gamus_tile_id("OMA_1")


def test_whole_city_plan_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    development = _development(tmp_path)
    first = build_city_holdout_contract(development, holdout_city="NYC")
    second = build_city_holdout_contract(development, holdout_city="NYC")

    assert first == second
    assert first["schema"] == GAMUS_GEOGRAPHIC_HOLDOUT_SCHEMA
    assert role_sample_ids(first, "geographic_holdout") == (
        "NYC_00001",
        "NYC_00002",
    )
    assert role_sample_ids(first, "learning") == (
        "DC_01_01",
        "DC_01_02",
        "PHL_0001",
        "PHL_0002",
    )
    assert role_sample_ids(first, "buffer") == ()
    assert all(first["verification"].values())
    assert first["official_test_policy"]["allowed"] is False
    assert first["use_policy"]["existing_full-train_candidates_are_eligible"] is False


def test_holdout_city_in_validation_fails_closed(tmp_path: Path) -> None:
    development = _development(tmp_path, _payload(nyc_in_val=True))
    with pytest.raises(ValueError, match="checkpoint selection would contaminate"):
        build_city_holdout_contract(development, holdout_city="NYC")


def test_training_inventory_guard_rejects_every_nonlearning_role(tmp_path: Path) -> None:
    contract = build_city_holdout_contract(_development(tmp_path), holdout_city="NYC")
    learning = role_sample_ids(contract, "learning")
    assert_training_ids_respect_contract(learning, contract)

    with pytest.raises(ValueError, match="leaks protected"):
        assert_training_ids_respect_contract([*learning, "NYC_00001"], contract)
    with pytest.raises(ValueError, match="leaks protected"):
        assert_training_ids_respect_contract([*learning, "DC_02_01"], contract)
    with pytest.raises(ValueError, match="outside the learning role"):
        assert_training_ids_respect_contract([*learning, "DC_99_99"], contract)
    with pytest.raises(ValueError, match="omits approved learning"):
        assert_training_ids_respect_contract(learning[:-1], contract)

    buffered = deepcopy(contract)
    buffered_row = next(
        row for row in buffered["assignments"] if row["sample_id"] == "DC_01_01"
    )
    buffered_row["role"] = "buffer"
    buffered["counts"]["by_role"]["learning"] -= 1
    buffered["counts"]["by_role"]["buffer"] += 1
    buffered["assignment_sha256"] = canonical_sha256(buffered["assignments"])
    with pytest.raises(ValueError, match="leaks protected"):
        assert_training_ids_respect_contract(learning, buffered)


def test_learning_index_is_loader_compatible_and_omits_test(tmp_path: Path) -> None:
    development = _development(tmp_path)
    contract = build_city_holdout_contract(development, holdout_city="NYC")
    derived = build_learning_approved_index(development, contract)

    assert list(derived["splits"]) == ["train", "val"]
    assert "NYC_99999" not in json.dumps(derived)
    assert derived["splits"]["train"]["approved_sample_ids"] == [
        "DC_01_01",
        "DC_01_02",
        "PHL_0001",
        "PHL_0002",
    ]
    assert derived["splits"]["train"]["approved_count"] == 4
    assert derived["splits"]["val"]["approved_count"] == 2
    assert derived["derivation"]["official_test_split_copied"] is False
    # Source payload is evidence and must remain unchanged.
    assert development["source_payload"]["splits"]["train"]["approved_count"] == 6


def _write_rgb(root: Path, split: str, sample_id: str, *, attr: bool = False) -> None:
    h5py = pytest.importorskip("h5py")
    suffix = "IMG" if sample_id.startswith("NYC_") else "RGB"
    path = root / "images" / split / f"{sample_id}_{suffix}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(
            "image", shape=(1024, 1024, 3), dtype="uint8", chunks=(64, 64, 3)
        )
        if attr:
            dataset.attrs["sensor_note"] = "synthetic"


def test_report_scans_only_development_rgb_and_seals_plan(tmp_path: Path) -> None:
    index_path = _write_index(tmp_path)
    dataset_root = tmp_path / "gamus"
    payload = _payload()
    for split in ("train", "val"):
        for sample_id in payload["splits"][split]["approved_sample_ids"]:
            _write_rgb(dataset_root, split, sample_id, attr=sample_id == "DC_01_01")

    report, derived, contract = PLANNER.build_report(
        index_path,
        dataset_root,
        expected_approved_index_sha256=file_sha256(index_path),
        holdout_city="NYC",
        seed="fixed",
    )

    metadata = report["rgb_spatial_metadata_audit"]
    assert metadata["files_scanned"] == 8
    assert metadata["files_with_any_attributes"] == 1
    assert metadata["files_with_geospatial_attributes"] == 0
    assert "test" not in derived["splits"]
    assert contract["selection"]["selection_uses_labels_or_pixels"] is False
    assert report["tile_id_audit"]["dc_released_grid_adjacency"] == {
        "validation_tiles": 1,
        "validation_tiles_touching_train_4_neighbour": 1,
        "validation_tiles_touching_train_8_neighbour": 1,
        "metric_distance_claimed": False,
    }


def test_wrong_index_hash_and_differing_sealed_artifact_fail_closed(tmp_path: Path) -> None:
    index_path = _write_index(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PLANNER.build_report(
            index_path,
            tmp_path / "unused",
            expected_approved_index_sha256="0" * 64,
            holdout_city="NYC",
            seed="fixed",
        )

    path = tmp_path / "sealed.json"
    PLANNER._immutable_write(path, b"same\n")
    PLANNER._immutable_write(path, b"same\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        PLANNER._immutable_write(path, b"different\n")
