import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from msr.data.gamus_dataset import load_gamus_split  # noqa: E402


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "data" / "audit_gamus_dataset.py"
SPEC = importlib.util.spec_from_file_location("audit_gamus_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("image", data=values)


def _triplet(
    root: Path,
    split: str,
    sample_id: str,
    *,
    shape: tuple[int, int] = (3, 4),
    classes: np.ndarray | None = None,
) -> None:
    rgb = np.full((*shape, 3), 120, dtype=np.uint8)
    height = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
    if classes is None:
        classes = np.full(shape, 3, dtype=np.uint8)
    suffix = "IMG" if sample_id.startswith("NYC") else "RGB"
    _write(root / "images" / split / f"{sample_id}_{suffix}.h5", rgb)
    _write(root / "heights" / split / f"{sample_id}_AGL.h5", height)
    _write(root / "classes" / split / f"{sample_id}_CLS.h5", classes)


def test_full_contract_keeps_source_and_filters_known_quarantine(tmp_path: Path) -> None:
    root = tmp_path / "GAMUS"
    _triplet(root, "train", "DC_01_01")
    _triplet(root, "val", "PHL_3001")
    _triplet(root, "test", "NYC_9001")
    _triplet(root, "test", "PHL_4001")
    output = tmp_path / "quality_v1"

    report = MODULE.build_quality_contract(
        root,
        output,
        expected_counts={"train": 1, "val": 1, "test": 2},
        show_progress=False,
        block_rows=2,
    )

    assert report["totals"] == {
        "audited_tiles": 4,
        "approved_tiles": 3,
        "quarantined_tiles": 1,
    }
    assert report["split_ids_disjoint"] is True
    index = json.loads((output / "approved_samples.json").read_text())
    assert index["height_unit_contract"]["status"] == "explicit_project_assumption"
    assert index["splits"]["test"]["approved_sample_ids"] == ["NYC_9001"]
    quarantine = json.loads((output / "quarantine.json").read_text())
    assert quarantine["mode"] == "logical_only_source_files_untouched"
    assert quarantine["records"][0]["sample_id"] == "PHL_4001"
    assert (root / "heights" / "test" / "PHL_4001_AGL.h5").is_file()
    persisted_report = json.loads((output / "qc_report.json").read_text())
    assert persisted_report["artifact_hash_scope"] == (
        "sha256_of_exact_persisted_file_bytes"
    )
    assert persisted_report["approved_index_sha256"] == _sha256(
        output / "approved_samples.json"
    )
    assert persisted_report["quarantine_contract_sha256"] == _sha256(
        output / "quarantine.json"
    )
    assert persisted_report["tile_audit_sha256"] == _sha256(
        output / "tile_audit.jsonl"
    )

    loaded = load_gamus_split(
        root, "test", approved_index_path=output / "approved_samples.json"
    )
    assert [record.sample_id for record in loaded] == ["NYC_9001"]


def test_partially_invalid_classes_are_masked_and_output_is_immutable(tmp_path: Path) -> None:
    root = tmp_path / "GAMUS"
    _triplet(root, "train", "DC_GOOD")
    invalid = np.full((3, 4), 3, dtype=np.float32)
    invalid[0, 0] = 255
    _triplet(root, "val", "PHL_BAD", classes=invalid)
    _triplet(root, "test", "NYC_GOOD")
    output = tmp_path / "quality_v1"

    report = MODULE.build_quality_contract(
        root,
        output,
        expected_counts={"train": 1, "val": 1, "test": 1},
        show_progress=False,
    )

    assert report["splits"]["val"]["approved_tiles"] == 1
    assert report["splits"]["val"]["quarantined_sample_ids"] == []
    line_records = [
        json.loads(line)
        for line in (output / "tile_audit.jsonl").read_text().splitlines()
    ]
    bad = next(record for record in line_records if record["sample_id"] == "PHL_BAD")
    assert bad["reasons"] == []
    assert "invalid_class_pixels_masked_as_ignore" in bad["warnings"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.build_quality_contract(
            root,
            output,
            expected_counts={"train": 1, "val": 1, "test": 1},
            show_progress=False,
        )


def test_zero_regression_tile_is_approved_for_semantics(tmp_path: Path) -> None:
    root = tmp_path / "GAMUS"
    semantic_only = np.full((3, 4), 5, dtype=np.uint8)  # road: semantic-only policy
    _triplet(root, "train", "DC_ROAD", classes=semantic_only)
    _triplet(root, "val", "PHL_GOOD")
    _triplet(root, "test", "NYC_GOOD")
    output = tmp_path / "quality_v1"

    # Force all road heights outside the accepted range; the tile still has
    # valid semantic supervision and must not be mislabeled as corrupt.
    height_path = root / "heights" / "train" / "DC_ROAD_AGL.h5"
    _write(height_path, np.full((3, 4), -5.0, dtype=np.float32))
    report = MODULE.build_quality_contract(
        root,
        output,
        expected_counts={"train": 1, "val": 1, "test": 1},
        show_progress=False,
    )

    assert report["splits"]["train"]["approved_tiles"] == 1
    assert report["splits"]["train"]["semantic_eligible_tiles"] == 1
    assert report["splits"]["train"]["height_regression_eligible_tiles"] == 0
    record = json.loads((output / "tile_audit.jsonl").read_text().splitlines()[0])
    assert record["status"] == "approved"
    assert record["eligible_tasks"] == ["semantic"]
    assert "no_regression_supervision_semantic_supervision_only" in record["warnings"]


def test_split_overlap_fails_before_atomic_publication(tmp_path: Path) -> None:
    root = tmp_path / "GAMUS"
    _triplet(root, "train", "DC_DUPLICATE")
    _triplet(root, "val", "DC_DUPLICATE")
    _triplet(root, "test", "NYC_ONLY")
    output = tmp_path / "quality_v1"

    with pytest.raises(ValueError, match="tile leakage"):
        MODULE.build_quality_contract(
            root,
            output,
            expected_counts={"train": 1, "val": 1, "test": 1},
            show_progress=False,
        )
    assert not output.exists()


def test_loader_rejects_unknown_or_inconsistent_approved_index(tmp_path: Path) -> None:
    root = tmp_path / "GAMUS"
    _triplet(root, "train", "DC_01")
    bad_index = tmp_path / "bad.json"
    bad_index.write_text(
        json.dumps(
            {
                "schema": "msr.gamus.approved_samples.v1",
                "splits": {
                    "train": {
                        "approved_count": 1,
                        "approved_sample_ids": ["DC_MISSING"],
                    }
                },
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="references missing"):
        load_gamus_split(root, "train", approved_index_path=bad_index)
