import importlib.util
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "data" / "audit_gamus_preparation.py"
SPEC = importlib.util.spec_from_file_location("audit_gamus_preparation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_h5(path: Path, values: np.ndarray, *, attrs: dict[str, str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("image", data=values)
        for key, value in (attrs or {}).items():
            dataset.attrs[key] = value


def _prior_pair(tmp_path: Path) -> tuple[Path, Path, str]:
    source = tmp_path / "GAMUS" / "images" / "train" / "DC_01_RGB.h5"
    prior = tmp_path / "priors" / "train" / "DC_01_REL.h5"
    model = "models/foundation/depth-anything-v2-small-hf"
    _write_h5(source, np.full((3, 4, 3), 100, dtype=np.uint8))
    _write_h5(
        prior,
        np.linspace(0.0, 1.0, 12, dtype=np.float16).reshape(3, 4),
        attrs={"units": "relative_0_1", "model": model, "source_rgb": str(source)},
    )
    return source, prior, model


def test_prior_identity_contract_passes_and_hashes_exact_files(tmp_path: Path) -> None:
    source, prior, model = _prior_pair(tmp_path)

    result = MODULE.validate_prior_pair(
        source_path=source,
        prior_path=prior,
        expected_model=model,
        include_content_hashes=True,
        block_rows=2,
    )

    assert result["status"] == "passed"
    assert result["errors"] == []
    assert result["source_shape"] == [3, 4, 3]
    assert result["prior_shape"] == [3, 4]
    assert result["numeric"]["finite_pixels"] == 12
    assert len(result["source_file_sha256"]) == 64
    assert len(result["prior_file_sha256"]) == 64
    assert "historical_prior_missing_source_rgb_sha256" in result["warnings"]


@pytest.mark.parametrize(
    ("changed_attribute", "value", "expected_error"),
    [
        ("model", "wrong/model", "recorded_model_mismatch"),
        ("source_rgb", "missing/source.h5", "recorded_source_rgb_mismatch"),
        ("units", "metres", "recorded_units_mismatch"),
    ],
)
def test_prior_identity_fails_closed_on_recorded_mismatch(
    tmp_path: Path,
    changed_attribute: str,
    value: str,
    expected_error: str,
) -> None:
    source, prior, model = _prior_pair(tmp_path)
    with h5py.File(prior, "r+") as handle:
        handle["image"].attrs[changed_attribute] = value

    result = MODULE.validate_prior_pair(
        source_path=source,
        prior_path=prior,
        expected_model=model,
        include_content_hashes=False,
    )

    assert result["status"] == "failed"
    assert expected_error in result["errors"]


def test_prior_identity_fails_closed_on_grid_and_numeric_corruption(tmp_path: Path) -> None:
    source, prior, model = _prior_pair(tmp_path)
    _write_h5(
        prior,
        np.asarray([[0.0, np.nan], [1.2, 0.5]], dtype=np.float32),
        attrs={"units": "relative_0_1", "model": model, "source_rgb": str(source)},
    )

    result = MODULE.validate_prior_pair(
        source_path=source,
        prior_path=prior,
        expected_model=model,
        include_content_hashes=False,
    )

    assert result["status"] == "failed"
    assert "prior_grid_mismatch" in result["errors"]
    assert "prior_contains_nonfinite_values" in result["errors"]
    assert "prior_outside_relative_0_1_range" in result["errors"]


def test_height_outlier_inventory_is_joined_to_fine_classes(tmp_path: Path) -> None:
    height = np.asarray([[0.0, 201.0, 250.0], [205.0, 4.0, np.nan]], dtype=np.float32)
    classes = np.asarray([[1, 3, 6], [1, 5, 2]], dtype=np.float32)
    height_path = tmp_path / "height.h5"
    class_path = tmp_path / "class.h5"
    _write_h5(height_path, height)
    _write_h5(class_path, classes)

    result = MODULE.inspect_height_outlier_tile(
        height_path=height_path,
        class_path=class_path,
        threshold=200.0,
    )

    assert result["all"]["count"] == 3
    assert result["all"]["maximum"] == pytest.approx(250.0)
    assert result["by_class"]["1"]["count"] == 1
    assert result["by_class"]["3"]["count"] == 1
    assert result["by_class"]["6"]["count"] == 1


def test_recompute_selection_is_stable_and_never_uses_official_test() -> None:
    values = {
        "train": ["DC_2", "DC_1", "PHL_2", "PHL_1"],
        "val": ["DC_4", "DC_3", "NYC_2", "NYC_1"],
        "test": ["DC_SECRET", "PHL_SECRET"],
    }

    first = MODULE.choose_recompute_samples(values, per_city=1)
    second = MODULE.choose_recompute_samples(values, per_city=1)

    assert first == second
    assert len(first) == 4
    assert all(split in {"train", "val"} for split, _ in first)
    assert not any("SECRET" in sample_id for _, sample_id in first)


def test_quality_contract_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    quality_root = tmp_path / "quality"
    quality_root.mkdir()
    index_path = quality_root / "approved_samples.json"
    tile_path = quality_root / "tile_audit.jsonl"
    index_path.write_text('{"splits": {}}\n', encoding="utf-8")
    tile_path.write_text("\n", encoding="utf-8")
    (quality_root / "qc_report.json").write_text(
        '{"approved_index_sha256": "wrong", "tile_audit_sha256": "wrong"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approved GAMUS index"):
        MODULE._load_quality_contract(quality_root, expected_index_sha256=None)


def test_versioned_output_refuses_overwrite_before_other_work(tmp_path: Path) -> None:
    output = tmp_path / "already_published"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MODULE.build_preparation_audit(
            gamus_root=tmp_path / "missing_gamus",
            prior_root=tmp_path / "missing_priors",
            quality_root=tmp_path / "missing_quality",
            output_root=output,
            project_root=tmp_path,
            expected_index_sha256=None,
            height_cutoff=200.0,
            include_content_hashes=False,
            recompute_samples_per_city=0,
            device="cpu",
        )
