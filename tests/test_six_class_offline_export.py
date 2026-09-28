import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from msr.io.six_class_export import (
    AUTHORIZATION_SCHEMA,
    CLASS_NAMES,
    INVALID_CLASS_ID,
    SixClassExportError,
    authorize_validated_candidate,
    export_six_class_bundle,
    file_sha256,
)


def _scores_for_ids(class_ids: np.ndarray) -> np.ndarray:
    scores = np.zeros((6, *class_ids.shape), dtype=np.float32)
    for class_id in range(6):
        scores[class_id, class_ids == class_id] = 1.0
    return scores


def _projected_profile(shape: tuple[int, int]) -> dict:
    return {
        "driver": "GTiff",
        "height": shape[0],
        "width": shape[1],
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:32643",
        "transform": Affine.translation(400000, 3000000) * Affine.scale(2, -3),
    }


def test_offline_export_is_aligned_masked_and_area_aware(tmp_path: Path) -> None:
    intended_ids = np.array([[0, 1, 2, 3], [4, 5, 1, 5]], dtype=np.uint8)
    image_valid = np.array(
        [[True, True, True, True], [True, True, True, False]]
    )
    classification_valid = np.array(
        [[True, True, True, False], [True, True, True, False]]
    )
    height_valid = np.array(
        [[True, True, True, False], [False, True, False, False]]
    )
    height = np.array(
        [[0, 10, 2, 999], [0, 30, 999, 999]], dtype=np.float32
    )
    output = tmp_path / "bundle"

    summary = export_six_class_bundle(
        output,
        fine_semantic_probability_maps=_scores_for_ids(intended_ids),
        height_map_m=height,
        reference_profile=_projected_profile(intended_ids.shape),
        image_valid_mask=image_valid,
        classification_valid_mask=classification_valid,
        height_valid_mask=height_valid,
        source_image=tmp_path / "rgb.tif",
        **_authorization_args(tmp_path),
    )

    with rasterio.open(output / "six_class_ids.tif") as raster:
        exported_ids = raster.read(1)
        assert raster.crs == rasterio.crs.CRS.from_epsg(32643)
        assert raster.transform == _projected_profile(intended_ids.shape)["transform"]
        assert raster.nodata == INVALID_CLASS_ID
        assert raster.tags()["MSR_CLASS_5"] == "trees"
    np.testing.assert_array_equal(
        exported_ids,
        np.array([[0, 1, 2, 255], [4, 5, 1, 255]], dtype=np.uint8),
    )
    with rasterio.open(output / "height_m.tif") as raster:
        exported_height = raster.read(1)
    assert np.isnan(exported_height[0, 3])
    assert np.isnan(exported_height[1, 2])
    assert exported_height[0, 1] == pytest.approx(10)

    assert summary["class_names"] == list(CLASS_NAMES)
    assert summary["classes"]["buildings"]["pixel_count"] == 2
    assert summary["classes"]["buildings"]["pixel_share"] == pytest.approx(2 / 6)
    assert summary["classes"]["buildings"]["coverage_m2"] == pytest.approx(12)
    assert summary["area"]["pixel_area_m2"] == pytest.approx(6)
    assert summary["area"]["classified_coverage_m2"] == pytest.approx(36)
    assert set(summary["height_summaries_m"]) == {
        "buildings",
        "low_vegetation",
        "trees",
    }
    assert summary["height_summaries_m"]["buildings"]["mean_m"] == pytest.approx(10)
    assert summary["height_summaries_m"]["low_vegetation"]["status"] == "unavailable"
    assert summary["height_summaries_m"]["trees"]["mean_m"] == pytest.approx(30)
    assert "water" not in summary["height_summaries_m"]
    assert "roads" not in summary["height_summaries_m"]
    assert summary["probability_or_confidence_exported"] is False
    assert summary["app_active_model_changed"] is False
    assert set(summary["artifacts"]) == {
        "six_class_ids.tif",
        "height_m.tif",
        "image_valid_mask.tif",
        "classification_valid_mask.tif",
        "height_valid_mask.tif",
    }
    on_disk = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["classes"] == summary["classes"]


def test_square_metres_are_omitted_for_geographic_crs(tmp_path: Path) -> None:
    shape = (2, 2)
    profile = _projected_profile(shape)
    profile.update(
        crs="EPSG:4326",
        transform=Affine.translation(77, 28) * Affine.scale(0.00001, -0.00001),
    )
    summary = export_six_class_bundle(
        tmp_path / "geographic",
        fine_semantic_probability_maps=_scores_for_ids(
            np.array([[0, 1], [4, 5]], dtype=np.uint8)
        ),
        height_map_m=np.ones(shape, dtype=np.float32),
        reference_profile=profile,
        image_valid_mask=np.ones(shape, dtype=bool),
        classification_valid_mask=np.ones(shape, dtype=bool),
        height_valid_mask=np.ones(shape, dtype=bool),
        source_image=tmp_path / "rgb.tif",
        **_authorization_args(tmp_path),
    )

    assert summary["area"]["status"] == "unavailable"
    assert summary["area"]["pixel_area_m2"] is None
    assert "not projected" in summary["area"]["reason"]
    assert all(value["coverage_m2"] is None for value in summary["classes"].values())


def test_validity_masks_remain_independent_and_cannot_escape_image(tmp_path: Path) -> None:
    shape = (2, 2)
    with pytest.raises(SixClassExportError, match="classification validity"):
        export_six_class_bundle(
            tmp_path / "bad",
            fine_semantic_probability_maps=np.ones((6, *shape), dtype=np.float32),
            height_map_m=np.ones(shape, dtype=np.float32),
            reference_profile=_projected_profile(shape),
            image_valid_mask=np.array([[True, False], [True, True]]),
            classification_valid_mask=np.ones(shape, dtype=bool),
            height_valid_mask=np.zeros(shape, dtype=bool),
            source_image=tmp_path / "rgb.tif",
            **_authorization_args(tmp_path),
        )
    assert not (tmp_path / "bad").exists()


def test_missing_height_is_not_interpreted_as_zero(tmp_path: Path) -> None:
    shape = (1, 2)
    height = np.array([[0.0, 12.0]], dtype=np.float32)
    summary = export_six_class_bundle(
        tmp_path / "missing_height",
        fine_semantic_probability_maps=_scores_for_ids(
            np.array([[1, 1]], dtype=np.uint8)
        ),
        height_map_m=height,
        reference_profile=_projected_profile(shape),
        image_valid_mask=np.ones(shape, dtype=bool),
        classification_valid_mask=np.ones(shape, dtype=bool),
        height_valid_mask=np.array([[False, True]]),
        source_image=tmp_path / "rgb.tif",
        **_authorization_args(tmp_path),
    )

    building = summary["height_summaries_m"]["buildings"]
    assert building["height_valid_pixels"] == 1
    assert building["mean_m"] == pytest.approx(12)
    with rasterio.open(tmp_path / "missing_height" / "height_m.tif") as raster:
        exported = raster.read(1)
    assert np.isnan(exported[0, 0])


def test_invalid_probability_maps_are_rejected(tmp_path: Path) -> None:
    shape = (1, 1)
    invalid = np.ones((6, *shape), dtype=np.float32)
    with pytest.raises(SixClassExportError, match="sum to one"):
        export_six_class_bundle(
            tmp_path / "invalid_probabilities",
            fine_semantic_probability_maps=invalid,
            height_map_m=np.ones(shape, dtype=np.float32),
            reference_profile=_projected_profile(shape),
            image_valid_mask=np.ones(shape, dtype=bool),
            classification_valid_mask=np.ones(shape, dtype=bool),
            height_valid_mask=np.ones(shape, dtype=bool),
            source_image=tmp_path / "rgb.tif",
            **_authorization_args(tmp_path),
        )


def _write_authorization(path: Path, checkpoint: Path, **gate_overrides: bool) -> None:
    gates = {
        "eligible": True,
        "six_class_validation_passed": True,
        "height_regression_guards_passed": True,
        "unseen_geography_evaluated": True,
    }
    gates.update(gate_overrides)
    path.write_text(
        json.dumps(
            {
                "schema": AUTHORIZATION_SCHEMA,
                "candidate_checkpoint": {
                    "path": str(checkpoint.resolve()),
                    "sha256": file_sha256(checkpoint),
                },
                "offline_export": gates,
            }
        ),
        encoding="utf-8",
    )


def _authorization_args(tmp_path: Path) -> dict:
    checkpoint = tmp_path / "export_candidate.pt"
    checkpoint.write_bytes(b"validated export candidate")
    report = tmp_path / "export_validation.json"
    _write_authorization(report, checkpoint)
    return {
        "checkpoint_path": checkpoint,
        "validation_report_path": report,
        "model_metadata": {
            "epoch": 3,
            "model_type": "domain_gated_surface_v4_six_class",
            "fine_semantic_head_type": "linear",
        },
    }


def test_authorization_binds_checkpoint_and_requires_all_gates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"sealed candidate")
    report = tmp_path / "validation.json"
    _write_authorization(report, checkpoint)

    identity = authorize_validated_candidate(report, checkpoint)
    assert identity["checkpoint_sha256"] == file_sha256(checkpoint)
    assert all(identity["gates"].values())

    failed_report = tmp_path / "failed.json"
    _write_authorization(
        failed_report, checkpoint, height_regression_guards_passed=False
    )
    with pytest.raises(SixClassExportError, match="height_regression_guards_passed"):
        authorize_validated_candidate(failed_report, checkpoint)

    other_checkpoint = tmp_path / "other.pt"
    other_checkpoint.write_bytes(b"other")
    with pytest.raises(SixClassExportError, match="different checkpoint"):
        authorize_validated_candidate(report, other_checkpoint)


def test_export_refuses_to_overwrite_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    shape = (1, 1)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        export_six_class_bundle(
            output,
            fine_semantic_probability_maps=np.ones((6, *shape), dtype=np.float32),
            height_map_m=np.ones(shape, dtype=np.float32),
            reference_profile=_projected_profile(shape),
            image_valid_mask=np.ones(shape, dtype=bool),
            classification_valid_mask=np.ones(shape, dtype=bool),
            height_valid_mask=np.ones(shape, dtype=bool),
            source_image=tmp_path / "rgb.tif",
            **_authorization_args(tmp_path),
        )
