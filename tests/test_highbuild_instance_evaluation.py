from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
import tarfile

from affine import Affine
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS

from msr.evaluation.highbuild_instances import (
    HighBuildInstanceAccumulator,
    HighBuildInstanceEvaluationError,
    authenticated_pixel_area_m2,
    evaluate_highbuild_tile,
    parse_highbuild_instances,
)


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_highbuild_instances.py"
SPEC = importlib.util.spec_from_file_location("highbuild_instance_cli_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def _coco(*, missing_estimated_flag: bool = False) -> dict[str, object]:
    annotations: list[dict[str, object]] = [
        {
            "id": 1,
            "image_id": 7,
            "category_id": 1,
            "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]],
            "iscrowd": 0,
            "attributes": {
                "height": 10.0,
                "is_estimated_height": False,
            },
        },
        {
            "id": 2,
            "image_id": 7,
            "category_id": 1,
            "segmentation": [[9, 2, 14, 2, 14, 7, 9, 7]],
            "iscrowd": 0,
            "attributes": {
                "height": 15.0,
                "is_estimated_height": True,
            },
        },
    ]
    if missing_estimated_flag:
        del annotations[0]["attributes"]["is_estimated_height"]  # type: ignore[index]
    return {
        "images": [{"id": 7, "file_name": "tile.jpg", "height": 12, "width": 16}],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "building"}],
    }


def _reference(
    *, protocol: str
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    coco = _coco()
    instances = parse_highbuild_instances(coco, height=12, width=16)
    building = np.zeros((12, 16), dtype=bool)
    measured = np.zeros_like(building)
    height = np.zeros((12, 16), dtype=np.float32)
    for instance in instances:
        building |= instance.mask
        assert instance.height_m is not None
        height[instance.mask] = instance.height_m
        if not instance.is_estimated_height:
            measured |= instance.mask
    valid = building if protocol == "inclusive_annotated" else measured
    return coco, height, building, valid


def _evaluate(
    *,
    protocol: str = "strict_measured",
    predicted_building: np.ndarray | None = None,
    classification_valid: np.ndarray | None = None,
) -> dict[str, object]:
    coco, target, building, height_valid = _reference(protocol=protocol)
    prediction_height = target.copy()
    prediction_height[building & (target == 10)] = 12.0
    return evaluate_highbuild_tile(
        sample_id="synthetic",
        prediction_height_m=prediction_height,
        prediction_building=building.astype(np.float32)
        if predicted_building is None
        else predicted_building,
        coco=coco,
        reference_height_m=target,
        building_footprints=building,
        height_valid_mask=height_valid,
        image_valid_mask=np.ones_like(building),
        classification_valid_mask=classification_valid,
        height_protocol=protocol,
        building_probability_threshold=0.5,
        instance_iou_threshold=0.5,
        minimum_predicted_instance_pixels=1,
        boundary_tolerance_pixels=1,
    )


def test_positive_only_contract_reports_recall_but_not_precision_or_count() -> None:
    result = _evaluate()

    assert result["support"]["reference_by_provenance"] == {
        "measured": 1,
        "estimated": 1,
        "missing_height": 0,
    }
    assert result["detection"]["annotated_building_recall"] == pytest.approx(1.0)
    assert result["detection"]["precision"] is None
    assert result["detection"]["count_error_predicted_minus_reference"] is None
    assert (
        result["counting"]["status"]
        == "unavailable_positive_annotations_are_not_exhaustive"
    )
    assert result["counting"]["metric_source"] == (
        "instance_counts_scored_separately_from_pixel_metrics"
    )
    assert result["counting"]["reference_count"] is None
    assert result["counting"]["absolute_error"] is None
    assert result["outline"]["mean_iou"] == pytest.approx(1.0)
    assert result["outline"]["mean_boundary_f1"] == pytest.approx(1.0)
    assert result["instance_height"]["measured"]["instance_count"] == 1
    assert result["instance_height"]["measured"]["rmse_m"] == pytest.approx(2.0)
    assert result["instance_height"]["measured"]["bias_m"] == pytest.approx(2.0)
    assert result["instance_height"]["estimated"]["instance_count"] == 0
    assert (
        result["instance_height"]["estimated"]["status"]
        == "excluded_by_strict_measured_protocol"
    )
    assert result["coverage"] == {
        "status": "unavailable_without_projected_crs_and_authenticated_scale"
    }
    measured_row = next(
        row for row in result["per_building"] if row["provenance"] == "measured"
    )
    assert measured_row["height_evaluation_status"] == "evaluated"
    assert measured_row["predicted_height_m"] == pytest.approx(12.0)
    assert measured_row["height_error_m"] == pytest.approx(2.0)


def test_exhaustive_support_enables_precision_recall_and_count() -> None:
    _, _, reference_building, _ = _reference(protocol="strict_measured")
    predicted = np.zeros_like(reference_building, dtype=np.float32)
    predicted[1:5, 1:5] = 1.0  # first object only
    predicted[9:11, 12:15] = 1.0  # certified false positive

    result = _evaluate(
        predicted_building=predicted,
        classification_valid=np.ones_like(reference_building),
    )

    assert result["support"]["predicted_instances"] == 2
    assert result["support"]["matched_instances"] == 1
    assert result["detection"]["precision"] == pytest.approx(0.5)
    assert result["detection"]["recall"] == pytest.approx(0.5)
    assert result["detection"]["f1"] == pytest.approx(0.5)
    assert result["detection"]["count_error_predicted_minus_reference"] == 0
    assert (
        result["counting"]["status"]
        == "validated_against_exhaustive_instance_reference"
    )
    assert result["counting"]["reference_count"] == 2
    assert result["counting"]["predicted_instances"] == 2
    assert result["counting"]["signed_error_predicted_minus_reference"] == 0
    assert result["counting"]["absolute_error"] == 0
    assert result["counting"]["exact_count"] is True


def test_counting_metrics_are_per_scene_so_opposite_errors_do_not_cancel() -> None:
    coco, target, building, height_valid = _reference(protocol="strict_measured")
    accumulator = HighBuildInstanceAccumulator(height_protocol="strict_measured")
    exhaustive = np.ones_like(building)

    one_prediction = np.zeros_like(building, dtype=np.float32)
    one_prediction[1:5, 1:5] = 1.0
    three_predictions = building.astype(np.float32)
    three_predictions[9:11, 12:15] = 1.0
    for sample_id, prediction in (
        ("undercount", one_prediction),
        ("overcount", three_predictions),
    ):
        accumulator.update(
            sample_id=sample_id,
            prediction_height_m=target,
            prediction_building=prediction,
            coco=coco,
            reference_height_m=target,
            building_footprints=building,
            height_valid_mask=height_valid,
            image_valid_mask=np.ones_like(building),
            classification_valid_mask=exhaustive,
            height_protocol="strict_measured",
        )

    counting = accumulator.compute()["counting"]
    assert counting["total_signed_error_predicted_minus_reference"] == 0
    assert counting["per_scene_mae"] == pytest.approx(1.0)
    assert counting["per_scene_rmse"] == pytest.approx(1.0)
    assert counting["exact_count_scene_rate"] == pytest.approx(0.0)
    assert counting["overcount_scene_count"] == 1
    assert counting["undercount_scene_count"] == 1


def test_inclusive_protocol_reports_estimated_heights_separately() -> None:
    result = _evaluate(protocol="inclusive_annotated")

    assert result["instance_height"]["combined"]["instance_count"] == 2
    assert result["instance_height"]["measured"]["rmse_m"] == pytest.approx(2.0)
    assert result["instance_height"]["estimated"]["instance_count"] == 1
    assert result["instance_height"]["estimated"]["rmse_m"] == pytest.approx(0.0)
    assert result["instance_height"]["estimated"]["status"] == "included_separately"


def test_mismatched_corrected_mask_fails_closed() -> None:
    coco, target, building, height_valid = _reference(protocol="strict_measured")
    building[0, 0] = True

    with pytest.raises(HighBuildInstanceEvaluationError, match="disagrees"):
        evaluate_highbuild_tile(
            sample_id="bad-contract",
            prediction_height_m=target,
            prediction_building=building,
            coco=coco,
            reference_height_m=target,
            building_footprints=building,
            height_valid_mask=height_valid,
            image_valid_mask=np.ones_like(building),
            height_protocol="strict_measured",
        )


def test_missing_estimated_flag_is_disclosed_under_corrected_contract_default() -> None:
    instances = parse_highbuild_instances(
        _coco(missing_estimated_flag=True), height=12, width=16
    )

    assert instances[0].is_estimated_height is False
    assert instances[0].estimated_flag_explicit is False
    assert instances[1].is_estimated_height is True
    assert instances[1].estimated_flag_explicit is True


def test_non_boolean_estimated_flag_fails_closed() -> None:
    coco = _coco()
    coco["annotations"][0]["attributes"]["is_estimated_height"] = "false"  # type: ignore[index]

    with pytest.raises(HighBuildInstanceEvaluationError, match="non-boolean"):
        parse_highbuild_instances(coco, height=12, width=16)


def test_subpixel_annotation_is_disclosed_not_counted_as_a_detection_miss() -> None:
    coco = {
        "images": [{"id": 7, "file_name": "tiny.jpg", "height": 4, "width": 4}],
        "annotations": [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 1,
                "segmentation": [[0.05, 0.05, 0.15, 0.05, 0.15, 0.15]],
                "iscrowd": 0,
                "attributes": {"height": 3.0, "is_estimated_height": False},
            }
        ],
        "categories": [{"id": 1, "name": "building"}],
    }
    empty = np.zeros((4, 4), dtype=np.float32)
    result = evaluate_highbuild_tile(
        sample_id="subpixel",
        prediction_height_m=empty,
        prediction_building=empty,
        coco=coco,
        reference_height_m=empty,
        building_footprints=empty,
        height_valid_mask=empty,
        image_valid_mask=np.ones_like(empty, dtype=bool),
        height_protocol="strict_measured",
    )

    assert result["support"]["source_annotations"] == 1
    assert result["support"]["unrasterizable_annotations_excluded"] == 1
    assert result["support"]["reference_instances"] == 0
    assert result["detection"]["annotated_building_recall"] is None


def test_square_metre_area_requires_projected_crs() -> None:
    assert authenticated_pixel_area_m2(crs=None, transform=Affine.scale(2, -2)) is None
    assert (
        authenticated_pixel_area_m2(
            crs=CRS.from_epsg(4326), transform=Affine.scale(0.01, -0.01)
        )
        is None
    )
    assert authenticated_pixel_area_m2(
        crs=CRS.from_epsg(3857), transform=Affine.scale(2, -2)
    ) == pytest.approx(4.0)


def test_accumulator_preserves_pixel_and_instance_metrics() -> None:
    coco, target, building, height_valid = _reference(protocol="strict_measured")
    accumulator = HighBuildInstanceAccumulator(height_protocol="strict_measured")
    for sample_id in ("one", "two"):
        accumulator.update(
            sample_id=sample_id,
            prediction_height_m=target + building.astype(np.float32),
            prediction_building=building,
            coco=coco,
            reference_height_m=target,
            building_footprints=building,
            height_valid_mask=height_valid,
            image_valid_mask=np.ones_like(building),
            height_protocol="strict_measured",
        )
    result = accumulator.compute()

    assert result["tile_count"] == 2
    assert result["support"]["reference_instances"] == 4
    assert result["support"]["matched_instances"] == 4
    assert result["pixel_height"]["rmse_m"] == pytest.approx(1.0)
    assert result["instance_height"]["measured"]["instance_count"] == 2
    assert result["instance_height"]["measured"]["rmse_m"] == pytest.approx(1.0)
    assert (
        result["counting"]["status"]
        == "unavailable_positive_annotations_are_not_exhaustive"
    )
    assert result["counting"]["per_scene_mae"] is None


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    crs: CRS | None = None,
    transform: Affine = Affine.identity(),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=str(values.dtype),
        crs=crs,
        transform=transform,
    ) as destination:
        destination.write(values, 1)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_model_agnostic_cli_runner_verifies_contract_and_coco_shard(tmp_path: Path) -> None:
    coco, target, building, height_valid = _reference(protocol="strict_measured")
    transform = Affine.translation(500_000, 2_000_000) * Affine.scale(2, -2)
    crs = CRS.from_epsg(3857)
    sample_id = "synthetic_tile"
    sample_dir = tmp_path / "sample"
    reference_path = sample_dir / "building_height_m.tif"
    building_path = sample_dir / "building_footprints.tif"
    valid_path = sample_dir / "measured_height_valid.tif"
    prediction_height_path = tmp_path / "predictions" / "height.tif"
    prediction_building_path = tmp_path / "predictions" / "building.tif"
    class_valid_path = tmp_path / "predictions" / "class_valid.tif"
    for path, values in (
        (reference_path, target),
        (building_path, building.astype(np.uint8)),
        (valid_path, height_valid.astype(np.uint8)),
        (prediction_height_path, target),
        (prediction_building_path, building.astype(np.uint8)),
        (class_valid_path, np.ones_like(building, dtype=np.uint8)),
    ):
        _write_raster(path, values, crs=crs, transform=transform)

    corrected_manifest = tmp_path / "corrected" / "manifests" / "validation.csv"
    manifest_fields = [
        "sample_id",
        "region",
        "landscape",
        "rgb_path",
        "surface_path",
        "target_kind",
        "dtm_path",
        "building_mask_path",
        "vegetation_mask_path",
        "valid_mask_path",
        "relative_prior_path",
        "gsd_m",
    ]
    rgb_path = sample_dir / "rgb.tif"
    _write_raster(rgb_path, np.ones_like(building, dtype=np.uint8), crs=crs, transform=transform)
    _write_csv(
        corrected_manifest,
        manifest_fields,
        [
            {
                "sample_id": sample_id,
                "region": "synthetic-city",
                "landscape": "urban",
                "rgb_path": str(rgb_path),
                "surface_path": str(reference_path),
                "target_kind": "building_height",
                "dtm_path": "",
                "building_mask_path": str(building_path),
                "vegetation_mask_path": "",
                "valid_mask_path": str(valid_path),
                "relative_prior_path": "",
                "gsd_m": "",
            }
        ],
    )
    contract_report = corrected_manifest.parent.parent / "contract_report.json"
    contract_report.write_text(
        json.dumps(
            {
                "schema": "msr.highbuild_annotation_contract.v2",
                "historical_artifacts_modified": False,
                "height_protocol": "strict_measured",
                "test_role": "synthetic",
                "splits": {
                    "validation": {
                        "output_manifest_sha256": SCRIPT.file_sha256(corrected_manifest)
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    split_root = tmp_path / "split_root"
    shard_path = split_root / "shards" / "validation" / "one.tar"
    shard_path.parent.mkdir(parents=True)
    encoded_coco = json.dumps(coco).encode("utf-8")
    with tarfile.open(shard_path, "w") as archive:
        info = tarfile.TarInfo(f"{sample_id}.json")
        info.size = len(encoded_coco)
        archive.addfile(info, io.BytesIO(encoded_coco))
    source_index = split_root / "validation.csv"
    _write_csv(
        source_index,
        ["webdataset_key", "msr_shard", "webdataset_json_member"],
        [
            {
                "webdataset_key": sample_id,
                "msr_shard": "shards/validation/one.tar",
                "webdataset_json_member": f"{sample_id}.json",
            }
        ],
    )
    prediction_manifest = tmp_path / "predictions" / "manifest.csv"
    _write_csv(
        prediction_manifest,
        [
            "sample_id",
            "prediction_height_path",
            "prediction_building_path",
            "classification_valid_mask_path",
            "classification_validity_contract",
        ],
        [
            {
                "sample_id": sample_id,
                "prediction_height_path": str(prediction_height_path),
                "prediction_building_path": str(prediction_building_path),
                "classification_valid_mask_path": str(class_valid_path),
                "classification_validity_contract": "exhaustive_building_annotations",
            }
        ],
    )

    report = SCRIPT.evaluate_prediction_manifest(
        corrected_manifest=corrected_manifest,
        contract_report=contract_report,
        source_index=source_index,
        split_root=split_root,
        prediction_manifest=prediction_manifest,
        split="validation",
    )

    evaluation = report["evaluation"]
    assert report["status"] == "complete_manifest"
    assert evaluation["detection"]["precision"] == pytest.approx(1.0)
    assert evaluation["detection"]["recall"] == pytest.approx(1.0)
    assert evaluation["pixel_height"]["rmse_m"] == pytest.approx(0.0)
    assert evaluation["coverage"]["reference_building_area_m2"] == pytest.approx(
        building.sum() * 4.0
    )
