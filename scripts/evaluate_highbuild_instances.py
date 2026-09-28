"""Evaluate dense predictions against corrected HighBuild objects and heights.

The prediction manifest is intentionally model-agnostic so the same evaluator
can compare the production model, the height-focused pilot and an expanded
six-class candidate.  Required columns are ``sample_id`` and
``prediction_height_path``, plus exactly one of
``prediction_building_path`` (probability/binary raster) or
``prediction_instance_path`` (positive integer instance labels).

Optional ``image_valid_mask_path`` records optical/prediction validity.
Optional ``classification_valid_mask_path`` may only be supplied when it
authenticates exhaustive labels over that area; without it, precision and
building-count error are reported unavailable rather than treating unknown
HighBuild background as a negative class.

Building-count metrics are emitted in a dedicated ``counting`` section. They
are computed from instance objects per scene, not inferred from pixel accuracy,
and remain unavailable for the current positive-only HighBuild annotations.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile
from typing import Any
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

from msr.data.highbuild import validate_highbuild_manifest_contract
from msr.data.surface_dataset import load_surface_manifest
from msr.evaluation.highbuild_instances import (
    HighBuildInstanceAccumulator,
    HighBuildInstanceEvaluationError,
)


SCHEMA = "msr.highbuild_instance_evaluation.v1"
CORRECTED_CONTRACT_SCHEMA = "msr.highbuild_annotation_contract.v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_hash(name: str, actual: str, expected: str | None) -> None:
    if expected is not None and actual.lower() != expected.lower():
        raise HighBuildInstanceEvaluationError(
            f"{name} hash mismatch: expected {expected}, found {actual}"
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise HighBuildInstanceEvaluationError(f"CSV has no header: {path}")
        return list(reader)


def _resolve_path(raw: str, *, relative_to: Path) -> Path:
    value = str(raw).strip()
    if not value:
        raise HighBuildInstanceEvaluationError("Required raster path is blank")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_band(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            if source.count != 1:
                raise HighBuildInstanceEvaluationError(
                    f"Expected one raster band at {path}, found {source.count}"
                )
            values = source.read(1)
            valid = source.read_masks(1) > 0
            profile = {
                "shape": source.shape,
                "crs": source.crs,
                "transform": source.transform,
            }
    return values, valid, profile


def _read_image_validity(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read optical validity independently from class and height labels."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            if source.count < 1:
                raise HighBuildInstanceEvaluationError(
                    f"Optical raster has no bands: {path}"
                )
            valid = np.all(source.read_masks() > 0, axis=0)
            profile = {
                "shape": source.shape,
                "crs": source.crs,
                "transform": source.transform,
            }
    return valid, profile


def _same_grid(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["shape"] == right["shape"]
        and left["crs"] == right["crs"]
        and np.allclose(tuple(left["transform"]), tuple(right["transform"]))
    )


def _require_same_grid(
    reference: dict[str, Any], candidate: dict[str, Any], *, name: str
) -> None:
    if not _same_grid(reference, candidate):
        raise HighBuildInstanceEvaluationError(
            f"{name} is not on the exact corrected HighBuild grid"
        )


class _CocoShardStore:
    def __init__(
        self, *, source_index: Path, split_root: Path, expected_split: str | None = None
    ) -> None:
        self.source_index = source_index
        self.split_root = split_root
        self._rows: dict[str, dict[str, str]] = {}
        self._archives: dict[Path, tarfile.TarFile] = {}
        required = {
            "webdataset_key",
            "msr_shard",
            "webdataset_json_member",
        }
        for row in _read_csv(source_index):
            if not required.issubset(row):
                raise HighBuildInstanceEvaluationError(
                    f"HighBuild source index lacks columns {sorted(required - set(row))}"
                )
            sample_id = row["webdataset_key"].strip()
            if not sample_id or sample_id in self._rows:
                raise HighBuildInstanceEvaluationError(
                    f"Blank or duplicate sample ID in HighBuild source index: {sample_id!r}"
                )
            declared_split = row.get("msr_split", "").strip()
            if expected_split is not None and declared_split and declared_split != expected_split:
                raise HighBuildInstanceEvaluationError(
                    f"HighBuild source row {sample_id} belongs to {declared_split!r}, "
                    f"not requested split {expected_split!r}"
                )
            self._rows[sample_id] = row

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __enter__(self) -> "_CocoShardStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def load(self, sample_id: str) -> dict[str, Any]:
        row = self._rows.get(sample_id)
        if row is None:
            raise HighBuildInstanceEvaluationError(
                f"HighBuild COCO provenance is missing for {sample_id}"
            )
        archive_path = _resolve_path(
            row["msr_shard"], relative_to=self.split_root
        )
        archive = self._archives.get(archive_path)
        if archive is None:
            archive = tarfile.open(archive_path, "r")
            self._archives[archive_path] = archive
        member_name = row["webdataset_json_member"].strip()
        if not member_name:
            raise HighBuildInstanceEvaluationError(
                f"HighBuild COCO member is blank for {sample_id}"
            )
        member = archive.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"{archive_path}!{member_name}")
        payload = json.load(member)
        if not isinstance(payload, dict):
            raise HighBuildInstanceEvaluationError(
                f"HighBuild COCO member is not an object for {sample_id}"
            )
        return payload


def _load_and_verify_contract(
    *, report_path: Path, manifest_path: Path, split: str
) -> tuple[str, dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema") != CORRECTED_CONTRACT_SCHEMA:
        raise HighBuildInstanceEvaluationError(
            "Unsupported or missing corrected HighBuild contract report"
        )
    if report.get("historical_artifacts_modified") is not False:
        raise HighBuildInstanceEvaluationError(
            "Corrected HighBuild contract does not certify immutable historical artifacts"
        )
    protocol = report.get("height_protocol")
    if protocol not in {"strict_measured", "inclusive_annotated"}:
        raise HighBuildInstanceEvaluationError(
            "Corrected HighBuild report has no recognized height protocol"
        )
    splits = report.get("splits")
    split_report = splits.get(split) if isinstance(splits, dict) else None
    if not isinstance(split_report, dict):
        raise HighBuildInstanceEvaluationError(
            f"Corrected HighBuild report has no {split!r} split"
        )
    expected_manifest_hash = split_report.get("output_manifest_sha256")
    actual_manifest_hash = file_sha256(manifest_path)
    if not isinstance(expected_manifest_hash, str) or (
        actual_manifest_hash.lower() != expected_manifest_hash.lower()
    ):
        raise HighBuildInstanceEvaluationError(
            "Corrected manifest bytes do not match its immutable contract report"
        )
    return protocol, report


def _prediction_rows(
    path: Path,
) -> dict[str, dict[str, str]]:
    required = {"sample_id", "prediction_height_path"}
    result: dict[str, dict[str, str]] = {}
    for row in _read_csv(path):
        missing = required - set(row)
        if missing:
            raise HighBuildInstanceEvaluationError(
                f"Prediction manifest lacks columns {sorted(missing)}"
            )
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in result:
            raise HighBuildInstanceEvaluationError(
                f"Blank or duplicate prediction sample ID: {sample_id!r}"
            )
        building_path = row.get("prediction_building_path", "").strip()
        instance_path = row.get("prediction_instance_path", "").strip()
        if bool(building_path) == bool(instance_path):
            raise HighBuildInstanceEvaluationError(
                f"Prediction {sample_id} must supply exactly one of "
                "prediction_building_path or prediction_instance_path"
            )
        result[sample_id] = row
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def evaluate_prediction_manifest(
    *,
    corrected_manifest: Path,
    contract_report: Path,
    source_index: Path,
    split_root: Path,
    prediction_manifest: Path,
    split: str,
    allow_subset: bool = False,
    building_probability_threshold: float = 0.5,
    instance_iou_threshold: float = 0.5,
    minimum_predicted_instance_pixels: int = 1,
    minimum_height_pixels: int = 1,
    boundary_tolerance_pixels: int = 2,
) -> dict[str, Any]:
    corrected_manifest = corrected_manifest.resolve()
    contract_report = contract_report.resolve()
    source_index = source_index.resolve()
    split_root = split_root.resolve()
    prediction_manifest = prediction_manifest.resolve()
    protocol, contract = _load_and_verify_contract(
        report_path=contract_report,
        manifest_path=corrected_manifest,
        split=split,
    )
    all_records = load_surface_manifest(corrected_manifest)
    records = [
        record
        for record in all_records
        if record.target_kind == "building_height"
    ]
    if not records:
        raise HighBuildInstanceEvaluationError(
            "Corrected manifest contains no HighBuild building-height records"
        )
    ids = [record.sample_id for record in records]
    if len(set(ids)) != len(ids):
        raise HighBuildInstanceEvaluationError(
            "Corrected manifest contains duplicate HighBuild sample IDs"
        )
    split_contract = contract["splits"][split]
    expected_highbuild_rows = split_contract.get("highbuild_rows")
    if expected_highbuild_rows is not None and int(expected_highbuild_rows) != len(ids):
        raise HighBuildInstanceEvaluationError(
            "Corrected manifest HighBuild row count disagrees with its contract report"
        )
    predictions = _prediction_rows(prediction_manifest)
    expected_ids = set(ids)
    prediction_ids = set(predictions)
    extras = prediction_ids - expected_ids
    missing = expected_ids - prediction_ids
    if extras:
        raise HighBuildInstanceEvaluationError(
            f"Prediction manifest contains unknown samples: {sorted(extras)[:5]}"
        )
    if missing and not allow_subset:
        raise HighBuildInstanceEvaluationError(
            f"Prediction manifest is missing {len(missing)} HighBuild samples; "
            "use --allow-subset only for an explicitly labelled development run"
        )
    records = [record for record in records if record.sample_id in predictions]
    if not records:
        raise HighBuildInstanceEvaluationError("No requested HighBuild samples remain")

    accumulator = HighBuildInstanceAccumulator(height_protocol=protocol)
    validity_sources: set[str] = set()
    input_hashes: dict[str, dict[str, str]] = {}
    prediction_parent = prediction_manifest.parent
    with _CocoShardStore(
        source_index=source_index,
        split_root=split_root,
        expected_split=split,
    ) as coco_store:
        for record in records:
            validate_highbuild_manifest_contract(
                sample_id=record.sample_id,
                surface_path=record.surface_path,
                target_kind=record.target_kind,
                building_mask_path=record.building_mask_path,
                valid_mask_path=record.valid_mask_path,
            )
            if record.building_mask_path is None or record.valid_mask_path is None:
                raise HighBuildInstanceEvaluationError(
                    f"Corrected masks are missing for {record.sample_id}"
                )
            prediction_row = predictions[record.sample_id]
            prediction_height_path = _resolve_path(
                prediction_row["prediction_height_path"], relative_to=prediction_parent
            )
            instance_value = prediction_row.get("prediction_instance_path", "").strip()
            prediction_building_path = _resolve_path(
                instance_value
                or prediction_row.get("prediction_building_path", ""),
                relative_to=prediction_parent,
            )
            reference_height, _, reference_profile = _read_band(record.surface_path)
            building_mask, _, building_profile = _read_band(record.building_mask_path)
            height_valid, _, height_valid_profile = _read_band(record.valid_mask_path)
            prediction_height, _, prediction_height_profile = _read_band(
                prediction_height_path
            )
            prediction_building, _, prediction_building_profile = _read_band(
                prediction_building_path
            )
            for name, profile in (
                ("corrected building mask", building_profile),
                ("corrected height-validity mask", height_valid_profile),
                ("predicted height", prediction_height_profile),
                ("predicted buildings", prediction_building_profile),
            ):
                _require_same_grid(reference_profile, profile, name=name)

            raw_image_valid_path = prediction_row.get("image_valid_mask_path", "").strip()
            if raw_image_valid_path:
                image_valid_path = _resolve_path(
                    raw_image_valid_path, relative_to=prediction_parent
                )
                image_valid, _, image_valid_profile = _read_band(image_valid_path)
                _require_same_grid(
                    reference_profile, image_valid_profile, name="image-validity mask"
                )
                image_valid_mask = image_valid > 0
                validity_sources.add("explicit_image_valid_mask")
            else:
                image_valid_mask, source_rgb_profile = _read_image_validity(
                    record.rgb_path
                )
                _require_same_grid(
                    reference_profile,
                    source_rgb_profile,
                    name="source RGB image-validity grid",
                )
                image_valid_path = record.rgb_path
                validity_sources.add("source_rgb_band_masks")

            raw_classification_valid = prediction_row.get(
                "classification_valid_mask_path", ""
            ).strip()
            classification_contract = prediction_row.get(
                "classification_validity_contract", ""
            ).strip()
            classification_valid_mask = None
            classification_valid_path = None
            if raw_classification_valid:
                if classification_contract != "exhaustive_building_annotations":
                    raise HighBuildInstanceEvaluationError(
                        f"{record.sample_id} supplies a classification-validity mask "
                        "without classification_validity_contract="
                        "'exhaustive_building_annotations'"
                    )
                classification_valid_path = _resolve_path(
                    raw_classification_valid, relative_to=prediction_parent
                )
                classification_valid, _, classification_profile = _read_band(
                    classification_valid_path
                )
                _require_same_grid(
                    reference_profile,
                    classification_profile,
                    name="classification-validity mask",
                )
                classification_valid_mask = classification_valid > 0
            elif classification_contract:
                raise HighBuildInstanceEvaluationError(
                    f"{record.sample_id} declares classification validity without its mask"
                )

            coco = coco_store.load(record.sample_id)
            accumulator.update(
                sample_id=record.sample_id,
                prediction_height_m=prediction_height,
                prediction_building=prediction_building,
                coco=coco,
                reference_height_m=reference_height,
                building_footprints=building_mask > 0,
                height_valid_mask=height_valid > 0,
                image_valid_mask=image_valid_mask,
                classification_valid_mask=classification_valid_mask,
                height_protocol=protocol,
                prediction_is_instance_labels=bool(instance_value),
                building_probability_threshold=building_probability_threshold,
                instance_iou_threshold=instance_iou_threshold,
                minimum_predicted_instance_pixels=minimum_predicted_instance_pixels,
                minimum_height_pixels=minimum_height_pixels,
                boundary_tolerance_pixels=boundary_tolerance_pixels,
                crs=reference_profile["crs"],
                transform=reference_profile["transform"],
            )
            paths = {
                "source_rgb": record.rgb_path,
                "reference_height": record.surface_path,
                "building_footprints": record.building_mask_path,
                "height_validity": record.valid_mask_path,
                "prediction_height": prediction_height_path,
                "prediction_building_or_instances": prediction_building_path,
            }
            if image_valid_path is not None:
                paths["image_validity"] = image_valid_path
            if classification_valid_path is not None:
                paths["classification_validity"] = classification_valid_path
            input_hashes[record.sample_id] = {
                name: file_sha256(path) for name, path in paths.items()
            }
            input_hashes[record.sample_id]["coco_annotation_canonical_json"] = (
                hashlib.sha256(
                    json.dumps(
                        coco, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
            )

    summary = accumulator.compute()
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_subset" if missing else "complete_manifest",
        "split": split,
        "test_role": contract.get("test_role"),
        "inputs": {
            "corrected_manifest": str(corrected_manifest),
            "corrected_manifest_sha256": file_sha256(corrected_manifest),
            "corrected_contract_report": str(contract_report),
            "corrected_contract_report_sha256": file_sha256(contract_report),
            "source_index": str(source_index),
            "source_index_sha256": file_sha256(source_index),
            "prediction_manifest": str(prediction_manifest),
            "prediction_manifest_sha256": file_sha256(prediction_manifest),
            "full_highbuild_sample_count": len(ids),
            "evaluated_highbuild_sample_count": len(records),
            "omitted_sample_count": len(missing),
            "image_validity_sources": sorted(validity_sources),
            "file_hashes": input_hashes,
        },
        "interpretation": {
            "height_bias": "prediction_minus_reference_metres",
            "outside_building_footprints": "unknown_not_zero_height_ground",
            "detection_precision_and_count": (
                "available only when an independent exhaustive classification-validity "
                "mask is supplied for every tile"
            ),
            "counting": (
                "separate instance-level evaluation; never inferred from pixel metrics"
            ),
            "estimated_heights": "never merged silently with measured references",
            "square_metres": (
                "reported only when every reference grid has a projected CRS with "
                "authenticated linear units"
            ),
        },
        "evaluation": summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract-report", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--building-threshold", type=float, default=0.5)
    parser.add_argument("--instance-iou-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-predicted-instance-pixels", type=int, default=1)
    parser.add_argument("--minimum-height-pixels", type=int, default=1)
    parser.add_argument("--boundary-tolerance-pixels", type=int, default=2)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-contract-report-sha256")
    parser.add_argument("--expected-source-index-sha256")
    parser.add_argument("--expected-prediction-manifest-sha256")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    inputs = {
        "manifest": args.manifest.expanduser().resolve(),
        "contract report": args.contract_report.expanduser().resolve(),
        "source index": args.source_index.expanduser().resolve(),
        "prediction manifest": args.prediction_manifest.expanduser().resolve(),
    }
    expected = {
        "manifest": args.expected_manifest_sha256,
        "contract report": args.expected_contract_report_sha256,
        "source index": args.expected_source_index_sha256,
        "prediction manifest": args.expected_prediction_manifest_sha256,
    }
    for name, path in inputs.items():
        _check_hash(name, file_sha256(path), expected[name])
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".json":
        raise HighBuildInstanceEvaluationError("Output report must be a .json file")
    if output in inputs.values():
        raise HighBuildInstanceEvaluationError("Output may not overwrite an input artifact")
    report = evaluate_prediction_manifest(
        corrected_manifest=inputs["manifest"],
        contract_report=inputs["contract report"],
        source_index=inputs["source index"],
        split_root=args.split_root.expanduser().resolve(),
        prediction_manifest=inputs["prediction manifest"],
        split=args.split,
        allow_subset=args.allow_subset,
        building_probability_threshold=args.building_threshold,
        instance_iou_threshold=args.instance_iou_threshold,
        minimum_predicted_instance_pixels=args.minimum_predicted_instance_pixels,
        minimum_height_pixels=args.minimum_height_pixels,
        boundary_tolerance_pixels=args.boundary_tolerance_pixels,
    )
    _atomic_json(output, report)
    print(json.dumps(report["evaluation"], indent=2, sort_keys=True))
    print(f"Saved immutable report: {output}")


if __name__ == "__main__":
    main()
