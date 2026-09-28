"""Fail-closed offline exports for validated GAMUS six-class candidates.

This module deliberately has no API or viewer integration.  It converts an
already-computed candidate prediction into an auditable raster bundle only
after a separate validation manifest authenticates the checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4
import warnings

import numpy as np
import rasterio
from affine import Affine
from rasterio.errors import NotGeoreferencedWarning

from .raster import write_float_raster


CLASS_NAMES = (
    "ground",
    "buildings",
    "water",
    "roads",
    "low_vegetation",
    "trees",
)
INVALID_CLASS_ID = 255
AUTHORIZATION_SCHEMA = "msr.validated_six_class_candidate.v1"
EXPORT_SCHEMA = "msr.offline_six_class_export.v1"


class SixClassExportError(ValueError):
    """Raised when a candidate or export violates the offline contract."""


def file_sha256(path: str | Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Return the SHA-256 identity of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authorize_validated_candidate(
    validation_report_path: str | Path,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Authenticate a checkpoint against a strict completed-validation report.

    The report is intentionally a small, explicit promotion-independent
    contract.  All three evidence gates must be true; a normal training report
    or a checkpoint path alone is insufficient.
    """

    report_path = Path(validation_report_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not report_path.is_file():
        raise SixClassExportError(f"Validation report does not exist: {report_path}")
    if not checkpoint.is_file():
        raise SixClassExportError(f"Candidate checkpoint does not exist: {checkpoint}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise SixClassExportError("Validation report is not valid JSON") from error
    if not isinstance(report, Mapping) or report.get("schema") != AUTHORIZATION_SCHEMA:
        raise SixClassExportError(
            f"Validation report must use schema {AUTHORIZATION_SCHEMA!r}"
        )

    identity = report.get("candidate_checkpoint")
    if not isinstance(identity, Mapping):
        raise SixClassExportError("Validation report has no candidate_checkpoint identity")
    reported_path = identity.get("path")
    reported_hash = identity.get("sha256")
    if not isinstance(reported_path, str) or not isinstance(reported_hash, str):
        raise SixClassExportError("Candidate checkpoint path/SHA-256 are required")
    if Path(reported_path).expanduser().resolve() != checkpoint:
        raise SixClassExportError("Validation report names a different checkpoint")
    actual_hash = file_sha256(checkpoint)
    if reported_hash.lower() != actual_hash:
        raise SixClassExportError("Candidate checkpoint SHA-256 does not match validation")

    gates = report.get("offline_export")
    required_gates = (
        "eligible",
        "six_class_validation_passed",
        "height_regression_guards_passed",
        "unseen_geography_evaluated",
    )
    if not isinstance(gates, Mapping):
        raise SixClassExportError("Validation report has no offline_export gate block")
    failed = [name for name in required_gates if gates.get(name) is not True]
    if failed:
        raise SixClassExportError(
            "Candidate is not eligible for offline export; required true gates: "
            + ", ".join(failed)
        )
    return {
        "path": str(report_path),
        "sha256": file_sha256(report_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_hash,
        "gates": {name: True for name in required_gates},
    }


def _expected_shape(reference_profile: Mapping[str, Any]) -> tuple[int, int]:
    try:
        height = int(reference_profile["height"])
        width = int(reference_profile["width"])
    except (KeyError, TypeError, ValueError) as error:
        raise SixClassExportError("Reference profile needs positive height and width") from error
    if height <= 0 or width <= 0:
        raise SixClassExportError("Reference profile needs positive height and width")
    return height, width


def _checked_mask(
    name: str,
    value: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape:
        raise SixClassExportError(f"{name} shape {mask.shape} != reference grid {shape}")
    if mask.dtype != np.bool_:
        if not np.all(np.isin(mask, (0, 1))):
            raise SixClassExportError(f"{name} must be boolean or contain only 0/1")
        mask = mask.astype(bool)
    return np.ascontiguousarray(mask, dtype=bool)


def _projected_pixel_area_m2(
    reference_profile: Mapping[str, Any],
) -> tuple[float | None, str | None]:
    crs_value = reference_profile.get("crs")
    if crs_value is None:
        return None, "source raster has no CRS"
    try:
        crs = rasterio.crs.CRS.from_user_input(crs_value)
    except Exception:
        return None, "source raster CRS is invalid"
    if not crs.is_projected:
        return None, "source CRS is not projected; angular pixels are not reported as square metres"
    transform = reference_profile.get("transform")
    if transform is None:
        return None, "source raster has no affine transform"
    try:
        affine = transform if isinstance(transform, Affine) else Affine(*transform[:6])
        if affine == Affine.identity():
            return None, "identity transform does not establish a known linear pixel scale"
        _, unit_to_metre = crs.linear_units_factor
        pixel_area = abs(affine.a * affine.e - affine.b * affine.d) * float(
            unit_to_metre
        ) ** 2
    except Exception:
        return None, "projected CRS has no usable linear pixel scale"
    if not math.isfinite(pixel_area) or pixel_area <= 0:
        return None, "projected CRS has a non-positive or non-finite pixel area"
    return float(pixel_area), None


def _height_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "status": "unavailable",
            "height_valid_pixels": 0,
            "reason": "no pixels satisfy image, classification, class and height validity",
        }
    values64 = values.astype(np.float64, copy=False)
    return {
        "status": "available",
        "height_valid_pixels": int(values64.size),
        "mean_m": float(np.mean(values64)),
        "median_m": float(np.median(values64)),
        "minimum_m": float(np.min(values64)),
        "maximum_m": float(np.max(values64)),
        "p05_m": float(np.percentile(values64, 5)),
        "p95_m": float(np.percentile(values64, 95)),
    }


def _clean_uint8_profile(
    reference_profile: Mapping[str, Any],
    *,
    nodata: int | None,
) -> dict[str, Any]:
    profile = dict(reference_profile)
    for key in ("photometric", "blockxsize", "blockysize", "interleave"):
        profile.pop(key, None)
    height, width = _expected_shape(profile)
    tiled = height >= 256 and width >= 256
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        compress="deflate",
        predictor=2,
        tiled=tiled,
    )
    if nodata is None:
        profile.pop("nodata", None)
    else:
        profile["nodata"] = int(nodata)
    if tiled:
        profile.update(blockxsize=256, blockysize=256)
    return profile


def _write_uint8_raster(
    path: Path,
    data: np.ndarray,
    *,
    reference_profile: Mapping[str, Any],
    nodata: int | None,
    tags: Mapping[str, str],
) -> None:
    profile = _clean_uint8_profile(reference_profile, nodata=nodata)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.tmp{path.suffix}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(temporary, "w", **profile) as destination:
                destination.write(np.asarray(data, dtype=np.uint8), 1)
                destination.update_tags(**dict(tags))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_six_class_bundle(
    output_dir: str | Path,
    *,
    fine_semantic_probability_maps: np.ndarray,
    height_map_m: np.ndarray,
    reference_profile: Mapping[str, Any],
    image_valid_mask: np.ndarray,
    classification_valid_mask: np.ndarray,
    height_valid_mask: np.ndarray,
    source_image: str | Path,
    checkpoint_path: str | Path,
    validation_report_path: str | Path,
    model_metadata: Mapping[str, Any] | None = None,
    mask_provenance: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write one immutable, grid-aligned offline six-class export bundle.

    No probability or confidence raster is written.  Height summaries are
    emitted only for buildings, low vegetation and trees, and only over the
    explicit height-valid support.
    """

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    # Re-authenticate here even when a caller already checked before inference.
    # This closes the direct library path and catches checkpoint/report changes
    # between inference and publication.
    authorization = authorize_validated_candidate(
        validation_report_path, checkpoint_path
    )
    shape = _expected_shape(reference_profile)
    probabilities = np.asarray(fine_semantic_probability_maps, dtype=np.float32)
    if probabilities.shape != (len(CLASS_NAMES), *shape):
        raise SixClassExportError(
            "fine_semantic_probability_maps must have shape "
            f"{(len(CLASS_NAMES), *shape)}, received {probabilities.shape}"
        )
    height = np.asarray(height_map_m, dtype=np.float32)
    if height.shape != shape:
        raise SixClassExportError(f"height_map_m shape {height.shape} != {shape}")
    image_valid = _checked_mask("image_valid_mask", image_valid_mask, shape)
    classification_valid = _checked_mask(
        "classification_valid_mask", classification_valid_mask, shape
    )
    height_valid = _checked_mask("height_valid_mask", height_valid_mask, shape)
    if np.any(classification_valid & ~image_valid):
        raise SixClassExportError("classification validity must not escape image validity")
    if np.any(height_valid & ~image_valid):
        raise SixClassExportError("height validity must not escape image validity")
    if not np.all(np.isfinite(probabilities[:, classification_valid])):
        raise SixClassExportError("Six-class values must be finite on classification-valid pixels")
    valid_probabilities = probabilities[:, classification_valid]
    if valid_probabilities.size and (
        np.any(valid_probabilities < -1.0e-6)
        or np.any(valid_probabilities > 1.0 + 1.0e-6)
        or not np.allclose(
            valid_probabilities.sum(axis=0), 1.0, rtol=1.0e-3, atol=1.0e-3
        )
    ):
        raise SixClassExportError(
            "Six-class probability maps must be bounded and sum to one on valid pixels"
        )
    if not np.all(np.isfinite(height[height_valid])):
        raise SixClassExportError("Height values must be finite on height-valid pixels")

    class_ids = np.full(shape, INVALID_CLASS_ID, dtype=np.uint8)
    class_ids[classification_valid] = np.argmax(
        probabilities[:, classification_valid], axis=0
    ).astype(np.uint8)
    exported_height = height.copy()
    exported_height[~height_valid] = np.nan

    pixel_area_m2, area_reason = _projected_pixel_area_m2(reference_profile)
    valid_count = int(np.count_nonzero(classification_valid))
    classes: dict[str, dict[str, Any]] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = int(np.count_nonzero(class_ids == class_id))
        classes[class_name] = {
            "class_id": class_id,
            "pixel_count": count,
            "pixel_share": float(count / valid_count) if valid_count else None,
            "coverage_m2": (
                float(count * pixel_area_m2) if pixel_area_m2 is not None else None
            ),
        }

    summary_support = classification_valid & height_valid
    height_summaries = {
        class_name: _height_summary(height[summary_support & (class_ids == class_id)])
        for class_id, class_name in ((1, "buildings"), (4, "low_vegetation"), (5, "trees"))
    }
    area = (
        {
            "status": "available",
            "pixel_area_m2": pixel_area_m2,
            "classified_coverage_m2": float(valid_count * pixel_area_m2),
            "reason": None,
        }
        if pixel_area_m2 is not None
        else {
            "status": "unavailable",
            "pixel_area_m2": None,
            "classified_coverage_m2": None,
            "reason": area_reason,
        }
    )

    source = Path(source_image).expanduser().resolve()
    staging = output.with_name(f".{output.name}.{uuid4().hex}.staging")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        class_path = staging / "six_class_ids.tif"
        height_path = staging / "height_m.tif"
        image_mask_path = staging / "image_valid_mask.tif"
        class_mask_path = staging / "classification_valid_mask.tif"
        height_mask_path = staging / "height_valid_mask.tif"
        class_tags = {
            "MSR_SCHEMA": EXPORT_SCHEMA,
            "MSR_INVALID_CLASS_ID": str(INVALID_CLASS_ID),
            **{
                f"MSR_CLASS_{class_id}": class_name
                for class_id, class_name in enumerate(CLASS_NAMES)
            },
        }
        _write_uint8_raster(
            class_path,
            class_ids,
            reference_profile=reference_profile,
            nodata=INVALID_CLASS_ID,
            tags=class_tags,
        )
        write_float_raster(
            height_path,
            exported_height,
            reference_profile=dict(reference_profile),
            tags={
                "MSR_SCHEMA": EXPORT_SCHEMA,
                "MSR_ROLE": "predicted_surface_height_m",
            },
        )
        for path, mask, role in (
            (image_mask_path, image_valid, "image_valid_mask"),
            (class_mask_path, classification_valid, "classification_valid_mask"),
            (height_mask_path, height_valid, "height_valid_mask"),
        ):
            _write_uint8_raster(
                path,
                mask,
                reference_profile=reference_profile,
                nodata=None,
                tags={
                    "MSR_SCHEMA": EXPORT_SCHEMA,
                    "MSR_ROLE": role,
                    "MSR_TRUE_VALUE": "1",
                    "MSR_FALSE_VALUE": "0",
                },
            )

        artifacts = {
            path.name: {"sha256": file_sha256(path)}
            for path in (
                class_path,
                height_path,
                image_mask_path,
                class_mask_path,
                height_mask_path,
            )
        }
        summary: dict[str, Any] = {
            "schema": EXPORT_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "offline_validated_candidate_export",
            "app_active_model_changed": False,
            "source_image": str(source),
            "grid": {
                "height": shape[0],
                "width": shape[1],
                "crs": (
                    str(reference_profile.get("crs"))
                    if reference_profile.get("crs") is not None
                    else None
                ),
                "transform": (
                    list(reference_profile["transform"][:6])
                    if reference_profile.get("transform") is not None
                    else None
                ),
            },
            "checkpoint": {
                "path": authorization["checkpoint_path"],
                "sha256": authorization["checkpoint_sha256"],
                **{
                    key: (model_metadata or {}).get(key)
                    for key in ("epoch", "model_type", "fine_semantic_head_type")
                },
            },
            "validation": {
                "path": authorization["path"],
                "sha256": authorization["sha256"],
                "gates": authorization["gates"],
            },
            "class_names": list(CLASS_NAMES),
            "invalid_class_id": INVALID_CLASS_ID,
            "decision_output": "categorical_argmax_only",
            "probability_or_confidence_exported": False,
            "validity": {
                "image_valid_pixels": int(np.count_nonzero(image_valid)),
                "classification_valid_pixels": valid_count,
                "height_valid_pixels": int(np.count_nonzero(height_valid)),
                "masks_are_independent_artifacts": True,
                "provenance": dict(mask_provenance or {}),
            },
            "classes": classes,
            "height_summaries_m": height_summaries,
            "height_summary_classes": ["buildings", "low_vegetation", "trees"],
            "water_and_road_height_reported": False,
            "area": area,
            "artifacts": artifacts,
        }
        _atomic_json(staging / "summary.json", summary)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "CLASS_NAMES",
    "EXPORT_SCHEMA",
    "INVALID_CLASS_ID",
    "SixClassExportError",
    "authorize_validated_candidate",
    "export_six_class_bundle",
    "file_sha256",
]
