"""Fail-closed HighBuild label-contract helpers.

HighBuild's exported ``building_height_m.tif`` is a sparse building-label
raster, not a complete nDSM.  COCO footprints provide semantic building
support.  A separate validity mask must state which annotated heights are
eligible for regression so the two protocols used by Monocular Surface Reconstruction remain
explicit and reproducible.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from affine import Affine
from rasterio.features import rasterize


HIGHBUILD_SURFACE_NAME = "building_height_m.tif"
HEIGHT_PROTOCOLS = ("strict_measured", "inclusive_annotated")


def is_highbuild_building_height(path: str | Path) -> bool:
    """Recognize the project's known sparse HighBuild export by basename."""

    return Path(path).name.lower() == HIGHBUILD_SURFACE_NAME


def validate_highbuild_manifest_contract(
    *,
    sample_id: str,
    surface_path: Path,
    target_kind: str,
    building_mask_path: Path | None,
    valid_mask_path: Path | None,
) -> None:
    """Reject ambiguous HighBuild labels before a dataset can be constructed.

    The semantic footprint and regression-support masks must be explicit,
    distinct files.  Reusing the sparse height raster as either mask would
    silently turn unknown background into ground and cannot distinguish
    measured-only from inclusive annotated-height evaluation.
    """

    if not is_highbuild_building_height(surface_path):
        return
    if target_kind != "building_height":
        raise ValueError(
            f"HighBuild sample {sample_id} uses {HIGHBUILD_SURFACE_NAME} and must "
            "declare target_kind='building_height'; it is not a complete nDSM"
        )
    if building_mask_path is None or valid_mask_path is None:
        raise ValueError(
            f"HighBuild sample {sample_id} requires explicit building_mask_path "
            "and valid_mask_path provenance"
        )
    paths = {
        "surface_path": surface_path.resolve(),
        "building_mask_path": building_mask_path.resolve(),
        "valid_mask_path": valid_mask_path.resolve(),
    }
    if len(set(paths.values())) != len(paths):
        raise ValueError(
            f"HighBuild sample {sample_id} requires distinct surface, semantic "
            "building-mask, and regression-validity files"
        )


def _annotation_polygons(annotation: dict[str, Any]) -> Iterable[dict[str, Any]]:
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list):
        raise ValueError("HighBuild annotation segmentation must be polygon lists")
    for flat in segmentation:
        if not isinstance(flat, list) or len(flat) < 6 or len(flat) % 2:
            raise ValueError("HighBuild polygon must contain at least three xy pairs")
        coordinates = [float(value) for value in flat]
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("HighBuild polygon contains a non-finite coordinate")
        ring = [
            [coordinates[index], coordinates[index + 1]]
            for index in range(0, len(coordinates), 2)
        ]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        yield {"type": "Polygon", "coordinates": [ring]}


def annotation_masks(
    coco: dict[str, Any], *, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Rasterize all building footprints and measured-height footprints."""

    images = coco.get("images")
    annotations = coco.get("annotations")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(annotations, list):
        raise ValueError("HighBuild COCO tile must contain one image and annotations")
    if int(images[0].get("height", -1)) != height or int(images[0].get("width", -1)) != width:
        raise ValueError("HighBuild COCO and height raster dimensions differ")

    all_shapes: list[tuple[dict[str, Any], int]] = []
    measured_shapes: list[tuple[dict[str, Any], int]] = []
    estimated = 0
    rejected_height = 0
    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)) != 0:
            raise ValueError("HighBuild crowd/RLE annotations are unsupported")
        attributes = annotation.get("attributes") or {}
        height_value = attributes.get("height")
        measured = (
            not bool(attributes.get("is_estimated_height", False))
            and isinstance(height_value, (int, float))
            and math.isfinite(float(height_value))
            and float(height_value) > 0.0
        )
        if bool(attributes.get("is_estimated_height", False)):
            estimated += 1
        elif not measured:
            rejected_height += 1
        for polygon in _annotation_polygons(annotation):
            all_shapes.append((polygon, 1))
            if measured:
                measured_shapes.append((polygon, 1))

    kwargs = {
        "out_shape": (height, width),
        "transform": Affine.identity(),
        "fill": 0,
        "dtype": "uint8",
        "all_touched": False,
    }
    empty = np.zeros((height, width), dtype=np.uint8)
    all_mask = rasterize(all_shapes, **kwargs) if all_shapes else empty.copy()
    measured_mask = (
        rasterize(measured_shapes, **kwargs) if measured_shapes else empty.copy()
    )
    return all_mask, measured_mask, {
        "annotations": len(annotations),
        "estimated_annotations": estimated,
        "rejected_nonpositive_or_missing_height_annotations": rejected_height,
        "trusted_annotations": len(annotations) - estimated - rejected_height,
    }


def regression_support_mask(
    *,
    all_footprints: np.ndarray,
    measured_footprints: np.ndarray,
    target: np.ndarray,
    nodata: float | None,
    protocol: str,
) -> np.ndarray:
    """Resolve explicit height support for a named HighBuild protocol."""

    if protocol not in HEIGHT_PROTOCOLS:
        raise ValueError(
            f"HighBuild height protocol must be one of {list(HEIGHT_PROTOCOLS)}, "
            f"got {protocol!r}"
        )
    if all_footprints.shape != target.shape or measured_footprints.shape != target.shape:
        raise ValueError("HighBuild masks and height raster dimensions differ")
    polygon_support = (
        measured_footprints if protocol == "strict_measured" else all_footprints
    ).astype(bool)
    support = polygon_support & np.isfinite(target) & (target > 0.0)
    if nodata is not None:
        support &= ~np.isclose(target, nodata, equal_nan=True)
    return support.astype(np.uint8)
