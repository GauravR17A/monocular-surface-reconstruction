from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "data"
    / "export_highbuild_surface_subset.py"
)
SPEC = importlib.util.spec_from_file_location("export_highbuild_surface_subset_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_target(path: Path) -> None:
    target = np.zeros((8, 8), dtype=np.float32)
    target[1:4, 1:4] = 4.0
    target[4:7, 4:7] = 9.0
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32643",
        "transform": Affine.identity(),
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(target, 1)


def _annotation(identifier: int, polygon: list[float], *, estimated: bool) -> dict:
    return {
        "id": identifier,
        "image_id": 1,
        "category_id": 1,
        "segmentation": [polygon],
        "iscrowd": 0,
        "attributes": {
            "height": 9.0 if estimated else 4.0,
            "is_estimated_height": estimated,
        },
    }


def test_exporter_writes_distinct_semantic_and_strict_height_masks(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "width": 8, "height": 8}],
                "annotations": [
                    _annotation(1, [1, 1, 4, 1, 4, 4, 1, 4], estimated=False),
                    _annotation(2, [4, 4, 7, 4, 7, 7, 4, 7], estimated=True),
                ],
            }
        ),
        encoding="utf-8",
    )
    surface_path = tmp_path / "building_height_m.tif"
    _write_target(surface_path)
    building_path = tmp_path / "building_footprints.tif"
    valid_path = tmp_path / "measured_height_valid.tif"

    audit = MODULE._build_contract_masks(
        annotation_path=annotation_path,
        surface_path=surface_path,
        building_mask_path=building_path,
        valid_mask_path=valid_path,
        height_protocol="strict_measured",
    )

    with rasterio.open(building_path) as source:
        buildings = source.read(1)
    with rasterio.open(valid_path) as source:
        valid = source.read(1)
    assert int(buildings.sum()) > int(valid.sum()) > 0
    assert audit["estimated_annotations"] == 1
    assert audit["semantic_building_pixels"] == int(buildings.sum())
    assert audit["regression_height_pixels"] == int(valid.sum())
