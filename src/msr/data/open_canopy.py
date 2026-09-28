"""Small, auditable helpers for the licensed Open-Canopy dataset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote

import numpy as np


OPEN_CANOPY_DATASET_URL = "https://huggingface.co/datasets/AI4Forest/Open-Canopy"
OPEN_CANOPY_RESOLVE_ROOT = f"{OPEN_CANOPY_DATASET_URL}/resolve/main/canopy_height"
OPEN_CANOPY_LICENSE = "Etalab Open Licence 2.0"
OPEN_CANOPY_CRS = "EPSG:2154"
OPEN_CANOPY_STORED_UNIT_M = 0.1

# IGN LiDAR HD classes used by Open-Canopy. Buildings and unclassified pixels are
# intentionally excluded from a forest expert so they cannot poison its gate.
FOREST_VALID_CLASSES = frozenset({2, 3, 4, 5, 9})
VEGETATION_CLASSES = frozenset({3, 4, 5})

_IMAGE_DATE_PATTERN = re.compile(r"pansharpened_(\d{8})")


@dataclass(frozen=True)
class OpenCanopyFeature:
    """One official 1 km Open-Canopy split cell."""

    sample_id: str
    split: str
    year: int
    image_name: str
    bounds: tuple[float, float, float, float]
    region: str
    lidar_points: int
    lidar_acquisition_date: str | None = None
    imagery_acquisition_date: str | None = None
    lidar_url: str | None = None


def _iso_acquisition_date(value: object, *, field: str) -> str | None:
    """Normalize an optional official YYYYMMDD acquisition date."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y%m%d").date().isoformat()
    except ValueError as error:
        raise ValueError(f"Invalid Open-Canopy {field} date {text!r}") from error


def imagery_acquisition_date(image_name: str) -> str | None:
    """Recover the SPOT acquisition date encoded in an official image name."""

    match = _IMAGE_DATE_PATTERN.search(str(image_name))
    if match is None:
        return None
    return _iso_acquisition_date(match.group(1), field="imagery")


def _polygon_bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    if geometry.get("type") != "Polygon":
        raise ValueError(f"Expected Polygon geometry, got {geometry.get('type')!r}")
    rings = geometry.get("coordinates") or []
    if not rings or not rings[0]:
        raise ValueError("Open-Canopy feature has an empty polygon")
    coordinates = rings[0]
    xs = [float(point[0]) for point in coordinates]
    ys = [float(point[1]) for point in coordinates]
    return min(xs), min(ys), max(xs), max(ys)


def _region_key(bounds: tuple[float, float, float, float], size_m: int = 5_000) -> str:
    """Return a coarse spatial group used to reject nearby split leakage."""

    left, bottom, right, top = bounds
    centre_x = (left + right) / 2
    centre_y = (bottom + top) / 2
    return f"open_canopy_r{int(centre_x // size_m)}_{int(centre_y // size_m)}"


def load_open_canopy_features(path: str | Path) -> list[OpenCanopyFeature]:
    """Load the official GeoJSON without adding a GeoPandas dependency."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        collection = json.load(handle)
    features: list[OpenCanopyFeature] = []
    for item in collection.get("features", []):
        properties = item.get("properties") or {}
        split = str(properties.get("split", "")).strip().lower()
        if split not in {"train", "val", "test", "buffer"}:
            continue
        year = int(properties["lidar_year"])
        image_name = str(properties["image_name"])
        lidar_date = _iso_acquisition_date(
            properties.get("lidar_acquisition_date"), field="LiDAR"
        )
        image_date = imagery_acquisition_date(image_name)
        bounds = _polygon_bounds(item["geometry"])
        x = int(round(float(properties.get("X", bounds[0]))))
        y = int(round(float(properties.get("Y", bounds[1]))))
        sample_id = f"oc_{year}_{x}_{y}"
        features.append(
            OpenCanopyFeature(
                sample_id=sample_id,
                split=split,
                year=year,
                image_name=image_name,
                bounds=bounds,
                region=_region_key(bounds),
                lidar_points=int(properties.get("n_lidar_points") or 0),
                lidar_acquisition_date=lidar_date,
                imagery_acquisition_date=image_date,
                lidar_url=(
                    str(properties["lidar_url"])
                    if properties.get("lidar_url")
                    else None
                ),
            )
        )
    if not features:
        raise ValueError(f"No Open-Canopy features found in {source}")
    return features


def open_canopy_urls(feature: OpenCanopyFeature) -> dict[str, str]:
    """Build the three official remote Cloud-Optimized GeoTIFF URLs."""

    image_name = feature.image_name
    lidar_name = image_name.replace("pansharpened", "lidar")
    class_name = image_name.replace("pansharpened", "lidar_classification")
    year_root = f"{OPEN_CANOPY_RESOLVE_ROOT}/{feature.year}"
    return {
        "rgb": f"{year_root}/spot/{quote(image_name)}",
        "canopy": f"{year_root}/lidar/{quote(lidar_name)}",
        "classification": (
            f"{year_root}/lidar_classification/{quote(class_name)}"
        ),
    }


def stored_canopy_to_metres(values: np.ndarray) -> np.ndarray:
    """Convert the released uint16 decimetre CHM to float32 metres."""

    return values.astype(np.float32) * OPEN_CANOPY_STORED_UNIT_M


def forest_semantic_masks(
    classification: np.ndarray, source_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid forest supervision and its vegetation subset."""

    classification = np.asarray(classification)
    source_valid = np.asarray(source_valid, dtype=bool)
    valid = source_valid & np.isin(classification, tuple(FOREST_VALID_CLASSES))
    vegetation = valid & np.isin(classification, tuple(VEGETATION_CLASSES))
    return valid, vegetation
