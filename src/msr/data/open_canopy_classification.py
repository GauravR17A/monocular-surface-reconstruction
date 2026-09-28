"""Lossless sidecar support for Open-Canopy LiDAR classification rasters.

The prepared Open-Canopy manifests intentionally remain binary-height manifests.
This module preserves the aligned upstream class identifiers in an optional,
versioned sidecar without changing those manifests or their existing masks.
"""

from __future__ import annotations

from collections import Counter
from typing import Final

import numpy as np


OPEN_CANOPY_CLASSIFICATION_SIDECAR_SCHEMA: Final = (
    "msr.open_canopy.lidar_classification_sidecar.v1"
)
OPEN_CANOPY_CLASSIFICATION_NODATA: Final = 255

# Open-Canopy exposes IGN LiDAR HD classification rasters.  These identifiers
# and names follow the IGN LiDAR HD classification table.  Other upstream byte
# identifiers are deliberately preserved but remain unnamed/unsupported here.
IGN_LIDAR_HD_CLASS_NAMES: Final[dict[int, str]] = {
    0: "never_classified",
    1: "unclassified",
    2: "ground",
    3: "low_vegetation",
    4: "medium_vegetation",
    5: "high_vegetation",
    6: "building",
    9: "water",
    17: "bridge_deck",
    64: "permanent_above_ground",
    65: "artifact",
    66: "synthetic",
    67: "miscellaneous_built",
}
RICH_VEGETATION_CLASS_IDS: Final = frozenset({3, 4, 5})


def preserve_classification_ids(
    values: np.ndarray,
    source_valid: np.ndarray,
) -> np.ndarray:
    """Return a uint8 copy with invalid pixels encoded as sidecar nodata.

    Valid upstream identifiers are copied exactly.  Float inputs are accepted
    only when every source-valid value is finite, integral, and in 0..254;
    malformed values fail closed rather than being silently rounded.
    """

    raw = np.asarray(values)
    valid = np.asarray(source_valid, dtype=bool)
    if raw.ndim != 2 or valid.shape != raw.shape:
        raise ValueError("classification values and validity must be aligned 2-D grids")

    output = np.full(raw.shape, OPEN_CANOPY_CLASSIFICATION_NODATA, dtype=np.uint8)
    selected = raw[valid]
    if selected.size == 0:
        return output
    if not np.issubdtype(selected.dtype, np.number):
        raise ValueError("classification identifiers must be numeric")
    numeric = selected.astype(np.float64, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("source-valid classification identifiers must be finite")
    rounded = np.rint(numeric)
    if not np.allclose(numeric, rounded, rtol=0.0, atol=0.0):
        raise ValueError("source-valid classification identifiers must be integral")
    if np.any((rounded < 0) | (rounded >= OPEN_CANOPY_CLASSIFICATION_NODATA)):
        raise ValueError("classification identifiers must be in the byte range 0..254")
    output[valid] = rounded.astype(np.uint8)
    return output


def classification_histogram(values: np.ndarray) -> dict[str, int]:
    """Return stable counts for every stored non-nodata upstream identifier."""

    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError("classification sidecar must be a 2-D grid")
    counts = Counter(int(value) for value in raw.ravel())
    counts.pop(OPEN_CANOPY_CLASSIFICATION_NODATA, None)
    return {str(class_id): counts[class_id] for class_id in sorted(counts)}


def rich_vegetation_target(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map IGN vegetation IDs 3/4/5 to dense targets 0/1/2 plus validity."""

    raw = np.asarray(values)
    if raw.ndim != 2:
        raise ValueError("classification sidecar must be a 2-D grid")
    valid = np.isin(raw, tuple(sorted(RICH_VEGETATION_CLASS_IDS)))
    target = np.zeros(raw.shape, dtype=np.uint8)
    target[raw == 3] = 0
    target[raw == 4] = 1
    target[raw == 5] = 2
    return target, valid
