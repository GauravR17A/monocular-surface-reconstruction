"""Deterministic overlap layout and seam-resistant blending weights."""

from __future__ import annotations

import numpy as np


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Return tile origins that cover a dimension and always touch its far edge."""

    if length <= 0 or tile_size <= 0:
        raise ValueError("length and tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_weight(height: int, width: int, *, floor: float = 1e-3) -> np.ndarray:
    """Create a positive Hann-like 2-D window for overlapping prediction tiles."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if not 0.0 < floor <= 1.0:
        raise ValueError("floor must satisfy 0 < floor <= 1")

    y = np.hanning(height) if height > 2 else np.ones(height)
    x = np.hanning(width) if width > 2 else np.ones(width)
    weight = np.outer(y, x).astype(np.float32)
    weight /= max(float(weight.max()), np.finfo(np.float32).eps)
    return np.maximum(weight, floor)

