"""Large-raster tiled inference helpers and public prediction API."""

from .predict import PredictionResult, load_predictor, predict_height
from .relative_depth import (
    DEFAULT_RELATIVE_DEPTH_MODEL,
    DepthAnythingV2Predictor,
    RelativeDepthResult,
    robust_normalize_depth,
)
from .tiling import blend_weight, tile_starts

__all__ = [
    "PredictionResult",
    "DEFAULT_RELATIVE_DEPTH_MODEL",
    "DepthAnythingV2Predictor",
    "RelativeDepthResult",
    "blend_weight",
    "load_predictor",
    "predict_height",
    "robust_normalize_depth",
    "tile_starts",
]
