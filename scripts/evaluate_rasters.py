"""Evaluate one predicted height GeoTIFF against an aligned reference raster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from msr.evaluation.metrics import compute_height_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--height-min-m", type=float, default=0.0)
    parser.add_argument("--height-max-m", type=float, default=1000.0)
    parser.add_argument(
        "--scope",
        choices=("all-valid", "positive-target"),
        default="all-valid",
        help="Use positive-target only for datasets whose zero nDSM denotes non-building pixels.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with rasterio.open(args.prediction) as prediction_source, rasterio.open(
        args.target
    ) as target_source:
        if (prediction_source.height, prediction_source.width) != (
            target_source.height,
            target_source.width,
        ):
            raise ValueError("Prediction and target rasters have different dimensions")
        if not np.allclose(tuple(prediction_source.transform), tuple(target_source.transform)):
            raise ValueError("Prediction and target geospatial transforms differ")
        prediction = prediction_source.read(1).astype(np.float64)
        target = target_source.read(1).astype(np.float64)
        valid = np.isfinite(prediction) & np.isfinite(target)
        if prediction_source.nodata is not None:
            valid &= ~np.isclose(prediction, prediction_source.nodata, equal_nan=True)
        if target_source.nodata is not None:
            valid &= ~np.isclose(target, target_source.nodata, equal_nan=True)
    valid &= target >= args.height_min_m
    valid &= target <= args.height_max_m
    if args.scope == "positive-target":
        valid &= target > 0.0
    if args.mask:
        with rasterio.open(args.mask) as mask_source:
            valid &= mask_source.read(1) > 0

    payload = compute_height_metrics(prediction, target, valid).to_dict()
    serialized = json.dumps(payload, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

