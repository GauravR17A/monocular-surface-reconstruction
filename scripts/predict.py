"""Run a Monocular Surface Reconstruction checkpoint on one RGB GeoTIFF and preserve georeferencing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio

from msr.inference.predict import load_predictor, predict_height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, metadata = load_predictor(args.checkpoint, device=args.device)
    with rasterio.open(args.image) as source:
        if source.count < 3:
            raise ValueError("Input raster must contain at least three RGB bands")
        image = source.read((1, 2, 3))
        profile = source.profile.copy()
    result = predict_height(
        image,
        model=model,
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        model_metadata=metadata,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.pop("photometric", None)
    profile.update(
        count=1,
        dtype="float32",
        nodata=float("nan"),
        compress="deflate",
        predictor=3,
    )
    with rasterio.open(args.output, "w", **profile) as destination:
        destination.write(result.height_map, 1)
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(result.model_metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
