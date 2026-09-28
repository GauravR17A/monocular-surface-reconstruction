"""Extract a licensed Depth Anything V2 relative-surface prior from RGB imagery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from msr.inference.relative_depth import (
    DEFAULT_RELATIVE_DEPTH_MODEL,
    DepthAnythingV2Predictor,
)
from msr.io.raster import read_rgb_raster, write_float_raster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=DEFAULT_RELATIVE_DEPTH_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--far-is-high",
        action="store_true",
        help="Invert the relative map when scene orientation requires it.",
    )
    args = parser.parse_args()

    source = read_rgb_raster(args.image)
    predictor = DepthAnythingV2Predictor(
        model_id=args.model,
        device=args.device,
        near_is_high=not args.far_is_high,
    )
    result = predictor.predict(source.rgb, valid_mask=source.valid_mask)
    tags = {
        "MSR_PRODUCT": "rDSM",
        "MSR_UNITS": "relative_0_1",
        "MSR_GEOREFERENCED": str(source.georeferenced).lower(),
        "MSR_MODEL": args.model,
    }
    output = write_float_raster(
        args.output,
        result.relative_surface,
        reference_profile=source.profile,
        tags=tags,
    )
    metadata = {
        **result.metadata,
        "source": str(source.source),
        "output": str(output),
        "georeferenced": source.georeferenced,
        "product": "relative_dsm",
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
