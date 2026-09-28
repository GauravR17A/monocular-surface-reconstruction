"""Create a presentation mesh raster from height and building probability.

This is a visualization-only semantic edge snap. It does not replace the raw
model output and must not be used when reporting validation metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio

from msr.inference.viewer_mesh import prepare_semantic_scene
from msr.io.raster import read_rgb_raster, write_float_raster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("height", type=Path)
    parser.add_argument("building_probability", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--terrain-prior", type=Path)
    parser.add_argument("--structures-output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--minimum-component-pixels", type=int, default=160)
    parser.add_argument("--roof-flatten", type=float, default=0.8)
    args = parser.parse_args()

    with rasterio.open(args.height) as source:
        height = source.read(1)
        profile = source.profile.copy()
    with rasterio.open(args.building_probability) as source:
        probability = source.read(1)
    rgb = read_rgb_raster(args.rgb).rgb if args.rgb else None
    terrain_prior = None
    if args.terrain_prior:
        with rasterio.open(args.terrain_prior) as source:
            terrain_prior = source.read(1)
    mesh, structures, counts = prepare_semantic_scene(
        height,
        probability,
        rgb=rgb,
        terrain_prior=terrain_prior,
        threshold=args.threshold,
        minimum_component_pixels=args.minimum_component_pixels,
        roof_flatten=args.roof_flatten,
    )
    write_float_raster(
        args.output,
        mesh,
        reference_profile=profile,
        tags={
            "MSR_PRODUCT": "presentation_ground_relief",
            "MSR_UNITS": "metres",
            "MSR_WARNING": "visualization_only_not_for_metric_evaluation",
            "MSR_COMPONENTS": str(counts["building_solids"]),
            "MSR_VEGETATION_PIXELS": str(counts["vegetation_pixels"]),
        },
    )
    if args.structures_output:
        args.structures_output.parent.mkdir(parents=True, exist_ok=True)
        args.structures_output.write_text(
            json.dumps({"structures": structures}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"Saved {args.output.resolve()} with {counts['building_solids']} retained "
        f"building components and {counts['vegetation_pixels']} vegetation pixels; "
        "visualization only."
    )


if __name__ == "__main__":
    main()
