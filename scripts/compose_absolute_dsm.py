"""Combine a georeferenced terrain DEM with Monocular Surface Reconstruction object heights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rasterio

from msr.geospatial.calibration import compose_absolute_dsm, reproject_dem_to_grid
from msr.io.raster import write_float_raster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_rgb", type=Path)
    parser.add_argument("ground_dem", type=Path)
    parser.add_argument("object_height", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with rasterio.open(args.reference_rgb) as reference:
        profile = reference.profile.copy()
    with rasterio.open(args.object_height) as height_source:
        height = height_source.read(1)
        if height.shape != (profile["height"], profile["width"]):
            raise ValueError("Object-height raster must already match the RGB grid")
    terrain, valid = reproject_dem_to_grid(args.ground_dem, profile)
    dsm = compose_absolute_dsm(terrain, height, valid_mask=valid)
    output = write_float_raster(
        args.output,
        dsm,
        reference_profile=profile,
        tags={
            "MSR_PRODUCT": "DSM",
            "MSR_UNITS": "metres",
            "MSR_TERRAIN_SOURCE": str(args.ground_dem),
            "MSR_HEIGHT_SOURCE": str(args.object_height),
        },
    )
    metadata = {
        "product": "absolute_dsm",
        "units": "metres",
        "reference_rgb": str(args.reference_rgb.resolve()),
        "ground_dem": str(args.ground_dem.resolve()),
        "object_height": str(args.object_height.resolve()),
        "output": str(output),
        "formula": "DSM = reprojected terrain DEM + max(predicted object height, 0)",
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
