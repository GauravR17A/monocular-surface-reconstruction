"""Run the current end-to-end Monocular Surface Reconstruction raster pipeline on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from msr.geospatial.calibration import (
    compose_absolute_dsm,
    fit_affine_calibration,
    reproject_dem_to_grid,
)
from msr.geospatial.gcps import sample_raster_at_gcps
from msr.inference.predict import load_predictor, predict_height
from msr.inference.relative_depth import (
    DEFAULT_RELATIVE_DEPTH_MODEL,
    DepthAnythingV2Predictor,
    robust_normalize_depth,
)
from msr.io.raster import read_rgb_raster, write_float_raster


def _jsonable_calibration(calibration) -> dict[str, float | int]:
    return {
        "scale": calibration.scale,
        "offset_m": calibration.offset_m,
        "rmse_m": calibration.rmse_m,
        "correlation": calibration.correlation,
        "points_used": calibration.points_used,
        "points_total": calibration.points_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create relative and, when calibrated, absolute surface products."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dem", type=Path, help="Ground-terrain DEM, e.g. SRTM.")
    parser.add_argument("--gcps", type=Path, help="CSV with elevation_m and x/y or row/col.")
    parser.add_argument("--relative-model", default=DEFAULT_RELATIVE_DEPTH_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--rgb-scale", type=float, default=255.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = read_rgb_raster(args.image)
    metadata: dict[str, object] = {
        "source": str(source.source),
        "georeferenced": source.georeferenced,
        "products": {},
        "warnings": [],
    }
    products: dict[str, str] = metadata["products"]  # type: ignore[assignment]
    warnings: list[str] = metadata["warnings"]  # type: ignore[assignment]

    relative_predictor = DepthAnythingV2Predictor(
        model_id=args.relative_model,
        device=args.device,
    )
    relative = relative_predictor.predict(source.rgb, valid_mask=source.valid_mask)
    relative_path = write_float_raster(
        args.output_dir / "relative_depth_prior.tif",
        relative.relative_surface,
        reference_profile=source.profile,
        tags={
            "MSR_PRODUCT": "relative_depth_prior",
            "MSR_UNITS": "dimensionless",
        },
    )
    products["relative_depth_prior"] = str(relative_path)

    height_result = None
    if args.checkpoint:
        model, model_metadata = load_predictor(args.checkpoint, device=args.device)
        height_result = predict_height(
            source.rgb,
            model=model,
            device=args.device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            rgb_scale=args.rgb_scale,
            model_metadata=model_metadata,
            relative_prior=relative.relative_surface,
        )
        height_path = write_float_raster(
            args.output_dir / "height_above_ground_m.tif",
            height_result.height_map,
            reference_profile=source.profile,
            tags={
                "MSR_PRODUCT": "nDSM",
                "MSR_UNITS": "metres",
                "MSR_CHECKPOINT": str(args.checkpoint),
            },
        )
        products["height_above_ground_m"] = str(height_path)
        if height_result.building_probability_map is not None:
            building_path = write_float_raster(
                args.output_dir / "building_probability.tif",
                height_result.building_probability_map,
                reference_profile=source.profile,
                tags={
                    "MSR_PRODUCT": "building_probability",
                    "MSR_UNITS": "probability_0_1",
                },
            )
            products["building_probability"] = str(building_path)
        if height_result.vegetation_probability_map is not None:
            vegetation_path = write_float_raster(
                args.output_dir / "vegetation_probability.tif",
                height_result.vegetation_probability_map,
                reference_profile=source.profile,
                tags={
                    "MSR_PRODUCT": "vegetation_probability",
                    "MSR_UNITS": "probability_0_1",
                },
            )
            products["vegetation_probability"] = str(vegetation_path)
        if height_result.confidence_map is not None:
            confidence_path = write_float_raster(
                args.output_dir / "prediction_confidence.tif",
                height_result.confidence_map,
                reference_profile=source.profile,
                tags={
                    "MSR_PRODUCT": "prediction_confidence",
                    "MSR_UNITS": "relative_confidence_0_1",
                    "MSR_WARNING": "requires_held_out_calibration",
                },
            )
            products["prediction_confidence"] = str(confidence_path)

    if not source.georeferenced:
        if height_result is None:
            rdsm = relative.relative_surface
            basis = "pretrained_relative_depth"
        else:
            # Shape-only output: without GSD/CRS, learned metre scale is not claimed.
            learned_shape = robust_normalize_depth(
                height_result.height_map,
                valid_mask=height_result.valid_mask,
                low_percentile=1,
                high_percentile=99,
            )
            rdsm = 0.8 * learned_shape + 0.2 * relative.relative_surface
            basis = "80% learned_height_shape + 20% pretrained_relative_geometry"
        rdsm[~source.valid_mask] = np.nan
        rdsm_path = write_float_raster(
            args.output_dir / "rdsm_relative.tif",
            rdsm,
            reference_profile=source.profile,
            tags={
                "MSR_PRODUCT": "rDSM",
                "MSR_UNITS": "relative_0_1",
                "MSR_BASIS": basis,
            },
        )
        products["rdsm"] = str(rdsm_path)
        warnings.append("No georeferencing: rDSM is relative and has no metre datum.")

    if args.dem:
        if not source.georeferenced:
            raise ValueError("A DEM can only be aligned to a georeferenced RGB raster")
        if height_result is None:
            raise ValueError("--dem currently requires --checkpoint for object-height prediction")
        terrain, terrain_valid = reproject_dem_to_grid(args.dem, source.profile)
        terrain_path = write_float_raster(
            args.output_dir / "terrain_dem_aligned_m.tif",
            terrain,
            reference_profile=source.profile,
            tags={"MSR_PRODUCT": "aligned_ground_DEM", "MSR_UNITS": "metres"},
        )
        products["terrain_dem_aligned_m"] = str(terrain_path)
        dsm = compose_absolute_dsm(
            terrain,
            height_result.height_map,
            valid_mask=terrain_valid & source.valid_mask,
        )
        dsm_path = write_float_raster(
            args.output_dir / "dsm_absolute_m.tif",
            dsm,
            reference_profile=source.profile,
            tags={
                "MSR_PRODUCT": "DSM",
                "MSR_UNITS": "metres",
                "MSR_FORMULA": "ground_DEM + nonnegative_object_height",
            },
        )
        products["dsm_absolute_m"] = str(dsm_path)
    elif source.georeferenced and height_result is not None:
        warnings.append("Georeferenced image has no DEM/GCP datum; absolute DSM was not claimed.")

    if args.gcps:
        values, elevations = sample_raster_at_gcps(
            relative.relative_surface,
            args.gcps,
            reference_profile=source.profile,
        )
        calibration = fit_affine_calibration(values, elevations)
        calibrated = calibration.apply(relative.relative_surface)
        calibrated[~source.valid_mask] = np.nan
        gcp_path = write_float_raster(
            args.output_dir / "gcp_calibrated_surface_m.tif",
            calibrated,
            reference_profile=source.profile,
            tags={
                "MSR_PRODUCT": "GCP_calibrated_surface",
                "MSR_UNITS": "metres",
            },
        )
        products["gcp_calibrated_surface_m"] = str(gcp_path)
        metadata["gcp_calibration"] = _jsonable_calibration(calibration)
        if abs(calibration.correlation) < 0.3:
            warnings.append("GCP calibration correlation is weak; treat its metric surface as low confidence.")

    metadata["relative_depth"] = relative.metadata
    metadata_path = args.output_dir / "pipeline_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
