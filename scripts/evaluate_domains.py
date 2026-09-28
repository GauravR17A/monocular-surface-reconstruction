"""Evaluate one DSM/nDSM against an aligned reference by landscape domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from msr.evaluation.domain_metrics import (
    evaluate_by_domain,
    terrain_slope_degrees,
)
from msr.io.raster import write_float_raster


def _aligned(left: rasterio.DatasetReader, right: rasterio.DatasetReader) -> bool:
    return (
        left.width == right.width
        and left.height == right.height
        and left.crs == right.crs
        and np.allclose(tuple(left.transform), tuple(right.transform), rtol=0.0, atol=1e-9)
    )


def _read_mask(path: Path, reference: rasterio.DatasetReader) -> np.ndarray:
    with rasterio.open(path) as source:
        if not _aligned(source, reference):
            raise ValueError(f"Auxiliary mask is not aligned with the reference: {path}")
        values = source.read(1, masked=False)
        valid = np.isfinite(values)
        if source.nodata is not None:
            valid &= ~np.isclose(values, source.nodata, equal_nan=True)
        return valid & (values > 0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report DSM accuracy separately for urban, forest, hilly terrain, "
            "and plains. Prediction and reference must represent the same surface."
        )
    )
    parser.add_argument("prediction", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--building-mask", type=Path)
    parser.add_argument("--vegetation-mask", type=Path)
    parser.add_argument(
        "--dtm",
        type=Path,
        help="Aligned bare-earth DTM used only to distinguish plains from hilly terrain.",
    )
    parser.add_argument("--hilly-threshold-deg", type=float, default=5.0)
    parser.add_argument("--height-min-m", type=float, default=-500.0)
    parser.add_argument("--height-max-m", type=float, default=9000.0)
    args = parser.parse_args()

    with rasterio.open(args.prediction) as prediction_source, rasterio.open(
        args.reference
    ) as reference_source:
        if not _aligned(prediction_source, reference_source):
            raise ValueError("Prediction and reference grids/CRS/transforms differ")
        prediction = prediction_source.read(1, masked=False).astype(np.float64)
        reference = reference_source.read(1, masked=False).astype(np.float64)
        valid = np.isfinite(prediction) & np.isfinite(reference)
        if prediction_source.nodata is not None:
            valid &= ~np.isclose(prediction, prediction_source.nodata, equal_nan=True)
        if reference_source.nodata is not None:
            valid &= ~np.isclose(reference, reference_source.nodata, equal_nan=True)
        valid &= reference >= args.height_min_m
        valid &= reference <= args.height_max_m
        profile = reference_source.profile.copy()
        building = (
            _read_mask(args.building_mask, reference_source)
            if args.building_mask
            else None
        )
        vegetation = (
            _read_mask(args.vegetation_mask, reference_source)
            if args.vegetation_mask
            else None
        )
        slope = None
        if args.dtm:
            with rasterio.open(args.dtm) as dtm_source:
                if not _aligned(dtm_source, reference_source):
                    raise ValueError("DTM is not aligned with the reference")
                dtm = dtm_source.read(1, masked=False).astype(np.float64)
                dtm_valid = np.isfinite(dtm)
                if dtm_source.nodata is not None:
                    dtm_valid &= ~np.isclose(dtm, dtm_source.nodata, equal_nan=True)
                if reference_source.crs is None or not reference_source.crs.is_projected:
                    raise ValueError(
                        "Slope domains require a projected metric CRS; reproject geographic data first"
                    )
                slope = terrain_slope_degrees(
                    dtm,
                    valid & dtm_valid,
                    pixel_size_x_m=abs(reference_source.transform.a),
                    pixel_size_y_m=abs(reference_source.transform.e),
                )

    result = evaluate_by_domain(
        prediction,
        reference,
        valid,
        building_mask=building,
        vegetation_mask=vegetation,
        terrain_slope_deg=slope,
        hilly_threshold_deg=args.hilly_threshold_deg,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "domain_metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2) + "\n", encoding="utf-8")

    signed_error = (prediction - reference).astype(np.float32)
    signed_error[~valid] = np.nan
    write_float_raster(
        args.output_dir / "signed_error_m.tif",
        signed_error,
        reference_profile=profile,
        tags={
            "MSR_PRODUCT": "signed_DSM_error",
            "MSR_UNITS": "metres_prediction_minus_reference",
        },
    )
    domain_profile = profile.copy()
    domain_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    domain_path = args.output_dir / "domain_map.tif"
    with rasterio.open(domain_path, "w", **domain_profile) as destination:
        destination.write(result.domain_map, 1)
        destination.update_tags(
            MSR_PRODUCT="exclusive_evaluation_domains",
            MSR_DOMAIN_CODES=json.dumps(result.metrics["domain_codes"]),
        )
    print(json.dumps(result.metrics, indent=2))
    print(f"Saved domain evaluation to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
