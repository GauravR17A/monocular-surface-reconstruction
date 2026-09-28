"""FastAPI bridge between browser uploads and the local CUDA prediction stack."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
except ImportError:  # Keep the core ML package importable without optional app extras.
    FastAPI = File = Form = HTTPException = UploadFile = None
    CORSMiddleware = StaticFiles = None

from msr.data.radiometry import percentile_stretch_rgb
from msr.inference.predict import load_predictor, predict_height
from msr.inference.relative_depth import (
    DEFAULT_RELATIVE_DEPTH_MODEL,
    DepthAnythingV2Predictor,
    robust_normalize_depth,
)
from msr.inference.viewer_mesh import prepare_semantic_scene, resolve_semantic_masks
from msr.geospatial.calibration import (
    compose_absolute_dsm,
    fit_affine_calibration,
    reproject_dem_to_grid,
)
from msr.geospatial.gcps import sample_raster_at_gcps
from msr.geospatial.public_terrain import download_public_terrain_dem
from msr.evaluation.raster_validation import evaluate_reference_raster
from msr.io.raster import read_rgb_raster, write_float_raster


MAX_UPLOAD_BYTES = 512 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024
MAX_INFERENCE_PIXELS = 3072 * 3072
MAX_INFERENCE_DIMENSION = 3072
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2", ".webp", ".bmp"}


def _preview_rgb(rgb: np.ndarray) -> np.ndarray:
    """Convert arbitrary raster bands into a robust uint8 browser texture."""

    output = percentile_stretch_rgb(
        rgb,
        low_percentile=1,
        high_percentile=99,
    ).astype(np.uint8)
    return np.moveaxis(output, 0, -1)


class InferenceRuntime:
    """Lazily loaded, single-GPU-safe model runtime."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        relative_model: str = DEFAULT_RELATIVE_DEPTH_MODEL,
        device: str = "cuda",
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.relative_model = relative_model
        self.device = device
        self._height_model = None
        self._height_metadata: dict[str, Any] | None = None
        self._relative_predictor = None
        self._lock = threading.Lock()

    def classify(self, input_path: Path, output_dir: Path) -> dict[str, Any]:
        """Optional RGB-only preview; shares the GPU lock, not the height machinery."""
        from msr.inference.rgb_preview import run_preview

        with self._lock:
            return run_preview(input_path, output_dir, self.device)

    def _load(self) -> None:
        if self._height_model is None:
            self._height_model, self._height_metadata = load_predictor(
                self.checkpoint, device=self.device
            )
        if self._relative_predictor is None:
            self._relative_predictor = DepthAnythingV2Predictor(
                model_id=self.relative_model, device=self.device
            )

    def predict(
        self,
        input_path: Path,
        output_dir: Path,
        *,
        dem_path: Path | None = None,
        gcps_path: Path | None = None,
        reference_path: Path | None = None,
        reference_kind: str = "ndsm",
        auto_dem: bool = False,
        auto_dem_if_georeferenced: bool = False,
    ) -> dict[str, Any]:
        """Run both geometry priors and return browser-addressable products."""

        with self._lock:
            self._load()
            source = read_rgb_raster(
                input_path,
                max_pixels=MAX_INFERENCE_PIXELS,
                max_dimension=MAX_INFERENCE_DIMENSION,
            )
            auto_dem = auto_dem or (auto_dem_if_georeferenced and source.georeferenced)
            output_dir.mkdir(parents=True, exist_ok=True)
            preview_path = output_dir / "texture.jpg"
            Image.fromarray(_preview_rgb(source.rgb), mode="RGB").save(
                preview_path, quality=92, optimize=True
            )

            relative = self._relative_predictor.predict(
                source.rgb, valid_mask=source.valid_mask
            )
            relative_path = write_float_raster(
                output_dir / "relative_depth_prior.tif",
                relative.relative_surface,
                reference_profile=source.profile,
                tags={
                    "MSR_PRODUCT": "relative_depth_prior",
                    "MSR_UNITS": "dimensionless",
                },
            )
            height = predict_height(
                source.rgb,
                model=self._height_model,
                device=self.device,
                tile_size=512,
                overlap=128,
                model_metadata=self._height_metadata,
                relative_prior=relative.relative_surface,
                valid_mask=source.valid_mask,
            )
            raw_height_path = write_float_raster(
                output_dir / "height_above_ground_m.tif",
                height.height_map,
                reference_profile=source.profile,
                tags={
                    "MSR_PRODUCT": "nDSM",
                    "MSR_UNITS": "metres",
                    "MSR_CHECKPOINT": str(self.checkpoint),
                },
            )

            display_height = height.height_map
            display_product = "raw_height_above_ground"
            building_path = None
            vegetation_path = None
            confidence_path = None
            semantic_class_path = None
            structures: list[dict[str, Any]] = []
            if height.vegetation_probability_map is not None:
                vegetation_path = write_float_raster(
                    output_dir / "vegetation_probability.tif",
                    height.vegetation_probability_map,
                    reference_profile=source.profile,
                    tags={
                        "MSR_PRODUCT": "vegetation_probability",
                        "MSR_UNITS": "probability_0_1",
                    },
                )
            if height.confidence_map is not None:
                confidence_path = write_float_raster(
                    output_dir / "prediction_confidence.tif",
                    height.confidence_map,
                    reference_profile=source.profile,
                    tags={
                        "MSR_PRODUCT": "prediction_confidence",
                        "MSR_UNITS": "relative_confidence_0_1",
                        "MSR_WARNING": "requires_held_out_calibration",
                    },
                )
            if height.building_probability_map is not None:
                building_path = write_float_raster(
                    output_dir / "building_probability.tif",
                    height.building_probability_map,
                    reference_profile=source.profile,
                    tags={
                        "MSR_PRODUCT": "building_probability",
                        "MSR_UNITS": "probability_0_1",
                    },
                )
                display_height, structures, display_counts = prepare_semantic_scene(
                    height.height_map,
                    height.building_probability_map,
                    rgb=source.rgb,
                    vegetation_probability=height.vegetation_probability_map,
                    terrain_prior=relative.relative_surface,
                    # This head was calibrated at 0.30. Raising the display
                    # cutoff clips complete but moderately confident roofs,
                    # especially in mixed vegetation/urban scenes.
                    threshold=0.30,
                    minimum_component_pixels=max(40, height.height_map.size // 6500),
                )
                _, _, semantic_class = resolve_semantic_masks(
                    height.height_map,
                    height.building_probability_map,
                    rgb=source.rgb,
                    vegetation_probability=height.vegetation_probability_map,
                    building_threshold=0.30,
                    minimum_building_pixels=max(40, height.height_map.size // 6500),
                )
                semantic_values = semantic_class.astype(np.float32)
                semantic_values[semantic_class == 255] = np.nan
                semantic_class_path = write_float_raster(
                    output_dir / "semantic_surface_class.tif",
                    semantic_values,
                    reference_profile=source.profile,
                    tags={
                        "MSR_PRODUCT": "semantic_surface_class",
                        "MSR_CLASSES": "0=surface,1=building,2=vegetation",
                        "MSR_SOURCE": "protected_building_plus_learned_vegetation_rgb_assist",
                    },
                )
                display_product = (
                    "semantic_solid_scene_"
                    f"{display_counts['building_solids']}_buildings_"
                    f"{display_counts['vegetation_pixels']}_vegetation_pixels"
                )

            structures_path = output_dir / "building_solids.json"
            structures_path.write_text(
                json.dumps({"structures": structures}, indent=2) + "\n",
                encoding="utf-8",
            )

            presentation_path = write_float_raster(
                output_dir / "presentation_mesh_height_m.tif",
                display_height,
                reference_profile=source.profile,
                tags={
                    "MSR_PRODUCT": "presentation_mesh_height",
                    "MSR_UNITS": "metres",
                    "MSR_WARNING": "visualization_only_not_for_metric_evaluation",
                },
            )
            relative_rdsm = robust_normalize_depth(
                height.height_map,
                valid_mask=height.valid_mask,
                low_percentile=1,
                high_percentile=99,
            )
            relative_rdsm = 0.8 * relative_rdsm + 0.2 * relative.relative_surface
            relative_rdsm[~source.valid_mask] = np.nan
            rdsm_path = write_float_raster(
                output_dir / "rdsm_relative.tif",
                relative_rdsm,
                reference_profile=source.profile,
                tags={
                    "MSR_PRODUCT": "rDSM",
                    "MSR_UNITS": "relative_0_1",
                },
            )

            aligned_dem_path = None
            absolute_dsm_path = None
            gcp_surface_path = None
            absolute_dsm = None
            gcp_surface = None
            calibration_metadata = None
            public_terrain_path = None
            public_terrain_metadata = None
            warnings = [
                "Presentation ground and building solids are semantic visualization, not the metric evaluation raster."
            ]
            if source.resampled:
                warnings.append(
                    "The source raster exceeded the interactive pixel budget and was resampled over its full geospatial extent. Object-height confidence may be lower at this overview resolution."
                )
            if auto_dem:
                if not source.georeferenced:
                    raise ValueError(
                        "Automatic terrain lookup requires a georeferenced RGB/GeoTIFF input."
                    )
                public_terrain_path = output_dir / "public_terrain_source_m.tif"
                public_terrain_metadata = download_public_terrain_dem(
                    source.profile,
                    public_terrain_path,
                ).to_dict()
                dem_path = public_terrain_path

            if dem_path is not None:
                if not source.georeferenced:
                    raise ValueError("A terrain DEM requires a georeferenced RGB/GeoTIFF input.")
                terrain, terrain_valid = reproject_dem_to_grid(dem_path, source.profile)
                aligned_dem_path = write_float_raster(
                    output_dir / "terrain_dem_aligned_m.tif",
                    terrain,
                    reference_profile=source.profile,
                    tags={
                        "MSR_PRODUCT": "aligned_ground_DEM",
                        "MSR_UNITS": "metres",
                    },
                )
                absolute_dsm = compose_absolute_dsm(
                    terrain,
                    height.height_map,
                    valid_mask=terrain_valid & source.valid_mask,
                )
                absolute_dsm_path = write_float_raster(
                    output_dir / "dsm_absolute_m.tif",
                    absolute_dsm,
                    reference_profile=source.profile,
                    tags={
                        "MSR_PRODUCT": "DSM",
                        "MSR_UNITS": "metres",
                        "MSR_FORMULA": "ground_DEM + nonnegative_object_height",
                    },
                )
            elif source.georeferenced and gcps_path is None:
                warnings.append(
                    "GeoTIFF has spatial metadata but no DEM/GCP datum; absolute DSM was not claimed."
                )

            if gcps_path is not None:
                values, elevations = sample_raster_at_gcps(
                    relative.relative_surface,
                    gcps_path,
                    reference_profile=source.profile,
                    pixel_coordinate_scale=(
                        source.profile["height"] / source.original_height,
                        source.profile["width"] / source.original_width,
                    ),
                )
                calibration = fit_affine_calibration(values, elevations)
                gcp_surface = calibration.apply(relative.relative_surface)
                gcp_surface[~source.valid_mask] = np.nan
                gcp_surface_path = write_float_raster(
                    output_dir / "gcp_calibrated_surface_m.tif",
                    gcp_surface,
                    reference_profile=source.profile,
                    tags={
                        "MSR_PRODUCT": "GCP_calibrated_surface",
                        "MSR_UNITS": "metres",
                    },
                )
                calibration_metadata = {
                    "scale": calibration.scale,
                    "offset_m": calibration.offset_m,
                    "rmse_m": calibration.rmse_m,
                    "correlation": calibration.correlation,
                    "points_used": calibration.points_used,
                    "points_total": calibration.points_total,
                }
                if abs(calibration.correlation) < 0.3:
                    warnings.append(
                        "GCP calibration correlation is weak; treat its metric surface as low confidence."
                    )

            validation_metrics_path = None
            validation_reference_path = None
            validation_error_path = None
            validation_metadata = None
            if reference_path is not None:
                if reference_kind == "ndsm":
                    validation_prediction = height.height_map
                    validation_product = "height_above_ground_ndsm"
                elif reference_kind == "dsm" and absolute_dsm is not None:
                    validation_prediction = absolute_dsm
                    validation_product = "dem_anchored_absolute_dsm"
                elif reference_kind == "dsm" and gcp_surface is not None:
                    validation_prediction = gcp_surface
                    validation_product = "gcp_calibrated_surface"
                elif reference_kind == "dsm":
                    raise ValueError(
                        "Reference DSM validation requires an attached terrain DEM or GCP CSV."
                    )
                else:
                    raise ValueError("reference_kind must be either 'ndsm' or 'dsm'")

                validation = evaluate_reference_raster(
                    validation_prediction,
                    reference_path,
                    prediction_profile=source.profile,
                    prediction_valid_mask=source.valid_mask,
                )
                validation_error_path = write_float_raster(
                    output_dir / "validation_signed_error_m.tif",
                    validation.error_m,
                    reference_profile=source.profile,
                    tags={
                        "MSR_PRODUCT": "signed_validation_error",
                        "MSR_UNITS": "metres_prediction_minus_reference",
                        "MSR_PREDICTION_PRODUCT": validation_product,
                    },
                )
                validation_reference_path = write_float_raster(
                    output_dir / "validation_reference_aligned_m.tif",
                    validation.reference_m,
                    reference_profile=source.profile,
                    tags={
                        "MSR_PRODUCT": "aligned_validation_reference",
                        "MSR_UNITS": "metres",
                        "MSR_REFERENCE_KIND": reference_kind,
                    },
                )
                validation_metadata = {
                    **validation.metrics.to_dict(),
                    "reference_kind": reference_kind,
                    "prediction_product": validation_product,
                    "alignment": validation.alignment,
                }
                validation_metrics_path = output_dir / "validation_metrics.json"
                validation_metrics_path.write_text(
                    json.dumps(validation_metadata, indent=2) + "\n",
                    encoding="utf-8",
                )

            metadata = {
                "georeferenced": source.georeferenced,
                "input_grid": {
                    "original_width": source.original_width,
                    "original_height": source.original_height,
                    "processing_width": source.profile["width"],
                    "processing_height": source.profile["height"],
                    "resampled_for_interactive_processing": source.resampled,
                },
                "display_product": display_product,
                "checkpoint": str(self.checkpoint),
                "relative_model": self.relative_model,
                "calibration": calibration_metadata,
                "terrain_datum": public_terrain_metadata,
                "validation": validation_metadata,
                "warnings": warnings,
            }
            (output_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            return {
                "texture": preview_path.name,
                "presentation_height": presentation_path.name,
                "raw_height": raw_height_path.name,
                "relative_depth": relative_path.name,
                "rdsm": rdsm_path.name,
                "building_probability": building_path.name if building_path else None,
                "vegetation_probability": vegetation_path.name if vegetation_path else None,
                "semantic_class": semantic_class_path.name if semantic_class_path else None,
                "prediction_confidence": confidence_path.name if confidence_path else None,
                "structures": structures_path.name,
                "aligned_dem": aligned_dem_path.name if aligned_dem_path else None,
                "public_terrain": public_terrain_path.name if public_terrain_path else None,
                "absolute_dsm": absolute_dsm_path.name if absolute_dsm_path else None,
                "gcp_surface": gcp_surface_path.name if gcp_surface_path else None,
                "validation_metrics": (
                    validation_metrics_path.name if validation_metrics_path else None
                ),
                "validation_reference": (
                    validation_reference_path.name if validation_reference_path else None
                ),
                "validation_error": (
                    validation_error_path.name if validation_error_path else None
                ),
                "metadata": metadata,
            }


def create_app(
    *,
    runtime: InferenceRuntime,
    results_root: str | Path,
):
    """Create the API separately so tests can inject a lightweight runtime."""

    if FastAPI is None:
        raise RuntimeError("Install the application dependencies with `pip install -e .[app]`.")

    root = Path(results_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Monocular Surface Reconstruction Local GPU API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in os.environ.get("MSR_VIEWER_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if origin.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.mount("/results", StaticFiles(directory=root), name="results")
    from msr.api.classification import register_classification_routes

    register_classification_routes(app, runtime, root)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ready", "device": runtime.device}

    @app.post("/api/predict")
    async def predict(
        image: UploadFile = File(...),
        dem: UploadFile | None = File(None),
        gcps: UploadFile | None = File(None),
        reference: UploadFile | None = File(None),
        reference_kind: str = Form("ndsm"),
        auto_dem: bool = Form(False),
        auto_dem_if_georeferenced: bool = Form(False),
    ) -> dict[str, Any]:
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise HTTPException(status_code=415, detail="Upload JPG, PNG, TIFF, WebP, BMP, or JP2 imagery.")
        if dem is not None and gcps is not None:
            raise HTTPException(status_code=400, detail="Choose either a DEM or GCP CSV, not both.")
        if (auto_dem or auto_dem_if_georeferenced) and (dem is not None or gcps is not None):
            raise HTTPException(
                status_code=400,
                detail="Choose automatic terrain, an uploaded DEM, or GCPsâ€”not more than one.",
            )
        dem_suffix = Path(dem.filename or "").suffix.lower() if dem else ""
        if dem is not None and dem_suffix not in {".tif", ".tiff"}:
            raise HTTPException(status_code=415, detail="The terrain DEM must be a GeoTIFF.")
        gcp_suffix = Path(gcps.filename or "").suffix.lower() if gcps else ""
        if gcps is not None and gcp_suffix != ".csv":
            raise HTTPException(status_code=415, detail="Ground control points must be a CSV file.")
        reference_suffix = Path(reference.filename or "").suffix.lower() if reference else ""
        if reference is not None and reference_suffix not in {".tif", ".tiff"}:
            raise HTTPException(status_code=415, detail="The validation reference must be a GeoTIFF.")
        if reference_kind not in {"ndsm", "dsm"}:
            raise HTTPException(status_code=400, detail="Reference kind must be nDSM or DSM.")
        job_id = uuid4().hex
        job_dir = root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        input_path = job_dir / f"input{suffix}"
        dem_path = job_dir / f"terrain_dem{dem_suffix}" if dem else None
        gcps_path = job_dir / "ground_control_points.csv" if gcps else None
        reference_path = job_dir / f"validation_reference{reference_suffix}" if reference else None
        try:
            uploaded_bytes = 0
            with input_path.open("wb") as output:
                while chunk := await image.read(UPLOAD_CHUNK_BYTES):
                    uploaded_bytes += len(chunk)
                    if uploaded_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Input exceeds the 512 MB local processing limit.",
                        )
                    output.write(chunk)
            if uploaded_bytes == 0:
                raise HTTPException(status_code=400, detail="The uploaded image is empty.")
            for auxiliary, destination, label in (
                (dem, dem_path, "terrain DEM"),
                (gcps, gcps_path, "GCP CSV"),
                (reference, reference_path, "validation reference"),
            ):
                if auxiliary is None or destination is None:
                    continue
                auxiliary_bytes = 0
                with destination.open("wb") as output:
                    while chunk := await auxiliary.read(UPLOAD_CHUNK_BYTES):
                        auxiliary_bytes += len(chunk)
                        if auxiliary_bytes > MAX_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"The {label} exceeds the 512 MB local processing limit.",
                            )
                        output.write(chunk)
                if auxiliary_bytes == 0:
                    raise HTTPException(status_code=400, detail=f"The {label} is empty.")
            products = await asyncio.to_thread(
                runtime.predict,
                input_path,
                job_dir,
                dem_path=dem_path,
                gcps_path=gcps_path,
                reference_path=reference_path,
                reference_kind=reference_kind,
                auto_dem=auto_dem,
                auto_dem_if_georeferenced=auto_dem_if_georeferenced,
            )
        except HTTPException:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        except ValueError as error:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"Inference failed: {error}") from error
        finally:
            await image.close()
            if dem is not None:
                await dem.close()
            if gcps is not None:
                await gcps.close()
            if reference is not None:
                await reference.close()
        base = f"/results/{job_id}"
        return {
            "job_id": job_id,
            "uploaded_bytes": uploaded_bytes,
            "texture_url": f"{base}/{products['texture']}",
            "height_url": f"{base}/{products['presentation_height']}",
            "raw_height_url": f"{base}/{products['raw_height']}",
            "rdsm_url": f"{base}/{products['rdsm']}",
            "relative_depth_url": f"{base}/{products['relative_depth']}",
            "building_probability_url": (
                f"{base}/{products['building_probability']}"
                if products["building_probability"] else None
            ),
            "vegetation_probability_url": (
                f"{base}/{products['vegetation_probability']}"
                if products.get("vegetation_probability") else None
            ),
            "semantic_class_url": (
                f"{base}/{products['semantic_class']}"
                if products.get("semantic_class") else None
            ),
            "prediction_confidence_url": (
                f"{base}/{products['prediction_confidence']}"
                if products.get("prediction_confidence") else None
            ),
            "structures_url": f"{base}/{products['structures']}",
            "aligned_dem_url": (
                f"{base}/{products['aligned_dem']}" if products.get("aligned_dem") else None
            ),
            "public_terrain_url": (
                f"{base}/{products['public_terrain']}"
                if products.get("public_terrain") else None
            ),
            "absolute_dsm_url": (
                f"{base}/{products['absolute_dsm']}" if products.get("absolute_dsm") else None
            ),
            "gcp_surface_url": (
                f"{base}/{products['gcp_surface']}" if products.get("gcp_surface") else None
            ),
            "validation_metrics_url": (
                f"{base}/{products['validation_metrics']}"
                if products.get("validation_metrics") else None
            ),
            "validation_reference_url": (
                f"{base}/{products['validation_reference']}"
                if products.get("validation_reference") else None
            ),
            "validation_error_url": (
                f"{base}/{products['validation_error']}"
                if products.get("validation_error") else None
            ),
            "validation": products["metadata"].get("validation"),
            "metadata_url": f"{base}/metadata.json",
            "metadata": products["metadata"],
        }

    return app
