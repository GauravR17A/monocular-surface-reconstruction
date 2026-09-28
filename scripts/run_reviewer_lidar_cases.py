"""Run the two fixed reviewer LiDAR cases through the already-running local API."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = PROJECT_ROOT / "outputs" / "presentation" / "monday_demo" / "reviewer_lidar_upload_demo"
WEB_JOBS = PROJECT_ROOT / "outputs" / "web_jobs"
API_URL = "http://127.0.0.1:8000/api/predict"

CASES = (
    (
        "01_URBAN_US3D_SATELLITE_LIDAR",
        "01_UPLOAD_urban_WorldView3_RGB.tif",
        "02_ATTACH_urban_airborne_LiDAR_AGL_nDSM.tif",
    ),
    (
        "02_RURAL_OPEN_CANOPY_SATELLITE_LIDAR",
        "01_UPLOAD_rural_SPOT_RGB.tif",
        "02_ATTACH_rural_IGN_LiDAR_canopy_nDSM.tif",
    ),
)


def main() -> None:
    for case_name, image_name, reference_name in CASES:
        case = DEMO_ROOT / case_name
        print(f"Running {case_name}...", flush=True)
        with (case / image_name).open("rb") as image, (case / reference_name).open("rb") as reference:
            response = requests.post(
                API_URL,
                files={
                    "image": (image_name, image, "image/tiff"),
                    "reference": (reference_name, reference, "image/tiff"),
                },
                data={"reference_kind": "ndsm"},
                timeout=900,
            )
        response.raise_for_status()
        payload = response.json()
        job_id = payload["job_id"]
        source = WEB_JOBS / job_id
        result = case / "Monocular Surface Reconstruction_RESULT"
        result.mkdir(parents=True, exist_ok=True)
        products = {
            "raw_height_url": "01_model_prediction_nDSM_m.tif",
            "validation_reference_url": "02_LiDAR_reference_aligned_m.tif",
            "validation_error_url": "03_signed_error_prediction_minus_LiDAR_m.tif",
            "validation_metrics_url": "04_validation_metrics.json",
            "metadata_url": "05_pipeline_metadata.json",
            "texture_url": "06_processed_RGB_preview.jpg",
        }
        for response_key, destination_name in products.items():
            url = payload.get(response_key)
            if not url:
                continue
            source_name = Path(url).name
            shutil.copy2(source / source_name, result / destination_name)
        summary = {
            "case": case_name,
            "job_id": job_id,
            "reference_usage": "Evaluation only. It was not supplied to the model as an input feature.",
            "reference_kind": "LiDAR-derived nDSM / height above ground",
            "validation": payload.get("validation"),
            "model": payload.get("metadata", {}).get("model"),
            "warnings": payload.get("metadata", {}).get("warnings", []),
        }
        (result / "HONEST_RESULT_SUMMARY.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        metrics = payload.get("validation") or {}
        print(
            f"{case_name}: RMSE={metrics.get('rmse_m'):.3f} m, "
            f"MAE={metrics.get('mae_m'):.3f} m, R2={metrics.get('r2'):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
