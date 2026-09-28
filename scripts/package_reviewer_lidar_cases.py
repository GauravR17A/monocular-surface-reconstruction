"""Package fixed, reproducible urban and rural optical/LiDAR demo pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import rasterio
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "presentation" / "monday_demo" / "reviewer_lidar_upload_demo"
URBAN_SOURCE = PROJECT_ROOT / "data" / "demo_sources" / "DFC19_candidate_selection"
RURAL_SOURCE = PROJECT_ROOT / "data" / "open_canopy_v2" / "test" / "oc_2021_728_6349"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grid(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        return {
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "dtype": dataset.dtypes[0],
            "crs": str(dataset.crs),
            "transform": list(dataset.transform),
            "bounds": list(dataset.bounds),
            "nodata": dataset.nodata,
        }


def aligned(left: Path, right: Path) -> bool:
    left_grid = grid(left)
    right_grid = grid(right)
    return all(
        left_grid[key] == right_grid[key]
        for key in ("width", "height", "crs", "transform", "bounds")
    )


def height_stats(path: Path) -> dict[str, float | int]:
    with rasterio.open(path) as dataset:
        values = dataset.read(1).astype(np.float32)
        valid = np.isfinite(values)
        if dataset.nodata is not None:
            valid &= values != dataset.nodata
    height = values[valid]
    return {
        "valid_pixels": int(height.size),
        "minimum_m": float(height.min()),
        "mean_m": float(height.mean()),
        "p50_m": float(np.percentile(height, 50)),
        "p90_m": float(np.percentile(height, 90)),
        "p95_m": float(np.percentile(height, 95)),
        "maximum_m": float(height.max()),
        "above_2m_percent": float((height > 2).mean() * 100),
        "above_5m_percent": float((height > 5).mean() * 100),
    }


def rgb_preview(source: Path, destination: Path) -> None:
    with rasterio.open(source) as dataset:
        values = dataset.read([1, 2, 3])
    if values.dtype != np.uint8:
        output = np.zeros(values.shape, dtype=np.uint8)
        for index in range(3):
            channel = values[index].astype(np.float32)
            finite = np.isfinite(channel)
            low, high = np.percentile(channel[finite], (2, 98)) if finite.any() else (0.0, 1.0)
            output[index] = np.clip((channel - low) * 255 / max(high - low, 1e-6), 0, 255)
        values = output
    Image.fromarray(np.moveaxis(values, 0, -1), mode="RGB").save(destination, quality=94)


def reference_preview(source: Path, destination: Path, maximum_m: float) -> None:
    with rasterio.open(source) as dataset:
        values = dataset.read(1).astype(np.float32)
    scaled = np.clip(values, 0, maximum_m) / maximum_m
    colors = np.stack(
        [
            np.clip(4 * scaled - 1.5, 0, 1),
            np.clip(2 - np.abs(4 * scaled - 2), 0, 1),
            np.clip(1.5 - 4 * scaled, 0, 1),
        ],
        axis=-1,
    )
    Image.fromarray((colors * 255).astype(np.uint8), mode="RGB").save(destination)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def package_urban() -> dict[str, Any]:
    case = OUTPUT_ROOT / "01_URBAN_US3D_SATELLITE_LIDAR"
    case.mkdir(parents=True, exist_ok=True)
    source_rgb = URBAN_SOURCE / "rgb" / "block_0_4401.tif"
    source_height = URBAN_SOURCE / "depth" / "block_0_4401.tif"
    source_semantic = URBAN_SOURCE / "seg" / "block_0_4401.png"
    rgb = case / "01_UPLOAD_urban_WorldView3_RGB.tif"
    reference = case / "02_ATTACH_urban_airborne_LiDAR_AGL_nDSM.tif"
    semantic = case / "03_REFERENCE_semantic_classes.png"
    shutil.copy2(source_rgb, rgb)
    shutil.copy2(source_height, reference)
    shutil.copy2(source_semantic, semantic)
    rgb_preview(rgb, case / "PREVIEW_urban_RGB.jpg")
    reference_preview(reference, case / "PREVIEW_LiDAR_height_0_to_30m.png", 30.0)

    metadata = {
        "case": "urban satellite + airborne LiDAR",
        "dataset": "US3D / 2019 IEEE GRSS Data Fusion Contest (DFC19), Omaha",
        "tile": "block_0_4401 (preprocessed held-out test tile)",
        "input_sensor": "WorldView-3 pan-sharpened optical satellite imagery",
        "height_reference": "airborne-LiDAR-derived above-ground-level (AGL) height / nDSM",
        "reference_kind_for_msr": "ndsm",
        "selection_integrity": (
            "Curated showcase selected after evaluating a fixed set of 15 pre-downloaded DFC19 held-out tiles. "
            "It must be presented as a showcase example, not as aggregate test accuracy. The complete ranking is "
            "saved under ../showcase_selection/urban_dfc19_candidate_metrics.json."
        ),
        "official_source": "https://www.grss-ieee.org/community/technical-committees/2019-ieee-grss-data-fusion-contest/",
        "official_dataset_record": "https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019",
        "redistributable_preprocessing_source": "https://github.com/JTRNEO/SynRS3D",
        "downloaded_tile_source": "https://huggingface.co/datasets/JasonXF/DFC2019-10k",
        "license_note": "The SynRS3D dataset documentation lists DFC19 as Creative Commons Attribution. Retain attribution when presenting or redistributing.",
        "rgb_grid": grid(rgb),
        "reference_grid": grid(reference),
        "exact_grid_alignment": aligned(rgb, reference),
        "reference_statistics": height_stats(reference),
        "files": {
            rgb.name: sha256(rgb),
            reference.name: sha256(reference),
            semantic.name: sha256(semantic),
        },
    }
    write_json(case / "source_provenance_and_alignment.json", metadata)
    return metadata


def package_rural() -> dict[str, Any]:
    case = OUTPUT_ROOT / "02_RURAL_OPEN_CANOPY_SATELLITE_LIDAR"
    case.mkdir(parents=True, exist_ok=True)
    rgb = case / "01_UPLOAD_rural_SPOT_RGB.tif"
    reference = case / "02_ATTACH_rural_IGN_LiDAR_canopy_nDSM.tif"
    vegetation = case / "03_REFERENCE_vegetation_mask.tif"
    valid = case / "04_REFERENCE_valid_mask.tif"
    shutil.copy2(RURAL_SOURCE / "rgb.tif", rgb)
    shutil.copy2(RURAL_SOURCE / "canopy_height_m.tif", reference)
    shutil.copy2(RURAL_SOURCE / "vegetation_mask.tif", vegetation)
    shutil.copy2(RURAL_SOURCE / "valid_mask.tif", valid)
    shutil.copy2(RURAL_SOURCE / "source.json", case / "original_OpenCanopy_source.json")
    rgb_preview(rgb, case / "PREVIEW_rural_RGB.jpg")
    reference_preview(reference, case / "PREVIEW_LiDAR_canopy_0_to_35m.png", 35.0)

    original = json.loads((RURAL_SOURCE / "source.json").read_text(encoding="utf-8"))
    metadata = {
        "case": "rural/forest satellite + airborne LiDAR",
        "dataset": "AI4Forest Open-Canopy, official test split",
        "sample_id": original["sample_id"],
        "input_sensor": "SPOT pan-sharpened optical satellite imagery",
        "height_reference": "IGN airborne-LiDAR-derived canopy height above ground",
        "reference_kind_for_msr": "ndsm",
        "selection_integrity": (
            "Selected before inference by the objective rule: closest held-out test chip to 50 percent reference "
            "vegetation coverage among chips with at least 90 percent valid labels."
        ),
        "official_source": original["source"],
        "license": original["license"],
        "original_download_urls": original["urls"],
        "rgb_grid": grid(rgb),
        "reference_grid": grid(reference),
        "exact_grid_alignment": aligned(rgb, reference),
        "reference_statistics": height_stats(reference),
        "files": {
            rgb.name: sha256(rgb),
            reference.name: sha256(reference),
            vegetation.name: sha256(vegetation),
            valid.name: sha256(valid),
        },
    }
    write_json(case / "source_provenance_and_alignment.json", metadata)
    return metadata


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    urban = package_urban()
    rural = package_rural()
    write_json(
        OUTPUT_ROOT / "TWO_CASE_INDEX.json",
        {
            "purpose": "Reviewer-facing upload pairs with exact co-registered LiDAR-derived height references",
            "important": "The reference rasters are measured/derived labels, not Monocular Surface Reconstruction predictions.",
            "cases": [urban, rural],
        },
    )
    print(f"Packaged urban alignment: {urban['exact_grid_alignment']}")
    print(f"Packaged rural alignment: {rural['exact_grid_alignment']}")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
