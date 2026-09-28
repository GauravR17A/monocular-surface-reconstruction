"""Build reproducible sparse and hilly candidate packs for the research demo.

Sparse candidates are selected from the untouched multi-domain test manifest using
only reference building coverage. Hilly candidates are cloud-screened Sentinel-2
L2A true-colour crops paired with aligned Copernicus DEM GLO-30 terrain data.

The Copernicus DEM is a calibration datum for the hilly functional demo. It must
not be reused as an "independent" truth raster when reporting model accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform, transform_bounds
import requests


EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
SENTINEL_COLLECTION = "sentinel-2-l2a"
SENTINEL_SOURCE = "https://registry.opendata.aws/sentinel-2-l2a-cogs/"
COPERNICUS_SOURCE = "https://registry.opendata.aws/copernicus-dem/"

HILLY_SITES = (
    {"slug": "manali_himachal", "label": "Manali, Himachal Pradesh", "lat": 32.2432, "lon": 77.1892},
    {"slug": "nainital_uttarakhand", "label": "Nainital, Uttarakhand", "lat": 29.3919, "lon": 79.4542},
    {"slug": "darjeeling_west_bengal", "label": "Darjeeling, West Bengal", "lat": 27.0410, "lon": 88.2663},
    {"slug": "munnar_kerala", "label": "Munnar, Kerala", "lat": 10.0889, "lon": 77.0595},
    {"slug": "aizawl_mizoram", "label": "Aizawl, Mizoram", "lat": 23.7271, "lon": 92.7176},
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_rgb_preview(path: Path, size: int = 512) -> Image.Image:
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        image = Image.open(path).convert("RGB")
    else:
        with rasterio.open(path) as dataset:
            values = dataset.read((1, 2, 3), out_shape=(3, size, size), resampling=Resampling.bilinear)
        if values.dtype != np.uint8:
            output = np.zeros(values.shape, dtype=np.uint8)
            for band in range(3):
                channel = values[band].astype(np.float32)
                finite = np.isfinite(channel)
                low, high = np.percentile(channel[finite], (2, 98)) if finite.any() else (0.0, 1.0)
                output[band] = np.clip((channel - low) * 255.0 / max(high - low, 1e-6), 0, 255)
            values = output
        image = Image.fromarray(np.moveaxis(values, 0, 2), mode="RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    return image


def _contact_sheet(cards: list[tuple[Path, str, str]], output_path: Path) -> None:
    if not cards:
        return
    card_width, image_height, footer_height = 420, 330, 76
    columns = 2
    rows = math.ceil(len(cards) / columns)
    sheet = Image.new("RGB", (columns * card_width, rows * (image_height + footer_height)), "#edf5f3")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (preview, title, detail) in enumerate(cards):
        col, row = index % columns, index // columns
        x, y = col * card_width, row * (image_height + footer_height)
        image = Image.open(preview).convert("RGB")
        image.thumbnail((card_width - 20, image_height - 20), Image.Resampling.LANCZOS)
        paste_x = x + (card_width - image.width) // 2
        paste_y = y + (image_height - image.height) // 2
        sheet.paste(image, (paste_x, paste_y))
        draw.rectangle((x, y + image_height, x + card_width - 1, y + image_height + footer_height - 1), fill="#123c46")
        draw.text((x + 12, y + image_height + 10), title, fill="white", font=font)
        draw.text((x + 12, y + image_height + 35), detail, fill="#bdebdc", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94)


def _surface_stats(path: Path) -> dict[str, float]:
    with rasterio.open(path) as dataset:
        values = dataset.read(1).astype(np.float32)
        valid = np.isfinite(values)
        if dataset.nodata is not None:
            valid &= values != dataset.nodata
    valid_values = values[valid]
    if not valid_values.size:
        raise ValueError(f"No valid surface pixels in {path}")
    elevated = valid_values > 2.0
    positive = valid_values[valid_values > 0.25]
    return {
        "elevated_fraction": float(elevated.mean()),
        "positive_fraction": float((valid_values > 0.25).mean()),
        "mean_positive_height_m": float(positive.mean()) if positive.size else 0.0,
        "p95_height_m": float(np.percentile(valid_values, 95)),
        "max_height_m": float(valid_values.max()),
    }


def prepare_sparse(manifest_path: Path, output_root: Path, limit: int) -> list[dict[str, Any]]:
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("landscape") == "urban"]

    measured: list[dict[str, Any]] = []
    for row in rows:
        stats = _surface_stats(Path(row["surface_path"]))
        if 0.005 <= stats["elevated_fraction"] <= 0.12:
            measured.append({**row, **stats})
    if not measured:
        raise RuntimeError("No objective sparse candidates were found in the held-out test split")

    targets = np.linspace(0.012, 0.09, limit)
    selected: list[dict[str, Any]] = []
    used_samples: set[str] = set()
    used_regions: set[str] = set()
    for target in targets:
        ranked = sorted(
            measured,
            key=lambda row: (
                row["region"] in used_regions,
                abs(float(row["elevated_fraction"]) - float(target)),
                row["sample_id"],
            ),
        )
        choice = next(row for row in ranked if row["sample_id"] not in used_samples)
        selected.append(choice)
        used_samples.add(choice["sample_id"])
        used_regions.add(choice["region"])

    sparse_root = output_root / "sparse_candidates"
    cards: list[tuple[Path, str, str]] = []
    index_rows: list[dict[str, Any]] = []
    for number, row in enumerate(selected, start=1):
        candidate = sparse_root / f"{number:02d}_{row['sample_id']}"
        candidate.mkdir(parents=True, exist_ok=True)
        rgb_destination = candidate / ("rgb" + Path(row["rgb_path"]).suffix.lower())
        reference_destination = candidate / "reference_building_height_m.tif"
        shutil.copy2(row["rgb_path"], rgb_destination)
        shutil.copy2(row["surface_path"], reference_destination)
        source_path = Path(row["rgb_path"]).parent / "source.json"
        if source_path.is_file():
            shutil.copy2(source_path, candidate / "original_source.json")
        preview = candidate / "preview.jpg"
        _read_rgb_preview(rgb_destination).save(preview, quality=94)
        metadata = {
            "candidate_type": "sparse held-out validation scene",
            "selection_rule": "selected using reference building coverage only; predictions were not inspected",
            "split": "test",
            "sample_id": row["sample_id"],
            "region": row["region"],
            "elevated_reference_threshold_m": 2.0,
            "elevated_reference_fraction": row["elevated_fraction"],
            "positive_reference_fraction": row["positive_fraction"],
            "reference_mean_positive_height_m": row["mean_positive_height_m"],
            "reference_p95_height_m": row["p95_height_m"],
            "reference_max_height_m": row["max_height_m"],
        }
        _write_json(candidate / "candidate_metadata.json", metadata)
        index_rows.append(metadata)
        cards.append(
            (
                preview,
                f"{number}. {row['region'].split('_')[-1]}",
                f"reference area >2 m: {100 * float(row['elevated_fraction']):.1f}%",
            )
        )

    _contact_sheet(cards, sparse_root / "SPARSE_CANDIDATES_CONTACT_SHEET.jpg")
    with (sparse_root / "candidate_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    return index_rows


def _target_grid(lat: float, lon: float, size_km: float, resolution_m: float) -> tuple[str, Any, int, int]:
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    crs = f"EPSG:{epsg}"
    centre_x, centre_y = transform("EPSG:4326", crs, [lon], [lat])
    width = height = int(round(size_km * 1000 / resolution_m))
    half = width * resolution_m / 2
    affine = from_origin(centre_x[0] - half, centre_y[0] + half, resolution_m, resolution_m)
    return crs, affine, width, height


def _search_sentinel(lat: float, lon: float, max_cloud: float) -> list[dict[str, Any]]:
    delta = 0.025
    payload = {
        "collections": [SENTINEL_COLLECTION],
        "bbox": [lon - delta, lat - delta, lon + delta, lat + delta],
        "datetime": "2024-01-01T00:00:00Z/2026-08-30T23:59:59Z",
        "limit": 50,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
    }
    response = requests.post(EARTH_SEARCH_URL, json=payload, timeout=60)
    response.raise_for_status()
    features = response.json().get("features", [])
    return sorted(features, key=lambda item: (item["properties"].get("eo:cloud_cover", 100.0), item["id"]))


def _read_remote_aligned(
    url: str,
    crs: str,
    affine: Any,
    width: int,
    height: int,
    indexes: int | tuple[int, ...],
    resampling: Resampling,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(url) as source:
        with WarpedVRT(
            source,
            crs=crs,
            transform=affine,
            width=width,
            height=height,
            resampling=resampling,
        ) as aligned:
            values = aligned.read(indexes)
            masks = aligned.read_masks(indexes)
    if masks.ndim == 3:
        valid = np.all(masks > 0, axis=0)
    else:
        valid = masks > 0
    return values, valid


def _choose_sentinel_item(
    items: list[dict[str, Any]], crs: str, affine: Any, width: int, height: int
) -> tuple[dict[str, Any], float, float]:
    evaluated: list[tuple[float, float, dict[str, Any]]] = []
    for item in items[:12]:
        scl_url = item.get("assets", {}).get("scl", {}).get("href")
        visual_url = item.get("assets", {}).get("visual", {}).get("href")
        if not scl_url or not visual_url:
            continue
        scl, valid = _read_remote_aligned(
            scl_url, crs, affine, width, height, 1, Resampling.nearest
        )
        usable = valid & (scl != 0)
        valid_fraction = float(usable.mean())
        bad = np.isin(scl, [1, 3, 8, 9, 10, 11]) & usable
        local_bad_fraction = float(bad.sum() / max(usable.sum(), 1))
        evaluated.append((local_bad_fraction, -valid_fraction, item))
    if not evaluated:
        raise RuntimeError("No readable Sentinel-2 visual/SCL candidates were found")
    local_bad, negative_valid, selected = min(evaluated, key=lambda value: (value[0], value[1]))
    return selected, float(local_bad), float(-negative_valid)


def _copernicus_url(lat_degree: int, lon_degree: int) -> str:
    lat_token = f"N{lat_degree:02d}" if lat_degree >= 0 else f"S{abs(lat_degree):02d}"
    lon_token = f"E{lon_degree:03d}" if lon_degree >= 0 else f"W{abs(lon_degree):03d}"
    stem = f"Copernicus_DSM_COG_10_{lat_token}_00_{lon_token}_00_DEM"
    return f"https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/{stem}/{stem}.tif"


def _read_copernicus_mosaic(
    crs: str, affine: Any, width: int, height: int
) -> tuple[np.ndarray, list[str]]:
    bounds = rasterio.transform.array_bounds(height, width, affine)
    west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    output = np.full((height, width), np.nan, dtype=np.float32)
    urls: list[str] = []
    for lat_degree in range(math.floor(south), math.floor(north) + 1):
        for lon_degree in range(math.floor(west), math.floor(east) + 1):
            url = _copernicus_url(lat_degree, lon_degree)
            try:
                values, valid = _read_remote_aligned(
                    url, crs, affine, width, height, 1, Resampling.bilinear
                )
            except rasterio.errors.RasterioIOError:
                continue
            finite = valid & np.isfinite(values)
            output[finite] = values[finite]
            urls.append(url)
    if not np.isfinite(output).any():
        raise RuntimeError("Copernicus DEM returned no valid elevation pixels")
    return output, urls


def _write_raster(path: Path, values: np.ndarray, crs: str, affine: Any, nodata: float | int | None) -> None:
    bands = values if values.ndim == 3 else values[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "width": bands.shape[2],
        "height": bands.shape[1],
        "count": bands.shape[0],
        "dtype": str(bands.dtype),
        "crs": crs,
        "transform": affine,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": nodata,
    }
    temporary = path.with_name(path.stem + ".partial.tif")
    with rasterio.open(temporary, "w", **profile) as destination:
        destination.write(bands)
    temporary.replace(path)


def prepare_hilly(output_root: Path, size_km: float, resolution_m: float) -> list[dict[str, Any]]:
    hilly_root = output_root / "hilly_candidates"
    cards: list[tuple[Path, str, str]] = []
    index_rows: list[dict[str, Any]] = []
    for number, site in enumerate(HILLY_SITES, start=1):
        candidate = hilly_root / f"{number:02d}_{site['slug']}"
        candidate.mkdir(parents=True, exist_ok=True)
        crs, affine, width, height = _target_grid(
            float(site["lat"]), float(site["lon"]), size_km, resolution_m
        )
        items = _search_sentinel(float(site["lat"]), float(site["lon"]), max_cloud=20.0)
        item, local_bad_fraction, valid_fraction = _choose_sentinel_item(
            items, crs, affine, width, height
        )
        visual_url = item["assets"]["visual"]["href"]
        rgb, rgb_valid = _read_remote_aligned(
            visual_url, crs, affine, width, height, (1, 2, 3), Resampling.bilinear
        )
        rgb = np.where(rgb_valid[None], rgb, 0).astype(np.uint8)
        terrain, terrain_urls = _read_copernicus_mosaic(crs, affine, width, height)
        finite = np.isfinite(terrain)
        fill = float(np.nanmedian(terrain))
        terrain = np.where(finite, terrain, fill).astype(np.float32)
        low, high = np.percentile(terrain[finite], (5, 95))
        relief = float(high - low)

        rgb_path = candidate / "input_rgb_georeferenced.tif"
        dem_path = candidate / "terrain_datum_copernicus_glo30_m.tif"
        preview = candidate / "preview.jpg"
        _write_raster(rgb_path, rgb, crs, affine, nodata=0)
        _write_raster(dem_path, terrain, crs, affine, nodata=None)
        Image.fromarray(np.moveaxis(rgb, 0, 2), mode="RGB").save(preview, quality=94)
        metadata = {
            "candidate_type": "hilly georeferenced functional demo",
            "location": site["label"],
            "latitude": site["lat"],
            "longitude": site["lon"],
            "size_km": size_km,
            "output_resolution_m": resolution_m,
            "crs": crs,
            "sentinel_item_id": item["id"],
            "acquired_at": item["properties"].get("datetime"),
            "catalog_scene_cloud_percent": item["properties"].get("eo:cloud_cover"),
            "local_cloud_shadow_snow_fraction": local_bad_fraction,
            "local_valid_fraction": valid_fraction,
            "sentinel_visual_cog": visual_url,
            "copernicus_dem_cogs": terrain_urls,
            "terrain_p05_m": float(low),
            "terrain_p95_m": float(high),
            "terrain_relief_p95_minus_p05_m": relief,
            "sentinel_source": SENTINEL_SOURCE,
            "copernicus_dem_source": COPERNICUS_SOURCE,
            "accuracy_note": (
                "Copernicus DEM is used as the low-resolution terrain calibration datum. "
                "It is not an independent reference for a model RMSE claim in this scene."
            ),
        }
        _write_json(candidate / "source_metadata.json", metadata)
        index_rows.append(metadata)
        cards.append(
            (
                preview,
                f"{number}. {site['label']}",
                f"relief {relief:.0f} m | local obstruction {100 * local_bad_fraction:.1f}%",
            )
        )
        print(
            f"[{number}/{len(HILLY_SITES)}] {site['label']}: "
            f"{item['id']}, local obstruction {100 * local_bad_fraction:.1f}%, relief {relief:.0f} m",
            flush=True,
        )

    _contact_sheet(cards, hilly_root / "HILLY_CANDIDATES_CONTACT_SHEET.jpg")
    with (hilly_root / "candidate_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    return index_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/multidomain_v2/manifests_with_priors/test.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/presentation/monday_demo/landscape_candidates"),
    )
    parser.add_argument("--mode", choices=("all", "sparse", "hilly"), default="all")
    parser.add_argument("--sparse-count", type=int, default=6)
    parser.add_argument("--hilly-size-km", type=float, default=7.68)
    parser.add_argument("--resolution-m", type=float, default=10.0)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "purpose": "Manual selection pack for honest sparse and hilly research demonstrations",
        "selection_integrity": (
            "Sparse scenes are chosen without prediction access. Hilly scenes are chosen by "
            "local cloud obstruction and terrain relief, not by model error."
        ),
    }
    if args.mode in {"all", "sparse"}:
        summary["sparse_candidates"] = prepare_sparse(
            args.manifest.resolve(), args.output_root.resolve(), args.sparse_count
        )
    if args.mode in {"all", "hilly"}:
        summary["hilly_candidates"] = prepare_hilly(
            args.output_root.resolve(), args.hilly_size_km, args.resolution_m
        )
    _write_json(args.output_root / "candidate_pack_summary.json", summary)
    print(f"Candidate pack ready: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
