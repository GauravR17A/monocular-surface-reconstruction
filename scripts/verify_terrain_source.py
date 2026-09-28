"""Cross-check browser evidence against the actual uploaded and generated GeoTIFFs.

Run after scripts/verify_terrain_analysis.mjs from the project root.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as project


def main() -> None:
    output = Path("outputs/runtime/terrain-final")
    report = json.loads((output / "browser-verification.json").read_text(encoding="utf-8"))
    job = report["products"][0]["url"].split("/results/")[1].split("/")[0]
    folder = Path("outputs/web_jobs") / job
    source = Path(report["source"])
    assert source.read_bytes() == (folder / "input.tif").read_bytes()
    with rasterio.open(folder / "dsm_absolute_m.tif") as raster:
        surface, tags, affine, shape, crs = raster.read(1), raster.tags(), raster.transform, raster.shape, raster.crs
    with rasterio.open(folder / "terrain_dem_aligned_m.tif") as raster:
        ground = raster.read(1)
    with rasterio.open(folder / "height_above_ground_m.tif") as raster:
        heights = raster.read(1)
    with rasterio.open(source) as raster:
        assert raster.shape == shape and raster.crs == crs and raster.transform == affine
    composition_error = float(np.max(np.abs(surface - ground - heights)))
    assert composition_error < .001
    spacing = (float(tags["MSR_PIXEL_SIZE_X_M"]), float(tags["MSR_PIXEL_SIZE_Y_M"]))
    x, y = shape[1] / 2 + .5, shape[0] / 2 + .5
    points = [affine * p for p in ((x, y), (x + 1, y), (x, y + 1))]
    lon, lat = project(crs, "EPSG:4326", [p[0] for p in points], [p[1] for p in points])
    # Independent local ground distances from GDAL-transformed coordinates.
    phi = math.radians(lat[0])
    denominator = 1 - 6.6943799901413165e-3 * math.sin(phi) ** 2
    east = 6378137 / math.sqrt(denominator) * math.cos(phi)
    north = 6378137 * (1 - 6.6943799901413165e-3) / denominator ** 1.5
    distances = [math.hypot(math.radians(lon[i] - lon[0]) * east, math.radians(lat[i] - lat[0]) * north) for i in (1, 2)]
    assert np.max(np.abs(np.array(distances) - spacing)) < .001
    u, v = report["inspection"]["u"], report["inspection"]["v"]
    col, row = round(u * (shape[1] - 1)), round(v * (shape[0] - 1))
    inspection = {"column": col, "row": row, "surface": float(surface[row, col]),
                  "terrain": float(ground[row, col]), "difference": float(surface[row, col] - ground[row, col])}
    assert f"{inspection['surface']:.1f} m" in report["inspection"]["text"]
    assert f"{inspection['terrain']:.1f} m" in report["inspection"]["text"]
    with (output / "downloaded-profile.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def bilinear(array, u, v):
        x, y = u * (array.shape[1] - 1), v * (array.shape[0] - 1)
        left, top = int(x), int(y)
        right, bottom = min(left + 1, array.shape[1] - 1), min(top + 1, array.shape[0] - 1)
        dx, dy = x - left, y - top
        return array[top, left] * (1-dx) * (1-dy) + array[top, right] * dx * (1-dy) + array[bottom, left] * (1-dx) * dy + array[bottom, right] * dx * dy

    errors = []
    for point in rows:
        for array, key in ((surface, "surface_m"), (ground, "terrain_reference_m")):
            errors.append(abs(bilinear(array, float(point["image_u"]), float(point["image_v"])) - float(point[key])))
    # Six-decimal UV output can move a sample by up to 0.000512 source pixels.
    assert max(errors) < .1
    a, b = rows[0], rows[-1]
    distance = math.hypot((float(b["image_u"]) - float(a["image_u"])) * (shape[1] - 1) * spacing[0],
                          (float(b["image_v"]) - float(a["image_v"])) * (shape[0] - 1) * spacing[1])
    assert abs(distance - float(b["distance_m"])) < .05
    result = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "source_unchanged": True,
              "job": job, "dimensions": shape, "crs": str(crs), "gsd_m": spacing,
              "independent_local_ground_spacing_m": distances, "elevation_min_m": float(surface.min()),
              "elevation_max_m": float(surface.max()), "relief_m": float(surface.max()-surface.min()),
              "composition_max_error_m": composition_error, "inspection": inspection,
              "profile_samples": len(rows), "profile_max_error_m_including_rounded_uv": float(max(errors)),
              "profile_distance_m": float(b["distance_m"])}
    (output / "source-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
