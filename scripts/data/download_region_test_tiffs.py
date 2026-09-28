"""Build a small, attributed RGB GeoTIFF import set from public EOX map tiles."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds

SCENES = [
    ("01_urban_new_delhi", "Urban / New Delhi", 77.2167, 28.6333),
    ("02_plains_punjab", "Plains and farms / Punjab", 75.68, 30.85),
    ("03_forest_dandeli", "Forest / Dandeli", 74.40, 15.12),
    ("04_hills_munnar", "Hills / Munnar", 77.06, 10.09),
    ("05_desert_thar", "Desert / Thar", 70.50, 26.85),
    ("06_coast_goa", "Coast / Goa", 73.77, 15.53),
    ("07_wetlands_sundarbans", "Wetlands / Sundarbans", 88.79, 21.90),
    ("08_rocky_ladakh", "Rocky mountains / Ladakh", 77.62, 34.17),
]
ZOOM, SIZE = 14, 1024
ATTRIBUTION = "Sentinel-2 cloudless 2024 by EOX IT Services GmbH; contains modified Copernicus Sentinel data 2024."


def main():
    output = Path.home() / "Downloads" / "Monocular Surface Reconstruction_Region_TIFFs"
    output.mkdir(parents=True, exist_ok=True)
    cache = Path("outputs/runtime/region-tiff-cache")
    cache.mkdir(parents=True, exist_ok=True)
    records, previews = [], []

    def fetch(item):
        x, y = item
        url = f"https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{ZOOM}/{y}/{x}.jpg"
        tile_path = cache / f"{ZOOM}_{x}_{y}.jpg"
        if tile_path.exists():
            payload = tile_path.read_bytes()
        else:
            for attempt in range(3):
                try:
                    with urlopen(Request(url, headers={"User-Agent": "Monocular Surface Reconstruction-region-preview/1.0"}), timeout=35) as response:
                        payload = response.read(2 * 1024 * 1024)
                    with Image.open(BytesIO(payload)) as decoded:
                        decoded.verify()
                    tile_path.write_bytes(payload)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        image = np.asarray(Image.open(BytesIO(payload)).convert("RGB"))
        if image.shape != (256, 256, 3):
            raise ValueError(f"Unexpected source tile dimensions: {image.shape}")
        return x, y, image, {"url": url, "sha256": sha256(payload).hexdigest()}

    for slug, label, longitude, latitude in SCENES:
        n = 2 ** ZOOM
        x0 = math.floor((longitude + 180) / 360 * n) - 1
        y0 = math.floor((1 - math.asinh(math.tan(math.radians(latitude))) / math.pi) / 2 * n) - 1
        mosaic = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        tiles = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for x, y, pixels, provenance in pool.map(fetch, [(x, y) for y in range(y0, y0 + 4) for x in range(x0, x0 + 4)]):
                mosaic[(y-y0)*256:(y-y0+1)*256, (x-x0)*256:(x-x0+1)*256] = pixels
                tiles.append(provenance)
        half = 20037508.342789244
        span = 2 * half / n
        bounds = [-half + x0*span, half-(y0+4)*span, -half+(x0+4)*span, half-y0*span]
        target = output / f"{slug}_s2cloudless_2024.tif"
        sidecar = target.with_suffix(".source.json")
        if target.exists() and not sidecar.exists():
            raise FileExistsError(f"Refusing to overwrite an unrelated existing file: {target}")
        with rasterio.open(target, "w", driver="GTiff", width=SIZE, height=SIZE, count=3,
                           dtype="uint8", crs="EPSG:3857", transform=from_bounds(*bounds, SIZE, SIZE),
                           tiled=True, blockxsize=256, blockysize=256, compress="deflate", photometric="RGB") as raster:
            raster.write(mosaic.transpose(2, 0, 1))
            raster.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            raster.update_tags(AREA_OR_POINT="Area", SOURCE=ATTRIBUTION, REGION=label,
                               PRODUCT="RGB optical imagery; not an elevation raster", SOURCE_URL="https://maps.eox.at/")
        with rasterio.open(target) as raster:
            assert raster.count == 3 and raster.crs.to_epsg() == 3857 and raster.shape == (SIZE, SIZE)
            assert np.array_equal(raster.read().transpose(1, 2, 0), mosaic)
        record = {"file": str(target), "region": label, "requested_center_lonlat": [longitude, latitude],
                  "bounds_epsg3857": bounds, "crs": "EPSG:3857", "dimensions": [SIZE, SIZE], "bands": "RGB uint8",
                  "map_pixel_size": span/256, "zoom": ZOOM, "attribution": ATTRIBUTION,
                  "source": "https://maps.eox.at/", "license_information": "https://cloudless.eox.at/",
                  "method": "Mosaic of 16 public JPEG map tiles repackaged losslessly as a georeferenced TIFF; not raw multispectral or elevation data.",
                  "downloaded_utc": datetime.now(timezone.utc).isoformat(), "tiles": tiles,
                  "file_sha256": sha256(target.read_bytes()).hexdigest(), "bytes": target.stat().st_size}
        sidecar.write_text(json.dumps(record, indent=2), encoding="utf-8")
        preview = Image.fromarray(mosaic)
        preview.resize((512, 512), Image.Resampling.LANCZOS).save(output / f"{slug}_preview.jpg", quality=90)
        previews.append((label, preview))
        records.append(record)
        print(f"READY {target.name}: {target.stat().st_size:,} bytes", flush=True)
    (output / "manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    sheet = Image.new("RGB", (4*320, 2*354), "#0c1714")
    draw = ImageDraw.Draw(sheet)
    for index, (label, preview) in enumerate(previews):
        x, y = index % 4 * 320, index // 4 * 354
        sheet.paste(preview.resize((312, 312), Image.Resampling.LANCZOS), (x+4, y+4))
        draw.text((x+10, y+326), label, fill="#d7eee6")
    sheet.save(output / "REGION_PREVIEWS.jpg", quality=94)
    (output / "README.txt").write_text(
        "Monocular Surface Reconstruction region import set\n\n" + "\n".join(f"{Path(r['file']).name}: {r['region']}" for r in records)
        + "\n\nAll eight are 1024 x 1024, three-band RGB GeoTIFFs with EPSG:3857 coordinates.\n"
        "They were assembled from real EOX Sentinel-2 cloudless 2024 map tiles. They are optical images, not DEMs or ground truth.\n"
        "Use Upload satellite image in Monocular Surface Reconstruction; leave public terrain enabled, or attach a DEM.\n"
        "The original joshimath_s2cloudless_2024.tif remains one folder above for the snow/high-mountain test.\n"
        "These are representative regions, not exhaustive coverage of every TIFF encoding or sensor.\n\n"
        + ATTRIBUTION + "\nSource and provider terms: https://maps.eox.at/ and https://cloudless.eox.at/\n"
        "Per-file source JSON contains source URLs, tile hashes, coordinates and TIFF hashes.\n", encoding="utf-8")
    print(f"COMPLETE {len(records)} validated TIFFs in {output}", flush=True)


if __name__ == "__main__":
    main()
