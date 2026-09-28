"""Create a clearly synthetic scene for viewer integration tests, NOT evaluation."""
from pathlib import Path
import json

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin


def main():
    root = Path(__file__).resolve().parents[1] / "outputs/web_jobs/_synthetic_roof_geometry_test"
    root.mkdir(parents=True, exist_ok=True)
    size = 101
    y, x = np.mgrid[0:size, 0:size] / (size-1)
    building = (x >= .2) & (x <= .8) & (y >= .1) & (y <= .9)
    raw = np.where(building, 15-10*np.abs(x-.5), 0).astype("float32")
    slab = np.where(building, 14, 0).astype("float32")
    for name, values in {"raw.tif": raw, "slab.tif": slab, "semantic.tif": building.astype("float32")}.items():
        with rasterio.open(root / name, "w", driver="GTiff", width=size, height=size, count=1, dtype="float32", transform=from_origin(0, size*.3, .3, .3)) as dst:
            dst.write(values, 1)
    rgb = np.full((size, size, 3), [45, 55, 50], dtype="uint8")
    rgb[building & (x < .5)] = [80, 64, 52]
    rgb[building & (x >= .5)] = [150, 120, 98]
    Image.fromarray(rgb).save(root / "SYNTHETIC-NOT-A-MODEL-OUTPUT.png")
    structures = {"structures": [{"id": 1, "pixels": int(building.sum()), "center_x": .5, "center_y": .5, "width": .6, "depth": .8, "angle_rad": 0, "height_m": 14, "base_m": 0, "confidence": 1, "outline": [[.2,.1],[.8,.1],[.8,.9],[.2,.9]]}]}
    (root / "building_solids.json").write_text(json.dumps(structures), encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()
