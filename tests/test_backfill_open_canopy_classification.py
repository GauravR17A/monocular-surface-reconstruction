import importlib.util
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "data"
    / "backfill_open_canopy_classification.py"
)
SPEC = importlib.util.spec_from_file_location("backfill_open_canopy_classification", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rgb(path: Path) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=3,
        dtype="uint8",
        crs="EPSG:2154",
        transform=from_origin(100, 200, 1.5, 1.5),
    ) as destination:
        destination.write(np.zeros((3, 16, 16), dtype=np.uint8))


def test_writes_versioned_sidecar_without_modifying_rgb(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.tif"
    _rgb(rgb)
    rgb_hash = MODULE._sha256(rgb)
    values = np.full((16, 16), 3, dtype=np.uint8)
    reference = {
        "width": 16,
        "height": 16,
        "crs": rasterio.crs.CRS.from_epsg(2154),
        "transform": from_origin(100, 200, 1.5, 1.5),
        "bounds": [100, 176, 124, 200],
    }

    metadata = MODULE.write_classification_artifact(
        tmp_path / "sidecar",
        sample_id="oc_2021_100_200",
        split="train",
        source_url="https://example.test/class.tif",
        rgb_path=rgb,
        values=values,
        reference=reference,
    )

    assert MODULE._sha256(rgb) == rgb_hash
    assert metadata["schema"].endswith(".v1")
    assert metadata["class_pixel_counts"] == {"3": 256}
    saved = json.loads((tmp_path / "sidecar" / "classification_source.json").read_text())
    assert saved["aligned_rgb_sha256"] == rgb_hash
    with rasterio.open(saved["classification_raster"]) as raster:
        assert raster.nodata == 255
        assert raster.transform == reference["transform"]
        np.testing.assert_array_equal(raster.read(1), values)
