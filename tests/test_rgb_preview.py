from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine
from fastapi.testclient import TestClient

from msr.api.app import create_app, InferenceRuntime
from msr.api.classification import resolve_source
from msr.inference.rgb_preview import coverage_report, predict_classes, tile_starts


class PixelModel:
    def __call__(self, image):
        labels = (image[:, 0] * 255).round().long().clamp(0, 5)
        return {"logits": torch.nn.functional.one_hot(labels, 6).permute(0, 3, 1, 2).float() * 10}


@pytest.mark.parametrize("shape", [(5, 6), (17, 29), (31, 7)])
def test_tiling_preserves_class_grid_validity_and_input(shape):
    labels = np.indices(shape)[1].astype(np.uint8) % 6
    rgb = np.stack([labels] * 3).astype(np.float32)
    original = rgb.copy()
    valid = np.ones(shape, bool); valid[0, 0] = False
    got = predict_classes(rgb, valid, PixelModel(), tile=8, overlap=2)
    np.testing.assert_array_equal(got[valid], labels[valid])
    assert got[0, 0] == 255
    np.testing.assert_array_equal(rgb, original)


def test_empty_invalid_and_bad_tiling_rejected():
    with pytest.raises(ValueError):
        predict_classes(np.zeros((3, 4, 4)), np.zeros((4, 4), bool), PixelModel())
    with pytest.raises(ValueError):
        tile_starts(1, 3, 3)


def test_coverage_denominator_excludes_unknown_and_no_invented_area():
    labels = np.array([[0, 1, 255], [2, 3, 4]], np.uint8)
    result = coverage_report(labels, {})
    assert result["valid_pixels"] == 5 and result["ignored_pixels"] == 1
    assert sum(x["coverage_percent"] for x in result["classes"]) == 100
    assert all(x["area_m2"] is None for x in result["classes"])
    geographic = coverage_report(labels, {"crs": "EPSG:4326", "transform": Affine.scale(.00001)})
    assert geographic["pixel_area_m2"] is None
    projected = coverage_report(labels, {"crs": "EPSG:32632", "transform": Affine(2, 1, 20, 0, -3, 40)})
    assert projected["pixel_area_m2"] == 6
    assert projected["classes"][0]["area_m2"] == 6


@pytest.mark.parametrize("job,demo", [("../escape", None), ("a"*32, "urban"), (None, "../urban"), (None, None)])
def test_source_selection_does_not_accept_paths(tmp_path, job, demo):
    with pytest.raises(ValueError):
        resolve_source(tmp_path, job, demo)


def test_classifier_route_cannot_call_height_or_modify_job(tmp_path):
    job = tmp_path / ("a"*32); job.mkdir()
    (job / "metadata.json").write_text("{}")
    (job / "input.png").write_bytes(b"source")
    (job / "height.tif").write_bytes(b"protected height output")
    class Runtime:
        device = "cpu"
        def predict(self, *args, **kwargs):
            raise AssertionError("Height prediction must never run for an overlay")
        def classify(self, source, output):
            assert source == job / "input.png"
            assert output != job and output.parent == tmp_path
            return {"height_pipeline_changed": False}
    client = TestClient(create_app(runtime=Runtime(), results_root=tmp_path))
    response = client.post("/api/classify", data={"job_id": job.name})
    assert response.status_code == 200
    assert response.json()["height_pipeline_changed"] is False
    assert (job / "height.tif").read_bytes() == b"protected height output"
    assert client.post("/api/classify", data={"job_id": "../../something"}).status_code == 422


def test_classify_shares_lock_and_does_not_load_height(monkeypatch, tmp_path):
    from msr.inference import rgb_preview
    runtime = InferenceRuntime(checkpoint=tmp_path / "not-a-model", device="cpu")
    def preview(*args):
        assert runtime._lock.locked()
        return {"ok": True}
    monkeypatch.setattr(rgb_preview, "run_preview", preview)
    assert runtime.classify(tmp_path / "input", tmp_path / "output") == {"ok": True}
    assert runtime._height_model is None and runtime._relative_predictor is None


def test_artifact_export_preserves_geogrid_and_nodata(monkeypatch, tmp_path):
    from msr.inference import rgb_preview
    source = tmp_path / "input.tif"
    rgb = np.stack([np.tile(np.arange(6, dtype=np.uint8), (6, 1))] * 3)
    transform = Affine(2, 0, 300000, 0, -2, 5000000)
    with rasterio.open(source, "w", driver="GTiff", width=6, height=6, count=3, dtype="uint8", crs="EPSG:32632", transform=transform) as dst:
        dst.write(rgb)
        mask = np.full((6, 6), 255, np.uint8); mask[0, 0] = 0; dst.write_mask(mask)
    monkeypatch.setattr(rgb_preview, "load_preview_model", lambda device: PixelModel())
    output = tmp_path / "output"
    report = rgb_preview.run_preview(source, output, "cpu")
    with rasterio.open(output / "classes.tif") as raster:
        assert raster.crs.to_epsg() == 32632 and raster.transform == transform
        assert raster.nodata == 255 and raster.read(1)[0, 0] == 255
    assert report["valid_pixels"] == 35
    assert report["height_pipeline_changed"] is False
    assert len((output / "classes.bin").read_bytes()) == 36


def test_high_bit_depth_rejected_before_model_load_without_altering_source(monkeypatch, tmp_path):
    from msr.inference import rgb_preview
    source = tmp_path / "input16.tif"
    with rasterio.open(source, "w", driver="GTiff", width=8, height=8, count=3,
                       dtype="uint16", transform=Affine(1, 0, 10, 0, -1, 20)) as dst:
        dst.write(np.full((3, 8, 8), 1000, np.uint16))
    before = source.read_bytes()
    def forbidden(*args):
        raise AssertionError("Unsupported radiometry must not start GPU classification")
    monkeypatch.setattr(rgb_preview, "load_preview_model", forbidden)
    with pytest.raises(ValueError, match="8-bit RGB"):
        rgb_preview.run_preview(source, tmp_path / "classes", "cpu")
    assert source.read_bytes() == before
    assert not (tmp_path / "classes").exists()
