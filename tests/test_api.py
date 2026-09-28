from pathlib import Path

from fastapi.testclient import TestClient

from msr.api.app import create_app


class FakeRuntime:
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, input_path: Path, output_dir: Path, **kwargs):
        self.calls.append({"input_path": input_path, "output_dir": output_dir, **kwargs})
        for name in ("texture.jpg", "presentation.tif", "height.tif", "relative.tif", "rdsm.tif", "structures.json"):
            (output_dir / name).write_bytes(b"test")
        return {
            "texture": "texture.jpg",
            "presentation_height": "presentation.tif",
            "raw_height": "height.tif",
            "relative_depth": "relative.tif",
            "rdsm": "rdsm.tif",
            "building_probability": None,
            "semantic_class": None,
            "structures": "structures.json",
            "metadata": {"georeferenced": False},
        }


def test_api_health_and_prediction_contract(tmp_path: Path) -> None:
    client = TestClient(create_app(runtime=FakeRuntime(), results_root=tmp_path))
    assert client.get("/api/health").json() == {"status": "ready", "device": "cpu"}

    response = client.post(
        "/api/predict",
        files={"image": ("scene.jpg", b"fake-image", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["height_url"].endswith("/presentation.tif")
    assert payload["texture_url"].endswith("/texture.jpg")
    assert payload["structures_url"].endswith("/structures.json")
    assert payload["uploaded_bytes"] == len(b"fake-image")
    assert payload["metadata"]["georeferenced"] is False


def test_api_streams_optional_dem_to_runtime(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime=runtime, results_root=tmp_path))

    response = client.post(
        "/api/predict",
        files={
            "image": ("scene.tif", b"fake-geotiff", "image/tiff"),
            "dem": ("terrain.tif", b"fake-dem", "image/tiff"),
        },
    )

    assert response.status_code == 200
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["gcps_path"] is None
    dem_path = runtime.calls[0]["dem_path"]
    assert isinstance(dem_path, Path)
    assert dem_path.read_bytes() == b"fake-dem"


def test_api_streams_optional_gcps_to_runtime(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime=runtime, results_root=tmp_path))

    response = client.post(
        "/api/predict",
        files={
            "image": ("scene.png", b"fake-image", "image/png"),
            "gcps": (
                "points.csv",
                b"row,col,elevation_m\n0,0,100\n",
                "text/csv",
            ),
        },
    )

    assert response.status_code == 200
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["dem_path"] is None
    gcps_path = runtime.calls[0]["gcps_path"]
    assert isinstance(gcps_path, Path)
    assert gcps_path.read_bytes().startswith(b"row,col,elevation_m")


def test_api_streams_optional_reference_and_kind_to_runtime(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime=runtime, results_root=tmp_path))

    response = client.post(
        "/api/predict",
        files={
            "image": ("scene.tif", b"fake-image", "image/tiff"),
            "reference": ("reference.tif", b"fake-reference", "image/tiff"),
        },
        data={"reference_kind": "ndsm"},
    )

    assert response.status_code == 200
    assert len(runtime.calls) == 1
    reference_path = runtime.calls[0]["reference_path"]
    assert isinstance(reference_path, Path)
    assert reference_path.read_bytes() == b"fake-reference"
    assert runtime.calls[0]["reference_kind"] == "ndsm"


def test_api_rejects_absolute_reference_without_valid_kind(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime=runtime, results_root=tmp_path))

    response = client.post(
        "/api/predict",
        files={
            "image": ("scene.tif", b"fake-image", "image/tiff"),
            "reference": ("reference.tif", b"fake-reference", "image/tiff"),
        },
        data={"reference_kind": "terrain"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Reference kind must be nDSM or DSM."
    assert runtime.calls == []


def test_api_rejects_dem_and_gcps_together(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime=runtime, results_root=tmp_path))

    response = client.post(
        "/api/predict",
        files={
            "image": ("scene.tif", b"fake-geotiff", "image/tiff"),
            "dem": ("terrain.tif", b"fake-dem", "image/tiff"),
            "gcps": ("points.csv", b"row,col,elevation_m\n", "text/csv"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Choose either a DEM or GCP CSV, not both."
    assert runtime.calls == []


def test_api_passes_automatic_terrain_choice_to_runtime(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime=runtime, results_root=tmp_path))

    response = client.post(
        "/api/predict",
        files={"image": ("scene.tif", b"fake-geotiff", "image/tiff")},
        data={"auto_dem": "true"},
    )

    assert response.status_code == 200
    assert runtime.calls[0]["auto_dem"] is True


def test_api_rejects_multiple_terrain_calibration_sources(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime=runtime, results_root=tmp_path))

    response = client.post(
        "/api/predict",
        files={
            "image": ("scene.tif", b"fake-geotiff", "image/tiff"),
            "dem": ("terrain.tif", b"fake-dem", "image/tiff"),
        },
        data={"auto_dem": "true"},
    )

    assert response.status_code == 400
    assert "not more than one" in response.json()["detail"]
    assert runtime.calls == []
