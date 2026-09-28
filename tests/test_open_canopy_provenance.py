import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "data"
    / "audit_open_canopy_provenance.py"
)
SPEC = importlib.util.spec_from_file_location("audit_open_canopy_provenance", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _geometry(path: Path, *, split: str = "train") -> None:
    path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "properties": {
                            "split": split,
                            "lidar_year": 2021,
                            "image_name": "compressed_pansharpened_20210406371164.tif",
                            "lidar_acquisition_date": "20210905",
                            "lidar_url": "https://example.test/lidar.copc.laz",
                            "n_lidar_points": 123,
                            "X": "0836",
                            "Y": "6453",
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [836000, 6453000],
                                    [837000, 6453000],
                                    [837000, 6454000],
                                    [836000, 6454000],
                                    [836000, 6453000],
                                ]
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _source(root: Path, *, split: str = "train", region: str | None = None) -> None:
    sample = root / split / "oc_2021_836_6453"
    sample.mkdir(parents=True)
    feature = MODULE.load_open_canopy_features(root.parent / "geometry.geojson")[0]
    (sample / "source.json").write_text(
        json.dumps(
            {
                "sample_id": feature.sample_id,
                "official_split": split,
                "region": region or feature.region,
                "year": 2021,
                "image_name": feature.image_name,
                "bounds": [836100, 6453100, 836900, 6453900],
                "gsd_m": 1.5,
                "urls": MODULE.open_canopy_urls(feature),
                "source": MODULE.OPEN_CANOPY_DATASET_URL,
                "license": MODULE.OPEN_CANOPY_LICENSE,
            }
        ),
        encoding="utf-8",
    )
    manifest_name = MODULE.MANIFEST_NAMES[split]
    manifest = root / "manifests" / manifest_name
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id"])
        writer.writeheader()
        writer.writerow({"sample_id": feature.sample_id})


def _empty_manifests(root: Path, *excluded: str) -> None:
    for split, name in MODULE.MANIFEST_NAMES.items():
        if split in excluded:
            continue
        manifest = root / "manifests" / name
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("sample_id\n", encoding="utf-8")


def test_report_preserves_dates_locations_and_locked_test_role(tmp_path: Path) -> None:
    geometry = tmp_path / "geometry.geojson"
    _geometry(geometry)
    subset = tmp_path / "subset"
    _source(subset)
    _empty_manifests(subset, "train")

    report = MODULE.build_provenance_report(geometry, subset)

    assert report["passes"] is True
    assert report["counts"]["train"]["samples"] == 1
    record = report["records"][0]
    assert record["protocol_role"] == "learning"
    assert record["imagery_acquisition_date"] == "2021-04-06"
    assert record["lidar_acquisition_date"] == "2021-09-05"
    assert record["lidar_source_url"] == "https://example.test/lidar.copc.laz"


def test_report_fails_on_provenance_mismatch(tmp_path: Path) -> None:
    geometry = tmp_path / "geometry.geojson"
    _geometry(geometry)
    subset = tmp_path / "subset"
    _source(subset, region="wrong-region")
    _empty_manifests(subset, "train")

    report = MODULE.build_provenance_report(geometry, subset)

    assert report["passes"] is False
    assert any("Region mismatch" in issue for issue in report["issues"])
