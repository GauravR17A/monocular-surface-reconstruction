import numpy as np

from msr.data.open_canopy import (
    OpenCanopyFeature,
    forest_semantic_masks,
    imagery_acquisition_date,
    load_open_canopy_features,
    open_canopy_urls,
    stored_canopy_to_metres,
)


def test_open_canopy_decimetres_are_converted_to_metres() -> None:
    stored = np.array([0, 10, 253], dtype=np.uint16)
    np.testing.assert_allclose(stored_canopy_to_metres(stored), [0.0, 1.0, 25.3])


def test_forest_masks_exclude_buildings_and_unknown_classes() -> None:
    classification = np.array([[2, 3, 4, 5, 6, 1, 9]], dtype=np.uint8)
    source_valid = np.ones_like(classification, dtype=bool)
    valid, vegetation = forest_semantic_masks(classification, source_valid)
    np.testing.assert_array_equal(valid, [[True, True, True, True, False, False, True]])
    np.testing.assert_array_equal(
        vegetation, [[False, True, True, True, False, False, False]]
    )


def test_open_canopy_urls_preserve_official_year_and_names() -> None:
    feature = OpenCanopyFeature(
        sample_id="sample",
        split="train",
        year=2022,
        image_name="compressed_pansharpened_scene.tif",
        bounds=(0, 0, 1000, 1000),
        region="region",
        lidar_points=1,
    )
    urls = open_canopy_urls(feature)
    assert "/2022/spot/compressed_pansharpened_scene.tif" in urls["rgb"]
    assert "/2022/lidar/compressed_lidar_scene.tif" in urls["canopy"]
    assert "compressed_lidar_classification_scene.tif" in urls["classification"]


def test_image_acquisition_date_is_recovered_from_official_name() -> None:
    assert imagery_acquisition_date(
        "compressed_pansharpened_20210406371164.tif"
    ) == "2021-04-06"
    assert imagery_acquisition_date("compressed_pansharpened_scene.tif") is None


def test_geometry_loader_preserves_acquisition_and_lidar_provenance(tmp_path) -> None:
    geometry = tmp_path / "geometry.geojson"
    geometry.write_text(
        """{
          "features": [{
            "properties": {
              "split": "train",
              "lidar_year": 2021,
              "image_name": "compressed_pansharpened_20210406371164.tif",
              "lidar_acquisition_date": "20210905",
              "lidar_url": "https://example.test/source.copc.laz",
              "n_lidar_points": 123,
              "X": "0836",
              "Y": "6453"
            },
            "geometry": {
              "type": "Polygon",
              "coordinates": [[[836000, 6453000], [837000, 6453000],
                [837000, 6454000], [836000, 6454000], [836000, 6453000]]]
            }
          }]
        }""",
        encoding="utf-8",
    )

    feature = load_open_canopy_features(geometry)[0]

    assert feature.imagery_acquisition_date == "2021-04-06"
    assert feature.lidar_acquisition_date == "2021-09-05"
    assert feature.lidar_url == "https://example.test/source.copc.laz"
    assert feature.lidar_points == 123
