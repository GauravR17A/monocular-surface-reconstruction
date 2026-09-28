from pathlib import Path

from affine import Affine
import numpy as np
import pytest
import rasterio

from msr.data.surface_dataset import SurfaceSampleRecord
from msr.evaluation.locked_benchmark import (
    RADIOMETRIC_VARIANTS,
    benchmark_dataset_digest,
    load_benchmark_reference,
    make_radiometric_variant,
    sha256_path_tree,
)
from msr.io.raster import read_rgb_raster


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    nodata: float | None = None,
) -> None:
    bands = values if values.ndim == 3 else values[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=bands.shape[-1],
        height=bands.shape[-2],
        count=bands.shape[0],
        dtype=str(bands.dtype),
        transform=Affine.identity(),
        nodata=nodata,
    ) as target:
        target.write(bands)


def test_radiometric_variants_are_deterministic_and_keep_invalid_pixels_zero() -> None:
    rgb = np.linspace(12, 190, 3 * 19 * 23, dtype=np.float32).reshape(3, 19, 23)
    valid = np.ones((19, 23), dtype=bool)
    valid[:2, :3] = False

    raw = make_radiometric_variant(rgb, valid, "raw")
    stretched = make_radiometric_variant(rgb, valid, "stretch01_99")
    first_jpeg = make_radiometric_variant(rgb, valid, "jpeg92")
    second_jpeg = make_radiometric_variant(rgb, valid, "jpeg92")

    assert raw.shape == stretched.shape == first_jpeg.shape == rgb.shape
    assert np.array_equal(first_jpeg, second_jpeg)
    assert np.all(raw[:, ~valid] == 0)
    assert np.all(stretched[:, ~valid] == 0)
    assert np.all(first_jpeg[:, ~valid] == 0)
    assert not np.array_equal(stretched[:, valid], raw[:, valid])

    for name in RADIOMETRIC_VARIANTS:
        first = make_radiometric_variant(rgb, valid, name)
        second = make_radiometric_variant(rgb, valid, name)
        assert np.array_equal(first, second)
        assert first.shape == rgb.shape
        assert np.all(first[:, ~valid] == 0)
        if name != "raw":
            assert float(first[:, valid].min()) >= 0
            assert float(first[:, valid].max()) <= 255


def test_full_raster_reference_preserves_training_manifest_semantics(
    tmp_path: Path,
) -> None:
    height, width = 11, 17
    rgb_path = tmp_path / "rgb.tif"
    surface_path = tmp_path / "height.tif"
    vegetation_path = tmp_path / "vegetation.tif"
    valid_path = tmp_path / "valid.tif"
    _write_raster(rgb_path, np.full((3, height, width), 90, dtype=np.uint8))
    surface = np.arange(height * width, dtype=np.float32).reshape(height, width) / 10
    _write_raster(surface_path, surface, nodata=np.nan)
    vegetation = np.zeros((height, width), dtype=np.uint8)
    vegetation[:, width // 2 :] = 1
    _write_raster(vegetation_path, vegetation)
    valid = np.ones((height, width), dtype=np.uint8)
    valid[0, :] = 0
    _write_raster(valid_path, valid)
    record = SurfaceSampleRecord(
        sample_id="forest-control",
        region="r1",
        landscape="forest",
        rgb_path=rgb_path,
        surface_path=surface_path,
        target_kind="ndsm",
        vegetation_mask_path=vegetation_path,
        valid_mask_path=valid_path,
    )

    source = read_rgb_raster(rgb_path)
    reference = load_benchmark_reference(record, source, height_max_m=200)

    assert reference.target_m.shape == (height, width)
    assert reference.regression_mask.shape == (height, width)
    assert not reference.regression_mask[0].any()
    assert reference.regression_mask[1:].all()
    assert not reference.vegetation_mask[:, : width // 2].any()
    assert reference.vegetation_mask[1:, width // 2 :].all()
    assert reference.alignments["surface"] == "pixel_aligned_same_shape"


def test_building_only_background_is_not_scored(tmp_path: Path) -> None:
    rgb_path = tmp_path / "urban_rgb.tif"
    height_path = tmp_path / "building_height.tif"
    _write_raster(rgb_path, np.full((3, 8, 9), 128, dtype=np.uint8))
    target = np.zeros((8, 9), dtype=np.float32)
    target[2:6, 3:7] = 6.0
    _write_raster(height_path, target)
    record = SurfaceSampleRecord(
        sample_id="urban-control",
        region="city-held-out",
        landscape="urban",
        rgb_path=rgb_path,
        surface_path=height_path,
        target_kind="building_height",
    )

    reference = load_benchmark_reference(record, read_rgb_raster(rgb_path))

    assert int(reference.regression_mask.sum()) == 16
    np.testing.assert_array_equal(reference.regression_mask, reference.building_mask)


def test_highbuild_benchmark_separates_semantics_from_strict_height_support(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "rgb.tif"
    height_path = tmp_path / "building_height_m.tif"
    building_path = tmp_path / "building_footprints.tif"
    valid_path = tmp_path / "measured_height_valid.tif"
    _write_raster(rgb_path, np.full((3, 4, 4), 128, dtype=np.uint8))
    height = np.zeros((4, 4), dtype=np.float32)
    height[0, 0] = 8.0
    height[0, 1] = 12.0
    _write_raster(height_path, height)
    buildings = np.zeros((4, 4), dtype=np.uint8)
    buildings[0, :2] = 1
    _write_raster(building_path, buildings)
    measured = np.zeros((4, 4), dtype=np.uint8)
    measured[0, 0] = 1
    _write_raster(valid_path, measured)
    record = SurfaceSampleRecord(
        sample_id="highbuild-control",
        region="city-held-out",
        landscape="urban",
        rgb_path=rgb_path,
        surface_path=height_path,
        target_kind="building_height",
        building_mask_path=building_path,
        valid_mask_path=valid_path,
    )

    reference = load_benchmark_reference(record, read_rgb_raster(rgb_path))

    assert int(reference.building_mask.sum()) == 2
    assert int(reference.regression_mask.sum()) == 1
    assert int(reference.valid_mask.sum()) == 1
    assert np.isnan(reference.target_m[0, 1])


def test_highbuild_benchmark_rejects_ambiguous_sparse_height_record(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "rgb.tif"
    height_path = tmp_path / "building_height_m.tif"
    _write_raster(rgb_path, np.full((3, 2, 2), 128, dtype=np.uint8))
    _write_raster(height_path, np.ones((2, 2), dtype=np.float32))
    record = SurfaceSampleRecord(
        sample_id="unsafe-highbuild",
        region="city-held-out",
        landscape="urban",
        rgb_path=rgb_path,
        surface_path=height_path,
        target_kind="ndsm",
    )

    with pytest.raises(ValueError, match="not a complete nDSM"):
        load_benchmark_reference(record, read_rgb_raster(rgb_path))


def test_dataset_digest_is_path_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    rgb_a = tmp_path / "a_rgb.tif"
    rgb_b = tmp_path / "b_rgb.tif"
    target_a = tmp_path / "a_target.tif"
    target_b = tmp_path / "b_target.tif"
    rgb = np.full((3, 3, 4), 17, dtype=np.uint8)
    target = np.full((3, 4), 2.0, dtype=np.float32)
    _write_raster(rgb_a, rgb)
    _write_raster(rgb_b, rgb)
    _write_raster(target_a, target)
    _write_raster(target_b, target)

    def record(rgb_path: Path, surface_path: Path) -> SurfaceSampleRecord:
        return SurfaceSampleRecord(
            sample_id="same-id",
            region="same-region",
            landscape="forest",
            rgb_path=rgb_path,
            surface_path=surface_path,
            target_kind="ndsm",
        )

    first, _ = benchmark_dataset_digest([record(rgb_a, target_a)])
    relocated, _ = benchmark_dataset_digest([record(rgb_b, target_b)])
    assert first == relocated

    changed = target.copy()
    changed[0, 0] = 3.0
    _write_raster(target_b, changed)
    modified, _ = benchmark_dataset_digest([record(rgb_b, target_b)])
    assert modified != first


def test_model_tree_digest_includes_relative_names_and_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.json").write_text("same", encoding="utf-8")
    (second / "config.json").write_text("same", encoding="utf-8")
    assert sha256_path_tree(first) == sha256_path_tree(second)

    (second / "config.json").write_text("changed", encoding="utf-8")
    assert sha256_path_tree(first) != sha256_path_tree(second)
