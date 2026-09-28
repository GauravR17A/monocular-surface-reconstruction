from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from msr.data.surface_dataset import (
    LANDSCAPE_CLASSES,
    MultiDomainSurfaceDataset,
    SurfaceSampleRecord,
    load_surface_manifest,
)


def _write(path: Path, values: np.ndarray, *, count: int = 1) -> None:
    array = values if values.ndim == 3 else values[None]
    profile = {
        "driver": "GTiff",
        "height": array.shape[-2],
        "width": array.shape[-1],
        "count": count,
        "dtype": str(array.dtype),
        "crs": "EPSG:32643",
        "transform": Affine.translation(500000, 3000000) * Affine.scale(1, -1),
        "nodata": None,
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(array)


def test_dsm_minus_dtm_produces_object_and_canopy_height(tmp_path: Path):
    rgb = np.full((3, 4, 4), 128, dtype=np.uint8)
    dtm = np.full((4, 4), 100.0, dtype=np.float32)
    dsm = dtm.copy()
    dsm[:2, :2] += 12.0
    dsm[2:, :2] += 7.0
    building = np.zeros((4, 4), dtype=np.uint8)
    building[:2, :2] = 1
    vegetation = np.zeros((4, 4), dtype=np.uint8)
    vegetation[2:, :2] = 1
    for name, values in (
        ("rgb.tif", rgb),
        ("dsm.tif", dsm),
        ("dtm.tif", dtm),
        ("building.tif", building),
        ("vegetation.tif", vegetation),
    ):
        _write(tmp_path / name, values, count=values.shape[0] if values.ndim == 3 else 1)
    manifest = tmp_path / "surface.csv"
    manifest.write_text(
        "sample_id,region,landscape,rgb_path,surface_path,target_kind,dtm_path,building_mask_path,vegetation_mask_path\n"
        "one,region_a,mixed,rgb.tif,dsm.tif,dsm,dtm.tif,building.tif,vegetation.tif\n",
        encoding="utf-8",
    )

    records = load_surface_manifest(manifest)
    sample = MultiDomainSurfaceDataset(records, patch_size=4)[0]

    assert sample["height"][0, 0, 0].item() == pytest.approx(12.0)
    assert sample["height"][0, 3, 0].item() == pytest.approx(7.0)
    assert sample["domain_target"][0, 0].item() == LANDSCAPE_CLASSES["building"]
    assert sample["domain_target"][3, 0].item() == LANDSCAPE_CLASSES["vegetation"]
    assert sample["domain_target"][3, 3].item() == LANDSCAPE_CLASSES["ground"]


def test_building_only_targets_do_not_label_background_as_ground(tmp_path: Path):
    rgb = np.full((3, 3, 3), 128, dtype=np.uint8)
    height = np.zeros((3, 3), dtype=np.float32)
    height[1, 1] = 10.0
    _write(tmp_path / "rgb.tif", rgb, count=3)
    _write(tmp_path / "height.tif", height)
    manifest = tmp_path / "surface.csv"
    manifest.write_text(
        "sample_id,region,landscape,rgb_path,surface_path,target_kind\n"
        "one,region_a,urban,rgb.tif,height.tif,building_height\n",
        encoding="utf-8",
    )

    sample = MultiDomainSurfaceDataset(load_surface_manifest(manifest), patch_size=3)[0]

    assert sample["regression_mask"].sum().item() == 1
    assert sample["domain_valid_mask"].sum().item() == 1


def test_supervised_crop_keeps_sparse_measured_building_height(tmp_path: Path) -> None:
    size = 12
    rgb = np.full((3, size, size), 128, dtype=np.uint8)
    height = np.zeros((size, size), dtype=np.float32)
    buildings = np.zeros((size, size), dtype=np.uint8)
    measured = np.zeros((size, size), dtype=np.uint8)
    height[-1, -1] = 18.0
    buildings[-1, -1] = 1
    measured[-1, -1] = 1
    _write(tmp_path / "rgb.tif", rgb, count=3)
    _write(tmp_path / "building_height_m.tif", height)
    _write(tmp_path / "building_footprints.tif", buildings)
    _write(tmp_path / "measured_height_valid.tif", measured)
    manifest = tmp_path / "surface.csv"
    manifest.write_text(
        "sample_id,region,landscape,rgb_path,surface_path,target_kind,"
        "building_mask_path,valid_mask_path\n"
        "one,region_a,urban,rgb.tif,building_height_m.tif,building_height,"
        "building_footprints.tif,measured_height_valid.tif\n",
        encoding="utf-8",
    )

    sample = MultiDomainSurfaceDataset(
        load_surface_manifest(manifest),
        patch_size=4,
        random_crop=True,
        supervised_crop_probability=1.0,
    )[0]

    assert sample["regression_mask"].sum().item() == 1
    assert sample["height"].max().item() == pytest.approx(18.0)


def test_supervised_crop_probability_is_bounded(tmp_path: Path) -> None:
    record = SurfaceSampleRecord(
        sample_id="one",
        region="region_a",
        landscape="plains",
        rgb_path=tmp_path / "rgb.tif",
        surface_path=tmp_path / "ordinary_ndsm.tif",
        target_kind="ndsm",
    )
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        MultiDomainSurfaceDataset([record], supervised_crop_probability=1.1)


def test_highbuild_sparse_height_cannot_be_declared_ndsm(tmp_path: Path) -> None:
    manifest = tmp_path / "surface.csv"
    manifest.write_text(
        "sample_id,region,landscape,rgb_path,surface_path,target_kind\n"
        "one,region_a,urban,rgb.tif,building_height_m.tif,ndsm\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a complete nDSM"):
        load_surface_manifest(manifest, check_files=False)


def test_highbuild_requires_distinct_semantic_and_regression_masks(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "surface.csv"
    manifest.write_text(
        "sample_id,region,landscape,rgb_path,surface_path,target_kind,"
        "building_mask_path,valid_mask_path\n"
        "one,region_a,urban,rgb.tif,building_height_m.tif,building_height,,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit building_mask_path"):
        load_surface_manifest(manifest, check_files=False)

    manifest.write_text(
        "sample_id,region,landscape,rgb_path,surface_path,target_kind,"
        "building_mask_path,valid_mask_path\n"
        "one,region_a,urban,rgb.tif,building_height_m.tif,building_height,"
        "building_height_m.tif,valid.tif\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires distinct"):
        load_surface_manifest(manifest, check_files=False)


def test_highbuild_semantic_footprints_are_separate_from_height_support(
    tmp_path: Path,
) -> None:
    rgb = np.full((3, 3, 3), 128, dtype=np.uint8)
    height = np.zeros((3, 3), dtype=np.float32)
    height[0, 0] = 8.0
    height[0, 1] = 12.0
    buildings = np.zeros((3, 3), dtype=np.uint8)
    buildings[0, :2] = 1
    measured = np.zeros((3, 3), dtype=np.uint8)
    measured[0, 0] = 1
    _write(tmp_path / "rgb.tif", rgb, count=3)
    _write(tmp_path / "building_height_m.tif", height)
    _write(tmp_path / "building_footprints.tif", buildings)
    _write(tmp_path / "measured_height_valid.tif", measured)
    manifest = tmp_path / "surface.csv"
    manifest.write_text(
        "sample_id,region,landscape,rgb_path,surface_path,target_kind,"
        "building_mask_path,valid_mask_path\n"
        "one,region_a,urban,rgb.tif,building_height_m.tif,building_height,"
        "building_footprints.tif,measured_height_valid.tif\n",
        encoding="utf-8",
    )

    sample = MultiDomainSurfaceDataset(load_surface_manifest(manifest), patch_size=3)[0]

    assert sample["building_mask"].sum().item() == 2
    assert sample["domain_valid_mask"].sum().item() == 2
    assert sample["regression_mask"].sum().item() == 1
    assert sample["height"][0, 0, 0].item() == pytest.approx(8.0)
    assert sample["height"][0, 0, 1].item() == 0.0


def test_non_highbuild_ndsm_remains_backward_compatible(tmp_path: Path) -> None:
    manifest = tmp_path / "surface.csv"
    manifest.write_text(
        "sample_id,region,landscape,rgb_path,surface_path,target_kind\n"
        "one,region_a,plains,rgb.tif,ordinary_ndsm.tif,ndsm\n",
        encoding="utf-8",
    )

    records = load_surface_manifest(manifest, check_files=False)

    assert records[0].target_kind == "ndsm"


def test_direct_highbuild_record_construction_is_also_fail_closed(
    tmp_path: Path,
) -> None:
    record = SurfaceSampleRecord(
        sample_id="one",
        region="region_a",
        landscape="urban",
        rgb_path=tmp_path / "rgb.tif",
        surface_path=tmp_path / "building_height_m.tif",
        target_kind="ndsm",
    )

    with pytest.raises(ValueError, match="not a complete nDSM"):
        MultiDomainSurfaceDataset([record], patch_size=3)


def test_stored_prior_policy_is_independent_of_reference_validity(tmp_path: Path):
    rgb = np.full((3, 3, 3), 128, dtype=np.uint8)
    prior = np.array(
        [[-0.2, 0.1, 0.2], [0.3, np.nan, 0.7], [0.8, 0.9, 1.2]],
        dtype=np.float32,
    )
    outputs = []
    for name, invalid_position in (("left", (0, 0)), ("right", (2, 2))):
        height = np.ones((3, 3), dtype=np.float32)
        height[invalid_position] = np.nan
        _write(tmp_path / f"{name}-rgb.tif", rgb, count=3)
        _write(tmp_path / f"{name}-height.tif", height)
        _write(tmp_path / f"{name}-prior.tif", prior)
        manifest = tmp_path / f"{name}.csv"
        manifest.write_text(
            "sample_id,region,landscape,rgb_path,surface_path,target_kind,relative_prior_path\n"
            f"{name},region_a,plains,{name}-rgb.tif,{name}-height.tif,ndsm,{name}-prior.tif\n",
            encoding="utf-8",
        )
        sample = MultiDomainSurfaceDataset(
            load_surface_manifest(manifest),
            patch_size=3,
            relative_prior_policy="stored_01",
        )[0]
        outputs.append(sample["relative_prior"][0].numpy())

    expected = np.nan_to_num(
        prior, nan=0.0, posinf=1.0, neginf=0.0
    ).clip(0.0, 1.0)
    np.testing.assert_array_equal(outputs[0], expected)
    np.testing.assert_array_equal(outputs[1], expected)
