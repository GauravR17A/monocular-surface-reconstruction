from pathlib import Path

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")

from msr.data.gamus_dataset import (  # noqa: E402
    GAMUS_SIX_CLASS_IGNORE_INDEX,
    GamusSurfaceDataset,
    load_gamus_split,
)
from msr.data.raster_dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from msr.data.surface_dataset import LANDSCAPE_CLASSES  # noqa: E402


def _write_h5(path: Path, values: np.ndarray, *, nodata: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("image", data=values)
        if nodata is not None:
            dataset.attrs["_FillValue"] = nodata


def _write_triplet(
    root: Path,
    *,
    sample_id: str = "DC_01_02",
    split: str = "train",
    rgb: np.ndarray,
    height: np.ndarray,
    classes: np.ndarray,
    nodata: float | None = None,
    image_suffix: str = "RGB",
) -> None:
    _write_h5(root / "images" / split / f"{sample_id}_{image_suffix}.h5", rgb)
    _write_h5(
        root / "heights" / split / f"{sample_id}_AGL.h5",
        height,
        nodata=nodata,
    )
    _write_h5(root / "classes" / split / f"{sample_id}_CLS.h5", classes)


def test_official_nyc_img_naming_is_discovered(tmp_path: Path) -> None:
    rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
    height = np.full((2, 2), 8.0, dtype=np.float32)
    classes = np.full((2, 2), 3, dtype=np.float32)
    _write_triplet(
        tmp_path,
        sample_id="NYC_22835",
        rgb=rgb,
        height=height,
        classes=classes,
        image_suffix="IMG",
    )

    records = load_gamus_split(tmp_path, "train")

    assert len(records) == 1
    assert records[0].sample_id == "NYC_22835"
    assert records[0].image_path.name == "NYC_22835_IMG.h5"


def test_precomputed_relative_prior_is_loaded_on_the_same_crop(tmp_path: Path) -> None:
    height = np.arange(30, dtype=np.float32).reshape(5, 6)
    classes = np.full((5, 6), 2, dtype=np.float32)
    rgb = np.stack((height, height, height), axis=-1).astype(np.uint8)
    prior = height / float(height.max())
    _write_triplet(tmp_path, rgb=rgb, height=height, classes=classes)
    prior_root = tmp_path / "priors"
    _write_h5(prior_root / "train" / "DC_01_02_REL.h5", prior)

    dataset = GamusSurfaceDataset(
        tmp_path,
        "train",
        patch_size=3,
        relative_prior_root=prior_root,
        require_relative_prior=True,
    )
    sample = dataset[0]

    np.testing.assert_allclose(
        sample["relative_prior"][0].numpy(), prior[1:4, 1:4], atol=1.0e-6
    )


def test_required_relative_prior_rejects_incomplete_set(tmp_path: Path) -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    values = np.ones((2, 2), dtype=np.float32)
    _write_triplet(tmp_path, rgb=rgb, height=values, classes=values)

    with pytest.raises(FileNotFoundError, match="REL.h5"):
        load_gamus_split(
            tmp_path,
            "train",
            relative_prior_root=tmp_path / "priors",
            require_relative_prior=True,
        )


def test_native_triplet_maps_semantics_and_masks_bad_heights(tmp_path: Path) -> None:
    rgb = np.full((3, 4, 3), 128, dtype=np.uint8)
    classes = np.asarray(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 1],
            [2, 3, 6, 5],
        ],
        dtype=np.float32,
    )
    height = np.asarray(
        [
            [1.0, 0.0, 0.5, 12.0],
            [-5.0, 999.0, 18.0, np.nan],
            [0.2, -0.1, 22.0, 3.0],
        ],
        dtype=np.float32,
    )
    _write_triplet(tmp_path, rgb=rgb, height=height, classes=classes, nodata=999.0)

    sample = GamusSurfaceDataset(tmp_path, "train", patch_size=4, height_max_m=100)[0]

    assert sample["image"].shape == (3, 4, 4)
    assert sample["height"].shape == (1, 4, 4)
    assert sample["sample_id"] == "DC_01_02"
    assert sample["region"] == "GAMUS_DC"
    assert sample["landscape"] == "mixed"
    assert sample["fine_class_target"][0].tolist() == [
        GAMUS_SIX_CLASS_IGNORE_INDEX,
        0,
        4,
        1,
    ]
    assert sample["fine_class_target"][1].tolist() == [2, 3, 5, 0]
    assert sample["classification_valid_mask"].sum().item() == 11
    assert sample["image_valid_mask"].sum().item() == 12
    assert sample["height_valid_mask"].sum().item() == 8
    assert torch.equal(sample["valid_mask"], sample["height_valid_mask"])
    assert sample["domain_target"][0].tolist() == [
        LANDSCAPE_CLASSES["ground"],
        LANDSCAPE_CLASSES["ground"],
        LANDSCAPE_CLASSES["vegetation"],
        LANDSCAPE_CLASSES["building"],
    ]
    assert sample["domain_target"][1].tolist() == [
        LANDSCAPE_CLASSES["ground"],
        LANDSCAPE_CLASSES["ground"],
        LANDSCAPE_CLASSES["vegetation"],
        LANDSCAPE_CLASSES["ground"],
    ]
    assert sample["domain_valid_mask"][0].tolist() == [False, True, True, True]
    assert sample["building_mask"].sum().item() == 2
    assert sample["vegetation_mask"].sum().item() == 4

    # All negative, explicit no-data, NaN, over-limit, and padded AGL values
    # must be excluded. Road/water remain semantic labels but not regressors.
    assert sample["valid_mask"].sum().item() == 8
    assert sample["regression_mask"].sum().item() == 6
    assert sample["regression_mask"][0, 0, 0].item() is False  # ignored class zero
    assert sample["regression_mask"][0, 1, 0].item() is False  # water
    assert sample["regression_mask"][0, 2, 3].item() is False  # road
    assert sample["height"][0, 1, 0].item() == 0.0
    assert sample["height"][0, 2, 0].item() == pytest.approx(0.2)
    assert sample["valid_mask"][0, 1, 0].item() is False
    assert sample["valid_mask"][0, 1, 1].item() is False
    assert sample["valid_mask"][0, 1, 3].item() is False
    assert sample["valid_mask"][0, 2, 1].item() is False
    assert not sample["valid_mask"][0, 3].any()  # padded row
    assert torch.count_nonzero(sample["relative_prior"]) == 0


def test_image_classification_and_height_validity_are_independent(
    tmp_path: Path,
) -> None:
    rgb = np.full((2, 3, 3), 80.0, dtype=np.float32)
    rgb[0, 0] = -999.0
    classes = np.asarray([[1, 4, 5], [0, 2, 6]], dtype=np.float32)
    height = np.asarray([[3.0, 4.0, 5.0], [6.0, -5.0, 7.0]], dtype=np.float32)
    _write_h5(tmp_path / "images" / "train" / "DC_01_02_RGB.h5", rgb, nodata=-999)
    _write_h5(tmp_path / "heights" / "train" / "DC_01_02_AGL.h5", height)
    _write_h5(tmp_path / "classes" / "train" / "DC_01_02_CLS.h5", classes)

    sample = GamusSurfaceDataset(
        tmp_path,
        "train",
        patch_size=3,
        radiometric_policy="percentile_stretch",
    )[0]

    assert sample["image_valid_mask"][0, 0, 0].item() is False
    assert sample["height_valid_mask"][0, 0, 0].item() is True
    assert sample["classification_valid_mask"][0, 0].item() is True
    assert sample["domain_valid_mask"][0, 0].item() is False
    assert sample["regression_mask"][0, 0, 0].item() is False
    assert sample["classification_valid_mask"][0, 1].item() is True  # water
    assert sample["classification_valid_mask"][0, 2].item() is True  # road
    assert sample["regression_mask"][0, 0, 1].item() is False
    assert sample["regression_mask"][0, 0, 2].item() is False
    assert sample["height"][0, 0, 1].item() == 0.0
    assert sample["height"][0, 0, 2].item() == 0.0
    assert sample["fine_class_target"][0, 1].item() == 2
    assert sample["fine_class_target"][0, 2].item() == 3
    # Missing AGL on low vegetation does not erase its class label.
    assert sample["height_valid_mask"][0, 1, 1].item() is False
    assert sample["classification_valid_mask"][1, 1].item() is True
    assert sample["fine_class_target"][1, 1].item() == 4


def test_raw_radiometry_neutralizes_invalid_image_pixels(tmp_path: Path) -> None:
    rgb = np.full((2, 2, 3), 80.0, dtype=np.float32)
    rgb[0, 0] = -999.0
    classes = np.ones((2, 2), dtype=np.float32)
    height = np.ones((2, 2), dtype=np.float32)
    _write_h5(tmp_path / "images" / "train" / "DC_01_02_RGB.h5", rgb, nodata=-999)
    _write_h5(tmp_path / "heights" / "train" / "DC_01_02_AGL.h5", height)
    _write_h5(tmp_path / "classes" / "train" / "DC_01_02_CLS.h5", classes)

    sample = GamusSurfaceDataset(
        tmp_path, "train", patch_size=2, radiometric_policy="raw"
    )[0]

    assert sample["image_valid_mask"][0, 0, 0].item() is False
    expected = torch.from_numpy(
        ((np.zeros((3, 1, 1), dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD)
        .reshape(3)
    )
    torch.testing.assert_close(sample["image"][:, 0, 0], expected)


def test_random_crop_is_spatially_aligned(tmp_path: Path) -> None:
    height = np.arange(30, dtype=np.float32).reshape(5, 6)
    classes = np.full((5, 6), 3, dtype=np.float32)
    rgb = np.stack((height, height + 40, height + 80), axis=-1).astype(np.uint8)
    _write_triplet(tmp_path, rgb=rgb, height=height, classes=classes)

    torch.manual_seed(17)
    expected_row = int(torch.randint(0, 5 - 3 + 1, size=(1,)).item())
    expected_col = int(torch.randint(0, 6 - 3 + 1, size=(1,)).item())
    torch.manual_seed(17)
    sample = GamusSurfaceDataset(
        tmp_path, "train", patch_size=3, random_crop=True
    )[0]

    expected_height = torch.from_numpy(
        height[expected_row : expected_row + 3, expected_col : expected_col + 3]
    )
    assert torch.equal(sample["height"][0], expected_height)
    recovered_red = (
        sample["image"][0] * float(IMAGENET_STD[0, 0, 0])
        + float(IMAGENET_MEAN[0, 0, 0])
    ) * 255.0
    assert torch.allclose(recovered_red, expected_height, atol=1.0e-4)
    assert sample["building_mask"].bool().all()


def test_geometry_and_radiometry_preserve_label_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    height = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    classes = np.asarray([[1, 2, 3], [6, 3, 1]], dtype=np.float32)
    rgb = np.stack((height * 10, height * 20, height * 30), axis=-1).astype(np.uint8)
    _write_triplet(tmp_path, rgb=rgb, height=height, classes=classes)

    draws = iter((torch.tensor([1]), torch.tensor([1])))
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: next(draws))
    sample = GamusSurfaceDataset(
        tmp_path,
        "train",
        patch_size=3,
        augment=True,
        radiometric_policy="percentile_stretch",
    )[0]

    original_height = np.pad(height, ((0, 1), (0, 0)), constant_values=0)
    expected_height = np.flip(np.rot90(original_height, 1), axis=-1).copy()
    assert np.array_equal(sample["height"][0].numpy(), expected_height)
    assert sample["building_mask"][0].sum().item() == 2
    assert sample["vegetation_mask"][0].sum().item() == 2
    assert torch.isfinite(sample["image"]).all()


def test_split_discovery_rejects_incomplete_or_malformed_data(tmp_path: Path) -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    image_path = tmp_path / "images" / "val" / "PHL_01_01_RGB.h5"
    _write_h5(image_path, rgb)
    with pytest.raises(FileNotFoundError, match="Incomplete GAMUS triplet"):
        load_gamus_split(tmp_path, "val")

    _write_h5(
        tmp_path / "heights" / "val" / "PHL_01_01_AGL.h5",
        np.zeros((2, 3), dtype=np.float32),
    )
    _write_h5(
        tmp_path / "classes" / "val" / "PHL_01_01_CLS.h5",
        np.zeros((2, 2), dtype=np.float32),
    )
    dataset = GamusSurfaceDataset(tmp_path, "val", patch_size=2)
    with pytest.raises(ValueError, match="Unaligned GAMUS arrays"):
        _ = dataset[0]
