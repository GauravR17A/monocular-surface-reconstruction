"""PyTorch dataset for aligned RGB and metric-height GeoTIFF pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from .manifest import SampleRecord


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class RasterSample:
    image: torch.Tensor
    height: torch.Tensor
    valid_mask: torch.Tensor
    building_mask: torch.Tensor
    sample_id: str
    region: str


def _crop_origin(length: int, patch_size: int, random_crop: bool) -> int:
    if length <= patch_size:
        return 0
    if random_crop:
        return int(torch.randint(0, length - patch_size + 1, size=(1,)).item())
    return (length - patch_size) // 2


def _pad_to_patch(array: np.ndarray, patch_size: int, *, fill: float) -> np.ndarray:
    height, width = array.shape[-2:]
    pad_height = max(patch_size - height, 0)
    pad_width = max(patch_size - width, 0)
    if pad_height == 0 and pad_width == 0:
        return array
    padding = [(0, 0)] * (array.ndim - 2) + [(0, pad_height), (0, pad_width)]
    return np.pad(array, padding, mode="constant", constant_values=fill)


class RasterHeightDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Read aligned raster windows without resampling height targets."""

    def __init__(
        self,
        records: list[SampleRecord],
        *,
        patch_size: int = 384,
        random_crop: bool = False,
        augment: bool = False,
        samples_per_epoch: int | None = None,
        rgb_scale: float = 255.0,
        height_min_m: float = 0.0,
        height_max_m: float = 185.0,
        building_threshold_m: float = 2.0,
    ) -> None:
        if not records:
            raise ValueError("At least one sample record is required")
        if patch_size <= 0 or rgb_scale <= 0:
            raise ValueError("patch_size and rgb_scale must be positive")
        self.records = records
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch
        self.rgb_scale = rgb_scale
        self.height_min_m = height_min_m
        self.height_max_m = height_max_m
        self.building_threshold_m = building_threshold_m

    def __len__(self) -> int:
        return self.samples_per_epoch or len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index % len(self.records)]
        with rasterio.open(record.rgb_path) as rgb_source, rasterio.open(
            record.height_path
        ) as height_source:
            if (rgb_source.height, rgb_source.width) != (
                height_source.height,
                height_source.width,
            ):
                raise ValueError(f"Unaligned raster dimensions for {record.sample_id}")
            if not np.allclose(tuple(rgb_source.transform), tuple(height_source.transform)):
                raise ValueError(f"Unaligned raster transforms for {record.sample_id}")
            row = _crop_origin(rgb_source.height, self.patch_size, self.random_crop)
            col = _crop_origin(rgb_source.width, self.patch_size, self.random_crop)
            read_height = min(self.patch_size, rgb_source.height - row)
            read_width = min(self.patch_size, rgb_source.width - col)
            window = Window(col, row, read_width, read_height)

            if rgb_source.count < 3:
                raise ValueError(f"RGB raster has fewer than 3 bands: {record.rgb_path}")
            image = rgb_source.read((1, 2, 3), window=window).astype(np.float32)
            height = height_source.read(1, window=window).astype(np.float32)
            nodata = height_source.nodata

        valid = np.isfinite(height)
        if nodata is not None:
            valid &= ~np.isclose(height, nodata, equal_nan=True)
        valid &= height >= self.height_min_m
        valid &= height <= self.height_max_m
        if record.mask_path is not None:
            with rasterio.open(record.mask_path) as mask_source:
                valid &= mask_source.read(1, window=window) > 0

        image = _pad_to_patch(image, self.patch_size, fill=0.0)
        height = _pad_to_patch(height, self.patch_size, fill=0.0)
        valid = _pad_to_patch(valid, self.patch_size, fill=False)
        height = np.where(valid, height, 0.0).astype(np.float32, copy=False)

        image = image / self.rgb_scale
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        building = valid & (height >= self.building_threshold_m)

        if self.augment:
            turns = int(torch.randint(0, 4, size=(1,)).item())
            if turns:
                image = np.rot90(image, turns, axes=(-2, -1)).copy()
                height = np.rot90(height, turns, axes=(-2, -1)).copy()
                valid = np.rot90(valid, turns, axes=(-2, -1)).copy()
                building = np.rot90(building, turns, axes=(-2, -1)).copy()
            if bool(torch.randint(0, 2, size=(1,)).item()):
                image = np.flip(image, axis=-1).copy()
                height = np.flip(height, axis=-1).copy()
                valid = np.flip(valid, axis=-1).copy()
                building = np.flip(building, axis=-1).copy()

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "height": torch.from_numpy(np.ascontiguousarray(height[None, ...])),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid[None, ...])),
            "building_mask": torch.from_numpy(np.ascontiguousarray(building[None, ...])).float(),
            "sample_id": record.sample_id,
            "region": record.region,
        }
