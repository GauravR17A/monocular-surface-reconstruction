"""Streaming transforms for locally repacked HighBuild WebDataset shards."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
import torch
import webdataset as wds
from PIL import Image
from rasterio.io import MemoryFile

from .raster_dataset import IMAGENET_MEAN, IMAGENET_STD, _crop_origin, _pad_to_patch


class HeightSampleTransform:
    def __init__(
        self,
        *,
        patch_size: int,
        training: bool,
        height_min_m: float,
        height_max_m: float,
        building_threshold_m: float = 2.0,
        foreground_crop_probability: float = 0.0,
        tall_crop_probability: float = 0.0,
        tall_threshold_m: float = 20.0,
    ) -> None:
        if not 0.0 <= tall_crop_probability <= foreground_crop_probability <= 1.0:
            raise ValueError(
                "crop probabilities must satisfy 0 <= tall <= foreground <= 1"
            )
        self.patch_size = patch_size
        self.training = training
        self.height_min_m = height_min_m
        self.height_max_m = height_max_m
        self.building_threshold_m = building_threshold_m
        self.foreground_crop_probability = foreground_crop_probability
        self.tall_crop_probability = tall_crop_probability
        self.tall_threshold_m = tall_threshold_m

    @staticmethod
    def _first(sample: dict, suffixes: Sequence[str]) -> bytes:
        for suffix in suffixes:
            value = sample.get(suffix)
            if value is not None:
                return value
        raise KeyError(f"Sample {sample.get('__key__')} lacks one of {tuple(suffixes)}")

    def __call__(self, sample: dict) -> dict[str, torch.Tensor | str]:
        image_bytes = self._first(sample, ("jpg", "jpeg", "png"))
        height_bytes = self._first(sample, ("tiff", "tif"))
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)
        # HighBuild height TIFFs use ZSTD compression. Pillow's bundled
        # libtiff on Windows cannot decode it, while Rasterio/GDAL can and
        # also preserves the geospatial nodata value.
        with MemoryFile(height_bytes) as memory_file, memory_file.open() as source:
            height = source.read(1).astype(np.float32, copy=False)
            nodata = source.nodata
        height = np.squeeze(height)
        if height.ndim != 2 or image.shape[-2:] != height.shape:
            raise ValueError(
                f"Unaligned WebDataset sample {sample.get('__key__')}: "
                f"RGB={image.shape[-2:]}, height={height.shape}"
            )

        full_valid = (
            np.isfinite(height)
            & (height >= self.height_min_m)
            & (height <= self.height_max_m)
        )
        if nodata is not None:
            full_valid &= ~np.isclose(height, nodata, equal_nan=True)
        row, col = self._crop_origin(height, full_valid)
        image = image[:, row : row + self.patch_size, col : col + self.patch_size]
        height = height[row : row + self.patch_size, col : col + self.patch_size]
        valid = full_valid[row : row + self.patch_size, col : col + self.patch_size]
        image = _pad_to_patch(image, self.patch_size, fill=0.0)
        height = _pad_to_patch(height, self.patch_size, fill=0.0)
        valid = _pad_to_patch(valid, self.patch_size, fill=False)
        height = np.where(valid, height, 0.0).astype(np.float32, copy=False)
        image = (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        building = valid & (height >= self.building_threshold_m)

        if self.training:
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

        sample_id = str(sample["__key__"])
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "height": torch.from_numpy(np.ascontiguousarray(height[None, ...])),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid[None, ...])),
            "building_mask": torch.from_numpy(
                np.ascontiguousarray(building[None, ...])
            ).float(),
            "sample_id": sample_id,
            "region": sample_id.split("__", maxsplit=1)[0],
        }

    def _crop_origin(
        self, height: np.ndarray, valid: np.ndarray
    ) -> tuple[int, int]:
        if not self.training:
            return (
                _crop_origin(height.shape[0], self.patch_size, False),
                _crop_origin(height.shape[1], self.patch_size, False),
            )
        choice = float(torch.rand(1).item())
        focus = None
        if choice < self.tall_crop_probability:
            tall = valid & (height >= self.tall_threshold_m)
            if np.any(tall):
                focus = tall
        if focus is None and choice < self.foreground_crop_probability:
            foreground = valid & (height >= self.building_threshold_m)
            if np.any(foreground):
                focus = foreground
        if focus is None:
            return (
                _crop_origin(height.shape[0], self.patch_size, True),
                _crop_origin(height.shape[1], self.patch_size, True),
            )

        coordinates = np.argwhere(focus)
        selected = coordinates[int(torch.randint(0, len(coordinates), (1,)).item())]
        origins = []
        for coordinate, length in zip(selected, height.shape):
            maximum_origin = max(length - self.patch_size, 0)
            lower = max(int(coordinate) - self.patch_size + 1, 0)
            upper = min(int(coordinate), maximum_origin)
            origin = (
                lower
                if upper <= lower
                else int(torch.randint(lower, upper + 1, (1,)).item())
            )
            origins.append(origin)
        return origins[0], origins[1]


def make_webdataset(
    shards: list[str],
    *,
    patch_size: int,
    training: bool,
    height_min_m: float,
    height_max_m: float,
    seed: int,
    foreground_crop_probability: float = 0.0,
    tall_crop_probability: float = 0.0,
    tall_threshold_m: float = 20.0,
) -> wds.WebDataset:
    if not shards:
        raise ValueError("At least one WebDataset shard is required")
    # ``file:///C:/...`` is parsed by WebDataset 1.0 as ``/C:/...`` on
    # Windows. The equivalent ``file:C:/...`` keeps a directly openable path.
    normalized_shards = [
        shard if "://" in shard or shard.startswith("file:") else f"file:{Path(shard).resolve().as_posix()}"
        for shard in shards
    ]
    dataset = wds.WebDataset(
        normalized_shards,
        resampled=training,
        # ResampledShards already chooses source shards randomly. WebDataset
        # ignores shardshuffle in that mode and emits a warning if it is set.
        shardshuffle=False,
        seed=seed,
    )
    if training:
        dataset = dataset.shuffle(1000, seed=seed)
    return dataset.map(
        HeightSampleTransform(
            patch_size=patch_size,
            training=training,
            height_min_m=height_min_m,
            height_max_m=height_max_m,
            foreground_crop_probability=foreground_crop_probability,
            tall_crop_probability=tall_crop_probability,
            tall_threshold_m=tall_threshold_m,
        )
    )
