"""Native GAMUS HDF5 samples for semantic surface-height training.

GAMUS stores one aligned RGB, above-ground-level (AGL), and semantic class
array per tile.  Reading the HDF5 files directly avoids creating a second
GeoTIFF/PNG copy of the roughly 80 GB dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
import json

from .radiometry import RADIOMETRIC_POLICIES, apply_radiometric_policy
from .raster_dataset import IMAGENET_MEAN, IMAGENET_STD, _crop_origin, _pad_to_patch
from .surface_dataset import LANDSCAPE_CLASSES

try:  # HDF5 support is deliberately kept in the optional ``train`` extra.
    import h5py
except ImportError:  # pragma: no cover - exercised only in minimal app installs.
    h5py = None


GAMUS_SPLITS = {"train", "val", "test"}
GAMUS_OFFICIAL_SPLIT_COUNTS = {"train": 5004, "val": 859, "test": 2861}
GAMUS_CLASSES = {
    0: "other/background",
    1: "ground",
    2: "low vegetation",
    3: "building",
    4: "water",
    5: "road",
    6: "tree",
}

# The protected height model intentionally keeps its three broad, mutually
# exclusive routing groups.  GAMUS identification is richer, so a second head
# uses this stable six-class order.  Source class 0 and malformed values are
# ignored rather than being rewritten as ground.
GAMUS_SIX_CLASS_NAMES = (
    "ground",
    "buildings",
    "water",
    "roads",
    "low_vegetation",
    "trees",
)
GAMUS_SIX_CLASS_SOURCE_IDS = (1, 3, 4, 5, 2, 6)
GAMUS_SOURCE_ID_TO_SIX_CLASS = {
    source_id: class_index
    for class_index, source_id in enumerate(GAMUS_SIX_CLASS_SOURCE_IDS)
}
GAMUS_SIX_CLASS_IGNORE_INDEX = 255

# GAMUS contains seven fine classes while Monocular Surface Reconstruction's protected expert
# router has three mutually exclusive branches.  Water and road belong to the
# ground branch for semantic routing.  Class zero is an ignore label.
GAMUS_GROUND_CLASS_IDS = frozenset({1, 4, 5})
GAMUS_BUILDING_CLASS_IDS = frozenset({3})
GAMUS_VEGETATION_CLASS_IDS = frozenset({2, 6})

# Road and water AGL values contain substantial label artefacts in the audited
# release.  They remain useful semantic supervision, but are excluded from the
# metric-height loss until a broader quality audit says otherwise.
GAMUS_REGRESSION_CLASS_IDS = frozenset({1, 2, 3, 6})
GAMUS_APPROVED_INDEX_SCHEMA = "msr.gamus.approved_samples.v1"


@dataclass(frozen=True)
class GamusSampleRecord:
    """Paths for one native, spatially aligned GAMUS tile triplet."""

    sample_id: str
    region: str
    image_path: Path
    height_path: Path
    class_path: Path
    relative_prior_path: Path | None = None


def _require_h5py() -> Any:
    if h5py is None:
        raise RuntimeError(
            "Native GAMUS loading requires h5py; install Monocular Surface Reconstruction with "
            "the 'train' optional dependencies"
        )
    return h5py


def load_gamus_split(
    root: str | Path,
    split: str,
    *,
    check_files: bool = True,
    relative_prior_root: str | Path | None = None,
    require_relative_prior: bool = False,
    approved_index_path: str | Path | None = None,
) -> list[GamusSampleRecord]:
    """Discover native GAMUS triplets without generating a duplicate manifest."""

    normalized_split = split.strip().lower()
    if normalized_split not in GAMUS_SPLITS:
        raise ValueError(
            f"GAMUS split must be one of {sorted(GAMUS_SPLITS)}, received {split!r}"
        )
    root_path = Path(root).expanduser().resolve()
    prior_root_path = (
        Path(relative_prior_root).expanduser().resolve()
        if relative_prior_root is not None
        else None
    )
    if require_relative_prior and prior_root_path is None:
        raise ValueError("require_relative_prior needs relative_prior_root")
    image_dir = root_path / "images" / normalized_split
    records: list[GamusSampleRecord] = []
    seen_sample_ids: set[str] = set()
    # Most GAMUS cities use ``*_RGB.h5``.  The official New York release uses
    # ``*_IMG.h5`` for the same input role, so both naming conventions must be
    # included or 2,167 valid NYC tiles silently disappear from train/test.
    image_paths = sorted(
        [*image_dir.glob("*_RGB.h5"), *image_dir.glob("*_IMG.h5")]
    )
    for image_path in image_paths:
        suffix = "_RGB.h5" if image_path.name.endswith("_RGB.h5") else "_IMG.h5"
        sample_id = image_path.name.removesuffix(suffix)
        if sample_id in seen_sample_ids:
            raise ValueError(
                f"Duplicate GAMUS imagery for {sample_id} in {image_dir}"
            )
        seen_sample_ids.add(sample_id)
        height_path = root_path / "heights" / normalized_split / f"{sample_id}_AGL.h5"
        class_path = root_path / "classes" / normalized_split / f"{sample_id}_CLS.h5"
        relative_prior_path = (
            prior_root_path / normalized_split / f"{sample_id}_REL.h5"
            if prior_root_path is not None
            else None
        )
        if check_files:
            missing = [path for path in (height_path, class_path) if not path.is_file()]
            if (
                require_relative_prior
                and relative_prior_path is not None
                and not relative_prior_path.is_file()
            ):
                missing.append(relative_prior_path)
            if missing:
                raise FileNotFoundError(
                    f"Incomplete GAMUS triplet {sample_id}: "
                    + ", ".join(str(path) for path in missing)
                )
        city = sample_id.split("_", maxsplit=1)[0]
        records.append(
            GamusSampleRecord(
                sample_id=sample_id,
                region=f"GAMUS_{city}",
                image_path=image_path,
                height_path=height_path,
                class_path=class_path,
                relative_prior_path=(
                    relative_prior_path
                    if relative_prior_path is not None and relative_prior_path.is_file()
                    else None
                ),
            )
        )
    if approved_index_path is not None:
        index_path = Path(approved_index_path).expanduser().resolve()
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("schema") != GAMUS_APPROVED_INDEX_SCHEMA:
            raise ValueError(
                f"Unsupported GAMUS approved index schema in {index_path}: "
                f"{payload.get('schema')!r}"
            )
        split_payload = payload.get("splits", {}).get(normalized_split)
        if not isinstance(split_payload, dict):
            raise ValueError(
                f"GAMUS approved index has no {normalized_split!r} split: {index_path}"
            )
        approved_values = split_payload.get("approved_sample_ids")
        if not isinstance(approved_values, list) or not all(
            isinstance(value, str) and value for value in approved_values
        ):
            raise ValueError(
                f"Malformed approved_sample_ids for {normalized_split}: {index_path}"
            )
        approved_ids = set(approved_values)
        if len(approved_ids) != len(approved_values):
            raise ValueError(
                f"Duplicate approved sample IDs for {normalized_split}: {index_path}"
            )
        discovered_ids = {record.sample_id for record in records}
        missing_from_source = approved_ids - discovered_ids
        if missing_from_source:
            raise FileNotFoundError(
                f"Approved GAMUS index references missing {normalized_split} tiles: "
                f"{sorted(missing_from_source)[:5]}"
            )
        records = [record for record in records if record.sample_id in approved_ids]
        declared_count = split_payload.get("approved_count")
        if declared_count is not None and int(declared_count) != len(records):
            raise ValueError(
                f"Approved GAMUS index count mismatch for {normalized_split}: "
                f"declared {declared_count}, loaded {len(records)}"
            )
    if not records:
        raise ValueError(f"No GAMUS RGB HDF5 tiles found in {image_dir}")
    return records


def _image_dataset(handle: Any, path: Path) -> Any:
    if "image" not in handle:
        raise ValueError(f"GAMUS HDF5 file has no root 'image' dataset: {path}")
    return handle["image"]


def _declared_nodata(dataset: Any, handle: Any) -> tuple[float, ...]:
    """Collect scalar no-data attributes used by common HDF5 exporters."""

    values: list[float] = []
    for attributes in (dataset.attrs, handle.attrs):
        for name in ("_FillValue", "fill_value", "nodata", "NoDataValue"):
            if name not in attributes:
                continue
            raw = np.asarray(attributes[name]).reshape(-1)
            for item in raw:
                try:
                    values.append(float(item))
                except (TypeError, ValueError):
                    continue
    return tuple(values)


class GamusSurfaceDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Read GAMUS HDF5 triplets into the existing Monocular Surface Reconstruction batch contract."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        patch_size: int = 384,
        random_crop: bool = False,
        augment: bool = False,
        samples_per_epoch: int | None = None,
        rgb_scale: float = 255.0,
        height_max_m: float = 200.0,
        dark_pixel_threshold: float = 0.15,
        radiometric_policy: str = "raw",
        check_files: bool = True,
        relative_prior_root: str | Path | None = None,
        require_relative_prior: bool = False,
        approved_index_path: str | Path | None = None,
    ) -> None:
        _require_h5py()
        if patch_size <= 0 or rgb_scale <= 0 or height_max_m <= 0:
            raise ValueError("patch_size, rgb_scale, and height_max_m must be positive")
        if not 0.0 <= dark_pixel_threshold <= 1.0:
            raise ValueError("dark_pixel_threshold must be within [0, 1]")
        if samples_per_epoch is not None and samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive when supplied")
        if radiometric_policy not in RADIOMETRIC_POLICIES:
            raise ValueError(
                f"radiometric_policy must be one of {sorted(RADIOMETRIC_POLICIES)}"
            )
        self.records = load_gamus_split(
            root,
            split,
            check_files=check_files,
            relative_prior_root=relative_prior_root,
            require_relative_prior=require_relative_prior,
            approved_index_path=approved_index_path,
        )
        self.split = split.strip().lower()
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch
        self.rgb_scale = rgb_scale
        self.height_max_m = height_max_m
        self.dark_pixel_threshold = float(dark_pixel_threshold)
        self.radiometric_policy = radiometric_policy
        self.approved_index_path = (
            Path(approved_index_path).expanduser().resolve()
            if approved_index_path is not None
            else None
        )

    def __len__(self) -> int:
        return self.samples_per_epoch or len(self.records)

    def _read_crop(self, record: GamusSampleRecord) -> dict[str, np.ndarray]:
        h5 = _require_h5py()
        with (
            h5.File(record.image_path, "r") as image_handle,
            h5.File(record.height_path, "r") as height_handle,
            h5.File(record.class_path, "r") as class_handle,
        ):
            image_dataset = _image_dataset(image_handle, record.image_path)
            height_dataset = _image_dataset(height_handle, record.height_path)
            class_dataset = _image_dataset(class_handle, record.class_path)
            if image_dataset.ndim != 3 or image_dataset.shape[-1] < 3:
                raise ValueError(
                    f"GAMUS RGB must be HWC with at least three channels: "
                    f"{record.image_path} has {image_dataset.shape}"
                )
            if height_dataset.ndim != 2 or class_dataset.ndim != 2:
                raise ValueError(
                    f"GAMUS AGL and CLS must be two-dimensional for {record.sample_id}"
                )
            spatial_shape = tuple(image_dataset.shape[:2])
            if tuple(height_dataset.shape) != spatial_shape or tuple(class_dataset.shape) != spatial_shape:
                raise ValueError(
                    f"Unaligned GAMUS arrays for {record.sample_id}: RGB {spatial_shape}, "
                    f"AGL {height_dataset.shape}, CLS {class_dataset.shape}"
                )

            row = _crop_origin(spatial_shape[0], self.patch_size, self.random_crop)
            col = _crop_origin(spatial_shape[1], self.patch_size, self.random_crop)
            read_height = min(self.patch_size, spatial_shape[0] - row)
            read_width = min(self.patch_size, spatial_shape[1] - col)
            rows = slice(row, row + read_height)
            columns = slice(col, col + read_width)
            image = np.asarray(
                image_dataset[rows, columns, :3], dtype=np.float32
            ).transpose(2, 0, 1)
            height = np.asarray(
                height_dataset[rows, columns], dtype=np.float32
            )
            class_values = np.asarray(
                class_dataset[rows, columns], dtype=np.float32
            )
            nodata_values = _declared_nodata(height_dataset, height_handle)
            image_nodata_values = _declared_nodata(image_dataset, image_handle)

        # Image coverage, class availability, and height availability are
        # independent facts.  In particular, a missing class or AGL value must
        # never turn into a supervised zero-height pixel.
        image_valid = np.all(np.isfinite(image), axis=0)
        for nodata in image_nodata_values:
            if np.isnan(nodata):
                image_valid &= ~np.all(np.isnan(image), axis=0)
            else:
                image_valid &= ~np.all(
                    np.isclose(image, nodata, equal_nan=True), axis=0
                )

        valid_height = np.isfinite(height)
        for nodata in nodata_values:
            if np.isnan(nodata):
                valid_height &= ~np.isnan(height)
            else:
                valid_height &= ~np.isclose(height, nodata, equal_nan=True)
        # The released AGL tiles use negative values (commonly -5) for invalid
        # pixels.  Mask every negative value rather than relying on one sentinel.
        valid_height &= height >= 0.0
        valid_height &= height <= self.height_max_m

        rounded_classes = np.rint(class_values)
        valid_class = (
            np.isfinite(class_values)
            & np.isclose(class_values, rounded_classes, rtol=0.0, atol=1.0e-5)
            & (rounded_classes >= min(GAMUS_CLASSES))
            & (rounded_classes <= max(GAMUS_CLASSES))
        )
        classes = np.where(valid_class, rounded_classes, 0).astype(np.int64)
        classification_valid = valid_class & (classes != 0)
        building = classification_valid & np.isin(
            classes, tuple(GAMUS_BUILDING_CLASS_IDS)
        )
        vegetation = classification_valid & np.isin(
            classes, tuple(GAMUS_VEGETATION_CLASS_IDS)
        )
        ground = classification_valid & np.isin(
            classes, tuple(GAMUS_GROUND_CLASS_IDS)
        )

        fine_class_target = np.full(
            classes.shape, GAMUS_SIX_CLASS_IGNORE_INDEX, dtype=np.int64
        )
        for source_id, class_index in GAMUS_SOURCE_ID_TO_SIX_CLASS.items():
            fine_class_target[classification_valid & (classes == source_id)] = (
                class_index
            )

        domain = np.full(classes.shape, LANDSCAPE_CLASSES["ground"], dtype=np.int64)
        domain[vegetation] = LANDSCAPE_CLASSES["vegetation"]
        domain[building] = LANDSCAPE_CLASSES["building"]
        domain_valid = image_valid & (ground | building | vegetation)
        regression_class = classification_valid & np.isin(
            classes, tuple(GAMUS_REGRESSION_CLASS_IDS)
        )
        regression_valid = image_valid & valid_height & regression_class

        # GAMUS has no shadow class.  This luminance mask is retained only for
        # an explicitly labelled water-vs-dark-pixel *proxy* diagnostic; it is
        # never a training target and must not be presented as shadow truth.
        scaled_rgb = np.clip(image / self.rgb_scale, 0.0, 1.0)
        dark_pixel_proxy = image_valid & (
            np.mean(scaled_rgb, axis=0) <= self.dark_pixel_threshold
        )

        relative_prior = np.zeros_like(height, dtype=np.float32)
        if record.relative_prior_path is not None:
            with h5.File(record.relative_prior_path, "r") as prior_handle:
                prior_dataset = _image_dataset(
                    prior_handle, record.relative_prior_path
                )
                if prior_dataset.ndim != 2 or tuple(prior_dataset.shape) != spatial_shape:
                    raise ValueError(
                        f"Unaligned GAMUS relative prior for {record.sample_id}: "
                        f"expected {spatial_shape}, found {prior_dataset.shape}"
                    )
                relative_prior = np.asarray(
                    prior_dataset[rows, columns], dtype=np.float32
                )
                relative_prior = np.nan_to_num(
                    relative_prior, nan=0.0, posinf=1.0, neginf=0.0
                ).clip(0.0, 1.0)

        return {
            "image": image,
            "height": np.where(regression_valid, height, 0.0).astype(np.float32),
            "image_valid_mask": image_valid,
            "classification_valid_mask": classification_valid,
            "height_valid_mask": valid_height,
            "fine_class_target": fine_class_target,
            "dark_pixel_proxy_mask": dark_pixel_proxy,
            # ``valid_mask`` is a compatibility alias used by existing code.
            # New code should select one of the three explicit validity masks.
            "valid_mask": valid_height,
            "regression_mask": regression_valid,
            "building_mask": building,
            "vegetation_mask": vegetation,
            "domain_target": domain,
            "domain_valid_mask": domain_valid,
            "relative_prior": relative_prior,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index % len(self.records)]
        arrays = self._read_crop(record)
        fill_values = {
            "image": 0.0,
            "height": 0.0,
            "image_valid_mask": False,
            "classification_valid_mask": False,
            "height_valid_mask": False,
            "fine_class_target": GAMUS_SIX_CLASS_IGNORE_INDEX,
            "dark_pixel_proxy_mask": False,
            "valid_mask": False,
            "regression_mask": False,
            "building_mask": False,
            "vegetation_mask": False,
            "domain_target": LANDSCAPE_CLASSES["ground"],
            "domain_valid_mask": False,
            "relative_prior": 0.0,
        }
        for name in arrays:
            arrays[name] = _pad_to_patch(
                arrays[name], self.patch_size, fill=fill_values[name]
            )

        if self.augment:
            turns = int(torch.randint(0, 4, size=(1,)).item())
            flip = bool(torch.randint(0, 2, size=(1,)).item())
            for name, value in arrays.items():
                if turns:
                    value = np.rot90(value, turns, axes=(-2, -1)).copy()
                if flip:
                    value = np.flip(value, axis=-1).copy()
                arrays[name] = value

        arrays["image"] = apply_radiometric_policy(
            arrays["image"],
            valid_mask=arrays["image_valid_mask"],
            policy=self.radiometric_policy,
            rgb_scale=self.rgb_scale,
        )
        # Every radiometric branch must present the network with the same
        # neutral value at invalid/padded pixels.  The raw branch otherwise
        # preserved finite HDF5 no-data sentinels (for example ``-999``), which
        # could contaminate neighbouring valid predictions through convolution
        # even though all three supervision masks correctly excluded the pixel.
        arrays["image"][:, ~arrays["image_valid_mask"]] = 0.0
        arrays["image"] = (
            arrays["image"] / self.rgb_scale - IMAGENET_MEAN
        ) / IMAGENET_STD

        return {
            "image": torch.from_numpy(np.ascontiguousarray(arrays["image"])).float(),
            "height": torch.from_numpy(
                np.ascontiguousarray(arrays["height"][None])
            ).float(),
            "image_valid_mask": torch.from_numpy(
                np.ascontiguousarray(arrays["image_valid_mask"][None])
            ).bool(),
            "classification_valid_mask": torch.from_numpy(
                np.ascontiguousarray(arrays["classification_valid_mask"])
            ).bool(),
            "height_valid_mask": torch.from_numpy(
                np.ascontiguousarray(arrays["height_valid_mask"][None])
            ).bool(),
            "fine_class_target": torch.from_numpy(
                np.ascontiguousarray(arrays["fine_class_target"])
            ).long(),
            "dark_pixel_proxy_mask": torch.from_numpy(
                np.ascontiguousarray(arrays["dark_pixel_proxy_mask"])
            ).bool(),
            "valid_mask": torch.from_numpy(
                np.ascontiguousarray(arrays["valid_mask"][None])
            ).bool(),
            "regression_mask": torch.from_numpy(
                np.ascontiguousarray(arrays["regression_mask"][None])
            ).bool(),
            "building_mask": torch.from_numpy(
                np.ascontiguousarray(arrays["building_mask"][None])
            ).float(),
            "vegetation_mask": torch.from_numpy(
                np.ascontiguousarray(arrays["vegetation_mask"][None])
            ).float(),
            "domain_target": torch.from_numpy(
                np.ascontiguousarray(arrays["domain_target"])
            ).long(),
            "domain_valid_mask": torch.from_numpy(
                np.ascontiguousarray(arrays["domain_valid_mask"])
            ).bool(),
            "relative_prior": torch.from_numpy(
                np.ascontiguousarray(arrays["relative_prior"][None])
            ).float(),
            "sample_id": record.sample_id,
            "region": record.region,
            "landscape": "mixed",
        }
