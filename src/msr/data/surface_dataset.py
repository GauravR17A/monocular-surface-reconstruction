"""Aligned multi-domain RGB/DSM/DTM samples for gated surface training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from .highbuild import validate_highbuild_manifest_contract
from .radiometry import RADIOMETRIC_POLICIES, apply_radiometric_policy
from .raster_dataset import IMAGENET_MEAN, IMAGENET_STD, _crop_origin, _pad_to_patch


LANDSCAPE_CLASSES = {"ground": 0, "building": 1, "vegetation": 2}
TARGET_KINDS = {"dsm", "ndsm", "building_height"}
LANDSCAPES = {"urban", "forest", "plains", "hilly", "mixed"}


@dataclass(frozen=True)
class SurfaceSampleRecord:
    sample_id: str
    region: str
    landscape: str
    rgb_path: Path
    surface_path: Path
    target_kind: str
    dtm_path: Path | None = None
    building_mask_path: Path | None = None
    vegetation_mask_path: Path | None = None
    valid_mask_path: Path | None = None
    relative_prior_path: Path | None = None
    gsd_m: float | None = None


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_surface_manifest(
    path: str | Path, *, check_files: bool = True
) -> list[SurfaceSampleRecord]:
    """Load the explicit multi-domain manifest described in the project README."""

    manifest_path = Path(path).resolve()
    required = {
        "sample_id",
        "region",
        "landscape",
        "rgb_path",
        "surface_path",
        "target_kind",
    }
    records: list[SurfaceSampleRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Surface manifest is missing columns: {sorted(missing)}")
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            region = (row.get("region") or "").strip()
            landscape = (row.get("landscape") or "").strip().lower()
            target_kind = (row.get("target_kind") or "").strip().lower()
            if not sample_id or not region:
                raise ValueError(f"Row {row_number} must define sample_id and region")
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id in surface manifest: {sample_id}")
            if landscape not in LANDSCAPES:
                raise ValueError(
                    f"Invalid landscape {landscape!r} at row {row_number}; "
                    f"expected one of {sorted(LANDSCAPES)}"
                )
            if target_kind not in TARGET_KINDS:
                raise ValueError(
                    f"Invalid target_kind {target_kind!r} at row {row_number}; "
                    f"expected one of {sorted(TARGET_KINDS)}"
                )
            seen.add(sample_id)

            def optional_path(column: str) -> Path | None:
                value = (row.get(column) or "").strip()
                return _resolve(value, manifest_path.parent) if value else None

            rgb_path = _resolve(row["rgb_path"], manifest_path.parent)
            surface_path = _resolve(row["surface_path"], manifest_path.parent)
            dtm_path = optional_path("dtm_path")
            building_mask_path = optional_path("building_mask_path")
            vegetation_mask_path = optional_path("vegetation_mask_path")
            valid_mask_path = optional_path("valid_mask_path")
            relative_prior_path = optional_path("relative_prior_path")
            if target_kind == "dsm" and dtm_path is None:
                raise ValueError(
                    f"DSM target {sample_id} requires dtm_path so object/canopy height "
                    "can be separated from absolute ground elevation"
                )
            validate_highbuild_manifest_contract(
                sample_id=sample_id,
                surface_path=surface_path,
                target_kind=target_kind,
                building_mask_path=building_mask_path,
                valid_mask_path=valid_mask_path,
            )
            record = SurfaceSampleRecord(
                sample_id=sample_id,
                region=region,
                landscape=landscape,
                rgb_path=rgb_path,
                surface_path=surface_path,
                target_kind=target_kind,
                dtm_path=dtm_path,
                building_mask_path=building_mask_path,
                vegetation_mask_path=vegetation_mask_path,
                valid_mask_path=valid_mask_path,
                relative_prior_path=relative_prior_path,
                gsd_m=(float(row["gsd_m"]) if (row.get("gsd_m") or "").strip() else None),
            )
            if record.landscape == "mixed" and (
                record.building_mask_path is None or record.vegetation_mask_path is None
            ):
                raise ValueError(
                    f"Mixed sample {sample_id} requires building and vegetation masks"
                )
            if check_files:
                candidates = (
                    record.rgb_path,
                    record.surface_path,
                    record.dtm_path,
                    record.building_mask_path,
                    record.vegetation_mask_path,
                    record.valid_mask_path,
                    record.relative_prior_path,
                )
                missing_paths = [item for item in candidates if item is not None and not item.is_file()]
                if missing_paths:
                    raise FileNotFoundError(
                        f"Missing file(s) for {sample_id}: "
                        + ", ".join(str(item) for item in missing_paths)
                    )
            records.append(record)
    if not records:
        raise ValueError(f"Surface manifest contains no samples: {manifest_path}")
    return records


def assert_surface_regions_disjoint(*splits: list[SurfaceSampleRecord]) -> None:
    region_sets = [{record.region for record in split} for split in splits]
    for left in range(len(region_sets)):
        for right in range(left + 1, len(region_sets)):
            overlap = region_sets[left] & region_sets[right]
            if overlap:
                raise ValueError(
                    f"Geographic leakage between surface splits: {sorted(overlap)}"
                )


def _aligned(source: rasterio.DatasetReader, reference: rasterio.DatasetReader) -> bool:
    return (
        source.width == reference.width
        and source.height == reference.height
        and source.crs == reference.crs
        and np.allclose(tuple(source.transform), tuple(reference.transform), rtol=0.0, atol=1e-9)
    )


def _read_band(
    path: Path,
    reference: rasterio.DatasetReader,
    window: Window,
    *,
    sample_id: str,
) -> tuple[np.ndarray, float | None]:
    with rasterio.open(path) as source:
        if not _aligned(source, reference):
            raise ValueError(f"Unaligned raster for {sample_id}: {path}")
        return source.read(1, window=window).astype(np.float32), source.nodata


def _valid_values(values: np.ndarray, nodata: float | None) -> np.ndarray:
    valid = np.isfinite(values)
    if nodata is not None:
        valid &= ~np.isclose(values, nodata, equal_nan=True)
    return valid


def _normalize_prior(prior: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros_like(prior, dtype=np.float32)
    finite = valid & np.isfinite(prior)
    if np.any(finite):
        low, high = np.percentile(prior[finite], [2, 98])
        if high > low:
            output[finite] = np.clip((prior[finite] - low) / (high - low), 0, 1)
    return output


class MultiDomainSurfaceDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Read aligned full-surface targets and semantic supervision safely."""

    def __init__(
        self,
        records: list[SurfaceSampleRecord],
        *,
        patch_size: int = 384,
        random_crop: bool = False,
        augment: bool = False,
        samples_per_epoch: int | None = None,
        rgb_scale: float = 255.0,
        height_max_m: float = 200.0,
        building_threshold_m: float = 2.0,
        radiometric_policy: str = "raw",
        relative_prior_policy: str = "target_crop_normalized",
        supervised_crop_probability: float = 0.0,
    ) -> None:
        if not records:
            raise ValueError("At least one surface record is required")
        if patch_size <= 0 or rgb_scale <= 0 or height_max_m <= 0:
            raise ValueError("patch_size, rgb_scale, and height_max_m must be positive")
        for record in records:
            # Callers may construct records directly instead of using the CSV
            # loader; keep the HighBuild contract fail-closed on both paths.
            validate_highbuild_manifest_contract(
                sample_id=record.sample_id,
                surface_path=record.surface_path,
                target_kind=record.target_kind,
                building_mask_path=record.building_mask_path,
                valid_mask_path=record.valid_mask_path,
            )
        self.records = records
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.augment = augment
        self.samples_per_epoch = samples_per_epoch
        self.rgb_scale = rgb_scale
        self.height_max_m = height_max_m
        self.building_threshold_m = building_threshold_m
        if not 0.0 <= supervised_crop_probability <= 1.0:
            raise ValueError("supervised_crop_probability must be within [0, 1]")
        self.supervised_crop_probability = supervised_crop_probability
        if radiometric_policy not in RADIOMETRIC_POLICIES:
            raise ValueError(
                f"radiometric_policy must be one of {sorted(RADIOMETRIC_POLICIES)}"
            )
        self.radiometric_policy = radiometric_policy
        if relative_prior_policy not in {
            "target_crop_normalized",
            "stored_01",
        }:
            raise ValueError(
                "relative_prior_policy must be 'target_crop_normalized' or 'stored_01'"
            )
        self.relative_prior_policy = relative_prior_policy

    def __len__(self) -> int:
        return self.samples_per_epoch or len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index % len(self.records)]
        with rasterio.open(record.rgb_path) as rgb, rasterio.open(record.surface_path) as surface:
            if not _aligned(surface, rgb):
                raise ValueError(f"Unaligned RGB/surface rasters for {record.sample_id}")
            focused_origin: tuple[int, int] | None = None
            if (
                self.random_crop
                and record.target_kind == "building_height"
                and record.valid_mask_path is not None
                and self.supervised_crop_probability > 0.0
                and float(torch.rand(1).item()) < self.supervised_crop_probability
            ):
                with rasterio.open(record.valid_mask_path) as supervision_source:
                    if not _aligned(supervision_source, rgb):
                        raise ValueError(
                            "Unaligned supervision mask for " f"{record.sample_id}"
                        )
                    supervision = supervision_source.read(1).astype(np.float32)
                    focus = _valid_values(
                        supervision, supervision_source.nodata
                    ) & (supervision > 0)
                coordinates = np.argwhere(focus)
                if len(coordinates):
                    selected = coordinates[
                        int(torch.randint(0, len(coordinates), (1,)).item())
                    ]
                    origins: list[int] = []
                    for coordinate, length in zip(
                        selected, (rgb.height, rgb.width)
                    ):
                        maximum = max(length - self.patch_size, 0)
                        lower = max(int(coordinate) - self.patch_size + 1, 0)
                        upper = min(int(coordinate), maximum)
                        origin = (
                            lower
                            if upper <= lower
                            else int(torch.randint(lower, upper + 1, (1,)).item())
                        )
                        origins.append(origin)
                    focused_origin = (origins[0], origins[1])
            if focused_origin is None:
                row = _crop_origin(rgb.height, self.patch_size, self.random_crop)
                col = _crop_origin(rgb.width, self.patch_size, self.random_crop)
            else:
                row, col = focused_origin
            read_height = min(self.patch_size, rgb.height - row)
            read_width = min(self.patch_size, rgb.width - col)
            window = Window(col, row, read_width, read_height)
            image = rgb.read((1, 2, 3), window=window).astype(np.float32)
            image_valid = np.all(np.isfinite(image), axis=0) & np.all(
                rgb.read_masks((1, 2, 3), window=window) > 0,
                axis=0,
            )
            target = surface.read(1, window=window).astype(np.float32)
            target_nodata = surface.nodata

            height_valid = _valid_values(target, target_nodata)
            if record.target_kind == "dsm":
                assert record.dtm_path is not None
                terrain, terrain_nodata = _read_band(
                    record.dtm_path, rgb, window, sample_id=record.sample_id
                )
                height_valid &= _valid_values(terrain, terrain_nodata)
                target = target - terrain
            target = np.maximum(target, 0.0)
            height_valid &= target <= self.height_max_m

            if record.valid_mask_path:
                validity, validity_nodata = _read_band(
                    record.valid_mask_path, rgb, window, sample_id=record.sample_id
                )
                height_valid &= _valid_values(validity, validity_nodata) & (
                    validity > 0
                )
            valid = image_valid & height_valid

            explicit_building = None
            if record.building_mask_path:
                values, nodata = _read_band(
                    record.building_mask_path, rgb, window, sample_id=record.sample_id
                )
                explicit_building = _valid_values(values, nodata) & (values > 0)
            explicit_vegetation = None
            if record.vegetation_mask_path:
                values, nodata = _read_band(
                    record.vegetation_mask_path, rgb, window, sample_id=record.sample_id
                )
                explicit_vegetation = _valid_values(values, nodata) & (values > 0)

            if record.target_kind == "building_height" and explicit_building is not None:
                # Footprints are valid semantic labels even when their height
                # is estimated or unavailable under the selected regression
                # protocol. Height eligibility is handled separately below.
                building = explicit_building
            elif explicit_building is not None:
                building = explicit_building & valid
            elif record.landscape == "urban" or record.target_kind == "building_height":
                building = valid & (target >= self.building_threshold_m)
            else:
                building = np.zeros_like(valid)
            if explicit_vegetation is not None:
                vegetation = explicit_vegetation & valid & ~building
            elif record.landscape == "forest":
                vegetation = valid & ~building
            else:
                vegetation = np.zeros_like(valid)

            domain = np.full(valid.shape, LANDSCAPE_CLASSES["ground"], dtype=np.int64)
            domain[vegetation] = LANDSCAPE_CLASSES["vegetation"]
            domain[building] = LANDSCAPE_CLASSES["building"]
            domain_valid = valid.copy()
            regression_valid = valid.copy()
            if record.target_kind == "building_height":
                # A building-only map says nothing about canopy/ground surface.
                domain_valid = image_valid & building
                regression_valid = valid & building

            if record.relative_prior_path:
                prior_values, prior_nodata = _read_band(
                    record.relative_prior_path, rgb, window, sample_id=record.sample_id
                )
                if self.relative_prior_policy == "stored_01":
                    # Precomputed priors are already globally normalized exactly
                    # as the app consumes them. Reference validity must not alter
                    # any model input used to learn a deployment-time route.
                    relative_prior = np.nan_to_num(
                        prior_values, nan=0.0, posinf=1.0, neginf=0.0
                    ).clip(0.0, 1.0).astype(np.float32)
                else:
                    prior_valid = _valid_values(prior_values, prior_nodata)
                    relative_prior = _normalize_prior(prior_values, valid & prior_valid)
            else:
                relative_prior = np.zeros_like(target, dtype=np.float32)

        arrays: dict[str, np.ndarray] = {
            "image": image,
            "height": np.where(regression_valid, target, 0.0).astype(np.float32),
            "image_valid_mask": image_valid,
            "classification_valid_mask": np.zeros_like(valid),
            "height_valid_mask": height_valid,
            # Generic legacy manifests do not carry the six native GAMUS
            # labels.  Keep an explicit ignored target so mixed replay remains
            # collatable without inventing fine classes.
            "fine_class_target": np.full(
                valid.shape, 255, dtype=np.int64
            ),
            "dark_pixel_proxy_mask": np.zeros_like(valid),
            "valid_mask": valid,
            "regression_mask": regression_valid,
            "building_mask": building,
            "vegetation_mask": vegetation,
            "domain_target": domain,
            "domain_valid_mask": domain_valid,
            "relative_prior": relative_prior,
        }
        fill_values = {
            "image": 0.0,
            "height": 0.0,
            "image_valid_mask": False,
            "classification_valid_mask": False,
            "height_valid_mask": False,
            "fine_class_target": 255,
            "dark_pixel_proxy_mask": False,
            "valid_mask": False,
            "regression_mask": False,
            "building_mask": False,
            "vegetation_mask": False,
            "domain_target": 0,
            "domain_valid_mask": False,
            "relative_prior": 0.0,
        }
        for name in arrays:
            arrays[name] = _pad_to_patch(arrays[name], self.patch_size, fill=fill_values[name])

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
        arrays["image"] = (arrays["image"] / self.rgb_scale - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(np.ascontiguousarray(arrays["image"])).float(),
            "height": torch.from_numpy(np.ascontiguousarray(arrays["height"][None])).float(),
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
            "valid_mask": torch.from_numpy(np.ascontiguousarray(arrays["valid_mask"][None])).bool(),
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
            "landscape": record.landscape,
        }
