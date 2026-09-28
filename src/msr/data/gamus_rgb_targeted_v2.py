"""Versioned, training-only RGB crop targeting without changing the V1 reader.

The image sampler remains V1's city-normalized water-hit sampler. This module
only chooses a spatial window after that sampler has selected a training tile.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .gamus_dataset import GAMUS_SIX_CLASS_NAMES, GAMUS_SOURCE_ID_TO_SIX_CLASS, _declared_nodata, _image_dataset, _require_h5py
from .gamus_rgb_segmentation import (
    ALLOWED_CITIES, IGNORE_INDEX, GamusRgbRecord, GamusRgbSegmentationDataset, authenticated_json,
    canonical_sha256, make_datasets, source_metadata_snapshot,
)
from .raster_dataset import _crop_origin


TARGET_INDEX_SCHEMA = "msr.gamus_rgb_targeted_crop_index.v2"
TARGET_NAMES = ("ground", "low_vegetation")
TARGET_CLASS_IDS = tuple(GAMUS_SIX_CLASS_NAMES.index(name) for name in TARGET_NAMES)


def read_rgb_class_window(record: GamusRgbRecord, *, native_shape: int, patch_size: int,
                          row: int, col: int, rgb_scale: float = 255.0,
                          dark_pixel_threshold: float = .15) -> dict[str, np.ndarray]:
    """Isolated V1-equivalent decoding and validity; opens RGB and CLS only."""
    if record.city not in ALLOWED_CITIES or any(path.parent.name != "train" for path in (record.image_path, record.class_path)):
        raise ValueError("V2 crop reader permits DC/PHL training RGB/CLS only")
    if record.image_path.name != f"{record.sample_id}_RGB.h5" or record.class_path.name != f"{record.sample_id}_CLS.h5":
        raise ValueError("V2 crop reader source roles do not match approved record")
    if not (0 <= row <= native_shape - patch_size and 0 <= col <= native_shape - patch_size):
        raise ValueError("Crop origin lies outside native tile")
    h5 = _require_h5py()
    with h5.File(record.image_path, "r") as ih, h5.File(record.class_path, "r") as ch:
        rgb, cls = _image_dataset(ih, record.image_path), _image_dataset(ch, record.class_path)
        if rgb.ndim != 3 or rgb.shape[-1] < 3 or cls.ndim != 2 or tuple(rgb.shape[:2]) != tuple(cls.shape):
            raise ValueError(f"Unaligned HWC RGB and 2D CLS: {record.sample_id}")
        if tuple(cls.shape) != (native_shape, native_shape):
            raise ValueError(f"Unexpected native tile shape: {record.sample_id}: {cls.shape}")
        window = np.s_[row:row + patch_size, col:col + patch_size]
        image = np.asarray(rgb[window[0], window[1], :3], dtype=np.float32).transpose(2, 0, 1)
        classes = np.asarray(cls[window], dtype=np.float32)
        image_nodata = _declared_nodata(rgb, ih)
    image_valid = np.isfinite(image).all(axis=0)
    for nodata in image_nodata:
        image_valid &= ~np.all(np.isclose(image, nodata, equal_nan=True), axis=0)
    if np.any((image[:, image_valid] < 0) | (image[:, image_valid] > rgb_scale)):
        raise ValueError(f"Valid RGB values outside [0,rgb_scale]: {record.sample_id}")
    rounded = np.rint(classes)
    class_available = np.isfinite(classes)
    legal = np.isclose(classes, rounded, rtol=0, atol=1e-5) & (rounded >= 0) & (rounded <= 6)
    if np.any(class_available & ~legal):
        raise ValueError(f"Unexpected source class IDs in {record.sample_id}: {np.unique(classes[class_available & ~legal])[:8]}")
    class_valid = class_available & legal & (rounded != 0)
    labels = np.full(classes.shape, IGNORE_INDEX, dtype=np.int64)
    for source_id, target in GAMUS_SOURCE_ID_TO_SIX_CLASS.items():
        labels[class_valid & (rounded == source_id)] = target
    image[:, ~image_valid] = 0
    image /= rgb_scale
    return {"image": image, "labels": labels, "image_valid_mask": image_valid,
            "classification_valid_mask": class_valid,
            "dark_pixel_proxy_mask": image_valid & (image.mean(axis=0) <= dark_pixel_threshold)}


def candidate_origins(native_shape: int, patch_size: int, stride: int) -> list[list[int]]:
    if not 0 < patch_size <= native_shape or stride <= 0:
        raise ValueError("Invalid candidate grid dimensions")
    edge = native_shape - patch_size
    axis = sorted(set(range(0, edge + 1, stride)) | {edge})
    return [[row, col] for row in axis for col in axis]


def window_counts(mask: np.ndarray, patch_size: int) -> np.ndarray:
    """Exact counts for all ordinary uniform integer crop origins."""
    integral = np.pad(mask.astype(np.int32).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32), ((1, 0), (1, 0)))
    p = patch_size
    return integral[p:, p:] - integral[:-p, p:] - integral[p:, :-p] + integral[:-p, :-p]


def target_candidate_eligibility(counts: np.ndarray, minimum: int, target_quantile: float) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Richer windows in this same selected tile, with an absolute support floor."""
    thresholds = {name: max(minimum, math.ceil(float(np.quantile(counts[:, cid], target_quantile))))
                  for name, cid in zip(TARGET_NAMES, TARGET_CLASS_IDS)}
    eligible = {name: np.flatnonzero(counts[:, cid] >= thresholds[name]).tolist()
                for name, cid in zip(TARGET_NAMES, TARGET_CLASS_IDS)}
    return thresholds, eligible


def training_tile_statistics(arrays: Mapping[str, np.ndarray], patch_size: int,
                             origins: list[list[int]], min_target_pixels: int,
                             target_quantile: float = .75) -> dict[str, Any]:
    """Count joint RGB/CLS-valid support; no height-derived eligibility."""
    labels = arrays["labels"]
    valid = arrays["image_valid_mask"] & arrays["classification_valid_mask"]
    if labels.ndim != 2 or labels.shape[0] != labels.shape[1] or patch_size > labels.shape[0]:
        raise ValueError("Expected square native tile and valid patch")
    count = np.zeros((len(origins), len(GAMUS_SIX_CLASS_NAMES)), dtype=np.int64)
    rr, cc = np.asarray(origins).T
    expected, tile_counts, hit, support = {}, {}, {}, {}
    for class_id, name in enumerate(GAMUS_SIX_CLASS_NAMES):
        mask = valid & (labels == class_id)
        all_counts = window_counts(mask, patch_size)
        count[:, class_id] = all_counts[rr, cc]
        expected[name] = float(all_counts.mean())
        tile_counts[name] = int(mask.sum())
        hit[name] = float((all_counts > 0).mean())
        support[name] = float((all_counts >= min_target_pixels).mean())
    thresholds, eligible = target_candidate_eligibility(count, min_target_pixels, target_quantile)
    return {"joint_valid_class_pixels": tile_counts, "joint_valid_pixels": int(valid.sum()),
            "class_valid_pixels": int(arrays["classification_valid_mask"].sum()),
            "image_valid_pixels": int(arrays["image_valid_mask"].sum()),
            "candidate_valid_class_counts": count.tolist(), "eligible_candidate_indices": eligible,
            "target_minimum_valid_pixels": thresholds,
            "ordinary_uniform_crop": {"expected_valid_class_pixels": expected,
                "class_hit_probability": hit, "meaningful_support_probability": support}}


class GamusRgbTargetedCropDatasetV2(GamusRgbSegmentationDataset):
    """Same V1 RGB/labels/augmentation, with independent RNG for crop targeting.

    Both arms always draw the V1 ordinary crop and V1 geometry from the normal
    worker RNG. A separate worker-seeded RNG chooses mixture/target/window with
    exactly three draws per item, including ordinary/fallback items. Thus p=0
    is byte-equivalent to V1 at the same torch seed, and p=0/.5 preserve the
    common ordinary proposals and geometry. Image sampling is external.
    """

    def __init__(self, root: str | Path, split: str = "train", *,
                 targeted_index_path: str | Path, targeted_index_sha256: str,
                 target_probability: float = .5, **kwargs: Any) -> None:
        if split != "train":
            raise ValueError("Targeted crops are training-only; no validation/test access")
        if not math.isfinite(target_probability) or not 0 <= target_probability <= 1:
            raise ValueError("target_probability must lie in [0,1]")
        kwargs.setdefault("random_crop", True)
        if kwargs["random_crop"] is not True:
            raise ValueError("Both V2 arms require ordinary uniform random crop proposals")
        super().__init__(root, split, **kwargs)
        payload = authenticated_json(targeted_index_path, targeted_index_sha256)
        if payload.get("schema") != TARGET_INDEX_SCHEMA or payload.get("split") != "train":
            raise ValueError("Target index must be the training-only V2 schema")
        if payload.get("approved_index_sha256") != self.approved_index_sha256:
            raise ValueError("Target index approved source identity differs")
        if payload.get("train_ids_sha256") != canonical_sha256(list(self.sample_ids)):
            raise ValueError("Target index ordered train IDs differ")
        if payload.get("native_shape") != self.native_shape or payload.get("patch_size") != self.patch_size:
            raise ValueError("Target index crop/native shape differs")
        if payload.get("rgb_scale") != self.rgb_scale:
            raise ValueError("Target index RGB validity scale differs")
        if payload.get("class_names") != list(GAMUS_SIX_CLASS_NAMES) or payload.get("target_names") != list(TARGET_NAMES):
            raise ValueError("Target index class order differs")
        if payload.get("source_metadata") != source_metadata_snapshot(self):
            raise ValueError("Training RGB/CLS source metadata differs from target index")
        minimum = payload.get("min_target_pixels")
        if type(minimum) is not int or not 0 < minimum <= self.patch_size ** 2:
            raise ValueError("Invalid meaningful target support threshold")
        quantile = payload.get("target_quantile")
        if not isinstance(quantile, (int, float)) or not math.isfinite(quantile) or not 0 <= quantile <= 1:
            raise ValueError("Invalid target crop quantile")
        origins = payload.get("candidate_origins")
        if not isinstance(origins, list) or not origins or any(
            not isinstance(x, list) or len(x) != 2 or any(type(v) is not int or not 0 <= v <= self.native_shape - self.patch_size for v in x)
            for x in origins
        ) or len({tuple(x) for x in origins}) != len(origins):
            raise ValueError("Invalid or duplicate target crop origins")
        rows = payload.get("tiles")
        if not isinstance(rows, list) or [r.get("sample_id") for r in rows] != list(self.sample_ids):
            raise ValueError("Target index must match every approved training ID in order")
        self._eligible = []
        for record, row in zip(self.records, rows):
            if row.get("city") != record.city:
                raise ValueError("Target index city identity differs")
            counts = np.asarray(row.get("candidate_valid_class_counts"))
            if counts.shape != (len(origins), len(GAMUS_SIX_CLASS_NAMES)) or counts.dtype.kind not in "iu" or np.any(counts < 0) or np.any(counts > self.patch_size ** 2) or np.any(counts.sum(1) > self.patch_size ** 2):
                raise ValueError("Invalid candidate class pixel counts")
            thresholds, eligible = target_candidate_eligibility(counts, minimum, quantile)
            if row.get("target_minimum_valid_pixels") != thresholds:
                raise ValueError("Target candidate support thresholds contradict the quantile rule")
            if row.get("eligible_candidate_indices") != eligible:
                raise ValueError("Target candidate eligibility contradicts valid class counts")
            self._eligible.append([(name, ids) for name, ids in eligible.items() if ids])
        self.target_probability = float(target_probability)
        self.targeted_index_path = Path(targeted_index_path).resolve()
        self.targeted_index_sha256 = targeted_index_sha256
        self.candidate_origins = origins
        self.min_target_pixels = minimum
        self.target_quantile = float(quantile)
        self._target_generator: torch.Generator | None = None
        self._target_seed: int | None = None

    def choose_crop(self, index: int) -> tuple[int, int, str]:
        row = _crop_origin(self.native_shape, self.patch_size, True)
        col = _crop_origin(self.native_shape, self.patch_size, True)
        seed = torch.initial_seed()
        if self._target_generator is None or self._target_seed != seed:
            self._target_generator = torch.Generator().manual_seed(seed ^ 0x47524F554E44)
            self._target_seed = seed
        mixture, target_draw, window_draw = torch.rand(3, generator=self._target_generator).tolist()
        eligible = self._eligible[index]
        if mixture < self.target_probability and eligible:
            target, candidates = eligible[min(int(target_draw * len(eligible)), len(eligible) - 1)]
            chosen = candidates[min(int(window_draw * len(candidates)), len(candidates) - 1)]
            row, col = self.candidate_origins[chosen]
            return row, col, target
        return row, col, "ordinary"

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        row, col, _ = self.choose_crop(index)
        arrays = read_rgb_class_window(record, native_shape=self.native_shape, patch_size=self.patch_size,
            row=row, col=col, rgb_scale=self.rgb_scale, dark_pixel_threshold=self.dark_pixel_threshold)
        if self.augment:
            turns = int(torch.randint(0, 4, (1,)).item())
            flip = bool(torch.randint(0, 2, (1,)).item())
            for name, value in arrays.items():
                if turns:
                    value = np.rot90(value, turns, axes=(-2, -1))
                if flip:
                    value = np.flip(value, axis=-1)
                arrays[name] = value
        return {**{key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in arrays.items()},
                "sample_id": record.sample_id, "city": record.city, "region": record.region}


def make_datasets_v2(config: Mapping[str, Any]):
    """V1 full-native validation and the matched-control/targeted training arm."""
    _, validation, contract = make_datasets(config)
    data, sampling = config["data"], config["sampling"]
    arm = sampling["arm"]
    if arm not in ("control", "targeted"):
        raise ValueError("V2 sampling arm must be control or targeted")
    probability = 0.0 if arm == "control" else float(sampling.get("target_probability", .5))
    train = GamusRgbTargetedCropDatasetV2(data["root"], "train",
        approved_index_path=data["approved_index_path"], approved_index_sha256=data["approved_index_file_sha256"],
        targeted_index_path=sampling["targeted_index_path"], targeted_index_sha256=sampling["targeted_index_sha256"],
        target_probability=probability, patch_size=int(data.get("patch_size", 384)), random_crop=True, augment=True,
        rgb_scale=float(data.get("rgb_scale", 255)), dark_pixel_threshold=float(data.get("dark_pixel_threshold", .15)))
    contract = {**contract, "schema": "msr.gamus_rgb_only_inputs.v2", "sampling_arm": arm,
        "target_probability": probability, "targeted_index_sha256": train.targeted_index_sha256,
        "min_target_pixels": train.min_target_pixels, "target_quantile": train.target_quantile,
        "image_sampler": "unchanged V1 city-normalized water-hit"}
    contract.pop("contract_sha256", None)
    contract["contract_sha256"] = canonical_sha256(contract)
    return train, validation, contract
