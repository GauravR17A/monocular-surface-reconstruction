"""Explicit dataset manifests and RGB/height alignment checks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio


REQUIRED_COLUMNS = {"sample_id", "region", "rgb_path", "height_path"}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    region: str
    rgb_path: Path
    height_path: Path
    mask_path: Path | None = None
    gsd_m: float | None = None


@dataclass(frozen=True)
class PairAudit:
    sample_id: str
    rgb_shape: tuple[int, int]
    height_shape: tuple[int, int]
    shapes_match: bool
    transforms_match: bool
    crs_match: bool
    rgb_band_count: int
    height_band_count: int
    finite_height_fraction: float
    valid_height_fraction: float
    below_height_range_fraction: float
    above_height_range_fraction: float
    height_min_m: float | None
    height_max_m: float | None
    raw_finite_height_min_m: float | None
    raw_finite_height_max_m: float | None
    height_nodata: float | None

    @property
    def aligned(self) -> bool:
        return self.shapes_match and self.transforms_match and self.crs_match


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_manifest(path: str | Path, *, check_files: bool = True) -> list[SampleRecord]:
    """Load a CSV manifest and resolve paths relative to the manifest directory."""

    manifest_path = Path(path).resolve()
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        records: list[SampleRecord] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            region = (row.get("region") or "").strip()
            if not sample_id or not region:
                raise ValueError(f"Row {row_number} must define sample_id and region")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
            seen_ids.add(sample_id)

            rgb_path = _resolve_path(row["rgb_path"], manifest_path.parent)
            height_path = _resolve_path(row["height_path"], manifest_path.parent)
            mask_value = (row.get("mask_path") or "").strip()
            mask_path = _resolve_path(mask_value, manifest_path.parent) if mask_value else None
            gsd_value = (row.get("gsd_m") or "").strip()
            gsd_m = float(gsd_value) if gsd_value else None
            record = SampleRecord(
                sample_id=sample_id,
                region=region,
                rgb_path=rgb_path,
                height_path=height_path,
                mask_path=mask_path,
                gsd_m=gsd_m,
            )
            if check_files:
                missing_paths = [
                    candidate
                    for candidate in (record.rgb_path, record.height_path, record.mask_path)
                    if candidate is not None and not candidate.is_file()
                ]
                if missing_paths:
                    raise FileNotFoundError(
                        f"Missing file(s) for {sample_id}: "
                        + ", ".join(str(candidate) for candidate in missing_paths)
                    )
            records.append(record)

    if not records:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")
    return records


def assert_disjoint_regions(*splits: Iterable[SampleRecord]) -> None:
    """Reject geographic leakage between any supplied data splits."""

    region_sets = [{record.region for record in split} for split in splits]
    for left in range(len(region_sets)):
        for right in range(left + 1, len(region_sets)):
            overlap = region_sets[left] & region_sets[right]
            if overlap:
                raise ValueError(
                    f"Geographic leakage between splits {left} and {right}: {sorted(overlap)}"
                )


def _transforms_match(left: rasterio.Affine, right: rasterio.Affine) -> bool:
    return bool(np.allclose(tuple(left), tuple(right), rtol=0.0, atol=1e-9))


def audit_pair(record: SampleRecord, *, height_min_m: float = 0.0, height_max_m: float = 1000.0) -> PairAudit:
    """Inspect dimensions, georeferencing, nodata, units range, and finite coverage."""

    with rasterio.open(record.rgb_path) as rgb, rasterio.open(record.height_path) as height:
        height_values = height.read(1, masked=False).astype(np.float64, copy=False)
        finite = np.isfinite(height_values)
        valid = finite.copy()
        if height.nodata is not None:
            valid &= ~np.isclose(height_values, height.nodata, equal_nan=True)
        eligible = valid.copy()
        below_range = eligible & (height_values < height_min_m)
        above_range = eligible & (height_values > height_max_m)
        valid &= height_values >= height_min_m
        valid &= height_values <= height_max_m

        valid_values = height_values[valid]
        finite_values = height_values[eligible]
        return PairAudit(
            sample_id=record.sample_id,
            rgb_shape=(rgb.height, rgb.width),
            height_shape=(height.height, height.width),
            shapes_match=(rgb.height, rgb.width) == (height.height, height.width),
            transforms_match=_transforms_match(rgb.transform, height.transform),
            crs_match=rgb.crs == height.crs,
            rgb_band_count=rgb.count,
            height_band_count=height.count,
            finite_height_fraction=float(finite.mean()),
            valid_height_fraction=float(valid.mean()),
            below_height_range_fraction=float(below_range.mean()),
            above_height_range_fraction=float(above_range.mean()),
            height_min_m=float(valid_values.min()) if valid_values.size else None,
            height_max_m=float(valid_values.max()) if valid_values.size else None,
            raw_finite_height_min_m=(
                float(finite_values.min()) if finite_values.size else None
            ),
            raw_finite_height_max_m=(
                float(finite_values.max()) if finite_values.size else None
            ),
            height_nodata=float(height.nodata) if height.nodata is not None else None,
        )
