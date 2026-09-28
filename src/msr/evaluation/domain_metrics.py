"""Landscape-aware DSM evaluation without letting easy ground pixels hide failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .metrics import compute_height_metrics


DOMAIN_CODES: Mapping[str, int] = {
    "plains": 1,
    "hilly_terrain": 2,
    "forest": 3,
    "urban": 4,
}


@dataclass(frozen=True)
class DomainEvaluation:
    """Metrics, exclusive landscape masks, and an auditable domain raster."""

    metrics: dict[str, object]
    masks: dict[str, np.ndarray]
    domain_map: np.ndarray


def terrain_slope_degrees(
    terrain_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> np.ndarray:
    """Estimate slope in degrees from a metric terrain grid."""

    terrain = np.asarray(terrain_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(terrain)
    if terrain.ndim != 2 or valid.shape != terrain.shape:
        raise ValueError("terrain and valid_mask must be aligned 2-D arrays")
    if pixel_size_x_m <= 0 or pixel_size_y_m <= 0:
        raise ValueError("metric pixel sizes must be positive")
    finite_values = terrain[valid]
    if finite_values.size == 0:
        return np.full(terrain.shape, np.nan, dtype=np.float32)

    # Invalid pixels are filled only to keep the finite-difference operation
    # numerical. Their slopes are removed again below.
    filled = np.where(valid, terrain, float(np.median(finite_values)))
    dz_dy, dz_dx = np.gradient(filled, pixel_size_y_m, pixel_size_x_m)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)
    slope[~valid] = np.nan
    return slope


def build_domain_masks(
    valid_mask: np.ndarray,
    *,
    building_mask: np.ndarray | None = None,
    vegetation_mask: np.ndarray | None = None,
    terrain_slope_deg: np.ndarray | None = None,
    hilly_threshold_deg: float = 5.0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Create mutually exclusive masks in urban, forest, hill, plain priority.

    Exclusivity is intentional: aggregate domain scores remain interpretable and
    cannot double-count the same easy or difficult pixels.
    """

    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("valid_mask must be a 2-D array")
    if hilly_threshold_deg < 0:
        raise ValueError("hilly_threshold_deg must be non-negative")

    def optional_mask(value: np.ndarray | None, name: str) -> np.ndarray:
        if value is None:
            return np.zeros_like(valid)
        result = np.asarray(value, dtype=bool)
        if result.shape != valid.shape:
            raise ValueError(f"{name} must match valid_mask")
        return result & valid

    urban = optional_mask(building_mask, "building_mask")
    forest = optional_mask(vegetation_mask, "vegetation_mask") & ~urban
    assigned = urban | forest

    if terrain_slope_deg is None:
        hilly = np.zeros_like(valid)
    else:
        slope = np.asarray(terrain_slope_deg, dtype=np.float64)
        if slope.shape != valid.shape:
            raise ValueError("terrain_slope_deg must match valid_mask")
        hilly = valid & np.isfinite(slope) & (slope >= hilly_threshold_deg) & ~assigned
    plains = valid & ~(assigned | hilly)

    masks = {
        "urban": urban,
        "forest": forest,
        "hilly_terrain": hilly,
        "plains": plains,
    }
    domain_map = np.zeros(valid.shape, dtype=np.uint8)
    for name, code in DOMAIN_CODES.items():
        domain_map[masks[name]] = code
    return masks, domain_map


def evaluate_by_domain(
    prediction_m: np.ndarray,
    reference_m: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    building_mask: np.ndarray | None = None,
    vegetation_mask: np.ndarray | None = None,
    terrain_slope_deg: np.ndarray | None = None,
    hilly_threshold_deg: float = 5.0,
) -> DomainEvaluation:
    """Compute overall and separate domain metrics plus macro/worst summaries."""

    prediction = np.asarray(prediction_m, dtype=np.float64)
    reference = np.asarray(reference_m, dtype=np.float64)
    if prediction.shape != reference.shape or prediction.ndim != 2:
        raise ValueError("prediction and reference must be aligned 2-D arrays")
    valid = np.isfinite(prediction) & np.isfinite(reference)
    if valid_mask is not None:
        supplied_valid = np.asarray(valid_mask, dtype=bool)
        if supplied_valid.shape != valid.shape:
            raise ValueError("valid_mask must match prediction and reference")
        valid &= supplied_valid
    if not np.any(valid):
        raise ValueError("No valid pixels remain for domain evaluation")

    masks, domain_map = build_domain_masks(
        valid,
        building_mask=building_mask,
        vegetation_mask=vegetation_mask,
        terrain_slope_deg=terrain_slope_deg,
        hilly_threshold_deg=hilly_threshold_deg,
    )
    overall = compute_height_metrics(prediction, reference, valid).to_dict()
    domains: dict[str, dict[str, object]] = {}
    for name in ("urban", "forest", "hilly_terrain", "plains"):
        mask = masks[name]
        if np.any(mask):
            result = compute_height_metrics(prediction, reference, mask).to_dict()
            result["coverage_fraction"] = float(mask.sum() / valid.sum())
            domains[name] = result

    rmse_values = [float(result["rmse_m"]) for result in domains.values()]
    summary: dict[str, object] = {
        "overall": overall,
        "domains": domains,
        "domain_macro_rmse_m": float(np.mean(rmse_values)) if rmse_values else None,
        "worst_domain_rmse_m": float(np.max(rmse_values)) if rmse_values else None,
        "worst_domain": (
            max(domains, key=lambda name: float(domains[name]["rmse_m"]))
            if domains
            else None
        ),
        "hilly_threshold_deg": float(hilly_threshold_deg),
        "domain_codes": dict(DOMAIN_CODES),
    }
    return DomainEvaluation(metrics=summary, masks=masks, domain_map=domain_map)
