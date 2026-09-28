"""Semantic display geometry derived from raw height-model outputs."""

from __future__ import annotations

import numpy as np
from rasterio.features import shapes
from scipy import ndimage


def _clean_building_labels(
    building_probability: np.ndarray,
    valid: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, int]:
    mask = valid & (building_probability >= threshold)
    structure = np.ones((3, 3), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
    mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
    return ndimage.label(mask, structure=structure)


def prepare_building_mesh(
    height_m: np.ndarray,
    building_probability: np.ndarray,
    *,
    threshold: float = 0.3,
    minimum_component_pixels: int = 160,
    roof_flatten: float = 0.8,
) -> tuple[np.ndarray, int]:
    """Snap urban height to cleaned footprints and locally flatten roofs."""

    height = np.asarray(height_m, dtype=np.float32)
    probability = np.asarray(building_probability, dtype=np.float32)
    if height.shape != probability.shape:
        raise ValueError("height and building probability must share a grid")
    if not 0 <= roof_flatten <= 1:
        raise ValueError("roof_flatten must be between 0 and 1")

    valid = np.isfinite(height) & np.isfinite(probability)
    labels, count = _clean_building_labels(probability, valid, threshold=threshold)
    output = np.zeros_like(height, dtype=np.float32)
    kept = 0
    for label_index in range(1, count + 1):
        component = labels == label_index
        size = int(component.sum())
        if size < minimum_component_pixels:
            continue
        finite_height = height[component]
        finite_height = finite_height[np.isfinite(finite_height)]
        if finite_height.size == 0:
            continue
        roof_height = float(np.percentile(finite_height, 65))
        local = np.maximum(height[component], 0.0)
        output[component] = roof_flatten * roof_height + (1.0 - roof_flatten) * local
        kept += 1
    output[~valid] = np.nan
    return output, kept


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError("rgb must have shape (3, height, width)")
    finite = image[np.isfinite(image)]
    scale = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    if scale > 1.5:
        scale = 255.0 if scale <= 255.0 else scale
    return np.clip(image[:3] / max(scale, 1e-6), 0.0, 1.0)


def vegetation_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Estimate visibly green vegetation for presentation, not evaluation."""

    image = _normalize_rgb(rgb)
    red, green, blue = image
    excess_green = 2.0 * green - red - blue
    mask = (
        (green > 0.13)
        & (green > red * 1.035)
        & (green > blue * 1.025)
        & (excess_green > 0.035)
    )
    structure = np.ones((3, 3), dtype=bool)
    mask = ndimage.binary_opening(mask, structure=structure, iterations=1)
    return ndimage.binary_closing(mask, structure=structure, iterations=2)


def _rgb_canopy_relief_m(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create conservative presentation relief for visibly green canopy.

    The learned height raster remains the source of all measurements. This
    low-amplitude prior exists only so vegetation that the mixed-domain pilot
    recognizes weakly does not render as paint on a flat plane.
    """

    image = _normalize_rgb(rgb)
    red, green, blue = image
    excess_green = np.clip(2.0 * green - red - blue, 0.0, 1.0)
    colour_strength = np.clip((excess_green - 0.035) / 0.265, 0.0, 1.0)
    # Satellite-visible shrubs and compact tree crowns get a restrained
    # 0.8-3.5 m display relief. It is intentionally well below the old
    # 9-10 m generic bush proxy that looked implausible on mixed scenes.
    relief = 0.8 + 2.7 * np.sqrt(colour_strength)
    edge_distance = ndimage.distance_transform_edt(mask)
    crown_shape = 0.55 + 0.45 * np.sqrt(np.clip(edge_distance / 6.0, 0.0, 1.0))
    relief *= crown_shape
    relief = ndimage.gaussian_filter(relief * mask, sigma=1.1)
    relief[~mask] = 0.0
    return relief.astype(np.float32)


def resolve_semantic_masks(
    height_m: np.ndarray,
    building_probability: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    vegetation_probability: np.ndarray | None = None,
    building_threshold: float = 0.3,
    vegetation_threshold: float = 0.35,
    minimum_building_pixels: int = 160,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve one mutually exclusive surface/building/vegetation map.

    Building footprints come from the protected urban expert. Vegetation uses
    the learned router plus a conservative visible-green fallback, then yields
    to cleaned building footprints. Returning this shared map prevents the API
    mesh and browser inspector from independently assigning different classes
    to the same mixed-scene pixel.

    Class ids are 0=surface, 1=building, 2=vegetation and 255=invalid.
    """

    height = np.asarray(height_m, dtype=np.float32)
    building = np.asarray(building_probability, dtype=np.float32)
    if height.shape != building.shape:
        raise ValueError("height and building probability must share a grid")
    if not 0 <= building_threshold <= 1:
        raise ValueError("building_threshold must be between 0 and 1")
    if not 0 <= vegetation_threshold <= 1:
        raise ValueError("vegetation_threshold must be between 0 and 1")

    valid = np.isfinite(height) & np.isfinite(building)
    labels, label_count = _clean_building_labels(
        building,
        valid,
        threshold=building_threshold,
    )
    building_mask = np.zeros_like(valid)
    for label_index in range(1, label_count + 1):
        component = labels == label_index
        if int(component.sum()) >= minimum_building_pixels:
            building_mask |= component

    if rgb is not None and np.asarray(rgb).shape[1:] != height.shape:
        raise ValueError("rgb and height must share a grid")
    rgb_vegetation = (
        vegetation_mask_from_rgb(rgb) & valid
        if rgb is not None
        else np.zeros_like(valid)
    )

    model_vegetation = np.zeros_like(valid)
    if vegetation_probability is not None:
        vegetation = np.asarray(vegetation_probability, dtype=np.float32)
        if vegetation.shape != height.shape:
            raise ValueError("vegetation probability and height must share a grid")
        model_vegetation = (
            np.isfinite(vegetation)
            & (vegetation >= vegetation_threshold)
            & valid
        )

    vegetation_mask = model_vegetation | rgb_vegetation
    structure = np.ones((3, 3), dtype=bool)
    vegetation_mask = ndimage.binary_opening(
        vegetation_mask,
        structure=structure,
        iterations=1,
    )
    vegetation_mask = ndimage.binary_closing(
        vegetation_mask,
        structure=structure,
        iterations=2,
    )
    # Protected footprints win every overlap. A one-pixel boundary guard stops
    # canopy smoothing from bleeding onto roofs without erasing nearby crowns.
    protected_buildings = ndimage.binary_dilation(building_mask, iterations=1)
    vegetation_mask &= valid & ~protected_buildings

    semantic_class = np.full(height.shape, 255, dtype=np.uint8)
    semantic_class[valid] = 0
    semantic_class[vegetation_mask] = 2
    semantic_class[building_mask] = 1
    return building_mask, vegetation_mask, semantic_class


def prepare_presentation_mesh(
    height_m: np.ndarray,
    building_probability: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    vegetation_probability: np.ndarray | None = None,
    terrain_prior: np.ndarray | None = None,
    threshold: float = 0.3,
    minimum_component_pixels: int = 160,
    roof_flatten: float = 0.8,
    maximum_visual_relief_m: float = 1.25,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compose buildings, vegetation, and subtle terrain for the 3D viewer.

    This result is deliberately presentation-only. Metric exports and reported
    validation scores must continue to use ``height_m`` unchanged.
    """

    height = np.asarray(height_m, dtype=np.float32)
    probability = np.asarray(building_probability, dtype=np.float32)
    building_height, components = prepare_building_mesh(
        height,
        probability,
        threshold=threshold,
        minimum_component_pixels=minimum_component_pixels,
        roof_flatten=roof_flatten,
    )
    valid = np.isfinite(height) & np.isfinite(probability)
    output = np.zeros_like(height, dtype=np.float32)

    if terrain_prior is not None:
        prior = np.asarray(terrain_prior, dtype=np.float32)
        if prior.shape != height.shape:
            raise ValueError("terrain prior and height must share a grid")
        prior_valid = valid & np.isfinite(prior)
        if np.any(prior_valid):
            low, high = np.percentile(prior[prior_valid], [3, 97])
            normalized = np.clip((prior - low) / max(float(high - low), 1e-6), 0, 1)
            terrain = ndimage.gaussian_filter(normalized, sigma=7.0)
            terrain -= float(np.nanpercentile(terrain[prior_valid], 5))
            output += np.clip(terrain, 0, 1) * maximum_visual_relief_m

    building_mask, vegetation_mask, _ = resolve_semantic_masks(
        height,
        probability,
        rgb=rgb,
        vegetation_probability=vegetation_probability,
        building_threshold=threshold,
        minimum_building_pixels=minimum_component_pixels,
    )
    rgb_vegetation = (
        vegetation_mask_from_rgb(rgb) & valid
        if rgb is not None
        else np.zeros_like(valid)
    )
    if np.any(vegetation_mask):
        # Preserve learned canopy height where present. When visible vegetation
        # has near-zero predicted height, use restrained RGB-derived relief so
        # it is still legible in 3D; inspection and exports keep raw height.
        canopy = ndimage.gaussian_filter(np.maximum(height, 0), sigma=2.0)
        if rgb is not None:
            canopy = np.maximum(
                canopy,
                _rgb_canopy_relief_m(rgb, vegetation_mask & rgb_vegetation),
            )
        finite_canopy = canopy[vegetation_mask]
        if finite_canopy.size:
            cap = max(float(np.percentile(finite_canopy, 97)), 1.0)
            canopy = np.clip(canopy, 0, cap)
            output[vegetation_mask] += canopy[vegetation_mask] * 0.72

    output[building_mask] += building_height[building_mask]
    output[~valid] = np.nan
    return output, {
        "building_components": components,
        "building_pixels": int(building_mask.sum()),
        "vegetation_pixels": int(vegetation_mask.sum()),
        "surface_pixels": int((valid & ~building_mask & ~vegetation_mask).sum()),
    }


def extract_building_solids(
    height_m: np.ndarray,
    building_probability: np.ndarray,
    *,
    ground_relief_m: np.ndarray | None = None,
    threshold: float = 0.3,
    minimum_component_pixels: int = 160,
    maximum_components: int = 250,
) -> list[dict[str, object]]:
    """Approximate semantic components as oriented solids for clean rendering.

    These boxes are a visualization abstraction derived from predictions. They
    are never used for metric scoring or exported as reference footprints.
    """

    height = np.asarray(height_m, dtype=np.float32)
    probability = np.asarray(building_probability, dtype=np.float32)
    if height.shape != probability.shape:
        raise ValueError("height and building probability must share a grid")
    ground = (
        np.zeros_like(height, dtype=np.float32)
        if ground_relief_m is None
        else np.asarray(ground_relief_m, dtype=np.float32)
    )
    if ground.shape != height.shape:
        raise ValueError("ground relief and height must share a grid")
    valid = np.isfinite(height) & np.isfinite(probability)
    labels, count = _clean_building_labels(probability, valid, threshold=threshold)
    image_height, image_width = height.shape
    outlines: dict[int, list[list[float]]] = {}
    for geometry, value in shapes(
        labels.astype(np.int32),
        mask=labels > 0,
        connectivity=8,
    ):
        label_value = int(value)
        rings = geometry.get("coordinates", [])
        if not rings:
            continue
        ring = list(rings[0])
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring.pop()
        if len(ring) > 320:
            stride = int(np.ceil(len(ring) / 320))
            ring = ring[::stride]
        outlines[label_value] = [
            [float(x / image_width), float(y / image_height)] for x, y in ring
        ]
    candidates: list[tuple[int, int]] = []
    for label_index in range(1, count + 1):
        size = int(np.count_nonzero(labels == label_index))
        if size >= minimum_component_pixels:
            candidates.append((size, label_index))
    candidates.sort(reverse=True)

    solids: list[dict[str, object]] = []
    for size, label_index in candidates[:maximum_components]:
        rows, cols = np.nonzero(labels == label_index)
        coordinates = np.column_stack((cols, rows)).astype(np.float64)
        center = coordinates.mean(axis=0)
        centered = coordinates - center
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = eigenvectors[:, int(np.argmax(eigenvalues))]
        minor = np.array([-major[1], major[0]])
        projected_major = centered @ major
        projected_minor = centered @ minor
        major_low, major_high = np.percentile(projected_major, [1, 99])
        minor_low, minor_high = np.percentile(projected_minor, [1, 99])
        width_pixels = max(float(major_high - major_low + 1), 2.0)
        depth_pixels = max(float(minor_high - minor_low + 1), 2.0)
        component_height = height[rows, cols]
        component_height = component_height[np.isfinite(component_height)]
        if component_height.size == 0:
            continue
        roof_height = max(float(np.percentile(component_height, 68)), 0.35)
        component_ground = ground[rows, cols]
        component_ground = component_ground[np.isfinite(component_ground)]
        base_height = float(np.median(component_ground)) if component_ground.size else 0.0
        solids.append(
            {
                "id": int(label_index),
                "pixels": size,
                "center_x": float(center[0] / max(image_width - 1, 1)),
                "center_y": float(center[1] / max(image_height - 1, 1)),
                "width": float(width_pixels / image_width),
                "depth": float(depth_pixels / image_height),
                "angle_rad": float(np.arctan2(major[1], major[0])),
                "height_m": roof_height,
                "base_m": base_height,
                "confidence": float(np.mean(probability[rows, cols])),
                "outline": outlines.get(label_index, []),
            }
        )
    return solids


def prepare_semantic_scene(
    height_m: np.ndarray,
    building_probability: np.ndarray,
    *,
    rgb: np.ndarray | None = None,
    vegetation_probability: np.ndarray | None = None,
    terrain_prior: np.ndarray | None = None,
    threshold: float = 0.3,
    minimum_component_pixels: int = 160,
    roof_flatten: float = 0.8,
    maximum_visual_relief_m: float = 1.25,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, int]]:
    """Return continuous natural relief plus separate clean building solids."""

    full_mesh, counts = prepare_presentation_mesh(
        height_m,
        building_probability,
        rgb=rgb,
        vegetation_probability=vegetation_probability,
        terrain_prior=terrain_prior,
        threshold=threshold,
        minimum_component_pixels=minimum_component_pixels,
        roof_flatten=roof_flatten,
        maximum_visual_relief_m=maximum_visual_relief_m,
    )
    height = np.asarray(height_m, dtype=np.float32)
    probability = np.asarray(building_probability, dtype=np.float32)
    valid = np.isfinite(height) & np.isfinite(probability)
    building_mask, _, _ = resolve_semantic_masks(
        height,
        probability,
        rgb=rgb,
        vegetation_probability=vegetation_probability,
        building_threshold=threshold,
        minimum_building_pixels=minimum_component_pixels,
    )
    # Remove the warped building heightfield but retain the local ground base.
    ground = full_mesh.copy()
    if np.any(building_mask):
        nearby_ground = ground.copy()
        nearby_ground[building_mask] = np.nan
        nearest_indices = ndimage.distance_transform_edt(
            building_mask,
            return_distances=False,
            return_indices=True,
        )
        ground[building_mask] = nearby_ground[tuple(nearest_indices[:, building_mask])]
    ground = ndimage.gaussian_filter(np.nan_to_num(ground, nan=0.0), sigma=1.2)
    ground[~valid] = np.nan
    solids = extract_building_solids(
        height,
        probability,
        ground_relief_m=ground,
        threshold=threshold,
        minimum_component_pixels=minimum_component_pixels,
    )
    counts = {**counts, "building_solids": len(solids)}
    return ground.astype(np.float32), solids, counts
