"""Roof3D plane/section supervision, kept separate from metric height labels.

Released annotations may wrap uncompressed COCO RLE dictionaries in a list.
Unknown background and overlapping instance labels are not valid negatives.
"""
from __future__ import annotations

import numpy as np


def decode_roof_mask(segmentation, shape: tuple[int, int]) -> np.ndarray:
    """Decode strict uncompressed COCO RLE, including lists of RLE parts."""
    height, width = shape
    if height < 1 or width < 1:
        raise ValueError("Invalid raster shape")
    parts = [segmentation] if isinstance(segmentation, dict) else segmentation
    if not isinstance(parts, list) or not parts:
        raise ValueError("Expected nonempty RLE parts")
    union = np.zeros(shape, dtype=bool)
    for part in parts:
        if not isinstance(part, dict) or part.get("size") != [height, width]:
            raise ValueError("RLE dimensions differ from RGB")
        counts = part.get("counts")
        if not isinstance(counts, list) or any(type(n) is not int or n < 0 for n in counts):
            raise ValueError("Expected nonnegative integer uncompressed RLE counts")
        if sum(counts) != height * width:
            raise ValueError("RLE length differs from raster")
        flat = np.zeros(height * width, dtype=bool)
        offset = 0
        for i, count in enumerate(counts):
            if i % 2:
                flat[offset:offset+count] = True
            offset += count
        union |= flat.reshape(shape, order="F")
    return union


def roof_instance_targets(annotations, image_valid: np.ndarray) -> dict[str, np.ndarray]:
    """Positive-only instance/boundary targets. Never manufactures height labels."""
    valid_image = np.asarray(image_valid, dtype=bool)
    if valid_image.ndim != 2:
        raise ValueError("Expected a 2D image validity mask")
    instances = np.zeros(valid_image.shape, dtype=np.int32)
    overlap = np.zeros(valid_image.shape, dtype=bool)
    ids = set()
    for sequence, annotation in enumerate(annotations, 1):
        identifier = annotation["id"]
        if identifier in ids:
            raise ValueError("Duplicate annotation ID")
        ids.add(identifier)
        if annotation.get("iscrowd", 0):
            raise ValueError("Crowd labels need an explicit ignore policy")
        mask = decode_roof_mask(annotation["segmentation"], valid_image.shape)
        overlap |= mask & (instances != 0)
        instances[mask & (instances == 0)] = sequence
    valid = valid_image & (instances > 0) & ~overlap
    boundaries = np.zeros(instances.shape, dtype=bool)
    boundaries[:-1] |= instances[:-1] != instances[1:]
    boundaries[1:] |= instances[1:] != instances[:-1]
    boundaries[:, :-1] |= instances[:, :-1] != instances[:, 1:]
    boundaries[:, 1:] |= instances[:, 1:] != instances[:, :-1]
    # A cropped image border is not a labelled roof edge. Exclude its pixels
    # and labels adjacent to image-invalid/overlapping regions.
    unsafe = ~valid_image | overlap
    margin = unsafe.copy()
    margin[1:] |= unsafe[:-1]
    margin[:-1] |= unsafe[1:]
    margin[:, 1:] |= unsafe[:, :-1]
    margin[:, :-1] |= unsafe[:, 1:]
    margin[[0, -1], :] = True
    margin[:, [0, -1]] = True
    boundary_valid = valid & ~margin
    instances[~valid] = 0
    return {
        "instance_id": instances,
        "image_valid": valid_image,
        "classification_valid": valid,
        "boundary": boundaries & boundary_valid,
        "boundary_valid": boundary_valid,
        "ambiguous_overlap": overlap,
        # No datum/unit contract has yet been certified for height regression.
        "height_valid": np.zeros_like(valid),
    }
