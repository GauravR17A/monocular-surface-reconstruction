"""Export an inactive, validated six-class candidate without wiring the app."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

from msr.inference.predict import load_predictor, predict_height
from msr.io.raster import RasterImage, read_rgb_raster
from msr.io.six_class_export import (
    CLASS_NAMES,
    SixClassExportError,
    authorize_validated_candidate,
    export_six_class_bundle,
)


def _same_transform(left: Any, right: Any) -> bool:
    return bool(np.allclose(tuple(left)[:6], tuple(right)[:6], rtol=0.0, atol=1.0e-9))


def _read_aligned_band(path: Path, source: RasterImage) -> tuple[np.ndarray, np.ndarray]:
    """Read one band only when it is already on the exact source grid."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path.expanduser().resolve()) as raster:
            expected = (int(source.profile["height"]), int(source.profile["width"]))
            if raster.shape != expected:
                raise SixClassExportError(
                    f"Aligned sidecar {path} shape {raster.shape} != {expected}"
                )
            if raster.crs != source.profile.get("crs"):
                raise SixClassExportError(f"Aligned sidecar {path} CRS differs from RGB")
            if not _same_transform(raster.transform, source.profile.get("transform")):
                raise SixClassExportError(
                    f"Aligned sidecar {path} transform differs from RGB"
                )
            values = raster.read(1)
            valid = raster.read_masks(1) > 0
    return values, valid


def _read_mask(path: Path, source: RasterImage, role: str) -> np.ndarray:
    values, raster_valid = _read_aligned_band(path, source)
    if not np.all(np.isfinite(values[raster_valid])):
        raise SixClassExportError(f"{role} contains non-finite valid pixels")
    if not np.all(np.isin(values[raster_valid], (0, 1))):
        raise SixClassExportError(f"{role} must contain only 0/1 on valid pixels")
    return raster_valid & (values == 1)


def _read_relative_prior(path: Path, source: RasterImage) -> np.ndarray:
    values, valid = _read_aligned_band(path, source)
    result = values.astype(np.float32, copy=False)
    result[~valid] = np.nan
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an offline six-class GeoTIFF bundle from a checkpoint bound "
            "to a completed validation report. This never changes the app model."
        )
    )
    parser.add_argument("image", type=Path, help="RGB image/GeoTIFF")
    parser.add_argument("checkpoint", type=Path, help="inactive six-class checkpoint")
    parser.add_argument("validation_report", type=Path, help="strict validation gate JSON")
    parser.add_argument("output_dir", type=Path, help="new output directory")
    parser.add_argument("--image-valid-mask", type=Path)
    parser.add_argument("--classification-valid-mask", type=Path)
    parser.add_argument("--height-valid-mask", type=Path)
    parser.add_argument("--relative-prior", type=Path)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--rgb-scale", type=float, default=255.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authorize_validated_candidate(args.validation_report, args.checkpoint)
    source = read_rgb_raster(args.image)

    image_valid = source.valid_mask.copy()
    image_valid_source = "RGB band validity"
    if args.image_valid_mask is not None:
        supplied = _read_mask(args.image_valid_mask, source, "image_valid_mask")
        if np.any(supplied & ~source.valid_mask):
            raise SixClassExportError(
                "Supplied image validity must not escape RGB band validity"
            )
        image_valid = supplied
        image_valid_source = str(args.image_valid_mask.expanduser().resolve())

    if args.classification_valid_mask is None:
        classification_valid = image_valid.copy()
        classification_valid_source = "copy of image validity for dense model inference"
    else:
        classification_valid = _read_mask(
            args.classification_valid_mask, source, "classification_valid_mask"
        )
        classification_valid_source = str(
            args.classification_valid_mask.expanduser().resolve()
        )

    relative_prior = (
        _read_relative_prior(args.relative_prior, source)
        if args.relative_prior is not None
        else None
    )
    model, metadata = load_predictor(args.checkpoint, device=args.device)
    if tuple(metadata.get("fine_semantic_classes") or ()) != CLASS_NAMES:
        raise SixClassExportError(
            "Authenticated checkpoint does not expose the exact six-class GAMUS head"
        )
    prediction = predict_height(
        source.rgb,
        model=model,
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        rgb_scale=args.rgb_scale,
        amp=not args.disable_amp,
        model_metadata=metadata,
        relative_prior=relative_prior,
        valid_mask=image_valid,
    )
    if prediction.fine_semantic_probability_maps is None:
        raise SixClassExportError("Candidate produced no six-class output")

    if args.height_valid_mask is None:
        height_valid = image_valid & np.isfinite(prediction.height_map)
        height_valid_source = "finite model height on image-valid pixels"
    else:
        height_valid = _read_mask(args.height_valid_mask, source, "height_valid_mask")
        height_valid_source = str(args.height_valid_mask.expanduser().resolve())

    summary = export_six_class_bundle(
        args.output_dir,
        fine_semantic_probability_maps=prediction.fine_semantic_probability_maps,
        height_map_m=prediction.height_map,
        reference_profile=source.profile,
        image_valid_mask=image_valid,
        classification_valid_mask=classification_valid,
        height_valid_mask=height_valid,
        source_image=source.source,
        checkpoint_path=args.checkpoint,
        validation_report_path=args.validation_report,
        model_metadata={
            "epoch": metadata.get("epoch"),
            "model_type": metadata.get("model_type"),
            "fine_semantic_head_type": metadata.get("fine_semantic_head_type"),
        },
        mask_provenance={
            "image_valid_mask": image_valid_source,
            "classification_valid_mask": classification_valid_source,
            "height_valid_mask": height_valid_source,
        },
    )
    print(f"Offline export complete: {args.output_dir.expanduser().resolve()}")
    print(
        f"Classification-valid pixels: "
        f"{summary['validity']['classification_valid_pixels']:,}"
    )
    print("Application model/pointer changed: no")


if __name__ == "__main__":
    main()
