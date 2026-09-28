"""Isolated research-only evaluation adapter for the FusedSeg-HE checkpoint.

This adapter does not copy FusedHE implementation code into Monocular Surface Reconstruction. It
imports a separately cloned repository, whose non-commercial license applies to
that model and its predictions. It is a benchmark candidate, not a distributable
Monocular Surface Reconstruction dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import rasterio
import torch

from msr.inference.tiling import blend_weight, tile_starts


def model_args() -> SimpleNamespace:
    return SimpleNamespace(
        option="full",
        pre_trained=False,
        enc_opt="fused",
        num_classes=1,
        in_channel_ffb=[2048, 1024, 512, 256],
        out_channel_ffb=[512, 320, 128, 64],
        in_channel_decoderb=[1024, 640, 256],
        out_channel_decoderb=[640, 256, 128],
        out_decoder=64,
    )


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    *,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 128,
    max_height_m: float = 185.0,
    precision: str = "fp32",
) -> np.ndarray:
    image = np.moveaxis(image_bgr.astype(np.float32) / 255.0, -1, 0)
    height, width = image.shape[-2:]
    output = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    for y in tile_starts(height, tile_size, overlap):
        for x in tile_starts(width, tile_size, overlap):
            tile = image[:, y : y + tile_size, x : x + tile_size]
            tile_height, tile_width = tile.shape[-2:]
            if tile_height != tile_size or tile_width != tile_size:
                tile = np.pad(
                    tile,
                    (
                        (0, 0),
                        (0, tile_size - tile_height),
                        (0, tile_size - tile_width),
                    ),
                )
            tensor = torch.from_numpy(np.ascontiguousarray(tile[None])).to(device)
            autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=precision != "fp32",
            ):
                normalized = model(tensor)["pred_d"]
            prediction = (
                normalized[0, 0, :tile_height, :tile_width].float().cpu().numpy()
                * max_height_m
            )
            if not np.all(np.isfinite(prediction)):
                raise FloatingPointError(
                    f"Non-finite FusedSeg-HE prediction at tile ({x}, {y}) using {precision}"
                )
            weight = blend_weight(tile_height, tile_width)
            output[y : y + tile_height, x : x + tile_width] += prediction * weight
            weights[y : y + tile_height, x : x + tile_width] += weight
    return (output / np.maximum(weights, 1e-12)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--max-height-m", type=float, default=185.0)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve()
    if not (repo_dir / "models" / "model.py").is_file():
        raise FileNotFoundError(f"FusedHE repository not found at {repo_dir}")
    sys.path.insert(0, str(repo_dir))
    from models.model import FHE  # type: ignore[import-not-found]  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("FusedSeg-HE benchmark requires CUDA")
    device = torch.device("cuda")
    model = FHE(model_args())
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {args.image}")
    prediction = predict(
        model,
        image_bgr,
        device=device,
        overlap=args.overlap,
        max_height_m=args.max_height_m,
        precision=args.precision,
    )

    with rasterio.open(args.image) as source:
        profile = source.profile.copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.pop("photometric", None)
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=float("nan"),
        compress="deflate",
        predictor=3,
    )
    with rasterio.open(args.output, "w", **profile) as destination:
        destination.write(prediction, 1)
    metadata = {
        "model": "FusedSeg-HE",
        "license_scope": "research/evaluation only; non-commercial",
        "checkpoint": str(args.checkpoint.resolve()),
        "source_repository": str(repo_dir),
        "max_height_m": args.max_height_m,
        "preprocessing": "OpenCV BGR, float32 divided by 255",
        "tile_size": 512,
        "overlap": args.overlap,
        "precision": args.precision,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
