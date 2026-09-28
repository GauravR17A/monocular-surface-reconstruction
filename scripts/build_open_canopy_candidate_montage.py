"""Create a contact sheet for preselected held-out Open-Canopy samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("sample_ids", nargs="+")
    args = parser.parse_args()

    rows: list[Image.Image] = []
    for sample_id in args.sample_ids:
        sample = args.root / sample_id
        metadata = json.loads((sample / "source.json").read_text(encoding="utf-8"))
        with rasterio.open(sample / "rgb.tif") as dataset:
            rgb = np.moveaxis(dataset.read([1, 2, 3]), 0, -1)
        with rasterio.open(sample / "canopy_height_m.tif") as dataset:
            height = dataset.read(1)
        with rasterio.open(sample / "vegetation_mask.tif") as dataset:
            vegetation = dataset.read(1) > 0

        rgb_image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").resize((384, 384))
        clipped = np.clip(height, 0, 35) / 35.0
        height_rgb = np.stack(
            [
                np.clip(4 * clipped - 1.5, 0, 1),
                np.clip(2 - np.abs(4 * clipped - 2), 0, 1),
                np.clip(1.5 - 4 * clipped, 0, 1),
            ],
            axis=-1,
        )
        height_image = Image.fromarray((height_rgb * 255).astype(np.uint8), mode="RGB").resize((384, 384))
        mask_rgb = np.zeros((*vegetation.shape, 3), dtype=np.uint8)
        mask_rgb[vegetation] = (34, 197, 94)
        mask_image = Image.fromarray(mask_rgb, mode="RGB").resize((384, 384))

        row = Image.new("RGB", (1152, 424), "white")
        row.paste(rgb_image, (0, 40))
        row.paste(height_image, (384, 40))
        row.paste(mask_image, (768, 40))
        draw = ImageDraw.Draw(row)
        draw.text((8, 8), f"{sample_id} | SPOT RGB", fill="black")
        draw.text((392, 8), "IGN LiDAR canopy height (0-35 m)", fill="black")
        draw.text(
            (776, 8),
            f"vegetation mask | coverage {100 * metadata['vegetation_fraction']:.1f}%",
            fill="black",
        )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    montage = Image.new("RGB", (1152, 424 * len(rows)), "white")
    for index, row in enumerate(rows):
        montage.paste(row, (0, 424 * index))
    montage.save(args.output, quality=92)


if __name__ == "__main__":
    main()
