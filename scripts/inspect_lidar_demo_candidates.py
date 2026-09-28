"""Inspect matched RGB, height, and optional semantic demo candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--montage", type=Path)
    args = parser.parse_args()

    montage_rows: list[Image.Image] = []

    for depth_path in sorted((args.root / "depth").glob("*.tif")):
        with rasterio.open(depth_path) as dataset:
            height = dataset.read(1)
            profile = {
                "shape": [dataset.height, dataset.width],
                "dtype": dataset.dtypes[0],
                "crs": str(dataset.crs),
                "transform": tuple(dataset.transform),
            }

        semantic_path = args.root / "seg" / f"{depth_path.stem}.png"
        semantic_summary: list[tuple[object, float]] = []
        if semantic_path.exists():
            semantic = np.asarray(Image.open(semantic_path))
            values = semantic.reshape(-1, semantic.shape[-1]) if semantic.ndim == 3 else semantic
            unique, counts = np.unique(values, axis=0, return_counts=True)
            semantic_summary = [
                (value.tolist() if hasattr(value, "tolist") else int(value), round(float(count / counts.sum() * 100), 1))
                for value, count in zip(unique, counts, strict=True)
            ]

        print(
            depth_path.stem,
            profile,
            {
                "min_m": float(np.nanmin(height)),
                "max_m": float(np.nanmax(height)),
                "mean_m": float(np.nanmean(height)),
                "above_2m_pct": round(float(np.mean(height > 2) * 100), 1),
                "above_5m_pct": round(float(np.mean(height > 5) * 100), 1),
            },
            semantic_summary,
        )

        if args.montage:
            with rasterio.open(args.root / "rgb" / f"{depth_path.stem}.tif") as dataset:
                rgb = np.moveaxis(dataset.read([1, 2, 3]), 0, -1)
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb_image = Image.fromarray(rgb, mode="RGB").resize((256, 256))

            clipped = np.clip(height, 0, 30) / 30.0
            height_rgb = np.stack(
                [
                    np.clip(4 * clipped - 1.5, 0, 1),
                    np.clip(2 - np.abs(4 * clipped - 2), 0, 1),
                    np.clip(1.5 - 4 * clipped, 0, 1),
                ],
                axis=-1,
            )
            height_image = Image.fromarray((height_rgb * 255).astype(np.uint8), mode="RGB").resize((256, 256))
            semantic_image = Image.open(semantic_path).convert("RGB").resize((256, 256))

            row = Image.new("RGB", (768, 286), "white")
            row.paste(rgb_image, (0, 30))
            row.paste(height_image, (256, 30))
            row.paste(semantic_image, (512, 30))
            draw = ImageDraw.Draw(row)
            draw.text((8, 8), f"{depth_path.stem} | RGB", fill="black")
            draw.text((264, 8), "LiDAR AGL (0-30 m)", fill="black")
            draw.text((520, 8), "semantic reference", fill="black")
            montage_rows.append(row)

    if args.montage and montage_rows:
        args.montage.parent.mkdir(parents=True, exist_ok=True)
        montage = Image.new("RGB", (768, 286 * len(montage_rows)), "white")
        for index, row in enumerate(montage_rows):
            montage.paste(row, (0, 286 * index))
        montage.save(args.montage, quality=90)


if __name__ == "__main__":
    main()
