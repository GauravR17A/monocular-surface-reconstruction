"""Render top DFC19 validation candidates for visual, metric-aware selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "presentation"
    / "monday_demo"
    / "reviewer_lidar_upload_demo"
    / "showcase_selection"
    / "urban_dfc19_candidate_metrics.json"
)
OUTPUT_PATH = REPORT_PATH.with_name("TOP_URBAN_SHOWCASE_CANDIDATES.jpg")


def read_float(path: Path) -> np.ndarray:
    with rasterio.open(path) as dataset:
        return dataset.read(1).astype(np.float32)


def height_image(values: np.ndarray, low: float, high: float) -> Image.Image:
    scaled = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    colors = np.stack(
        [
            np.clip(4 * scaled - 1.5, 0, 1),
            np.clip(2 - np.abs(4 * scaled - 2), 0, 1),
            np.clip(1.5 - 4 * scaled, 0, 1),
        ],
        axis=-1,
    )
    return Image.fromarray((colors * 255).astype(np.uint8), mode="RGB")


def error_image(values: np.ndarray) -> Image.Image:
    finite = values[np.isfinite(values)]
    limit = max(float(np.percentile(np.abs(finite), 95)), 1.0)
    scaled = np.clip(values / limit, -1, 1)
    red = np.where(scaled > 0, 255, 255 * (1 + scaled))
    blue = np.where(scaled < 0, 255, 255 * (1 - scaled))
    green = 255 * (1 - np.abs(scaled))
    return Image.fromarray(np.stack([red, green, blue], axis=-1).astype(np.uint8), mode="RGB")


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    rows: list[Image.Image] = []
    for candidate in report["candidates"][:5]:
        job = PROJECT_ROOT / "outputs" / "web_jobs" / candidate["job_id"]
        prediction = read_float(job / "height_above_ground_m.tif")
        reference = read_float(job / "validation_reference_aligned_m.tif")
        error = read_float(job / "validation_signed_error_m.tif")
        finite = np.concatenate([prediction[np.isfinite(prediction)], reference[np.isfinite(reference)]])
        low, high = np.percentile(finite, (1, 99))
        cards = [
            Image.open(job / "texture.jpg").convert("RGB"),
            height_image(prediction, float(low), float(high)),
            height_image(reference, float(low), float(high)),
            error_image(error),
        ]
        card_size = 320
        row = Image.new("RGB", (card_size * 4, card_size + 60), "#07100e")
        for index, card in enumerate(cards):
            row.paste(card.resize((card_size, card_size), Image.Resampling.LANCZOS), (index * card_size, 60))
        draw = ImageDraw.Draw(row)
        draw.text((10, 8), f"#{candidate['rank_by_r2_then_correlation_then_rmse']} {candidate['sample_id']}", fill="#45f3c4")
        draw.text(
            (10, 29),
            f"RMSE {candidate['rmse_m']:.2f} m | MAE {candidate['mae_m']:.2f} m | corr {candidate['correlation']:.3f} | R2 {candidate['r2']:.3f}",
            fill="white",
        )
        for index, label in enumerate(("RGB", "MODEL", "LiDAR", "ERROR")):
            draw.text((index * card_size + card_size - 58, 29), label, fill="#bbf451")
        rows.append(row)

    sheet = Image.new("RGB", (1280, 380 * len(rows)), "#07100e")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * 380))
    sheet.save(OUTPUT_PATH, quality=92)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
