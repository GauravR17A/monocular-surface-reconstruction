"""Evaluate the fixed DFC19 candidate set and rank transparent showcase options."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = PROJECT_ROOT / "data" / "demo_sources" / "DFC19_candidate_selection"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "presentation"
    / "monday_demo"
    / "reviewer_lidar_upload_demo"
    / "showcase_selection"
)
API_URL = "http://127.0.0.1:8000/api/predict"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    rgb_paths = sorted((CANDIDATE_ROOT / "rgb").glob("*.tif"))
    for index, rgb_path in enumerate(rgb_paths, start=1):
        reference_path = CANDIDATE_ROOT / "depth" / rgb_path.name
        if not reference_path.exists():
            continue
        print(f"[{index:02d}/{len(rgb_paths):02d}] {rgb_path.stem}", flush=True)
        with rgb_path.open("rb") as image, reference_path.open("rb") as reference:
            response = requests.post(
                API_URL,
                files={
                    "image": (rgb_path.name, image, "image/tiff"),
                    "reference": (reference_path.name, reference, "image/tiff"),
                },
                data={"reference_kind": "ndsm"},
                timeout=900,
            )
        response.raise_for_status()
        payload = response.json()
        metrics = payload["validation"]
        row: dict[str, object] = {
            "sample_id": rgb_path.stem,
            "job_id": payload["job_id"],
            "rmse_m": metrics["rmse_m"],
            "mae_m": metrics["mae_m"],
            "median_ae_m": metrics["median_ae_m"],
            "p90_ae_m": metrics["p90_ae_m"],
            "bias_m": metrics["bias_m"],
            "correlation": metrics["correlation"],
            "r2": metrics["r2"],
            "pixel_count": metrics["pixel_count"],
        }
        rows.append(row)
        print(
            f"    RMSE {row['rmse_m']:.3f} | MAE {row['mae_m']:.3f} | "
            f"corr {row['correlation']:.3f} | R2 {row['r2']:.3f}",
            flush=True,
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["r2"]),
            -float(row["correlation"]),
            float(row["rmse_m"]),
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank_by_r2_then_correlation_then_rmse"] = rank

    report = {
        "selection_disclosure": (
            "All 15 pre-downloaded DFC19 candidates were evaluated. The chosen visual is a "
            "curated held-out showcase selected after evaluation, not an unbiased aggregate test result."
        ),
        "candidate_count": len(ranked),
        "ranking_rule": "highest R2, then correlation, then lowest RMSE",
        "candidates": ranked,
    }
    (OUTPUT_ROOT / "urban_dfc19_candidate_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (OUTPUT_ROOT / "urban_dfc19_candidate_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    print(json.dumps(ranked[:5], indent=2), flush=True)


if __name__ == "__main__":
    main()
