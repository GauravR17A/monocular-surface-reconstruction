"""Precompute licensed Depth Anything priors for multi-domain training records."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

from msr.data.surface_dataset import load_surface_manifest
from msr.inference.relative_depth import (
    DEFAULT_RELATIVE_DEPTH_MODEL,
    DepthAnythingV2Predictor,
)
from msr.io.raster import read_rgb_raster, write_float_raster


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create grid-aligned relative-depth priors once, rather than running "
            "the foundation model inside every training epoch."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--model", default=DEFAULT_RELATIVE_DEPTH_MODEL)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = load_surface_manifest(args.manifest)
    predictor = DepthAnythingV2Predictor(model_id=args.model, device=args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    prior_paths: dict[str, Path] = {}
    for index, record in enumerate(records, start=1):
        destination = (
            args.output_root
            / _safe_name(record.region)
            / f"{_safe_name(record.sample_id)}_relative_prior.tif"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            source = read_rgb_raster(record.rgb_path)
            relative = predictor.predict(source.rgb, valid_mask=source.valid_mask)
            write_float_raster(
                destination,
                relative.relative_surface,
                reference_profile=source.profile,
                tags={
                    "MSR_PRODUCT": "training_relative_depth_prior",
                    "MSR_UNITS": "relative_0_1",
                    "MSR_MODEL": args.model,
                },
            )
        prior_paths[record.sample_id] = destination.resolve()
        print(f"[{index}/{len(records)}] {record.sample_id} -> {destination}")

    with args.manifest.resolve().open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        if "relative_prior_path" not in fieldnames:
            fieldnames.append("relative_prior_path")
        rows = list(reader)
    for row in rows:
        row["relative_prior_path"] = str(prior_paths[row["sample_id"].strip()])

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_manifest.with_suffix(args.output_manifest.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output_manifest)
    print(f"Saved enriched manifest to {args.output_manifest.resolve()}")


if __name__ == "__main__":
    main()
