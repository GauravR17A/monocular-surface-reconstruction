"""Precompute aligned Depth Anything priors for native GAMUS HDF5 tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
from tqdm import tqdm

from msr.data.gamus_dataset import load_gamus_split
from msr.inference.relative_depth import (
    DEFAULT_RELATIVE_DEPTH_MODEL,
    DepthAnythingV2Predictor,
)


def _valid_prior(path: Path, expected_shape: tuple[int, int]) -> bool:
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return "image" in handle and tuple(handle["image"].shape) == expected_shape
    except OSError:
        return False


def _write_prior(
    destination: Path,
    values: np.ndarray,
    *,
    model_id: str,
    source_path: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        dataset = handle.create_dataset(
            "image",
            data=np.asarray(values, dtype=np.float16),
            compression="lzf",
            shuffle=True,
        )
        dataset.attrs["units"] = "relative_0_1"
        dataset.attrs["model"] = model_id
        dataset.attrs["source_rgb"] = str(source_path)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create resumable, grid-aligned Depth Anything priors for GAMUS."
    )
    parser.add_argument("--gamus-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val")
    )
    parser.add_argument("--model", default=DEFAULT_RELATIVE_DEPTH_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    predictor = DepthAnythingV2Predictor(model_id=args.model, device=args.device)
    counts: dict[str, int] = {}
    started = time.time()
    for split in args.splits:
        records = load_gamus_split(args.gamus_root, split, check_files=True)
        if args.max_samples is not None:
            records = records[: args.max_samples]
        completed = 0
        progress = tqdm(records, desc=f"GAMUS {split} priors", unit="tile")
        for record in progress:
            destination = args.output_root / split / f"{record.sample_id}_REL.h5"
            with h5py.File(record.image_path, "r") as handle:
                image_dataset = handle["image"]
                expected_shape = tuple(image_dataset.shape[:2])
                if _valid_prior(destination, expected_shape):
                    completed += 1
                    continue
                image = np.asarray(image_dataset[..., :3], dtype=np.uint8).transpose(2, 0, 1)
            relative = predictor.predict(image).relative_surface
            if tuple(relative.shape) != expected_shape:
                raise ValueError(
                    f"Prior shape mismatch for {record.sample_id}: "
                    f"{relative.shape} != {expected_shape}"
                )
            _write_prior(
                destination,
                np.clip(relative, 0.0, 1.0),
                model_id=args.model,
                source_path=record.image_path,
            )
            completed += 1
            progress.set_postfix(completed=completed)
        counts[split] = completed

    provenance = {
        "dataset": "GAMUS",
        "product": "Depth Anything V2 relative-depth priors",
        "model": args.model,
        "splits": counts,
        "elapsed_seconds": time.time() - started,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = args.output_root / "msr_prior_provenance.json.tmp"
    temporary.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    temporary.replace(args.output_root / "msr_prior_provenance.json")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
