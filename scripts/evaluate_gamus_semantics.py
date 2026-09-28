"""Compare GAMUS semantic identification for two protected checkpoints.

This evaluator is deliberately read-only.  It evaluates the protected app
checkpoint and one candidate on the same official GAMUS validation crops and
never selects or promotes a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from msr.data.gamus_dataset import (
    GAMUS_OFFICIAL_SPLIT_COUNTS,
    GamusSurfaceDataset,
)
from msr.data.surface_dataset import LANDSCAPE_CLASSES
from msr.evaluation.classification_metrics import (
    compute_multiclass_metrics,
    multiclass_metric_deltas,
)
from msr.inference.predict import load_predictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTER = PROJECT_ROOT / "outputs" / "runtime" / "showcase_checkpoint.txt"
CLASS_NAMES = tuple(
    name for name, _ in sorted(LANDSCAPE_CLASSES.items(), key=lambda item: item[1])
)


def semantic_metrics(
    confusion: np.ndarray,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, Any]:
    """Return multiclass metrics for a row-truth, column-prediction matrix."""

    return compute_multiclass_metrics(confusion, class_names)


def compare_semantic_metrics(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Summarize candidate-minus-baseline changes in reviewer-readable units."""

    return multiclass_metric_deltas(candidate, baseline)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pointer_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "contents_sha256": None, "target": None}
    contents = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "contents_sha256": hashlib.sha256(contents).hexdigest(),
        # PowerShell's historical UTF-8 output can include a BOM.  Treat it as
        # encoding metadata, not as part of the protected checkpoint path.
        "target": contents.decode("utf-8-sig").strip(),
    }


def _checkpoint_data_config(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError(f"Checkpoint has no usable data config: {checkpoint}")
    return dict(config["data"])


def _autocast(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: Path,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    label: str,
) -> dict[str, Any]:
    model, metadata = load_predictor(checkpoint, device=device)
    model.eval()
    confusion = torch.zeros(
        (len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64, device=device
    )
    try:
        for batch in tqdm(loader, desc=f"GAMUS semantics ({label})"):
            image = batch["image"].to(device, non_blocking=True)
            prior = batch["relative_prior"].to(device, non_blocking=True)
            target = batch["domain_target"].to(device, non_blocking=True)
            valid = batch["domain_valid_mask"].to(device, non_blocking=True).bool()
            with _autocast(device, precision):
                output = model(image, prior)
            logits = output.get("domain_logits")
            if logits is None or logits.ndim != 4 or logits.shape[1] != len(CLASS_NAMES):
                raise ValueError(
                    f"{label} checkpoint does not expose three-class domain_logits"
                )
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite domain logits in {label} checkpoint")
            prediction = logits.argmax(dim=1)
            encoded = target[valid] * len(CLASS_NAMES) + prediction[valid]
            confusion += torch.bincount(
                encoded, minlength=len(CLASS_NAMES) ** 2
            ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = semantic_metrics(confusion.cpu().numpy())
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_sha256"] = _sha256(checkpoint)
    result["checkpoint_epoch"] = metadata.get("epoch")
    result["model_type"] = metadata.get("model_type")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare protected-baseline and candidate semantic identification on "
            "the official GAMUS validation split without promoting either model."
        )
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        help="Defaults to the checkpoint currently named by the protected app pointer.",
    )
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--gamus-root", type=Path)
    parser.add_argument("--relative-prior-root", type=Path)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Development smoke test only; omit for the official full validation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples must be positive")

    pointer = args.pointer.expanduser().resolve()
    pointer_before = _pointer_snapshot(pointer)
    if args.baseline_checkpoint is None:
        if not pointer_before["exists"] or not pointer_before["target"]:
            raise ValueError("No baseline checkpoint supplied and app pointer is unavailable")
        baseline_checkpoint = Path(str(pointer_before["target"]))
    else:
        baseline_checkpoint = args.baseline_checkpoint
    baseline_checkpoint = baseline_checkpoint.expanduser().resolve()
    candidate_checkpoint = args.candidate_checkpoint.expanduser().resolve()
    for checkpoint in (baseline_checkpoint, candidate_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    data_config = _checkpoint_data_config(candidate_checkpoint)
    gamus_root = (
        args.gamus_root.expanduser().resolve()
        if args.gamus_root is not None
        else Path(data_config["root"]).expanduser().resolve()
    )
    prior_setting = args.relative_prior_root or data_config.get("relative_prior_root")
    if prior_setting is None:
        raise ValueError(
            "A relative-prior root is required for an app-equivalent GAMUS comparison"
        )
    relative_prior_root = Path(prior_setting).expanduser().resolve()
    patch_size = int(
        data_config.get("validation_patch_size", data_config.get("patch_size", 384))
    )
    dataset: Dataset = GamusSurfaceDataset(
        gamus_root,
        "val",
        patch_size=patch_size,
        random_crop=False,
        augment=False,
        rgb_scale=float(data_config.get("rgb_scale", 255.0)),
        height_max_m=float(data_config.get("height_max_m", 200.0)),
        radiometric_policy=str(data_config.get("validation_radiometric_policy", "raw")),
        relative_prior_root=relative_prior_root,
        require_relative_prior=True,
    )
    official_count = GAMUS_OFFICIAL_SPLIT_COUNTS["val"]
    discovered_count = len(dataset)
    if discovered_count != official_count:
        raise ValueError(
            f"Official GAMUS validation requires {official_count} tiles; found {discovered_count}"
        )
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, discovered_count)))

    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(data_config.get("num_workers", 0))
    )
    if workers < 0:
        raise ValueError("num-workers cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    precision = args.precision if device.type == "cuda" else "fp32"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        precision = "fp16"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    baseline = evaluate_checkpoint(
        baseline_checkpoint,
        loader,
        device=device,
        precision=precision,
        label="protected baseline",
    )
    candidate = evaluate_checkpoint(
        candidate_checkpoint,
        loader,
        device=device,
        precision=precision,
        label="candidate",
    )
    pointer_after = _pointer_snapshot(pointer)
    if pointer_after != pointer_before:
        raise RuntimeError(
            "The protected app pointer changed during evaluation; no report was written"
        )

    result = {
        "report_type": "GAMUS validation semantic identification comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "validation_protocol": {
            "split": "official GAMUS val",
            "official_tile_count": official_count,
            "evaluated_tile_count": len(dataset),
            "full_official_validation": len(dataset) == official_count,
            "patch_size": patch_size,
            "radiometric_policy": str(
                data_config.get("validation_radiometric_policy", "raw")
            ),
            "relative_priors_required": True,
            "gamus_root": str(gamus_root),
            "relative_prior_root": str(relative_prior_root),
            "precision": precision,
        },
        "protected_pointer": {
            "before": pointer_before,
            "after": pointer_after,
            "unchanged": True,
        },
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": compare_semantic_metrics(baseline, candidate),
    }
    serialized = json.dumps(result, indent=2)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(output)
    print(serialized)


if __name__ == "__main__":
    main()
