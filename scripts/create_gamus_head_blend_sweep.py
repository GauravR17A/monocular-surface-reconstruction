"""Create evaluation-only head interpolations without touching source checkpoints.

The protected checkpoint is the template.  Only the semantic/domain and two
height-head parameter groups are interpolated toward a GAMUS checkpoint.  The
utility aborts unless every other model tensor is exactly identical in both
sources, and publishes the completed sweep with one atomic directory rename.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import torch


BLEND_PREFIXES = (
    "domain_head.",
    "building_residual_head.",
    "canopy_height_head.",
)
NON_RUNTIME_MODEL_CONFIG_KEYS = frozenset({"initial_checkpoint"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state(payload: Mapping[str, Any], label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{label} checkpoint has no non-empty model state")
    non_tensors = sorted(name for name, value in state.items() if not torch.is_tensor(value))
    if non_tensors:
        raise TypeError(f"{label} model state contains non-tensors: {non_tensors}")
    return state


def _runtime_model_config(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("model"), Mapping):
        raise ValueError(f"{label} checkpoint has no usable model config")
    return {
        key: value
        for key, value in config["model"].items()
        if key not in NON_RUNTIME_MODEL_CONFIG_KEYS
    }


def validate_blend_sources(
    baseline_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that the sources differ only in the explicitly allowed heads."""

    if baseline_payload.get("model_type") != candidate_payload.get("model_type"):
        raise ValueError("Source checkpoint model_type values do not match")
    if _runtime_model_config(baseline_payload, "baseline") != _runtime_model_config(
        candidate_payload, "candidate"
    ):
        raise ValueError("Source runtime model configurations do not match")

    baseline = _model_state(baseline_payload, "baseline")
    candidate = _model_state(candidate_payload, "candidate")
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(f"Source model keys differ; missing={missing}, extra={extra}")

    selected = sorted(
        name for name in baseline if name.startswith(BLEND_PREFIXES)
    )
    for prefix in BLEND_PREFIXES:
        if not any(name.startswith(prefix) for name in selected):
            raise ValueError(f"No model tensors match required prefix {prefix!r}")

    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    non_head_differences: list[str] = []
    changed: list[str] = []
    for name in sorted(baseline):
        left = baseline[name]
        right = candidate[name]
        if left.shape != right.shape:
            shape_mismatches.append(name)
            continue
        if left.dtype != right.dtype:
            dtype_mismatches.append(name)
            continue
        if not torch.equal(left, right):
            changed.append(name)
            if name not in selected:
                non_head_differences.append(name)
    if shape_mismatches:
        raise ValueError(f"Source tensor shapes differ: {shape_mismatches}")
    if dtype_mismatches:
        raise ValueError(f"Source tensor dtypes differ: {dtype_mismatches}")
    if non_head_differences:
        raise ValueError(
            "Source checkpoints differ outside the allowed heads: "
            f"{non_head_differences}"
        )
    non_float = [
        name
        for name in selected
        if not (torch.is_floating_point(baseline[name]) or torch.is_complex(baseline[name]))
    ]
    if non_float:
        raise TypeError(f"Selected tensors cannot be interpolated: {non_float}")

    return {
        "model_tensor_count": len(baseline),
        "blend_tensor_count": len(selected),
        "non_blend_tensor_count": len(baseline) - len(selected),
        "blend_tensor_names": selected,
        "changed_tensor_names": changed,
        "model_keys_shapes_dtypes_identical": True,
        "non_blend_tensors_identical": True,
        "runtime_model_config_identical": True,
    }


def _validate_alphas(alphas: list[float]) -> list[float]:
    if not alphas:
        raise ValueError("At least one alpha is required")
    normalized = [float(alpha) for alpha in alphas]
    if any(not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0 for alpha in normalized):
        raise ValueError("Every alpha must be finite and within [0, 1]")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Alpha values must be unique")
    return normalized


def _alpha_label(alpha: float) -> str:
    text = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def build_blended_payload(
    baseline_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    *,
    alpha: float,
    baseline_path: Path,
    candidate_path: Path,
    source_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one non-resumable checkpoint from the protected model/config."""

    alpha = _validate_alphas([alpha])[0]
    baseline = _model_state(baseline_payload, "baseline")
    candidate = _model_state(candidate_payload, "candidate")
    selected = frozenset(source_verification["blend_tensor_names"])
    blended: dict[str, torch.Tensor] = {}
    for name, baseline_tensor in baseline.items():
        if name not in selected:
            blended[name] = baseline_tensor.clone()
        elif alpha == 0.0:
            blended[name] = baseline_tensor.clone()
        elif alpha == 1.0:
            blended[name] = candidate[name].clone()
        else:
            blended[name] = torch.lerp(baseline_tensor, candidate[name], alpha)

    return {
        "model_type": baseline_payload.get("model_type"),
        "epoch": None,
        "model": blended,
        "config": deepcopy(baseline_payload["config"]),
        "checkpoint_role": "evaluation_only_derived_head_blend",
        "derivation": {
            "method": "linear_parameter_interpolation",
            "formula": "baseline + alpha * (gamus_epoch8 - baseline)",
            "alpha": alpha,
            "baseline_checkpoint": str(baseline_path),
            "candidate_checkpoint": str(candidate_path),
            "blend_prefixes": list(BLEND_PREFIXES),
            "blend_tensor_names": sorted(selected),
            "non_blend_tensors_identical_to_baseline": True,
            "evaluation_status": "unevaluated",
            "resumable_training_checkpoint": False,
        },
    }


def _verify_derived_payload(
    payload: Mapping[str, Any],
    baseline_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    *,
    alpha: float,
    selected_names: set[str],
) -> None:
    derived = _model_state(payload, "derived")
    baseline = _model_state(baseline_payload, "baseline")
    candidate = _model_state(candidate_payload, "candidate")
    if set(derived) != set(baseline):
        raise RuntimeError("Derived checkpoint model keys changed during serialization")
    for name, baseline_tensor in baseline.items():
        actual = derived[name]
        if name not in selected_names:
            expected = baseline_tensor
        elif alpha == 0.0:
            expected = baseline_tensor
        elif alpha == 1.0:
            expected = candidate[name]
        else:
            expected = torch.lerp(baseline_tensor, candidate[name], alpha)
        if not torch.equal(actual, expected):
            raise RuntimeError(f"Derived tensor verification failed for {name}")


def create_blend_sweep(
    baseline_path: Path,
    candidate_path: Path,
    output_dir: Path,
    alphas: list[float],
) -> dict[str, Any]:
    """Create and atomically publish a fully verified interpolation sweep."""

    baseline_path = baseline_path.expanduser().resolve()
    candidate_path = candidate_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    for source in (baseline_path, candidate_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    if baseline_path == candidate_path:
        raise ValueError("Baseline and candidate checkpoints must be different files")
    alphas = _validate_alphas(alphas)
    labels = [_alpha_label(alpha) for alpha in alphas]
    if len(set(labels)) != len(labels):
        raise ValueError("Alpha values collide at six-decimal filename precision")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    source_hashes = {
        "baseline": _sha256(baseline_path),
        "candidate": _sha256(candidate_path),
    }
    baseline_payload = torch.load(baseline_path, map_location="cpu", weights_only=False)
    candidate_payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
    source_verification = validate_blend_sources(baseline_payload, candidate_payload)
    selected_names = set(source_verification["blend_tensor_names"])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        outputs: list[dict[str, Any]] = []
        for alpha, label in zip(alphas, labels, strict=True):
            name = f"checkpoint_head_blend_alpha_{label}.pt"
            destination = staging / name
            payload = build_blended_payload(
                baseline_payload,
                candidate_payload,
                alpha=alpha,
                baseline_path=baseline_path,
                candidate_path=candidate_path,
                source_verification=source_verification,
            )
            torch.save(payload, destination)
            reloaded = torch.load(destination, map_location="cpu", weights_only=False)
            _verify_derived_payload(
                reloaded,
                baseline_payload,
                candidate_payload,
                alpha=alpha,
                selected_names=selected_names,
            )
            outputs.append(
                {
                    "alpha": alpha,
                    "checkpoint": name,
                    "checkpoint_sha256": _sha256(destination),
                    "evaluation_status": "unevaluated",
                    "non_blend_tensors_identical_to_baseline": True,
                }
            )
            del payload, reloaded

        if _sha256(baseline_path) != source_hashes["baseline"]:
            raise RuntimeError("Baseline checkpoint changed while creating sweep")
        if _sha256(candidate_path) != source_hashes["candidate"]:
            raise RuntimeError("Candidate checkpoint changed while creating sweep")

        manifest = {
            "report_type": "GAMUS evaluation-only head interpolation sweep",
            "baseline": {
                "checkpoint": str(baseline_path),
                "checkpoint_sha256": source_hashes["baseline"],
                "epoch": baseline_payload.get("epoch"),
            },
            "candidate": {
                "checkpoint": str(candidate_path),
                "checkpoint_sha256": source_hashes["candidate"],
                "epoch": candidate_payload.get("epoch"),
            },
            "source_verification": source_verification,
            "outputs": outputs,
            "showcase_pointer_action": "not_read_or_modified",
        }
        temporary_manifest = staging / "manifest.json.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary_manifest.replace(staging / "manifest.json")
        staging.replace(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create verified, evaluation-only checkpoint blends for the three "
            "GAMUS-trained heads. This utility never reads or changes the app pointer."
        )
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
        help="Interpolation weights in [0, 1]; default: 0.25 0.50 0.75",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = create_blend_sweep(
        args.baseline, args.candidate, args.output_dir, args.alphas
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
