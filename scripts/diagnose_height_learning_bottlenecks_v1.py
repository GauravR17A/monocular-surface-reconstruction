"""Training-only causal audit of Monocular Surface Reconstruction's metric-height bottlenecks.

This diagnostic performs no optimizer step and never constructs validation or
test data.  It compares the protected checkpoint with the completed height-safe
v2 control, inspects a few mechanically selected GAMUS *training* crops, and
measures how strongly final-height losses reach the protected height, building,
canopy, routing, and adapter tensors.  The report is descriptive evidence for
designing the next candidate; it is not an accuracy evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from msr.data.gamus_dataset import GamusSurfaceDataset
from msr.data.surface_dataset import LANDSCAPE_CLASSES
from msr.inference.predict import load_predictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "20260915T105854Z_multidomain_surface_gamus_height_safe_v2"
    / "config.yaml"
)
DEFAULT_PROTECTED = (
    PROJECT_ROOT
    / "experiments"
    / "20260829T203146Z_multidomain_surface_v2_guarded_vegetation"
    / "checkpoint_best_guarded_vegetation.pt"
)
DEFAULT_CONTROL = (
    PROJECT_ROOT
    / "experiments"
    / "20260915T105854Z_multidomain_surface_gamus_height_safe_v2"
    / "checkpoint_epoch_004.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "height_learning_bottlenecks_v1"
    / "report.json"
)
DIAGNOSTIC_PREFIXES = (
    "base_model.height_head.",
    "adapter.",
    "domain_head.",
    "building_residual_head.",
    "canopy_height_head.",
    "refinement_strength_head.",
)
CONTROL_TRAINABLE_PREFIXES = (
    "base_model.height_head.",
    "canopy_height_head.",
)


def file_sha256(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parameter_count(
    model: torch.nn.Module, prefixes: Iterable[str] | None = None
) -> int:
    selected = tuple(prefixes or ())
    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not selected or name.startswith(selected)
    )


def tensor_stats(value: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float | int]:
    detached = value.detach().float()
    if mask is not None:
        selected_mask = mask.bool()
        while selected_mask.ndim < detached.ndim:
            selected_mask = selected_mask.unsqueeze(1)
        selected_mask = selected_mask.expand_as(detached)
        detached = detached[selected_mask]
    else:
        detached = detached.reshape(-1)
    if detached.numel() == 0:
        return {"count": 0}
    return {
        "count": int(detached.numel()),
        "mean": float(detached.mean()),
        "std": float(detached.std(unbiased=False)),
        "minimum": float(detached.min()),
        "maximum": float(detached.max()),
        "mean_abs": float(detached.abs().mean()),
    }


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 2.0,
) -> torch.Tensor | None:
    valid = mask.bool()
    if not torch.any(valid):
        return None
    return F.smooth_l1_loss(prediction[valid], target[valid], beta=beta)


def gradient_stats(
    loss: torch.Tensor | None,
    tensors: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor | None],
) -> dict[str, Any]:
    if loss is None:
        return {"available": False}
    names = list(tensors)
    gradients = torch.autograd.grad(
        loss,
        [tensors[name] for name in names],
        retain_graph=True,
        allow_unused=True,
    )
    result: dict[str, Any] = {"available": True, "loss": float(loss.detach())}
    for name, gradient in zip(names, gradients):
        result[name] = (
            {"count": 0, "disconnected": True}
            if gradient is None
            else tensor_stats(gradient, masks.get(name))
        )
    return result


def ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        left = float(numerator["mean_abs"])
        right = float(denominator["mean_abs"])
    except (KeyError, TypeError, ValueError):
        return None
    return left / right if right > 0 else None


def checkpoint_change_audit(
    protected_path: Path, control_path: Path
) -> dict[str, Any]:
    protected = torch.load(protected_path, map_location="cpu", weights_only=False)
    control = torch.load(control_path, map_location="cpu", weights_only=False)
    protected_state = protected["model"]
    control_state = control["model"]
    if set(protected_state) != set(control_state):
        raise ValueError("Control checkpoint tensor roster differs from protected")
    changed: list[dict[str, Any]] = []
    for name in protected_state:
        left = protected_state[name]
        right = control_state[name]
        if not torch.equal(left, right):
            delta = (right.float() - left.float()).reshape(-1)
            changed.append(
                {
                    "name": name,
                    "parameters": int(delta.numel()),
                    "delta_l2": float(torch.linalg.vector_norm(delta)),
                    "delta_max_abs": float(delta.abs().max()),
                }
            )
    unexpected = [
        item["name"]
        for item in changed
        if not item["name"].startswith(CONTROL_TRAINABLE_PREFIXES)
    ]
    return {
        "protected_epoch": int(protected.get("epoch", -1)),
        "control_epoch": int(control.get("epoch", -1)),
        "tensor_count": len(protected_state),
        "changed_tensor_count": len(changed),
        "changed_parameter_count": sum(item["parameters"] for item in changed),
        "changed_tensors": changed,
        "unexpected_changed_tensors": unexpected,
        "only_predeclared_control_tensors_changed": not unexpected,
    }


def make_training_dataset(config: Mapping[str, Any]) -> GamusSurfaceDataset:
    data = dict(config["data"])
    # Deliberately instantiate only the training split. Centre crops remove RNG
    # and augmentation as confounders while retaining the real train records,
    # normalisation, masks, raw-radiometry policy, and cached DAV2 priors.
    return GamusSurfaceDataset(
        Path(data["root"]).expanduser().resolve(),
        "train",
        patch_size=int(data["patch_size"]),
        random_crop=False,
        augment=False,
        samples_per_epoch=None,
        rgb_scale=float(data.get("rgb_scale", 255.0)),
        height_max_m=float(data.get("height_max_m", 200.0)),
        radiometric_policy=str(data.get("train_radiometric_policy", "raw")),
        relative_prior_root=data.get("relative_prior_root"),
        require_relative_prior=bool(data.get("require_relative_priors", False)),
        approved_index_path=data.get("approved_index_path"),
    )


def inspect_sample(
    model: torch.nn.Module,
    sample: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    image = sample["image"].unsqueeze(0).to(device)
    prior = sample["relative_prior"].unsqueeze(0).to(device)
    target = sample["height"].unsqueeze(0).to(device)
    regression = sample["regression_mask"].unsqueeze(0).to(device).bool()
    domain = sample["domain_target"].unsqueeze(0).to(device)
    masks = {
        "all": regression,
        "ground": regression
        & (domain[:, None] == LANDSCAPE_CLASSES["ground"]),
        "building": regression
        & (domain[:, None] == LANDSCAPE_CLASSES["building"]),
        "vegetation": regression
        & (domain[:, None] == LANDSCAPE_CLASSES["vegetation"]),
    }
    output = model(image, prior)
    watched = {
        "final_height": output["height"],
        "base_height": output["base_height"],
        "building_height": output["building_height"],
        "canopy_height": output["canopy_height"],
        "refinement_logits": output["refinement_logits"],
        "adapter_features": output["adapter_features"],
        "domain_logits": output["domain_logits"],
    }

    final_all = gradient_stats(
        masked_huber(output["height"], target, masks["all"]),
        watched,
        {name: masks["all"] for name in watched},
    )
    final_building = gradient_stats(
        masked_huber(output["height"], target, masks["building"]),
        watched,
        {name: masks["building"] for name in watched},
    )
    final_vegetation = gradient_stats(
        masked_huber(output["height"], target, masks["vegetation"]),
        watched,
        {name: masks["vegetation"] for name in watched},
    )
    direct_building = gradient_stats(
        masked_huber(output["building_height"], target, masks["building"]),
        {
            "building_height": output["building_height"],
            "adapter_features": output["adapter_features"],
        },
        {
            "building_height": masks["building"],
            "adapter_features": masks["building"],
        },
    )
    direct_canopy = gradient_stats(
        masked_huber(output["canopy_height"], target, masks["vegetation"]),
        {
            "canopy_height": output["canopy_height"],
            "adapter_features": output["adapter_features"],
        },
        {
            "canopy_height": masks["vegetation"],
            "adapter_features": masks["vegetation"],
        },
    )

    vegetation_mask = masks["vegetation"]
    vegetation_probability = output["domain_probabilities"][:, 2:3]
    hard_gate = output["vegetation_expert_fusion_gate"] > 0.5
    true_vegetation_count = int(torch.count_nonzero(vegetation_mask))
    hard_gate_on_true_vegetation = int(
        torch.count_nonzero(hard_gate & vegetation_mask)
    )
    final_minus_base = (output["height"] - output["base_height"]).abs()
    refinement = output["refinement_strength"]
    effective_refinement = output["effective_refinement_strength"]

    return {
        "sample_id": str(sample["sample_id"]),
        "region": str(sample["region"]),
        "pixel_counts": {
            name: int(torch.count_nonzero(mask)) for name, mask in masks.items()
        },
        "activations": {
            "adapter_features": tensor_stats(output["adapter_features"]),
            "relative_prior": tensor_stats(prior, sample["image_valid_mask"].unsqueeze(0).to(device)),
            "base_height": tensor_stats(output["base_height"], regression),
            "building_height": tensor_stats(output["building_height"], masks["building"]),
            "canopy_height": tensor_stats(output["canopy_height"], vegetation_mask),
            "final_height": tensor_stats(output["height"], regression),
            "vegetation_probability_on_true_vegetation": tensor_stats(
                vegetation_probability, vegetation_mask
            ),
            "refinement_strength": tensor_stats(refinement, regression),
            "effective_refinement_strength": tensor_stats(
                effective_refinement, regression
            ),
            "absolute_final_minus_base_ground": tensor_stats(
                final_minus_base, masks["ground"]
            ),
            "absolute_final_minus_base_building": tensor_stats(
                final_minus_base, masks["building"]
            ),
            "absolute_final_minus_base_vegetation": tensor_stats(
                final_minus_base, vegetation_mask
            ),
        },
        "fusion": {
            "vegetation_expert_threshold": float(
                model.vegetation_expert_fusion_threshold
            ),
            "vegetation_expert_strength": float(
                model.vegetation_expert_fusion_strength
            ),
            "true_vegetation_pixels": true_vegetation_count,
            "hard_expert_gate_on_true_vegetation_pixels": hard_gate_on_true_vegetation,
            "hard_expert_gate_recall_on_true_vegetation": (
                hard_gate_on_true_vegetation / true_vegetation_count
                if true_vegetation_count
                else None
            ),
        },
        "gradients": {
            "final_all": final_all,
            "final_building": final_building,
            "final_vegetation": final_vegetation,
            "direct_building_expert": direct_building,
            "direct_canopy_expert": direct_canopy,
            "final_building_to_expert_gradient_ratio": ratio(
                final_building.get("building_height"),
                final_building.get("final_height"),
            ),
            "final_vegetation_to_canopy_gradient_ratio": ratio(
                final_vegetation.get("canopy_height"),
                final_vegetation.get("final_height"),
            ),
        },
    }


def numeric_mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    protected_path = args.protected_checkpoint.resolve()
    control_path = args.control_checkpoint.resolve()
    output_path = args.output.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["contracts"].get("official_test_used") is not False:
        raise ValueError("Diagnostic config does not forbid official test use")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostic requested but unavailable")

    model, metadata = load_predictor(protected_path, device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(DIAGNOSTIC_PREFIXES):
            parameter.requires_grad_(True)

    dataset = make_training_dataset(config)
    if dataset.split != "train":
        raise AssertionError("Diagnostic constructed a non-training split")
    sample_indices = sorted({0, len(dataset) // 3, 2 * len(dataset) // 3, len(dataset) - 1})
    sample_reports = [inspect_sample(model, dataset[index], device) for index in sample_indices]

    total_parameters = parameter_count(model)
    control_parameters = parameter_count(model, CONTROL_TRAINABLE_PREFIXES)
    trainable_diagnostic_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    gate_recalls = [
        sample["fusion"]["hard_expert_gate_recall_on_true_vegetation"]
        for sample in sample_reports
    ]
    canopy_ratios = [
        sample["gradients"]["final_vegetation_to_canopy_gradient_ratio"]
        for sample in sample_reports
    ]
    building_ratios = [
        sample["gradients"]["final_building_to_expert_gradient_ratio"]
        for sample in sample_reports
    ]
    effective_refinement = [
        sample["activations"]["effective_refinement_strength"].get("mean")
        for sample in sample_reports
    ]

    report = {
        "schema": "msr.height_learning_bottleneck_diagnostic.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_training_only_no_updates",
        "scope": {
            "dataset_split_constructed": "train",
            "sample_selection": "four mechanically spaced centre crops",
            "sample_indices": sample_indices,
            "validation_used": False,
            "test_used": False,
            "external_holdout_used": False,
            "optimizer_steps": 0,
            "checkpoint_or_pointer_writes": 0,
        },
        "artifacts": {
            "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
            "protected_checkpoint": {
                "path": str(protected_path),
                "sha256": file_sha256(protected_path),
            },
            "completed_control_checkpoint": {
                "path": str(control_path),
                "sha256": file_sha256(control_path),
            },
        },
        "model_metadata": metadata,
        "capacity": {
            "total_parameters": total_parameters,
            "height_safe_v2_trainable_parameters": control_parameters,
            "height_safe_v2_fraction_of_model": control_parameters / total_parameters,
            "diagnostic_gradient_enabled_parameters": trainable_diagnostic_parameters,
            "module_parameters": {
                name: parameter_count(model, (prefix,))
                for name, prefix in {
                    "base_height_head": "base_model.height_head.",
                    "base_decoder": "base_model.decoder_stages.",
                    "base_deepest": "base_model.deepest.",
                    "surface_adapter": "adapter.",
                    "domain_head": "domain_head.",
                    "building_residual_head": "building_residual_head.",
                    "canopy_height_head": "canopy_height_head.",
                    "refinement_strength_head": "refinement_strength_head.",
                }.items()
            },
        },
        "architecture_path": {
            "surface_adapter_inputs": [
                "normalised RGB (3 channels)",
                "fixed cached DAV2 scalar prior (1 channel)",
                "log-normalised protected base height (1 channel)",
                "protected building probability (1 channel)",
            ],
            "surface_adapter_direct_encoder_feature_channels": 0,
            "surface_adapter_direct_decoder_feature_channels": 0,
            "surface_adapter_nominal_receptive_field_pixels": 5,
            "dav2_trainable": False,
            "semantic_router_trainable_in_control": False,
            "refinement_gate_trainable_in_control": False,
            "protected_building_detector_trainable_in_control": False,
        },
        "checkpoint_change_audit": checkpoint_change_audit(
            protected_path, control_path
        ),
        "samples": sample_reports,
        "aggregate": {
            "mean_true_vegetation_hard_expert_gate_recall": numeric_mean(gate_recalls),
            "mean_final_vegetation_to_canopy_gradient_ratio": numeric_mean(canopy_ratios),
            "mean_final_building_to_expert_gradient_ratio": numeric_mean(building_ratios),
            "mean_effective_refinement_strength": numeric_mean(effective_refinement),
        },
        "interpretation_contract": {
            "accuracy_claim": "none; training crops only",
            "causal_question": (
                "whether available expert gradients can materially alter the final "
                "height through the frozen protected fusion path"
            ),
            "next_candidate_must_be_evaluated_with": (
                "locked corrected HighBuild/OpenCanopy replay before any app change"
            ),
        },
    }
    atomic_json(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protected-checkpoint", type=Path, default=DEFAULT_PROTECTED)
    parser.add_argument("--control-checkpoint", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "output": str(DEFAULT_OUTPUT),
        "status": result["status"],
        "scope": result["scope"],
        "capacity": result["capacity"],
        "aggregate": result["aggregate"],
        "checkpoint_change_audit": result["checkpoint_change_audit"],
    }, indent=2))
