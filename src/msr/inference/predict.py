"""Checkpoint loading and seam-resistant inference over arbitrary RGB arrays."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from msr.data.raster_dataset import IMAGENET_MEAN, IMAGENET_STD
from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.height_net import HeightNet
from msr.models.residual_height import ResidualHeightSurfaceNet

from .tiling import blend_weight, tile_starts


@dataclass(frozen=True)
class PredictionResult:
    height_map: np.ndarray
    building_probability_map: np.ndarray | None
    vegetation_probability_map: np.ndarray | None
    fine_semantic_probability_maps: np.ndarray | None
    confidence_map: np.ndarray | None
    valid_mask: np.ndarray
    model_metadata: dict[str, Any]


def load_predictor(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstruct a protected urban or guarded multi-domain checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = config["model"]
    model_type = str(checkpoint.get("model_type", "height_net_v1"))
    base_metadata = None
    multidomain_model_types = {
        "domain_gated_surface_v1",
        "domain_gated_surface_v2",
        "domain_gated_surface_v3",
        "domain_gated_surface_v4_six_class",
    }
    protected_state = None
    if model_type == ResidualHeightSurfaceNet.model_type:
        protected_path = Path(model_config["protected_checkpoint"])
        digest = hashlib.sha256()
        with protected_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != model_config["protected_checkpoint_sha256"]:
            raise ValueError("Residual predictor protected checkpoint hash mismatch")
        protected, base_metadata = load_predictor(protected_path, device="cpu")
        protected_state = {key: value.clone() for key, value in protected.state_dict().items()}
        model = ResidualHeightSurfaceNet(
            protected,
            hidden_channels=int(model_config.get("hidden_channels", 32)),
            maximum_correction_m=float(model_config.get("maximum_correction_m", 80.0)),
        )
    elif model_type in multidomain_model_types:
        if "base_architecture" in model_config:
            # Public inference bundles carry the architecture and all base tensors.
            architecture = model_config["base_architecture"]
            base_model = HeightNet(pretrained=False, **architecture)
            base_metadata = {"backbone": architecture["backbone"], "checkpoint": "embedded base tensors"}
        else:
            base_checkpoint = Path(model_config["base_checkpoint"])
            base_model, base_metadata = load_predictor(base_checkpoint, device="cpu")
        if not isinstance(base_model, HeightNet):
            raise ValueError("A multi-domain checkpoint must reference a HeightNet base")
        model = DomainGatedSurfaceNet(
            base_model,
            hidden_channels=int(model_config.get("hidden_channels", 48)),
            maximum_building_residual_m=float(
                model_config.get("maximum_building_residual_m", 30.0)
            ),
            initial_canopy_height_m=float(
                model_config.get("initial_canopy_height_m", 8.0)
            ),
            initial_refinement_strength=float(
                model_config.get("initial_refinement_strength", 0.08)
            ),
            fusion_mode=str(model_config.get("fusion_mode", "legacy")),
            building_protection_power=float(
                model_config.get("building_protection_power", 2.0)
            ),
            building_fusion_min_height_m=float(
                model_config.get("building_fusion_min_height_m", 10.0)
            ),
            building_fusion_temperature_m=float(
                model_config.get("building_fusion_temperature_m", 2.0)
            ),
            building_fusion_score_threshold=float(
                model_config.get("building_fusion_score_threshold", 0.5)
            ),
            building_fusion_score_temperature=float(
                model_config.get("building_fusion_score_temperature", 0.05)
            ),
            building_fusion_strength=float(
                model_config.get("building_fusion_strength", 1.0)
            ),
            vegetation_fusion_temperature=float(
                model_config.get("vegetation_fusion_temperature", 1.0)
            ),
            vegetation_expert_fusion_threshold=float(
                model_config.get("vegetation_expert_fusion_threshold", 1.0)
            ),
            vegetation_expert_fusion_strength=float(
                model_config.get("vegetation_expert_fusion_strength", 0.0)
            ),
            fine_semantic_classes=int(
                model_config.get("fine_semantic_classes", 0)
            ),
            fine_semantic_head_type=str(
                model_config.get("fine_semantic_head_type", "linear")
            ),
            freeze_base=True,
        )
    else:
        model = HeightNet(
            backbone=model_config["backbone"],
            pretrained=False,
            decoder_channels=model_config["decoder_channels"],
            auxiliary_building_head=bool(model_config["auxiliary_building_head"]),
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    if protected_state is not None:
        for key, value in model.protected.state_dict().items():
            if not torch.equal(value, protected_state[key]):
                raise ValueError(f"Residual predictor altered protected tensor: {key}")
    model.to(device).eval()
    is_surface = model_type in multidomain_model_types or isinstance(model, ResidualHeightSurfaceNet)
    semantic_model = model.protected if isinstance(model, ResidualHeightSurfaceNet) else model
    metadata = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "epoch": checkpoint.get("epoch"),
        "validation_metrics": checkpoint.get("metrics", checkpoint.get("validation")),
        "model_type": model_type,
        "backbone": (
            model_config.get("backbone")
            if not is_surface
            else (base_metadata or {}).get("backbone")
        ),
        "task": "multi_domain_surface_height_m"
        if is_surface
        else "height_above_ground_m",
        "base_checkpoint": (
            (base_metadata or {}).get("checkpoint")
            if is_surface
            else None
        ),
        "fine_semantic_classes": (
            list(DomainGatedSurfaceNet.fine_semantic_names)
            if isinstance(semantic_model, DomainGatedSurfaceNet)
            and semantic_model.fine_semantic_head is not None
            else None
        ),
        "fine_semantic_head_type": (
            semantic_model.fine_semantic_head_type
            if isinstance(semantic_model, DomainGatedSurfaceNet)
            and semantic_model.fine_semantic_head is not None
            else None
        ),
    }
    return model, metadata


def _as_chw_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"RGB image must be 3-D, received shape {image.shape}")
    if image.shape[0] == 3:
        chw = image
    elif image.shape[-1] >= 3:
        chw = np.moveaxis(image[..., :3], -1, 0)
    else:
        raise ValueError(f"Could not identify three RGB bands in shape {image.shape}")
    return chw.astype(np.float32, copy=False)


def _normalize_tile(tile: np.ndarray, rgb_scale: float) -> torch.Tensor:
    tile = tile / rgb_scale
    tile = (tile - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(tile[None, ...]))


@torch.inference_mode()
def predict_height(
    image: np.ndarray,
    *,
    model: torch.nn.Module,
    device: str | torch.device = "cuda",
    tile_size: int = 512,
    overlap: int = 128,
    rgb_scale: float = 255.0,
    amp: bool = True,
    model_metadata: dict[str, Any] | None = None,
    relative_prior: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    padding_policy: str = "fixed_tile",
) -> PredictionResult:
    """Predict a dense metric height map from one RGB image.

    The model is evaluated on overlapping tiles. A positive 2-D blending window
    suppresses seams while retaining full coverage at image boundaries.
    ``minimal`` pads smaller inputs only to the model's 32-pixel stride,
    avoiding large artificial black borders that affect spatial normalization.
    The default retains the original fixed-tile behavior for old callers.
    """

    if rgb_scale <= 0:
        raise ValueError("rgb_scale must be positive")
    if padding_policy not in {"fixed_tile", "minimal"}:
        raise ValueError("padding_policy must be 'fixed_tile' or 'minimal'")
    device = torch.device(device)
    chw = _as_chw_rgb(image)
    height, width = chw.shape[-2:]
    if relative_prior is not None:
        relative_prior = np.asarray(relative_prior, dtype=np.float32)
        if relative_prior.shape != (height, width):
            raise ValueError(
                "relative_prior must match the RGB height/width: "
                f"{relative_prior.shape} != {(height, width)}"
            )
        # A no-data pixel must not poison an entire convolutional inference
        # tile. The authoritative validity mask is applied to final outputs;
        # inside the model, missing prior values are neutral geometry hints.
        relative_prior = np.nan_to_num(
            relative_prior,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            copy=False,
        )
    finite_rgb = np.all(np.isfinite(chw), axis=0)
    if valid_mask is None:
        valid_mask = finite_rgb
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != (height, width):
            raise ValueError(
                "valid_mask must match the RGB height/width: "
                f"{valid_mask.shape} != {(height, width)}"
            )
        valid_mask = valid_mask & finite_rgb
    chw = np.nan_to_num(chw, copy=False)

    y_starts = tile_starts(height, tile_size, overlap)
    x_starts = tile_starts(width, tile_size, overlap)
    output = np.zeros((height, width), dtype=np.float64)
    total_weight = np.zeros((height, width), dtype=np.float64)
    building_output: np.ndarray | None = None
    vegetation_output: np.ndarray | None = None
    confidence_output: np.ndarray | None = None
    fine_semantic_output: np.ndarray | None = None

    for y in y_starts:
        for x in x_starts:
            tile = chw[:, y : min(y + tile_size, height), x : min(x + tile_size, width)]
            actual_height, actual_width = tile.shape[-2:]
            padded_height = tile_size
            padded_width = tile_size
            if padding_policy == "minimal":
                if actual_height < tile_size:
                    padded_height = ((actual_height + 31) // 32) * 32
                if actual_width < tile_size:
                    padded_width = ((actual_width + 31) // 32) * 32
            if actual_height < padded_height or actual_width < padded_width:
                tile = np.pad(
                    tile,
                    (
                        (0, 0),
                        (0, padded_height - actual_height),
                        (0, padded_width - actual_width),
                    ),
                    mode="constant",
                )
            tensor = _normalize_tile(tile, rgb_scale).to(device, non_blocking=True)
            prior_tensor = None
            if relative_prior is not None:
                prior_tile = relative_prior[
                    y : y + actual_height, x : x + actual_width
                ]
                if actual_height < padded_height or actual_width < padded_width:
                    prior_tile = np.pad(
                        prior_tile,
                        (
                            (0, padded_height - actual_height),
                            (0, padded_width - actual_width),
                        ),
                        mode="constant",
                    )
                prior_tensor = torch.from_numpy(
                    np.ascontiguousarray(prior_tile[None, None])
                ).to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                if isinstance(model, (DomainGatedSurfaceNet, ResidualHeightSurfaceNet)):
                    model_output = model(tensor, prior_tensor)
                else:
                    model_output = model(tensor)
                prediction = model_output["height"]
            predicted_tile = (
                prediction[0, 0, :actual_height, :actual_width]
                .float()
                .cpu()
                .numpy()
            )
            weight = blend_weight(actual_height, actual_width)
            output[y : y + actual_height, x : x + actual_width] += predicted_tile * weight
            total_weight[y : y + actual_height, x : x + actual_width] += weight
            building_logits = model_output.get(
                "protected_building_logits", model_output.get("building_logits")
            )
            if building_logits is not None:
                if building_output is None:
                    building_output = np.zeros((height, width), dtype=np.float64)
                building_tile = (
                    building_logits[0, 0, :actual_height, :actual_width]
                    .sigmoid()
                    .float()
                    .cpu()
                    .numpy()
                )
                building_output[y : y + actual_height, x : x + actual_width] += (
                    building_tile * weight
                )
            if "vegetation_logits" in model_output:
                if vegetation_output is None:
                    vegetation_output = np.zeros((height, width), dtype=np.float64)
                vegetation_tile = (
                    model_output["vegetation_logits"][0, 0, :actual_height, :actual_width]
                    .sigmoid()
                    .float()
                    .cpu()
                    .numpy()
                )
                vegetation_output[y : y + actual_height, x : x + actual_width] += (
                    vegetation_tile * weight
                )
            if "fine_semantic_probabilities" in model_output:
                fine_semantic_tile = (
                    model_output["fine_semantic_probabilities"][
                        0, :, :actual_height, :actual_width
                    ]
                    .float()
                    .cpu()
                    .numpy()
                )
                if fine_semantic_output is None:
                    fine_semantic_output = np.zeros(
                        (fine_semantic_tile.shape[0], height, width),
                        dtype=np.float64,
                    )
                fine_semantic_output[
                    :, y : y + actual_height, x : x + actual_width
                ] += fine_semantic_tile * weight[None]
            if "log_variance" in model_output:
                if confidence_output is None:
                    confidence_output = np.zeros((height, width), dtype=np.float64)
                sigma = (
                    (0.5 * model_output["log_variance"][0, 0, :actual_height, :actual_width])
                    .exp()
                    .float()
                    .cpu()
                    .numpy()
                )
                confidence_output[y : y + actual_height, x : x + actual_width] += (
                    (1.0 / (1.0 + sigma)) * weight
                )

    height_map = (output / np.maximum(total_weight, 1e-12)).astype(np.float32)
    height_map[~valid_mask] = np.nan
    building_probability_map = None
    if building_output is not None:
        building_probability_map = (
            building_output / np.maximum(total_weight, 1e-12)
        ).astype(np.float32)
        building_probability_map[~valid_mask] = np.nan
    vegetation_probability_map = None
    if vegetation_output is not None:
        vegetation_probability_map = (
            vegetation_output / np.maximum(total_weight, 1e-12)
        ).astype(np.float32)
        vegetation_probability_map[~valid_mask] = np.nan
    confidence_map = None
    if confidence_output is not None:
        confidence_map = (
            confidence_output / np.maximum(total_weight, 1e-12)
        ).astype(np.float32)
        confidence_map[~valid_mask] = np.nan
    fine_semantic_probability_maps = None
    if fine_semantic_output is not None:
        fine_semantic_probability_maps = (
            fine_semantic_output / np.maximum(total_weight[None], 1e-12)
        ).astype(np.float32)
        fine_semantic_probability_maps[:, ~valid_mask] = np.nan
    return PredictionResult(
        height_map=height_map,
        building_probability_map=building_probability_map,
        vegetation_probability_map=vegetation_probability_map,
        fine_semantic_probability_maps=fine_semantic_probability_maps,
        confidence_map=confidence_map,
        valid_mask=valid_mask,
        model_metadata=dict(model_metadata or {}),
    )
