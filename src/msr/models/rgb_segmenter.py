"""Isolated RGB-only six-class segmenter; never used by the production height path.

An ImageNet-pretrained MiT encoder is fine-tuned together with a fresh SegFormer
decoder. Inputs are raw RGB in [0, 1]. Normalization lives in this wrapper so the
same contract can be retained by a future, explicitly validated overlay adapter.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


CLASS_NAMES = ("ground", "buildings", "water", "roads", "low_vegetation", "trees")


class RgbSegformer(nn.Module):
    model_type = "rgb_segformer_six_class_v1"

    def __init__(self, config, *, encoder: nn.Module | None = None) -> None:
        super().__init__()
        # Lazy import keeps dataset/evaluation tooling independent of Transformers.
        from transformers import SegformerForSemanticSegmentation

        config = deepcopy(config)
        if config.num_channels != 3:
            raise ValueError("The independent segmenter accepts exactly three RGB channels")
        config.num_labels = len(CLASS_NAMES)
        config.id2label = dict(enumerate(CLASS_NAMES))
        config.label2id = {name: index for index, name in enumerate(CLASS_NAMES)}
        config.semantic_loss_ignore_index = 255
        self.network = SegformerForSemanticSegmentation(config)
        if encoder is not None:
            self.network.segformer = encoder
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(True)

    @classmethod
    def from_local_pretrained(cls, directory: str | Path) -> "RgbSegformer":
        from transformers import SegformerModel

        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Pinned local MiT encoder directory is missing: {directory}")
        encoder, loading = SegformerModel.from_pretrained(
            str(directory), local_files_only=True, weights_only=True, output_loading_info=True
        )
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise RuntimeError(f"Pretrained encoder did not load completely: {loading}")
        # ImageNet classifier weights are intentionally not part of the encoder.
        unexpected = loading.get("unexpected_keys", [])
        if any(not key.startswith("classifier.") for key in unexpected):
            raise RuntimeError(f"Unexpected non-classifier pretrained tensors: {unexpected}")
        model = cls(encoder.config, encoder=encoder)
        model.pretrained_loading_info = loading
        return model

    @classmethod
    def from_architecture(cls, architecture: dict) -> "RgbSegformer":
        """Reconstruct from a standalone checkpoint, without any network access."""
        from transformers import SegformerConfig

        return cls(SegformerConfig.from_dict(architecture))

    @property
    def architecture(self) -> dict:
        return self.network.config.to_dict()

    def parameter_groups(self, encoder_lr: float, decoder_lr: float, weight_decay: float) -> list[dict]:
        if encoder_lr <= 0 or decoder_lr <= 0 or weight_decay < 0:
            raise ValueError("Learning rates must be positive and weight decay nonnegative")
        return [
            {"name": "encoder", "params": list(self.network.segformer.parameters()), "lr": encoder_lr,
             "initial_lr": encoder_lr, "weight_decay": weight_decay},
            {"name": "decoder", "params": list(self.network.decode_head.parameters()), "lr": decoder_lr,
             "initial_lr": decoder_lr, "weight_decay": weight_decay},
        ]

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("Expected floating RGB tensor shaped [batch, 3, height, width]")
        normalized = (image - self.rgb_mean) / self.rgb_std
        logits = self.network(pixel_values=normalized).logits
        logits = F.interpolate(logits.float(), size=image.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits}
