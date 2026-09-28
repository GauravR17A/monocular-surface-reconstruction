"""An independent spatial height decoder over a frozen production backbone.

All classification and original height tensors stay frozen. The copied decoder
learns corrections to the final height directly, using the encoder's multi-scale
image features. No semantic or reference mask is accepted as a model input.
"""

from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .domain_surface_net import DomainGatedSurfaceNet
from .height_net import ConvBlock


class ResidualHeightSurfaceNet(nn.Module):
    model_type = "residual_height_surface_v1"

    def __init__(
        self,
        protected: DomainGatedSurfaceNet,
        *,
        hidden_channels: int = 32,
        maximum_correction_m: float = 80.0,
    ) -> None:
        super().__init__()
        if not isinstance(protected, DomainGatedSurfaceNet):
            raise TypeError("Residual height requires a DomainGatedSurfaceNet")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if not math.isfinite(maximum_correction_m) or maximum_correction_m <= 0:
            raise ValueError("maximum_correction_m must be finite and positive")
        self.protected = protected
        self.protected.requires_grad_(False).eval()
        self.maximum_correction_m = float(maximum_correction_m)
        # Copy pretrained urban decoding weights, with independent storage.
        # One frozen encoder forward supplies both decoders.
        self.deepest = deepcopy(protected.base_model.deepest)
        self.decoder_stages = deepcopy(protected.base_model.decoder_stages)
        self.deepest.requires_grad_(True)
        self.decoder_stages.requires_grad_(True)
        decoded_channels = protected.base_model.height_head.in_channels
        # Image RGB + DAV2 relative prior + protected surface height provide
        # local texture/geometry alongside the contextual learned features.
        self.spatial_correction = ConvBlock(decoded_channels + 5, hidden_channels)
        self.correction_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.protected.eval()
        return self

    def forward(
        self, image: torch.Tensor, relative_prior: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if relative_prior is None:
            raise ValueError("Residual height requires the cached relative-depth prior")
        with torch.no_grad():
            original = self.protected(image, relative_prior, return_features=True)
        features = original.pop("encoder_features")
        original.pop("decoded_features")
        decoded = self.deepest(features[-1])
        for stage, skip in zip(self.decoder_stages, reversed(features[:-1])):
            decoded = F.interpolate(
                decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            decoded = stage(torch.cat((decoded, skip), dim=1))
        native_size = original["height"].shape[-2:]
        if relative_prior.shape[-2:] != native_size:
            relative_prior = F.interpolate(
                relative_prior, size=native_size, mode="bilinear", align_corners=False
            )
        context = torch.cat(
            (image, relative_prior, torch.log1p(original["height"].float()) / 5.0),
            dim=1,
        )
        context = F.interpolate(
            context, size=decoded.shape[-2:], mode="bilinear", align_corners=False
        )
        spatial = self.spatial_correction(torch.cat((decoded, context), dim=1))
        residual_logits = self.correction_head(spatial)
        residual_logits = F.interpolate(
            residual_logits, size=native_size, mode="bilinear", align_corners=False
        )
        # Do metric arithmetic in fp32 even when convolution uses bf16.
        correction = self.maximum_correction_m * torch.tanh(residual_logits.float())
        height = (original["height"].float() + correction).clamp_min(0.0)
        return {
            **original,
            "height": height,
            "uncorrected_height": original["height"],
            "height_correction": correction,
        }
