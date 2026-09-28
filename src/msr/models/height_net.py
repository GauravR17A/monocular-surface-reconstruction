"""Multi-scale pretrained encoder with height and optional building heads."""

from __future__ import annotations

from collections.abc import Sequence

import timm
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        group_count = min(32, output_channels)
        while output_channels % group_count != 0:
            group_count -= 1
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group_count, output_channels),
            nn.GELU(),
        )


class HeightNet(nn.Module):
    """U-Net-like decoder over a timm encoder; pretrained weight terms apply."""

    def __init__(
        self,
        backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        *,
        pretrained: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        auxiliary_building_head: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        feature_channels = list(self.encoder.feature_info.channels())
        if len(decoder_channels) != len(feature_channels):
            raise ValueError(
                "decoder_channels must have one entry per encoder feature level: "
                f"{len(decoder_channels)} != {len(feature_channels)}"
            )

        self.deepest = ConvBlock(feature_channels[-1], decoder_channels[0])
        stages: list[nn.Module] = []
        current_channels = decoder_channels[0]
        for skip_channels, output_channels in zip(
            reversed(feature_channels[:-1]), decoder_channels[1:]
        ):
            stages.append(ConvBlock(current_channels + skip_channels, output_channels))
            current_channels = output_channels
        self.decoder_stages = nn.ModuleList(stages)
        self.height_head = nn.Conv2d(current_channels, 1, kernel_size=1)
        self.building_head = (
            nn.Conv2d(current_channels, 1, kernel_size=1) if auxiliary_building_head else None
        )

    def forward(
        self, image: torch.Tensor, *, return_features: bool = False
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        input_size = image.shape[-2:]
        features = self.encoder(image)
        decoded = self.deepest(features[-1])
        for stage, skip in zip(self.decoder_stages, reversed(features[:-1])):
            decoded = F.interpolate(decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            decoded = stage(torch.cat((decoded, skip), dim=1))

        height_raw = self.height_head(decoded)
        height = F.softplus(
            F.interpolate(height_raw, size=input_size, mode="bilinear", align_corners=False)
        )
        output = {"height": height}
        if self.building_head is not None:
            output["building_logits"] = F.interpolate(
                self.building_head(decoded), size=input_size, mode="bilinear", align_corners=False
            )
        if return_features:
            # Opt-in access for an independent height decoder. Normal inference
            # and every existing checkpoint keep the identical computation.
            output["encoder_features"] = tuple(features)
            output["decoded_features"] = decoded
        return output
