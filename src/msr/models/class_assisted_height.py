"""Isolated ablation: frozen RGB probabilities condition a height correction.

Reference classes/height masks are never forward inputs. Uniform and predicted
arms have identical parameters. This model is intentionally not registered in
the production inference loader: feasibility is not deployment approval.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .height_net import ConvBlock
from .residual_height import ResidualHeightSurfaceNet


def conditioning_channels(probabilities: torch.Tensor, image_valid: torch.Tensor, *, arm: str):
    if arm not in {'uniform', 'predicted'}:
        raise ValueError('arm must be uniform or predicted')
    if probabilities.ndim != 4 or probabilities.shape[1] != 6:
        raise ValueError('Expected six probability channels')
    if image_valid.shape != (probabilities.shape[0], 1, *probabilities.shape[-2:]) or image_valid.dtype != torch.bool:
        raise ValueError('Independent boolean image-validity mask required')
    valid = image_valid.expand_as(probabilities)
    selected = probabilities[valid]
    if not torch.isfinite(selected).all() or (selected < 0).any() or (selected > 1).any():
        raise ValueError('Invalid probabilities on valid image pixels')
    sums = probabilities.detach().float().sum(1, keepdim=True)[image_valid]
    if not torch.allclose(sums, torch.ones_like(sums), atol=.003, rtol=0):
        raise ValueError('Probabilities must sum to one on valid pixels')
    # Invalid RGB is explicitly unavailable, never a ground class or zero-height label.
    values = probabilities.detach().float() if arm == 'predicted' else torch.full_like(probabilities, 1/6, dtype=torch.float32)
    values = torch.where(valid, values, torch.zeros_like(values))
    return torch.cat((values, image_valid.float()), dim=1)


class ClassAssistedHeightNet(ResidualHeightSurfaceNet):
    model_type = 'class_assisted_residual_height_pilot_v1'

    def __init__(self, protected, *, hidden_channels=32, maximum_correction_m=80.):
        super().__init__(protected, hidden_channels=hidden_channels, maximum_correction_m=maximum_correction_m)
        self.spatial_correction = ConvBlock(protected.base_model.height_head.in_channels + 12, hidden_channels)

    def forward(self, image, relative_prior, *, probabilities, image_valid, arm):
        if relative_prior is None:
            raise ValueError('Cached source-bound relative prior required')
        if probabilities.shape[0] != image.shape[0] or probabilities.shape[-2:] != image.shape[-2:]:
            raise ValueError('Class/image grid mismatch')
        if relative_prior.shape != (image.shape[0], 1, *image.shape[-2:]):
            raise ValueError('Prior/image grid mismatch')
        condition = conditioning_channels(probabilities, image_valid, arm=arm)
        with torch.no_grad():
            original = self.protected(image, relative_prior, return_features=True)
        features = original.pop('encoder_features')
        original.pop('decoded_features')
        decoded = self.deepest(features[-1])
        for stage, skip in zip(self.decoder_stages, reversed(features[:-1])):
            decoded = F.interpolate(decoded, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            decoded = stage(torch.cat((decoded, skip), dim=1))
        context = torch.cat((image, relative_prior, torch.log1p(original['height'].float())/5., condition), dim=1)
        context = F.interpolate(context, size=decoded.shape[-2:], mode='bilinear', align_corners=False)
        residual = self.correction_head(self.spatial_correction(torch.cat((decoded, context), dim=1)))
        residual = F.interpolate(residual, size=original['height'].shape[-2:], mode='bilinear', align_corners=False)
        correction = self.maximum_correction_m * torch.tanh(residual.float())
        return {**original, 'height': (original['height'].float()+correction).clamp_min(0),
                'uncorrected_height': original['height'], 'height_correction': correction}
