"""Experimental soft six-class residual routing, not a production model.

Each class has a learned residual output over shared spatial features. Frozen
RGB probabilities mix corrections at native pixel resolution. A uniform-router
control has the exact same capacity. No reference or height-label masks enter
forward; a class never implies a fixed height (including water and roads).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .class_assisted_height import conditioning_channels
from .residual_height import ResidualHeightSurfaceNet


def mix_expert_corrections(logits, probabilities, image_valid, *, arm, maximum_correction_m):
    weights = conditioning_channels(probabilities, image_valid, arm=arm)[:, :6]
    if logits.shape != probabilities.shape:
        raise ValueError('Expert/class native grids must match')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite expert logits')
    corrections = maximum_correction_m * torch.tanh(logits.float())
    # Invalid RGB carries no correction, not an invented class or ground label.
    return (corrections * weights).sum(1, keepdim=True)


class SemanticExpertHeightNet(ResidualHeightSurfaceNet):
    model_type = 'soft_six_class_expert_height_v1'

    def __init__(self, protected, *, hidden_channels=32, maximum_correction_m=80.):
        super().__init__(protected, hidden_channels=hidden_channels,
                         maximum_correction_m=maximum_correction_m)
        self.correction_head = nn.Conv2d(hidden_channels, 6, kernel_size=1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, image, relative_prior, *, probabilities, image_valid, arm):
        if relative_prior is None or relative_prior.shape != (image.shape[0], 1, *image.shape[-2:]):
            raise ValueError('Source-bound prior/image grid mismatch')
        if probabilities.shape != (image.shape[0], 6, *image.shape[-2:]):
            raise ValueError('Class/image grid mismatch')
        # Validate before the expensive model call. Probability channels remain
        # detached; the independent classifier is never trained by height loss.
        conditioning_channels(probabilities, image_valid, arm=arm)
        with torch.no_grad():
            original = self.protected(image, relative_prior, return_features=True)
        features = original.pop('encoder_features')
        original.pop('decoded_features')
        decoded = self.deepest(features[-1])
        for stage, skip in zip(self.decoder_stages, reversed(features[:-1])):
            decoded = F.interpolate(decoded, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            decoded = stage(torch.cat((decoded, skip), dim=1))
        # Both arms have identical RGB/prior/height context. Only the final
        # router differs; probabilities are not hidden in normalised features.
        context = torch.cat((image, relative_prior, torch.log1p(original['height'].float())/5.), dim=1)
        context = F.interpolate(context, size=decoded.shape[-2:], mode='bilinear', align_corners=False)
        logits = self.correction_head(self.spatial_correction(torch.cat((decoded, context), dim=1)))
        logits = F.interpolate(logits, size=image.shape[-2:], mode='bilinear', align_corners=False)
        correction = mix_expert_corrections(logits, probabilities, image_valid, arm=arm,
                                            maximum_correction_m=self.maximum_correction_m)
        return {**original, 'height':(original['height'].float()+correction).clamp_min(0),
                'uncorrected_height':original['height'], 'height_correction':correction}
