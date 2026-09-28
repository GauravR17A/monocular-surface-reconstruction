"""Explicit short-vegetation supervision plus a labelled low-surface no-harm loss.

Only a training objective: references/masks never enter model.forward. Missing
labels are never converted to zero height. The frozen model is a comparator,
not ground truth: improving its errors is allowed, exceeding them is penalized.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F
from .balanced_height_loss import SourceBalancedHeightLoss


class LowSurfaceProtectedHeightLoss(SourceBalancedHeightLoss):
    def __init__(self, *, huber_delta_m=2., mse_weight=.02,
                 short_supervision_weight=.5, low_surface_regret_weight=1.):
        super().__init__(huber_delta_m=huber_delta_m, mse_weight=mse_weight)
        for name, value in [('short_supervision_weight',short_supervision_weight),
                            ('low_surface_regret_weight',low_surface_regret_weight)]:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            setattr(self, name, float(value))

    def forward(self, output, batch):
        base, components = super().forward(output, batch)
        prediction, target = output['height'], batch['height']
        original = output['uncorrected_height'].detach()
        if original.shape != prediction.shape:
            raise ValueError('Frozen prediction must match height shape')
        valid = batch['regression_mask'].bool()[:,0]
        domain = batch['domain_target']
        zero = prediction.reshape(-1)[:0].sum().float()
        short_losses, regrets = [], []
        # The bounded experiment contains only legacy HighBuild and OpenCanopy.
        # Do not silently apply this recipe to unverified GAMUS labels.
        for index, (source, landscape) in enumerate(zip(batch['source'],batch['landscape'])):
            if source != 'legacy' or landscape not in ('urban','forest'):
                raise ValueError('Low-surface experiment permits only verified legacy sources')
            short = valid[index] & (domain[index] == 2) & (target[index,0] > 0) & (target[index,0] <= 2)
            ground = valid[index] & (domain[index] == 0)
            for name, mask in [('ground',ground), ('short_vegetation',short)]:
                if not bool(mask.any()):
                    continue
                p, y, reference = (t[index,0][mask].float() for t in (prediction,target,original))
                if not bool(torch.isfinite(reference).all()):
                    raise ValueError('Nonfinite frozen height on valid low-surface pixels')
                error, prior_error = (p-y).abs(), (reference-y).abs()
                # Huber slope stays bounded; MSE captures large new mistakes.
                excess = (error-prior_error).clamp_min(0)
                regrets.append((excess + self.mse_weight*excess.square()).mean())
                if name == 'short_vegetation':
                    short_losses.append(F.huber_loss(p,y,delta=self.huber_delta_m) + self.mse_weight*F.mse_loss(p,y))
        short_loss = torch.stack(short_losses).mean() if short_losses else zero
        regret = torch.stack(regrets).mean() if regrets else zero
        total = base + self.short_supervision_weight*short_loss + self.low_surface_regret_weight*regret
        return total, {**components, 'short_vegetation_supervision':short_loss,
                       'low_surface_regret':regret, 'total':total}
