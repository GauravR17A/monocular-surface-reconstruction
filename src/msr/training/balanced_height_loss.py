"""Source- and domain-balanced metric-height supervision.

The corrected replay combines dense GAMUS/OpenCanopy labels with sparse
HighBuild building measurements.  A single pixel mean would let the dense
sources dominate.  This loss therefore reduces in three explicit stages:

1. mean over valid pixels for each sample and domain;
2. mean over samples, then present domains, within each source;
3. mean over the present GAMUS, HighBuild, and OpenCanopy sources.

Missing labels are selected out before any arithmetic, so placeholder values
outside ``regression_mask`` neither affect the loss nor receive gradients.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


DOMAIN_NAMES = ("ground", "building", "vegetation")
SOURCE_NAMES = ("gamus", "highbuild", "open_canopy")


def _batch_strings(value: object, *, name: str, batch_size: int) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[object] = (value,)
    elif isinstance(value, Sequence):
        values = value
    else:
        raise TypeError(f"batch[{name!r}] must contain one string per sample")
    if len(values) != batch_size or any(not isinstance(item, str) for item in values):
        raise ValueError(f"batch[{name!r}] must contain one string per sample")
    normalized = tuple(str(item).strip().lower() for item in values)
    if any(not item for item in normalized):
        raise ValueError(f"batch[{name!r}] contains an empty label")
    return normalized


def _source_group(source: str, landscape: str) -> str:
    if source == "gamus":
        return "gamus"
    if source != "legacy":
        raise ValueError(f"Unsupported height-supervision source {source!r}")
    if landscape == "urban":
        return "highbuild"
    if landscape == "forest":
        return "open_canopy"
    raise ValueError(
        "Legacy height replay must identify landscape as 'urban' (HighBuild) "
        f"or 'forest' (OpenCanopy), received {landscape!r}"
    )


class SourceBalancedHeightLoss(nn.Module):
    """Huber + small MSE height loss with source/domain/sample balancing."""

    def __init__(self, *, huber_delta_m: float = 2.0, mse_weight: float = 0.02) -> None:
        super().__init__()
        if not math.isfinite(huber_delta_m) or huber_delta_m <= 0:
            raise ValueError("huber_delta_m must be finite and positive")
        if not math.isfinite(mse_weight) or mse_weight < 0:
            raise ValueError("mse_weight must be finite and nonnegative")
        self.huber_delta_m = float(huber_delta_m)
        self.mse_weight = float(mse_weight)

    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, object],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if "height" not in output:
            raise KeyError("output['height'] is required")
        required = {"height", "regression_mask", "domain_target", "source", "landscape"}
        missing = required - set(batch)
        if missing:
            raise KeyError(f"Height batch is missing fields: {sorted(missing)}")

        prediction = output["height"]
        target = batch["height"]
        regression_mask = batch["regression_mask"]
        domain_target = batch["domain_target"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (prediction, target, regression_mask, domain_target)
        ):
            raise TypeError("height, regression_mask, and domain_target must be tensors")
        if prediction.ndim != 4 or prediction.shape[1] != 1:
            raise ValueError("output['height'] must have shape (batch, 1, height, width)")
        if target.shape != prediction.shape or regression_mask.shape != prediction.shape:
            raise ValueError("target and regression_mask must match output['height']")
        expected_domain_shape = (
            prediction.shape[0],
            prediction.shape[2],
            prediction.shape[3],
        )
        if domain_target.shape != expected_domain_shape:
            raise ValueError("domain_target must have shape (batch, height, width)")

        batch_size = prediction.shape[0]
        sources = _batch_strings(batch["source"], name="source", batch_size=batch_size)
        landscapes = _batch_strings(
            batch["landscape"], name="landscape", batch_size=batch_size
        )
        source_groups = tuple(
            _source_group(source, landscape)
            for source, landscape in zip(sources, landscapes)
        )

        valid = regression_mask.bool()[:, 0]
        invalid_domains = valid & ((domain_target < 0) | (domain_target >= len(DOMAIN_NAMES)))
        if bool(torch.any(invalid_domains)):
            raise ValueError("Valid regression pixels must use domain IDs 0, 1, or 2")

        components: dict[str, torch.Tensor] = {}
        source_losses: list[torch.Tensor] = []
        for source_name in SOURCE_NAMES:
            sample_indices = [
                index for index, group in enumerate(source_groups) if group == source_name
            ]
            domain_losses: list[torch.Tensor] = []
            for domain_id, domain_name in enumerate(DOMAIN_NAMES):
                sample_losses: list[torch.Tensor] = []
                for sample_index in sample_indices:
                    sample_valid = valid[sample_index] & (
                        domain_target[sample_index] == domain_id
                    )
                    if not bool(torch.any(sample_valid)):
                        continue
                    predicted_values = prediction[sample_index, 0][sample_valid].float()
                    target_values = target[sample_index, 0][sample_valid].float()
                    if not bool(torch.all(torch.isfinite(predicted_values))):
                        raise ValueError(
                            f"Non-finite prediction on valid {source_name}/{domain_name} pixels"
                        )
                    if not bool(torch.all(torch.isfinite(target_values))):
                        raise ValueError(
                            f"Non-finite target on valid {source_name}/{domain_name} pixels"
                        )
                    huber = F.huber_loss(
                        predicted_values,
                        target_values,
                        reduction="mean",
                        delta=self.huber_delta_m,
                    )
                    mse = F.mse_loss(predicted_values, target_values, reduction="mean")
                    sample_losses.append(huber + self.mse_weight * mse)
                if sample_losses:
                    domain_loss = torch.stack(sample_losses).mean()
                    domain_losses.append(domain_loss)
                    components[f"{source_name}.{domain_name}"] = domain_loss
            if domain_losses:
                source_loss = torch.stack(domain_losses).mean()
                source_losses.append(source_loss)
                components[source_name] = source_loss

        if source_losses:
            balanced = torch.stack(source_losses).mean()
        else:
            # Empty selection yields a finite scalar attached to the prediction
            # graph, so backward() is valid and produces zero gradients.
            balanced = prediction.reshape(-1)[:0].sum().float()
        components["balanced_height"] = balanced
        components["total"] = balanced
        return balanced, components

