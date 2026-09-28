"""Deployable, label-free scene features for a future Stage-4c semantic router.

The extractor deliberately accepts only tensors that are available during
ordinary endpoint inference.  It does not accept reference targets, validity
masks, source names, city/group identifiers, or any other audit metadata.

RGB is expected in the ImageNet-normalized representation consumed by the
shared Monocular Surface Reconstruction model.  Statistics and vegetation indices are calculated
after converting it back to the nominal ``[0, 1]`` RGB range.  The protected
final height is the stable base-height summary; the candidate final height is
represented relative to it by candidate-minus-protected quantiles.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


STAGE4C_SEMANTIC_FEATURE_SCHEMA = (
    "msr.stage4c_semantic_routing_features.v1"
)
# The deployed DomainGatedSurfaceNet checkpoints use a 64-channel adapter.
# Keep the schema default aligned with that production contract; tests and
# experimental architectures can still request another width explicitly.
STAGE4C_DEFAULT_ADAPTER_CHANNELS = 64
STAGE4C_SEMANTIC_CLASSES = ("ground", "building", "vegetation")
STAGE4C_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)

_RGB_CHANNELS = ("red", "green", "blue")
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _quantile_name(value: float) -> str:
    return f"q{int(round(value * 100)):02d}"


def stage4c_semantic_feature_names(
    adapter_channels: int = STAGE4C_DEFAULT_ADAPTER_CHANNELS,
) -> tuple[str, ...]:
    """Return the canonical descriptor layout for an adapter width."""

    if adapter_channels <= 0:
        raise ValueError("adapter_channels must be positive")
    names: list[str] = []
    names.extend(f"adapter_mean_c{index}" for index in range(adapter_channels))
    names.extend(f"adapter_std_c{index}" for index in range(adapter_channels))
    names.extend(f"rgb_mean_{channel}" for channel in _RGB_CHANNELS)
    names.extend(f"rgb_std_{channel}" for channel in _RGB_CHANNELS)
    for index_name in ("exg", "vari"):
        names.extend(
            f"{index_name}_{_quantile_name(quantile)}"
            for quantile in STAGE4C_QUANTILES
        )
    for map_name in ("relative_prior", "protected_base_height"):
        names.extend(
            f"{map_name}_{_quantile_name(quantile)}"
            for quantile in STAGE4C_QUANTILES
        )
    for endpoint in ("protected", "candidate"):
        names.extend(
            f"{endpoint}_prob_mean_{class_name}"
            for class_name in STAGE4C_SEMANTIC_CLASSES
        )
        names.extend(
            f"{endpoint}_prob_std_{class_name}"
            for class_name in STAGE4C_SEMANTIC_CLASSES
        )
        for class_name in STAGE4C_SEMANTIC_CLASSES:
            names.extend(
                f"{endpoint}_prob_{class_name}_{_quantile_name(quantile)}"
                for quantile in STAGE4C_QUANTILES
            )
        names.extend(
            (f"{endpoint}_entropy_mean", f"{endpoint}_entropy_std")
        )
        names.extend(
            f"{endpoint}_entropy_{_quantile_name(quantile)}"
            for quantile in STAGE4C_QUANTILES
        )
        names.extend(
            f"{endpoint}_hard_fraction_{class_name}"
            for class_name in STAGE4C_SEMANTIC_CLASSES
        )
        for class_name in STAGE4C_SEMANTIC_CLASSES:
            for row in range(2):
                for column in range(2):
                    names.append(
                        f"{endpoint}_pyramid_r{row}c{column}_{class_name}"
                    )
    for protected_class in STAGE4C_SEMANTIC_CLASSES:
        for candidate_class in STAGE4C_SEMANTIC_CLASSES:
            names.append(
                "hard_disagreement_"
                f"protected_{protected_class}_candidate_{candidate_class}"
            )
    names.extend(
        f"height_delta_{_quantile_name(quantile)}"
        for quantile in STAGE4C_QUANTILES
    )
    if len(names) != len(set(names)):
        raise AssertionError("Stage-4c semantic feature names must be unique")
    return tuple(names)


def stage4c_semantic_descriptor_size(
    adapter_channels: int = STAGE4C_DEFAULT_ADAPTER_CHANNELS,
) -> int:
    """Return the fixed descriptor width for an adapter architecture."""

    return len(stage4c_semantic_feature_names(adapter_channels))


STAGE4C_SEMANTIC_DESCRIPTOR_SIZE = stage4c_semantic_descriptor_size()


def _spatial_mean_and_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=(-2, -1))
    variance = torch.square(values - mean[:, :, None, None]).mean(dim=(-2, -1))
    return mean, torch.sqrt(variance.clamp_min(0.0))


def _map_quantiles(values: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    flattened = values.flatten(start_dim=-2)
    result = torch.quantile(flattened, quantiles, dim=-1, interpolation="linear")
    # torch.quantile returns (quantiles, batch, channels).  The canonical
    # layout is batch, then channel, then increasing quantile.
    return result.permute(1, 2, 0).reshape(values.shape[0], -1)


def _scalar_quantiles(values: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    if values.ndim == 3:
        values = values[:, None]
    return _map_quantiles(values, quantiles)


def _endpoint_features(
    logits: torch.Tensor,
    quantiles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = logits.softmax(dim=1)
    probability_mean, probability_std = _spatial_mean_and_std(probabilities)
    probability_quantiles = _map_quantiles(probabilities, quantiles)

    entropy = -torch.sum(
        probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log(),
        dim=1,
    ) / math.log(len(STAGE4C_SEMANTIC_CLASSES))
    entropy_mean, entropy_std = _spatial_mean_and_std(entropy[:, None])
    entropy_quantiles = _scalar_quantiles(entropy, quantiles)

    hard_classes = probabilities.argmax(dim=1)
    hard_fractions = torch.stack(
        [
            (hard_classes == class_index).float().mean(dim=(-2, -1))
            for class_index in range(len(STAGE4C_SEMANTIC_CLASSES))
        ],
        dim=1,
    )

    # Adaptive pooling is the conventional 2x2 spatial-pyramid definition and
    # remains well-defined even for very small endpoint maps.
    pyramid = F.adaptive_avg_pool2d(probabilities, output_size=(2, 2)).flatten(1)
    features = torch.cat(
        (
            probability_mean,
            probability_std,
            probability_quantiles,
            entropy_mean,
            entropy_std,
            entropy_quantiles,
            hard_fractions,
            pyramid,
        ),
        dim=1,
    )
    return features, hard_classes


def _validate_inputs(
    *,
    adapter_features: torch.Tensor,
    rgb: torch.Tensor,
    relative_prior: torch.Tensor,
    protected_domain_logits: torch.Tensor,
    candidate_domain_logits: torch.Tensor,
    protected_height: torch.Tensor,
    candidate_height: torch.Tensor,
    adapter_channels: int,
) -> tuple[tuple[torch.Tensor, ...], bool]:
    named = (
        ("adapter_features", adapter_features),
        ("rgb", rgb),
        ("relative_prior", relative_prior),
        ("protected_domain_logits", protected_domain_logits),
        ("candidate_domain_logits", candidate_domain_logits),
        ("protected_height", protected_height),
        ("candidate_height", candidate_height),
    )
    if any(not isinstance(value, torch.Tensor) for _, value in named):
        raise TypeError("all Stage-4c semantic feature inputs must be tensors")
    dimensions = {value.ndim for _, value in named}
    if dimensions not in ({3}, {4}):
        raise ValueError(
            "inputs must all be unbatched CHW tensors or all be batched BCHW tensors"
        )
    unbatched = dimensions == {3}
    values = tuple(value[None] if unbatched else value for _, value in named)
    if len({value.device for value in values}) != 1:
        raise ValueError("all Stage-4c semantic feature inputs must share one device")

    expected_channels = (adapter_channels, 3, 1, 3, 3, 1, 1)
    batch_spatial = (values[0].shape[0], values[0].shape[-2:])
    for (name, _), value, channels in zip(named, values, expected_channels):
        if value.shape[1] != channels:
            raise ValueError(
                f"{name} must have {channels} channels, found {value.shape[1]}"
            )
        if (value.shape[0], value.shape[-2:]) != batch_spatial:
            raise ValueError(
                f"{name} must match adapter_features batch and spatial dimensions"
            )
        if value.shape[0] <= 0 or value.shape[-2] <= 0 or value.shape[-1] <= 0:
            raise ValueError("Stage-4c semantic feature inputs cannot be empty")
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{name} must contain only finite values")
    return values, unbatched


class Stage4CSemanticFeatureExtractor(nn.Module):
    """Create one stable rich semantic-routing descriptor per scene.

    Batched BCHW inputs produce ``(batch, descriptor_size)``.  Supplying every
    input as unbatched CHW produces one one-dimensional descriptor.  The output
    is always float32 and stays on the input device.
    """

    def __init__(
        self,
        adapter_channels: int = STAGE4C_DEFAULT_ADAPTER_CHANNELS,
        *,
        quantiles: Sequence[float] = STAGE4C_QUANTILES,
    ) -> None:
        super().__init__()
        if adapter_channels <= 0:
            raise ValueError("adapter_channels must be positive")
        canonical_quantiles = tuple(float(value) for value in quantiles)
        if canonical_quantiles != STAGE4C_QUANTILES:
            raise ValueError(
                "Stage-4c descriptor quantiles are schema-fixed at "
                f"{STAGE4C_QUANTILES}"
            )
        self.adapter_channels = int(adapter_channels)
        self.feature_names = stage4c_semantic_feature_names(self.adapter_channels)
        self.descriptor_size = len(self.feature_names)
        self.register_buffer(
            "quantiles",
            torch.tensor(canonical_quantiles, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        adapter_features: torch.Tensor,
        rgb: torch.Tensor,
        relative_prior: torch.Tensor,
        protected_domain_logits: torch.Tensor,
        candidate_domain_logits: torch.Tensor,
        protected_height: torch.Tensor,
        candidate_height: torch.Tensor,
    ) -> torch.Tensor:
        values, unbatched = _validate_inputs(
            adapter_features=adapter_features,
            rgb=rgb,
            relative_prior=relative_prior,
            protected_domain_logits=protected_domain_logits,
            candidate_domain_logits=candidate_domain_logits,
            protected_height=protected_height,
            candidate_height=candidate_height,
            adapter_channels=self.adapter_channels,
        )
        (
            adapter_features,
            rgb,
            relative_prior,
            protected_domain_logits,
            candidate_domain_logits,
            protected_height,
            candidate_height,
        ) = values

        with torch.autocast(device_type=adapter_features.device.type, enabled=False):
            adapter_features = adapter_features.float()
            rgb = rgb.float()
            relative_prior = relative_prior.float()
            protected_domain_logits = protected_domain_logits.float()
            candidate_domain_logits = candidate_domain_logits.float()
            protected_height = protected_height.float()
            candidate_height = candidate_height.float()
            quantiles = self.quantiles.to(device=adapter_features.device)

            adapter_mean, adapter_std = _spatial_mean_and_std(adapter_features)

            imagenet_mean = rgb.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            imagenet_std = rgb.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            rgb_unit = (rgb * imagenet_std + imagenet_mean).clamp(0.0, 1.0)
            rgb_mean, rgb_std = _spatial_mean_and_std(rgb_unit)
            red, green, blue = rgb_unit.unbind(dim=1)
            exg = 2.0 * green - red - blue
            vari_denominator = green + red - blue
            defined_vari = vari_denominator.abs() >= 1.0e-6
            # VARI is undefined when its denominator is zero.  Map that
            # measure-zero case to the neutral value instead of creating an
            # enormous feature that would dominate the routing descriptor.
            safe_denominator = torch.where(
                defined_vari, vari_denominator, torch.ones_like(vari_denominator)
            )
            vari = torch.where(
                defined_vari,
                (green - red) / safe_denominator,
                torch.zeros_like(vari_denominator),
            )

            protected_features, protected_hard = _endpoint_features(
                protected_domain_logits, quantiles
            )
            candidate_features, candidate_hard = _endpoint_features(
                candidate_domain_logits, quantiles
            )
            disagreement = torch.stack(
                [
                    (
                        (protected_hard == protected_class)
                        & (candidate_hard == candidate_class)
                    )
                    .float()
                    .mean(dim=(-2, -1))
                    for protected_class in range(
                        len(STAGE4C_SEMANTIC_CLASSES)
                    )
                    for candidate_class in range(
                        len(STAGE4C_SEMANTIC_CLASSES)
                    )
                ],
                dim=1,
            )

            descriptor = torch.cat(
                (
                    adapter_mean,
                    adapter_std,
                    rgb_mean,
                    rgb_std,
                    _scalar_quantiles(exg, quantiles),
                    _scalar_quantiles(vari, quantiles),
                    _map_quantiles(relative_prior, quantiles),
                    _map_quantiles(protected_height, quantiles),
                    protected_features,
                    candidate_features,
                    disagreement,
                    _map_quantiles(candidate_height - protected_height, quantiles),
                ),
                dim=1,
            )

        if descriptor.shape[1] != self.descriptor_size:
            raise AssertionError(
                "Stage-4c semantic descriptor layout disagrees with its schema"
            )
        if descriptor.dtype != torch.float32 or not bool(
            torch.all(torch.isfinite(descriptor))
        ):
            raise ValueError("Stage-4c semantic descriptor must be finite float32")
        return descriptor[0] if unbatched else descriptor


__all__ = [
    "STAGE4C_DEFAULT_ADAPTER_CHANNELS",
    "STAGE4C_QUANTILES",
    "STAGE4C_SEMANTIC_CLASSES",
    "STAGE4C_SEMANTIC_DESCRIPTOR_SIZE",
    "STAGE4C_SEMANTIC_FEATURE_SCHEMA",
    "Stage4CSemanticFeatureExtractor",
    "stage4c_semantic_descriptor_size",
    "stage4c_semantic_feature_names",
]
