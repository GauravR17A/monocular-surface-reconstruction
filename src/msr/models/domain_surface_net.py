"""Semantically gated surface model that preserves a protected urban expert."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .height_net import ConvBlock, HeightNet


class RefinedFineSemanticHead(nn.Module):
    """Small residual spatial classifier isolated from every height path.

    The depthwise 3x3 layer lets the auxiliary GAMUS classifier use local
    boundary/texture context without changing the shared adapter features.
    Every tensor lives below ``fine_semantic_head.*``, so a head-only training
    policy can freeze the complete height model by prefix.
    """

    def __init__(self, channels: int, classes: int) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.normalization = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()
        self.classifier = nn.Conv2d(channels, classes, kernel_size=1)

        # Preserve the same neutral six-class starting point as the existing
        # linear head. The residual branch and classifier become learnable as
        # soon as the head-only objective runs, but cannot influence height.
        nn.init.zeros_(self.spatial.weight)
        nn.init.ones_(self.normalization.weight)
        nn.init.zeros_(self.normalization.bias)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        refined = self.activation(self.normalization(self.spatial(features)))
        return self.classifier(features + refined)


class HierarchicalVegetationFineSemanticHead(nn.Module):
    """Six-class head with a soft, conditional vegetation subdivision.

    The five-way classifier predicts ground, buildings, water, roads, and
    combined vegetation.  A separate binary logit then divides only the soft
    vegetation probability between low vegetation and trees.  The returned
    six-class scores are normalized log-probabilities, so they remain fully
    compatible with the existing softmax and cross-entropy interfaces.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.normalization = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()
        self.coarse_classifier = nn.Conv2d(channels, 5, kernel_size=1)
        self.vegetation_split = nn.Conv2d(channels, 1, kernel_size=1)

        nn.init.zeros_(self.spatial.weight)
        nn.init.ones_(self.normalization.weight)
        nn.init.zeros_(self.normalization.bias)
        nn.init.normal_(self.coarse_classifier.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.coarse_classifier.bias)
        nn.init.constant_(self.coarse_classifier.bias[4], math.log(2.0))
        nn.init.normal_(self.vegetation_split.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.vegetation_split.bias)

    def forward_with_components(
        self, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return composed predictions and the logits needed by auxiliary loss."""

        refined = self.activation(self.normalization(self.spatial(features)))
        refined_features = features + refined
        coarse_logits = self.coarse_classifier(refined_features)
        vegetation_split_logits = self.vegetation_split(refined_features)

        # Compose in log space for stable gradients even when either branch is
        # confident. Positive split logits mean tree; negative logits mean low
        # vegetation. No argmax or hard mask sits between the two branches.
        coarse_log_probabilities = F.log_softmax(coarse_logits, dim=1)
        low_vegetation_log_probability = (
            coarse_log_probabilities[:, 4:5]
            + F.logsigmoid(-vegetation_split_logits)
        )
        tree_log_probability = (
            coarse_log_probabilities[:, 4:5]
            + F.logsigmoid(vegetation_split_logits)
        )
        fine_semantic_logits = torch.cat(
            (
                coarse_log_probabilities[:, :4],
                low_vegetation_log_probability,
                tree_log_probability,
            ),
            dim=1,
        )
        return {
            "logits": fine_semantic_logits,
            "probabilities": fine_semantic_logits.exp(),
            "coarse_logits": coarse_logits,
            "coarse_probabilities": coarse_log_probabilities.exp(),
            "vegetation_split_logits": vegetation_split_logits,
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.forward_with_components(features)["logits"]


def fuse_surface_endpoint(
    *,
    base_height: torch.Tensor,
    protected_building_logits: torch.Tensor,
    domain_probabilities: torch.Tensor,
    building_height: torch.Tensor,
    canopy_height: torch.Tensor,
    refinement_logits: torch.Tensor,
    fusion_mode: str = "legacy",
    building_protection_power: float = 2.0,
    building_fusion_min_height_m: float = 10.0,
    building_fusion_temperature_m: float = 2.0,
    building_fusion_score_threshold: float = 0.5,
    building_fusion_score_temperature: float = 0.05,
    building_fusion_strength: float = 1.0,
    vegetation_fusion_temperature: float = 1.0,
    vegetation_expert_fusion_threshold: float = 1.0,
    vegetation_expert_fusion_strength: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Apply the model's complete, deterministic surface-fusion policy.

    The endpoint heads and the shared model trunk are intentionally separated
    here.  A normal :class:`DomainGatedSurfaceNet` and a scene-routed frozen
    endpoint can therefore use exactly the same final-height calculation.
    """

    if fusion_mode not in {"legacy", "protected_vegetation", "calibrated_surface"}:
        raise ValueError(
            "fusion_mode must be 'legacy', 'protected_vegetation', or "
            "'calibrated_surface'"
        )

    legacy_gated_height = (
        domain_probabilities[:, 1:2] * building_height
        + domain_probabilities[:, 2:3] * canopy_height
    )
    refinement_strength = refinement_logits.sigmoid()
    protected_building_probability = protected_building_logits.sigmoid()
    building_fusion_gate = torch.zeros_like(refinement_strength)
    vegetation_fusion_probability = domain_probabilities[:, 2:3]
    if fusion_mode in {"protected_vegetation", "calibrated_surface"}:
        learned_building_probability = domain_probabilities[:, 1:2]
        vegetation_probability = vegetation_fusion_probability
        if vegetation_fusion_temperature != 1.0:
            vegetation_probability = torch.sigmoid(
                torch.logit(vegetation_probability.clamp(1.0e-5, 1.0 - 1.0e-5))
                / vegetation_fusion_temperature
            )
            vegetation_fusion_probability = vegetation_probability
        non_building_probability = (
            (1.0 - protected_building_probability)
            * (1.0 - learned_building_probability)
        ).clamp(0.0, 1.0)
        building_protection = (
            1.0 - non_building_probability.pow(building_protection_power)
        ) * (1.0 - vegetation_probability)
        effective_refinement_strength = refinement_strength * (
            1.0 - building_protection
        )
        gated_height = (
            learned_building_probability * base_height
            + vegetation_probability * canopy_height
        )
    else:
        building_protection = torch.zeros_like(refinement_strength)
        effective_refinement_strength = refinement_strength
        gated_height = legacy_gated_height

    height = torch.clamp_min(
        base_height
        + effective_refinement_strength * (gated_height - base_height),
        0.0,
    )
    if fusion_mode == "calibrated_surface":
        building_score = (
            protected_building_probability
            * domain_probabilities[:, 1:2]
            * (1.0 - domain_probabilities[:, 2:3])
        )
        score_gate = torch.sigmoid(
            (building_score - building_fusion_score_threshold)
            / building_fusion_score_temperature
        )
        height_gate = torch.sigmoid(
            (base_height - building_fusion_min_height_m)
            / building_fusion_temperature_m
        )
        building_fusion_gate = score_gate * height_gate
        height = torch.clamp_min(
            height
            + building_fusion_strength
            * building_fusion_gate
            * (building_height - base_height),
            0.0,
        )

    vegetation_expert_fusion_gate = (
        domain_probabilities[:, 2:3] >= vegetation_expert_fusion_threshold
    ).to(height.dtype)
    if vegetation_expert_fusion_strength > 0:
        height = torch.clamp_min(
            height
            + vegetation_expert_fusion_strength
            * vegetation_expert_fusion_gate
            * (canopy_height - height),
            0.0,
        )

    return {
        "height": height,
        "gated_height": gated_height,
        "refinement_strength": refinement_strength,
        "effective_refinement_strength": effective_refinement_strength,
        "building_protection": building_protection,
        "building_fusion_gate": building_fusion_gate,
        "vegetation_fusion_probability": vegetation_fusion_probability,
        "vegetation_expert_fusion_gate": vegetation_expert_fusion_gate,
    }


class DomainGatedSurfaceNet(nn.Module):
    """Fuse mutually exclusive ground, building, and canopy predictions.

    The supplied ``base_model`` remains a complete, independently usable urban
    checkpoint. This wrapper learns residual building correction, canopy
    height, semantic gates, and uncertainty. Softmax gates sum to one at every
    pixel, preventing experts from being added on top of one another.
    """

    domain_names = ("ground", "building", "vegetation")
    fine_semantic_names = (
        "ground",
        "buildings",
        "water",
        "roads",
        "low_vegetation",
        "trees",
    )

    def __init__(
        self,
        base_model: HeightNet,
        *,
        hidden_channels: int = 48,
        maximum_building_residual_m: float = 30.0,
        initial_canopy_height_m: float = 8.0,
        initial_refinement_strength: float = 0.08,
        fusion_mode: str = "legacy",
        building_protection_power: float = 2.0,
        building_fusion_min_height_m: float = 10.0,
        building_fusion_temperature_m: float = 2.0,
        building_fusion_score_threshold: float = 0.5,
        building_fusion_score_temperature: float = 0.05,
        building_fusion_strength: float = 1.0,
        vegetation_fusion_temperature: float = 1.0,
        vegetation_expert_fusion_threshold: float = 1.0,
        vegetation_expert_fusion_strength: float = 0.0,
        fine_semantic_classes: int = 0,
        fine_semantic_head_type: str = "linear",
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if (
            hidden_channels <= 0
            or maximum_building_residual_m <= 0
            or initial_canopy_height_m <= 0
        ):
            raise ValueError("channel count and height limits must be positive")
        if not 0 < initial_refinement_strength < 1:
            raise ValueError("initial_refinement_strength must be between zero and one")
        if fusion_mode not in {"legacy", "protected_vegetation", "calibrated_surface"}:
            raise ValueError(
                "fusion_mode must be 'legacy', 'protected_vegetation', or "
                "'calibrated_surface'"
            )
        if building_protection_power <= 0:
            raise ValueError("building_protection_power must be positive")
        if building_fusion_temperature_m <= 0 or building_fusion_score_temperature <= 0:
            raise ValueError("building fusion temperatures must be positive")
        if not 0 <= building_fusion_score_threshold <= 1:
            raise ValueError("building_fusion_score_threshold must be within [0, 1]")
        if not 0 <= building_fusion_strength <= 1:
            raise ValueError("building_fusion_strength must be within [0, 1]")
        if vegetation_fusion_temperature <= 0:
            raise ValueError("vegetation_fusion_temperature must be positive")
        if not 0 <= vegetation_expert_fusion_threshold <= 1:
            raise ValueError("vegetation_expert_fusion_threshold must be within [0, 1]")
        if not 0 <= vegetation_expert_fusion_strength <= 1:
            raise ValueError("vegetation_expert_fusion_strength must be within [0, 1]")
        if fine_semantic_classes not in {0, len(self.fine_semantic_names)}:
            raise ValueError(
                "fine_semantic_classes must be 0 (disabled) or 6 (GAMUS classes)"
            )
        if fine_semantic_head_type not in {
            "linear",
            "spatial_refined",
            "hierarchical_vegetation",
        }:
            raise ValueError(
                "fine_semantic_head_type must be 'linear', 'spatial_refined', "
                "or 'hierarchical_vegetation'"
            )
        self.base_model = base_model
        self.maximum_building_residual_m = float(maximum_building_residual_m)
        self.fusion_mode = fusion_mode
        self.building_protection_power = float(building_protection_power)
        self.building_fusion_min_height_m = float(building_fusion_min_height_m)
        self.building_fusion_temperature_m = float(building_fusion_temperature_m)
        self.building_fusion_score_threshold = float(building_fusion_score_threshold)
        self.building_fusion_score_temperature = float(building_fusion_score_temperature)
        self.building_fusion_strength = float(building_fusion_strength)
        self.vegetation_fusion_temperature = float(vegetation_fusion_temperature)
        self.vegetation_expert_fusion_threshold = float(
            vegetation_expert_fusion_threshold
        )
        self.vegetation_expert_fusion_strength = float(
            vegetation_expert_fusion_strength
        )
        self.fine_semantic_classes = int(fine_semantic_classes)
        self.fine_semantic_head_type = str(fine_semantic_head_type)
        self.adapter = ConvBlock(6, hidden_channels)
        self.domain_head = nn.Conv2d(hidden_channels, 3, kernel_size=1)
        if not self.fine_semantic_classes:
            self.fine_semantic_head = None
        elif self.fine_semantic_head_type == "linear":
            self.fine_semantic_head = nn.Conv2d(
                hidden_channels, self.fine_semantic_classes, kernel_size=1
            )
        elif self.fine_semantic_head_type == "spatial_refined":
            self.fine_semantic_head = RefinedFineSemanticHead(
                hidden_channels, self.fine_semantic_classes
            )
        else:
            # Keep the global CPU RNG stream identical to the paired
            # ``spatial_refined`` control.  DataLoader worker seeds and the
            # weighted sampler are derived from that stream in the sealed V3
            # recipe, so letting a different head constructor consume a
            # different number of random values would silently give V4
            # different samples/crops despite using the same experiment seed.
            # The fork gives the hierarchical head deterministic initial
            # weights without advancing the outer stream; the discarded V3
            # head then advances that stream exactly as the comparator did.
            with torch.random.fork_rng(devices=[]):
                hierarchical_head = HierarchicalVegetationFineSemanticHead(
                    hidden_channels
                )
            self.fine_semantic_head = hierarchical_head
            RefinedFineSemanticHead(
                hidden_channels, self.fine_semantic_classes
            )
        self.building_residual_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.canopy_height_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.refinement_strength_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.log_variance_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        # Start as a conservative refinement rather than a random destructive
        # residual on top of the protected base model.
        nn.init.zeros_(self.domain_head.weight)
        nn.init.zeros_(self.domain_head.bias)
        if (
            self.fine_semantic_head is not None
            and self.fine_semantic_head_type == "linear"
        ):
            # A warm-started six-class experiment begins as an uninformative
            # auxiliary classifier.  Its existence cannot perturb height until
            # the explicitly weighted classification objective is optimized.
            nn.init.zeros_(self.fine_semantic_head.weight)
            nn.init.zeros_(self.fine_semantic_head.bias)
        nn.init.zeros_(self.building_residual_head.weight)
        nn.init.zeros_(self.building_residual_head.bias)
        nn.init.zeros_(self.canopy_height_head.weight)
        # A near-zero softplus output has a near-zero derivative and made the
        # first canopy pilot unable to escape its initialization. Start at a
        # plausible low-canopy height while still learning the full metric map.
        canopy_bias = math.log(math.expm1(float(initial_canopy_height_m)))
        nn.init.constant_(self.canopy_height_head.bias, canopy_bias)
        nn.init.zeros_(self.refinement_strength_head.weight)
        refinement_bias = math.log(
            float(initial_refinement_strength) / (1.0 - float(initial_refinement_strength))
        )
        nn.init.constant_(self.refinement_strength_head.bias, refinement_bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.zeros_(self.log_variance_head.bias)
        self.set_base_trainable(not freeze_base)

    def set_base_trainable(self, trainable: bool) -> None:
        self.base_trainable = bool(trainable)
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(self.base_trainable)
        if not self.base_trainable:
            self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.base_trainable:
            self.base_model.eval()
        return self

    def load_legacy_compatible_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Warm-start an expanded model from an older three-group checkpoint.

        Only the newly introduced six-class head may be absent.  Every legacy
        height/routing tensor remains strict, preventing an apparently
        successful warm start with missing production weights.
        """

        expected_state = self.state_dict()
        allowed_missing = (
            {
                key
                for key in expected_state
                if key.startswith("fine_semantic_head.")
            }
            if self.fine_semantic_head is not None
            else set()
        )
        supplied_keys = set(state_dict)
        expected_keys = set(expected_state)
        missing = expected_keys - supplied_keys
        unexpected = supplied_keys - expected_keys
        # A checkpoint is either a complete expanded checkpoint or a genuine
        # legacy checkpoint with the entire new head absent.  Accepting only
        # one missing head tensor would silently warm-start from a truncated or
        # corrupt expanded checkpoint.
        allowed_missing_sets = ({frozenset(), frozenset(allowed_missing)})
        shape_mismatches = {
            key: (tuple(state_dict[key].shape), tuple(expected_state[key].shape))
            for key in supplied_keys & expected_keys
            if tuple(state_dict[key].shape) != tuple(expected_state[key].shape)
        }
        if unexpected or frozenset(missing) not in allowed_missing_sets or shape_mismatches:
            raise RuntimeError(
                "Legacy-compatible surface checkpoint load failed: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
                f"shape_mismatches={shape_mismatches}"
            )
        incompatible = super().load_state_dict(state_dict, strict=False)
        return incompatible

    def forward(
        self,
        image: torch.Tensor,
        relative_prior: torch.Tensor | None = None,
        *,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
        base = (
            self.base_model(image, return_features=True)
            if return_features
            else self.base_model(image)
        )
        base_height = base["height"]
        base_building_logits = base.get("building_logits")
        if base_building_logits is None:
            base_building_logits = torch.zeros_like(base_height)
        if relative_prior is None:
            relative_prior = torch.zeros_like(base_height)
        if relative_prior.shape != base_height.shape:
            relative_prior = F.interpolate(
                relative_prior,
                size=base_height.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        normalized_base_height = torch.log1p(base_height) / 5.0
        features = self.adapter(
            torch.cat(
                (
                    image,
                    relative_prior,
                    normalized_base_height,
                    base_building_logits.sigmoid(),
                ),
                dim=1,
            )
        )

        learned_domain_residual = self.domain_head(features)
        prior_domain_logits = torch.cat(
            (
                torch.zeros_like(base_building_logits),
                base_building_logits,
                torch.full_like(base_building_logits, -2.0),
            ),
            dim=1,
        )
        domain_logits = learned_domain_residual + prior_domain_logits
        domain_probabilities = domain_logits.softmax(dim=1)

        building_residual = self.maximum_building_residual_m * torch.tanh(
            self.building_residual_head(features)
        )
        building_height = torch.clamp_min(base_height + building_residual, 0.0)
        canopy_height = F.softplus(self.canopy_height_head(features))
        # Start within ~1.8% of the protected urban checkpoint. The adapter can
        # earn a larger local correction from supervised data, but random/new
        # domain heads cannot abruptly replace a working urban result.
        refinement_logits = self.refinement_strength_head(features)
        fusion = fuse_surface_endpoint(
            base_height=base_height,
            protected_building_logits=base_building_logits,
            domain_probabilities=domain_probabilities,
            building_height=building_height,
            canopy_height=canopy_height,
            refinement_logits=refinement_logits,
            fusion_mode=self.fusion_mode,
            building_protection_power=self.building_protection_power,
            building_fusion_min_height_m=self.building_fusion_min_height_m,
            building_fusion_temperature_m=self.building_fusion_temperature_m,
            building_fusion_score_threshold=self.building_fusion_score_threshold,
            building_fusion_score_temperature=self.building_fusion_score_temperature,
            building_fusion_strength=self.building_fusion_strength,
            vegetation_fusion_temperature=self.vegetation_fusion_temperature,
            vegetation_expert_fusion_threshold=self.vegetation_expert_fusion_threshold,
            vegetation_expert_fusion_strength=self.vegetation_expert_fusion_strength,
        )
        log_variance = self.log_variance_head(features).clamp(-6.0, 6.0)
        result = {
            **fusion,
            # Stage-2 semantic repair snapshots only the small domain head and
            # reuses these frozen features/priors for inexpensive distillation.
            "adapter_features": features,
            "prior_domain_logits": prior_domain_logits,
            "building_height": building_height,
            "canopy_height": canopy_height,
            "refinement_logits": refinement_logits,
            "domain_logits": domain_logits,
            "domain_probabilities": domain_probabilities,
            "building_logits": domain_logits[:, 1:2] - torch.logsumexp(
                domain_logits[:, (0, 2)], dim=1, keepdim=True
            ),
            # Preserve the proven urban detector for application-level
            # footprints. The learned domain logits remain available above for
            # training/evaluation, but must not silently replace this expert.
            "protected_building_logits": base_building_logits,
            "vegetation_logits": domain_logits[:, 2:3] - torch.logsumexp(
                domain_logits[:, :2], dim=1, keepdim=True
            ),
            "log_variance": log_variance,
            "base_height": base_height,
        }
        if self.fine_semantic_head is not None:
            if isinstance(
                self.fine_semantic_head,
                HierarchicalVegetationFineSemanticHead,
            ):
                fine_semantic = self.fine_semantic_head.forward_with_components(
                    features
                )
                result["fine_semantic_logits"] = fine_semantic["logits"]
                result["fine_semantic_probabilities"] = fine_semantic[
                    "probabilities"
                ]
                result["fine_semantic_coarse_logits"] = fine_semantic[
                    "coarse_logits"
                ]
                result["fine_semantic_coarse_probabilities"] = fine_semantic[
                    "coarse_probabilities"
                ]
                result["fine_semantic_vegetation_split_logits"] = fine_semantic[
                    "vegetation_split_logits"
                ]
            else:
                fine_semantic_logits = self.fine_semantic_head(features)
                result["fine_semantic_logits"] = fine_semantic_logits
                result["fine_semantic_probabilities"] = fine_semantic_logits.softmax(
                    dim=1
                )
        if return_features:
            result["encoder_features"] = base["encoder_features"]
            result["decoded_features"] = base["decoded_features"]
        return result
