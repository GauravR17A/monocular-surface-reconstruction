"""Frozen surface-head endpoints and conservative scene-level routing.

This module provides the small, isolated core needed for a guarded two-endpoint
surface model.  It deliberately does not alter :class:`DomainGatedSurfaceNet`
or choose a runtime checkpoint.  Existing checkpoints therefore retain their
exact loading and inference behaviour until a routed model is explicitly wired
and validated.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .domain_surface_net import fuse_surface_endpoint


_FUSION_DEFAULTS: dict[str, object] = {
    "fusion_mode": "legacy",
    "building_protection_power": 2.0,
    "building_fusion_min_height_m": 10.0,
    "building_fusion_temperature_m": 2.0,
    "building_fusion_score_threshold": 0.5,
    "building_fusion_score_temperature": 0.05,
    "building_fusion_strength": 1.0,
    "vegetation_fusion_temperature": 1.0,
    "vegetation_expert_fusion_threshold": 1.0,
    "vegetation_expert_fusion_strength": 0.0,
}


class FrozenSurfaceHeadPack(nn.Module):
    """Immutable copy of one complete three-head surface endpoint.

    The semantic-domain, building-residual, and canopy-height heads are the
    only tensors that differ between the protected and GAMUS Stage-1 endpoint
    checkpoints.  Keeping them together prevents a semantic decision from
    being paired with height experts from a different endpoint.
    """

    def __init__(
        self,
        domain_head: nn.Module,
        building_residual_head: nn.Module,
        canopy_height_head: nn.Module,
        *,
        maximum_building_residual_m: float,
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
    ) -> None:
        super().__init__()
        if maximum_building_residual_m <= 0:
            raise ValueError("maximum_building_residual_m must be positive")
        expected_outputs = {
            "domain_head": 3,
            "building_residual_head": 1,
            "canopy_height_head": 1,
        }
        feature_channels: int | None = None
        for name, head in (
            ("domain_head", domain_head),
            ("building_residual_head", building_residual_head),
            ("canopy_height_head", canopy_height_head),
        ):
            if not isinstance(head, nn.Conv2d) or head.kernel_size != (1, 1):
                raise TypeError(f"{name} must be a 1x1 Conv2d")
            if head.out_channels != expected_outputs[name]:
                raise ValueError(
                    f"{name} must emit {expected_outputs[name]} channel(s)"
                )
            if feature_channels is None:
                feature_channels = head.in_channels
            elif head.in_channels != feature_channels:
                raise ValueError("all endpoint heads must share their input channels")
        assert feature_channels is not None
        self.feature_channels = int(feature_channels)
        self.domain_head = deepcopy(domain_head)
        self.building_residual_head = deepcopy(building_residual_head)
        self.canopy_height_head = deepcopy(canopy_height_head)
        self.maximum_building_residual_m = float(maximum_building_residual_m)
        self.fusion_mode = str(fusion_mode)
        self.building_protection_power = float(building_protection_power)
        self.building_fusion_min_height_m = float(building_fusion_min_height_m)
        self.building_fusion_temperature_m = float(building_fusion_temperature_m)
        self.building_fusion_score_threshold = float(
            building_fusion_score_threshold
        )
        self.building_fusion_score_temperature = float(
            building_fusion_score_temperature
        )
        self.building_fusion_strength = float(building_fusion_strength)
        self.vegetation_fusion_temperature = float(vegetation_fusion_temperature)
        self.vegetation_expert_fusion_threshold = float(
            vegetation_expert_fusion_threshold
        )
        self.vegetation_expert_fusion_strength = float(
            vegetation_expert_fusion_strength
        )
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_model(cls, model: nn.Module) -> "FrozenSurfaceHeadPack":
        """Snapshot the three endpoint heads from a compatible surface model."""

        required = (
            "domain_head",
            "building_residual_head",
            "canopy_height_head",
            "maximum_building_residual_m",
        )
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise TypeError(
                "Surface model does not expose the complete three-head endpoint: "
                f"{missing}"
            )
        fusion_settings = {
            name: getattr(model, name, default)
            for name, default in _FUSION_DEFAULTS.items()
        }
        return cls(
            model.domain_head,
            model.building_residual_head,
            model.canopy_height_head,
            maximum_building_residual_m=float(
                model.maximum_building_residual_m
            ),
            **fusion_settings,
        )

    def train(self, mode: bool = True):
        """Keep a frozen endpoint deterministic inside a training wrapper."""

        super().train(False)
        return self

    def forward(
        self,
        adapter_features: torch.Tensor,
        prior_domain_logits: torch.Tensor,
        base_height: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate all three heads using already-computed shared features."""

        if adapter_features.ndim != 4:
            raise ValueError("adapter_features must have shape (batch, channels, H, W)")
        if prior_domain_logits.ndim != 4 or prior_domain_logits.shape[1] != 3:
            raise ValueError(
                "prior_domain_logits must have shape (batch, 3, H, W)"
            )
        if base_height.ndim != 4 or base_height.shape[1] != 1:
            raise ValueError("base_height must have shape (batch, 1, H, W)")
        if (
            adapter_features.shape[0] != base_height.shape[0]
            or adapter_features.shape[-2:] != base_height.shape[-2:]
            or prior_domain_logits.shape[0] != base_height.shape[0]
            or prior_domain_logits.shape[-2:] != base_height.shape[-2:]
        ):
            raise ValueError("endpoint inputs must share batch and spatial dimensions")

        # A frozen endpoint must not become an accidental gradient path into
        # the shared adapter.  The separate router still learns from the same
        # values through its own trainable layers.
        adapter_features = adapter_features.detach()
        prior_domain_logits = prior_domain_logits.detach()
        base_height = base_height.detach()

        learned_domain_residual = self.domain_head(adapter_features)
        if learned_domain_residual.shape != prior_domain_logits.shape:
            raise ValueError(
                "domain head output must match the three-channel prior logits"
            )
        domain_logits = learned_domain_residual + prior_domain_logits
        domain_probabilities = domain_logits.softmax(dim=1)

        building_residual = self.maximum_building_residual_m * torch.tanh(
            self.building_residual_head(adapter_features)
        )
        canopy_height = F.softplus(self.canopy_height_head(adapter_features))
        if building_residual.shape != base_height.shape:
            raise ValueError("building residual head output must match base_height")
        if canopy_height.shape != base_height.shape:
            raise ValueError("canopy height head output must match base_height")
        building_height = torch.clamp_min(base_height + building_residual, 0.0)

        return {
            "learned_domain_residual": learned_domain_residual,
            "domain_logits": domain_logits,
            "domain_probabilities": domain_probabilities,
            "building_residual": building_residual,
            "building_height": building_height,
            "canopy_height": canopy_height,
            "building_logits": domain_logits[:, 1:2]
            - torch.logsumexp(domain_logits[:, (0, 2)], dim=1, keepdim=True),
            "vegetation_logits": domain_logits[:, 2:3]
            - torch.logsumexp(domain_logits[:, :2], dim=1, keepdim=True),
        }

    def forward_fused(
        self,
        adapter_features: torch.Tensor,
        prior_domain_logits: torch.Tensor,
        base_height: torch.Tensor,
        *,
        protected_building_logits: torch.Tensor,
        refinement_logits: torch.Tensor,
        log_variance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reproduce this endpoint's complete final surface-model output.

        The adapter, protected detector, refinement head, and uncertainty head
        are common to both frozen checkpoints.  This pack supplies its atomic
        semantic/building/canopy endpoint and its snapshotted fusion policy.
        """

        raw = self(adapter_features, prior_domain_logits, base_height)
        for name, value in (
            ("protected_building_logits", protected_building_logits),
            ("refinement_logits", refinement_logits),
        ):
            if value.shape != base_height.shape:
                raise ValueError(f"{name} must match base_height")
        if log_variance is not None and log_variance.shape != base_height.shape:
            raise ValueError("log_variance must match base_height")

        fusion = fuse_surface_endpoint(
            base_height=base_height.detach(),
            protected_building_logits=protected_building_logits.detach(),
            domain_probabilities=raw["domain_probabilities"],
            building_height=raw["building_height"],
            canopy_height=raw["canopy_height"],
            refinement_logits=refinement_logits.detach(),
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
        complete = {
            **raw,
            **fusion,
            "refinement_logits": refinement_logits.detach(),
            "protected_building_logits": protected_building_logits.detach(),
            "base_height": base_height.detach(),
        }
        if log_variance is not None:
            complete["log_variance"] = log_variance.detach()
        return complete


class ConservativeSceneRouter(nn.Module):
    """Tiny global router whose untrained state always chooses the fallback.

    The router summarizes a scene with the per-channel mean and standard
    deviation of frozen adapter features.  It emits one decision per image,
    never one decision per pixel, which prevents hard endpoint seams within an
    image.  A high threshold and low candidate prior make an untrained or
    uncertain router retain the protected endpoint.
    """

    def __init__(
        self,
        feature_channels: int,
        *,
        hidden_features: int = 16,
        decision_threshold: float = 0.90,
        initial_candidate_probability: float = 0.01,
    ) -> None:
        super().__init__()
        if feature_channels <= 0 or hidden_features <= 0:
            raise ValueError("router feature sizes must be positive")
        if not 0.0 < decision_threshold < 1.0:
            raise ValueError("decision_threshold must be strictly within (0, 1)")
        if not 0.0 < initial_candidate_probability < decision_threshold:
            raise ValueError(
                "initial_candidate_probability must be positive and below the "
                "decision threshold"
            )
        self.feature_channels = int(feature_channels)
        self.hidden_features = int(hidden_features)
        self.register_buffer(
            "decision_threshold",
            torch.tensor(float(decision_threshold), dtype=torch.float32),
        )
        self.network = nn.Sequential(
            nn.Linear(2 * self.feature_channels, self.hidden_features),
            nn.GELU(),
            nn.Linear(self.hidden_features, 1),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        prior_logit = math.log(
            initial_candidate_probability / (1.0 - initial_candidate_probability)
        )
        nn.init.constant_(final.bias, prior_logit)

    def _scene_descriptor(
        self,
        adapter_features: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if adapter_features.ndim != 4:
            raise ValueError("adapter_features must have shape (batch, channels, H, W)")
        if adapter_features.shape[1] != self.feature_channels:
            raise ValueError(
                "adapter feature channel mismatch: expected "
                f"{self.feature_channels}, found {adapter_features.shape[1]}"
            )
        if valid_mask is None:
            mean = adapter_features.mean(dim=(-2, -1))
            variance = torch.mean(
                torch.square(adapter_features - mean[:, :, None, None]),
                dim=(-2, -1),
            )
        else:
            if valid_mask.ndim == 3:
                valid_mask = valid_mask[:, None]
            if (
                valid_mask.ndim != 4
                or valid_mask.shape[1] != 1
                or valid_mask.shape[0] != adapter_features.shape[0]
                or valid_mask.shape[-2:] != adapter_features.shape[-2:]
            ):
                raise ValueError(
                    "valid_mask must have shape (batch, 1, H, W) or (batch, H, W)"
                )
            mask = valid_mask.to(dtype=adapter_features.dtype)
            count = mask.sum(dim=(-2, -1))
            if bool(torch.any(count <= 0)):
                raise ValueError("every routed scene must contain at least one valid pixel")
            mean = (adapter_features * mask).sum(dim=(-2, -1)) / count
            variance = (
                torch.square(adapter_features - mean[:, :, None, None]) * mask
            ).sum(dim=(-2, -1)) / count
        standard_deviation = torch.sqrt(variance.clamp_min(0.0))
        return torch.cat((mean, standard_deviation), dim=1)

    def forward(
        self,
        adapter_features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Endpoint inference may run under bf16/fp16 autocast, but a rounded
        # routing threshold can turn an actually-sub-threshold score into a
        # candidate selection.  Keep the complete decision path in float32 so
        # the conservative fail-safe has the same meaning at every precision.
        with torch.autocast(
            device_type=adapter_features.device.type,
            enabled=False,
        ):
            descriptor = self._scene_descriptor(adapter_features.float(), valid_mask)
            candidate_logit = self.network(descriptor)
            candidate_probability = candidate_logit.sigmoid()
        threshold = self.decision_threshold.to(
            device=candidate_probability.device,
            dtype=torch.float32,
        )
        candidate_selected = candidate_probability >= threshold
        return {
            "candidate_logit": candidate_logit,
            "candidate_probability": candidate_probability,
            "candidate_selected": candidate_selected,
            "scene_descriptor": descriptor,
        }


def select_scene_endpoint(
    fallback: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    candidate_selected: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Choose one complete endpoint per batch item without spatial mixing."""

    if set(fallback) != set(candidate):
        missing = sorted(set(fallback) - set(candidate))
        extra = sorted(set(candidate) - set(fallback))
        raise ValueError(
            f"endpoint output keys differ; missing={missing}, extra={extra}"
        )
    if candidate_selected.ndim == 1:
        candidate_selected = candidate_selected[:, None]
    if candidate_selected.ndim != 2 or candidate_selected.shape[1] != 1:
        raise ValueError("candidate_selected must have shape (batch,) or (batch, 1)")

    selected: dict[str, torch.Tensor] = {}
    for name in fallback:
        left = fallback[name]
        right = candidate[name]
        if left.shape != right.shape:
            raise ValueError(f"endpoint tensor shape differs for {name!r}")
        if left.ndim < 1 or left.shape[0] != candidate_selected.shape[0]:
            raise ValueError(f"endpoint tensor {name!r} has no matching batch dimension")
        selection = candidate_selected.reshape(
            candidate_selected.shape[0], *([1] * (left.ndim - 1))
        )
        selected[name] = torch.where(selection, right, left)
    return selected


class FrozenDualSurfaceHeadRouter(nn.Module):
    """Evaluate two frozen head packs and conservatively select one per scene."""

    def __init__(
        self,
        fallback: FrozenSurfaceHeadPack,
        candidate: FrozenSurfaceHeadPack,
        router: ConservativeSceneRouter,
    ) -> None:
        super().__init__()
        if not math.isclose(
            fallback.maximum_building_residual_m,
            candidate.maximum_building_residual_m,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "fallback and candidate endpoints must share the building residual limit"
            )
        if fallback.feature_channels != candidate.feature_channels:
            raise ValueError("fallback and candidate endpoints must share feature channels")
        if router.feature_channels != fallback.feature_channels:
            raise ValueError("router feature channels must match both endpoint packs")
        for head_name in (
            "domain_head",
            "building_residual_head",
            "canopy_height_head",
        ):
            fallback_state = getattr(fallback, head_name).state_dict()
            candidate_state = getattr(candidate, head_name).state_dict()
            if set(fallback_state) != set(candidate_state) or any(
                fallback_state[name].shape != candidate_state[name].shape
                or fallback_state[name].dtype != candidate_state[name].dtype
                for name in fallback_state
            ):
                raise ValueError(
                    f"fallback and candidate {head_name} architectures do not match"
                )
        self.fallback = fallback
        self.candidate = candidate
        self.router = router
        self.fallback.requires_grad_(False)
        self.candidate.requires_grad_(False)
        self.fallback.eval()
        self.candidate.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.fallback.eval()
        self.candidate.eval()
        return self

    def forward(
        self,
        adapter_features: torch.Tensor,
        prior_domain_logits: torch.Tensor,
        base_height: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        adapter_features = adapter_features.detach()
        prior_domain_logits = prior_domain_logits.detach()
        base_height = base_height.detach()
        fallback = self.fallback(
            adapter_features, prior_domain_logits, base_height
        )
        candidate = self.candidate(
            adapter_features, prior_domain_logits, base_height
        )
        route = self.router(adapter_features, valid_mask)
        selected = select_scene_endpoint(
            fallback, candidate, route["candidate_selected"]
        )
        return {
            **selected,
            "route_candidate_logit": route["candidate_logit"],
            "route_candidate_probability": route["candidate_probability"],
            "route_candidate_selected": route["candidate_selected"],
            "route_scene_descriptor": route["scene_descriptor"],
        }

    def forward_fused(
        self,
        adapter_features: torch.Tensor,
        prior_domain_logits: torch.Tensor,
        base_height: torch.Tensor,
        *,
        protected_building_logits: torch.Tensor,
        refinement_logits: torch.Tensor,
        log_variance: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Select one endpoint per scene after applying each full fusion policy."""

        adapter_features = adapter_features.detach()
        prior_domain_logits = prior_domain_logits.detach()
        base_height = base_height.detach()
        protected_building_logits = protected_building_logits.detach()
        refinement_logits = refinement_logits.detach()
        if log_variance is not None:
            log_variance = log_variance.detach()
        common = {
            "protected_building_logits": protected_building_logits,
            "refinement_logits": refinement_logits,
            "log_variance": log_variance,
        }
        fallback = self.fallback.forward_fused(
            adapter_features,
            prior_domain_logits,
            base_height,
            **common,
        )
        candidate = self.candidate.forward_fused(
            adapter_features,
            prior_domain_logits,
            base_height,
            **common,
        )
        route = self.router(adapter_features, valid_mask)
        selected = select_scene_endpoint(
            fallback, candidate, route["candidate_selected"]
        )
        return {
            **selected,
            "route_candidate_logit": route["candidate_logit"],
            "route_candidate_probability": route["candidate_probability"],
            "route_candidate_selected": route["candidate_selected"],
            "route_scene_descriptor": route["scene_descriptor"],
        }


class RoutedDomainGatedSurfaceNet(nn.Module):
    """Whole-scene model that routes between two complete frozen endpoints.

    ``shared_model`` supplies the unchanged base model, adapter, refinement,
    and uncertainty tensors.  Only ``endpoint_router.router`` is trainable.
    Existing :class:`DomainGatedSurfaceNet` inference remains untouched unless
    this explicit wrapper is constructed.
    """

    def __init__(
        self,
        shared_model: nn.Module,
        endpoint_router: FrozenDualSurfaceHeadRouter,
    ) -> None:
        super().__init__()
        required = (
            "adapter",
            "domain_head",
            "refinement_strength_head",
            "log_variance_head",
        )
        missing = [name for name in required if not hasattr(shared_model, name)]
        if missing:
            raise TypeError(
                "shared_model is not a compatible surface model; missing "
                f"{missing}"
            )
        shared_feature_channels = int(shared_model.domain_head.in_channels)
        if shared_feature_channels != endpoint_router.fallback.feature_channels:
            raise ValueError(
                "shared model adapter channels must match the routed endpoints"
            )
        self.shared_model = shared_model
        self.endpoint_router = endpoint_router
        self.shared_model.requires_grad_(False)
        self.shared_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.shared_model.eval()
        return self

    def forward(
        self,
        image: torch.Tensor,
        relative_prior: torch.Tensor | None = None,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Reusing the normal forward guarantees checkpoint compatibility.  Its
        # endpoint result is then atomically replaced by the routed result.
        with torch.no_grad():
            shared = self.shared_model(image, relative_prior)
        required = (
            "adapter_features",
            "prior_domain_logits",
            "base_height",
            "protected_building_logits",
            "refinement_logits",
            "log_variance",
        )
        missing = [name for name in required if name not in shared]
        if missing:
            raise RuntimeError(
                "shared surface model did not expose routed inputs: "
                f"{missing}"
            )
        routed = self.endpoint_router.forward_fused(
            shared["adapter_features"],
            shared["prior_domain_logits"],
            shared["base_height"],
            protected_building_logits=shared["protected_building_logits"],
            refinement_logits=shared["refinement_logits"],
            log_variance=shared["log_variance"],
            valid_mask=valid_mask,
        )
        return {
            **shared,
            **routed,
        }


__all__ = [
    "ConservativeSceneRouter",
    "FrozenDualSurfaceHeadRouter",
    "FrozenSurfaceHeadPack",
    "RoutedDomainGatedSurfaceNet",
    "select_scene_endpoint",
]
