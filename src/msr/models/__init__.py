"""Height-estimation model registry."""

from .domain_surface_net import DomainGatedSurfaceNet, fuse_surface_endpoint
from .height_net import HeightNet
from .routed_surface import (
    ConservativeSceneRouter,
    FrozenDualSurfaceHeadRouter,
    FrozenSurfaceHeadPack,
    RoutedDomainGatedSurfaceNet,
    select_scene_endpoint,
)
from .stage3_endpoints import (
    BaseCheckpointDiagnostics,
    EndpointCheckpointDiagnostics,
    EndpointCompatibilityDiagnostics,
    Stage3CheckpointValidationError,
    Stage3EndpointDiagnostics,
    Stage3EndpointPacks,
    load_stage3_endpoint_packs,
    load_stage3_endpoints,
)

__all__ = [
    "BaseCheckpointDiagnostics",
    "ConservativeSceneRouter",
    "DomainGatedSurfaceNet",
    "EndpointCheckpointDiagnostics",
    "EndpointCompatibilityDiagnostics",
    "FrozenDualSurfaceHeadRouter",
    "FrozenSurfaceHeadPack",
    "HeightNet",
    "RoutedDomainGatedSurfaceNet",
    "Stage3CheckpointValidationError",
    "Stage3EndpointDiagnostics",
    "Stage3EndpointPacks",
    "fuse_surface_endpoint",
    "load_stage3_endpoint_packs",
    "load_stage3_endpoints",
    "select_scene_endpoint",
]
