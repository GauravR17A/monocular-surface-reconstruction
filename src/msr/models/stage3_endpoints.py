"""Fail-closed loading for the two frozen Stage-3 surface endpoints.

Stage 3 routes whole scenes between the protected application endpoint and the
GAMUS Stage-1 endpoint. That is safe only when the checkpoints share exactly
the same trunk and differ solely in the semantic-domain, building-residual,
and canopy-height heads. This module verifies that invariant before exposing
independent :class:`FrozenSurfaceHeadPack` snapshots.

The loader intentionally does not construct a routed model, update a runtime
pointer, or select an endpoint. It only validates and extracts immutable head
packs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .domain_surface_net import DomainGatedSurfaceNet
from .height_net import HeightNet
from .routed_surface import FrozenSurfaceHeadPack


_SUPPORTED_MODEL_TYPES = frozenset(
    {
        "domain_gated_surface_v1",
        "domain_gated_surface_v2",
        "domain_gated_surface_v3",
    }
)
_HEAD_PREFIXES = (
    "domain_head.",
    "building_residual_head.",
    "canopy_height_head.",
)
_HEAD_TENSOR_KEYS = frozenset(
    f"{prefix}{suffix}"
    for prefix in _HEAD_PREFIXES
    for suffix in ("weight", "bias")
)
_NON_ARCHITECTURE_MODEL_CONFIG_KEYS = frozenset({"initial_checkpoint"})
_FUSION_POLICY_DEFAULTS: dict[str, object] = {
    "maximum_building_residual_m": 30.0,
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


class Stage3CheckpointValidationError(ValueError):
    """Raised whenever a checkpoint pair cannot be proven routing-safe."""


@dataclass(frozen=True)
class EndpointCheckpointDiagnostics:
    """Immutable provenance and state hashes for one endpoint checkpoint."""

    role: str
    path: str
    sha256: str
    file_size_bytes: int
    epoch: int | None
    model_type: str
    model_tensor_count: int
    model_numel: int
    model_state_sha256: str
    head_state_sha256: str
    architecture_config_sha256: str


@dataclass(frozen=True)
class EndpointCompatibilityDiagnostics:
    """Evidence that the two endpoint trunks and architectures are identical."""

    shared_tensor_count: int
    shared_numel: int
    shared_state_sha256: str
    differing_head_tensors: tuple[str, ...]
    identical_head_tensors: tuple[str, ...]
    allowed_head_tensors: tuple[str, ...]
    architecture_config_items: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BaseCheckpointDiagnostics:
    """Immutable provenance for the base checkpoint used to rebuild the trunk."""

    path: str
    sha256: str
    file_size_bytes: int
    model_type: str
    model_tensor_count: int
    model_numel: int
    model_state_sha256: str
    architecture_config_sha256: str


@dataclass(frozen=True)
class Stage3EndpointDiagnostics:
    """Complete validation diagnostics for a Stage-3 endpoint pair."""

    protected: EndpointCheckpointDiagnostics
    gamus_stage1: EndpointCheckpointDiagnostics
    compatibility: EndpointCompatibilityDiagnostics
    base_checkpoint: BaseCheckpointDiagnostics | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy suitable for an audit artifact."""

        return asdict(self)


@dataclass(frozen=True)
class Stage3EndpointPacks:
    """Two validated, independently frozen endpoint head packs."""

    protected: FrozenSurfaceHeadPack
    gamus_stage1: FrozenSurfaceHeadPack
    shared_model: DomainGatedSurfaceNet | None
    diagnostics: Stage3EndpointDiagnostics

    @property
    def protected_pack(self) -> FrozenSurfaceHeadPack:
        """Explicit alias useful at routed-model call sites."""

        return self.protected

    @property
    def gamus_stage1_pack(self) -> FrozenSurfaceHeadPack:
        """Explicit alias useful at routed-model call sites."""

        return self.gamus_stage1

    def require_shared_model(self) -> DomainGatedSurfaceNet:
        """Return the verified shared model or fail if reconstruction was disabled."""

        if self.shared_model is None:
            raise Stage3CheckpointValidationError(
                "This endpoint bundle was loaded without shared-model reconstruction"
            )
        return self.shared_model


@dataclass(frozen=True)
class _LoadedCheckpoint:
    path: Path
    sha256: str
    size_bytes: int
    payload: Mapping[str, Any]
    model_type: str
    model_config: Mapping[str, Any]
    architecture_config: Mapping[str, Any]
    state: Mapping[str, torch.Tensor]
    epoch: int | None


@dataclass(frozen=True)
class _LoadedBaseCheckpoint:
    path: Path
    sha256: str
    size_bytes: int
    model_type: str
    model_config: Mapping[str, Any]
    state: Mapping[str, torch.Tensor]


def _sha256_file(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.layout != torch.strided:
        raise Stage3CheckpointValidationError(
            f"Only strided checkpoint tensors are supported, found {tensor.layout}"
        )
    value = tensor.detach().cpu().contiguous().reshape(-1)
    return value.view(torch.uint8).numpy().tobytes()


def _update_tensor_hash(digest: Any, name: str, tensor: torch.Tensor) -> None:
    metadata = json.dumps(
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(metadata).to_bytes(8, byteorder="big"))
    digest.update(metadata)
    raw = _tensor_bytes(tensor)
    digest.update(len(raw).to_bytes(8, byteorder="big"))
    digest.update(raw)


def _state_sha256(
    state: Mapping[str, torch.Tensor], names: tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        _update_tensor_hash(digest, name, state[name])
    return digest.hexdigest()


def _is_bit_identical(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return _tensor_bytes(left) == _tensor_bytes(right)


def _canonical_config_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Stage3CheckpointValidationError(
                "Architecture config contains a non-finite floating-point value"
            )
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise Stage3CheckpointValidationError(
                "Architecture config mapping keys must be strings"
            )
        return {
            key: _canonical_config_value(value[key]) for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    raise Stage3CheckpointValidationError(
        "Architecture config contains an unsupported value of type "
        f"{type(value).__name__}"
    )


def _architecture_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _canonical_config_value(value)
        for key, value in sorted(model_config.items())
        if key not in _NON_ARCHITECTURE_MODEL_CONFIG_KEYS
    }


def _config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_config_value(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_weights_load(path: Path) -> Mapping[str, Any]:
    """Load primitive checkpoint data with PyTorch's restricted unpickler."""

    try:
        from numpy.core.multiarray import _reconstruct

        # Historical checkpoints contain NumPy RNG state. These four data
        # container types decode it while arbitrary checkpoint globals remain
        # forbidden by the weights-only unpickler.
        safe_globals = [
            _reconstruct,
            np.ndarray,
            np.dtype,
            type(np.dtype(np.uint32)),
        ]
        with torch.serialization.safe_globals(safe_globals):
            payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise Stage3CheckpointValidationError(
            f"Could not safely load checkpoint {path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise Stage3CheckpointValidationError(
            f"Checkpoint {path} payload must be a mapping"
        )
    return payload


def _validate_expected_hash(role: str, actual: str, expected: str | None) -> None:
    if expected is None:
        return
    normalized = expected.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise Stage3CheckpointValidationError(
            f"Expected {role} SHA-256 is not a 64-character hexadecimal digest"
        )
    if actual != normalized:
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint SHA-256 mismatch: expected {normalized}, "
            f"found {actual}"
        )


def _load_checkpoint(
    path_like: str | Path,
    *,
    role: str,
    expected_sha256: str | None,
) -> _LoadedCheckpoint:
    try:
        path = Path(path_like).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint is unavailable: {path_like}"
        ) from error
    if not path.is_file():
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint is not a regular file: {path}"
        )

    stat_before = path.stat()
    hash_before = _sha256_file(path)
    _validate_expected_hash(role, hash_before, expected_sha256)
    payload = _safe_weights_load(path)
    stat_after = path.stat()
    hash_after = _sha256_file(path)
    if (
        hash_before != hash_after
        or stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint changed while it was being loaded"
        )

    model_type = payload.get("model_type")
    if not isinstance(model_type, str) or model_type not in _SUPPORTED_MODEL_TYPES:
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint has unsupported model_type {model_type!r}"
        )
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint is missing a config mapping"
        )
    model_config = config.get("model")
    if not isinstance(model_config, Mapping) or not all(
        isinstance(key, str) for key in model_config
    ):
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint is missing a valid config.model mapping"
        )
    architecture_config = _architecture_config(model_config)

    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint is missing a non-empty model state mapping"
        )
    if not all(isinstance(name, str) for name in state):
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint model-state keys must be strings"
        )
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint model state may contain tensors only"
        )
    for name, tensor in state.items():
        if tensor.layout != torch.strided:
            raise Stage3CheckpointValidationError(
                f"{role} tensor {name!r} has unsupported layout {tensor.layout}"
            )
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise Stage3CheckpointValidationError(
                f"{role} tensor {name!r} contains non-finite values"
            )

    found_head_keys = frozenset(
        name for name in state if name.startswith(_HEAD_PREFIXES)
    )
    if found_head_keys != _HEAD_TENSOR_KEYS:
        missing = sorted(_HEAD_TENSOR_KEYS - found_head_keys)
        extra = sorted(found_head_keys - _HEAD_TENSOR_KEYS)
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint does not expose exactly the required three heads; "
            f"missing={missing}, extra={extra}"
        )

    epoch_value = payload.get("epoch")
    if epoch_value is not None and (
        isinstance(epoch_value, bool) or not isinstance(epoch_value, int)
    ):
        raise Stage3CheckpointValidationError(
            f"{role} checkpoint epoch must be an integer or null"
        )

    return _LoadedCheckpoint(
        path=path,
        sha256=hash_before,
        size_bytes=int(stat_before.st_size),
        payload=payload,
        model_type=model_type,
        model_config=model_config,
        architecture_config=architecture_config,
        state=state,
        epoch=epoch_value,
    )


def _resolve_base_checkpoint_path(
    reference: Any,
    *,
    protected_path: Path,
    project_root: str | Path | None,
) -> Path:
    if not isinstance(reference, (str, Path)) or not str(reference).strip():
        raise Stage3CheckpointValidationError(
            "Protected checkpoint config.model.base_checkpoint must be a path"
        )
    source = Path(reference).expanduser()
    if source.is_absolute():
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise Stage3CheckpointValidationError(
                f"Referenced base checkpoint is unavailable: {source}"
            ) from error
        if not resolved.is_file():
            raise Stage3CheckpointValidationError(
                f"Referenced base checkpoint is not a regular file: {resolved}"
            )
        return resolved

    roots: list[Path] = []
    if project_root is not None:
        roots.append(Path(project_root).expanduser())
    roots.append(Path.cwd())
    roots.extend(protected_path.parents)
    matches: dict[str, Path] = {}
    for root in roots:
        candidate = (root / source).resolve()
        if candidate.is_file():
            matches[str(candidate).casefold()] = candidate
    if not matches:
        raise Stage3CheckpointValidationError(
            "Could not resolve the protected checkpoint's relative base checkpoint "
            f"reference {source!s}"
        )
    if len(matches) != 1:
        raise Stage3CheckpointValidationError(
            "Relative base checkpoint reference is ambiguous: "
            f"{tuple(str(path) for path in matches.values())}"
        )
    return next(iter(matches.values()))


def _load_base_checkpoint(
    path: Path, *, expected_sha256: str | None
) -> _LoadedBaseCheckpoint:
    stat_before = path.stat()
    hash_before = _sha256_file(path)
    _validate_expected_hash("base", hash_before, expected_sha256)
    payload = _safe_weights_load(path)
    stat_after = path.stat()
    hash_after = _sha256_file(path)
    if (
        hash_before != hash_after
        or stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise Stage3CheckpointValidationError(
            "Base checkpoint changed while it was being loaded"
        )

    model_type_value = payload.get("model_type", "height_net_v1")
    if model_type_value is None:
        model_type_value = "height_net_v1"
    if model_type_value != "height_net_v1":
        raise Stage3CheckpointValidationError(
            f"Base checkpoint must contain HeightNet, found {model_type_value!r}"
        )
    config = payload.get("config")
    model_config = config.get("model") if isinstance(config, Mapping) else None
    if not isinstance(model_config, Mapping) or not all(
        isinstance(name, str) for name in model_config
    ):
        raise Stage3CheckpointValidationError(
            "Base checkpoint is missing a valid config.model mapping"
        )
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise Stage3CheckpointValidationError(
            "Base checkpoint is missing a non-empty model state mapping"
        )
    if not all(isinstance(name, str) for name in state) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise Stage3CheckpointValidationError(
            "Base checkpoint model state must be a string-to-tensor mapping"
        )
    for name, tensor in state.items():
        if tensor.layout != torch.strided:
            raise Stage3CheckpointValidationError(
                f"Base tensor {name!r} has unsupported layout {tensor.layout}"
            )
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise Stage3CheckpointValidationError(
                f"Base tensor {name!r} contains non-finite values"
            )
    return _LoadedBaseCheckpoint(
        path=path,
        sha256=hash_before,
        size_bytes=int(stat_before.st_size),
        model_type="height_net_v1",
        model_config=model_config,
        state=state,
    )


def _numeric_model_setting(
    config: Mapping[str, Any],
    name: str,
    default: float,
    *,
    role: str,
) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage3CheckpointValidationError(f"{role} {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Stage3CheckpointValidationError(f"{role} {name} must be finite")
    return result


def _reconstruct_shared_model(
    protected: _LoadedCheckpoint,
    *,
    project_root: str | Path | None,
    expected_base_sha256: str | None,
) -> tuple[DomainGatedSurfaceNet, BaseCheckpointDiagnostics]:
    reference = protected.architecture_config.get("base_checkpoint")
    base_path = _resolve_base_checkpoint_path(
        reference,
        protected_path=protected.path,
        project_root=project_root,
    )
    base = _load_base_checkpoint(base_path, expected_sha256=expected_base_sha256)

    embedded_base = {
        name.removeprefix("base_model."): tensor
        for name, tensor in protected.state.items()
        if name.startswith("base_model.")
    }
    if set(embedded_base) != set(base.state):
        raise Stage3CheckpointValidationError(
            "Protected checkpoint's embedded base-model keys do not match its "
            "referenced base checkpoint"
        )
    base_mismatches = tuple(
        name
        for name in sorted(base.state)
        if not _is_bit_identical(embedded_base[name], base.state[name])
    )
    if base_mismatches:
        raise Stage3CheckpointValidationError(
            "Protected checkpoint's embedded base model differs from its referenced "
            f"base checkpoint: {base_mismatches}"
        )

    backbone = base.model_config.get("backbone")
    decoder_channels = base.model_config.get("decoder_channels")
    auxiliary_building_head = base.model_config.get("auxiliary_building_head")
    if not isinstance(backbone, str) or not backbone:
        raise Stage3CheckpointValidationError(
            "Base checkpoint backbone must be a non-empty string"
        )
    if (
        not isinstance(decoder_channels, (list, tuple))
        or not decoder_channels
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in decoder_channels
        )
    ):
        raise Stage3CheckpointValidationError(
            "Base checkpoint decoder_channels must be positive integers"
        )
    if not isinstance(auxiliary_building_head, bool):
        raise Stage3CheckpointValidationError(
            "Base checkpoint auxiliary_building_head must be boolean"
        )

    hidden_channels = protected.architecture_config.get("hidden_channels", 48)
    if (
        isinstance(hidden_channels, bool)
        or not isinstance(hidden_channels, int)
        or hidden_channels <= 0
    ):
        raise Stage3CheckpointValidationError(
            "Protected checkpoint hidden_channels must be a positive integer"
        )
    initial_canopy_height_m = _numeric_model_setting(
        protected.architecture_config,
        "initial_canopy_height_m",
        8.0,
        role="protected",
    )
    initial_refinement_strength = _numeric_model_setting(
        protected.architecture_config,
        "initial_refinement_strength",
        0.08,
        role="protected",
    )
    if initial_canopy_height_m <= 0:
        raise Stage3CheckpointValidationError(
            "Protected initial_canopy_height_m must be positive"
        )
    if not 0.0 < initial_refinement_strength < 1.0:
        raise Stage3CheckpointValidationError(
            "Protected initial_refinement_strength must be within (0, 1)"
        )
    policy = _fusion_policy(protected.architecture_config, role="protected")

    try:
        base_model = HeightNet(
            backbone=backbone,
            pretrained=False,
            decoder_channels=tuple(decoder_channels),
            auxiliary_building_head=auxiliary_building_head,
        )
        base_model.load_state_dict(base.state, strict=True)
        model = DomainGatedSurfaceNet(
            base_model,
            hidden_channels=hidden_channels,
            initial_canopy_height_m=initial_canopy_height_m,
            initial_refinement_strength=initial_refinement_strength,
            freeze_base=True,
            **policy,
        )
        model.load_state_dict(protected.state, strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise Stage3CheckpointValidationError(
            f"Could not reconstruct the validated shared surface model: {error}"
        ) from error

    reconstructed_state = model.state_dict()
    if set(reconstructed_state) != set(protected.state):
        raise Stage3CheckpointValidationError(
            "Reconstructed shared-model state keys do not match the protected state"
        )
    reconstruction_mismatches = tuple(
        name
        for name in sorted(protected.state)
        if not _is_bit_identical(reconstructed_state[name], protected.state[name])
    )
    if reconstruction_mismatches:
        raise Stage3CheckpointValidationError(
            "Reconstructed shared model does not exactly match the protected "
            f"checkpoint: {reconstruction_mismatches}"
        )
    model.requires_grad_(False)
    model.eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise Stage3CheckpointValidationError(
            "Reconstructed shared model is not frozen in evaluation mode"
        )

    base_names = tuple(base.state)
    diagnostics = BaseCheckpointDiagnostics(
        path=str(base.path),
        sha256=base.sha256,
        file_size_bytes=base.size_bytes,
        model_type=base.model_type,
        model_tensor_count=len(base_names),
        model_numel=sum(int(tensor.numel()) for tensor in base.state.values()),
        model_state_sha256=_state_sha256(base.state, base_names),
        architecture_config_sha256=_config_sha256(base.model_config),
    )
    return model, diagnostics


def _validate_head_architecture(
    state: Mapping[str, torch.Tensor], *, role: str
) -> int:
    expected_out_channels = {
        "domain_head": 3,
        "building_residual_head": 1,
        "canopy_height_head": 1,
    }
    feature_channels: int | None = None
    for head_name, out_channels in expected_out_channels.items():
        weight = state[f"{head_name}.weight"]
        bias = state[f"{head_name}.bias"]
        if weight.ndim != 4 or tuple(weight.shape[-2:]) != (1, 1):
            raise Stage3CheckpointValidationError(
                f"{role} {head_name} weight must describe a 1x1 convolution"
            )
        if weight.shape[0] != out_channels or bias.shape != (out_channels,):
            raise Stage3CheckpointValidationError(
                f"{role} {head_name} output shape is incompatible"
            )
        if weight.dtype != bias.dtype or not weight.is_floating_point():
            raise Stage3CheckpointValidationError(
                f"{role} {head_name} weight and bias must share a floating dtype"
            )
        if feature_channels is None:
            feature_channels = int(weight.shape[1])
        elif int(weight.shape[1]) != feature_channels:
            raise Stage3CheckpointValidationError(
                f"{role} endpoint heads do not share their input channels"
            )
    assert feature_channels is not None
    if feature_channels <= 0:
        raise Stage3CheckpointValidationError(
            f"{role} endpoint feature-channel count must be positive"
        )
    return feature_channels


def _head_module(
    state: Mapping[str, torch.Tensor],
    name: str,
    *,
    in_channels: int,
    out_channels: int,
) -> nn.Conv2d:
    weight = state[f"{name}.weight"]
    bias = state[f"{name}.bias"]
    head = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        bias=True,
        device="cpu",
        dtype=weight.dtype,
    )
    with torch.no_grad():
        head.weight.copy_(weight)
        head.bias.copy_(bias)
    return head


def _fusion_policy(
    architecture_config: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    """Read and validate the complete DomainGatedSurfaceNet fusion policy."""

    policy = {
        name: architecture_config.get(name, default)
        for name, default in _FUSION_POLICY_DEFAULTS.items()
    }
    fusion_mode = policy["fusion_mode"]
    if not isinstance(fusion_mode, str) or fusion_mode not in {
        "legacy",
        "protected_vegetation",
        "calibrated_surface",
    }:
        raise Stage3CheckpointValidationError(
            f"{role} fusion_mode is unsupported: {fusion_mode!r}"
        )

    numeric_names = tuple(
        name for name in _FUSION_POLICY_DEFAULTS if name != "fusion_mode"
    )
    for name in numeric_names:
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Stage3CheckpointValidationError(
                f"{role} {name} must be numeric"
            )
        converted = float(value)
        if not math.isfinite(converted):
            raise Stage3CheckpointValidationError(
                f"{role} {name} must be finite"
            )
        policy[name] = converted

    for name in (
        "maximum_building_residual_m",
        "building_protection_power",
        "building_fusion_temperature_m",
        "building_fusion_score_temperature",
        "vegetation_fusion_temperature",
    ):
        if policy[name] <= 0:
            raise Stage3CheckpointValidationError(
                f"{role} {name} must be positive"
            )
    for name in (
        "building_fusion_score_threshold",
        "building_fusion_strength",
        "vegetation_expert_fusion_threshold",
        "vegetation_expert_fusion_strength",
    ):
        if not 0.0 <= policy[name] <= 1.0:
            raise Stage3CheckpointValidationError(
                f"{role} {name} must be within [0, 1]"
            )
    return policy


def _extract_pack(checkpoint: _LoadedCheckpoint, *, role: str) -> FrozenSurfaceHeadPack:
    feature_channels = _validate_head_architecture(checkpoint.state, role=role)
    policy = _fusion_policy(checkpoint.architecture_config, role=role)
    try:
        pack = FrozenSurfaceHeadPack(
            _head_module(
                checkpoint.state,
                "domain_head",
                in_channels=feature_channels,
                out_channels=3,
            ),
            _head_module(
                checkpoint.state,
                "building_residual_head",
                in_channels=feature_channels,
                out_channels=1,
            ),
            _head_module(
                checkpoint.state,
                "canopy_height_head",
                in_channels=feature_channels,
                out_channels=1,
            ),
            **policy,
        )
    except (TypeError, ValueError) as error:
        raise Stage3CheckpointValidationError(
            f"{role} fusion policy is incompatible with FrozenSurfaceHeadPack: "
            f"{error}"
        ) from error
    if pack.training or any(parameter.requires_grad for parameter in pack.parameters()):
        raise Stage3CheckpointValidationError(
            f"{role} endpoint extraction did not produce a frozen evaluation pack"
        )
    return pack


def _endpoint_diagnostics(
    checkpoint: _LoadedCheckpoint, *, role: str
) -> EndpointCheckpointDiagnostics:
    state_names = tuple(checkpoint.state)
    head_names = tuple(sorted(_HEAD_TENSOR_KEYS))
    return EndpointCheckpointDiagnostics(
        role=role,
        path=str(checkpoint.path),
        sha256=checkpoint.sha256,
        file_size_bytes=checkpoint.size_bytes,
        epoch=checkpoint.epoch,
        model_type=checkpoint.model_type,
        model_tensor_count=len(state_names),
        model_numel=sum(int(tensor.numel()) for tensor in checkpoint.state.values()),
        model_state_sha256=_state_sha256(checkpoint.state, state_names),
        head_state_sha256=_state_sha256(checkpoint.state, head_names),
        architecture_config_sha256=_config_sha256(checkpoint.architecture_config),
    )


def load_stage3_endpoint_packs(
    protected_checkpoint: str | Path,
    gamus_stage1_checkpoint: str | Path,
    *,
    expected_protected_sha256: str | None = None,
    expected_gamus_stage1_sha256: str | None = None,
    expected_base_sha256: str | None = None,
    project_root: str | Path | None = None,
    reconstruct_shared_model: bool = True,
) -> Stage3EndpointPacks:
    """Validate and extract the protected and GAMUS Stage-1 endpoints.

    By default the protected checkpoint's referenced HeightNet is also loaded
    through the restricted loader, proven identical to the embedded base
    tensors, and used to reconstruct a fully frozen shared surface model.
    ``reconstruct_shared_model=False`` is reserved for head-only audit tooling.

    Any unproven condition raises :class:`Stage3CheckpointValidationError` and
    returns no partial endpoint bundle.
    """

    if not isinstance(reconstruct_shared_model, bool):
        raise Stage3CheckpointValidationError(
            "reconstruct_shared_model must be boolean"
        )
    protected = _load_checkpoint(
        protected_checkpoint,
        role="protected",
        expected_sha256=expected_protected_sha256,
    )
    gamus = _load_checkpoint(
        gamus_stage1_checkpoint,
        role="gamus_stage1",
        expected_sha256=expected_gamus_stage1_sha256,
    )
    if protected.path == gamus.path:
        raise Stage3CheckpointValidationError(
            "Protected and GAMUS Stage-1 endpoints must be different checkpoint files"
        )
    if protected.model_type != gamus.model_type:
        raise Stage3CheckpointValidationError(
            "Endpoint model types differ: "
            f"protected={protected.model_type!r}, gamus_stage1={gamus.model_type!r}"
        )
    if protected.architecture_config != gamus.architecture_config:
        differing_keys = tuple(
            sorted(
                key
                for key in set(protected.architecture_config)
                | set(gamus.architecture_config)
                if protected.architecture_config.get(key)
                != gamus.architecture_config.get(key)
            )
        )
        raise Stage3CheckpointValidationError(
            "Endpoint model architecture configs differ at keys "
            f"{differing_keys}"
        )

    protected_keys = set(protected.state)
    gamus_keys = set(gamus.state)
    if protected_keys != gamus_keys:
        raise Stage3CheckpointValidationError(
            "Endpoint model-state keys differ; "
            f"missing_from_gamus={sorted(protected_keys - gamus_keys)}, "
            f"extra_in_gamus={sorted(gamus_keys - protected_keys)}"
        )

    for name in sorted(protected_keys):
        left = protected.state[name]
        right = gamus.state[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise Stage3CheckpointValidationError(
                f"Endpoint tensor architecture differs for {name!r}: "
                f"protected={tuple(left.shape)}/{left.dtype}, "
                f"gamus_stage1={tuple(right.shape)}/{right.dtype}"
            )

    shared_names = tuple(sorted(protected_keys - _HEAD_TENSOR_KEYS))
    shared_mismatches = tuple(
        name
        for name in shared_names
        if not _is_bit_identical(protected.state[name], gamus.state[name])
    )
    if shared_mismatches:
        raise Stage3CheckpointValidationError(
            "Endpoint checkpoints differ outside the three permitted heads: "
            f"{shared_mismatches}"
        )

    differing_heads = tuple(
        name
        for name in sorted(_HEAD_TENSOR_KEYS)
        if not _is_bit_identical(protected.state[name], gamus.state[name])
    )
    identical_heads = tuple(sorted(_HEAD_TENSOR_KEYS - set(differing_heads)))
    protected_shared_hash = _state_sha256(protected.state, shared_names)
    gamus_shared_hash = _state_sha256(gamus.state, shared_names)
    if protected_shared_hash != gamus_shared_hash:
        raise Stage3CheckpointValidationError(
            "Shared endpoint tensor digest differs after bitwise validation"
        )

    protected_pack = _extract_pack(protected, role="protected")
    gamus_pack = _extract_pack(gamus, role="gamus_stage1")
    shared_model: DomainGatedSurfaceNet | None = None
    base_diagnostics: BaseCheckpointDiagnostics | None = None
    if reconstruct_shared_model:
        shared_model, base_diagnostics = _reconstruct_shared_model(
            protected,
            project_root=project_root,
            expected_base_sha256=expected_base_sha256,
        )
    architecture_items = tuple(
        (key, json.dumps(value, sort_keys=True, separators=(",", ":")))
        for key, value in sorted(protected.architecture_config.items())
    )
    diagnostics = Stage3EndpointDiagnostics(
        protected=_endpoint_diagnostics(protected, role="protected"),
        gamus_stage1=_endpoint_diagnostics(gamus, role="gamus_stage1"),
        compatibility=EndpointCompatibilityDiagnostics(
            shared_tensor_count=len(shared_names),
            shared_numel=sum(
                int(protected.state[name].numel()) for name in shared_names
            ),
            shared_state_sha256=protected_shared_hash,
            differing_head_tensors=differing_heads,
            identical_head_tensors=identical_heads,
            allowed_head_tensors=tuple(sorted(_HEAD_TENSOR_KEYS)),
            architecture_config_items=architecture_items,
        ),
        base_checkpoint=base_diagnostics,
    )
    return Stage3EndpointPacks(
        protected=protected_pack,
        gamus_stage1=gamus_pack,
        shared_model=shared_model,
        diagnostics=diagnostics,
    )


load_stage3_endpoints = load_stage3_endpoint_packs


__all__ = [
    "BaseCheckpointDiagnostics",
    "EndpointCheckpointDiagnostics",
    "EndpointCompatibilityDiagnostics",
    "Stage3CheckpointValidationError",
    "Stage3EndpointDiagnostics",
    "Stage3EndpointPacks",
    "load_stage3_endpoint_packs",
    "load_stage3_endpoints",
]
