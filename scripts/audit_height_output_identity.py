"""Fail-closed height-output identity audit for semantic-only checkpoints.

This audit complements the head-only state-dict auditors.  It proves that a
candidate:

* adds state only below ``fine_semantic_head.``;
* preserves every inherited tensor bit-for-bit;
* declares the same effective RGB preprocessing and height-fusion policy;
* reconstructs with the same effective fusion attributes; and
* emits bit-identical shared/height tensors and app-level height maps for
  deterministic identical inputs.

The script is read-only with respect to checkpoints and the application
pointer.  It writes a report only after every guard passes and never promotes
or selects a checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from msr.data.raster_dataset import IMAGENET_MEAN, IMAGENET_STD
from msr.inference.predict import (
    _normalize_tile,
    load_predictor,
    predict_height,
)
from msr.models.domain_surface_net import DomainGatedSurfaceNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "msr.height_output_identity_audit.v1"
HEAD_PREFIX = "fine_semantic_head."
HEAD_ONLY_MODEL_CONFIG_KEYS = frozenset(
    {
        "fine_semantic_classes",
        "fine_semantic_head_type",
    }
)
PROVENANCE_ONLY_MODEL_CONFIG_KEYS = frozenset({"initial_checkpoint"})

HEIGHT_MODEL_DEFAULTS: dict[str, Any] = {
    "hidden_channels": 48,
    "maximum_building_residual_m": 30.0,
    "initial_canopy_height_m": 8.0,
    "initial_refinement_strength": 0.08,
}
FUSION_DEFAULTS: dict[str, Any] = {
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
LOADED_FUSION_ATTRIBUTES = tuple(FUSION_DEFAULTS)

# These tensors either determine the final height or are frozen inputs to that
# calculation.  ``fine_semantic_*`` is intentionally absent: that side output
# is allowed to change, but it must not feed back into any entry below.
SHARED_FORWARD_OUTPUTS = (
    "height",
    "gated_height",
    "refinement_strength",
    "effective_refinement_strength",
    "building_protection",
    "building_fusion_gate",
    "vegetation_fusion_probability",
    "vegetation_expert_fusion_gate",
    "adapter_features",
    "prior_domain_logits",
    "building_height",
    "canopy_height",
    "refinement_logits",
    "domain_logits",
    "domain_probabilities",
    "building_logits",
    "protected_building_logits",
    "vegetation_logits",
    "log_variance",
    "base_height",
)
APP_SHARED_OUTPUTS = (
    "height_map",
    "building_probability_map",
    "vegetation_probability_map",
    "confidence_map",
    "valid_mask",
)
SOURCE_FILES = (
    PROJECT_ROOT / "src/msr/data/raster_dataset.py",
    PROJECT_ROOT / "src/msr/inference/predict.py",
    PROJECT_ROOT / "src/msr/inference/tiling.py",
    PROJECT_ROOT / "src/msr/models/domain_surface_net.py",
    PROJECT_ROOT / "src/msr/models/height_net.py",
)


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    return value


def _positive_float(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{role} must be finite and positive")
    return result


def _checkpoint_payload(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    raw_payload = torch.load(path, map_location="cpu", weights_only=False)
    payload = dict(_mapping(raw_payload, f"checkpoint {path}"))
    raw_state = _mapping(payload.get("model"), f"model state in {path}")
    state: dict[str, torch.Tensor] = {}
    for name, tensor in raw_state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"invalid model-state entry {name!r} in {path}")
        state[name] = tensor
    if not state:
        raise ValueError(f"checkpoint contains an empty model state: {path}")
    return payload, state


def _tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().reshape(-1)
    raw = value.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _array_bytes_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8).tobytes(order="C")).hexdigest()


def audit_inherited_state(
    protected_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Require a bit-identical inherited state and one isolated semantic head."""

    protected_keys = set(protected_state)
    candidate_keys = set(candidate_state)
    protected_head_keys = sorted(
        name for name in protected_keys if name.startswith(HEAD_PREFIX)
    )
    if protected_head_keys:
        raise ValueError(
            "protected checkpoint unexpectedly contains fine-semantic state: "
            + ", ".join(protected_head_keys[:20])
        )
    missing = sorted(protected_keys - candidate_keys)
    added = sorted(candidate_keys - protected_keys)
    unexpected = [name for name in added if not name.startswith(HEAD_PREFIX)]
    if missing:
        raise ValueError(f"candidate is missing inherited tensors: {missing[:20]}")
    if unexpected:
        raise ValueError(f"candidate has non-head tensors: {unexpected[:20]}")
    if not added:
        raise ValueError("candidate adds no fine-semantic head tensors")

    inherited_elements = 0
    inherited_digests: list[tuple[str, str]] = []
    for name in sorted(protected_keys):
        protected = protected_state[name]
        candidate = candidate_state[name]
        if protected.shape != candidate.shape:
            raise ValueError(
                f"inherited tensor shape changed for {name}: "
                f"{tuple(protected.shape)} != {tuple(candidate.shape)}"
            )
        if protected.dtype != candidate.dtype:
            raise ValueError(
                f"inherited tensor dtype changed for {name}: "
                f"{protected.dtype} != {candidate.dtype}"
            )
        protected_digest = _tensor_bytes_sha256(protected)
        candidate_digest = _tensor_bytes_sha256(candidate)
        if not torch.equal(protected, candidate) or protected_digest != candidate_digest:
            raise ValueError(f"inherited tensor changed bit-for-bit: {name}")
        inherited_elements += protected.numel()
        inherited_digests.append((name, protected_digest))

    head_elements = 0
    head_details: dict[str, Any] = {}
    for name in added:
        tensor = candidate_state[name]
        if tensor.is_floating_point() or tensor.is_complex():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"candidate head tensor is non-finite: {name}")
        head_elements += tensor.numel()
        head_details[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": _tensor_bytes_sha256(tensor),
        }

    return {
        "passes": True,
        "comparison": "torch.equal plus raw tensor-byte SHA-256",
        "inherited_tensor_count": len(protected_keys),
        "inherited_element_count": inherited_elements,
        "inherited_state_sha256": canonical_json_sha256(inherited_digests),
        "changed_inherited_tensors": [],
        "allowed_new_prefix": HEAD_PREFIX,
        "new_head_tensor_count": len(added),
        "new_head_element_count": head_elements,
        "new_head_tensors": head_details,
    }


def _effective_model_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    known = {
        name: model_config.get(name, default)
        for name, default in HEIGHT_MODEL_DEFAULTS.items()
    }
    known["base_checkpoint_sha256"] = file_sha256(
        _resolve(str(model_config.get("base_checkpoint", "")))
    )
    ignored = (
        set(HEIGHT_MODEL_DEFAULTS)
        | set(FUSION_DEFAULTS)
        | set(HEAD_ONLY_MODEL_CONFIG_KEYS)
        | set(PROVENANCE_ONLY_MODEL_CONFIG_KEYS)
        | {"base_checkpoint"}
    )
    unknown = {
        str(name): value for name, value in model_config.items() if name not in ignored
    }
    known["other_height_model_fields"] = unknown
    return known


def _effective_fusion_config(model_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: model_config.get(name, default)
        for name, default in FUSION_DEFAULTS.items()
    }


def extract_pipeline_contract(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    tile_size: int,
    overlap: int,
) -> dict[str, Any]:
    config = _mapping(payload.get("config"), f"config in {checkpoint_path}")
    data_config = _mapping(config.get("data"), f"data config in {checkpoint_path}")
    model_config = _mapping(
        config.get("model"), f"model config in {checkpoint_path}"
    )
    rgb_scale = _positive_float(
        data_config.get("rgb_scale", 255.0),
        f"data.rgb_scale in {checkpoint_path}",
    )
    radiometric_policy = str(
        data_config.get("validation_radiometric_policy", "raw")
    )
    if radiometric_policy != "raw":
        raise ValueError(
            "height-output identity audit requires validation_radiometric_policy "
            f"'raw', found {radiometric_policy!r} in {checkpoint_path}"
        )
    base_checkpoint_value = model_config.get("base_checkpoint")
    if not isinstance(base_checkpoint_value, str) or not base_checkpoint_value.strip():
        raise ValueError(f"model.base_checkpoint is missing in {checkpoint_path}")
    base_checkpoint = _resolve(base_checkpoint_value)
    if not base_checkpoint.is_file():
        raise FileNotFoundError(
            f"base checkpoint referenced by {checkpoint_path} is missing: "
            f"{base_checkpoint}"
        )

    preprocessing = {
        "rgb_scale": rgb_scale,
        "validation_radiometric_policy": radiometric_policy,
        "channel_order": "RGB_CHW",
        "normalization": "(rgb / rgb_scale - imagenet_mean) / imagenet_std",
        "imagenet_mean": np.asarray(IMAGENET_MEAN).reshape(3).tolist(),
        "imagenet_std": np.asarray(IMAGENET_STD).reshape(3).tolist(),
        "rgb_nonfinite_policy": "valid_mask_intersection_then_numpy_nan_to_num",
        "relative_prior_nonfinite_policy": "nan_posinf_neginf_to_zero",
        "relative_prior_range_policy": "explicit_identical_input_no_rescaling",
    }
    fusion = _effective_fusion_config(model_config)
    height_model = _effective_model_config(model_config)
    contract = {
        "preprocessing": preprocessing,
        "height_model": height_model,
        "fusion": fusion,
        "inference": {
            "device": "cpu",
            "amp": False,
            "tile_size": tile_size,
            "overlap": overlap,
            "blend_weight": "msr.inference.tiling.blend_weight",
        },
    }
    return {
        **contract,
        "preprocessing_sha256": canonical_json_sha256(preprocessing),
        "height_model_sha256": canonical_json_sha256(height_model),
        "fusion_sha256": canonical_json_sha256(fusion),
        "full_contract_sha256": canonical_json_sha256(contract),
    }


def compare_pipeline_contracts(
    protected: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    for section in ("preprocessing", "height_model", "fusion", "inference"):
        if protected.get(section) != candidate.get(section):
            raise ValueError(
                f"candidate {section} contract differs from protected: "
                f"protected={protected.get(section)!r}, "
                f"candidate={candidate.get(section)!r}"
            )
    if protected.get("full_contract_sha256") != candidate.get(
        "full_contract_sha256"
    ):
        raise ValueError("pipeline contract hashes differ despite section comparison")
    return {
        "passes": True,
        "comparison": "canonical effective configuration equality",
        "preprocessing_sha256": protected["preprocessing_sha256"],
        "height_model_sha256": protected["height_model_sha256"],
        "fusion_sha256": protected["fusion_sha256"],
        "full_contract_sha256": protected["full_contract_sha256"],
        "contract": {
            section: protected[section]
            for section in ("preprocessing", "height_model", "fusion", "inference")
        },
    }


def _validate_candidate_provenance(
    protected_path: Path,
    protected_sha256: str,
    candidate_payload: Mapping[str, Any],
) -> dict[str, Any]:
    config = _mapping(candidate_payload.get("config"), "candidate config")
    model_config = _mapping(config.get("model"), "candidate model config")
    initial_value = model_config.get("initial_checkpoint")
    if not isinstance(initial_value, str) or not initial_value.strip():
        raise ValueError("candidate model.initial_checkpoint must name the protected model")
    initial_path = _resolve(initial_value)
    if not initial_path.is_file():
        raise FileNotFoundError(f"candidate initial checkpoint is missing: {initial_path}")
    initial_sha256 = file_sha256(initial_path)
    if initial_sha256 != protected_sha256:
        raise ValueError(
            "candidate initial checkpoint is not the supplied protected checkpoint: "
            f"{initial_sha256} != {protected_sha256}"
        )
    protocol = _mapping(config.get("protocol"), "candidate protocol")
    declared_sha256 = protocol.get("protected_checkpoint_sha256")
    if declared_sha256 != protected_sha256:
        raise ValueError(
            "candidate protocol protected_checkpoint_sha256 does not match the "
            f"supplied protected checkpoint: {declared_sha256!r} != "
            f"{protected_sha256!r}"
        )
    return {
        "passes": True,
        "protected_path": str(protected_path),
        "protected_sha256": protected_sha256,
        "initial_checkpoint_path": str(initial_path),
        "initial_checkpoint_sha256": initial_sha256,
        "declared_protected_checkpoint_sha256": declared_sha256,
    }


def _loaded_fusion_values(model: DomainGatedSurfaceNet) -> dict[str, Any]:
    return {name: getattr(model, name) for name in LOADED_FUSION_ATTRIBUTES}


def _validate_loaded_models(
    protected_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    pipeline: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(protected_model, DomainGatedSurfaceNet):
        raise ValueError("protected checkpoint did not reconstruct a DomainGatedSurfaceNet")
    if not isinstance(candidate_model, DomainGatedSurfaceNet):
        raise ValueError("candidate checkpoint did not reconstruct a DomainGatedSurfaceNet")
    if protected_model.training or candidate_model.training:
        raise ValueError("both reconstructed models must be in evaluation mode")
    if protected_model.fine_semantic_head is not None:
        raise ValueError("protected reconstructed model unexpectedly has a semantic head")
    if candidate_model.fine_semantic_head is None:
        raise ValueError("candidate reconstructed model has no semantic head")
    protected_fusion = _loaded_fusion_values(protected_model)
    candidate_fusion = _loaded_fusion_values(candidate_model)
    expected_fusion = dict(_mapping(pipeline.get("fusion"), "pipeline fusion"))
    if protected_fusion != expected_fusion:
        raise ValueError(
            "protected loaded fusion attributes differ from sealed configuration: "
            f"{protected_fusion!r} != {expected_fusion!r}"
        )
    if candidate_fusion != expected_fusion:
        raise ValueError(
            "candidate loaded fusion attributes differ from sealed configuration: "
            f"{candidate_fusion!r} != {expected_fusion!r}"
        )
    return {
        "passes": True,
        "protected_type": type(protected_model).__name__,
        "candidate_type": type(candidate_model).__name__,
        "protected_eval": True,
        "candidate_eval": True,
        "effective_fusion": expected_fusion,
    }


def _assert_tensor_identity(
    name: str, protected: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    if protected.shape != candidate.shape:
        raise ValueError(
            f"height-path output shape changed for {name}: "
            f"{tuple(protected.shape)} != {tuple(candidate.shape)}"
        )
    if protected.dtype != candidate.dtype:
        raise ValueError(
            f"height-path output dtype changed for {name}: "
            f"{protected.dtype} != {candidate.dtype}"
        )
    if not torch.isfinite(protected).all() or not torch.isfinite(candidate).all():
        raise ValueError(f"height-path output is non-finite for {name}")
    protected_digest = _tensor_bytes_sha256(protected)
    candidate_digest = _tensor_bytes_sha256(candidate)
    if not torch.equal(protected, candidate) or protected_digest != candidate_digest:
        difference = (protected.float() - candidate.float()).abs()
        raise ValueError(
            f"height-path output changed for {name}: "
            f"max_abs_difference={float(difference.max().item())}, "
            f"protected_sha256={protected_digest}, "
            f"candidate_sha256={candidate_digest}"
        )
    return {
        "shape": list(protected.shape),
        "dtype": str(protected.dtype),
        "sha256": protected_digest,
        "max_abs_difference": 0.0,
        "bit_identical": True,
    }


def _assert_array_identity(
    name: str, protected: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    protected = np.asarray(protected)
    candidate = np.asarray(candidate)
    if protected.shape != candidate.shape:
        raise ValueError(
            f"app output shape changed for {name}: "
            f"{protected.shape} != {candidate.shape}"
        )
    if protected.dtype != candidate.dtype:
        raise ValueError(
            f"app output dtype changed for {name}: "
            f"{protected.dtype} != {candidate.dtype}"
        )
    protected_digest = _array_bytes_sha256(protected)
    candidate_digest = _array_bytes_sha256(candidate)
    if protected_digest != candidate_digest:
        finite = np.isfinite(protected) & np.isfinite(candidate)
        max_abs_difference = (
            float(np.max(np.abs(protected[finite] - candidate[finite])))
            if np.any(finite)
            else None
        )
        raise ValueError(
            f"app output changed for {name}: "
            f"max_abs_difference={max_abs_difference}, "
            f"protected_sha256={protected_digest}, "
            f"candidate_sha256={candidate_digest}"
        )
    return {
        "shape": list(protected.shape),
        "dtype": str(protected.dtype),
        "sha256": protected_digest,
        "max_abs_difference": 0.0,
        "bit_identical": True,
    }


def _synthetic_raw_cases(
    *, tile_size: int, rgb_scale: float
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, (height, width, include_prior) in enumerate(
        (
            (tile_size, tile_size, False),
            (max(32, tile_size - 5), max(32, tile_size - 9), True),
            (tile_size + 17, tile_size + 9, True),
        )
    ):
        rows, columns = np.indices((height, width), dtype=np.float32)
        image = np.stack(
            (
                np.mod(rows * 17.0 + columns * 3.0 + 11.0, 251.0),
                np.mod(rows * 5.0 + columns * 19.0 + 29.0, 251.0),
                np.mod(rows * 13.0 + columns * 7.0 + 47.0, 251.0),
            ),
            axis=0,
        ).astype(np.float32)
        image *= np.float32(rgb_scale / 250.0)
        valid_mask = ((rows.astype(np.int64) + columns.astype(np.int64)) % 23) != 0
        prior: np.ndarray | None = None
        if include_prior:
            prior = np.mod(rows * 0.037 + columns * 0.019, 1.0).astype(np.float32)
            prior[0, 0] = np.nan
            prior[0, 1] = np.inf
            prior[0, 2] = -np.inf
            image[0, 1, 1] = np.nan
            valid_mask[1, 1] = False
        cases.append(
            {
                "name": f"synthetic_{index}_{'prior' if include_prior else 'zero_prior'}",
                "image": image,
                "relative_prior": prior,
                "valid_mask": valid_mask,
            }
        )
    return cases


def _forward_surface(
    model: torch.nn.Module,
    image: torch.Tensor,
    relative_prior: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    if isinstance(model, DomainGatedSurfaceNet):
        output = model(image, relative_prior)
    else:  # Kept for focused unit tests; production validation rejects this type.
        output = model(image)
    return _mapping(output, "model output")  # type: ignore[return-value]


@torch.inference_mode()
def audit_direct_outputs(
    protected_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    *,
    tile_size: int,
    rgb_scale: float,
) -> dict[str, Any]:
    rows, columns = np.indices((tile_size, tile_size), dtype=np.float32)
    raw = np.stack(
        (
            np.mod(rows * 7.0 + columns * 11.0, 251.0),
            np.mod(rows * 17.0 + columns * 5.0 + 31.0, 251.0),
            np.mod(rows * 3.0 + columns * 23.0 + 73.0, 251.0),
        ),
        axis=0,
    ).astype(np.float32)
    raw *= np.float32(rgb_scale / 250.0)
    image = _normalize_tile(raw, rgb_scale)
    prior_array = np.mod(rows * 0.011 + columns * 0.029, 1.0).astype(np.float32)
    prior = torch.from_numpy(np.ascontiguousarray(prior_array[None, None]))
    protected_output = _forward_surface(protected_model, image, prior)
    candidate_output = _forward_surface(candidate_model, image, prior)

    outputs: dict[str, Any] = {}
    for name in SHARED_FORWARD_OUTPUTS:
        if name not in protected_output or name not in candidate_output:
            raise ValueError(f"required shared forward output is missing: {name}")
        protected = protected_output[name]
        candidate = candidate_output[name]
        if not isinstance(protected, torch.Tensor) or not isinstance(
            candidate, torch.Tensor
        ):
            raise ValueError(f"shared forward output is not a tensor: {name}")
        outputs[name] = _assert_tensor_identity(name, protected, candidate)

    return {
        "passes": True,
        "comparison": "torch.equal plus raw tensor-byte SHA-256",
        "input": {
            "shape": list(image.shape),
            "normalized_rgb_sha256": _tensor_bytes_sha256(image),
            "relative_prior_sha256": _tensor_bytes_sha256(prior),
        },
        "outputs": outputs,
    }


@torch.inference_mode()
def audit_app_outputs(
    protected_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    *,
    tile_size: int,
    overlap: int,
    rgb_scale: float,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for case in _synthetic_raw_cases(tile_size=tile_size, rgb_scale=rgb_scale):
        kwargs = {
            "device": "cpu",
            "tile_size": tile_size,
            "overlap": overlap,
            "rgb_scale": rgb_scale,
            "amp": False,
            "relative_prior": case["relative_prior"],
            "valid_mask": case["valid_mask"],
        }
        protected = predict_height(case["image"], model=protected_model, **kwargs)
        candidate = predict_height(case["image"], model=candidate_model, **kwargs)
        outputs: dict[str, Any] = {}
        for name in APP_SHARED_OUTPUTS:
            protected_value = getattr(protected, name)
            candidate_value = getattr(candidate, name)
            if protected_value is None or candidate_value is None:
                raise ValueError(f"required app-level shared output is missing: {name}")
            outputs[name] = _assert_array_identity(
                f"{case['name']}.{name}", protected_value, candidate_value
            )
        reports.append(
            {
                "name": case["name"],
                "raw_rgb_shape": list(case["image"].shape),
                "raw_rgb_sha256": _array_bytes_sha256(case["image"]),
                "relative_prior_sha256": (
                    _array_bytes_sha256(case["relative_prior"])
                    if case["relative_prior"] is not None
                    else None
                ),
                "requested_valid_mask_sha256": _array_bytes_sha256(
                    case["valid_mask"]
                ),
                "outputs": outputs,
            }
        )
    return {
        "passes": True,
        "comparison": "raw NumPy output-byte SHA-256",
        "cases": reports,
    }


@contextmanager
def _deterministic_cpu_context() -> Iterator[None]:
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic


def _source_fingerprints() -> dict[str, str]:
    missing = [path for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"height-pipeline source files are missing: {missing}")
    return {
        str(path.relative_to(PROJECT_ROOT)).replace(os.sep, "/"): file_sha256(path)
        for path in SOURCE_FILES
    }


def build_report(
    protected_path: Path,
    candidate_path: Path,
    *,
    tile_size: int = 64,
    overlap: int = 16,
    expected_protected_sha256: str | None = None,
    expected_candidate_sha256: str | None = None,
) -> dict[str, Any]:
    protected_path = protected_path.resolve()
    candidate_path = candidate_path.resolve()
    # The protected ConvNeXt/MobileNet encoders downsample by 32.  A 32-pixel
    # tile leaves one spatial value at the deepest GroupNorm layer, which is
    # invalid even in evaluation mode.  Keep the audit portable across both.
    if tile_size < 64:
        raise ValueError("tile_size must be at least 64")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    for path in (protected_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    protected_sha256_before = file_sha256(protected_path)
    candidate_sha256_before = file_sha256(candidate_path)
    if (
        expected_protected_sha256 is not None
        and protected_sha256_before != expected_protected_sha256
    ):
        raise ValueError(
            "protected checkpoint SHA-256 mismatch: "
            f"{protected_sha256_before} != {expected_protected_sha256}"
        )
    if (
        expected_candidate_sha256 is not None
        and candidate_sha256_before != expected_candidate_sha256
    ):
        raise ValueError(
            "candidate checkpoint SHA-256 mismatch: "
            f"{candidate_sha256_before} != {expected_candidate_sha256}"
        )

    protected_payload, protected_state = _checkpoint_payload(protected_path)
    candidate_payload, candidate_state = _checkpoint_payload(candidate_path)
    provenance = _validate_candidate_provenance(
        protected_path, protected_sha256_before, candidate_payload
    )
    state_audit = audit_inherited_state(protected_state, candidate_state)
    protected_contract = extract_pipeline_contract(
        protected_payload,
        checkpoint_path=protected_path,
        tile_size=tile_size,
        overlap=overlap,
    )
    candidate_contract = extract_pipeline_contract(
        candidate_payload,
        checkpoint_path=candidate_path,
        tile_size=tile_size,
        overlap=overlap,
    )
    pipeline_audit = compare_pipeline_contracts(
        protected_contract, candidate_contract
    )

    with _deterministic_cpu_context():
        protected_model, protected_metadata = load_predictor(
            protected_path, device="cpu"
        )
        candidate_model, candidate_metadata = load_predictor(
            candidate_path, device="cpu"
        )
        loaded_model_audit = _validate_loaded_models(
            protected_model, candidate_model, pipeline_audit["contract"]
        )
        direct_output_audit = audit_direct_outputs(
            protected_model,
            candidate_model,
            tile_size=tile_size,
            rgb_scale=float(
                pipeline_audit["contract"]["preprocessing"]["rgb_scale"]
            ),
        )
        app_output_audit = audit_app_outputs(
            protected_model,
            candidate_model,
            tile_size=tile_size,
            overlap=overlap,
            rgb_scale=float(
                pipeline_audit["contract"]["preprocessing"]["rgb_scale"]
            ),
        )

    protected_sha256_after = file_sha256(protected_path)
    candidate_sha256_after = file_sha256(candidate_path)
    if protected_sha256_after != protected_sha256_before:
        raise RuntimeError("protected checkpoint changed while it was being audited")
    if candidate_sha256_after != candidate_sha256_before:
        raise RuntimeError("candidate checkpoint changed while it was being audited")

    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passes": True,
        "protected": {
            "path": str(protected_path),
            "sha256_before": protected_sha256_before,
            "sha256_after": protected_sha256_after,
            "model_type": protected_metadata.get("model_type"),
            "epoch": protected_metadata.get("epoch"),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256_before": candidate_sha256_before,
            "sha256_after": candidate_sha256_after,
            "model_type": candidate_metadata.get("model_type"),
            "epoch": candidate_metadata.get("epoch"),
        },
        "provenance_audit": provenance,
        "state_audit": state_audit,
        "pipeline_contract_audit": pipeline_audit,
        "loaded_model_audit": loaded_model_audit,
        "direct_output_audit": direct_output_audit,
        "app_output_audit": app_output_audit,
        "source_fingerprints": _source_fingerprints(),
        "checkpoint_files_unchanged_during_audit": True,
        "app_pointer": {
            "inspected": False,
            "modified": False,
            "reason": "pointer sealing belongs to the external run wrapper",
        },
        "promotion_performed": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-protected-sha256")
    parser.add_argument("--expected-candidate-sha256")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    protected_path = _resolve(args.protected_checkpoint)
    candidate_path = _resolve(args.candidate_checkpoint)
    report = build_report(
        protected_path,
        candidate_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        expected_protected_sha256=args.expected_protected_sha256,
        expected_candidate_sha256=args.expected_candidate_sha256,
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "passes": report["passes"],
                "inherited_tensor_count": report["state_audit"][
                    "inherited_tensor_count"
                ],
                "pipeline_contract_sha256": report["pipeline_contract_audit"][
                    "full_contract_sha256"
                ],
                "direct_output_count": len(
                    report["direct_output_audit"]["outputs"]
                ),
                "app_case_count": len(report["app_output_audit"]["cases"]),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
