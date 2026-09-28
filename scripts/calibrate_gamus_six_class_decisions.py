"""Fit and evaluate external six-class decision biases on DC+PHL development data.

This utility is intentionally separate from training and application inference.
It never writes a checkpoint or application pointer.  The fitted values change
only ``argmax(raw_six_class_logits + bias)``; they are not confidence or
probability calibration.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from audit_gamus_head_only_checkpoint import (  # noqa: E402
    audit_states as audit_linear_head_states,
    load_model_state,
)
from audit_gamus_spatial_refined_head_checkpoint import (  # noqa: E402
    EXPECTED_HEAD_KEYS as SPATIAL_REFINED_HEAD_KEYS,
    audit_states as audit_spatial_refined_head_states,
)
from msr.data.gamus_dataset import (  # noqa: E402
    GAMUS_APPROVED_INDEX_SCHEMA,
    GAMUS_SIX_CLASS_NAMES,
    GamusSurfaceDataset,
)
from msr.evaluation.classification_metrics import (  # noqa: E402
    StreamingClassBoundaryMetrics,
    StreamingMulticlassMetrics,
    StreamingWaterDarkPixelProxy,
)
from msr.evaluation.decision_bias_calibration import (  # noqa: E402
    DeterministicBoundedClassSampler,
    centered_biases,
    decision_gate_report,
    fit_additive_decision_biases,
)
from msr.inference.predict import load_predictor  # noqa: E402
from msr.models import DomainGatedSurfaceNet  # noqa: E402


PROTOCOL_SCHEMA = "msr.gamus_six_class_decision_bias_protocol.v1"
FIT_SCHEMA = "msr.gamus_six_class_decision_bias_fit.v1"
CONSUMPTION_SCHEMA = "msr.gamus_six_class_decision_bias_full_eval_consumption.v1"
RESULT_SCHEMA = "msr.gamus_six_class_decision_bias_full_eval.v1"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/gamus_six_class_decision_bias_v1.yaml"
LINEAR_HEAD_KEYS = frozenset(
    {"fine_semantic_head.weight", "fine_semantic_head.bias"}
)
HEAD_AUDIT_SCHEMAS = {
    "linear": "msr.gamus_head_only_checkpoint_audit.v1",
    "spatial_refined": "msr.gamus_spatial_refined_head_checkpoint_audit.v1",
}


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_once(path: Path, value: object, *, accept_identical: bool = False) -> None:
    content = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if accept_identical and path.read_bytes() == content:
            return
        raise FileExistsError(f"calibration artifact already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _pointer_snapshot(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"protected application pointer is missing: {path}")
    contents = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(contents).hexdigest(),
        "target": contents.decode("utf-8-sig").strip(),
    }


def _load_hashed_yaml(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"{name} SHA-256 mismatch")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a mapping")
    return value


def _validate_head_only_training_config(
    config: Mapping[str, Any], protected_checkpoint_sha256: str
) -> None:
    """Require a six-class-only recipe while remaining candidate-config driven."""

    model = _mapping(config.get("model"), "candidate model config")
    training = _mapping(config.get("training"), "candidate training config")
    protocol = _mapping(config.get("protocol"), "candidate protocol")
    if int(model.get("fine_semantic_classes", 0)) != 6:
        raise ValueError("candidate recipe does not declare six fine classes")
    groups = training.get("parameter_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("candidate recipe must have one explicit parameter group")
    group = _mapping(groups[0], "candidate parameter group")
    if tuple(group.get("prefixes", [])) != ("fine_semantic_head.",):
        raise ValueError("candidate recipe is not fine-semantic-head-only")
    if _number(training.get("fine_semantic_weight"), "fine semantic weight") <= 0.0:
        raise ValueError("candidate recipe must train the fine-semantic head")
    forbidden_losses = (
        "height_weight",
        "semantic_weight",
        "building_weight",
        "building_mse_weight",
        "tall_building_weight",
        "canopy_weight",
        "canopy_mse_weight",
        "tall_canopy_weight",
        "fused_building_weight",
        "fused_building_mse_weight",
        "fused_vegetation_weight",
        "fused_tall_building_weight",
        "fused_tall_vegetation_weight",
        "building_distillation_weight",
        "ground_suppression_weight",
        "refinement_weight",
        "uncertainty_weight",
    )
    nonzero = [
        name
        for name in forbidden_losses
        if _number(training.get(name, 0.0), f"candidate {name}") != 0.0
    ]
    if nonzero:
        raise ValueError(
            "candidate recipe enables non-classification losses: " + ", ".join(nonzero)
        )
    if protocol.get("protected_checkpoint_sha256") != protected_checkpoint_sha256:
        raise ValueError("candidate recipe names another protected checkpoint")
    if protocol.get("auto_promotion") is not False:
        raise ValueError("candidate recipe must forbid automatic promotion")
    if not str(protocol.get("official_test_policy", "")).startswith("never_"):
        raise ValueError("candidate recipe does not forbid official-test development use")


def _head_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the isolated head's exact state and independent-audit contract."""

    model = _mapping(config.get("model"), "candidate model config")
    head_type = str(model.get("fine_semantic_head_type", "linear"))
    if head_type == "linear":
        keys = LINEAR_HEAD_KEYS
        auditor = audit_linear_head_states
    elif head_type == "spatial_refined":
        keys = SPATIAL_REFINED_HEAD_KEYS
        auditor = audit_spatial_refined_head_states
    else:
        raise ValueError(f"unsupported isolated fine-semantic head type: {head_type}")
    return {
        "head_type": head_type,
        "allowed_trained_tensors": frozenset(keys),
        "audit_schema": HEAD_AUDIT_SCHEMAS[head_type],
        "audit_states": auditor,
    }


def validate_protocol(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load the protocol and reject any route to NYC or official test data."""

    protocol_path = _resolve(path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported six-class decision-bias protocol")
    output_root = _resolve(str(protocol.get("output_root")))
    evaluation_root = (PROJECT_ROOT / "outputs" / "evaluation").resolve()
    try:
        output_root.relative_to(evaluation_root)
    except ValueError as error:
        raise ValueError("calibration output_root must remain under outputs/evaluation") from error

    data = _mapping(protocol.get("data"), "calibration data")
    if data.get("source_split") != "val":
        raise ValueError("decision-bias fitting is restricted to GAMUS val")
    allowed_cities = tuple(data.get("allowed_cities", []))
    forbidden_cities = tuple(data.get("forbidden_cities", []))
    if allowed_cities != ("DC", "PHL") or "NYC" not in forbidden_cities:
        raise ValueError("calibration must allow only DC+PHL and forbid NYC")
    if int(data.get("expected_tiles", -1)) != 859:
        raise ValueError("DC+PHL development validation must contain 859 tiles")
    approved_index = _resolve(str(data.get("approved_index")))
    expected_index_sha = str(data.get("approved_index_sha256", ""))
    if file_sha256(approved_index) != expected_index_sha:
        raise ValueError("development approved-index SHA-256 mismatch")
    index = json.loads(approved_index.read_text(encoding="utf-8"))
    if index.get("schema") != GAMUS_APPROVED_INDEX_SCHEMA:
        raise ValueError("development approved index has an unsupported schema")
    if "test" in _mapping(index.get("splits"), "approved-index splits"):
        raise ValueError("calibration approved index must not expose official test")
    val = _mapping(
        _mapping(index.get("splits"), "approved-index splits").get("val"),
        "approved-index val split",
    )
    sample_ids = val.get("approved_sample_ids")
    if not isinstance(sample_ids, list) or len(sample_ids) != 859:
        raise ValueError("development approved index must name exactly 859 val tiles")
    cities = {str(sample_id).split("_", maxsplit=1)[0] for sample_id in sample_ids}
    if cities != set(allowed_cities):
        raise ValueError("development approved index is not exactly DC+PHL")

    candidate = _mapping(protocol.get("candidate"), "candidate")
    candidate_config = _resolve(str(candidate.get("training_config")))
    candidate_config_payload = _load_hashed_yaml(
        candidate_config,
        str(candidate.get("training_config_sha256", "")),
        "candidate training config",
    )
    _validate_head_only_training_config(
        candidate_config_payload,
        str(_mapping(protocol.get("protected_state"), "protected state").get("checkpoint_sha256", "")),
    )
    head_contract = _head_contract(candidate_config_payload)
    if set(candidate.get("allowed_trained_tensors", [])) != set(
        head_contract["allowed_trained_tensors"]
    ):
        raise ValueError(
            "calibration candidate tensors do not match its isolated head architecture"
        )
    declared_audit_schema = candidate.get(
        "required_tensor_audit_schema", head_contract["audit_schema"]
    )
    if declared_audit_schema != head_contract["audit_schema"]:
        raise ValueError("calibration candidate declares the wrong tensor-audit schema")
    if candidate.get("required_checkpoint_name") != "checkpoint_best_landscape.pt":
        raise ValueError("calibration must use the frozen best development checkpoint")
    tensor_audit_path = _resolve(str(candidate.get("required_tensor_audit")))
    try:
        tensor_audit_path.relative_to(evaluation_root)
    except ValueError as error:
        raise ValueError(
            "candidate tensor audit must remain under outputs/evaluation"
        ) from error
    protected = _mapping(protocol.get("protected_state"), "protected state")
    protected_checkpoint = _resolve(str(protected.get("checkpoint")))
    if file_sha256(protected_checkpoint) != str(protected.get("checkpoint_sha256", "")):
        raise ValueError("protected checkpoint SHA-256 mismatch")
    pointer = _resolve(str(protected.get("live_pointer")))
    if file_sha256(pointer) != str(protected.get("live_pointer_sha256", "")):
        raise ValueError("live application pointer SHA-256 mismatch")

    sampling = _mapping(protocol.get("sampling"), "sampling")
    per_tile = int(sampling.get("per_tile_class_cap", 0))
    per_class = int(sampling.get("maximum_pixels_per_class", 0))
    declared_max = int(sampling.get("hard_maximum_cached_pixels", -1))
    if per_tile <= 0 or per_class <= 0:
        raise ValueError("sampling caps must be positive")
    if declared_max != len(GAMUS_SIX_CLASS_NAMES) * per_class:
        raise ValueError("hard pixel-cache maximum is inconsistent")

    search = _mapping(protocol.get("search"), "search")
    if search.get("identifiability_constraint") != "mean_zero":
        raise ValueError("decision biases must use a mean-zero constraint")
    _number(search.get("maximum_absolute_bias"), "maximum absolute bias")
    steps = search.get("step_schedule")
    if not isinstance(steps, list) or not steps:
        raise ValueError("search step_schedule must be a non-empty list")

    acceptance = _mapping(protocol.get("acceptance"), "acceptance")
    if acceptance.get("objective") != "macro_f1":
        raise ValueError("the predeclared decision objective must be macro_f1")
    # Exercise the gate parser now rather than failing after a GPU pass.
    synthetic = {
        "macro_f1": 0.5,
        "macro_iou": 0.35,
        "per_class": {
            name: {"precision": 0.5, "recall": 0.5, "f1": 0.5, "iou": 0.35}
            for name in GAMUS_SIX_CLASS_NAMES
        },
    }
    decision_gate_report(synthetic, synthetic, acceptance, GAMUS_SIX_CLASS_NAMES)

    safety = _mapping(protocol.get("safety"), "safety")
    required_safety = {
        "official_test_policy": "never_constructed_or_discovered",
        "locked_nyc_policy": "never_opened_or_used_for_fitting_or_full_validation",
        "checkpoint_mutation": "forbidden",
        "application_model_change": "forbidden",
        "auto_promotion": False,
        "probability_confidence_calibration": False,
    }
    for key, required in required_safety.items():
        if safety.get(key) != required:
            raise ValueError(f"unsafe protocol setting: {key}")
    if safety.get("decision_calibration_kind") != "additive_logit_bias_argmax_only":
        raise ValueError("protocol mislabels decision calibration")
    if safety.get("full_validation_use") != "exactly_one_dc_phl_pass_per_fit":
        raise ValueError("full validation must be a single DC+PHL pass")
    if not str(safety.get("full_validation_confirmation_phrase", "")):
        raise ValueError("full validation confirmation phrase is missing")
    return protocol_path, protocol


def _verify_candidate(
    checkpoint: Path, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"candidate checkpoint is missing: {checkpoint}")
    candidate_spec = _mapping(protocol.get("candidate"), "candidate")
    if checkpoint.name != candidate_spec.get("required_checkpoint_name"):
        raise ValueError("candidate is not the predeclared best validation checkpoint")
    protected_spec = _mapping(protocol.get("protected_state"), "protected state")
    protected_checkpoint = _resolve(str(protected_spec.get("checkpoint")))
    config_path = _resolve(str(candidate_spec.get("training_config")))
    expected_config = _load_hashed_yaml(
        config_path,
        str(candidate_spec.get("training_config_sha256", "")),
        "candidate training config",
    )
    head_contract = _head_contract(expected_config)
    base_state, _ = load_model_state(protected_checkpoint)
    candidate_state, payload = load_model_state(checkpoint)
    state_audit = head_contract["audit_states"](base_state, candidate_state)
    embedded_config = _mapping(payload.get("config"), "checkpoint config")
    if dict(embedded_config) != expected_config:
        raise ValueError("checkpoint embedded config differs from sealed training config")
    if payload.get("model_type") != "domain_gated_surface_v4_six_class":
        raise ValueError("candidate does not expose the required six-class model type")
    audit_path = _resolve(str(candidate_spec.get("required_tensor_audit")))
    if not audit_path.is_file():
        raise FileNotFoundError(
            "the independent best-checkpoint tensor audit must pass before calibration: "
            f"{audit_path}"
        )
    audit_report = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_report.get("schema") != head_contract["audit_schema"]:
        raise ValueError("candidate tensor audit has an unsupported schema")
    audited_candidate = _mapping(audit_report.get("candidate"), "audited candidate")
    audited_state = _mapping(audit_report.get("state_audit"), "audited state")
    if (
        audited_candidate.get("sha256") != file_sha256(checkpoint)
        or audited_candidate.get("path") != str(checkpoint)
        or audited_state.get("passes") is not True
        or audited_state.get("changed_inherited_tensors") != []
    ):
        raise ValueError("independent tensor audit does not authenticate this checkpoint")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "training_config": str(config_path),
        "training_config_sha256": file_sha256(config_path),
        "independent_tensor_audit": {
            "path": str(audit_path),
            "sha256": file_sha256(audit_path),
            "passes": True,
        },
        "state_audit": state_audit,
    }


def _make_dataset(protocol: Mapping[str, Any]) -> GamusSurfaceDataset:
    data = _mapping(protocol.get("data"), "calibration data")
    dataset = GamusSurfaceDataset(
        _resolve(str(data.get("root"))),
        "val",
        patch_size=int(data.get("patch_size", 1024)),
        random_crop=False,
        augment=False,
        rgb_scale=float(data.get("rgb_scale", 255.0)),
        height_max_m=float(data.get("height_max_m", 200.0)),
        dark_pixel_threshold=float(data.get("dark_pixel_threshold", 0.15)),
        radiometric_policy=str(data.get("radiometric_policy", "raw")),
        relative_prior_root=_resolve(str(data.get("relative_prior_root"))),
        require_relative_prior=bool(data.get("require_relative_priors", True)),
        approved_index_path=_resolve(str(data.get("approved_index"))),
    )
    expected = int(data.get("expected_tiles", -1))
    if len(dataset) != expected:
        raise ValueError(f"expected {expected} DC+PHL tiles, constructed {len(dataset)}")
    allowed = set(data.get("allowed_cities", []))
    ids = tuple(record.sample_id for record in dataset.records)
    if len(set(ids)) != expected:
        raise ValueError("development validation contains duplicate IDs")
    if {sample_id.split("_", maxsplit=1)[0] for sample_id in ids} != allowed:
        raise ValueError("constructed development validation is not exactly DC+PHL")
    return dataset


def _make_loader(dataset: GamusSurfaceDataset, protocol: Mapping[str, Any]) -> DataLoader:
    data = _mapping(protocol.get("data"), "calibration data")
    workers = int(data.get("num_workers", 0))
    if workers < 0:
        raise ValueError("num_workers cannot be negative")
    return DataLoader(
        dataset,
        batch_size=int(data.get("batch_size", 1)),
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _device_and_precision(protocol: Mapping[str, Any]) -> tuple[torch.device, str]:
    runtime = _mapping(protocol.get("runtime"), "runtime")
    device = torch.device(str(runtime.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this calibration protocol")
    precision = str(runtime.get("precision", "bf16"))
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("runtime precision must be fp32, bf16, or fp16")
    if device.type != "cuda":
        precision = "fp32"
    elif precision == "bf16" and not torch.cuda.is_bf16_supported():
        precision = "fp16"
    return device, precision


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_model(checkpoint: Path, device: torch.device) -> DomainGatedSurfaceNet:
    model, metadata = load_predictor(checkpoint, device=device)
    if not isinstance(model, DomainGatedSurfaceNet):
        raise ValueError("candidate is not a DomainGatedSurfaceNet")
    if tuple(metadata.get("fine_semantic_classes") or ()) != tuple(
        GAMUS_SIX_CLASS_NAMES
    ):
        raise ValueError("candidate fine-class order differs from the protocol")
    model.eval()
    return model


def _assert_batch_cities(sample_ids: list[str], allowed: set[str]) -> None:
    observed = {sample_id.split("_", maxsplit=1)[0] for sample_id in sample_ids}
    if not observed <= allowed or "NYC" in observed:
        raise ValueError(f"forbidden city reached calibration batch: {sorted(observed)}")


@torch.inference_mode()
def fit(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    candidate = _verify_candidate(checkpoint, protocol)
    protected = _mapping(protocol.get("protected_state"), "protected state")
    pointer_path = _resolve(str(protected.get("live_pointer")))
    protected_path = _resolve(str(protected.get("checkpoint")))
    pointer_before = _pointer_snapshot(pointer_path)
    protected_before = file_sha256(protected_path)
    checkpoint_before = file_sha256(checkpoint)

    dataset = _make_dataset(protocol)
    loader = _make_loader(dataset, protocol)
    device, precision = _device_and_precision(protocol)
    model = _load_model(checkpoint, device)
    sampling = _mapping(protocol.get("sampling"), "sampling")
    sampler = DeterministicBoundedClassSampler(
        GAMUS_SIX_CLASS_NAMES,
        seed=int(sampling.get("seed")),
        per_tile_class_cap=int(sampling.get("per_tile_class_cap")),
        maximum_pixels_per_class=int(sampling.get("maximum_pixels_per_class")),
    )
    allowed = set(_mapping(protocol.get("data"), "calibration data").get("allowed_cities", []))
    try:
        for batch in tqdm(loader, desc="DC+PHL bounded decision sample"):
            sample_ids = [str(value) for value in batch["sample_id"]]
            _assert_batch_cities(sample_ids, allowed)
            image = batch["image"].to(device, non_blocking=True)
            prior = batch["relative_prior"].to(device, non_blocking=True)
            with _autocast(device, precision):
                output = model(image, prior)
            logits = output.get("fine_semantic_logits")
            if logits is None or logits.ndim != 4 or logits.shape[1] != 6:
                raise ValueError("candidate did not return six fine-semantic logits")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("candidate returned non-finite fine logits")
            logits_cpu = logits.float().cpu().numpy()
            target_cpu = batch["fine_class_target"].numpy()
            valid_cpu = (
                batch["classification_valid_mask"]
                & batch["image_valid_mask"][:, 0]
            ).numpy()
            for index, sample_id in enumerate(sample_ids):
                sampler.update(
                    logits_cpu[index],
                    target_cpu[index],
                    valid_cpu[index],
                    sample_id=sample_id,
                )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (
        sampled_logits,
        sampled_target,
        sampled_weight,
        sample_provenance,
    ) = sampler.finalize()
    if sample_provenance["tiles_seen"] != len(dataset):
        raise ValueError("bounded sampler did not see every DC+PHL tile")
    search = _mapping(protocol.get("search"), "search")
    calibration = fit_additive_decision_biases(
        sampled_logits,
        sampled_target,
        GAMUS_SIX_CLASS_NAMES,
        gate_config=_mapping(protocol.get("acceptance"), "acceptance"),
        maximum_absolute_bias=float(search.get("maximum_absolute_bias")),
        step_schedule=tuple(search.get("step_schedule", [])),
        maximum_sweeps_per_step=int(search.get("maximum_sweeps_per_step", 0)),
        sample_weight=sampled_weight,
    )

    pointer_after = _pointer_snapshot(pointer_path)
    protected_after = file_sha256(protected_path)
    checkpoint_after = file_sha256(checkpoint)
    if (
        pointer_after != pointer_before
        or protected_after != protected_before
        or checkpoint_after != checkpoint_before
    ):
        raise RuntimeError("a protected artifact changed during decision calibration")
    return {
        "schema": FIT_SCHEMA,
        "protocol": {
            "path": str(protocol_path),
            "sha256": file_sha256(protocol_path),
            "protocol_id": protocol.get("protocol_id"),
        },
        "candidate": candidate,
        "development_data": {
            "split": "official GAMUS val",
            "cities": ["DC", "PHL"],
            "tiles": len(dataset),
            "locked_nyc_opened": False,
            "official_test_constructed_or_discovered": False,
        },
        "bounded_sample": sample_provenance,
        "calibration": calibration,
        "eligible_for_one_full_dc_phl_evaluation": bool(
            calibration["gate_report"]["passes_all"]
        ),
        "interpretation": {
            "kind": "class-decision calibration",
            "effect": "adds six external offsets before class argmax only",
            "probability_or_confidence_calibration": False,
            "softmax_values_are_probabilities_of_correctness": False,
            "height_prediction_changed": False,
        },
        "safety": {
            "candidate_checkpoint_unchanged": True,
            "protected_checkpoint_unchanged": True,
            "live_pointer_unchanged": True,
            "application_model_changed": False,
            "promotion_performed": False,
        },
    }


def _validate_fit_artifact(
    artifact: Mapping[str, Any],
    protocol_path: Path,
    protocol: Mapping[str, Any],
    checkpoint: Path,
) -> np.ndarray:
    if artifact.get("schema") != FIT_SCHEMA:
        raise ValueError("unsupported decision-bias fit artifact")
    artifact_protocol = _mapping(artifact.get("protocol"), "fit protocol")
    if artifact_protocol.get("sha256") != file_sha256(protocol_path):
        raise ValueError("fit artifact protocol hash differs")
    artifact_candidate = _mapping(artifact.get("candidate"), "fit candidate")
    if artifact_candidate.get("checkpoint_sha256") != file_sha256(checkpoint):
        raise ValueError("fit artifact belongs to another checkpoint")
    if artifact.get("eligible_for_one_full_dc_phl_evaluation") is not True:
        raise ValueError("sampled calibration gates failed; full evaluation is forbidden")
    calibration = _mapping(artifact.get("calibration"), "fit calibration")
    if calibration.get("probability_confidence_calibration_performed") is not False:
        raise ValueError("artifact mislabels decision biases as confidence calibration")
    search = _mapping(protocol.get("search"), "search")
    return centered_biases(
        calibration.get("biases", []),
        class_count=len(GAMUS_SIX_CLASS_NAMES),
        maximum_absolute_bias=float(search.get("maximum_absolute_bias")),
    )


def _boundary_gate_report(
    calibrated: Mapping[str, Any],
    baseline: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    minimum = _number(config.get("minimum_f1"), "road boundary minimum F1")
    max_drop = _number(
        config.get("maximum_drop_from_uncalibrated"),
        "road boundary maximum drop",
    )
    value = _number(calibrated.get("f1"), "calibrated road boundary F1")
    reference = _number(baseline.get("f1"), "uncalibrated road boundary F1")
    checks = {
        "absolute_minimum": {
            "value": value,
            "minimum": minimum,
            "passes": value >= minimum,
        },
        "retention": {
            "value": value,
            "uncalibrated_value": reference,
            "maximum_drop": max_drop,
            "passes": value >= reference - max_drop,
        },
    }
    return {
        "passes_all": all(bool(item["passes"]) for item in checks.values()),
        "checks": checks,
    }


@torch.inference_mode()
def evaluate_full_once(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    checkpoint: Path,
    fit_path: Path,
    confirmation: str,
) -> dict[str, Any]:
    safety = _mapping(protocol.get("safety"), "safety")
    if confirmation != safety.get("full_validation_confirmation_phrase"):
        raise ValueError("explicit full-validation confirmation phrase is missing")
    output_root = _resolve(str(protocol.get("output_root")))
    marker_path = output_root / "full_validation_consumed.json"
    result_path = output_root / "full_validation_result.json"
    if marker_path.exists() or result_path.exists():
        raise FileExistsError("the one full DC+PHL evaluation has already been consumed")
    fit_artifact = json.loads(fit_path.read_text(encoding="utf-8"))
    biases = _validate_fit_artifact(
        _mapping(fit_artifact, "fit artifact"),
        protocol_path,
        protocol,
        checkpoint,
    )
    candidate = _verify_candidate(checkpoint, protocol)
    protected = _mapping(protocol.get("protected_state"), "protected state")
    pointer_path = _resolve(str(protected.get("live_pointer")))
    protected_path = _resolve(str(protected.get("checkpoint")))
    pointer_before = _pointer_snapshot(pointer_path)
    protected_before = file_sha256(protected_path)
    checkpoint_before = file_sha256(checkpoint)
    marker = {
        "schema": CONSUMPTION_SCHEMA,
        "protocol_sha256": file_sha256(protocol_path),
        "fit_artifact_sha256": file_sha256(fit_path),
        "candidate_checkpoint_sha256": checkpoint_before,
        "split": "official GAMUS val",
        "cities": ["DC", "PHL"],
        "state": "consumed_before_full_dataset_construction",
        "locked_nyc_opened": False,
        "official_test_constructed_or_discovered": False,
        "automatic_retry_permitted": False,
    }
    _write_once(marker_path, marker)

    dataset = _make_dataset(protocol)
    loader = _make_loader(dataset, protocol)
    device, precision = _device_and_precision(protocol)
    model = _load_model(checkpoint, device)
    baseline_metrics = StreamingMulticlassMetrics(GAMUS_SIX_CLASS_NAMES)
    calibrated_metrics = StreamingMulticlassMetrics(GAMUS_SIX_CLASS_NAMES)
    road_index = GAMUS_SIX_CLASS_NAMES.index("roads")
    water_index = GAMUS_SIX_CLASS_NAMES.index("water")
    baseline_boundary = StreamingClassBoundaryMetrics(road_index, tolerance_pixels=2)
    calibrated_boundary = StreamingClassBoundaryMetrics(road_index, tolerance_pixels=2)
    baseline_water_proxy = StreamingWaterDarkPixelProxy(water_index)
    calibrated_water_proxy = StreamingWaterDarkPixelProxy(water_index)
    allowed = set(_mapping(protocol.get("data"), "calibration data").get("allowed_cities", []))
    try:
        for batch in tqdm(loader, desc="one full DC+PHL decision evaluation"):
            sample_ids = [str(value) for value in batch["sample_id"]]
            _assert_batch_cities(sample_ids, allowed)
            image = batch["image"].to(device, non_blocking=True)
            prior = batch["relative_prior"].to(device, non_blocking=True)
            with _autocast(device, precision):
                output = model(image, prior)
            logits = output.get("fine_semantic_logits")
            if logits is None or logits.ndim != 4 or logits.shape[1] != 6:
                raise ValueError("candidate did not return six fine-semantic logits")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("candidate returned non-finite fine logits")
            decision_logits = logits.float()
            baseline_prediction = decision_logits.argmax(dim=1).cpu().numpy()
            bias_tensor = torch.as_tensor(
                biases, dtype=decision_logits.dtype, device=decision_logits.device
            ).view(1, -1, 1, 1)
            calibrated_prediction = (
                decision_logits + bias_tensor
            ).argmax(dim=1).cpu().numpy()
            target = batch["fine_class_target"].numpy()
            valid = (
                batch["classification_valid_mask"]
                & batch["image_valid_mask"][:, 0]
            ).numpy()
            dark = batch["dark_pixel_proxy_mask"].numpy()
            baseline_metrics.update(baseline_prediction, target, valid)
            calibrated_metrics.update(calibrated_prediction, target, valid)
            baseline_boundary.update(baseline_prediction, target, valid)
            calibrated_boundary.update(calibrated_prediction, target, valid)
            baseline_water_proxy.update(baseline_prediction, target, dark, valid)
            calibrated_water_proxy.update(calibrated_prediction, target, dark, valid)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_identification = baseline_metrics.compute()
    calibrated_identification = calibrated_metrics.compute()
    identification_gates = decision_gate_report(
        calibrated_identification,
        baseline_identification,
        _mapping(protocol.get("acceptance"), "acceptance"),
        GAMUS_SIX_CLASS_NAMES,
    )
    baseline_road = baseline_boundary.compute()
    calibrated_road = calibrated_boundary.compute()
    boundary_gates = _boundary_gate_report(
        calibrated_road,
        baseline_road,
        _mapping(protocol.get("full_validation_boundary_gate"), "boundary gate"),
    )
    pointer_after = _pointer_snapshot(pointer_path)
    protected_after = file_sha256(protected_path)
    checkpoint_after = file_sha256(checkpoint)
    if (
        pointer_after != pointer_before
        or protected_after != protected_before
        or checkpoint_after != checkpoint_before
    ):
        raise RuntimeError("a protected artifact changed during full validation")
    result = {
        "schema": RESULT_SCHEMA,
        "protocol": {
            "path": str(protocol_path),
            "sha256": file_sha256(protocol_path),
            "protocol_id": protocol.get("protocol_id"),
        },
        "fit_artifact": {"path": str(fit_path), "sha256": file_sha256(fit_path)},
        "consumption_marker_sha256": file_sha256(marker_path),
        "candidate": candidate,
        "decision_biases": {
            name: float(biases[index])
            for index, name in enumerate(GAMUS_SIX_CLASS_NAMES)
        },
        "validation": {
            "split": "official GAMUS val",
            "cities": ["DC", "PHL"],
            "tiles": len(dataset),
            "interpretation": (
                "development-set calibration evaluation, not an independent "
                "geographic holdout and not the official GAMUS test"
            ),
        },
        "uncalibrated": {
            "identification": baseline_identification,
            "road_boundary_quality": baseline_road,
            "water_dark_pixel_proxy": baseline_water_proxy.compute(),
        },
        "decision_calibrated": {
            "identification": calibrated_identification,
            "road_boundary_quality": calibrated_road,
            "water_dark_pixel_proxy": calibrated_water_proxy.compute(),
        },
        "acceptance": {
            "passes_all": bool(identification_gates["passes_all"])
            and bool(boundary_gates["passes_all"]),
            "identification": identification_gates,
            "road_boundary": boundary_gates,
        },
        "interpretation": {
            "kind": "class-decision calibration",
            "decision_rule": "argmax(raw_logits + additive_biases)",
            "probability_or_confidence_calibration": False,
            "softmax_values_are_probabilities_of_correctness": False,
            "height_prediction_changed": False,
            "locked_nyc_opened": False,
            "official_test_constructed_or_discovered": False,
        },
        "safety": {
            "candidate_checkpoint_unchanged": True,
            "protected_checkpoint_unchanged": True,
            "live_pointer_unchanged": True,
            "application_model_changed": False,
            "promotion_performed": False,
        },
    }
    _write_once(result_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-protocol", action="store_true")
    modes.add_argument("--fit", action="store_true")
    modes.add_argument("--evaluate-full", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--fit-artifact", type=Path)
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path, protocol = validate_protocol(args.protocol)
    if args.validate_protocol:
        print(
            json.dumps(
                {
                    "protocol": str(protocol_path),
                    "sha256": file_sha256(protocol_path),
                    "valid": True,
                    "data": "DC+PHL GAMUS development validation only",
                    "locked_nyc_opened": False,
                    "official_test_constructed_or_discovered": False,
                },
                indent=2,
            )
        )
        return
    if args.candidate_checkpoint is None:
        raise ValueError("--candidate-checkpoint is required for fit/evaluation")
    checkpoint = _resolve(args.candidate_checkpoint)
    output_root = _resolve(str(protocol.get("output_root")))
    if args.fit:
        result = fit(protocol_path, protocol, checkpoint)
        fit_path = output_root / "fit.json"
        _write_once(fit_path, result, accept_identical=True)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    fit_path = (
        _resolve(args.fit_artifact)
        if args.fit_artifact is not None
        else output_root / "fit.json"
    )
    if not fit_path.is_file():
        raise FileNotFoundError(f"fit artifact is missing: {fit_path}")
    result = evaluate_full_once(
        protocol_path,
        protocol,
        checkpoint,
        fit_path,
        args.confirmation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
