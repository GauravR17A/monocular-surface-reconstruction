"""Fail-closed audit for the isolated spatial six-class GAMUS head.

The v3 candidate may add and optimize only ``fine_semantic_head.*``. Every
tensor inherited from the protected production height model must be present
with the same shape, dtype, and exact value according to ``torch.equal``.
This auditor is read-only and deliberately does not inspect or modify the live
application pointer; the launch wrapper seals that pointer before and after the
run.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "msr.gamus_spatial_refined_head_checkpoint_audit.v1"
PROTOCOL_SCHEMA = "msr.gamus_six_class_spatial_refined.v3"
HEAD_PREFIX = "fine_semantic_head."
EXPECTED_HEAD_KEYS = frozenset(
    {
        "fine_semantic_head.spatial.weight",
        "fine_semantic_head.normalization.weight",
        "fine_semantic_head.normalization.bias",
        "fine_semantic_head.classifier.weight",
        "fine_semantic_head.classifier.bias",
    }
)


def _load_v2_audit_module():
    """Reuse the sealed v2 classification-gate implementation exactly."""

    path = Path(__file__).with_name("audit_gamus_head_only_checkpoint.py")
    spec = importlib.util.spec_from_file_location("_msr_v2_head_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load the v2 head-only auditor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V2_AUDIT = _load_v2_audit_module()


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


def _mapping(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def load_model_state(path: Path) -> tuple[dict[str, torch.Tensor], Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload = _mapping(payload, f"checkpoint {path}")
    raw_state = _mapping(payload.get("model"), f"model state in {path}")
    state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"invalid model-state entry {name!r} in {path}")
        state[name] = value
    if not state:
        raise ValueError(f"checkpoint contains an empty model state: {path}")
    return state, payload


def audit_states(
    base_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Prove that the candidate is the protected model plus one isolated head."""

    base_keys = set(base_state)
    candidate_keys = set(candidate_state)
    if any(name.startswith(HEAD_PREFIX) for name in base_keys):
        raise ValueError("protected base unexpectedly contains a fine-semantic head")

    missing = sorted(base_keys - candidate_keys)
    added = set(candidate_keys - base_keys)
    unexpected = sorted(name for name in added if not name.startswith(HEAD_PREFIX))
    if missing:
        raise ValueError(f"candidate is missing inherited tensors: {missing}")
    if unexpected:
        raise ValueError(f"candidate has unexpected new tensors: {unexpected}")
    if added != EXPECTED_HEAD_KEYS:
        raise ValueError(
            "candidate must add exactly the spatial-refined six-class head tensors; "
            f"found {sorted(added)}"
        )

    compared_elements = 0
    changed: list[str] = []
    for name in sorted(base_keys):
        base = base_state[name]
        candidate = candidate_state[name]
        if base.shape != candidate.shape or base.dtype != candidate.dtype:
            changed.append(name)
            continue
        compared_elements += base.numel()
        if not torch.equal(base, candidate):
            changed.append(name)
    if changed:
        raise ValueError(
            "protected height/shared tensors changed during head-only training: "
            + ", ".join(changed[:20])
        )

    head_details: dict[str, Any] = {}
    changed_from_neutral: list[str] = []
    for name in sorted(EXPECTED_HEAD_KEYS):
        tensor = candidate_state[name]
        if not torch.isfinite(tensor).all():
            raise ValueError(f"candidate head tensor is non-finite: {name}")
        neutral = (
            torch.ones_like(tensor)
            if name == "fine_semantic_head.normalization.weight"
            else torch.zeros_like(tensor)
        )
        changed_from_initial = not torch.equal(tensor, neutral)
        if changed_from_initial:
            changed_from_neutral.append(name)
        head_details[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "l2_norm": float(torch.linalg.vector_norm(tensor.float()).item()),
            "changed_from_neutral_initialization": changed_from_initial,
        }

    # A valid trained classifier must at minimum move its output projection.
    # Other trainable refined-head parameters are allowed to remain at their
    # neutral initialization if optimization legitimately leaves them there.
    if "fine_semantic_head.classifier.weight" not in changed_from_neutral:
        raise ValueError("spatial-refined classifier projection was not trained")

    return {
        "passes": True,
        "inherited_tensor_count": len(base_keys),
        "inherited_element_count": compared_elements,
        "inherited_equality": "torch.equal",
        "changed_inherited_tensors": [],
        "allowed_trainable_prefix": HEAD_PREFIX,
        "allowed_trainable_tensors": head_details,
        "head_tensors_changed_from_neutral": changed_from_neutral,
    }


def validate_recipe_contract(
    source_v2: Mapping[str, Any],
    candidate_v3: Mapping[str, Any],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    """Require exact v2 behavior except for identity and head architecture."""

    source = deepcopy(dict(source_v2))
    candidate = deepcopy(dict(candidate_v3))
    source_experiment = _mapping(source.get("experiment"), "v2 experiment")
    candidate_experiment = _mapping(candidate.get("experiment"), "v3 experiment")
    if candidate_experiment.get("name") != (
        "multidomain_surface_gamus_six_class_spatial_refined_v3"
    ):
        raise ValueError("candidate experiment identity is not the sealed v3 name")
    for key in ("seed", "output_root"):
        if candidate_experiment.get(key) != source_experiment.get(key):
            raise ValueError(f"v3 experiment.{key} must match v2 exactly")

    for section in ("data", "training", "evaluation"):
        if candidate.get(section) != source.get(section):
            raise ValueError(f"v3 {section} contract differs from sealed v2")

    source_model = dict(_mapping(source.get("model"), "v2 model"))
    candidate_model = dict(_mapping(candidate.get("model"), "v3 model"))
    if candidate_model.pop("fine_semantic_head_type", None) != "spatial_refined":
        raise ValueError("v3 must use model.fine_semantic_head_type: spatial_refined")
    if candidate_model != source_model:
        raise ValueError("v3 model differs from v2 beyond the head architecture")

    source_protocol = _mapping(source.get("protocol"), "v2 protocol")
    protocol = _mapping(candidate.get("protocol"), "v3 protocol")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("candidate protocol schema is not the sealed v3 schema")
    if protocol.get("source_recipe_config_sha256") != source_sha256:
        raise ValueError("candidate does not pin the exact source v2 recipe")
    inherited_protocol_keys = (
        "comparison_height_pilot",
        "comparison_joint_six_class",
        "protected_checkpoint_sha256",
        "live_pointer_file_sha256",
        "official_test_policy",
        "auto_promotion",
        "classification_acceptance",
    )
    for key in inherited_protocol_keys:
        if protocol.get(key) != source_protocol.get(key):
            raise ValueError(f"v3 protocol.{key} must match v2 exactly")
    if protocol.get("official_test_policy") != "never_constructed_for_development":
        raise ValueError("official test construction is forbidden")
    if protocol.get("auto_promotion") is not False:
        raise ValueError("automatic application-model promotion must remain disabled")

    data = _mapping(candidate.get("data"), "v3 data")
    forbidden = sorted(str(key) for key in data if "test" in str(key).lower())
    if forbidden:
        raise ValueError(f"v3 data exposes forbidden official-test fields: {forbidden}")

    training = _mapping(candidate.get("training"), "v3 training")
    groups = training.get("parameter_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("v3 must declare exactly one trainable parameter group")
    group = _mapping(groups[0], "v3 trainable parameter group")
    if list(group.get("prefixes", [])) != [HEAD_PREFIX]:
        raise ValueError("v3 may train only fine_semantic_head.*")

    return {
        "passes": True,
        "source_v2_sha256": source_sha256,
        "identical_sections": ["data", "training", "evaluation"],
        "only_model_difference": "fine_semantic_head_type=spatial_refined",
        "official_test_used": False,
        "auto_promotion": False,
    }


def build_report(
    base_path: Path,
    candidate_path: Path,
    source_recipe_path: Path,
    comparison_report_path: Path | None = None,
) -> dict[str, Any]:
    base_state, base_payload = load_model_state(base_path)
    candidate_state, candidate_payload = load_model_state(candidate_path)
    source_sha256 = file_sha256(source_recipe_path)
    source_recipe = _load_yaml(source_recipe_path)
    candidate_config = _mapping(candidate_payload.get("config"), "candidate config")
    recipe_audit = validate_recipe_contract(
        source_recipe,
        candidate_config,
        source_sha256=source_sha256,
    )
    state_audit = audit_states(base_state, candidate_state)
    metrics = _mapping(candidate_payload.get("metrics"), "candidate metrics")

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base": {
            "path": str(base_path),
            "sha256": file_sha256(base_path),
            "epoch": base_payload.get("epoch"),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
            "epoch": candidate_payload.get("epoch"),
        },
        "source_recipe": {
            "path": str(source_recipe_path),
            "sha256": source_sha256,
        },
        "recipe_audit": recipe_audit,
        "state_audit": state_audit,
        "candidate_metrics": metrics,
        "app_pointer_verification": {
            "status": "external_wrapper_required",
            "performed_by_this_auditor": False,
        },
        "promotion_performed": False,
    }
    if comparison_report_path is not None:
        comparison = json.loads(comparison_report_path.read_text(encoding="utf-8"))
        protocol = _mapping(candidate_config.get("protocol"), "candidate protocol")
        thresholds = _mapping(
            protocol.get("classification_acceptance"),
            "classification acceptance thresholds",
        )
        report["classification_acceptance"] = V2_AUDIT.classification_acceptance(
            metrics,
            _mapping(comparison, "v1 comparison report"),
            thresholds,
        )
        report["comparison_report"] = {
            "path": str(comparison_report_path),
            "sha256": file_sha256(comparison_report_path),
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--source-recipe-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comparison-report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = _resolve(args.base_checkpoint)
    candidate_path = _resolve(args.candidate_checkpoint)
    source_recipe_path = _resolve(args.source_recipe_config)
    for path, role in (
        (base_path, "base checkpoint"),
        (candidate_path, "candidate checkpoint"),
        (source_recipe_path, "source v2 recipe"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{role} does not exist: {path}")
    comparison_path = (
        _resolve(args.comparison_report) if args.comparison_report else None
    )
    if comparison_path is not None and not comparison_path.is_file():
        raise FileNotFoundError(f"comparison report does not exist: {comparison_path}")

    report = build_report(
        base_path,
        candidate_path,
        source_recipe_path,
        comparison_path,
    )
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report["recipe_audit"], indent=2))
    print(json.dumps(report["state_audit"], indent=2))
    if "classification_acceptance" in report:
        print(json.dumps(report["classification_acceptance"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
