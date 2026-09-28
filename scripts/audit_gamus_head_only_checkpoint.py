"""Fail-closed audit for a six-class-head-only GAMUS checkpoint.

The candidate is allowed to add and train only ``fine_semantic_head``.  Every
tensor inherited from the protected height model must remain exactly equal.
This script is intentionally independent of the trainer and never changes the
application checkpoint pointer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "msr.gamus_head_only_checkpoint_audit.v1"
ALLOWED_NEW_KEYS = {
    "fine_semantic_head.weight",
    "fine_semantic_head.bias",
}


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


def _number(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{role} must be finite")
    return result


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
    base_keys = set(base_state)
    candidate_keys = set(candidate_state)
    missing = sorted(base_keys - candidate_keys)
    unexpected = sorted(candidate_keys - base_keys - ALLOWED_NEW_KEYS)
    added = sorted(candidate_keys - base_keys)
    if missing:
        raise ValueError(f"candidate is missing inherited tensors: {missing}")
    if unexpected:
        raise ValueError(f"candidate has unexpected new tensors: {unexpected}")
    if set(added) != ALLOWED_NEW_KEYS:
        raise ValueError(
            "candidate must add exactly the six-class head tensors; "
            f"found {added}"
        )

    changed: list[str] = []
    compared_elements = 0
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
            "height/shared tensors changed during head-only training: "
            + ", ".join(changed[:20])
        )

    head_details: dict[str, Any] = {}
    for name in sorted(ALLOWED_NEW_KEYS):
        tensor = candidate_state[name]
        if not torch.isfinite(tensor).all():
            raise ValueError(f"candidate head tensor is non-finite: {name}")
        if torch.count_nonzero(tensor).item() == 0:
            raise ValueError(f"candidate head tensor was not trained: {name}")
        head_details[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "l2_norm": float(torch.linalg.vector_norm(tensor.float()).item()),
        }

    return {
        "passes": True,
        "inherited_tensor_count": len(base_keys),
        "inherited_element_count": compared_elements,
        "changed_inherited_tensors": [],
        "allowed_trained_tensors": head_details,
    }


def classification_acceptance(
    candidate_metrics: Mapping[str, Any],
    comparison_report: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    current = _mapping(
        candidate_metrics.get("six_class_identification"),
        "candidate six-class metrics",
    )
    current_classes = _mapping(current.get("per_class"), "candidate per-class metrics")
    baseline = _mapping(
        comparison_report.get("expanded_classification"),
        "v1 expanded classification",
    )
    baseline_classes = _mapping(baseline.get("per_class"), "v1 per-class metrics")
    baseline_macro = _mapping(baseline.get("macro"), "v1 macro metrics")

    def current_class(name: str, metric: str) -> float:
        return _number(
            _mapping(current_classes.get(name), f"candidate class {name}").get(metric),
            f"candidate {name} {metric}",
        )

    gates: dict[str, dict[str, Any]] = {}

    def minimum(name: str, value: float, limit: float) -> None:
        gates[name] = {
            "value": value,
            "minimum": limit,
            "passes": value >= limit,
        }

    minimum(
        "macro_f1",
        _number(current.get("macro_f1"), "candidate macro_f1"),
        _number(thresholds.get("macro_f1_min"), "macro_f1_min"),
    )
    minimum(
        "water_recall",
        current_class("water", "recall"),
        _number(thresholds.get("water_recall_min"), "water_recall_min"),
    )
    minimum(
        "water_f1",
        current_class("water", "f1"),
        _number(thresholds.get("water_f1_min"), "water_f1_min"),
    )
    minimum(
        "road_f1",
        current_class("roads", "f1"),
        _number(thresholds.get("road_f1_min"), "road_f1_min"),
    )

    max_drop = _number(
        thresholds.get("max_f1_drop_other_classes"),
        "max_f1_drop_other_classes",
    )
    for name in ("ground", "buildings", "low_vegetation", "trees"):
        baseline_value = _number(
            _mapping(baseline_classes.get(name), f"v1 class {name}").get("f1"),
            f"v1 {name} f1",
        )
        value = current_class(name, "f1")
        gates[f"{name}_f1_retention"] = {
            "value": value,
            "v1_value": baseline_value,
            "maximum_drop": max_drop,
            "passes": value >= baseline_value - max_drop,
        }

    return {
        "passes": all(bool(gate["passes"]) for gate in gates.values()),
        "comparison_v1_macro_f1": _number(baseline_macro.get("f1"), "v1 macro f1"),
        "gates": gates,
    }


def build_report(
    base_path: Path,
    candidate_path: Path,
    comparison_report_path: Path | None = None,
) -> dict[str, Any]:
    base_state, base_payload = load_model_state(base_path)
    candidate_state, candidate_payload = load_model_state(candidate_path)
    audit = audit_states(base_state, candidate_state)
    candidate_config = _mapping(candidate_payload.get("config"), "candidate config")
    model_config = _mapping(candidate_config.get("model"), "candidate model config")
    training_config = _mapping(
        candidate_config.get("training"), "candidate training config"
    )
    if int(model_config.get("fine_semantic_classes", 0)) != 6:
        raise ValueError("candidate config does not declare six fine classes")
    groups = training_config.get("parameter_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("candidate must contain exactly one explicit parameter group")
    group = _mapping(groups[0], "candidate parameter group")
    if list(group.get("prefixes", [])) != ["fine_semantic_head."]:
        raise ValueError("candidate parameter group is not fine-head-only")

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
        "state_audit": audit,
        "candidate_metrics": metrics,
        # This standalone auditor deliberately never reads or mutates the live
        # application pointer.  The PowerShell run wrapper performs the
        # before/after pointer and protected-checkpoint hash verification.
        # Do not claim a global pointer result that was not observed here.
        "app_pointer_verification": {
            "status": "external_wrapper_required",
            "performed_by_this_auditor": False,
        },
        "promotion_performed": False,
    }
    if comparison_report_path is not None:
        protocol = _mapping(candidate_config.get("protocol"), "candidate protocol")
        thresholds = _mapping(
            protocol.get("classification_acceptance"),
            "classification acceptance thresholds",
        )
        comparison = json.loads(comparison_report_path.read_text(encoding="utf-8"))
        report["classification_acceptance"] = classification_acceptance(
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--comparison-report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = _resolve(args.base_checkpoint)
    candidate_path = _resolve(args.candidate_checkpoint)
    if not base_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError("base and candidate checkpoints must both exist")
    comparison_path = (
        _resolve(args.comparison_report) if args.comparison_report else None
    )
    if comparison_path is not None and not comparison_path.is_file():
        raise FileNotFoundError(f"comparison report does not exist: {comparison_path}")
    report = build_report(base_path, candidate_path, comparison_path)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report["state_audit"], indent=2))
    if "classification_acceptance" in report:
        print(json.dumps(report["classification_acceptance"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
