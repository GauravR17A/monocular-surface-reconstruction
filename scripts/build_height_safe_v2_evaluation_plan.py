"""Seal a height-safe v2 checkpoint into a read-only full-scene eval plan.

The generated plan contains validation manifests only and keeps the protected
application checkpoint as the first model.  It is consumed by
``evaluate_height_safe_v2_candidate.py`` and cannot promote either model.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PLAN = PROJECT_ROOT / "configs" / "corrected_legacy_triplet_v1.yaml"
PROTECTED_SHA256 = (
    "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
)


def file_sha256(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def mapping(value: Any, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    return dict(value)


def build(checkpoint: Path, output_dir: Path) -> tuple[Path, Path]:
    checkpoint = checkpoint.resolve()
    output_dir = output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload = mapping(payload, "candidate checkpoint")
    config = mapping(payload.get("config"), "candidate embedded config")
    contracts = mapping(config.get("contracts"), "candidate contracts")
    evaluation = mapping(config.get("evaluation"), "candidate evaluation")
    if contracts.get("official_test_used") is not False:
        raise ValueError("Candidate does not certify official-test exclusion")
    if contracts.get("height_unit_status") != "metre_assumed":
        raise ValueError("Candidate is not the explicit metre-assumed dev protocol")
    if evaluation.get("promotion_eligible") is not False:
        raise ValueError("Candidate config must remain non-promotable")
    epoch = int(payload.get("epoch", -1))
    if epoch < 1:
        raise ValueError("Candidate checkpoint has no completed epoch")
    candidate_sha = file_sha256(checkpoint)

    saved_config = checkpoint.parent / "config.yaml"
    if not saved_config.is_file():
        raise FileNotFoundError(saved_config)
    saved = yaml.safe_load(saved_config.read_text(encoding="utf-8"))
    if saved != config:
        raise ValueError("Checkpoint embedded config differs from saved config")

    base = yaml.safe_load(BASE_PLAN.read_text(encoding="utf-8"))
    base = mapping(base, "base corrected full-scene plan")
    protected = deepcopy(mapping(base["models"]["protected"], "protected model"))
    if str(protected.get("checkpoint_sha256")) != PROTECTED_SHA256:
        raise ValueError("Base plan no longer names the protected checkpoint")

    comparison_path = output_dir / "candidate_comparison_seal.json"
    comparison = {
        "schema": "msr.height_safe_v2_candidate_seal.v1",
        "purpose": "checkpoint_identity_only_for_corrected_development_evaluation",
        "official_gamus_test_constructed": False,
        "production_promotion_permitted": False,
        "protected_baseline": {
            "selected_checkpoint_sha256": PROTECTED_SHA256,
            "selected_epoch": int(protected["epoch"]),
        },
        "height_safe_v2_candidate": {
            "selected_checkpoint_sha256": candidate_sha,
            "selected_epoch": epoch,
        },
    }
    atomic_text(
        comparison_path,
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
    )

    plan = deepcopy(base)
    plan["protocol"]["output"] = str(output_dir / "report.json")
    plan["protocol"]["work_dir"] = str(output_dir / "partials")
    plan["protocol"]["comparison_report"] = str(comparison_path)
    plan["protocol"]["comparison_report_sha256"] = file_sha256(comparison_path)
    plan["models"] = {
        "protected": protected,
        "height_safe_v2": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": candidate_sha,
            "epoch": epoch,
            "model_type": str(payload.get("model_type", "")),
            "comparison_section": "height_safe_v2_candidate",
            "saved_config": str(saved_config.resolve()),
            "saved_config_sha256": file_sha256(saved_config.resolve()),
            "expected_model_config": deepcopy(config["model"]),
        },
    }
    plan["gates"] = {
        "candidate_vs_protected_max_regression": deepcopy(
            base["gates"]["candidate_vs_protected_max_regression"]
        )
    }
    plan_path = output_dir / "evaluation_plan.yaml"
    atomic_text(plan_path, yaml.safe_dump(plan, sort_keys=False))
    return plan_path, comparison_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plan_path, comparison_path = build(args.checkpoint, args.output_dir)
    print(
        json.dumps(
            {
                "plan": str(plan_path),
                "plan_sha256": file_sha256(plan_path),
                "comparison_seal": str(comparison_path),
                "comparison_seal_sha256": file_sha256(comparison_path),
                "official_test_used": False,
                "promotion_permitted": False,
            },
            indent=2,
        )
    )
