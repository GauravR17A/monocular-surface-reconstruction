"""Train the isolated Stage-4b semantic domain/OOD controller.

Only authenticated Stage-3 TRAIN descriptors are fitted.  Stage-4 v3
calibration is reused once as development proof and is explicitly not treated
as fresh validation.  Official test splits and application pointer contents
are never opened; the pointer is only SHA-256 checked before and after work.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msr.training.scene_router import read_scene_router_records  # noqa: E402
from msr.training.scene_router_record_store import (  # noqa: E402
    RECORD_STORE_SCHEMA as STAGE3_STORE_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    validate_record_selection_provenance,
)
from msr.training.stage4_dual_router_record_store import (  # noqa: E402
    load_completed_stage4_record_store,
)
from msr.training.stage4b_semantic_ood_router import (  # noqa: E402
    DESCRIPTOR_CONTRACT,
    DEVELOPMENT_DISCLAIMER,
    DevelopmentGateConfig,
    FitNormalizedLinearClassifier,
    LOWER_ENSEMBLE_FORMULA,
    LinearTrainingConfig,
    STAGE4B_ARTIFACT_SCHEMA,
    STAGE4B_COMPLETION_SCHEMA,
    STAGE4B_REPORT_SCHEMA,
    Stage4bSemanticOODRouter,
    build_stage4b_training_examples,
    fit_ood_guard,
    make_group_held_out_folds,
    select_development_operating_point,
    state_sha256,
    train_linear_member,
)


CONFIG_SECTION = "stage4b_semantic_ood_router"
TOP_LEVEL_FIELDS = {
    "stage3_record_store",
    "stage4_development_store",
    "output_dir",
    "app_pointer",
    "app_pointer_sha256",
    "required_gamus_cities",
    "folds",
    "seeds",
    "ensemble_std_multiplier",
    "classifier",
    "ood",
    "development_gates",
}
CLASSIFIER_FIELDS = {"epochs", "learning_rate", "weight_decay", "batch_size"}
OOD_FIELDS = {"shrinkage", "cutoff_quantile", "minimum_distance_margin"}
GATE_FIELDS = {
    "minimum_gamus_coverage",
    "minimum_gamus_selections",
    "minimum_global_exact_gain",
    "minimum_gamus_exact_gain",
    "minimum_stratum_support",
    "minimum_decision_consistency",
    "minimum_candidate_gain_for_reporting",
}


def _resolved(value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _require_sha256(value: object, name: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = _resolved(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(value.get(CONFIG_SECTION), Mapping):
        raise ValueError(f"config must contain {CONFIG_SECTION!r}")
    config = dict(value[CONFIG_SECTION])
    unknown = sorted(set(config) - TOP_LEVEL_FIELDS)
    missing = sorted(TOP_LEVEL_FIELDS - set(config))
    if unknown or missing:
        raise ValueError(f"Stage-4b config mismatch; unknown={unknown}, missing={missing}")
    for key in ("stage3_record_store", "stage4_development_store", "output_dir", "app_pointer"):
        config[key] = _resolved(config[key])
    config["app_pointer_sha256"] = _require_sha256(
        config["app_pointer_sha256"], "app_pointer_sha256"
    )
    for section, fields in (
        ("classifier", CLASSIFIER_FIELDS),
        ("ood", OOD_FIELDS),
        ("development_gates", GATE_FIELDS),
    ):
        if not isinstance(config[section], Mapping) or set(config[section]) != fields:
            raise ValueError(f"Stage-4b {section} fields must be exactly {sorted(fields)}")
        config[section] = dict(config[section])
    config["config_path"] = config_path
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("folds") != 3:
        raise ValueError("Stage-4b requires exactly three geographic folds")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 5
        or len(set(seeds)) != 5
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("Stage-4b requires exactly five unique integer seeds")
    cities = config.get("required_gamus_cities")
    if sorted(str(city).lower() for city in cities) != ["dc", "nyc", "phl"]:
        raise ValueError("Stage-4b must fit all GAMUS cities: DC, NYC, PHL")
    z = float(config["ensemble_std_multiplier"])
    if not math.isfinite(z) or z < 1.0:
        raise ValueError("Stage-4b lower-ensemble multiplier must be at least 1")
    classifier = config["classifier"]
    LinearTrainingConfig(
        epochs=int(classifier["epochs"]),
        learning_rate=float(classifier["learning_rate"]),
        weight_decay=float(classifier["weight_decay"]),
        batch_size=int(classifier["batch_size"]),
    )
    ood = config["ood"]
    if not 0.0 < float(ood["shrinkage"]) <= 1.0:
        raise ValueError("OOD shrinkage must be within (0, 1]")
    if not 0.9 <= float(ood["cutoff_quantile"]) <= 1.0:
        raise ValueError("OOD cutoff quantile must be conservative")
    DevelopmentGateConfig(**config["development_gates"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train isolated Stage-4b semantic OOD router")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "stage4b_semantic_ood_router.yaml",
    )
    return parser.parse_args(argv)


def _read_json(path: Path, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a mapping")
    return value


def load_authenticated_stage3(
    root: Path, *, stage4_ancestry: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Authenticate Stage-3 through its completion proof and Stage-4 ancestry."""

    provenance_path = root / "provenance.json"
    completion_path = root / "completion.json"
    records_path = root / "records.jsonl"
    provenance = _read_json(provenance_path, "Stage-3 provenance")
    completion = _read_json(completion_path, "Stage-3 completion")
    if provenance.get("schema") != STAGE3_STORE_SCHEMA or completion.get("schema") != STAGE3_STORE_SCHEMA:
        raise ValueError("unsupported Stage-3 record store")
    if provenance.get("test_splits_excluded") is not True:
        raise ValueError("Stage-3 provenance does not exclude official test splits")
    data = provenance.get("data")
    validate_record_selection_provenance(data, completed_record_count=completion.get("record_count"))
    if data.get("excluded_source_splits") != ["gamus/test", "legacy/test"]:
        raise ValueError("Stage-3 official test exclusions changed")
    generation = provenance.get("generation_config")
    if not isinstance(generation, Mapping) or generation.get("descriptor_mask") != DESCRIPTOR_CONTRACT:
        raise ValueError("Stage-3 descriptors violate the all-pixel serving contract")
    checks = {
        "provenance_sha256": file_sha256(provenance_path),
        "provenance_canonical_sha256": canonical_json_sha256(provenance),
        "completion_sha256": file_sha256(completion_path),
        "completion_canonical_sha256": canonical_json_sha256(completion),
        "records_sha256": file_sha256(records_path),
    }
    for name, actual in checks.items():
        if stage4_ancestry.get(name) != actual:
            raise ValueError(f"Stage-3 {name} differs from authenticated Stage-4 ancestry")
    records = read_scene_router_records(records_path)
    if len(records) != completion.get("record_count") or len(records) != stage4_ancestry.get("record_count"):
        raise ValueError("Stage-3 record count differs from its proofs")
    keys = [record.sample_id for record in records]
    if canonical_json_sha256(keys) != stage4_ancestry.get("record_keys_sha256"):
        raise ValueError("Stage-3 record ordering changed")
    return records, {
        "schema": STAGE3_STORE_SCHEMA,
        "provenance_sha256": checks["provenance_sha256"],
        "provenance_canonical_sha256": checks["provenance_canonical_sha256"],
        "completion_sha256": checks["completion_sha256"],
        "completion_canonical_sha256": checks["completion_canonical_sha256"],
        "records_sha256": checks["records_sha256"],
        "record_count": len(records),
        "descriptor_contract": DESCRIPTOR_CONTRACT,
        "test_splits_used": False,
    }


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    positive = int(target.sum())
    negative = len(target) - positive
    if not positive or not negative:
        return float("nan")
    return float((ranks[target].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _member_audit(
    member: FitNormalizedLinearClassifier,
    *,
    fold: int,
    seed: int,
    target: np.ndarray,
    probability: np.ndarray,
    final_loss: float,
) -> dict[str, Any]:
    prediction = probability >= 0.5
    return {
        "fold": fold,
        "seed": seed,
        "state_sha256": state_sha256(member.state_dict()),
        "fit_normalizer_sha256": state_sha256(
            {"feature_mean": member.feature_mean, "feature_scale": member.feature_scale}
        ),
        "heldout_auc": _binary_auc(target, probability),
        "heldout_accuracy_at_0_5": float(np.mean(prediction == target)),
        "heldout_gamus_mean_probability": float(probability[target].mean()),
        "heldout_legacy_mean_probability": float(probability[~target].mean()),
        "final_train_bce": final_loss,
    }


def _clone_router(router: Stage4bSemanticOODRouter) -> Stage4bSemanticOODRouter:
    members = [
        FitNormalizedLinearClassifier(member.feature_mean, member.feature_scale)
        for member in router.members
    ]
    from msr.training.stage4b_semantic_ood_router import MahalanobisFit

    ood = MahalanobisFit(
        router.ood_feature_mean,
        router.ood_feature_scale,
        router.ood_gamus_center,
        router.ood_legacy_center,
        router.ood_precision,
    )
    clone = Stage4bSemanticOODRouter(
        members,
        ood,
        ensemble_std_multiplier=float(router.ensemble_std_multiplier),
        ood_max_distance=float(router.ood_max_distance),
        minimum_distance_margin=float(router.minimum_distance_margin),
    )
    clone.load_state_dict(router.state_dict(), strict=True)
    clone.eval()
    return clone


def _decision_checks(
    router: Stage4bSemanticOODRouter, descriptors: torch.Tensor
) -> dict[str, Any]:
    with torch.no_grad():
        first = router.forward_descriptor(descriptors)
        second = router.forward_descriptor(descriptors.clone())
        order = torch.arange(len(descriptors) - 1, -1, -1)
        permuted = router.forward_descriptor(descriptors[order])
        restored = permuted["semantic_candidate_selected"][order]
    return {
        "repeat_scores_bit_identical": bool(
            torch.equal(first["lower_probability"], second["lower_probability"])
        ),
        "repeat_decisions_identical": bool(
            torch.equal(
                first["semantic_candidate_selected"],
                second["semantic_candidate_selected"],
            )
        ),
        "batch_permutation_decisions_identical": bool(
            torch.equal(first["semantic_candidate_selected"], restored)
        ),
        "height_route_enabled": router.height_route_enabled,
        "height_selections": int(first["height_candidate_selected"].sum()),
    }


def _corruption_check(router: Stage4bSemanticOODRouter) -> dict[str, Any]:
    mean = router.ood_feature_mean
    scale = router.ood_feature_scale
    alternating = torch.where(
        torch.arange(router.descriptor_size) % 2 == 0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    corrupted = torch.stack(
        (mean + 20 * scale, mean - 20 * scale, mean + 20 * scale * alternating)
    )
    with torch.no_grad():
        scored = router.score_descriptor(corrupted)
    would_select = (
        scored["ood_accepted"]
        & scored["distance_margin_accepted"]
        & (scored["lower_probability"] >= router.semantic_decision_threshold)
    )
    return {
        "suite": "fit-normalizer mean +/- 20 sigma deterministic probes",
        "probe_count": len(corrupted),
        "selected": int(would_select.sum()),
        "selection_rate": float(would_select.float().mean()),
        "all_rejected": not bool(torch.any(would_select)),
    }


def _atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    config = load_config(parse_args(argv).config)
    pointer = Path(config["app_pointer"])
    pointer_before = file_sha256(pointer)
    if pointer_before != config["app_pointer_sha256"]:
        raise ValueError("protected app pointer hash changed before Stage-4b")

    output_dir = Path(config["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Stage-4b output: {output_dir}")

    development_records, development_provenance, development_completion = (
        load_completed_stage4_record_store(config["stage4_development_store"])
    )
    if development_provenance.get("test_splits_excluded") is not True or development_completion.get("test_splits_excluded") is not True:
        raise ValueError("Stage-4 development evidence does not exclude tests")
    ancestry = development_provenance.get("stage3")
    if not isinstance(ancestry, Mapping) or ancestry.get("authenticated") is not True:
        raise ValueError("Stage-4 development store lacks authenticated Stage-3 ancestry")
    stage3_records, stage3_audit = load_authenticated_stage3(
        Path(config["stage3_record_store"]), stage4_ancestry=ancestry
    )
    legacy_groups = {
        record.sample_id: record.group_id
        for record in development_records
        if record.partition == "train" and record.source == "legacy"
    }
    examples, example_audit = build_stage4b_training_examples(
        stage3_records,
        legacy_group_ids=legacy_groups,
        required_gamus_cities=config["required_gamus_cities"],
    )
    descriptors = torch.stack([example.descriptor for example in examples])
    targets = torch.tensor([example.candidate_domain for example in examples], dtype=torch.float32)
    folds, fold_audit = make_group_held_out_folds(
        examples, folds=int(config["folds"]), seed=min(config["seeds"])
    )

    classifier_config = LinearTrainingConfig(**config["classifier"])
    members: list[FitNormalizedLinearClassifier] = []
    member_audits: list[dict[str, Any]] = []
    oof_seed_scores = torch.full((len(examples), len(config["seeds"])), float("nan"))
    for fold in folds:
        train_index = torch.tensor(fold.train_indices, dtype=torch.long)
        validation_index = torch.tensor(fold.validation_indices, dtype=torch.long)
        validation_target = targets[validation_index].bool().numpy()
        for seed_index, seed in enumerate(config["seeds"]):
            member, history = train_linear_member(
                descriptors[train_index],
                targets[train_index],
                seed=int(seed) + 1009 * fold.index,
                config=classifier_config,
            )
            with torch.no_grad():
                probability = torch.sigmoid(member(descriptors[validation_index]))
            oof_seed_scores[validation_index, seed_index] = probability
            members.append(member)
            member_audits.append(
                _member_audit(
                    member,
                    fold=fold.index,
                    seed=int(seed),
                    target=validation_target,
                    probability=probability.numpy(),
                    final_loss=float(history[-1]["train_bce"]),
                )
            )
    if not bool(torch.all(torch.isfinite(oof_seed_scores))):
        raise RuntimeError("Stage-4b OOF ensemble did not score every TRAIN scene")
    z = float(config["ensemble_std_multiplier"])
    oof_lower = (
        oof_seed_scores.mean(dim=1) - z * oof_seed_scores.std(dim=1, unbiased=False)
    ).clamp(0, 1)
    ood_fit, ood_cutoff, distance_margin, ood_audit = fit_ood_guard(
        descriptors,
        targets,
        folds,
        shrinkage=float(config["ood"]["shrinkage"]),
        quantile=float(config["ood"]["cutoff_quantile"]),
        minimum_distance_margin=float(config["ood"]["minimum_distance_margin"]),
    )
    router = Stage4bSemanticOODRouter(
        members,
        ood_fit,
        ensemble_std_multiplier=z,
        ood_max_distance=ood_cutoff,
        minimum_distance_margin=distance_margin,
    )
    router.eval()

    calibration_records = [
        record for record in development_records if record.partition == "calibration"
    ]
    calibration_descriptors = torch.stack([record.descriptor for record in calibration_records])
    with torch.no_grad():
        development_score = router.score_descriptor(calibration_descriptors)
    operating_point = select_development_operating_point(
        calibration_records,
        lower_probabilities=development_score["lower_probability"].numpy(),
        member_probabilities=development_score["member_probabilities"].numpy(),
        ood_accepted=development_score["ood_accepted"].numpy(),
        margin_accepted=development_score["distance_margin_accepted"].numpy(),
        std_multiplier=z,
        config=DevelopmentGateConfig(**config["development_gates"]),
    )
    router.set_semantic_operating_point(
        threshold=operating_point.threshold, enabled=operating_point.eligible
    )
    consistency = _decision_checks(router, calibration_descriptors)
    corruption = _corruption_check(router)
    post_checks = {
        "repeat_and_permutation_consistent": all(
            consistency[key]
            for key in (
                "repeat_scores_bit_identical",
                "repeat_decisions_identical",
                "batch_permutation_decisions_identical",
            )
        ),
        "height_hard_disabled": consistency["height_route_enabled"] is False
        and consistency["height_selections"] == 0,
        "corruption_suite_rejected": corruption["all_rejected"],
    }
    semantic_enabled = bool(operating_point.eligible and all(post_checks.values()))
    if not semantic_enabled:
        router.set_semantic_operating_point(
            threshold=operating_point.threshold, enabled=False
        )

    pointer_after_fit = file_sha256(pointer)
    if pointer_after_fit != pointer_before:
        raise RuntimeError("protected app pointer changed during Stage-4b fitting")
    clone = _clone_router(router)
    with torch.no_grad():
        original = router.forward_descriptor(calibration_descriptors)
        reloaded = clone.forward_descriptor(calibration_descriptors)
    roundtrip = {
        "state_dict_strict_reload": True,
        "lower_probability_bit_identical": bool(
            torch.equal(original["lower_probability"], reloaded["lower_probability"])
        ),
        "semantic_decisions_identical": bool(
            torch.equal(
                original["semantic_candidate_selected"],
                reloaded["semantic_candidate_selected"],
            )
        ),
    }
    if not all(roundtrip.values()):
        raise RuntimeError("Stage-4b state round-trip changed decisions")

    oof_target = targets.bool().numpy()
    oof_audit = {
        "scene_count": len(examples),
        "gamus_count": int(oof_target.sum()),
        "legacy_count": int((~oof_target).sum()),
        "lower_probability_formula": LOWER_ENSEMBLE_FORMULA,
        "lower_probability_mean": float(oof_lower.mean()),
        "domain_auc": _binary_auc(oof_target, oof_lower.numpy()),
        "per_seed_scores_shape": list(oof_seed_scores.shape),
        "member_audits": member_audits,
    }
    config_path = Path(config["config_path"])
    training_launch = {
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "trainer_sha256": file_sha256(Path(__file__).resolve()),
        "architecture_sha256": file_sha256(
            PROJECT_ROOT
            / "src"
            / "msr"
            / "training"
            / "stage4b_semantic_ood_router.py"
        ),
        "folds": 3,
        "seeds": list(config["seeds"]),
        "fixed_epochs_no_development_peeking": classifier_config.epochs,
    }
    development_proof = {
        "role": "reused_development_proof_only",
        "disclaimer": DEVELOPMENT_DISCLAIMER,
        "fresh_holdout_required": True,
        "records": len(calibration_records),
        "operating_point": {
            "threshold": operating_point.threshold,
            "eligible_before_post_checks": operating_point.eligible,
            "reason": operating_point.reason,
            "evaluated_thresholds": operating_point.evaluated_thresholds,
            "metrics": operating_point.metrics,
        },
        "decision_consistency": consistency,
        "corruption_check": corruption,
        "post_checks": post_checks,
    }
    payload: dict[str, Any] = {
        "artifact_schema": STAGE4B_ARTIFACT_SCHEMA,
        "artifact_type": "offline_semantic_domain_ood_router_only",
        "descriptor_size": router.descriptor_size,
        "member_count": len(router.members),
        "router_state_dict": {
            name: value.detach().cpu() for name, value in router.state_dict().items()
        },
        "semantic_route_enabled": semantic_enabled,
        "height_route_enabled": False,
        "safe_to_evaluate": True,
        "fresh_holdout_required": True,
        "development_evidence_reused": True,
        "development_proof": development_proof,
        "training_examples": example_audit,
        "group_holdout": fold_audit,
        "oof_training": oof_audit,
        "ood_guard": ood_audit,
        "stage3_training_ancestry": stage3_audit,
        "stage4_development_provenance": development_provenance,
        "stage4_development_completion": development_completion,
        "stage4_development_provenance_canonical_sha256": canonical_json_sha256(
            development_provenance
        ),
        "stage4_development_completion_canonical_sha256": canonical_json_sha256(
            development_completion
        ),
        "app_pointer": {"path": str(pointer), "sha256": pointer_before},
        "training_launch": training_launch,
        "state_roundtrip": roundtrip,
        "test_splits_used": False,
        "live_application_pointer_changed": False,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.pending-", dir=output_dir.parent)
    )
    try:
        checkpoint = staging / "stage4b_semantic_ood_router.pt"
        report_path = staging / "stage4b_semantic_ood_router_report.json"
        completion_path = staging / "completion.json"
        _atomic_torch_save(payload, checkpoint)
        artifact_hash = file_sha256(checkpoint)
        report = {
            **{key: value for key, value in payload.items() if key != "router_state_dict"},
            "artifact_schema": STAGE4B_REPORT_SCHEMA,
            "router_artifact_path": str(
                (output_dir / "stage4b_semantic_ood_router.pt").resolve()
            ),
            "router_artifact_sha256": artifact_hash,
        }
        _atomic_json_save(report, report_path)
        completion_manifest = {
            "schema": STAGE4B_COMPLETION_SCHEMA,
            "artifact_schema": STAGE4B_ARTIFACT_SCHEMA,
            "report_schema": STAGE4B_REPORT_SCHEMA,
            "artifact_sha256": artifact_hash,
            "report_sha256": file_sha256(report_path),
            "semantic_route_enabled": semantic_enabled,
            "height_route_enabled": False,
            "fresh_holdout_required": True,
            "test_splits_used": False,
            "live_application_pointer_changed": False,
            "complete": True,
        }
        _atomic_json_save(completion_manifest, completion_path)
        if file_sha256(pointer) != pointer_before:
            raise RuntimeError("protected app pointer changed before Stage-4b publication")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Stage-4b artifact: {output_dir / 'stage4b_semantic_ood_router.pt'}")
    print(f"Semantic route enabled on reused development proof: {semantic_enabled}")
    print("Height route enabled: False; official test used: False; app pointer untouched")


if __name__ == "__main__":
    main()
