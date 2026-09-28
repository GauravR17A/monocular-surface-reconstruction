"""Train versioned Stage-4 semantic and height endpoint-gain routers offline.

The command consumes only a completed, authenticated Stage-4 rich-record
journal.  It never loads an application pointer, never edits endpoint
checkpoints, and never accesses official test splits.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msr.training.scene_router_record_store import (  # noqa: E402
    canonical_json_sha256,
    file_sha256,
)
from msr.training.stage4_dual_router import (  # noqa: E402
    BalancedStage4Sampler,
    EndpointGainRouter,
    GainRouterTrainingConfig,
    STAGE4_ARTIFACT_SCHEMA,
    STAGE4_RECORD_SCHEMA,
    STAGE4_RECORD_STORE_SCHEMA,
    STAGE4_REPORT_SCHEMA,
    Stage4GainDataset,
    partition_group_held_out_records,
    train_gain_router,
)
from msr.training.stage4_dual_router_record_store import (  # noqa: E402
    load_completed_stage4_record_store,
    validate_stage4_record_selection_provenance,
)


CONFIG_SECTION = "stage4_dual_router_training"
TOP_LEVEL_FIELDS = {
    "record_store",
    "output_dir",
    "batch_size",
    "hidden_features",
    "epochs",
    "learning_rate",
    "weight_decay",
    "huber_delta",
    "seed",
    "device",
    "require_group_disjoint",
    "height",
    "semantic",
}
HEAD_FIELDS = {
    "target_scale",
    "minimum_score",
    "minimum_candidate_gain",
    "minimum_precision",
    "minimum_candidate_selections",
    "minimum_stratum_support",
    "maximum_mean_utility_regression",
    "maximum_stratum_utility_regression",
}
DESCRIPTOR_MASK_CONTRACT = "full_center_crop_all_pixels_no_reference_mask"
LEGACY_PRIOR_CONTRACT = "stored_01"
COMPLETION_SCHEMA = "msr.stage4_dual_endpoint_router_completion.v1"


def _resolved(path_like: object) -> Path:
    path = Path(str(path_like)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_training_config(path: str | Path) -> dict[str, Any]:
    config_path = _resolved(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(value.get(CONFIG_SECTION), Mapping):
        raise ValueError(f"training config must contain {CONFIG_SECTION!r}")
    config = dict(value[CONFIG_SECTION])
    unknown = sorted(set(config) - TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"unknown Stage-4 training config fields: {unknown}")
    missing_top = sorted(TOP_LEVEL_FIELDS - set(config))
    if missing_top:
        raise ValueError(f"Stage-4 training config is missing: {missing_top}")
    for name in ("record_store", "output_dir"):
        if name not in config:
            raise ValueError(f"Stage-4 training config requires {name}")
        config[name] = _resolved(config[name])
    for component in ("height", "semantic"):
        head = config.get(component)
        if not isinstance(head, Mapping):
            raise ValueError(f"Stage-4 training config requires {component} mapping")
        head = dict(head)
        unknown_head = sorted(set(head) - HEAD_FIELDS)
        if unknown_head:
            raise ValueError(
                f"unknown Stage-4 {component} config fields: {unknown_head}"
            )
        missing = sorted(HEAD_FIELDS - set(head))
        if missing:
            raise ValueError(f"Stage-4 {component} config is missing: {missing}")
        config[component] = head
    config["config_path"] = config_path
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train isolated Stage-4 semantic and height gain routers."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "stage4_dual_router_training.yaml",
    )
    return parser.parse_args(argv)


def _validate_number(
    value: object, *, name: str, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} cannot be boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def validate_training_config(config: Mapping[str, Any]) -> None:
    for name in ("batch_size", "hidden_features", "epochs"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _validate_number(config.get("learning_rate"), name="learning_rate", positive=True)
    _validate_number(config.get("weight_decay"), name="weight_decay", nonnegative=True)
    _validate_number(config.get("huber_delta"), name="huber_delta", positive=True)
    if config.get("require_group_disjoint") is not True:
        raise ValueError("checked Stage-4 training requires source/group holdout")
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    if not isinstance(config.get("device"), str) or not str(config["device"]).strip():
        raise ValueError("device must be a non-empty string")
    for component, expected_gain in (("height", 0.25), ("semantic", 0.01)):
        head = config[component]
        if float(head["minimum_candidate_gain"]) != expected_gain:
            raise ValueError(
                f"{component} minimum candidate gain must remain {expected_gain}"
            )
        if float(head["minimum_precision"]) < 0.98:
            raise ValueError(f"{component} precision guard must be at least 0.98")
        for field in ("minimum_candidate_selections", "minimum_stratum_support"):
            value = head[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 32:
                raise ValueError(f"{component} {field} must be at least 32")
        for field in (
            "maximum_mean_utility_regression",
            "maximum_stratum_utility_regression",
        ):
            if float(head[field]) != 0.0:
                raise ValueError(f"{component} {field} must remain zero")
        _validate_number(head["target_scale"], name=f"{component}.target_scale", positive=True)
        _validate_number(head["minimum_score"], name=f"{component}.minimum_score")


def validate_record_evidence(
    records: Sequence[Any],
    provenance: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> None:
    if provenance.get("schema") != STAGE4_RECORD_STORE_SCHEMA:
        raise ValueError("unsupported Stage-4 record-store schema")
    if provenance.get("record_schema") != STAGE4_RECORD_SCHEMA:
        raise ValueError("record store is not the rich Stage-4 v2 schema")
    if provenance.get("test_splits_excluded") is not True or completion.get(
        "test_splits_excluded"
    ) is not True:
        raise ValueError("Stage-4 training evidence does not exclude test splits")
    if provenance.get("app_pointer_changed") is not False or completion.get(
        "app_pointer_changed"
    ) is not False:
        raise ValueError("Stage-4 record preparation touched the app pointer")
    if (
        completion.get("schema") != STAGE4_RECORD_STORE_SCHEMA
        or completion.get("record_schema") != STAGE4_RECORD_SCHEMA
        or completion.get("complete") is not True
        or completion.get("identity_sha256") != provenance.get("identity_sha256")
    ):
        raise ValueError("Stage-4 completion proof identity is invalid")
    generation = provenance.get("generation_config")
    if not isinstance(generation, Mapping):
        raise ValueError("Stage-4 generation configuration is missing")
    if generation.get("descriptor_mask") != DESCRIPTOR_MASK_CONTRACT:
        raise ValueError("Stage-4 descriptors are not deployable all-pixel descriptors")
    if generation.get("legacy_relative_prior_policy") != LEGACY_PRIOR_CONTRACT:
        raise ValueError("Stage-4 legacy priors do not use stored 0..1 inputs")
    expected_generation_contracts = {
        "descriptor_source": "authenticated_stage3_records.jsonl",
        "height_evidence": "float64_sse+count+rmse_on_dataset_regression_mask",
        "semantic_evidence": (
            "reference_row_prediction_column_3x3_confusion+count+macro_error"
        ),
    }
    for name, expected in expected_generation_contracts.items():
        if generation.get(name) != expected:
            raise ValueError(f"Stage-4 generation contract mismatch: {name}")
    data = provenance.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Stage-4 data provenance is missing")
    stage3_data = data.get("stage3_data_provenance")
    if not isinstance(stage3_data, Mapping):
        raise ValueError("Stage-4 data provenance has no Stage-3 data ancestry")
    included = stage3_data.get("included_source_splits")
    excluded = stage3_data.get("excluded_source_splits")
    if not isinstance(included, list) or any(
        str(item).lower().endswith("/test") for item in included
    ):
        raise ValueError("Stage-4 included source splits are not test-safe")
    if excluded != ["gamus/test", "legacy/test"]:
        raise ValueError("Stage-4 provenance must exclude both official test splits")
    validate_stage4_record_selection_provenance(
        data, completed_record_count=completion.get("record_count")
    )
    ancestry = provenance.get("stage3")
    if not isinstance(ancestry, Mapping) or ancestry.get("authenticated") is not True:
        raise ValueError("Stage-4 records have no authenticated Stage-3 ancestry")
    endpoints = provenance.get("endpoints")
    if not isinstance(endpoints, Mapping) or not {
        "protected",
        "gamus_stage1",
        "compatibility",
    }.issubset(endpoints):
        raise ValueError("Stage-4 endpoint identities are incomplete")
    if int(completion.get("record_count", -1)) != len(records):
        raise ValueError("Stage-4 completion count differs from loaded records")
    descriptor_sizes = {record.descriptor.numel() for record in records}
    if len(descriptor_sizes) != 1 or int(completion.get("descriptor_size", -1)) not in descriptor_sizes:
        raise ValueError("Stage-4 descriptor size differs from completion proof")


def _head_training_config(
    common: Mapping[str, Any], head: Mapping[str, Any]
) -> GainRouterTrainingConfig:
    return GainRouterTrainingConfig(
        epochs=int(common["epochs"]),
        learning_rate=float(common["learning_rate"]),
        weight_decay=float(common["weight_decay"]),
        huber_delta=float(common["huber_delta"]),
        minimum_score=float(head["minimum_score"]),
        minimum_candidate_gain=float(head["minimum_candidate_gain"]),
        minimum_precision=float(head["minimum_precision"]),
        minimum_candidate_selections=int(head["minimum_candidate_selections"]),
        minimum_stratum_support=int(head["minimum_stratum_support"]),
        maximum_mean_utility_regression=float(
            head["maximum_mean_utility_regression"]
        ),
        maximum_stratum_utility_regression=float(
            head["maximum_stratum_utility_regression"]
        ),
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_training_config(args.config)
    validate_training_config(config)

    torch.manual_seed(int(config["seed"]))
    if str(config["device"]).startswith("cuda"):
        torch.cuda.manual_seed_all(int(config["seed"]))
    torch.use_deterministic_algorithms(True)

    output_dir = Path(config["output_dir"])
    checkpoint_path = output_dir / "stage4_dual_router.pt"
    report_path = output_dir / "stage4_dual_router_report.json"
    completion_path = output_dir / "completion.json"
    if output_dir.exists():
        raise FileExistsError(
            "refusing to overwrite an existing Stage-4 output directory: "
            + str(output_dir)
        )

    records, provenance, completion = load_completed_stage4_record_store(
        config["record_store"]
    )
    validate_record_evidence(records, provenance, completion)
    train_records, calibration_records, group_evidence = (
        partition_group_held_out_records(
            records, require_group_disjoint=bool(config["require_group_disjoint"])
        )
    )
    descriptor_size = int(completion["descriptor_size"])
    component_results: dict[str, Any] = {}
    routers: dict[str, EndpointGainRouter] = {}
    for component in ("height", "semantic"):
        train_dataset = Stage4GainDataset(train_records, component=component)
        calibration_dataset = Stage4GainDataset(
            calibration_records, component=component
        )
        sampler = BalancedStage4Sampler(
            train_dataset.records, seed=int(config["seed"])
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            sampler=sampler,
            num_workers=0,
        )
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=0,
        )
        head = config[component]
        router = EndpointGainRouter(
            descriptor_size,
            hidden_features=int(config["hidden_features"]),
            target_scale=float(head["target_scale"]),
        )
        training_config = _head_training_config(config, head)
        result = train_gain_router(
            router,
            train_loader,
            calibration_loader,
            component=component,
            config=training_config,
            device=str(config["device"]),
        )
        routers[component] = router
        component_results[component] = {
            "training_config": asdict(training_config),
            "records": {
                "train": len(train_dataset),
                "calibration": len(calibration_dataset),
            },
            "balanced_train_strata": {
                name: len(indices) for name, indices in sampler.groups.items()
            },
            "checkpoint_selection": "fixed_final_epoch_no_calibration_peeking",
            "calibration_gain_mae": result.calibration_gain_mae,
            "calibration_gain_rmse": result.calibration_gain_rmse,
            "history": list(result.history),
            "threshold": {
                "value": result.threshold.threshold,
                "eligible": result.threshold.eligible,
                "reason": result.threshold.reason,
                "metrics": result.threshold.metrics,
                "evaluated_thresholds": result.threshold.evaluated_thresholds,
            },
        }

    eligible_components = [
        name
        for name in ("height", "semantic")
        if bool(component_results[name]["threshold"]["eligible"])
    ]
    any_route_enabled = bool(eligible_components)
    both_routes_enabled = len(eligible_components) == 2
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
            / "stage4_dual_router.py"
        ),
        "device": str(config["device"]),
        "seed": int(config["seed"]),
    }
    payload = {
        "artifact_schema": STAGE4_ARTIFACT_SCHEMA,
        "artifact_type": "offline_dual_endpoint_gain_router_only",
        "descriptor_size": descriptor_size,
        "hidden_features": int(config["hidden_features"]),
        "height_router_state_dict": {
            name: value.detach().cpu()
            for name, value in routers["height"].state_dict().items()
        },
        "semantic_router_state_dict": {
            name: value.detach().cpu()
            for name, value in routers["semantic"].state_dict().items()
        },
        "component_results": component_results,
        "safe_to_evaluate": True,
        "eligible_components": eligible_components,
        "any_route_enabled": any_route_enabled,
        "both_routes_enabled": both_routes_enabled,
        "record_provenance": provenance,
        "record_completion": completion,
        "record_provenance_canonical_sha256": canonical_json_sha256(provenance),
        "record_completion_canonical_sha256": canonical_json_sha256(completion),
        "group_holdout": group_evidence,
        "training_launch": training_launch,
        "test_splits_used": False,
        "live_application_pointer_changed": False,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.pending-", dir=output_dir.parent
        )
    )
    staging_checkpoint = staging_dir / checkpoint_path.name
    staging_report = staging_dir / report_path.name
    staging_completion = staging_dir / completion_path.name
    _atomic_torch_save(payload, staging_checkpoint)
    artifact_sha256 = file_sha256(staging_checkpoint)
    report = {
        "artifact_schema": STAGE4_REPORT_SCHEMA,
        "artifact_type": "offline_dual_endpoint_gain_router_report",
        "router_artifact_path": str(checkpoint_path.resolve()),
        "router_artifact_sha256": artifact_sha256,
        "safe_to_evaluate": True,
        "eligible_components": eligible_components,
        "any_route_enabled": any_route_enabled,
        "both_routes_enabled": both_routes_enabled,
        "component_results": component_results,
        "group_holdout": group_evidence,
        "record_provenance": provenance,
        "record_completion": completion,
        "record_provenance_canonical_sha256": canonical_json_sha256(provenance),
        "record_completion_canonical_sha256": canonical_json_sha256(completion),
        "training_launch": training_launch,
        "test_splits_used": False,
        "live_application_pointer_changed": False,
    }
    _atomic_json_save(report, staging_report)
    completion_manifest = {
        "schema": COMPLETION_SCHEMA,
        "artifact_schema": STAGE4_ARTIFACT_SCHEMA,
        "report_schema": STAGE4_REPORT_SCHEMA,
        "artifact_path": str(checkpoint_path.resolve()),
        "artifact_sha256": artifact_sha256,
        "report_path": str(report_path.resolve()),
        "report_sha256": file_sha256(staging_report),
        "safe_to_evaluate": True,
        "test_splits_used": False,
        "live_application_pointer_changed": False,
        "complete": True,
    }
    _atomic_json_save(completion_manifest, staging_completion)
    # Publish checkpoint, report, and their final integrity proof in one
    # same-volume directory rename. A crash can leave an ignorable pending
    # directory, never a half-published audited output directory.
    os.replace(staging_dir, output_dir)
    print(f"Stage-4 router checkpoint: {checkpoint_path}")
    print(f"Stage-4 audit report: {report_path}")
    print(
        "Safe for later offline evaluation; enabled components: "
        + (", ".join(eligible_components) if eligible_components else "none")
    )


if __name__ == "__main__":
    main()
