"""Freeze once, then evaluate once on the locked NYC GAMUS development holdout.

The first mode authenticates a selected DC+PHL-only checkpoint without opening
NYC data.  The second mode requires an explicit confirmation, creates an
irreversible consumption marker before loading NYC, reports every metric even
on a failed gate, and never touches the official GAMUS test split or live app.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader
import yaml


PROJECT_ROOT = Path(__file__).parents[1]
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
    audit_states as audit_spatial_refined_head_states,
)
from msr.data.gamus_dataset import (  # noqa: E402
    GAMUS_SIX_CLASS_NAMES,
    GamusSurfaceDataset,
)
from msr.data.gamus_geographic_candidate import (  # noqa: E402
    load_json_file_with_hash,
    load_json_or_yaml_with_hash,
    resolve_project_path,
    validate_geographic_candidate_config,
)
from msr.data.gamus_geographic_holdout import (  # noqa: E402
    file_sha256,
    load_sealed_holdout_contract,
    role_sample_ids,
)
from msr.inference.predict import load_predictor  # noqa: E402
from msr.models import DomainGatedSurfaceNet  # noqa: E402
from msr.training.losses import MultiDomainSurfaceLoss  # noqa: E402
from train_multidomain import validate  # noqa: E402


PROTOCOL_SCHEMA = "msr.gamus.locked_geographic_evaluation.v1"
FREEZE_SCHEMA = "msr.gamus.locked_geographic_freeze.v1"
CONSUMPTION_SCHEMA = "msr.gamus.locked_geographic_consumption.v1"
RESULT_SCHEMA = "msr.gamus.locked_geographic_result.v1"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs/gamus_locked_nyc_evaluation_v1.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/evaluation/gamus_locked_nyc_v1"


def _head_state_auditor(config: Mapping[str, Any]):
    """Select the exact isolated-head auditor declared by the frozen recipe."""

    model = _mapping(config.get("model"), "candidate model")
    head_type = str(model.get("fine_semantic_head_type", "linear"))
    if head_type == "linear":
        return audit_linear_head_states
    if head_type == "spatial_refined":
        return audit_spatial_refined_head_states
    raise ValueError(f"unsupported locked-evaluation head type: {head_type}")


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


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_once(path: Path, value: object, *, accept_identical: bool) -> None:
    content = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if accept_identical and path.read_bytes() == content:
            return
        raise FileExistsError(f"locked artifact already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def load_protocol(path: str | Path) -> tuple[Path, dict[str, Any]]:
    protocol_path = Path(path).expanduser().resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported locked NYC evaluation protocol")
    holdout = _mapping(protocol.get("holdout"), "holdout protocol")
    safety = _mapping(protocol.get("safety"), "safety protocol")
    if holdout.get("city") != "NYC" or holdout.get("source_split") != "train":
        raise ValueError("locked evaluation must target NYC official-train tiles")
    if holdout.get("use") != "exactly_one_post_freeze_evaluation":
        raise ValueError("locked NYC holdout use policy is invalid")
    if holdout.get("tuning_after_opening") != "forbidden":
        raise ValueError("locked NYC tuning must be forbidden")
    if safety.get("official_test_policy") != "never_constructed_or_discovered":
        raise ValueError("official GAMUS test discovery must remain forbidden")
    if safety.get("create_consumption_lock_before_loading_holdout") is not True:
        raise ValueError("evaluation must lock consumption before loading NYC")
    if safety.get("refuse_second_evaluation") is not True:
        raise ValueError("evaluation protocol must refuse a second NYC pass")
    if safety.get("auto_promotion") is not False:
        raise ValueError("locked evaluation cannot promote a model")
    if safety.get("application_model_change") != "forbidden":
        raise ValueError("locked evaluation must forbid app-model changes")
    return protocol_path, protocol


def _load_candidate_config(protocol: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    candidate = _mapping(protocol.get("candidate"), "candidate protocol")
    config_path = resolve_project_path(PROJECT_ROOT, candidate.get("config"))
    config = load_json_or_yaml_with_hash(
        config_path,
        expected_sha256=str(candidate.get("config_sha256", "")),
        name="geographic candidate config",
    )
    validate_geographic_candidate_config(config, project_root=PROJECT_ROOT)
    return config_path, config


def _validate_protocol_bindings(
    protocol: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_protocol = _mapping(candidate_config.get("protocol"), "candidate protocol")
    protocol_candidate = _mapping(protocol.get("candidate"), "evaluation candidate")
    protected = _mapping(protocol.get("protected_state"), "protected state")
    holdout = _mapping(protocol.get("holdout"), "holdout")
    if protocol_candidate.get("experiment_name") != _mapping(
        candidate_config.get("experiment"), "candidate experiment"
    ).get("name"):
        raise ValueError("evaluation protocol names a different experiment")
    if protected.get("checkpoint_sha256") != candidate_protocol.get(
        "protected_checkpoint_sha256"
    ):
        raise ValueError("evaluation and candidate protected-checkpoint hashes differ")
    if protected.get("live_pointer_sha256") != candidate_protocol.get(
        "live_pointer_file_sha256"
    ):
        raise ValueError("evaluation and candidate live-pointer hashes differ")
    if holdout.get("contract_sha256") != candidate_protocol.get(
        "holdout_contract_sha256"
    ):
        raise ValueError("evaluation and candidate holdout-contract hashes differ")
    if holdout.get("approved_index_sha256") != candidate_protocol.get(
        "locked_holdout_approved_index_sha256"
    ):
        raise ValueError("evaluation and candidate NYC-index hashes differ")

    contract_path = resolve_project_path(PROJECT_ROOT, holdout.get("contract"))
    contract = load_sealed_holdout_contract(
        contract_path,
        expected_sha256=str(holdout.get("contract_sha256", "")),
    )
    holdout_index_path = resolve_project_path(PROJECT_ROOT, holdout.get("approved_index"))
    holdout_index = load_json_file_with_hash(
        holdout_index_path,
        expected_sha256=str(holdout.get("approved_index_sha256", "")),
        name="locked NYC approved index",
    )
    split_payload = _mapping(
        _mapping(holdout_index.get("splits"), "NYC index splits").get("train"),
        "NYC index train split",
    )
    ids = split_payload.get("approved_sample_ids")
    if not isinstance(ids, list) or tuple(sorted(ids)) != role_sample_ids(
        contract, "geographic_holdout"
    ):
        raise ValueError("locked NYC index differs from the sealed contract")
    if int(holdout.get("expected_tiles", -1)) != len(ids):
        raise ValueError("locked NYC tile count differs from protocol")
    return {
        "contract_path": contract_path,
        "contract": contract,
        "holdout_index_path": holdout_index_path,
        "holdout_ids": tuple(sorted(ids)),
    }


def build_freeze_manifest(
    protocol_path: str | Path,
    candidate_checkpoint: str | Path,
) -> dict[str, Any]:
    """Authenticate the selected model without constructing a holdout dataset."""

    protocol_file, protocol = load_protocol(protocol_path)
    config_path, config = _load_candidate_config(protocol)
    bindings = _validate_protocol_bindings(protocol, config)
    candidate_spec = _mapping(protocol.get("candidate"), "candidate protocol")
    checkpoint = Path(candidate_checkpoint).expanduser().resolve()
    if checkpoint.name != candidate_spec.get("required_checkpoint_name"):
        raise ValueError(
            "only the predeclared best validation checkpoint may be frozen for NYC"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"candidate checkpoint does not exist: {checkpoint}")
    protected_spec = _mapping(protocol.get("protected_state"), "protected state")
    protected_path = resolve_project_path(PROJECT_ROOT, protected_spec.get("checkpoint"))
    expected_protected_sha = str(protected_spec.get("checkpoint_sha256", ""))
    if file_sha256(protected_path) != expected_protected_sha:
        raise ValueError("protected checkpoint SHA-256 mismatch")
    base_state, _ = load_model_state(protected_path)
    candidate_state, candidate_payload = load_model_state(checkpoint)
    state_audit = _head_state_auditor(config)(base_state, candidate_state)
    embedded_config = _mapping(candidate_payload.get("config"), "checkpoint config")
    if embedded_config != config:
        raise ValueError("checkpoint embedded config differs from the frozen config")
    metrics = _mapping(candidate_payload.get("metrics"), "checkpoint validation metrics")
    six_class = _mapping(
        metrics.get("six_class_identification"),
        "checkpoint six-class validation metrics",
    )
    if set(six_class.get("class_names", [])) != set(GAMUS_SIX_CLASS_NAMES):
        raise ValueError("selected checkpoint lacks complete six-class validation metrics")
    checkpoint_epoch = candidate_payload.get("epoch")
    if isinstance(checkpoint_epoch, bool) or not isinstance(checkpoint_epoch, int):
        raise ValueError("selected checkpoint epoch is missing or malformed")

    # Crucially, only IDs and hashes are inspected above. No GamusSurfaceDataset
    # is constructed until the explicitly confirmed consumption mode.
    return {
        "schema": FREEZE_SCHEMA,
        "protocol": {
            "path": str(protocol_file),
            "sha256": file_sha256(protocol_file),
            "protocol_id": protocol.get("protocol_id"),
        },
        "candidate": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "epoch": checkpoint_epoch,
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "development_validation_metrics": deepcopy(dict(metrics)),
        },
        "state_audit": state_audit,
        "protected_checkpoint": {
            "path": str(protected_path),
            "sha256": file_sha256(protected_path),
        },
        "locked_holdout": {
            "city": "NYC",
            "tiles": len(bindings["holdout_ids"]),
            "contract": str(bindings["contract_path"]),
            "contract_sha256": file_sha256(bindings["contract_path"]),
            "approved_index": str(bindings["holdout_index_path"]),
            "approved_index_sha256": file_sha256(bindings["holdout_index_path"]),
            "dataset_constructed": False,
            "imagery_or_labels_opened": False,
        },
        "official_test_constructed_or_discovered": False,
        "thresholds_frozen": deepcopy(
            dict(_mapping(protocol.get("reporting"), "reporting protocol"))
        ),
        "promotion_performed": False,
    }


def evaluate_readiness(
    metrics: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only the thresholds declared before the locked city was opened."""

    reporting = _mapping(protocol.get("reporting"), "reporting protocol")
    required_classes = tuple(reporting.get("required_classes", []))
    if required_classes != tuple(GAMUS_SIX_CLASS_NAMES):
        raise ValueError("reporting class order differs from the model contract")
    required_metrics = tuple(reporting.get("required_per_class_metrics", []))
    identification = _mapping(
        metrics.get("six_class_identification"), "six-class metrics"
    )
    if tuple(identification.get("class_names", [])) != required_classes:
        raise ValueError("locked result class order is incomplete or changed")
    per_class = _mapping(identification.get("per_class"), "per-class metrics")
    if set(per_class) != set(required_classes):
        raise ValueError("locked result does not report all six classes")
    for class_name in required_classes:
        values = _mapping(per_class[class_name], f"{class_name} metrics")
        for metric_name in required_metrics:
            value = _number(values.get(metric_name), f"{class_name} {metric_name}")
            if metric_name == "support_pixels" and value <= 0:
                raise ValueError(f"locked NYC has no reference support for {class_name}")
            if metric_name != "support_pixels" and not 0.0 <= value <= 1.0:
                raise ValueError(f"{class_name} {metric_name} is outside [0, 1]")
    road_boundary = _mapping(
        metrics.get("road_boundary_quality"), "road boundary metrics"
    )
    _mapping(metrics.get("water_shadow_proxy"), "water-shadow proxy metrics")
    for metric_name in ("rmse_m", "mae_m", "bias_m", "correlation", "r2"):
        _number(metrics.get(metric_name), f"height {metric_name}")

    gates = _mapping(reporting.get("readiness_gates"), "readiness gates")
    checks: dict[str, dict[str, Any]] = {}

    def minimum(name: str, value: float, threshold: float) -> None:
        checks[name] = {
            "value": value,
            "minimum": threshold,
            "passes": value >= threshold,
        }

    minimum(
        "macro_f1",
        _number(identification.get("macro_f1"), "macro F1"),
        _number(gates.get("macro_f1_min"), "macro F1 gate"),
    )
    minimum(
        "macro_iou",
        _number(identification.get("macro_iou"), "macro IoU"),
        _number(gates.get("macro_iou_min"), "macro IoU gate"),
    )
    per_class_gates = _mapping(gates.get("per_class_f1_min"), "per-class gates")
    if set(per_class_gates) != set(required_classes):
        raise ValueError("predeclared per-class gates do not cover all six classes")
    for class_name in required_classes:
        minimum(
            f"{class_name}_f1",
            _number(
                _mapping(per_class[class_name], f"{class_name} metrics").get("f1"),
                f"{class_name} F1",
            ),
            _number(per_class_gates.get(class_name), f"{class_name} F1 gate"),
        )
    minimum(
        "road_boundary_f1",
        _number(road_boundary.get("f1"), "road-boundary F1"),
        _number(gates.get("road_boundary_f1_min"), "road-boundary F1 gate"),
    )
    return {
        "passes_all_predeclared_readiness_gates": all(
            bool(check["passes"]) for check in checks.values()
        ),
        "checks": checks,
    }


def _make_criterion(config: Mapping[str, Any], device: torch.device) -> MultiDomainSurfaceLoss:
    training = _mapping(config.get("training"), "candidate training")
    return MultiDomainSurfaceLoss(
        huber_delta_m=float(training.get("huber_delta_m", 2.0)),
        height_weight=float(training.get("height_weight", 1.0)),
        semantic_weight=float(training.get("semantic_weight", 0.3)),
        fine_semantic_weight=float(training.get("fine_semantic_weight", 0.0)),
        building_weight=float(training.get("building_weight", 1.0)),
        building_mse_weight=float(training.get("building_mse_weight", 0.0)),
        tall_building_weight=float(training.get("tall_building_weight", 0.0)),
        canopy_weight=float(training.get("canopy_weight", 1.0)),
        canopy_mse_weight=float(training.get("canopy_mse_weight", 0.0)),
        tall_canopy_weight=float(training.get("tall_canopy_weight", 0.0)),
        fused_building_weight=float(training.get("fused_building_weight", 0.0)),
        fused_building_mse_weight=float(
            training.get("fused_building_mse_weight", 0.0)
        ),
        fused_vegetation_weight=float(training.get("fused_vegetation_weight", 0.0)),
        fused_tall_building_weight=float(
            training.get("fused_tall_building_weight", 0.0)
        ),
        fused_tall_vegetation_weight=float(
            training.get("fused_tall_vegetation_weight", 0.0)
        ),
        building_distillation_weight=float(
            training.get("building_distillation_weight", 0.0)
        ),
        ground_suppression_weight=float(training.get("ground_suppression_weight", 0.25)),
        refinement_weight=float(training.get("refinement_weight", 0.0)),
        fine_semantic_class_weights=training.get("fine_semantic_class_weights"),
        fine_semantic_focal_gamma=float(
            training.get("fine_semantic_focal_gamma", 0.0)
        ),
        uncertainty_weight=float(training.get("uncertainty_weight", 0.0)),
    ).to(device)


def consume_and_evaluate(
    *,
    protocol_path: Path,
    candidate_checkpoint: Path,
    output_root: Path,
    confirmation: str,
) -> dict[str, Any]:
    protocol_file, protocol = load_protocol(protocol_path)
    safety = _mapping(protocol.get("safety"), "safety protocol")
    if confirmation != safety.get("confirmation_phrase"):
        raise ValueError("explicit locked-holdout confirmation phrase is missing")
    freeze_path = output_root / "freeze_manifest.json"
    consumption_path = output_root / "holdout_consumed.json"
    result_path = output_root / "locked_nyc_result.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("freeze_manifest.json must be created first")
    if consumption_path.exists() or result_path.exists():
        raise FileExistsError(
            "NYC holdout is already consumed; a second pass is forbidden"
        )
    expected_freeze = build_freeze_manifest(protocol_file, candidate_checkpoint)
    actual_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if actual_freeze != expected_freeze:
        raise ValueError("freeze manifest differs from current authenticated inputs")

    pointer_spec = _mapping(protocol.get("protected_state"), "protected state")
    pointer_path = resolve_project_path(PROJECT_ROOT, pointer_spec.get("live_pointer"))
    protected_path = resolve_project_path(PROJECT_ROOT, pointer_spec.get("checkpoint"))
    pointer_before = file_sha256(pointer_path)
    protected_before = file_sha256(protected_path)

    # This marker is intentionally written before the dataset is constructed.
    # If evaluation crashes, NYC remains consumed and an independent protocol
    # review is required; silently retrying would enable holdout fishing.
    consumption = {
        "schema": CONSUMPTION_SCHEMA,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": file_sha256(protocol_file),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "candidate_checkpoint_sha256": file_sha256(candidate_checkpoint),
        "city": "NYC",
        "expected_tiles": int(_mapping(protocol["holdout"], "holdout")["expected_tiles"]),
        "state": "consumed_before_dataset_construction",
        "automatic_retry_permitted": False,
        "tuning_or_reselection_after_this_point": "forbidden",
        "official_test_constructed_or_discovered": False,
    }
    _write_once(consumption_path, consumption, accept_identical=False)

    config_path, config = _load_candidate_config(protocol)
    bindings = _validate_protocol_bindings(protocol, config)
    data = _mapping(protocol.get("data"), "holdout data")
    dataset = GamusSurfaceDataset(
        resolve_project_path(PROJECT_ROOT, data.get("root")),
        "train",
        patch_size=int(data.get("patch_size", 1024)),
        random_crop=False,
        augment=False,
        rgb_scale=float(data.get("rgb_scale", 255.0)),
        height_max_m=float(data.get("height_max_m", 200.0)),
        radiometric_policy=str(data.get("radiometric_policy", "raw")),
        relative_prior_root=resolve_project_path(
            PROJECT_ROOT, data.get("relative_prior_root")
        ),
        require_relative_prior=bool(data.get("require_relative_priors", True)),
        approved_index_path=bindings["holdout_index_path"],
    )
    dataset_ids = tuple(sorted(record.sample_id for record in dataset.records))
    if dataset_ids != bindings["holdout_ids"]:
        raise ValueError("constructed NYC dataset differs from the locked contract")
    if len(dataset) != int(_mapping(protocol["holdout"], "holdout")["expected_tiles"]):
        raise ValueError("constructed NYC dataset has an unexpected size")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(data.get("num_workers", 0)) > 0,
        drop_last=False,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full locked NYC evaluation")
    device = torch.device("cuda")
    previous_directory = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        model, _ = load_predictor(candidate_checkpoint, device=device)
    finally:
        os.chdir(previous_directory)
    if not isinstance(model, DomainGatedSurfaceNet):
        raise ValueError("locked candidate is not a DomainGatedSurfaceNet")
    criterion = _make_criterion(config, device)
    precision = str(_mapping(config["training"], "candidate training").get("precision", "bf16"))
    losses, metrics = validate(
        model,
        loader,
        criterion,
        device,
        precision=precision,
    )
    readiness = evaluate_readiness(metrics, protocol)
    pointer_after = file_sha256(pointer_path)
    protected_after = file_sha256(protected_path)
    if pointer_after != pointer_before or protected_after != protected_before:
        raise RuntimeError("live pointer or protected checkpoint changed during evaluation")
    result = {
        "schema": RESULT_SCHEMA,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": file_sha256(protocol_file),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "consumption_marker_sha256": file_sha256(consumption_path),
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": file_sha256(candidate_checkpoint),
        "candidate_config": str(config_path),
        "city": "NYC",
        "tiles": len(dataset),
        "losses": losses,
        "metrics": metrics,
        "readiness": readiness,
        "interpretation": (
            "one locked unseen-city development result; it is not the official "
            "GAMUS test benchmark and does not prove cross-country or cross-sensor transfer"
        ),
        "official_test_constructed_or_discovered": False,
        "model_or_threshold_tuning_after_holdout": False,
        "live_pointer_unchanged": True,
        "protected_checkpoint_unchanged": True,
        "promotion_performed": False,
    }
    _write_once(result_path, result, accept_identical=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--evaluate-locked", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--confirmation", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = args.protocol.expanduser().resolve()
    checkpoint = args.candidate_checkpoint.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.freeze_only:
        manifest = build_freeze_manifest(protocol, checkpoint)
        _write_once(
            output_root / "freeze_manifest.json",
            manifest,
            accept_identical=True,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    result = consume_and_evaluate(
        protocol_path=protocol,
        candidate_checkpoint=checkpoint,
        output_root=output_root,
        confirmation=args.confirmation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
