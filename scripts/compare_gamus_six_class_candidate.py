"""Fail-closed comparison for the GAMUS direct-height and six-class runs.

The command reads validation artifacts only.  It never constructs the official
GAMUS test split, writes a checkpoint, or changes the application pointer.
Corrected legacy and unseen-geography reports are external evidence: omitting
either leaves the candidate blocked from manual promotion review while still
allowing an honest GAMUS development comparison to be written.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


# Independent CUDA validation passes accumulate hundreds of millions of pixels
# in a different kernel/reduction order.  Their observed baseline drift is at
# most 9.9e-6, far below the smallest predeclared guard (0.01).  Use a fixed,
# recorded numerical-reproducibility tolerance instead of requiring bitwise
# equality between two honest recomputations.
BASELINE_REPRODUCIBILITY_ABS_TOLERANCE = 1.0e-4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "gamus_six_class_comparison_v1.yaml"
REPORT_SCHEMA = "msr.gamus_six_class_candidate_comparison.v1"
EXPECTED_HEIGHT_EXPERIMENT = "multidomain_surface_gamus_direct_height_pilot"
EXPECTED_EXPANDED_EXPERIMENT = "multidomain_surface_gamus_six_class_candidate_v1"


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
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(_mapping(value, f"YAML document {path}"))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(_mapping(value, f"JSON document {path}"))


def _metric(metrics: Mapping[str, Any], path: str) -> float:
    value: object = metrics
    for component in path.split("."):
        value = _mapping(value, f"parent of metric {path!r}").get(component)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"required metric {path!r} is missing or non-numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"required metric {path!r} is non-finite")
    return result


def _normalise_pair_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove exactly the declared ablation differences from one run config."""

    result = deepcopy(dict(config))
    result.pop("protocol", None)
    experiment = dict(_mapping(result.get("experiment"), "experiment config"))
    experiment.pop("name", None)
    result["experiment"] = experiment
    model = dict(_mapping(result.get("model"), "model config"))
    model.pop("fine_semantic_classes", None)
    result["model"] = model
    training = dict(_mapping(result.get("training"), "training config"))
    training.pop("fine_semantic_weight", None)
    training.pop("fine_semantic_class_weights", None)
    groups = []
    for raw_group in training.get("parameter_groups", []):
        group = deepcopy(dict(_mapping(raw_group, "parameter group")))
        group["prefixes"] = [
            prefix
            for prefix in group.get("prefixes", [])
            if str(prefix) != "fine_semantic_head."
        ]
        groups.append(group)
    if groups:
        training["parameter_groups"] = groups
    result["training"] = training
    evaluation = dict(_mapping(result.get("evaluation"), "evaluation config"))
    # These declarations cannot change optimizer/model behavior.  Both remain
    # false for promotion, but the expanded candidate lists extra blockers.
    evaluation.pop("promotion_blockers", None)
    result["evaluation"] = evaluation
    return result


def validate_fair_source_configs(
    height_config: Mapping[str, Any], expanded_config: Mapping[str, Any]
) -> None:
    if height_config.get("experiment", {}).get("name") != EXPECTED_HEIGHT_EXPERIMENT:
        raise ValueError("unexpected height-focused experiment name")
    if (
        expanded_config.get("experiment", {}).get("name")
        != EXPECTED_EXPANDED_EXPERIMENT
    ):
        raise ValueError("unexpected expanded experiment name")
    if int(height_config.get("model", {}).get("fine_semantic_classes", 0)) != 0:
        raise ValueError("height-focused comparison must not enable six classes")
    if int(expanded_config.get("model", {}).get("fine_semantic_classes", 0)) != 6:
        raise ValueError("expanded candidate must enable exactly six classes")
    if float(height_config.get("training", {}).get("fine_semantic_weight", 0.0)) != 0:
        raise ValueError("height-focused comparison unexpectedly has a fine loss")
    if float(expanded_config.get("training", {}).get("fine_semantic_weight", 0.0)) <= 0:
        raise ValueError("expanded candidate requires a positive six-class loss")
    if _normalise_pair_config(height_config) != _normalise_pair_config(expanded_config):
        raise ValueError(
            "height and expanded configs differ outside the six-class head/loss ablation"
        )
    for role, config in (("height", height_config), ("expanded", expanded_config)):
        data = _mapping(config.get("data"), f"{role} data config")
        forbidden = [key for key in data if "test" in str(key).lower()]
        if forbidden:
            raise ValueError(
                f"{role} experiment data config exposes test fields: {forbidden}"
            )
        if str(data.get("dataset", "")).lower() != "gamus":
            raise ValueError(f"{role} comparison must use the native GAMUS loader")
        if str(data.get("train_radiometric_policy", "")) != "raw":
            raise ValueError("colour augmentation is not part of this frozen comparison")


def authenticate_protocol(protocol_path: Path) -> dict[str, Any]:
    config = _load_yaml(protocol_path)
    protocol = _mapping(config.get("protocol"), "comparison protocol")
    if protocol.get("schema") != "msr.gamus_six_class_comparison.v1":
        raise ValueError("unsupported comparison protocol schema")
    if protocol.get("official_test_constructed") is not False:
        raise ValueError("comparison protocol must forbid official-test construction")
    if protocol.get("promotion_permitted") is not False:
        raise ValueError("comparison protocol cannot permit promotion")

    authenticated: dict[str, Any] = {}
    artifact_roles = {
        "protected_checkpoint": "protected_checkpoint_sha256",
        "live_pointer": "live_pointer_file_sha256",
        "approved_index": "approved_index_sha256",
        "quality_report": "quality_report_sha256",
        "preparation_artifact_index": "preparation_artifact_index_sha256",
        "preparation_report": "preparation_report_sha256",
        "dav2_identity_manifest": "dav2_identity_manifest_sha256",
        "height_outlier_manifest": "height_outlier_manifest_sha256",
        "height_pilot_config": "height_pilot_config_sha256",
        "expanded_candidate_config": "expanded_candidate_config_sha256",
    }
    for role, hash_key in artifact_roles.items():
        path = _resolve(str(protocol[role]))
        if not path.is_file():
            raise FileNotFoundError(f"required protocol artifact is missing: {path}")
        actual = file_sha256(path)
        expected = str(protocol[hash_key]).lower()
        if actual != expected:
            raise ValueError(
                f"{role} SHA-256 mismatch: expected {expected}, found {actual}"
            )
        authenticated[role] = {"path": str(path), "sha256": actual}

    pointer_path = Path(authenticated["live_pointer"]["path"])
    pointer_target = _resolve(pointer_path.read_text(encoding="utf-8-sig").strip())
    protected_path = Path(authenticated["protected_checkpoint"]["path"])
    if pointer_target != protected_path:
        raise ValueError("live pointer no longer targets the protected checkpoint")

    quality = _load_json(Path(authenticated["quality_report"]["path"]))
    approved = _load_json(Path(authenticated["approved_index"]["path"]))
    preparation = _load_json(Path(authenticated["preparation_report"]["path"]))
    for split, expected_count in (
        ("train", int(protocol["approved_train_count"])),
        ("val", int(protocol["approved_validation_count"])),
    ):
        quality_count = int(quality["splits"][split]["approved_tiles"])
        approved_count = int(approved["splits"][split]["approved_count"])
        if quality_count != expected_count or approved_count != expected_count:
            raise ValueError(
                f"authenticated {split} count mismatch: "
                f"quality={quality_count}, index={approved_count}, expected={expected_count}"
            )
    if preparation.get("training_gate", {}).get("status") != "passed":
        raise ValueError("GAMUS preparation training gate is not passed")
    if preparation.get("validity_mask_contract", {}).get("status") != "passed":
        raise ValueError("GAMUS independent-validity-mask audit is not passed")
    if preparation.get("official_split_policy", {}).get("test") != (
        "inventory_audit_only; never constructed by the pilot trainer"
    ):
        raise ValueError("preparation report does not lock the official test split")

    height_source = _load_yaml(Path(authenticated["height_pilot_config"]["path"]))
    expanded_source = _load_yaml(
        Path(authenticated["expanded_candidate_config"]["path"])
    )
    validate_fair_source_configs(height_source, expanded_source)
    return {
        "config": config,
        "authenticated_artifacts": authenticated,
        "height_source_config": height_source,
        "expanded_source_config": expanded_source,
        "pointer_target": str(pointer_target),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        records.append(dict(_mapping(value, f"epoch record {line_number}")))
    if not records:
        raise ValueError(f"experiment has no completed epochs: {path}")
    epochs = [int(record["epoch"]) for record in records]
    if epochs != list(range(1, len(records) + 1)):
        raise ValueError(f"epoch history is non-contiguous: {epochs}")
    return records


def _gamus_metrics(value: Mapping[str, Any]) -> Mapping[str, Any]:
    by_suite = value.get("validation_metrics_by_suite")
    if isinstance(by_suite, Mapping):
        return _mapping(by_suite.get("gamus"), "GAMUS validation metrics")
    return _mapping(value.get("validation_metrics"), "GAMUS validation metrics")


def _record_eligible(record: Mapping[str, Any]) -> bool:
    metrics = _gamus_metrics(record)
    return bool(
        metrics.get("passes_validation_guards", False)
        and metrics.get("passes_initial_checkpoint_guard", False)
        and metrics.get("passes_protected_base_guard", False)
    )


def load_experiment(
    experiment_dir: Path,
    source_config: Mapping[str, Any],
    *,
    expected_name: str,
) -> dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"experiment directory is missing: {experiment_dir}")
    saved_config_path = experiment_dir / "config.yaml"
    metrics_path = experiment_dir / "metrics.jsonl"
    checkpoint_latest = experiment_dir / "checkpoint_latest.pt"
    for path in (saved_config_path, metrics_path, checkpoint_latest):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete experiment; missing {path.name}: {path}")
    saved_config = _load_yaml(saved_config_path)
    if expected_name == EXPECTED_HEIGHT_EXPERIMENT:
        if saved_config != dict(source_config):
            raise ValueError(
                "saved height-focused config differs from its authenticated source config"
            )
    elif _normalise_pair_config(saved_config) != _normalise_pair_config(source_config):
        raise ValueError(
            "saved expanded config differs from its authenticated source outside "
            "the sealed comparison metadata"
        )
    if saved_config.get("experiment", {}).get("name") != expected_name:
        raise ValueError(f"unexpected saved experiment name for {experiment_dir}")
    records = _load_jsonl(metrics_path)
    best_raw = min(records, key=lambda record: _metric(_gamus_metrics(record), "rmse_m"))
    eligible_records = [record for record in records if _record_eligible(record)]
    best_eligible = (
        min(
            eligible_records,
            key=lambda record: _metric(_gamus_metrics(record), "rmse_m"),
        )
        if eligible_records
        else None
    )
    selected = best_eligible or best_raw
    best_checkpoint = experiment_dir / "checkpoint_best_landscape.pt"
    if best_eligible is not None and not best_checkpoint.is_file():
        raise FileNotFoundError(
            "a gate-passing best epoch has no checkpoint_best_landscape.pt"
        )
    latest_record = records[-1]
    selected_checkpoint: Path | None
    if best_eligible is not None:
        selected_checkpoint = best_checkpoint
    elif int(best_raw["epoch"]) == int(latest_record["epoch"]):
        # checkpoint_latest is reproducibly paired only with the final epoch.
        selected_checkpoint = checkpoint_latest
    else:
        # The trainer did not save every raw, gate-failing epoch. Never attach
        # the final weights to metrics from an earlier raw-best epoch.
        selected_checkpoint = None
    return {
        "path": str(experiment_dir),
        "config_path": str(saved_config_path),
        "config_sha256": file_sha256(saved_config_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": file_sha256(metrics_path),
        "completed_epochs": len(records),
        "best_raw_epoch": int(best_raw["epoch"]),
        "best_gate_passing_epoch": (
            int(best_eligible["epoch"]) if best_eligible is not None else None
        ),
        "latest_epoch": int(latest_record["epoch"]),
        "latest_metrics": dict(_gamus_metrics(latest_record)),
        "latest_checkpoint": str(checkpoint_latest),
        "latest_checkpoint_sha256": file_sha256(checkpoint_latest),
        "selected_epoch": int(selected["epoch"]),
        "selected_is_gate_passing": best_eligible is not None,
        "selected_metrics": dict(_gamus_metrics(selected)),
        "selected_checkpoint": (
            str(selected_checkpoint) if selected_checkpoint is not None else None
        ),
        "selected_checkpoint_sha256": (
            file_sha256(selected_checkpoint) if selected_checkpoint is not None else None
        ),
        "selected_checkpoint_available": selected_checkpoint is not None,
        "saved_protocol": dict(saved_config.get("protocol", {})),
    }


def load_protected_baseline(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "initial_checkpoint_validation_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing protected validation baseline: {path}")
    document = _load_json(path)
    if document.get("source") != "current_validation":
        raise ValueError("protected baseline was not recomputed on current validation")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "metrics": dict(_gamus_metrics(document)),
    }


def seal_height_pilot_comparison(
    authenticated: Mapping[str, Any],
    height_experiment: Mapping[str, Any],
    height_baseline: Mapping[str, Any],
    *,
    lock_path: Path,
    launch_config_path: Path,
) -> dict[str, Any]:
    """Freeze the terminal height run into the exact config used for training."""

    lock = {
        "schema": "msr.gamus_height_pilot_comparison_lock.v1",
        "official_gamus_test_constructed": False,
        "height_experiment": {
            key: value
            for key, value in height_experiment.items()
            if key not in {"selected_metrics", "saved_protocol"}
        },
        "protected_baseline": {
            key: value for key, value in height_baseline.items() if key != "metrics"
        },
        "selected_height_metrics": height_experiment["selected_metrics"],
        "protected_baseline_metrics": height_baseline["metrics"],
    }
    _write_json_atomic(lock_path, lock)
    lock_sha256 = file_sha256(lock_path)
    launch_config = deepcopy(dict(authenticated["expanded_source_config"]))
    launch_protocol = dict(
        _mapping(launch_config.get("protocol"), "expanded launch protocol")
    )
    try:
        lock_reference = str(lock_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        lock_reference = str(lock_path.resolve())
    launch_protocol.update(
        {
            "height_pilot_comparison_artifact": lock_reference,
            "height_pilot_comparison_artifact_sha256": lock_sha256,
            "height_pilot_experiment": height_experiment["path"],
            "height_pilot_metrics_sha256": height_experiment["metrics_sha256"],
            "height_pilot_selected_checkpoint_sha256": height_experiment[
                "selected_checkpoint_sha256"
            ],
            "height_pilot_selected_epoch": height_experiment["selected_epoch"],
            "height_pilot_latest_checkpoint_sha256": height_experiment[
                "latest_checkpoint_sha256"
            ],
            "height_pilot_latest_epoch": height_experiment["latest_epoch"],
        }
    )
    launch_config["protocol"] = launch_protocol
    launch_config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = launch_config_path.with_suffix(launch_config_path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(launch_config, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(launch_config_path)
    return {
        "height_pilot_comparison_artifact": str(lock_path.resolve()),
        "height_pilot_comparison_artifact_sha256": lock_sha256,
        "launch_config": str(launch_config_path.resolve()),
        "launch_config_sha256": file_sha256(launch_config_path),
    }


def validate_embedded_height_pilot_lock(
    expanded_experiment: Mapping[str, Any],
    height_experiment: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = _mapping(
        expanded_experiment.get("saved_protocol"), "saved expanded protocol"
    )
    reference = protocol.get("height_pilot_comparison_artifact")
    expected_hash = protocol.get("height_pilot_comparison_artifact_sha256")
    if not isinstance(reference, str) or not isinstance(expected_hash, str):
        raise ValueError("expanded run did not embed a sealed height-pilot comparison")
    lock_path = _resolve(reference)
    if not lock_path.is_file():
        raise FileNotFoundError(f"height-pilot comparison lock is missing: {lock_path}")
    actual_hash = file_sha256(lock_path)
    if actual_hash != expected_hash.lower():
        raise ValueError("embedded height-pilot comparison lock hash mismatch")
    lock = _load_json(lock_path)
    if lock.get("schema") != "msr.gamus_height_pilot_comparison_lock.v1":
        raise ValueError("unsupported height-pilot comparison lock schema")
    locked_experiment = _mapping(lock.get("height_experiment"), "locked height run")
    expected_fields = {
        "path": height_experiment["path"],
        "metrics_sha256": height_experiment["metrics_sha256"],
        "selected_checkpoint_sha256": height_experiment[
            "selected_checkpoint_sha256"
        ],
        "selected_epoch": height_experiment["selected_epoch"],
        "latest_checkpoint_sha256": height_experiment[
            "latest_checkpoint_sha256"
        ],
        "latest_epoch": height_experiment["latest_epoch"],
    }
    for key, value in expected_fields.items():
        if locked_experiment.get(key) != value:
            raise ValueError(f"expanded run is pinned to a different height run: {key}")
    for key, value in (
        ("height_pilot_metrics_sha256", height_experiment["metrics_sha256"]),
        (
            "height_pilot_selected_checkpoint_sha256",
            height_experiment["selected_checkpoint_sha256"],
        ),
        ("height_pilot_selected_epoch", height_experiment["selected_epoch"]),
        (
            "height_pilot_latest_checkpoint_sha256",
            height_experiment["latest_checkpoint_sha256"],
        ),
        ("height_pilot_latest_epoch", height_experiment["latest_epoch"]),
    ):
        if protocol.get(key) != value:
            raise ValueError(f"saved expanded protocol has a mismatched {key}")
    return {"path": str(lock_path), "sha256": actual_hash, "status": "passed"}


def compare_gates(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for path, raw_rule in gate_config.items():
        rule = _mapping(raw_rule, f"gate {path}")
        direction = str(rule["direction"])
        limit = float(rule["limit"])
        current = _metric(candidate, str(path))
        baseline = _metric(reference, str(path))
        delta = current - baseline
        if direction == "min_improvement":
            passed = baseline - current >= limit
        elif direction == "max_regression":
            passed = delta <= limit
        elif direction == "max_drop":
            passed = baseline - current <= limit
        else:
            raise ValueError(f"unsupported gate direction {direction!r} for {path}")
        checks[str(path)] = {
            "candidate": current,
            "reference": baseline,
            "candidate_minus_reference": delta,
            "direction": direction,
            "limit": limit,
            "passes": bool(passed),
        }
    return {"passes": all(item["passes"] for item in checks.values()), "checks": checks}


def validate_six_class_metrics(
    metrics: Mapping[str, Any], six_class_config: Mapping[str, Any]
) -> dict[str, Any]:
    identification = _mapping(
        metrics.get("six_class_identification"), "six-class identification metrics"
    )
    names = [str(name) for name in six_class_config["class_names"]]
    if identification.get("class_names") != names:
        raise ValueError("six-class metric order does not match the frozen protocol")
    matrix = identification.get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != len(names)
        or any(not isinstance(row, list) or len(row) != len(names) for row in matrix)
    ):
        raise ValueError("six-class confusion matrix must be complete and 6 x 6")
    per_class = _mapping(identification.get("per_class"), "per-class metrics")
    required_metrics = [str(name) for name in six_class_config["required_per_class_metrics"]]
    class_checks: dict[str, Any] = {}
    for name in names:
        values = _mapping(per_class.get(name), f"metrics for class {name}")
        scores = {metric: _metric(values, metric) for metric in required_metrics}
        support = int(values.get("support_pixels", 0))
        if bool(six_class_config.get("require_positive_support", True)) and support <= 0:
            raise ValueError(f"six-class validation has no reference support for {name}")
        class_checks[name] = {**scores, "support_pixels": support}

    road = _mapping(metrics.get("road_boundary_quality"), "road-boundary metrics")
    expected_tolerance = int(six_class_config["road_boundary_tolerance_pixels"])
    if int(road.get("tolerance_pixels", -1)) != expected_tolerance:
        raise ValueError("road-boundary tolerance differs from the frozen protocol")
    road_scores = {
        name: _metric(road, name)
        for name in ("precision", "recall", "f1", "dilated_boundary_iou")
    }
    if int(road.get("reference_boundary_pixels", 0)) <= 0:
        raise ValueError("road-boundary validation has no reference boundary pixels")

    water = _mapping(metrics.get("water_shadow_proxy"), "water-dark diagnostic")
    if water.get("is_shadow_proxy_not_ground_truth") is not True:
        raise ValueError("water-dark diagnostic must be labelled as a proxy")
    water_scores = {
        name: _metric(water, name)
        for name in (
            "false_water_rate_on_dark_non_water",
            "dark_share_of_all_water_false_positives",
        )
    }
    if int(water.get("dark_non_water_pixels", 0)) <= 0:
        raise ValueError("water-dark diagnostic has no valid dark non-water pixels")
    return {
        "passes_completeness": True,
        "class_names": names,
        "per_class": class_checks,
        "macro": {
            metric: _metric(identification, f"macro_{metric}")
            for metric in required_metrics
        },
        "confusion_matrix": matrix,
        "road_boundary": {**road_scores, "tolerance_pixels": expected_tolerance},
        "water_dark_proxy": water_scores,
    }


def _normalise_legacy_metrics(raw: Mapping[str, Any]) -> dict[str, Any]:
    metrics = raw.get("metrics", raw)
    metrics = _mapping(metrics, "corrected legacy model metrics")
    if "expert_metrics" in metrics:
        height = _mapping(metrics["expert_metrics"], "expert metrics").get("height")
        landscapes = _mapping(
            _mapping(metrics.get("per_landscape_metrics"), "landscape metrics").get(
                "height"
            ),
            "height landscape metrics",
        )
        domains = _mapping(
            _mapping(metrics.get("per_domain_metrics"), "domain metrics").get(
                "height"
            ),
            "height domain metrics",
        )
        return {
            **dict(_mapping(height, "overall height metrics")),
            "landscapes": dict(landscapes),
            "domains": dict(domains),
        }
    return dict(metrics)


def validate_corrected_legacy_report(
    path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    report = _load_json(path)
    if report.get("schema") != config["report_schema"]:
        raise ValueError("unsupported corrected-legacy report schema")
    if report.get("identical_protocol_for_all_models") is not True:
        raise ValueError("corrected legacy report is not an identical-protocol comparison")
    if report.get("official_test_used") is not False:
        raise ValueError("corrected legacy report must not use an official test split")
    protocols = set(report.get("protocols", []))
    missing_protocols = set(config.get("required_protocols", [])) - protocols
    if missing_protocols:
        raise ValueError(
            f"corrected legacy report is missing protocols: {sorted(missing_protocols)}"
        )
    models = _mapping(report.get("models"), "corrected legacy models")
    required_names = [str(name) for name in config["required_model_names"]]
    normalised = {
        name: _normalise_legacy_metrics(
            _mapping(models.get(name), f"corrected legacy model {name}")
        )
        for name in required_names
    }
    for name, metrics in normalised.items():
        for metric_path in config["required_metric_paths"]:
            _metric(metrics, str(metric_path))

    comparisons: dict[str, Any] = {}
    all_pass = True
    for reference_name, gate_key in (
        ("protected", "candidate_vs_protected_max_regression"),
        ("height_pilot", "candidate_vs_height_pilot_max_regression"),
    ):
        checks = {}
        for metric_path, limit_value in config[gate_key].items():
            candidate = _metric(normalised["expanded_candidate"], str(metric_path))
            reference = _metric(normalised[reference_name], str(metric_path))
            delta = candidate - reference
            passed = delta <= float(limit_value)
            checks[str(metric_path)] = {
                "candidate": candidate,
                "reference": reference,
                "candidate_minus_reference": delta,
                "max_regression": float(limit_value),
                "passes": passed,
            }
            all_pass &= passed
        comparisons[f"expanded_vs_{reference_name}"] = {
            "passes": all(item["passes"] for item in checks.values()),
            "checks": checks,
        }
    return {
        "status": "passed" if all_pass else "failed",
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "comparisons": comparisons,
    }


def validate_unseen_geography_report(
    path: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    report = _load_json(path)
    if report.get("schema") != config["report_schema"]:
        raise ValueError("unsupported unseen-geography report schema")
    if report.get("geography_locked_before_evaluation") is not True:
        raise ValueError("unseen geography was not locked before evaluation")
    if report.get("official_gamus_test_used") is not False:
        raise ValueError("official GAMUS test cannot be used for this development decision")
    passed = report.get("passes_predeclared_guards") is True
    return {
        "status": "passed" if passed else "failed",
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
    }


def compare_recomputed_baselines(
    height_metrics: Mapping[str, Any],
    expanded_metrics: Mapping[str, Any],
    metric_paths: Iterable[str],
) -> dict[str, Any]:
    """Verify two independent baseline passes agree within numerical noise."""

    comparisons: dict[str, Any] = {}
    for metric_path in sorted(metric_paths):
        height_value = _metric(height_metrics, metric_path)
        expanded_value = _metric(expanded_metrics, metric_path)
        difference = abs(height_value - expanded_value)
        equal = math.isclose(
            height_value,
            expanded_value,
            rel_tol=0.0,
            abs_tol=BASELINE_REPRODUCIBILITY_ABS_TOLERANCE,
        )
        comparisons[metric_path] = {
            "height_run_baseline": height_value,
            "expanded_run_baseline": expanded_value,
            "absolute_difference": difference,
            "absolute_tolerance": BASELINE_REPRODUCIBILITY_ABS_TOLERANCE,
            "equal_within_tolerance": equal,
        }
        if not equal:
            raise ValueError(
                f"protected GAMUS baseline differs between experiments at {metric_path}"
            )
    return comparisons


def build_report(
    authenticated: Mapping[str, Any],
    height_experiment: Mapping[str, Any],
    expanded_experiment: Mapping[str, Any],
    height_baseline: Mapping[str, Any],
    expanded_baseline: Mapping[str, Any],
    *,
    corrected_legacy_path: Path | None,
    unseen_geography_path: Path | None,
) -> dict[str, Any]:
    protocol_config = _mapping(authenticated["config"], "protocol document")
    protected_gates = _mapping(
        protocol_config["protected_height_gates"], "protected height gates"
    )
    pilot_gates = _mapping(
        protocol_config["height_pilot_protection_gates"], "height-pilot gates"
    )
    all_height_paths = set(protected_gates) | set(pilot_gates)
    baseline_equality = compare_recomputed_baselines(
        height_baseline["metrics"],
        expanded_baseline["metrics"],
        all_height_paths,
    )
    protected_metrics = height_baseline["metrics"]
    height_metrics = height_experiment["selected_metrics"]
    expanded_metrics = expanded_experiment["selected_metrics"]
    height_pilot_lock = validate_embedded_height_pilot_lock(
        expanded_experiment, height_experiment
    )
    gamus_comparisons = {
        "height_pilot_vs_protected": compare_gates(
            height_metrics, protected_metrics, protected_gates
        ),
        "expanded_vs_protected": compare_gates(
            expanded_metrics, protected_metrics, protected_gates
        ),
        "expanded_vs_height_pilot": compare_gates(
            expanded_metrics, height_metrics, pilot_gates
        ),
    }
    six_class = validate_six_class_metrics(
        expanded_metrics,
        _mapping(protocol_config["six_class"], "six-class protocol"),
    )

    blockers: list[str] = []
    if not height_experiment["selected_is_gate_passing"]:
        blockers.append("height-focused pilot has no gate-passing checkpoint")
    if not expanded_experiment["selected_is_gate_passing"]:
        blockers.append("expanded candidate has no gate-passing checkpoint")
    for name, comparison in gamus_comparisons.items():
        if not comparison["passes"]:
            blockers.append(f"GAMUS height gates failed: {name}")

    corrected_config = _mapping(
        protocol_config["corrected_legacy"], "corrected legacy config"
    )
    if corrected_legacy_path is None:
        corrected_legacy: dict[str, Any] = {"status": "required_pending"}
        blockers.append("corrected legacy urban/forest comparison is pending")
    else:
        corrected_legacy = validate_corrected_legacy_report(
            corrected_legacy_path, corrected_config
        )
        if corrected_legacy["status"] != "passed":
            blockers.append("corrected legacy urban/forest height gates failed")

    unseen_config = _mapping(
        protocol_config["unseen_geography"], "unseen geography config"
    )
    if unseen_geography_path is None:
        unseen_geography: dict[str, Any] = {"status": "required_pending"}
        blockers.append("genuinely unseen geographic evaluation is pending")
    else:
        unseen_geography = validate_unseen_geography_report(
            unseen_geography_path, unseen_config
        )
        if unseen_geography["status"] != "passed":
            blockers.append("unseen-geography guards failed")

    return {
        "schema": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_role": "GAMUS train/validation development comparison; official test unused",
        "official_gamus_test_constructed": False,
        "authenticated_artifacts": authenticated["authenticated_artifacts"],
        "protected_baseline": {
            "checkpoint": authenticated["pointer_target"],
            "checkpoint_sha256": authenticated["authenticated_artifacts"]
            ["protected_checkpoint"]["sha256"],
            "height_experiment_baseline_artifact": {
                key: value for key, value in height_baseline.items() if key != "metrics"
            },
            "expanded_experiment_baseline_artifact": {
                key: value for key, value in expanded_baseline.items() if key != "metrics"
            },
            "identical_metric_check": baseline_equality,
            "metrics": protected_metrics,
        },
        "height_focused_pilot": dict(height_experiment),
        "expanded_six_class_candidate": dict(expanded_experiment),
        "height_pilot_comparison_lock": height_pilot_lock,
        "gamus_height_comparisons": gamus_comparisons,
        "expanded_classification": six_class,
        "corrected_legacy_external_evidence": corrected_legacy,
        "unseen_geography_external_evidence": unseen_geography,
        "manual_promotion_review_ready": not blockers,
        "promotion_blockers": blockers,
        "promotion_performed": False,
        "active_model_changed": False,
        "decision": (
            "eligible for explicit human review; never auto-promoted"
            if not blockers
            else "candidate remains isolated"
        ),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--height-experiment", type=Path, required=True)
    parser.add_argument("--expanded-experiment", type=Path)
    parser.add_argument("--corrected-legacy-report", type=Path)
    parser.add_argument("--unseen-geography-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--seal-height-pilot", action="store_true")
    parser.add_argument("--height-lock-output", type=Path)
    parser.add_argument("--launch-config-output", type=Path)
    args = parser.parse_args()

    authenticated = authenticate_protocol(args.protocol.resolve())
    height_experiment = load_experiment(
        args.height_experiment.resolve(),
        authenticated["height_source_config"],
        expected_name=EXPECTED_HEIGHT_EXPERIMENT,
    )
    if args.seal_height_pilot:
        height_baseline = load_protected_baseline(args.height_experiment.resolve())
        protocol_output = _resolve(
            authenticated["config"]["protocol"]["output_dir"]
        )
        result = seal_height_pilot_comparison(
            authenticated,
            height_experiment,
            height_baseline,
            lock_path=(
                args.height_lock_output.resolve()
                if args.height_lock_output
                else protocol_output / "height_pilot_comparison_lock.json"
            ),
            launch_config_path=(
                args.launch_config_output.resolve()
                if args.launch_config_output
                else protocol_output / "six_class_launch_config.yaml"
            ),
        )
        print(
            json.dumps(
                {
                    "status": "sealed",
                    "official_gamus_test_constructed": False,
                    **result,
                },
                indent=2,
            )
        )
        return
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "official_gamus_test_constructed": False,
                    "height_experiment": height_experiment["path"],
                    "height_completed_epochs": height_experiment["completed_epochs"],
                    "expanded_training_started": False,
                },
                indent=2,
            )
        )
        return
    if args.expanded_experiment is None:
        parser.error("--expanded-experiment is required unless --preflight-only is set")
    expanded_experiment = load_experiment(
        args.expanded_experiment.resolve(),
        authenticated["expanded_source_config"],
        expected_name=EXPECTED_EXPANDED_EXPERIMENT,
    )
    height_baseline = load_protected_baseline(args.height_experiment.resolve())
    expanded_baseline = load_protected_baseline(args.expanded_experiment.resolve())
    report = build_report(
        authenticated,
        height_experiment,
        expanded_experiment,
        height_baseline,
        expanded_baseline,
        corrected_legacy_path=(
            args.corrected_legacy_report.resolve()
            if args.corrected_legacy_report
            else None
        ),
        unseen_geography_path=(
            args.unseen_geography_report.resolve()
            if args.unseen_geography_report
            else None
        ),
    )
    output = (
        args.output.resolve()
        if args.output
        else _resolve(authenticated["config"]["protocol"]["output_dir"])
        / "report.json"
    )
    pointer_before = authenticated["authenticated_artifacts"]["live_pointer"][
        "sha256"
    ]
    _write_json_atomic(output, report)
    pointer_path = Path(
        authenticated["authenticated_artifacts"]["live_pointer"]["path"]
    )
    if file_sha256(pointer_path) != pointer_before:
        raise RuntimeError("live pointer changed while writing the comparison report")
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
