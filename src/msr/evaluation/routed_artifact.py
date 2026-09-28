"""Fail-closed loading of an offline, provenance-bound Stage-3 router."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch

from msr.models.routed_surface import (
    ConservativeSceneRouter,
    FrozenDualSurfaceHeadRouter,
    RoutedDomainGatedSurfaceNet,
)
from msr.models.stage3_endpoints import (
    Stage3CheckpointValidationError,
    Stage3EndpointDiagnostics,
    load_stage3_endpoint_packs,
)
from msr.training.scene_router_record_store import (
    RECORD_STORE_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    validate_record_selection_provenance,
)


ROUTER_ARTIFACT_SCHEMA = "msr.stage3_scene_router.v1"
ROUTER_REPORT_SCHEMA = "msr.stage3_scene_router_report.v1"
STAGE3B_ROUTER_ARTIFACT_SCHEMA = "msr.stage3b_scene_domain_router.v1"
STAGE3B_ROUTER_REPORT_SCHEMA = (
    "msr.stage3b_scene_domain_router_report.v1"
)
STAGE3B_TARGET_PROVENANCE = (
    "candidate target is one iff authenticated record.source is GAMUS; "
    "source creates supervision and balanced sampling only, while the "
    "router network receives the 128-D scene descriptor alone; held-out "
    "endpoint utilities are used only for the no-regression threshold guard"
)
_STAGE3B_SELECTION_PRIORITY = (
    "maximum_gamus_coverage",
    "maximum_routed_utility_gain_vs_protected",
    "highest_threshold",
)


class RoutedArtifactValidationError(ValueError):
    """Raised when no safely authenticated routed model can be returned."""


@dataclass(frozen=True)
class RoutedArtifactDiagnostics:
    router_artifact_path: str
    router_artifact_sha256: str
    router_report_path: str
    router_report_sha256: str
    feature_channels: int
    hidden_features: int
    decision_threshold: float
    endpoint_diagnostics: Stage3EndpointDiagnostics
    record_config_sha256: str
    record_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GuardedRoutedSurfaceBundle:
    model: RoutedDomainGatedSurfaceNet
    diagnostics: RoutedArtifactDiagnostics
    router_report: Mapping[str, Any]


def _resolved_file(path_like: str | Path, *, role: str) -> Path:
    try:
        path = Path(path_like).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RoutedArtifactValidationError(
            f"{role} is unavailable: {path_like}"
        ) from error
    if not path.is_file():
        raise RoutedArtifactValidationError(f"{role} is not a regular file: {path}")
    return path


def _stable_json(path: Path, *, role: str) -> tuple[Mapping[str, Any], str]:
    before = path.stat()
    digest_before = file_sha256(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RoutedArtifactValidationError(
            f"Could not read {role} as JSON: {path}: {error}"
        ) from error
    after = path.stat()
    digest_after = file_sha256(path)
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or digest_before != digest_after
    ):
        raise RoutedArtifactValidationError(f"{role} changed while it was read")
    if not isinstance(value, Mapping):
        raise RoutedArtifactValidationError(f"{role} must contain a JSON object")
    return value, digest_before


def _stable_router_artifact(path: Path) -> tuple[Mapping[str, Any], str]:
    before = path.stat()
    digest_before = file_sha256(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RoutedArtifactValidationError(
            f"Could not safely load router artifact {path}: {error}"
        ) from error
    after = path.stat()
    digest_after = file_sha256(path)
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or digest_before != digest_after
    ):
        raise RoutedArtifactValidationError("Router artifact changed while it was read")
    if not isinstance(payload, Mapping):
        raise RoutedArtifactValidationError("Router artifact payload must be a mapping")
    return payload, digest_before


def _stable_file_digest(path: Path, *, role: str) -> str:
    before = path.stat()
    digest_before = file_sha256(path)
    after = path.stat()
    digest_after = file_sha256(path)
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or digest_before != digest_after
    ):
        raise RoutedArtifactValidationError(f"{role} changed while it was read")
    return digest_before


def _hex_digest(value: object, *, role: str) -> str:
    if not isinstance(value, str):
        raise RoutedArtifactValidationError(f"{role} must be a SHA-256 string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RoutedArtifactValidationError(
            f"{role} must be a 64-character hexadecimal SHA-256"
        )
    return normalized


def _positive_int(value: object, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RoutedArtifactValidationError(f"{role} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RoutedArtifactValidationError(
            f"{role} must be a non-negative integer"
        )
    return int(value)


def _finite_float(value: object, *, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RoutedArtifactValidationError(f"{role} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise RoutedArtifactValidationError(f"{role} must be finite")
    return converted


def _require_mapping(value: object, *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutedArtifactValidationError(f"{role} must be a mapping")
    return value


def _canonical_equal(left: object, right: object) -> bool:
    return canonical_json_sha256(left) == canonical_json_sha256(right)


def _validate_embedded_record_identity(
    payload: Mapping[str, Any], *, role: str
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    provenance = payload.get("record_provenance")
    completion = payload.get("record_completion")
    if not isinstance(provenance, Mapping) or not isinstance(completion, Mapping):
        raise RoutedArtifactValidationError(
            f"{role} must embed record_provenance and record_completion mappings"
        )
    if provenance.get("schema") != RECORD_STORE_SCHEMA:
        raise RoutedArtifactValidationError(f"{role} record provenance schema mismatch")
    if completion.get("schema") != RECORD_STORE_SCHEMA:
        raise RoutedArtifactValidationError(f"{role} record completion schema mismatch")
    if provenance.get("test_splits_excluded") is not True:
        raise RoutedArtifactValidationError(
            f"{role} record provenance did not exclude test splits"
        )
    if completion.get("complete") is not True:
        raise RoutedArtifactValidationError(f"{role} record store is incomplete")

    config_hash = _hex_digest(
        provenance.get("config_sha256"), role=f"{role} record config hash"
    )
    if completion.get("config_sha256") != config_hash:
        raise RoutedArtifactValidationError(
            f"{role} provenance/completion config hash mismatch"
        )
    required_identity = {
        "generation_config",
        "endpoints",
        "shared_model",
        "data",
    }
    missing = required_identity - set(provenance)
    if missing:
        raise RoutedArtifactValidationError(
            f"{role} record provenance is missing identity fields: {sorted(missing)}"
        )
    generation = provenance.get("generation_config")
    if not isinstance(generation, Mapping):
        raise RoutedArtifactValidationError(
            f"{role} record generation config must be a mapping"
        )
    if (
        generation.get("descriptor_mask")
        != "full_center_crop_all_pixels_no_reference_mask"
    ):
        raise RoutedArtifactValidationError(
            f"{role} router descriptors are not deployable all-pixel descriptors"
        )
    if generation.get("legacy_relative_prior_policy") != "stored_01":
        raise RoutedArtifactValidationError(
            f"{role} legacy router priors are not deployable stored 0..1 inputs"
        )
    recomputed_config_hash = canonical_json_sha256(
        {name: provenance[name] for name in sorted(required_identity)}
    )
    if recomputed_config_hash != config_hash:
        raise RoutedArtifactValidationError(f"{role} record config hash mismatch")

    try:
        validate_record_selection_provenance(
            provenance.get("data"),
            completed_record_count=completion.get("record_count"),
        )
    except ValueError as error:
        raise RoutedArtifactValidationError(f"{role} {error}") from error

    provenance_hash = canonical_json_sha256(provenance)
    completion_hash = canonical_json_sha256(completion)
    if payload.get("record_provenance_canonical_sha256") != provenance_hash:
        raise RoutedArtifactValidationError(
            f"{role} embedded provenance canonical hash mismatch"
        )
    if payload.get("record_completion_canonical_sha256") != completion_hash:
        raise RoutedArtifactValidationError(
            f"{role} embedded completion canonical hash mismatch"
        )
    _hex_digest(completion.get("records_sha256"), role=f"{role} records hash")
    _positive_int(completion.get("record_count"), role=f"{role} record count")
    _positive_int(completion.get("part_count"), role=f"{role} part count")
    descriptor_size = _positive_int(
        completion.get("descriptor_size"), role=f"{role} descriptor size"
    )
    if descriptor_size % 2:
        raise RoutedArtifactValidationError(
            f"{role} descriptor size must be twice the endpoint feature count"
        )
    _hex_digest(
        completion.get("parts_manifest_canonical_sha256"),
        role=f"{role} parts-manifest hash",
    )
    records_path = completion.get("records_path")
    if not isinstance(records_path, str) or not records_path.strip():
        raise RoutedArtifactValidationError(
            f"{role} records path must be a non-empty string"
        )
    return provenance, completion


def _endpoint_field(
    provenance: Mapping[str, Any], role: str
) -> Mapping[str, Any]:
    endpoints = provenance.get("endpoints")
    if not isinstance(endpoints, Mapping):
        raise RoutedArtifactValidationError("Record provenance endpoints must be a mapping")
    endpoint = endpoints.get(role)
    if not isinstance(endpoint, Mapping):
        raise RoutedArtifactValidationError(
            f"Record provenance has no {role!r} endpoint mapping"
        )
    return endpoint


def _verify_endpoint_diagnostics(
    provenance: Mapping[str, Any], diagnostics: Stage3EndpointDiagnostics
) -> None:
    expected = diagnostics.as_dict()
    endpoint_mapping = provenance.get("endpoints")
    assert isinstance(endpoint_mapping, Mapping)
    expected_endpoints = {
        name: expected[name]
        for name in ("protected", "gamus_stage1", "compatibility")
    }
    if not _canonical_equal(endpoint_mapping, expected_endpoints):
        raise RoutedArtifactValidationError(
            "Loaded endpoint diagnostics do not match router-record provenance"
        )
    shared = provenance.get("shared_model")
    if not isinstance(shared, Mapping):
        raise RoutedArtifactValidationError(
            "Record provenance shared_model must be a mapping"
        )
    if shared.get("shared_state_sha256") != expected["compatibility"][
        "shared_state_sha256"
    ]:
        raise RoutedArtifactValidationError(
            "Shared model-state hash does not match endpoint compatibility proof"
        )
    if shared.get("base_checkpoint") != expected.get("base_checkpoint"):
        raise RoutedArtifactValidationError(
            "Base-checkpoint diagnostics do not match router-record provenance"
        )


def _same_float(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)


def _validate_stage3b_record_files(
    completion: Mapping[str, Any],
    provenance: Mapping[str, Any],
    training_launch: Mapping[str, Any],
) -> None:
    """Re-authenticate the immutable record evidence used by Stage-3b."""

    records_path_value = completion.get("records_path")
    launch_records_value = training_launch.get("records")
    if not isinstance(records_path_value, str) or not isinstance(
        launch_records_value, str
    ):
        raise RoutedArtifactValidationError("Stage-3b record paths must be strings")
    records_path = _resolved_file(records_path_value, role="Stage-3b records")
    if Path(launch_records_value).expanduser().resolve() != records_path:
        raise RoutedArtifactValidationError(
            "Stage-3b training launch points to different records"
        )
    expected_records_hash = _hex_digest(
        completion.get("records_sha256"), role="Stage-3b records hash"
    )
    if (
        _stable_file_digest(records_path, role="Stage-3b records")
        != expected_records_hash
    ):
        raise RoutedArtifactValidationError(
            "Stage-3b records no longer match record completion"
        )

    actual_provenance, _ = _stable_json(
        _resolved_file(
            records_path.parent / "provenance.json",
            role="Stage-3b record provenance",
        ),
        role="Stage-3b record provenance",
    )
    actual_completion, _ = _stable_json(
        _resolved_file(
            records_path.parent / "completion.json",
            role="Stage-3b record completion",
        ),
        role="Stage-3b record completion",
    )
    if not _canonical_equal(actual_provenance, provenance):
        raise RoutedArtifactValidationError(
            "Stage-3b embedded provenance differs from its record sidecar"
        )
    if not _canonical_equal(actual_completion, completion):
        raise RoutedArtifactValidationError(
            "Stage-3b embedded completion differs from its record sidecar"
        )

    parts_manifest, _ = _stable_json(
        _resolved_file(
            records_path.parent / "parts_manifest.json",
            role="Stage-3b parts manifest",
        ),
        role="Stage-3b parts manifest",
    )
    expected_manifest_hash = _hex_digest(
        completion.get("parts_manifest_canonical_sha256"),
        role="Stage-3b parts-manifest hash",
    )
    if canonical_json_sha256(parts_manifest) != expected_manifest_hash:
        raise RoutedArtifactValidationError(
            "Stage-3b parts manifest no longer matches record completion"
        )
    if (
        parts_manifest.get("schema") != RECORD_STORE_SCHEMA
        or parts_manifest.get("config_sha256") != provenance.get("config_sha256")
    ):
        raise RoutedArtifactValidationError(
            "Stage-3b parts manifest identity is inconsistent"
        )
    parts = parts_manifest.get("parts")
    if not isinstance(parts, list) or len(parts) != int(completion["part_count"]):
        raise RoutedArtifactValidationError(
            "Stage-3b parts manifest count is inconsistent"
        )
    part_record_count = 0
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise RoutedArtifactValidationError(
                "Stage-3b parts manifest contains a non-mapping entry"
            )
        if part.get("index") != index:
            raise RoutedArtifactValidationError(
                "Stage-3b parts manifest indices are not contiguous"
            )
        if part.get("descriptor_size") != completion.get("descriptor_size"):
            raise RoutedArtifactValidationError(
                "Stage-3b part descriptor size is inconsistent"
            )
        part_record_count += _positive_int(
            part.get("record_count"), role="Stage-3b part record count"
        )
    if part_record_count != int(completion["record_count"]):
        raise RoutedArtifactValidationError(
            "Stage-3b part record counts do not match completion"
        )


def _validate_stage3b_training_identity(
    artifact: Mapping[str, Any],
    report: Mapping[str, Any],
    provenance: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate the source-supervised router contract before endpoints load."""

    if artifact.get("target_mode") != "source_gamus" or report.get(
        "target_mode"
    ) != "source_gamus":
        raise RoutedArtifactValidationError(
            "Stage-3b target_mode must be source_gamus"
        )
    artifact_target = artifact.get("target_provenance")
    report_target = report.get("target_provenance")
    if (
        artifact_target != report_target
        or artifact_target != STAGE3B_TARGET_PROVENANCE
    ):
        raise RoutedArtifactValidationError(
            "Stage-3b artifact/report target provenance mismatch"
        )
    if artifact.get("stage3b_eligible_record_provenance") is not True:
        raise RoutedArtifactValidationError(
            "Stage-3b artifact lacks eligible record provenance"
        )

    artifact_guards = _require_mapping(
        artifact.get("source_threshold_guards"),
        role="Stage-3b source threshold guards",
    )
    report_guards = _require_mapping(
        report.get("source_threshold_guards"),
        role="reported Stage-3b source threshold guards",
    )
    if not _canonical_equal(artifact_guards, report_guards):
        raise RoutedArtifactValidationError(
            "Stage-3b artifact/report source threshold guards mismatch"
        )
    if set(artifact_guards) != {
        "minimum_source_precision",
        "minimum_gamus_coverage",
        "minimum_gamus_selections",
        "minimum_routed_utility_gain_vs_protected",
        "selection_priority",
    }:
        raise RoutedArtifactValidationError(
            "Stage-3b source threshold guards have an unexpected contract"
        )
    selection_priority = artifact_guards.get("selection_priority")
    if not isinstance(selection_priority, list) or tuple(
        selection_priority
    ) != _STAGE3B_SELECTION_PRIORITY:
        raise RoutedArtifactValidationError(
            "Stage-3b threshold selection priority is unsupported"
        )
    minimum_precision = _finite_float(
        artifact_guards.get("minimum_source_precision"),
        role="minimum Stage-3b source precision",
    )
    minimum_coverage = _finite_float(
        artifact_guards.get("minimum_gamus_coverage"),
        role="minimum Stage-3b GAMUS coverage",
    )
    minimum_gamus = _positive_int(
        artifact_guards.get("minimum_gamus_selections"),
        role="minimum Stage-3b GAMUS selections",
    )
    minimum_utility_gain = _finite_float(
        artifact_guards.get("minimum_routed_utility_gain_vs_protected"),
        role="minimum Stage-3b routed utility gain",
    )
    if minimum_precision < 0.99 or minimum_precision > 1.0:
        raise RoutedArtifactValidationError(
            "Stage-3b configured source precision must be at least 0.99"
        )
    if not 0.0 <= minimum_coverage <= 1.0:
        raise RoutedArtifactValidationError(
            "Stage-3b configured GAMUS coverage is outside [0, 1]"
        )
    if minimum_utility_gain < 0.0:
        raise RoutedArtifactValidationError(
            "Stage-3b cannot permit routed utility regression"
        )

    artifact_training = _require_mapping(
        artifact.get("training_config"), role="Stage-3b training config"
    )
    report_training = _require_mapping(
        report.get("training_config"), role="reported Stage-3b training config"
    )
    if not _canonical_equal(artifact_training, report_training):
        raise RoutedArtifactValidationError(
            "Stage-3b artifact/report training config mismatch"
        )
    configured_precision = _finite_float(
        artifact_training.get("minimum_precision"),
        role="Stage-3b training minimum precision",
    )
    maximum_regression = _finite_float(
        artifact_training.get("maximum_mean_utility_regression"),
        role="Stage-3b maximum mean utility regression",
    )
    if not _same_float(configured_precision, minimum_precision):
        raise RoutedArtifactValidationError(
            "Stage-3b precision differs between training and source guards"
        )
    if maximum_regression != 0.0:
        raise RoutedArtifactValidationError(
            "Stage-3b training config permits utility regression"
        )

    artifact_launch = _require_mapping(
        artifact.get("training_launch"), role="Stage-3b training launch"
    )
    report_launch = _require_mapping(
        report.get("training_launch"), role="reported Stage-3b training launch"
    )
    if not _canonical_equal(artifact_launch, report_launch):
        raise RoutedArtifactValidationError(
            "Stage-3b artifact/report training launch mismatch"
        )
    launch_precision = _finite_float(
        artifact_launch.get("minimum_source_precision"),
        role="Stage-3b launch minimum source precision",
    )
    launch_coverage = _finite_float(
        artifact_launch.get("minimum_gamus_coverage"),
        role="Stage-3b launch minimum GAMUS coverage",
    )
    launch_gamus = _positive_int(
        artifact_launch.get("minimum_gamus_selections"),
        role="Stage-3b launch minimum GAMUS selections",
    )
    if (
        artifact_launch.get("target_mode") != "source_gamus"
        or not _same_float(launch_precision, minimum_precision)
        or not _same_float(launch_coverage, minimum_coverage)
        or launch_gamus != minimum_gamus
    ):
        raise RoutedArtifactValidationError(
            "Stage-3b training launch differs from source guards"
        )
    if artifact_launch.get("hidden_features") != artifact.get("hidden_features"):
        raise RoutedArtifactValidationError(
            "Stage-3b launch hidden-feature count differs from artifact"
        )
    if artifact_launch.get("epochs") != artifact_training.get("epochs"):
        raise RoutedArtifactValidationError(
            "Stage-3b launch epoch count differs from training config"
        )

    config_path_value = artifact_launch.get("config_path")
    if not isinstance(config_path_value, str) or not config_path_value.strip():
        raise RoutedArtifactValidationError(
            "Stage-3b launch has no authenticated config path"
        )
    config_path = _resolved_file(config_path_value, role="Stage-3b training config")
    expected_config_hash = _hex_digest(
        artifact_launch.get("config_sha256"), role="Stage-3b training config hash"
    )
    if (
        _stable_file_digest(config_path, role="Stage-3b training config")
        != expected_config_hash
    ):
        raise RoutedArtifactValidationError(
            "Stage-3b training config bytes no longer match the launch"
        )
    expected_trainer_hash = _hex_digest(
        artifact_launch.get("trainer_sha256"), role="Stage-3b trainer hash"
    )
    trainer_path = Path(__file__).resolve().parents[3] / "scripts" / "train_scene_router.py"
    trainer_path = _resolved_file(trainer_path, role="Stage-3b trainer")
    if (
        _stable_file_digest(trainer_path, role="Stage-3b trainer")
        != expected_trainer_hash
    ):
        raise RoutedArtifactValidationError(
            "Stage-3b trainer bytes no longer match the launch"
        )

    _validate_stage3b_record_files(completion, provenance, artifact_launch)
    return artifact_guards


def _validate_stage3b_source_metrics(
    report: Mapping[str, Any],
    threshold_report: Mapping[str, Any],
    provenance: Mapping[str, Any],
    completion: Mapping[str, Any],
    guards: Mapping[str, Any],
) -> None:
    """Prove that the reported source gate actually passes every guard."""

    records = _require_mapping(report.get("records"), role="Stage-3b record counts")
    if set(records) != {"train", "calibration"}:
        raise RoutedArtifactValidationError(
            "Stage-3b report must contain exact train/calibration counts"
        )
    parsed_records = {
        key: _nonnegative_int(value, role=f"Stage-3b {key} record count")
        for key, value in records.items()
    }
    if sum(parsed_records.values()) != int(completion["record_count"]):
        raise RoutedArtifactValidationError(
            "Stage-3b report count does not match completed records"
        )

    source_counts = _require_mapping(
        report.get("source_partition_counts"),
        role="Stage-3b source partition counts",
    )
    if set(source_counts) != {"train", "calibration"}:
        raise RoutedArtifactValidationError(
            "Stage-3b source counts must contain train and calibration"
        )
    parsed_source_counts: dict[str, dict[str, int]] = {}
    for partition in ("train", "calibration"):
        partition_counts = _require_mapping(
            source_counts.get(partition),
            role=f"Stage-3b {partition} source counts",
        )
        if set(partition_counts) != {"gamus", "legacy"}:
            raise RoutedArtifactValidationError(
                f"Stage-3b {partition} source counts must be GAMUS and legacy"
            )
        parsed_source_counts[partition] = {
            source: _nonnegative_int(
                value, role=f"Stage-3b {partition}/{source} count"
            )
            for source, value in partition_counts.items()
        }
        if sum(parsed_source_counts[partition].values()) != parsed_records[partition]:
            raise RoutedArtifactValidationError(
                f"Stage-3b {partition} source counts disagree with records"
            )

    data = _require_mapping(provenance.get("data"), role="Stage-3b record data")
    if data.get("included_source_splits") != [
        "gamus/train",
        "gamus/val",
        "legacy/train",
        "legacy/val",
    ] or data.get("excluded_source_splits") != ["gamus/test", "legacy/test"]:
        raise RoutedArtifactValidationError(
            "Stage-3b record provenance has unexpected source splits"
        )
    if not _canonical_equal(data.get("scored_partition_counts"), parsed_records):
        raise RoutedArtifactValidationError(
            "Stage-3b scored partition counts disagree with its report"
        )
    expected_source_split_counts = {
        "gamus/train": parsed_source_counts["train"]["gamus"],
        "gamus/val": parsed_source_counts["calibration"]["gamus"],
        "legacy/train": parsed_source_counts["train"]["legacy"],
        "legacy/val": parsed_source_counts["calibration"]["legacy"],
    }
    if not _canonical_equal(
        data.get("scored_source_split_counts"), expected_source_split_counts
    ):
        raise RoutedArtifactValidationError(
            "Stage-3b scored source/split counts disagree with its report"
        )

    metrics = _require_mapping(
        threshold_report.get("metrics"), role="Stage-3b threshold metrics"
    )
    minimum_precision = _finite_float(
        guards.get("minimum_source_precision"),
        role="minimum Stage-3b source precision",
    )
    minimum_coverage = _finite_float(
        guards.get("minimum_gamus_coverage"),
        role="minimum Stage-3b GAMUS coverage",
    )
    minimum_gamus = _positive_int(
        guards.get("minimum_gamus_selections"),
        role="minimum Stage-3b GAMUS selections",
    )
    minimum_gain = _finite_float(
        guards.get("minimum_routed_utility_gain_vs_protected"),
        role="minimum Stage-3b routed utility gain",
    )
    selected_gamus = _nonnegative_int(
        metrics.get("selected_gamus"), role="selected GAMUS scenes"
    )
    selected_legacy = _nonnegative_int(
        metrics.get("selected_legacy"), role="selected legacy scenes"
    )
    selected_total = _positive_int(
        metrics.get("candidate_selected"), role="selected candidate scenes"
    )
    if selected_total != selected_gamus + selected_legacy:
        raise RoutedArtifactValidationError(
            "Stage-3b selected legacy/GAMUS counts are inconsistent"
        )
    calibration_gamus = parsed_source_counts["calibration"]["gamus"]
    calibration_legacy = parsed_source_counts["calibration"]["legacy"]
    if selected_gamus > calibration_gamus or selected_legacy > calibration_legacy:
        raise RoutedArtifactValidationError(
            "Stage-3b selected source counts exceed calibration support"
        )
    measured_precision = _finite_float(
        metrics.get("source_precision"), role="measured Stage-3b source precision"
    )
    measured_coverage = _finite_float(
        metrics.get("gamus_coverage"), role="measured Stage-3b GAMUS coverage"
    )
    expected_precision = selected_gamus / selected_total
    expected_coverage = (
        selected_gamus / calibration_gamus if calibration_gamus else 0.0
    )
    if (
        not _same_float(measured_precision, expected_precision)
        or not _same_float(measured_coverage, expected_coverage)
    ):
        raise RoutedArtifactValidationError(
            "Stage-3b precision/coverage do not match selected source counts"
        )
    if (
        measured_precision < minimum_precision
        or measured_coverage < minimum_coverage
        or selected_gamus < minimum_gamus
    ):
        raise RoutedArtifactValidationError(
            "Stage-3b source metrics do not pass configured guards"
        )

    for alias, expected in (
        ("precision", measured_precision),
        ("source_recall", measured_coverage),
        ("recall", measured_coverage),
    ):
        if not _same_float(
            _finite_float(metrics.get(alias), role=f"Stage-3b {alias}"), expected
        ):
            raise RoutedArtifactValidationError(
                f"Stage-3b {alias} is inconsistent with source counts"
            )
    for field, expected in (
        ("tp", selected_gamus),
        ("fp", selected_legacy),
        ("fn", calibration_gamus - selected_gamus),
        ("tn", calibration_legacy - selected_legacy),
        ("scene_count", calibration_gamus + calibration_legacy),
    ):
        if _nonnegative_int(metrics.get(field), role=f"Stage-3b {field}") != expected:
            raise RoutedArtifactValidationError(
                f"Stage-3b {field} is inconsistent with source counts"
            )

    measured_gain = _finite_float(
        metrics.get("routed_utility_gain_vs_protected"),
        role="measured Stage-3b routed utility gain",
    )
    routed_gain_alias = _finite_float(
        metrics.get("routed_gain_vs_fallback"),
        role="measured Stage-3b routed gain alias",
    )
    if (
        measured_gain < minimum_gain
        or measured_gain < 0.0
        or not _same_float(measured_gain, routed_gain_alias)
    ):
        raise RoutedArtifactValidationError(
            "Stage-3b routed utility gain does not pass its no-regression guard"
        )
    fallback_mean = _finite_float(
        metrics.get("fallback_mean_utility"),
        role="Stage-3b fallback mean utility",
    )
    routed_mean = _finite_float(
        metrics.get("routed_mean_utility"), role="Stage-3b routed mean utility"
    )
    if not _same_float(fallback_mean - routed_mean, measured_gain):
        raise RoutedArtifactValidationError(
            "Stage-3b routed utility gain is arithmetically inconsistent"
        )


def load_guarded_routed_surface(
    router_artifact_path: str | Path,
    router_report_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> GuardedRoutedSurfaceBundle:
    """Authenticate endpoints/router/report and return a read-only routed model.

    Every mismatch raises before a model is returned.  This function never
    writes a checkpoint, updates the application pointer, or evaluates a test
    split.
    """

    artifact_path = _resolved_file(router_artifact_path, role="router artifact")
    report_path = _resolved_file(router_report_path, role="router report")
    artifact, artifact_hash = _stable_router_artifact(artifact_path)
    report, report_hash = _stable_json(report_path, role="router report")
    artifact_schema = artifact.get("artifact_schema")
    report_schema = report.get("artifact_schema")
    is_stage3b = (
        artifact_schema == STAGE3B_ROUTER_ARTIFACT_SCHEMA
        and report_schema == STAGE3B_ROUTER_REPORT_SCHEMA
    )
    is_stage3_v1 = (
        artifact_schema == ROUTER_ARTIFACT_SCHEMA
        and report_schema == ROUTER_REPORT_SCHEMA
    )
    if not is_stage3_v1 and not is_stage3b:
        raise RoutedArtifactValidationError("Unsupported router artifact schema")
    expected_artifact_type = (
        "offline_scene_domain_router_only"
        if is_stage3b
        else "offline_scene_router_only"
    )
    expected_report_type = (
        "offline_scene_domain_router_report"
        if is_stage3b
        else "offline_scene_router_report"
    )
    if artifact.get("artifact_type") != expected_artifact_type:
        raise RoutedArtifactValidationError("Unexpected router artifact type")
    if report.get("artifact_type") != expected_report_type:
        raise RoutedArtifactValidationError("Unexpected router report type")
    if artifact.get("stage3_eligible_record_provenance") is not True:
        raise RoutedArtifactValidationError(
            "Router artifact was not trained from a complete Stage-3 record store"
        )
    if artifact.get("threshold_eligible") is not True:
        raise RoutedArtifactValidationError("Router artifact is not guard-eligible")
    threshold_report = report.get("threshold")
    if not isinstance(threshold_report, Mapping) or threshold_report.get(
        "eligible"
    ) is not True:
        raise RoutedArtifactValidationError("Router report says the route is ineligible")
    if _hex_digest(
        report.get("router_artifact_sha256"), role="reported router artifact hash"
    ) != artifact_hash:
        raise RoutedArtifactValidationError("Router report/artifact SHA-256 mismatch")
    reported_path = report.get("router_artifact_path")
    if (
        not isinstance(reported_path, str)
        or Path(reported_path).expanduser().resolve() != artifact_path
    ):
        raise RoutedArtifactValidationError("Router report points to a different artifact")

    artifact_provenance, artifact_completion = _validate_embedded_record_identity(
        artifact, role="router artifact"
    )
    report_provenance, report_completion = _validate_embedded_record_identity(
        report, role="router report"
    )
    if not _canonical_equal(artifact_provenance, report_provenance):
        raise RoutedArtifactValidationError(
            "Router artifact/report record provenance mismatch"
        )
    if not _canonical_equal(artifact_completion, report_completion):
        raise RoutedArtifactValidationError(
            "Router artifact/report record completion mismatch"
        )
    reported_records = report.get("records")
    if not isinstance(reported_records, Mapping) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in reported_records.values()
    ):
        raise RoutedArtifactValidationError("Router report record counts are invalid")
    if sum(int(value) for value in reported_records.values()) != int(
        artifact_completion["record_count"]
    ):
        raise RoutedArtifactValidationError(
            "Router report count does not match completed record store"
        )

    artifact_gain = _finite_float(
        artifact.get("minimum_candidate_gain"), role="minimum candidate gain"
    )
    report_gain = _finite_float(
        report.get("minimum_candidate_gain"), role="reported minimum candidate gain"
    )
    generation_config = artifact_provenance.get("generation_config")
    if not isinstance(generation_config, Mapping):
        raise RoutedArtifactValidationError("Record generation config is invalid")
    utility_config = generation_config.get("utility")
    if not isinstance(utility_config, Mapping):
        raise RoutedArtifactValidationError("Record utility config is invalid")
    provenance_gain = _finite_float(
        utility_config.get("minimum_candidate_gain"),
        role="provenance minimum candidate gain",
    )
    if artifact_gain < 0 or artifact_gain != report_gain or artifact_gain != provenance_gain:
        raise RoutedArtifactValidationError(
            "Minimum candidate gain differs across artifact/report/provenance"
        )

    if is_stage3b:
        stage3b_guards = _validate_stage3b_training_identity(
            artifact,
            report,
            artifact_provenance,
            artifact_completion,
        )
        _validate_stage3b_source_metrics(
            report,
            threshold_report,
            artifact_provenance,
            artifact_completion,
            stage3b_guards,
        )
    else:
        training_config = artifact.get("training_config")
        threshold_metrics = threshold_report.get("metrics")
        if not isinstance(training_config, Mapping) or not isinstance(
            threshold_metrics, Mapping
        ):
            raise RoutedArtifactValidationError(
                "Router training guards or threshold metrics are missing"
            )
        minimum_precision = _finite_float(
            training_config.get("minimum_precision"), role="minimum router precision"
        )
        maximum_regression = _finite_float(
            training_config.get("maximum_mean_utility_regression"),
            role="maximum router utility regression",
        )
        minimum_selections = _positive_int(
            training_config.get("minimum_candidate_selections"),
            role="minimum candidate selections",
        )
        measured_precision = _finite_float(
            threshold_metrics.get("precision"), role="measured router precision"
        )
        measured_gain = _finite_float(
            threshold_metrics.get("routed_gain_vs_fallback"),
            role="measured routed gain",
        )
        measured_selections = _positive_int(
            threshold_metrics.get("candidate_selected"),
            role="measured candidate selections",
        )
        if (
            not 0.0 <= minimum_precision <= 1.0
            or maximum_regression < 0.0
            or measured_precision < minimum_precision
            or measured_gain < -maximum_regression
            or measured_selections < minimum_selections
        ):
            raise RoutedArtifactValidationError(
                "Router report metrics do not actually pass its training guards"
            )

    protected = _endpoint_field(artifact_provenance, "protected")
    candidate = _endpoint_field(artifact_provenance, "gamus_stage1")
    protected_path = protected.get("path")
    candidate_path = candidate.get("path")
    if not isinstance(protected_path, str) or not isinstance(candidate_path, str):
        raise RoutedArtifactValidationError("Endpoint provenance paths must be strings")
    protected_hash = _hex_digest(
        protected.get("sha256"), role="protected endpoint hash"
    )
    candidate_hash = _hex_digest(
        candidate.get("sha256"), role="GAMUS Stage-1 endpoint hash"
    )
    shared_provenance = _require_mapping(
        artifact_provenance.get("shared_model"),
        role="record shared-model provenance",
    )
    base_provenance = _require_mapping(
        shared_provenance.get("base_checkpoint"),
        role="record base-checkpoint provenance",
    )
    base_hash = _hex_digest(
        base_provenance.get("sha256"), role="base checkpoint hash"
    )
    try:
        endpoints = load_stage3_endpoint_packs(
            protected_path,
            candidate_path,
            expected_protected_sha256=protected_hash,
            expected_gamus_stage1_sha256=candidate_hash,
            expected_base_sha256=base_hash,
            reconstruct_shared_model=True,
        )
    except Stage3CheckpointValidationError as error:
        raise RoutedArtifactValidationError(
            f"Endpoint pair failed guarded reconstruction: {error}"
        ) from error
    _verify_endpoint_diagnostics(artifact_provenance, endpoints.diagnostics)

    feature_channels = _positive_int(
        artifact.get("feature_channels"), role="router feature_channels"
    )
    hidden_features = _positive_int(
        artifact.get("hidden_features"), role="router hidden_features"
    )
    threshold = _finite_float(
        artifact.get("decision_threshold"), role="router decision_threshold"
    )
    reported_threshold = _finite_float(
        threshold_report.get("value"), role="reported decision threshold"
    )
    if not 0.0 < threshold < 1.0 or threshold != reported_threshold:
        raise RoutedArtifactValidationError(
            "Router decision threshold is invalid or differs from its report"
        )
    if feature_channels != endpoints.protected.feature_channels:
        raise RoutedArtifactValidationError(
            "Router feature count does not match the verified endpoints"
        )
    if int(artifact_completion["descriptor_size"]) != 2 * feature_channels:
        raise RoutedArtifactValidationError(
            "Router feature count does not match completed record descriptors"
        )
    state = artifact.get("router_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise RoutedArtifactValidationError("Router artifact has no state dictionary")
    if not all(isinstance(name, str) for name in state) or not all(
        isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise RoutedArtifactValidationError(
            "Router state dictionary must contain named tensors only"
        )
    if any(
        (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all())
        for value in state.values()
    ):
        raise RoutedArtifactValidationError("Router state contains non-finite tensors")

    router = ConservativeSceneRouter(
        feature_channels,
        hidden_features=hidden_features,
        decision_threshold=threshold,
    )
    try:
        router.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise RoutedArtifactValidationError(
            f"Router state/configuration mismatch: {error}"
        ) from error
    loaded_threshold = float(router.decision_threshold.detach().cpu())
    if not math.isclose(loaded_threshold, threshold, rel_tol=0.0, abs_tol=1.0e-7):
        raise RoutedArtifactValidationError(
            "Router state threshold differs from authenticated metadata"
        )

    endpoint_router = FrozenDualSurfaceHeadRouter(
        endpoints.protected,
        endpoints.gamus_stage1,
        router,
    )
    model = RoutedDomainGatedSurfaceNet(
        endpoints.require_shared_model(), endpoint_router
    )
    model.to(device).eval().requires_grad_(False)
    return GuardedRoutedSurfaceBundle(
        model=model,
        diagnostics=RoutedArtifactDiagnostics(
            router_artifact_path=str(artifact_path),
            router_artifact_sha256=artifact_hash,
            router_report_path=str(report_path),
            router_report_sha256=report_hash,
            feature_channels=feature_channels,
            hidden_features=hidden_features,
            decision_threshold=threshold,
            endpoint_diagnostics=endpoints.diagnostics,
            record_config_sha256=str(artifact_provenance["config_sha256"]),
            record_count=int(artifact_completion["record_count"]),
        ),
        router_report=report,
    )


__all__ = [
    "GuardedRoutedSurfaceBundle",
    "ROUTER_ARTIFACT_SCHEMA",
    "ROUTER_REPORT_SCHEMA",
    "STAGE3B_ROUTER_ARTIFACT_SCHEMA",
    "STAGE3B_ROUTER_REPORT_SCHEMA",
    "STAGE3B_TARGET_PROVENANCE",
    "RoutedArtifactDiagnostics",
    "RoutedArtifactValidationError",
    "load_guarded_routed_surface",
]
