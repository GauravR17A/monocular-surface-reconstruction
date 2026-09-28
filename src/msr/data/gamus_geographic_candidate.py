"""Strict preparation for a GAMUS head trained only on DC and Philadelphia.

This module never discovers the official GAMUS test split.  It authenticates a
predeclared whole-city contract, derives learning-only sampling statistics, and
proves that a candidate configuration differs from the frozen head-only recipe
only where the geographic protocol requires it to differ.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .gamus_geographic_holdout import (
    GAMUS_GEOGRAPHIC_HOLDOUT_SCHEMA,
    assert_training_ids_respect_contract,
    file_sha256,
    load_sealed_holdout_contract,
    parse_gamus_tile_id,
    role_sample_ids,
)


GAMUS_GEOGRAPHIC_CANDIDATE_SCHEMA = (
    "msr.gamus_six_class_head_only_geographic.v1"
)
GAMUS_SAMPLING_INDEX_SCHEMA = "msr.gamus.train_six_class_tile_index.v1"
EXPECTED_EXPERIMENT_NAME = (
    "multidomain_surface_gamus_six_class_head_only_geographic_v1"
)
GAMUS_SPATIAL_REFINED_GEOGRAPHIC_CANDIDATE_SCHEMA = (
    "msr.gamus_six_class_spatial_refined_geographic.v1"
)
EXPECTED_SPATIAL_REFINED_EXPERIMENT_NAME = (
    "multidomain_surface_gamus_six_class_spatial_refined_geographic_v1"
)
SUPPORTED_GEOGRAPHIC_CANDIDATES = {
    GAMUS_GEOGRAPHIC_CANDIDATE_SCHEMA: EXPECTED_EXPERIMENT_NAME,
    GAMUS_SPATIAL_REFINED_GEOGRAPHIC_CANDIDATE_SCHEMA: (
        EXPECTED_SPATIAL_REFINED_EXPERIMENT_NAME
    ),
}
_SAMPLE_ID_PATTERN = re.compile(r'"sample_id"\s*:\s*"([A-Z0-9_]+)"')
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _normalized_sha(value: object, name: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{name} must contain 64 lowercase SHA-256 hex digits")
    return digest


def resolve_project_path(project_root: str | Path, value: object) -> Path:
    """Resolve one config path relative to the project, never the CWD."""

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("required path is blank")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(project_root).expanduser().resolve() / path
    return path.resolve()


def load_json_file_with_hash(
    path: str | Path,
    *,
    expected_sha256: str,
    name: str,
) -> dict[str, Any]:
    """Load a JSON object after authenticating its exact persisted bytes."""

    artifact = Path(path).expanduser().resolve()
    expected = _normalized_sha(expected_sha256, f"{name} SHA-256")
    if not artifact.is_file():
        raise FileNotFoundError(f"{name} does not exist: {artifact}")
    actual = file_sha256(artifact)
    if actual != expected:
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected}, found {actual}"
        )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def derive_learning_sampling_index(
    source_path: str | Path,
    *,
    expected_source_sha256: str,
    contract: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Regenerate a deterministic sampling index for learning IDs only.

    The source is the already sealed full-train scan.  Holdout rows are used
    only for exact inventory accounting: their JSON payloads are not decoded,
    aggregated, weighted, or copied.  Therefore no holdout statistic can
    influence the learning sampler.
    """

    source = Path(source_path).expanduser().resolve()
    expected = _normalized_sha(
        expected_source_sha256, "source GAMUS sampling-index SHA-256"
    )
    if not source.is_file():
        raise FileNotFoundError(f"source GAMUS sampling index is missing: {source}")
    actual = file_sha256(source)
    if actual != expected:
        raise ValueError(
            "source GAMUS sampling-index SHA-256 mismatch: "
            f"expected {expected}, found {actual}"
        )

    learning = set(role_sample_ids(contract, "learning"))
    holdout = set(role_sample_ids(contract, "geographic_holdout"))
    if not learning or not holdout or learning & holdout:
        raise ValueError("holdout contract has invalid learning/holdout roles")

    selected: dict[str, dict[str, Any]] = {}
    observed_ids: set[str] = set()
    skipped_holdout_rows = 0
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            match = _SAMPLE_ID_PATTERN.search(line)
            if match is None:
                raise ValueError(
                    f"sampling-index line {line_number} has no extractable sample_id"
                )
            sample_id = match.group(1)
            if sample_id in observed_ids:
                raise ValueError(f"duplicate sampling-index ID: {sample_id}")
            observed_ids.add(sample_id)
            if sample_id in holdout:
                skipped_holdout_rows += 1
                continue
            if sample_id not in learning:
                raise ValueError(
                    "sampling index contains an ID outside the sealed train "
                    f"partition: {sample_id}"
                )
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"sampling-index line {line_number} is not an object")
            if row.get("schema") != GAMUS_SAMPLING_INDEX_SCHEMA:
                raise ValueError(
                    f"unexpected sampling schema for learning tile {sample_id}"
                )
            if str(row.get("sample_id", "")).strip() != sample_id:
                raise ValueError(f"sampling-index identity mismatch for {sample_id}")
            city = str(row.get("city", "")).strip()
            if parse_gamus_tile_id(sample_id).city != city:
                raise ValueError(f"sampling-index city mismatch for {sample_id}")
            class_path = Path(str(row.get("class_path", "")))
            normalized_parts = [part.lower() for part in class_path.parts]
            if "classes" not in normalized_parts or "train" not in normalized_parts:
                raise ValueError(f"sampling-index class path is not train-only: {sample_id}")
            if class_path.name != f"{sample_id}_CLS.h5":
                raise ValueError(f"sampling-index class path mismatch for {sample_id}")
            crop = _mapping(row.get("uniform_random_crop_384"), "crop statistics")
            probability = float(crop.get("water_hit_probability", -1.0))
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"invalid water-hit probability for {sample_id}")
            selected[sample_id] = row

    if observed_ids != learning | holdout:
        missing = sorted((learning | holdout) - observed_ids)
        extra = sorted(observed_ids - (learning | holdout))
        raise ValueError(
            "source sampling index does not exactly cover the sealed official-train "
            f"partition: missing={missing[:5]}, extra={extra[:5]}"
        )
    assert_training_ids_respect_contract(selected, contract)
    if skipped_holdout_rows != len(holdout):
        raise AssertionError("internal error: not every holdout row was excluded")

    serialized = b"".join(
        (
            json.dumps(
                selected[sample_id],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
        for sample_id in sorted(selected)
    )
    class_totals: Counter[str] = Counter()
    city_counts: Counter[str] = Counter()
    water_positive_tiles = 0
    weighted_probability_sum = 0.0
    for row in selected.values():
        city_counts[str(row["city"])] += 1
        counts = _mapping(row.get("class_counts"), "class counts")
        for name, count in counts.items():
            class_totals[str(name)] += int(count)
        probability = float(
            _mapping(row["uniform_random_crop_384"], "crop statistics")[
                "water_hit_probability"
            ]
        )
        weighted_probability_sum += probability
        water_positive_tiles += int(probability > 0.0)
    audit = {
        "schema": "msr.gamus.geographic_learning_sampling_audit.v1",
        "contract_id": contract.get("contract_id"),
        "contract_schema": GAMUS_GEOGRAPHIC_HOLDOUT_SCHEMA,
        "contract_assignment_sha256": contract.get("assignment_sha256"),
        "source_sampling_index": str(source),
        "source_sampling_index_sha256": actual,
        "derivation": (
            "exact-ID filter of the sealed full-train scan; excluded holdout rows "
            "were identified by sample_id only and were not JSON-decoded"
        ),
        "learning_rows": len(selected),
        "excluded_geographic_holdout_rows": skipped_holdout_rows,
        "official_test_rows_discovered_or_used": 0,
        "city_counts": dict(sorted(city_counts.items())),
        "learning_class_pixel_totals": dict(sorted(class_totals.items())),
        "learning_water_positive_crop_tiles": water_positive_tiles,
        "learning_mean_water_hit_probability": (
            weighted_probability_sum / len(selected)
        ),
        "output_sha256": __import__("hashlib").sha256(serialized).hexdigest(),
    }
    return serialized, audit


def _approved_ids(index: Mapping[str, Any], split: str) -> tuple[str, ...]:
    split_payload = _mapping(
        _mapping(index.get("splits"), "approved-index splits").get(split),
        f"approved-index {split} split",
    )
    values = split_payload.get("approved_sample_ids")
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value.strip() for value in values
    ):
        raise ValueError(f"approved-index {split} IDs are malformed")
    normalized = tuple(sorted(value.strip() for value in values))
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"approved-index {split} IDs contain duplicates")
    if int(split_payload.get("approved_count", -1)) != len(normalized):
        raise ValueError(f"approved-index {split} count is inconsistent")
    return normalized


def _sampling_ids(path: Path) -> tuple[str, ...]:
    values: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != GAMUS_SAMPLING_INDEX_SCHEMA:
                raise ValueError(f"unexpected sampling schema on line {line_number}")
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"sampling index line {line_number} has no sample ID")
            values.append(sample_id)
    if len(values) != len(set(values)):
        raise ValueError("learning sampling index contains duplicate IDs")
    return tuple(sorted(values))


def _compare_frozen_recipe(candidate: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    if candidate.get("model") != source.get("model"):
        raise ValueError("geographic candidate model recipe differs from frozen source")
    if candidate.get("training") != source.get("training"):
        raise ValueError("geographic candidate training recipe differs from frozen source")
    if candidate.get("evaluation") != source.get("evaluation"):
        raise ValueError("geographic candidate validation/selection recipe differs from source")
    candidate_experiment = deepcopy(
        dict(_mapping(candidate.get("experiment"), "candidate experiment"))
    )
    source_experiment = deepcopy(dict(_mapping(source.get("experiment"), "source experiment")))
    candidate_experiment.pop("name", None)
    source_experiment.pop("name", None)
    if candidate_experiment != source_experiment:
        raise ValueError("geographic candidate experiment controls differ from source")
    allowed_data_differences = {
        "approved_index_path",
        "approved_index_file_sha256",
        "fine_class_sampling_index_path",
        "fine_class_sampling_index_sha256",
    }
    candidate_data = dict(_mapping(candidate.get("data"), "candidate data"))
    source_data = dict(_mapping(source.get("data"), "source data"))
    for key in allowed_data_differences:
        candidate_data.pop(key, None)
        source_data.pop(key, None)
    if candidate_data != source_data:
        raise ValueError("geographic candidate data recipe has unapproved differences")


def validate_geographic_candidate_config(
    config: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Authenticate inputs and prove NYC exclusion from this classifier's training."""

    root = Path(project_root).expanduser().resolve()
    experiment = _mapping(config.get("experiment"), "candidate experiment")
    protocol = _mapping(config.get("protocol"), "candidate protocol")
    schema = str(protocol.get("schema", ""))
    expected_experiment_name = SUPPORTED_GEOGRAPHIC_CANDIDATES.get(schema)
    if expected_experiment_name is None:
        raise ValueError("unsupported GAMUS geographic candidate protocol schema")
    if experiment.get("name") != expected_experiment_name:
        raise ValueError("unexpected geographic candidate experiment name")
    if protocol.get("official_test_policy") != "never_constructed_or_discovered":
        raise ValueError("geographic candidate must forbid official-test discovery")
    if protocol.get("auto_promotion") is not False:
        raise ValueError("geographic candidate must disable automatic promotion")

    source_recipe_path = resolve_project_path(root, protocol.get("source_recipe_config"))
    source_recipe = load_json_or_yaml_with_hash(
        source_recipe_path,
        expected_sha256=str(protocol.get("source_recipe_config_sha256", "")),
        name="frozen source recipe",
    )
    _compare_frozen_recipe(config, source_recipe)

    contract_path = resolve_project_path(root, protocol.get("holdout_contract"))
    contract_sha = _normalized_sha(
        protocol.get("holdout_contract_sha256"), "holdout-contract SHA-256"
    )
    contract = load_sealed_holdout_contract(
        contract_path, expected_sha256=contract_sha
    )
    if contract.get("contract_id") != "gamus-development-city-nyc-v1":
        raise ValueError("candidate is not bound to the locked NYC contract")
    if tuple(protocol.get("learning_cities", [])) != ("DC", "PHL"):
        raise ValueError("candidate learning cities must be exactly DC and PHL")
    if tuple(protocol.get("development_validation_cities", [])) != ("DC", "PHL"):
        raise ValueError("candidate validation cities must be exactly DC and PHL")
    if protocol.get("locked_holdout_city") != "NYC":
        raise ValueError("candidate locked holdout city must be NYC")
    holdout_index_path = resolve_project_path(
        root, protocol.get("locked_holdout_approved_index")
    )
    holdout_index = load_json_file_with_hash(
        holdout_index_path,
        expected_sha256=str(
            protocol.get("locked_holdout_approved_index_sha256", "")
        ),
        name="locked NYC approved index",
    )
    if set(_mapping(holdout_index.get("splits"), "holdout index splits")) != {
        "train"
    }:
        raise ValueError("locked NYC approved index must contain only a train key")
    locked_ids = _approved_ids(holdout_index, "train")
    if locked_ids != role_sample_ids(contract, "geographic_holdout"):
        raise ValueError("locked NYC approved index differs from the holdout contract")
    if {parse_gamus_tile_id(value).city for value in locked_ids} != {"NYC"}:
        raise ValueError("locked holdout index contains a non-NYC tile")

    data = _mapping(config.get("data"), "candidate data")
    if data.get("dataset") != "gamus":
        raise ValueError("geographic candidate must use the GAMUS loader")
    if data.get("require_approved_index") is not True:
        raise ValueError("geographic candidate must require an approved index")
    if data.get("require_complete_official_splits") is not True:
        raise ValueError("geographic candidate must authenticate source split counts")
    learning_index_path = resolve_project_path(root, data.get("approved_index_path"))
    learning_index = load_json_file_with_hash(
        learning_index_path,
        expected_sha256=str(data.get("approved_index_file_sha256", "")),
        name="learning approved index",
    )
    if set(_mapping(learning_index.get("splits"), "learning index splits")) != {
        "train",
        "val",
    }:
        raise ValueError("learning approved index must contain only train and val")
    learning_ids = _approved_ids(learning_index, "train")
    validation_ids = _approved_ids(learning_index, "val")
    assert_training_ids_respect_contract(learning_ids, contract)
    if validation_ids != role_sample_ids(contract, "development_validation"):
        raise ValueError("development validation IDs differ from the holdout contract")
    if {parse_gamus_tile_id(value).city for value in learning_ids} != {"DC", "PHL"}:
        raise ValueError("learning index contains an unexpected city")
    if {parse_gamus_tile_id(value).city for value in validation_ids} != {"DC", "PHL"}:
        raise ValueError("validation index contains an unexpected city")
    expected_counts = _mapping(protocol.get("expected_counts"), "expected counts")
    actual_counts = {
        "learning": len(learning_ids),
        "development_validation": len(validation_ids),
        "locked_geographic_holdout": len(locked_ids),
    }
    if {key: int(expected_counts.get(key, -1)) for key in actual_counts} != actual_counts:
        raise ValueError(
            f"candidate partition counts differ from the frozen protocol: {actual_counts}"
        )

    sampling_path = resolve_project_path(
        root, data.get("fine_class_sampling_index_path")
    )
    expected_sampling_sha = _normalized_sha(
        data.get("fine_class_sampling_index_sha256"),
        "learning sampling-index SHA-256",
    )
    if file_sha256(sampling_path) != expected_sampling_sha:
        raise ValueError("learning sampling-index SHA-256 mismatch")
    sampling_ids = _sampling_ids(sampling_path)
    if sampling_ids != learning_ids:
        raise ValueError("learning sampling index does not exactly match learning IDs")

    model = _mapping(config.get("model"), "candidate model")
    training = _mapping(config.get("training"), "candidate training")
    groups = training.get("parameter_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("geographic candidate requires exactly one parameter group")
    group = _mapping(groups[0], "candidate parameter group")
    if group.get("prefixes") != ["fine_semantic_head."]:
        raise ValueError("only fine_semantic_head may be trainable")
    if int(model.get("fine_semantic_classes", 0)) != 6:
        raise ValueError("geographic candidate requires six semantic classes")
    nonzero_height_terms = [
        key
        for key, value in training.items()
        if (
            key.endswith("weight")
            and key not in {"fine_semantic_weight", "weight_decay"}
            and isinstance(value, (int, float))
            and float(value) != 0.0
        )
    ]
    if nonzero_height_terms:
        raise ValueError(
            f"geographic head-only candidate has active auxiliary losses: {nonzero_height_terms}"
        )
    if float(training.get("fine_semantic_weight", 0.0)) <= 0.0:
        raise ValueError("fine-semantic supervision must remain active")

    protected_path = resolve_project_path(root, model.get("initial_checkpoint"))
    protected_sha = _normalized_sha(
        protocol.get("protected_checkpoint_sha256"),
        "protected-checkpoint SHA-256",
    )
    if file_sha256(protected_path) != protected_sha:
        raise ValueError("protected checkpoint SHA-256 mismatch")

    pointer_path = resolve_project_path(root, protocol.get("live_pointer_file"))
    pointer_sha = _normalized_sha(
        protocol.get("live_pointer_file_sha256"), "live-pointer SHA-256"
    )
    if file_sha256(pointer_path) != pointer_sha:
        raise ValueError("live pointer SHA-256 mismatch")
    # PowerShell 5 may have created the legacy pointer with a UTF-8 BOM.  Its
    # exact bytes are authenticated above; remove only that encoding marker
    # before interpreting the path value.
    pointer_value = pointer_path.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    pointer_target = resolve_project_path(root, pointer_value)
    if pointer_target != protected_path:
        raise ValueError("live pointer does not target the protected checkpoint")

    return {
        "schema": "msr.gamus.geographic_candidate_preflight.v1",
        "experiment": experiment["name"],
        "source_recipe": {
            "path": str(source_recipe_path),
            "sha256": file_sha256(source_recipe_path),
        },
        "contract": {
            "path": str(contract_path),
            "sha256": contract_sha,
            "assignment_sha256": contract.get("assignment_sha256"),
        },
        "learning": {
            "tiles": len(learning_ids),
            "cities": ["DC", "PHL"],
            "approved_index": str(learning_index_path),
            "approved_index_sha256": file_sha256(learning_index_path),
            "sampling_index": str(sampling_path),
            "sampling_index_sha256": expected_sampling_sha,
        },
        "epoch_selection": {
            "tiles": len(validation_ids),
            "cities": ["DC", "PHL"],
            "source": "existing approved validation split",
        },
        "locked_post_freeze_evaluation": {
            "tiles": len(locked_ids),
            "city": "NYC",
            "approved_index": str(holdout_index_path),
            "approved_index_sha256": file_sha256(holdout_index_path),
            "opened_by_preflight": False,
        },
        "trainable_prefixes": ["fine_semantic_head."],
        "height_state_policy": "all inherited tensors must remain bitwise equal",
        "official_test_constructed_or_discovered": False,
        "promotion_performed": False,
    }


def load_json_or_yaml_with_hash(
    path: str | Path,
    *,
    expected_sha256: str,
    name: str,
) -> dict[str, Any]:
    """Load authenticated JSON or YAML without weakening file identity checks."""

    artifact = Path(path).expanduser().resolve()
    expected = _normalized_sha(expected_sha256, f"{name} SHA-256")
    if not artifact.is_file():
        raise FileNotFoundError(f"{name} does not exist: {artifact}")
    actual = file_sha256(artifact)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, found {actual}")
    if artifact.suffix.lower() == ".json":
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    else:
        import yaml

        payload = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must decode to a mapping")
    return payload


__all__ = [
    "EXPECTED_EXPERIMENT_NAME",
    "EXPECTED_SPATIAL_REFINED_EXPERIMENT_NAME",
    "GAMUS_GEOGRAPHIC_CANDIDATE_SCHEMA",
    "GAMUS_SPATIAL_REFINED_GEOGRAPHIC_CANDIDATE_SCHEMA",
    "GAMUS_SAMPLING_INDEX_SCHEMA",
    "SUPPORTED_GEOGRAPHIC_CANDIDATES",
    "derive_learning_sampling_index",
    "load_json_file_with_hash",
    "load_json_or_yaml_with_hash",
    "resolve_project_path",
    "validate_geographic_candidate_config",
]
