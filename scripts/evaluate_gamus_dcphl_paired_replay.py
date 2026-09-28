"""Independent paired V3/V4 replay on the sealed DC+Philadelphia validation set.

By default this command deliberately reconstructs both checkpoints and
recomputes every identification score from pixels.  ``--baseline-only`` runs
the same exact replay for V3 alone so its city-specific reference can be sealed
before V4 exists.  Neither mode trusts training-log metrics, constructs an NYC
or official-test dataset, or changes the protected application pointer or any
checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msr.data.gamus_dataset import (  # noqa: E402
    GAMUS_APPROVED_INDEX_SCHEMA,
    GAMUS_OFFICIAL_SPLIT_COUNTS,
    GAMUS_SIX_CLASS_NAMES,
    GamusSampleRecord,
    GamusSurfaceDataset,
)
from msr.evaluation.classification_metrics import (  # noqa: E402
    StreamingClassBoundaryMetrics,
    StreamingMulticlassMetrics,
    StreamingWaterDarkPixelProxy,
    multiclass_metric_deltas,
)
from msr.inference.predict import load_predictor  # noqa: E402


SCHEMA = "msr.gamus_dcphl_paired_independent_replay.v1"
V3_PROTOCOL = "msr.gamus_six_class_spatial_refined_geographic.v1"
V4_PROTOCOL = "msr.gamus_six_class_hierarchical_dcphl.v4"
CLASS_NAMES = tuple(GAMUS_SIX_CLASS_NAMES)
ALLOWED_CITIES = ("DC", "PHL")
ROAD_INDEX = CLASS_NAMES.index("roads")
WATER_INDEX = CLASS_NAMES.index("water")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
BOUNDARY_COUNT_FIELDS = (
    "predicted_boundary_pixels",
    "reference_boundary_pixels",
    "matched_predicted_pixels",
    "matched_reference_pixels",
    "dilated_intersection_pixels",
    "dilated_union_pixels",
)
WATER_COUNT_FIELDS = (
    "dark_non_water_pixels",
    "false_water_on_dark_non_water_pixels",
    "all_false_water_pixels",
)


def _mapping(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a mapping")
    return value


def _sequence(value: object, role: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{role} must be a sequence")
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _required_sha256(value: object, role: str) -> str:
    result = str(value or "").strip().lower()
    if not SHA256_PATTERN.fullmatch(result):
        raise ValueError(f"{role} must be 64 lowercase SHA-256 hex digits")
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_identity(path: Path) -> dict[str, Any]:
    """Hash a file and fail if it changes while being hashed."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    before = resolved.stat()
    digest = file_sha256(resolved)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while it was being hashed: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be a mapping: {path}")
    return payload


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _city(sample_id: str) -> str:
    city = str(sample_id).split("_", maxsplit=1)[0]
    if city not in ALLOWED_CITIES:
        raise ValueError(
            f"Validation replay is DC+PHL-only; forbidden sample ID: {sample_id}"
        )
    return city


def _approved_ids(split: Mapping[str, Any], *, role: str) -> tuple[str, ...]:
    values = _sequence(split.get("approved_sample_ids"), f"{role} approved IDs")
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"{role} approved IDs contain a malformed value")
    result = tuple(str(value).strip() for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{role} approved IDs contain duplicates")
    if int(split.get("approved_count", -1)) != len(result):
        raise ValueError(f"{role} approved count is inconsistent")
    return result


def authenticate_replay_config(
    config_path: Path,
    *,
    expected_sha256: str,
    expected_protocol: str,
    expected_head_type: str,
    expected_validation_count: int = GAMUS_OFFICIAL_SPLIT_COUNTS["val"],
) -> dict[str, Any]:
    """Authenticate a V3/V4 config and its exact DC+PHL approved index."""

    path = config_path.expanduser().resolve()
    expected = _required_sha256(expected_sha256, "expected config SHA-256")
    identity = _stable_file_identity(path)
    if identity["sha256"] != expected:
        raise ValueError(
            f"Config SHA-256 mismatch: expected {expected}, found {identity['sha256']}"
        )
    config = _load_yaml(path)
    protocol = _mapping(config.get("protocol"), "config protocol")
    if protocol.get("schema") != expected_protocol:
        raise ValueError(
            f"Config protocol must be {expected_protocol!r}, found {protocol.get('schema')!r}"
        )
    if list(protocol.get("development_validation_cities", ())) != list(
        ALLOWED_CITIES
    ):
        raise ValueError("Development validation cities must be exactly DC and PHL")
    official_policy = protocol.get("official_test_policy")
    accepted_policy = {
        V3_PROTOCOL: "never_constructed_or_discovered",
        V4_PROTOCOL: "previously_consumed_forbidden_for_reuse",
    }[expected_protocol]
    if official_policy != accepted_policy:
        raise ValueError("Config does not carry the required official-test policy")
    if protocol.get("auto_promotion") is not False:
        raise ValueError("Evaluation config must disable automatic promotion")

    model = _mapping(config.get("model"), "config model")
    if int(model.get("fine_semantic_classes", -1)) != len(CLASS_NAMES):
        raise ValueError("Replay checkpoint must expose exactly six classes")
    if model.get("fine_semantic_head_type") != expected_head_type:
        raise ValueError(
            f"Expected {expected_head_type!r} head, found "
            f"{model.get('fine_semantic_head_type')!r}"
        )

    data = _mapping(config.get("data"), "config data")
    if data.get("dataset") != "gamus":
        raise ValueError("Independent replay only supports the GAMUS dataset")
    if data.get("require_approved_index") is not True:
        raise ValueError("A sealed approved index is mandatory")
    if data.get("require_relative_priors") is not True:
        raise ValueError("Exact replay requires the sealed relative-depth priors")
    if str(data.get("validation_radiometric_policy", "")) != "raw":
        raise ValueError("Exact paired replay requires raw validation radiometry")

    index_path = _resolve(str(data.get("approved_index_path", "")))
    declared_index_sha = _required_sha256(
        data.get("approved_index_file_sha256"), "approved-index SHA-256"
    )
    index_identity = _stable_file_identity(index_path)
    if index_identity["sha256"] != declared_index_sha:
        raise ValueError("Approved-index SHA-256 differs from the sealed config")
    index = _load_json(index_path)
    if index.get("schema") != GAMUS_APPROVED_INDEX_SCHEMA:
        raise ValueError("Approved index has an unexpected schema")
    if index.get("dataset") != "GAMUS":
        raise ValueError("Approved index does not identify GAMUS")
    splits = _mapping(index.get("splits"), "approved-index splits")
    if set(splits) != {"train", "val"}:
        raise ValueError(
            "DC+PHL replay index must contain only train and val; test/holdout access is forbidden"
        )
    train_ids = _approved_ids(_mapping(splits["train"], "train split"), role="train")
    val_split = _mapping(splits["val"], "validation split")
    validation_ids = _approved_ids(val_split, role="validation")
    for sample_id in (*train_ids, *validation_ids):
        _city(sample_id)
    if len(validation_ids) != expected_validation_count:
        raise ValueError(
            "Sealed development validation count mismatch: "
            f"expected {expected_validation_count}, found {len(validation_ids)}"
        )
    if int(val_split.get("source_count", -1)) != expected_validation_count:
        raise ValueError("Validation source count is not the sealed official-val count")
    semantic_ids = tuple(
        str(value)
        for value in _sequence(
            val_split.get("semantic_eligible_sample_ids"),
            "validation semantic-eligible IDs",
        )
    )
    if semantic_ids != validation_ids:
        raise ValueError(
            "Every approved validation tile must be semantic-eligible in exact replay order"
        )

    validation_contract = {
        "approved_index_sha256": declared_index_sha,
        "validation_ids_sha256": _canonical_sha256(list(validation_ids)),
        "validation_count": len(validation_ids),
        "city_counts": dict(sorted(Counter(_city(value) for value in validation_ids).items())),
        "dataset_root": str(_resolve(str(data.get("root", "")))),
        "relative_prior_root": str(
            _resolve(str(data.get("relative_prior_root", "")))
        ),
        "validation_patch_size": int(data.get("validation_patch_size", -1)),
        "rgb_scale": float(data.get("rgb_scale", -1.0)),
        "height_max_m": float(data.get("height_max_m", -1.0)),
        "validation_radiometric_policy": str(
            data.get("validation_radiometric_policy", "")
        ),
        "relative_priors_required": True,
    }
    if validation_contract["validation_patch_size"] <= 0:
        raise ValueError("validation_patch_size must be positive")
    if validation_contract["rgb_scale"] <= 0.0:
        raise ValueError("rgb_scale must be positive")
    if validation_contract["height_max_m"] <= 0.0:
        raise ValueError("height_max_m must be positive")
    validation_contract["contract_sha256"] = _canonical_sha256(validation_contract)
    return {
        "config": config,
        "config_identity": identity,
        "approved_index_identity": index_identity,
        "approved_index_declared_source_inventory_metadata_sha256": index.get(
            "source_inventory_metadata_sha256"
        ),
        "validation_ids": validation_ids,
        "validation_contract": validation_contract,
    }


def assert_paired_contracts(v3: Mapping[str, Any], v4: Mapping[str, Any]) -> None:
    """Prove that the two independent replays use exactly the same inputs."""

    if v3["validation_ids"] != v4["validation_ids"]:
        raise ValueError("V3 and V4 approved validation IDs differ")
    if v3["validation_contract"] != v4["validation_contract"]:
        raise ValueError("V3 and V4 validation/preprocessing contracts differ")


def authenticate_checkpoint(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    external_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a checkpoint's exact bytes and embedded recipe to its sealed YAML."""

    path = checkpoint_path.expanduser().resolve()
    expected = _required_sha256(expected_sha256, "expected checkpoint SHA-256")
    identity = _stable_file_identity(path)
    if identity["sha256"] != expected:
        raise ValueError(
            "Checkpoint SHA-256 mismatch: "
            f"expected {expected}, found {identity['sha256']}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    embedded = payload.get("config")
    if not isinstance(embedded, Mapping) or dict(embedded) != dict(external_config):
        raise ValueError("Checkpoint's embedded config differs from the sealed YAML")
    if not isinstance(payload.get("model"), Mapping):
        raise ValueError("Checkpoint has no model state")
    return {
        **identity,
        "epoch": payload.get("epoch"),
        "model_type": payload.get("model_type"),
        "embedded_config_sha256": _canonical_sha256(embedded),
    }


def _expected_record_path(root: Path, role: str, sample_id: str) -> Path:
    suffix = {
        "image": f"{sample_id}_RGB.h5",
        "height": f"{sample_id}_AGL.h5",
        "class": f"{sample_id}_CLS.h5",
    }[role]
    directory = {"image": "images", "height": "heights", "class": "classes"}[role]
    return (root / directory / "val" / suffix).resolve()


def validate_dataset_records(
    records: Sequence[GamusSampleRecord],
    *,
    expected_ids: Sequence[str],
    dataset_root: Path,
    prior_root: Path,
) -> None:
    """Reject any inventory drift or path capable of touching NYC/test data."""

    actual_ids = tuple(record.sample_id for record in records)
    if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
        raise ValueError("Discovered validation records differ from the sealed ID set")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Discovered validation records contain duplicate IDs")
    for record in records:
        sample_id = record.sample_id
        _city(sample_id)
        for role, actual in (
            ("image", record.image_path),
            ("height", record.height_path),
            ("class", record.class_path),
        ):
            expected = _expected_record_path(dataset_root, role, sample_id)
            if actual.resolve() != expected:
                raise ValueError(
                    f"Unexpected {role} source for {sample_id}: {actual.resolve()}"
                )
        expected_prior = (prior_root / "val" / f"{sample_id}_REL.h5").resolve()
        if record.relative_prior_path is None or record.relative_prior_path.resolve() != expected_prior:
            raise ValueError(f"Unexpected or missing relative prior for {sample_id}")


def build_source_binding(records: Sequence[GamusSampleRecord]) -> dict[str, Any]:
    """Cryptographically bind the exact RGB/label/prior files used by replay."""

    role_rows: dict[str, list[dict[str, Any]]] = {
        "image": [],
        "height": [],
        "class": [],
        "relative_prior": [],
    }
    for record in sorted(records, key=lambda item: item.sample_id):
        sources = {
            "image": record.image_path,
            "height": record.height_path,
            "class": record.class_path,
            "relative_prior": record.relative_prior_path,
        }
        for role, source in sources.items():
            if source is None:
                raise ValueError(f"Missing {role} source for {record.sample_id}")
            identity = _stable_file_identity(source)
            role_rows[role].append(
                {
                    "sample_id": record.sample_id,
                    "size_bytes": identity["size_bytes"],
                    "sha256": identity["sha256"],
                }
            )
    per_role: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for role in sorted(role_rows):
        rows = role_rows[role]
        tagged = [{"role": role, **row} for row in rows]
        all_rows.extend(tagged)
        per_role[role] = {
            "file_count": len(rows),
            "total_bytes": sum(int(row["size_bytes"]) for row in rows),
            "content_manifest_sha256": _canonical_sha256(tagged),
        }
    return {
        "algorithm": "SHA-256 over every file, then canonical SHA-256 over identities",
        "individual_hashes_embedded": False,
        "sample_count": len(records),
        "total_file_count": sum(item["file_count"] for item in per_role.values()),
        "total_bytes": sum(item["total_bytes"] for item in per_role.values()),
        "per_role": per_role,
        "combined_content_manifest_sha256": _canonical_sha256(all_rows),
    }


def source_metadata_snapshot(records: Sequence[GamusSampleRecord]) -> dict[str, Any]:
    """Cheap before/after guard around the content-bound validation sources."""

    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.sample_id):
        for role, source in (
            ("image", record.image_path),
            ("height", record.height_path),
            ("class", record.class_path),
            ("relative_prior", record.relative_prior_path),
        ):
            if source is None:
                raise ValueError(f"Missing {role} source for {record.sample_id}")
            path = source.resolve()
            stat = path.stat()
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "role": role,
                    "path": str(path),
                    "size_bytes": int(stat.st_size),
                    "modified_time_ns": int(stat.st_mtime_ns),
                }
            )
    return {
        "file_count": len(rows),
        "metadata_manifest_sha256": _canonical_sha256(rows),
    }


@dataclass
class _MetricGroup:
    six_class: StreamingMulticlassMetrics
    road_boundary: StreamingClassBoundaryMetrics
    water_dark: StreamingWaterDarkPixelProxy

    @classmethod
    def create(cls) -> "_MetricGroup":
        return cls(
            six_class=StreamingMulticlassMetrics(CLASS_NAMES),
            road_boundary=StreamingClassBoundaryMetrics(
                ROAD_INDEX, tolerance_pixels=2
            ),
            water_dark=StreamingWaterDarkPixelProxy(WATER_INDEX),
        )

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        dark: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        self.six_class.update(prediction, target, valid)
        self.road_boundary.update(prediction, target, valid)
        self.water_dark.update(prediction, target, dark, valid)

    def compute(self) -> dict[str, Any]:
        water_dark = self.water_dark.compute()
        water_dark["water_class_index"] = WATER_INDEX
        return {
            "six_class_identification": self.six_class.compute(),
            "road_boundary_quality": self.road_boundary.compute(),
            "water_dark_pixel_proxy": water_dark,
        }


def _number(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{role} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{role} must be finite")
    return result


def _probability(value: object, role: str) -> float:
    result = _number(value, role)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{role} must be within [0, 1]")
    return result


def validate_identification_metrics(metrics: Mapping[str, Any]) -> None:
    """Fail closed on invalid score ranges, formulas, or count relationships."""

    six = _mapping(metrics.get("six_class_identification"), "six-class metrics")
    if tuple(six.get("class_names", ())) != CLASS_NAMES:
        raise ValueError("Six-class metrics use an unexpected class order")
    matrix = np.asarray(six.get("confusion_matrix"))
    if matrix.shape != (6, 6) or not np.issubdtype(matrix.dtype, np.integer):
        raise ValueError("Confusion matrix must be an integer 6x6 matrix")
    if np.any(matrix < 0):
        raise ValueError("Confusion matrix cannot contain negative counts")
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    if int(six.get("total_valid_pixels", -1)) != total:
        raise ValueError("Six-class total does not match its confusion matrix")
    if int(six.get("correct_pixels", -1)) != correct:
        raise ValueError("Six-class correct count does not match its matrix")
    accuracy = _probability(six.get("accuracy"), "six-class accuracy")
    error = _probability(six.get("error_rate"), "six-class error rate")
    expected_accuracy = correct / total if total else 0.0
    if abs(accuracy - expected_accuracy) > 1.0e-12 or abs(error - (1.0 - accuracy if total else 0.0)) > 1.0e-12:
        raise ValueError("Six-class accuracy/error formula is inconsistent")

    per_class = _mapping(six.get("per_class"), "per-class metrics")
    macro = _mapping(six.get("macro"), "macro metrics")
    for index, name in enumerate(CLASS_NAMES):
        item = _mapping(per_class.get(name), f"{name} metrics")
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        true_positive = int(matrix[index, index])
        false_positive = predicted - true_positive
        false_negative = support - true_positive
        expected_counts = {
            "support_pixels": support,
            "predicted_pixels": predicted,
            "true_positive_pixels": true_positive,
            "false_positive_pixels": false_positive,
            "false_negative_pixels": false_negative,
        }
        for key, expected in expected_counts.items():
            if int(item.get(key, -1)) != expected:
                raise ValueError(f"{name}.{key} is inconsistent with the matrix")
        expected_scores = {
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / support if support else 0.0,
            "f1": (
                2.0 * true_positive
                / (2.0 * true_positive + false_positive + false_negative)
                if 2 * true_positive + false_positive + false_negative
                else 0.0
            ),
            "iou": (
                true_positive
                / (true_positive + false_positive + false_negative)
                if true_positive + false_positive + false_negative
                else 0.0
            ),
        }
        for metric_name, expected_score in expected_scores.items():
            actual_score = _probability(
                item.get(metric_name), f"{name} {metric_name}"
            )
            if abs(actual_score - expected_score) > 1.0e-12:
                raise ValueError(
                    f"{name} {metric_name} formula is inconsistent with the matrix"
                )
    for metric_name in ("precision", "recall", "f1", "iou"):
        values = [
            _probability(
                _mapping(per_class[name], f"{name} metrics").get(metric_name),
                f"{name} {metric_name}",
            )
            for name in CLASS_NAMES
        ]
        actual = _probability(macro.get(metric_name), f"macro {metric_name}")
        if abs(actual - sum(values) / len(values)) > 1.0e-12:
            raise ValueError(f"Macro {metric_name} is not the six-class mean")
        duplicate = _probability(
            six.get(f"macro_{metric_name}"), f"top-level macro {metric_name}"
        )
        if abs(duplicate - actual) > 1.0e-12:
            raise ValueError(f"Duplicate macro {metric_name} is inconsistent")

    boundary = _mapping(metrics.get("road_boundary_quality"), "road-boundary metrics")
    if int(boundary.get("class_index", -1)) != ROAD_INDEX:
        raise ValueError("Road-boundary metrics use the wrong class index")
    if int(boundary.get("tolerance_pixels", -1)) != 2:
        raise ValueError("Road-boundary tolerance must remain fixed at two pixels")
    for name in ("precision", "recall", "f1", "dilated_boundary_iou"):
        _probability(boundary.get(name), f"road-boundary {name}")
    boundary_counts = {name: int(boundary.get(name, -1)) for name in BOUNDARY_COUNT_FIELDS}
    if any(value < 0 for value in boundary_counts.values()):
        raise ValueError("Road-boundary counts must be nonnegative")
    if boundary_counts["matched_predicted_pixels"] > boundary_counts["predicted_boundary_pixels"]:
        raise ValueError("Matched predicted road boundaries exceed predicted boundaries")
    if boundary_counts["matched_reference_pixels"] > boundary_counts["reference_boundary_pixels"]:
        raise ValueError("Matched reference road boundaries exceed reference boundaries")
    if boundary_counts["dilated_intersection_pixels"] > boundary_counts["dilated_union_pixels"]:
        raise ValueError("Dilated road-boundary intersection exceeds its union")
    expected_precision = boundary_counts["matched_predicted_pixels"] / max(
        boundary_counts["predicted_boundary_pixels"], 1
    )
    expected_recall = boundary_counts["matched_reference_pixels"] / max(
        boundary_counts["reference_boundary_pixels"], 1
    )
    expected_f1 = (
        2.0 * expected_precision * expected_recall / (expected_precision + expected_recall)
        if expected_precision + expected_recall
        else 0.0
    )
    if abs(float(boundary["precision"]) - expected_precision) > 1.0e-12:
        raise ValueError("Road-boundary precision formula is inconsistent")
    if abs(float(boundary["recall"]) - expected_recall) > 1.0e-12:
        raise ValueError("Road-boundary recall formula is inconsistent")
    if abs(float(boundary["f1"]) - expected_f1) > 1.0e-12:
        raise ValueError("Road-boundary F1 formula is inconsistent")
    expected_boundary_iou = boundary_counts["dilated_intersection_pixels"] / max(
        boundary_counts["dilated_union_pixels"], 1
    )
    if (
        abs(float(boundary["dilated_boundary_iou"]) - expected_boundary_iou)
        > 1.0e-12
    ):
        raise ValueError("Road-boundary IoU formula is inconsistent")

    water = _mapping(metrics.get("water_dark_pixel_proxy"), "water-dark proxy")
    if int(water.get("water_class_index", WATER_INDEX)) != WATER_INDEX:
        raise ValueError("Water-dark proxy uses the wrong water class index")
    if water.get("is_shadow_proxy_not_ground_truth") is not True:
        raise ValueError("Water-dark diagnostic must be labelled as a proxy")
    for name in (
        "false_water_rate_on_dark_non_water",
        "dark_share_of_all_water_false_positives",
    ):
        _probability(water.get(name), f"water-dark {name}")
    water_counts = {name: int(water.get(name, -1)) for name in WATER_COUNT_FIELDS}
    if any(value < 0 for value in water_counts.values()):
        raise ValueError("Water-dark proxy counts must be nonnegative")
    if water_counts["false_water_on_dark_non_water_pixels"] > water_counts["dark_non_water_pixels"]:
        raise ValueError("Dark false-water count exceeds dark non-water support")
    if water_counts["false_water_on_dark_non_water_pixels"] > water_counts["all_false_water_pixels"]:
        raise ValueError("Dark false-water count exceeds all false-water count")
    water_false_positives = int(matrix[:, WATER_INDEX].sum() - matrix[WATER_INDEX, WATER_INDEX])
    if water_counts["all_false_water_pixels"] != water_false_positives:
        raise ValueError("Water false-positive count differs from the confusion matrix")
    expected_dark_rate = water_counts[
        "false_water_on_dark_non_water_pixels"
    ] / max(water_counts["dark_non_water_pixels"], 1)
    expected_dark_share = water_counts[
        "false_water_on_dark_non_water_pixels"
    ] / max(water_counts["all_false_water_pixels"], 1)
    if (
        abs(float(water["false_water_rate_on_dark_non_water"]) - expected_dark_rate)
        > 1.0e-12
    ):
        raise ValueError("Water-dark false-water rate formula is inconsistent")
    if (
        abs(
            float(water["dark_share_of_all_water_false_positives"])
            - expected_dark_share
        )
        > 1.0e-12
    ):
        raise ValueError("Water-dark false-water share formula is inconsistent")


def validate_replay_result(result: Mapping[str, Any]) -> None:
    overall = _mapping(result.get("overall"), "overall replay metrics")
    cities = _mapping(result.get("by_city"), "city replay metrics")
    if set(cities) != set(ALLOWED_CITIES):
        raise ValueError("Replay must report exactly DC and PHL")
    validate_identification_metrics(overall)
    for city in ALLOWED_CITIES:
        validate_identification_metrics(_mapping(cities[city], f"{city} metrics"))

    total_matrix = np.asarray(
        _mapping(overall["six_class_identification"], "overall six-class")[
            "confusion_matrix"
        ],
        dtype=np.int64,
    )
    city_matrix = sum(
        (
            np.asarray(
                _mapping(cities[city]["six_class_identification"], f"{city} six-class")[
                    "confusion_matrix"
                ],
                dtype=np.int64,
            )
            for city in ALLOWED_CITIES
        ),
        start=np.zeros((6, 6), dtype=np.int64),
    )
    if not np.array_equal(total_matrix, city_matrix):
        raise ValueError("Per-city confusion matrices do not sum to overall")
    for section, fields in (
        ("road_boundary_quality", BOUNDARY_COUNT_FIELDS),
        ("water_dark_pixel_proxy", WATER_COUNT_FIELDS),
    ):
        total_section = _mapping(overall[section], f"overall {section}")
        for field in fields:
            city_sum = sum(int(cities[city][section][field]) for city in ALLOWED_CITIES)
            if int(total_section[field]) != city_sum:
                raise ValueError(f"Per-city {section}.{field} does not sum to overall")
    sample_counts = _mapping(result.get("evaluated_samples_by_city"), "sample counts")
    if set(sample_counts) != set(ALLOWED_CITIES):
        raise ValueError("Sample counts must contain exactly DC and PHL")
    if sum(int(value) for value in sample_counts.values()) != int(
        result.get("evaluated_sample_count", -1)
    ):
        raise ValueError("Per-city sample counts do not sum to overall")
    city_id_hashes = _mapping(
        result.get("evaluated_ids_by_city_sha256"), "per-city ID hashes"
    )
    if set(city_id_hashes) != set(ALLOWED_CITIES) or any(
        not SHA256_PATTERN.fullmatch(str(value)) for value in city_id_hashes.values()
    ):
        raise ValueError("Per-city evaluated-ID hashes are incomplete or malformed")


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def replay_checkpoint(
    checkpoint: Path,
    loader: Iterable[Mapping[str, Any]],
    *,
    expected_ids: Sequence[str],
    device: torch.device,
    precision: str,
    expected_head_type: str,
    label: str,
    model_loader: Callable[..., tuple[torch.nn.Module, dict[str, Any]]] = load_predictor,
) -> dict[str, Any]:
    """Independently reconstruct one checkpoint and replay the sealed pixels."""

    model, metadata = model_loader(checkpoint, device=device)
    model.eval()
    if metadata.get("fine_semantic_head_type") != expected_head_type:
        raise ValueError(
            f"{label} reconstructed the wrong head type: "
            f"{metadata.get('fine_semantic_head_type')!r}"
        )
    overall = _MetricGroup.create()
    per_city = {city: _MetricGroup.create() for city in ALLOWED_CITIES}
    observed: Counter[str] = Counter()
    try:
        for batch in tqdm(loader, desc=f"independent DC+PHL replay ({label})"):
            image = batch["image"].to(device, non_blocking=True)
            prior = batch["relative_prior"].to(device, non_blocking=True)
            with _autocast(device, precision):
                output = model(image, prior)
            logits = output.get("fine_semantic_logits")
            if (
                logits is None
                or logits.ndim != 4
                or int(logits.shape[1]) != len(CLASS_NAMES)
            ):
                raise ValueError(f"{label} does not emit six-class logits")
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"{label} emitted non-finite logits")
            prediction = logits.argmax(dim=1).cpu().numpy()
            target = batch["fine_class_target"].cpu().numpy()
            classification_valid = batch["classification_valid_mask"].bool()
            image_valid = batch["image_valid_mask"][:, 0].bool()
            valid = (classification_valid & image_valid).cpu().numpy()
            dark = batch["dark_pixel_proxy_mask"].bool().cpu().numpy()
            sample_ids = [str(value) for value in batch["sample_id"]]
            if prediction.shape[0] != len(sample_ids):
                raise ValueError("Batch sample IDs do not align with predictions")
            if prediction.shape != target.shape or prediction.shape != valid.shape or prediction.shape != dark.shape:
                raise ValueError("Prediction, target, validity, and dark proxy shapes differ")
            if np.any(valid & ((target < 0) | (target >= len(CLASS_NAMES)))):
                raise ValueError("A valid classification pixel has an out-of-range target")
            for index, sample_id in enumerate(sample_ids):
                city = _city(sample_id)
                observed[sample_id] += 1
                overall.update(
                    prediction[index], target[index], dark[index], valid[index]
                )
                per_city[city].update(
                    prediction[index], target[index], dark[index], valid[index]
                )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    expected = Counter(str(value) for value in expected_ids)
    if observed != expected:
        missing = sorted((expected - observed).elements())
        extra = sorted((observed - expected).elements())
        raise ValueError(
            "Replay did not evaluate each sealed ID exactly once: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    result = {
        "checkpoint_epoch": metadata.get("epoch"),
        "model_type": metadata.get("model_type"),
        "fine_semantic_head_type": metadata.get("fine_semantic_head_type"),
        "evaluated_sample_count": sum(observed.values()),
        "evaluated_samples_by_city": dict(
            sorted(Counter(_city(value) for value in observed).items())
        ),
        "evaluated_ids_sha256": _canonical_sha256(sorted(observed.elements())),
        "evaluated_ids_by_city_sha256": {
            city: _canonical_sha256(
                sorted(
                    sample_id
                    for sample_id in observed.elements()
                    if _city(sample_id) == city
                )
            )
            for city in ALLOWED_CITIES
        },
        "overall": overall.compute(),
        "by_city": {city: per_city[city].compute() for city in ALLOWED_CITIES},
    }
    validate_replay_result(result)
    return result


def paired_deltas(v3: Mapping[str, Any], v4: Mapping[str, Any]) -> dict[str, Any]:
    def one(v3_metrics: Mapping[str, Any], v4_metrics: Mapping[str, Any]) -> dict[str, Any]:
        v3_six = _mapping(v3_metrics["six_class_identification"], "V3 six-class")
        v4_six = _mapping(v4_metrics["six_class_identification"], "V4 six-class")
        v3_boundary = _mapping(v3_metrics["road_boundary_quality"], "V3 boundary")
        v4_boundary = _mapping(v4_metrics["road_boundary_quality"], "V4 boundary")
        v3_water = _mapping(v3_metrics["water_dark_pixel_proxy"], "V3 water proxy")
        v4_water = _mapping(v4_metrics["water_dark_pixel_proxy"], "V4 water proxy")
        return {
            "six_class_higher_is_better": multiclass_metric_deltas(v4_six, v3_six),
            "road_boundary_f1_delta_higher_is_better": float(v4_boundary["f1"])
            - float(v3_boundary["f1"]),
            "dark_non_water_false_water_rate_delta_lower_is_better": float(
                v4_water["false_water_rate_on_dark_non_water"]
            )
            - float(v3_water["false_water_rate_on_dark_non_water"]),
        }

    return {
        "overall": one(
            _mapping(v3["overall"], "V3 overall"),
            _mapping(v4["overall"], "V4 overall"),
        ),
        "by_city": {
            city: one(
                _mapping(v3["by_city"][city], f"V3 {city}"),
                _mapping(v4["by_city"][city], f"V4 {city}"),
            )
            for city in ALLOWED_CITIES
        },
    }


def _pointer_snapshot(path: Path, expected_sha256: str) -> dict[str, Any]:
    identity = _stable_file_identity(path)
    expected = _required_sha256(expected_sha256, "protected pointer SHA-256")
    if identity["sha256"] != expected:
        raise ValueError("Protected app pointer differs from its sealed hash")
    target_text = path.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    if not target_text:
        raise ValueError("Protected app pointer is blank")
    target = _resolve(target_text)
    return {
        **identity,
        "target": str(target),
        "target_sha256": file_sha256(target),
    }


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "evaluator": Path(__file__).resolve(),
        "gamus_dataset": PROJECT_ROOT / "src/msr/data/gamus_dataset.py",
        "classification_metrics": PROJECT_ROOT
        / "src/msr/evaluation/classification_metrics.py",
        "predictor": PROJECT_ROOT / "src/msr/inference/predict.py",
        "surface_model": PROJECT_ROOT
        / "src/msr/models/domain_surface_net.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help=(
            "Evaluate only the sealed V3 control now. Omit every --v4-* option; "
            "the report retains the same validated global/per-city V3 structure."
        ),
    )
    for suffix, kwargs in (
        ("checkpoint", {"type": Path}),
        ("config", {"type": Path}),
        ("expected-checkpoint-sha256", {}),
        ("expected-config-sha256", {}),
    ):
        parser.add_argument(f"--v3-{suffix}", required=True, **kwargs)
        parser.add_argument(
            f"--v4-{suffix}",
            help="Required unless --baseline-only is selected.",
            **kwargs,
        )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int)
    return parser.parse_args(argv)


def _paired_mode(args: argparse.Namespace) -> bool:
    v4_values = (
        args.v4_checkpoint,
        args.v4_config,
        args.v4_expected_checkpoint_sha256,
        args.v4_expected_config_sha256,
    )
    if args.baseline_only:
        if any(value is not None for value in v4_values):
            raise ValueError("--baseline-only cannot be combined with --v4-* options")
        return False
    if any(value is None for value in v4_values):
        raise ValueError(
            "Paired mode requires all four --v4-* options; use --baseline-only "
            "for an independent V3-only replay"
        )
    return True


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    paired = _paired_mode(args)
    v3_config_path = _resolve(args.v3_config)
    v3 = authenticate_replay_config(
        v3_config_path,
        expected_sha256=args.v3_expected_config_sha256,
        expected_protocol=V3_PROTOCOL,
        expected_head_type="spatial_refined",
    )
    v4_config_path: Path | None = None
    v4: dict[str, Any] | None = None
    if paired:
        assert args.v4_config is not None
        assert args.v4_expected_config_sha256 is not None
        v4_config_path = _resolve(args.v4_config)
        v4 = authenticate_replay_config(
            v4_config_path,
            expected_sha256=args.v4_expected_config_sha256,
            expected_protocol=V4_PROTOCOL,
            expected_head_type="hierarchical_vegetation",
        )
        assert_paired_contracts(v3, v4)

    v3_checkpoint_path = _resolve(args.v3_checkpoint)
    v3_checkpoint = authenticate_checkpoint(
        v3_checkpoint_path,
        expected_sha256=args.v3_expected_checkpoint_sha256,
        external_config=v3["config"],
    )
    v4_checkpoint_path: Path | None = None
    v4_checkpoint: dict[str, Any] | None = None
    if paired:
        assert args.v4_checkpoint is not None
        assert args.v4_expected_checkpoint_sha256 is not None
        assert v4 is not None
        v4_checkpoint_path = _resolve(args.v4_checkpoint)
        v4_checkpoint = authenticate_checkpoint(
            v4_checkpoint_path,
            expected_sha256=args.v4_expected_checkpoint_sha256,
            external_config=v4["config"],
        )

    v3_protocol = _mapping(v3["config"].get("protocol"), "V3 protocol")
    pointer_path = _resolve(str(v3_protocol.get("live_pointer_file", "")))
    v4_protocol: Mapping[str, Any] | None = None
    if paired:
        assert v4 is not None
        v4_protocol = _mapping(v4["config"].get("protocol"), "V4 protocol")
        if pointer_path != _resolve(str(v4_protocol.get("live_pointer_file", ""))):
            raise ValueError("V3 and V4 do not bind the same protected app pointer")
        if v3_protocol.get("live_pointer_file_sha256") != v4_protocol.get(
            "live_pointer_file_sha256"
        ):
            raise ValueError("V3 and V4 pointer hashes differ")
    pointer_before = _pointer_snapshot(
        pointer_path, str(v3_protocol["live_pointer_file_sha256"])
    )
    if pointer_before["target_sha256"] != v3_protocol.get(
        "protected_checkpoint_sha256"
    ):
        raise ValueError("Protected pointer target differs from the frozen checkpoint")
    if paired and pointer_before["target_sha256"] != v4_protocol.get(
        "protected_checkpoint_sha256"
    ):
        raise ValueError("V4 does not bind the frozen protected checkpoint")

    contract = _mapping(v3["validation_contract"], "validation contract")
    dataset_root = Path(str(contract["dataset_root"]))
    prior_root = Path(str(contract["relative_prior_root"]))
    dataset = GamusSurfaceDataset(
        dataset_root,
        "val",
        patch_size=int(contract["validation_patch_size"]),
        random_crop=False,
        augment=False,
        rgb_scale=float(contract["rgb_scale"]),
        height_max_m=float(contract["height_max_m"]),
        radiometric_policy=str(contract["validation_radiometric_policy"]),
        relative_prior_root=prior_root,
        require_relative_prior=True,
        approved_index_path=Path(v3["approved_index_identity"]["path"]),
    )
    validate_dataset_records(
        dataset.records,
        expected_ids=v3["validation_ids"],
        dataset_root=dataset_root,
        prior_root=prior_root,
    )
    output_path = _resolve(args.output)
    protected_paths = {
        pointer_path,
        Path(pointer_before["target"]),
        v3_checkpoint_path,
        v3_config_path,
        Path(v3["approved_index_identity"]["path"]),
    }
    if paired:
        assert v4_checkpoint_path is not None and v4_config_path is not None
        protected_paths.update({v4_checkpoint_path, v4_config_path})
    for record in dataset.records:
        protected_paths.update(
            {
                record.image_path.resolve(),
                record.height_path.resolve(),
                record.class_path.resolve(),
                record.relative_prior_path.resolve()
                if record.relative_prior_path is not None
                else Path(),
            }
        )
    if output_path in protected_paths:
        raise ValueError("Output path may not overwrite any sealed or source artifact")
    implementation_before = _implementation_hashes()
    source_metadata_before = source_metadata_snapshot(dataset.records)
    print("Hashing exact DC+PHL RGB, class, height, and prior sources...")
    source_binding = build_source_binding(dataset.records)

    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(_mapping(v3["config"].get("data"), "V3 data").get("num_workers", 0))
    )
    if workers < 0:
        raise ValueError("num-workers cannot be negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    precision = args.precision if device.type == "cuda" else "fp32"
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        precision = "fp16"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )
    v3_metrics = replay_checkpoint(
        v3_checkpoint_path,
        loader,
        expected_ids=v3["validation_ids"],
        device=device,
        precision=precision,
        expected_head_type="spatial_refined",
        label="paired V3 control" if paired else "V3 control baseline",
    )
    v4_metrics: dict[str, Any] | None = None
    if paired:
        assert v4_checkpoint_path is not None and v4 is not None
        v4_metrics = replay_checkpoint(
            v4_checkpoint_path,
            loader,
            expected_ids=v4["validation_ids"],
            device=device,
            precision=precision,
            expected_head_type="hierarchical_vegetation",
            label="paired V4 candidate",
        )

    pointer_after = _pointer_snapshot(
        pointer_path, str(v3_protocol["live_pointer_file_sha256"])
    )
    if pointer_after != pointer_before:
        raise RuntimeError("Protected app pointer or target changed during replay")
    immutable_after = {
        "v3_checkpoint_sha256": file_sha256(v3_checkpoint_path),
        "v3_config_sha256": file_sha256(v3_config_path),
        "approved_index_sha256": file_sha256(
            Path(v3["approved_index_identity"]["path"])
        ),
    }
    immutable_expected = {
        "v3_checkpoint_sha256": v3_checkpoint["sha256"],
        "v3_config_sha256": v3["config_identity"]["sha256"],
        "approved_index_sha256": v3["approved_index_identity"]["sha256"],
    }
    if paired:
        assert v4_checkpoint_path is not None and v4_config_path is not None
        assert v4_checkpoint is not None and v4 is not None
        immutable_after.update(
            {
                "v4_checkpoint_sha256": file_sha256(v4_checkpoint_path),
                "v4_config_sha256": file_sha256(v4_config_path),
            }
        )
        immutable_expected.update(
            {
                "v4_checkpoint_sha256": v4_checkpoint["sha256"],
                "v4_config_sha256": v4["config_identity"]["sha256"],
            }
        )
    if immutable_after != immutable_expected:
        raise RuntimeError("A sealed checkpoint/config/index changed during replay")
    source_metadata_after = source_metadata_snapshot(dataset.records)
    if source_metadata_after != source_metadata_before:
        raise RuntimeError("A validation source changed during independent replay")
    implementation_after = _implementation_hashes()
    if implementation_after != implementation_before:
        raise RuntimeError("Evaluator implementation changed during replay")

    report = {
        "schema": SCHEMA,
        "mode": "paired_v3_v4" if paired else "baseline_v3_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passes": True,
        "interpretation": (
            "Independent pixel replay on the fixed DC+Philadelphia development "
            "validation partition; this is not NYC, official GAMUS test, or a "
            "genuinely system-unseen geography."
        ),
        "access_policy": {
            "split_constructed": "val",
            "cities_opened": list(ALLOWED_CITIES),
            "nyc_opened": False,
            "official_test_opened_or_reused": False,
            "promotion_performed": False,
        },
        "validation_contract": dict(contract),
        "approved_index": {
            **v3["approved_index_identity"],
            "declared_source_inventory_metadata_sha256": v3[
                "approved_index_declared_source_inventory_metadata_sha256"
            ],
        },
        "source_content_binding": {
            **source_binding,
            "metadata_before": source_metadata_before,
            "metadata_after": source_metadata_after,
            "unchanged_during_replay": True,
        },
        "implementation_sha256": {
            "before": implementation_before,
            "after": implementation_after,
            "unchanged_during_replay": True,
        },
        "runtime": {
            "device": str(device),
            "precision": precision,
            "batch_size": args.batch_size,
            "num_workers": workers,
        },
        "protected_pointer": {
            "before": pointer_before,
            "after": pointer_after,
            "unchanged": True,
        },
        "sealed_artifacts": {
            "v3_config": v3["config_identity"],
            "v3_checkpoint": v3_checkpoint,
            "after_sha256": immutable_after,
            "unchanged": True,
        },
        "v3": v3_metrics,
    }
    if paired:
        assert v4 is not None and v4_checkpoint is not None and v4_metrics is not None
        report["sealed_artifacts"].update(
            {
                "v4_config": v4["config_identity"],
                "v4_checkpoint": v4_checkpoint,
            }
        )
        report["v4"] = v4_metrics
        report["v4_minus_v3"] = paired_deltas(v3_metrics, v4_metrics)
    atomic_write_json(output_path, report)
    summary = {
        "passes": True,
        "mode": report["mode"],
        "output": str(output_path),
        "v3_macro_f1": v3_metrics["overall"]["six_class_identification"]["macro_f1"],
        "source_content_manifest_sha256": source_binding["combined_content_manifest_sha256"],
    }
    if v4_metrics is not None:
        summary["v4_macro_f1"] = v4_metrics["overall"][
            "six_class_identification"
        ]["macro_f1"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
