"""Evaluate the frozen model triplet on corrected legacy validation data.

This runner is deliberately read-only with respect to checkpoints, source data,
and the live application pointer.  It reconstructs every model from the exact
architecture embedded in its checkpoint, authenticates the sealed model choice,
and feeds all three models the same full-scene tensors.  HighBuild's two valid
height contracts are scored independently; OpenCanopy is evaluated once and is
never double-counted in a combined result.

Completed model results are stored as authenticated partial reports.  Re-running
with ``--resume`` skips only partials whose complete run identity still matches,
so a long three-model pass can safely continue after interruption.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile
from typing import Any, Iterable, Mapping
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
import yaml
from tqdm import tqdm

from msr.data.surface_dataset import (
    LANDSCAPE_CLASSES,
    MultiDomainSurfaceDataset,
    SurfaceSampleRecord,
    load_surface_manifest,
)
from msr.evaluation.highbuild_instances import HighBuildInstanceAccumulator
from msr.evaluation.metrics import StreamingRegressionMetrics
from msr.inference.predict import load_predictor
from msr.models.domain_surface_net import DomainGatedSurfaceNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "msr.corrected_legacy_triplet_plan.v1"
REPORT_SCHEMA = "msr.corrected_legacy_triplet.v1"
PARTIAL_SCHEMA = "msr.corrected_legacy_triplet_partial.v1"
SUBSET_REPORT_SCHEMA = "msr.corrected_legacy_triplet.development_subset.v1"
MODEL_NAMES = ("protected", "height_pilot", "expanded_candidate")
METRIC_KEYS = ("pixel_count", "rmse_m", "mae_m", "bias_m", "correlation", "r2")
SHARED_HIGHBUILD_FIELDS = (
    "sample_id",
    "region",
    "landscape",
    "rgb_path",
    "surface_path",
    "target_kind",
    "dtm_path",
    "vegetation_mask_path",
    "relative_prior_path",
    "gsd_m",
)


class CorrectedTripletError(ValueError):
    """Raised when an evaluation contract cannot be authenticated safely."""


def file_sha256(path: str | Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CorrectedTripletError(f"{role} must be a mapping")
    return dict(value)


def _resolve(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _check_file(
    raw: str | Path, expected_sha256: str | None, role: str
) -> dict[str, Any]:
    path = _resolve(raw)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != str(expected_sha256).lower():
        raise CorrectedTripletError(
            f"{role} SHA-256 mismatch: expected {expected_sha256}, found {digest}"
        )
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def _atomic_json(path: Path, payload: Mapping[str, Any], *, replace: bool) -> Path:
    path = path.resolve()
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite evaluation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, str(path))


def load_plan(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    plan = _mapping(value, "triplet plan")
    protocol = _mapping(plan.get("protocol"), "protocol")
    if protocol.get("schema") != PLAN_SCHEMA:
        raise CorrectedTripletError("Unsupported corrected-triplet plan schema")
    if protocol.get("report_schema") != REPORT_SCHEMA:
        raise CorrectedTripletError("Plan does not request the supported report schema")
    if protocol.get("split") != "validation":
        raise CorrectedTripletError("Corrected legacy evaluation must use validation")
    if protocol.get("official_test_used") is not False:
        raise CorrectedTripletError("Official test use is forbidden")
    if protocol.get("promotion_permitted") is not False:
        raise CorrectedTripletError("This evaluation may not promote a model")
    models = _mapping(plan.get("models"), "models")
    if tuple(models) != MODEL_NAMES:
        raise CorrectedTripletError(
            f"Models must appear exactly in the frozen order {MODEL_NAMES}"
        )
    data = _mapping(plan.get("data"), "data")
    highbuild = _mapping(data.get("highbuild_protocols"), "HighBuild protocols")
    if set(highbuild) != {
        "highbuild_all_annotated_positive_v1",
        "highbuild_measured_coco_intersection_v1",
    }:
        raise CorrectedTripletError("Both corrected HighBuild protocols are required")
    if protocol.get("primary_height_protocol") not in highbuild:
        raise CorrectedTripletError("Primary height protocol is not a HighBuild protocol")
    runtime = _mapping(plan.get("runtime"), "runtime")
    if runtime.get("building_probability_source") != "protected_building_logits":
        raise CorrectedTripletError(
            "HighBuild instances must use the protected application building head"
        )
    if data.get("radiometric_policy") != "raw":
        raise CorrectedTripletError("Corrected triplet requires raw RGB")
    if data.get("relative_prior_policy") != "stored_01":
        raise CorrectedTripletError(
            "Corrected triplet requires target-independent stored_01 priors"
        )
    return plan


def _pointer_target(pointer_path: Path) -> Path:
    raw = pointer_path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise CorrectedTripletError("Live application pointer is blank")
    return _resolve(raw)


def _comparison_checkpoint_identity(section: Mapping[str, Any]) -> tuple[str, int | None]:
    checkpoint_hash = section.get("selected_checkpoint_sha256")
    epoch = section.get("selected_epoch")
    if checkpoint_hash is None:
        checkpoint_hash = section.get("checkpoint_sha256")
    if epoch is None:
        epoch = section.get("epoch")
    if not isinstance(checkpoint_hash, str):
        raise CorrectedTripletError("Comparison report lacks a checkpoint identity")
    return checkpoint_hash.lower(), int(epoch) if epoch is not None else None


def authenticate_models(
    plan: Mapping[str, Any], comparison_report: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    authenticated: dict[str, dict[str, Any]] = {}
    for name in MODEL_NAMES:
        config = _mapping(plan["models"][name], f"model {name}")
        identity = _check_file(
            config["checkpoint"], config["checkpoint_sha256"], f"{name} checkpoint"
        )
        payload = torch.load(identity["path"], map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise CorrectedTripletError(f"{name} checkpoint payload is not a mapping")
        actual_epoch = int(payload.get("epoch", -1))
        actual_type = str(payload.get("model_type", ""))
        expected_epoch = int(config["epoch"])
        expected_type = str(config["model_type"])
        if actual_epoch != expected_epoch:
            raise CorrectedTripletError(
                f"{name} epoch mismatch: expected {expected_epoch}, found {actual_epoch}"
            )
        if actual_type != expected_type:
            raise CorrectedTripletError(
                f"{name} model_type mismatch: expected {expected_type}, found {actual_type}"
            )
        embedded_config = _mapping(payload.get("config"), f"{name} embedded config")
        embedded_model = _mapping(
            embedded_config.get("model"), f"{name} embedded model config"
        )
        expected_model = _mapping(
            config.get("expected_model_config"), f"{name} expected model config"
        )
        if embedded_model != expected_model:
            raise CorrectedTripletError(
                f"{name} checkpoint model config differs from the frozen exact config"
            )
        saved_config_identity = None
        if config.get("saved_config") is not None:
            saved_config_identity = _check_file(
                config["saved_config"],
                config.get("saved_config_sha256"),
                f"{name} saved config",
            )
            saved = yaml.safe_load(
                Path(saved_config_identity["path"]).read_text(encoding="utf-8")
            )
            if saved != embedded_config:
                raise CorrectedTripletError(
                    f"{name} checkpoint config differs from its immutable saved config"
                )

        section_name = str(config["comparison_section"])
        section = _mapping(
            comparison_report.get(section_name), f"comparison section {section_name}"
        )
        selected_hash, selected_epoch = _comparison_checkpoint_identity(section)
        if selected_hash != identity["sha256"]:
            raise CorrectedTripletError(
                f"{name} checkpoint is not the sealed comparison checkpoint"
            )
        if selected_epoch is not None and selected_epoch != expected_epoch:
            raise CorrectedTripletError(
                f"{name} selected epoch disagrees with the checkpoint epoch"
            )
        authenticated[name] = {
            **identity,
            "epoch": actual_epoch,
            "model_type": actual_type,
            "model_config": embedded_model,
            "model_config_sha256": canonical_json_sha256(embedded_model),
            "saved_config": saved_config_identity,
            "comparison_section": section_name,
        }
        del payload
    return authenticated


def _record_value(record: SurfaceSampleRecord, field: str) -> Any:
    value = getattr(record, field)
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def _same_file(left: Path | None, right: Path | None, role: str) -> None:
    if left is None or right is None:
        if left != right:
            raise CorrectedTripletError(f"{role} availability differs across protocols")
        return
    if left.resolve() == right.resolve():
        return
    if file_sha256(left) != file_sha256(right):
        raise CorrectedTripletError(f"{role} bytes differ across protocols")


def _contract_protocol(
    *, report_path: Path, manifest_path: Path, split: str, expected: str
) -> dict[str, Any]:
    report = _load_json(report_path)
    if report.get("schema") != "msr.highbuild_annotation_contract.v2":
        raise CorrectedTripletError("Unsupported corrected HighBuild contract")
    if report.get("historical_artifacts_modified") is not False:
        raise CorrectedTripletError("Historical HighBuild artifacts were not preserved")
    if report.get("height_protocol") != expected:
        raise CorrectedTripletError("HighBuild height protocol mismatch")
    split_report = _mapping(
        _mapping(report.get("splits"), "HighBuild contract splits").get(split),
        f"HighBuild contract split {split}",
    )
    if split_report.get("output_manifest_sha256") != file_sha256(manifest_path):
        raise CorrectedTripletError(
            "Corrected HighBuild manifest disagrees with its contract report"
        )
    return report


def authenticate_data(plan: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(plan["data"], "data")
    split = str(plan["protocol"]["split"])
    protocol_records: dict[str, list[SurfaceSampleRecord]] = {}
    protocol_identities: dict[str, Any] = {}
    contracts: dict[str, Any] = {}
    for name, raw in data["highbuild_protocols"].items():
        config = _mapping(raw, f"HighBuild protocol {name}")
        manifest_identity = _check_file(
            config["manifest"], config["manifest_sha256"], f"{name} manifest"
        )
        contract_identity = _check_file(
            config["contract_report"],
            config["contract_report_sha256"],
            f"{name} contract",
        )
        manifest_path = Path(manifest_identity["path"])
        contract = _contract_protocol(
            report_path=Path(contract_identity["path"]),
            manifest_path=manifest_path,
            split=split,
            expected=str(config["height_protocol"]),
        )
        records = load_surface_manifest(manifest_path)
        protocol_records[name] = records
        protocol_identities[name] = {
            "manifest": manifest_identity,
            "contract_report": contract_identity,
            "height_protocol": config["height_protocol"],
        }
        contracts[name] = contract

    names = list(protocol_records)
    inclusive = protocol_records[names[0]]
    other = protocol_records[names[1]]
    left_by_id = {record.sample_id: record for record in inclusive}
    right_by_id = {record.sample_id: record for record in other}
    if list(left_by_id) != list(right_by_id):
        raise CorrectedTripletError(
            "Corrected HighBuild manifests do not have the same ordered sample IDs"
        )
    for sample_id, left in left_by_id.items():
        right = right_by_id[sample_id]
        for field in SHARED_HIGHBUILD_FIELDS:
            if _record_value(left, field) != _record_value(right, field):
                raise CorrectedTripletError(
                    f"{sample_id} differs across HighBuild protocols at {field}"
                )
        _same_file(
            left.building_mask_path,
            right.building_mask_path,
            f"{sample_id} building mask",
        )

    highbuild_ids = [
        record.sample_id
        for record in inclusive
        if record.target_kind == "building_height"
    ]
    if len(highbuild_ids) != int(data["expected_highbuild_samples"]):
        raise CorrectedTripletError("Unexpected corrected HighBuild validation count")
    for records in protocol_records.values():
        if [
            record.sample_id
            for record in records
            if record.target_kind == "building_height"
        ] != highbuild_ids:
            raise CorrectedTripletError("HighBuild protocol sample order differs")

    open_config = _mapping(data.get("open_canopy"), "OpenCanopy config")
    open_identity = _check_file(
        open_config["manifest"],
        open_config["manifest_sha256"],
        "OpenCanopy validation manifest",
    )
    open_records = load_surface_manifest(open_identity["path"])
    if len(open_records) != int(data["expected_open_canopy_samples"]):
        raise CorrectedTripletError("Unexpected OpenCanopy validation count")
    if any(record.landscape != "forest" for record in open_records):
        raise CorrectedTripletError("OpenCanopy validation contains a non-forest record")
    open_by_id = {record.sample_id: record for record in open_records}
    if set(open_by_id) & set(highbuild_ids):
        raise CorrectedTripletError("HighBuild and OpenCanopy sample IDs overlap")
    embedded_open_by_protocol: dict[str, dict[str, SurfaceSampleRecord]] = {}
    for protocol_name, records in protocol_records.items():
        embedded_open = {
            record.sample_id: record
            for record in records
            if record.target_kind != "building_height"
        }
        embedded_open_by_protocol[protocol_name] = embedded_open
        if set(embedded_open) != set(open_by_id):
            raise CorrectedTripletError(
                "Corrected combined manifest does not contain exact OpenCanopy IDs"
            )
        for sample_id, expected in open_by_id.items():
            actual = embedded_open[sample_id]
            for field in SurfaceSampleRecord.__dataclass_fields__:
                # The standalone source manifest intentionally predates cached
                # DAV2 enrichment.  Its label/input identity is authoritative;
                # the corrected combined manifests add the authenticated prior.
                if field == "relative_prior_path":
                    continue
                if _record_value(actual, field) != _record_value(expected, field):
                    raise CorrectedTripletError(
                        f"OpenCanopy record {sample_id} differs at {field}"
                    )

    highbuild_records = [left_by_id[sample_id] for sample_id in highbuild_ids]
    if bool(data.get("require_highbuild_relative_prior")) and any(
        record.relative_prior_path is None for record in highbuild_records
    ):
        raise CorrectedTripletError("A HighBuild cached relative prior is missing")
    if (
        data.get("open_canopy_relative_prior_policy")
        != "require_enriched_cached_stored_01"
    ):
        raise CorrectedTripletError("Unsupported OpenCanopy prior policy")
    enriched_open_records = [
        embedded_open_by_protocol[names[0]][record.sample_id]
        for record in open_records
    ]
    if any(record.relative_prior_path is None for record in enriched_open_records):
        raise CorrectedTripletError(
            "An enriched OpenCanopy cached relative prior is missing"
        )

    source_index = _check_file(
        data["highbuild_source_index"],
        data["highbuild_source_index_sha256"],
        "HighBuild source index",
    )
    return {
        "protocol_records": protocol_records,
        "highbuild_records": highbuild_records,
        "open_canopy_records": enriched_open_records,
        "protocol_identities": protocol_identities,
        "open_canopy_identity": open_identity,
        "open_canopy_protocol": str(open_config["protocol_name"]),
        "source_index": source_index,
        "split_root": str(_resolve(data["highbuild_split_root"])),
        "contracts": contracts,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise CorrectedTripletError(f"CSV has no header: {path}")
        return list(reader)


class CocoShardStore:
    """Read only the COCO members named by the immutable HighBuild index."""

    def __init__(self, *, source_index: Path, split_root: Path, split: str) -> None:
        self.split_root = split_root.resolve()
        self.rows: dict[str, dict[str, str]] = {}
        self.archives: dict[Path, tarfile.TarFile] = {}
        required = {
            "webdataset_key",
            "msr_shard",
            "webdataset_json_member",
        }
        for row in _read_csv(source_index):
            if not required.issubset(row):
                raise CorrectedTripletError("HighBuild source index columns are incomplete")
            sample_id = row["webdataset_key"].strip()
            if not sample_id or sample_id in self.rows:
                raise CorrectedTripletError("HighBuild source index has duplicate IDs")
            declared = row.get("msr_split", "").strip()
            if declared and declared != split:
                raise CorrectedTripletError(
                    f"HighBuild source row {sample_id} is not in {split}"
                )
            self.rows[sample_id] = row

    def __enter__(self) -> "CocoShardStore":
        return self

    def __exit__(self, *_: object) -> None:
        for archive in self.archives.values():
            archive.close()
        self.archives.clear()

    def load(self, sample_id: str) -> dict[str, Any]:
        row = self.rows.get(sample_id)
        if row is None:
            raise CorrectedTripletError(f"COCO provenance is missing for {sample_id}")
        archive_path = Path(row["msr_shard"].strip())
        if not archive_path.is_absolute():
            archive_path = self.split_root / archive_path
        archive_path = archive_path.resolve()
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        archive = self.archives.get(archive_path)
        if archive is None:
            archive = tarfile.open(archive_path, "r")
            self.archives[archive_path] = archive
        member_name = row["webdataset_json_member"].strip()
        member = archive.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"{archive_path}!{member_name}")
        value = json.load(member)
        return _mapping(value, f"COCO annotation {sample_id}")


class MetricScopes:
    """Pooled height metrics over overall, landscape, domain, and region scopes."""

    def __init__(self) -> None:
        self.overall = StreamingRegressionMetrics()
        self.landscapes: dict[str, StreamingRegressionMetrics] = defaultdict(
            StreamingRegressionMetrics
        )
        self.domains: dict[str, StreamingRegressionMetrics] = defaultdict(
            StreamingRegressionMetrics
        )
        self.regions: dict[str, StreamingRegressionMetrics] = defaultdict(
            StreamingRegressionMetrics
        )

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        *,
        landscape: str,
        region: str,
        domain_target: np.ndarray,
    ) -> None:
        self.overall.update(prediction, target, mask)
        self.landscapes[landscape].update(prediction, target, mask)
        self.regions[region].update(prediction, target, mask)
        for name, code in LANDSCAPE_CLASSES.items():
            self.domains[name].update(
                prediction, target, mask & (domain_target == code)
            )

    @staticmethod
    def _compute_nonempty(
        values: Mapping[str, StreamingRegressionMetrics]
    ) -> dict[str, Any]:
        return {
            name: metric.compute()
            for name, metric in sorted(values.items())
            if metric.count
        }

    def compute(self) -> dict[str, Any]:
        if not self.overall.count:
            raise CorrectedTripletError("No valid pixels were accumulated")
        return {
            **self.overall.compute(),
            "landscapes": self._compute_nonempty(self.landscapes),
            "domains": self._compute_nonempty(self.domains),
            "regions": self._compute_nonempty(self.regions),
        }


def metric_at(metrics: Mapping[str, Any], path: str) -> float:
    value: Any = metrics
    for part in path.split("."):
        value = _mapping(value, f"metric parent {path}").get(part)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorrectedTripletError(f"Metric {path} is missing or non-numeric")
    number = float(value)
    if not np.isfinite(number):
        raise CorrectedTripletError(f"Metric {path} is non-finite")
    return number


def compare_candidate(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for path, raw_limit in gates.items():
        candidate_value = metric_at(candidate, str(path))
        reference_value = metric_at(reference, str(path))
        delta = candidate_value - reference_value
        limit = float(raw_limit)
        checks[str(path)] = {
            "candidate": candidate_value,
            "reference": reference_value,
            "candidate_minus_reference": delta,
            "max_regression": limit,
            "passes": bool(delta <= limit),
        }
    return {
        "passes": all(item["passes"] for item in checks.values()),
        "checks": checks,
    }


def _hash_array(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(
        value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    )
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes())


def _assert_shared_highbuild_samples(
    inclusive: Mapping[str, Any], strict: Mapping[str, Any], sample_id: str
) -> None:
    shared = (
        "image",
        "image_valid_mask",
        "building_mask",
        "vegetation_mask",
        "domain_target",
        "domain_valid_mask",
        "relative_prior",
    )
    for field in shared:
        if not torch.equal(inclusive[field], strict[field]):
            raise CorrectedTripletError(
                f"{sample_id} model input/semantic field {field} differs by height protocol"
            )
    inclusive_mask = inclusive["regression_mask"].bool()
    strict_mask = strict["regression_mask"].bool()
    if torch.any(strict_mask & ~inclusive_mask):
        raise CorrectedTripletError(
            f"{sample_id} strict height support is not a subset of inclusive support"
        )
    if not torch.equal(
        inclusive["height"][strict_mask], strict["height"][strict_mask]
    ):
        raise CorrectedTripletError(
            f"{sample_id} target values differ on shared strict support"
        )


def _read_raster_band(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            if source.count != 1:
                raise CorrectedTripletError(f"Expected one band: {path}")
            return source.read(1), {
                "height": source.height,
                "width": source.width,
                "crs": source.crs,
                "transform": source.transform,
            }


def _require_native_shape(record: SurfaceSampleRecord, patch_size: int) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(record.rgb_path) as source:
            if source.height != patch_size or source.width != patch_size:
                raise CorrectedTripletError(
                    f"{record.sample_id} is {source.height}x{source.width}; expected "
                    f"the full-scene {patch_size}x{patch_size} grid"
                )
            return {
                "height": source.height,
                "width": source.width,
                "crs": source.crs,
                "transform": source.transform,
            }


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise CorrectedTripletError(f"{precision} evaluation requires CUDA")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    if dtype is None:
        raise CorrectedTripletError("precision must be fp32, bf16, or fp16")
    return torch.autocast(device_type="cuda", dtype=dtype)


def _dataset(
    records: list[SurfaceSampleRecord], *, patch_size: int, data: Mapping[str, Any]
) -> MultiDomainSurfaceDataset:
    return MultiDomainSurfaceDataset(
        records,
        patch_size=patch_size,
        random_crop=False,
        augment=False,
        rgb_scale=float(data["rgb_scale"]),
        height_max_m=float(data["height_max_m"]),
        building_threshold_m=float(data["building_threshold_m"]),
        radiometric_policy=str(data["radiometric_policy"]),
        relative_prior_policy=str(data["relative_prior_policy"]),
    )


def _validate_loaded_model(
    model: torch.nn.Module,
    metadata: Mapping[str, Any],
    authenticated: Mapping[str, Any],
) -> None:
    if metadata.get("model_type") != authenticated["model_type"]:
        raise CorrectedTripletError("Loaded model_type differs from authenticated payload")
    if int(metadata.get("epoch", -1)) != int(authenticated["epoch"]):
        raise CorrectedTripletError("Loaded model epoch differs from authenticated payload")
    if not isinstance(model, DomainGatedSurfaceNet):
        raise CorrectedTripletError("Every corrected-triplet model must be domain-gated")
    expected = authenticated["model_config"]
    attributes = (
        "fusion_mode",
        "building_protection_power",
        "vegetation_fusion_temperature",
        "vegetation_expert_fusion_threshold",
        "vegetation_expert_fusion_strength",
    )
    for attribute in attributes:
        expected_value = expected.get(attribute)
        actual_value = getattr(model, attribute)
        if actual_value != expected_value:
            raise CorrectedTripletError(
                f"Loaded {attribute}={actual_value!r}, expected {expected_value!r}"
            )
    expected_classes = int(expected.get("fine_semantic_classes", 0))
    actual_classes = 0 if model.fine_semantic_head is None else model.fine_semantic_classes
    if actual_classes != expected_classes:
        raise CorrectedTripletError(
            f"Loaded fine-semantic head has {actual_classes} classes; "
            f"expected {expected_classes}"
        )


def _native_arrays_for_instances(
    inclusive_record: SurfaceSampleRecord,
    strict_record: SurfaceSampleRecord,
) -> dict[str, Any]:
    if inclusive_record.building_mask_path is None or strict_record.building_mask_path is None:
        raise CorrectedTripletError("Corrected HighBuild building mask is missing")
    if inclusive_record.valid_mask_path is None or strict_record.valid_mask_path is None:
        raise CorrectedTripletError("Corrected HighBuild validity mask is missing")
    target, profile = _read_raster_band(inclusive_record.surface_path)
    inclusive_building, building_profile = _read_raster_band(
        inclusive_record.building_mask_path
    )
    strict_building, strict_building_profile = _read_raster_band(
        strict_record.building_mask_path
    )
    inclusive_valid, inclusive_valid_profile = _read_raster_band(
        inclusive_record.valid_mask_path
    )
    strict_valid, strict_valid_profile = _read_raster_band(strict_record.valid_mask_path)
    expected_shape = (profile["height"], profile["width"])
    for role, values, other_profile in (
        ("inclusive building", inclusive_building, building_profile),
        ("strict building", strict_building, strict_building_profile),
        ("inclusive validity", inclusive_valid, inclusive_valid_profile),
        ("strict validity", strict_valid, strict_valid_profile),
    ):
        if values.shape != expected_shape or (
            other_profile["crs"] != profile["crs"]
            or not np.allclose(
                tuple(other_profile["transform"]),
                tuple(profile["transform"]),
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise CorrectedTripletError(
                f"{inclusive_record.sample_id} {role} grid differs from reference"
            )
    if not np.array_equal(inclusive_building > 0, strict_building > 0):
        raise CorrectedTripletError("Corrected HighBuild building masks differ")
    if np.any((strict_valid > 0) & ~(inclusive_valid > 0)):
        raise CorrectedTripletError("Strict HighBuild mask is not inclusive")
    return {
        "target": target.astype(np.float32, copy=False),
        "building": inclusive_building > 0,
        "valid": {
            "highbuild_all_annotated_positive_v1": inclusive_valid > 0,
            "highbuild_measured_coco_intersection_v1": strict_valid > 0,
        },
        "profile": profile,
    }


def evaluate_one_model(
    *,
    model_name: str,
    authenticated_model: Mapping[str, Any],
    data_auth: Mapping[str, Any],
    plan: Mapping[str, Any],
    device: torch.device,
    precision: str,
    highbuild_limit: int | None,
    open_canopy_limit: int | None,
) -> dict[str, Any]:
    data = _mapping(plan["data"], "data")
    runtime = _mapping(plan["runtime"], "runtime")
    highbuild_records = list(data_auth["highbuild_records"])
    open_records = list(data_auth["open_canopy_records"])
    if highbuild_limit is not None:
        highbuild_records = highbuild_records[:highbuild_limit]
    if open_canopy_limit is not None:
        open_records = open_records[:open_canopy_limit]
    protocol_names = tuple(data_auth["protocol_records"])
    strict_name = "highbuild_measured_coco_intersection_v1"
    inclusive_name = "highbuild_all_annotated_positive_v1"
    protocol_maps = {
        name: {
            record.sample_id: record
            for record in data_auth["protocol_records"][name]
            if record.target_kind == "building_height"
        }
        for name in protocol_names
    }
    inclusive_records = [
        protocol_maps[inclusive_name][record.sample_id] for record in highbuild_records
    ]
    strict_records = [
        protocol_maps[strict_name][record.sample_id] for record in highbuild_records
    ]
    highbuild_patch = int(data["highbuild_patch_size"])
    open_patch = int(data["open_canopy_patch_size"])
    for record in inclusive_records:
        _require_native_shape(record, highbuild_patch)
    for record in open_records:
        _require_native_shape(record, open_patch)
    inclusive_dataset = _dataset(
        inclusive_records, patch_size=highbuild_patch, data=data
    )
    strict_dataset = _dataset(strict_records, patch_size=highbuild_patch, data=data)
    open_dataset = _dataset(open_records, patch_size=open_patch, data=data)

    model, metadata = load_predictor(authenticated_model["path"], device=device)
    _validate_loaded_model(model, metadata, authenticated_model)
    model.eval()
    books = {
        name: {
            "combined": MetricScopes(),
            "highbuild": MetricScopes(),
            "open_canopy": MetricScopes(),
        }
        for name in protocol_names
    }
    instances = {
        name: HighBuildInstanceAccumulator(
            height_protocol=str(data_auth["protocol_identities"][name]["height_protocol"])
        )
        for name in protocol_names
    }
    model_input_digest = hashlib.sha256()
    supervision_digest = hashlib.sha256()
    coco_digest = hashlib.sha256()
    scene_order: list[str] = []
    instance_thresholds = {
        "building_probability_threshold": float(
            runtime["building_probability_threshold"]
        ),
        "instance_iou_threshold": float(runtime["instance_iou_threshold"]),
        "minimum_predicted_instance_pixels": int(
            runtime["minimum_predicted_instance_pixels"]
        ),
        "minimum_height_pixels": int(runtime["minimum_height_pixels"]),
        "boundary_tolerance_pixels": int(runtime["boundary_tolerance_pixels"]),
    }

    with CocoShardStore(
        source_index=Path(data_auth["source_index"]["path"]),
        split_root=Path(data_auth["split_root"]),
        split=str(plan["protocol"]["split"]),
    ) as coco_store, torch.inference_mode():
        progress = tqdm(
            range(len(inclusive_dataset)),
            desc=f"{model_name}/HighBuild",
            unit="scene",
        )
        for index in progress:
            inclusive = inclusive_dataset[index]
            strict = strict_dataset[index]
            record = inclusive_records[index]
            strict_record = strict_records[index]
            _assert_shared_highbuild_samples(inclusive, strict, record.sample_id)
            scene_order.append(record.sample_id)
            model_input_digest.update(record.sample_id.encode("utf-8") + b"\0")
            for field in ("image", "relative_prior", "image_valid_mask"):
                _hash_array(model_input_digest, field, inclusive[field])
            for field, value in (
                ("inclusive_target", inclusive["height"]),
                ("inclusive_regression", inclusive["regression_mask"]),
                ("strict_regression", strict["regression_mask"]),
                ("domain_target", inclusive["domain_target"]),
                ("domain_valid", inclusive["domain_valid_mask"]),
            ):
                _hash_array(supervision_digest, field, value)
            image = inclusive["image"].unsqueeze(0).to(device, non_blocking=True)
            prior = inclusive["relative_prior"].unsqueeze(0).to(
                device, non_blocking=True
            )
            with _autocast(device, precision):
                output = model(image, prior)
            prediction = output["height"][0, 0].float().cpu().numpy()
            building_probability = (
                output["protected_building_logits"][0, 0]
                .sigmoid()
                .float()
                .cpu()
                .numpy()
            )
            target = inclusive["height"][0].numpy()
            domain_target = inclusive["domain_target"].numpy()
            masks = {
                inclusive_name: inclusive["regression_mask"][0].numpy(),
                strict_name: strict["regression_mask"][0].numpy(),
            }
            for protocol_name, mask in masks.items():
                for scope in ("combined", "highbuild"):
                    books[protocol_name][scope].update(
                        prediction,
                        target,
                        mask,
                        landscape=record.landscape,
                        region=record.region,
                        domain_target=domain_target,
                    )

            native = _native_arrays_for_instances(record, strict_record)
            height = int(native["profile"]["height"])
            width = int(native["profile"]["width"])
            image_valid = inclusive["image_valid_mask"][0, :height, :width].numpy()
            coco = coco_store.load(record.sample_id)
            coco_digest.update(record.sample_id.encode("utf-8") + b"\0")
            coco_digest.update(
                json.dumps(coco, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            for protocol_name in protocol_names:
                instances[protocol_name].update(
                    sample_id=record.sample_id,
                    prediction_height_m=prediction[:height, :width],
                    prediction_building=building_probability[:height, :width],
                    coco=coco,
                    reference_height_m=native["target"],
                    building_footprints=native["building"],
                    height_valid_mask=native["valid"][protocol_name],
                    image_valid_mask=image_valid,
                    height_protocol=str(
                        data_auth["protocol_identities"][protocol_name][
                            "height_protocol"
                        ]
                    ),
                    classification_valid_mask=None,
                    prediction_is_instance_labels=False,
                    crs=native["profile"]["crs"],
                    transform=native["profile"]["transform"],
                    **instance_thresholds,
                )
            del image, prior, output

        progress = tqdm(
            range(len(open_dataset)),
            desc=f"{model_name}/OpenCanopy",
            unit="scene",
        )
        for index in progress:
            sample = open_dataset[index]
            record = open_records[index]
            scene_order.append(record.sample_id)
            model_input_digest.update(record.sample_id.encode("utf-8") + b"\0")
            for field in ("image", "relative_prior", "image_valid_mask"):
                _hash_array(model_input_digest, field, sample[field])
            for field, value in (
                ("open_target", sample["height"]),
                ("open_regression", sample["regression_mask"]),
                ("open_domain_target", sample["domain_target"]),
                ("open_domain_valid", sample["domain_valid_mask"]),
            ):
                _hash_array(supervision_digest, field, value)
            image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
            prior = sample["relative_prior"].unsqueeze(0).to(
                device, non_blocking=True
            )
            with _autocast(device, precision):
                output = model(image, prior)
            prediction = output["height"][0, 0].float().cpu().numpy()
            target = sample["height"][0].numpy()
            mask = sample["regression_mask"][0].numpy()
            domain_target = sample["domain_target"].numpy()
            for protocol_name in protocol_names:
                for scope in ("combined", "open_canopy"):
                    books[protocol_name][scope].update(
                        prediction,
                        target,
                        mask,
                        landscape=record.landscape,
                        region=record.region,
                        domain_target=domain_target,
                    )
            del image, prior, output

    results: dict[str, Any] = {}
    for protocol_name in protocol_names:
        results[protocol_name] = {
            "height_protocol": data_auth["protocol_identities"][protocol_name][
                "height_protocol"
            ],
            "combined": books[protocol_name]["combined"].compute(),
            "highbuild": books[protocol_name]["highbuild"].compute(),
            "open_canopy": books[protocol_name]["open_canopy"].compute(),
            "highbuild_instances": instances[protocol_name].compute(),
        }
    if results[protocol_names[0]]["open_canopy"] != results[protocol_names[1]][
        "open_canopy"
    ]:
        raise RuntimeError("OpenCanopy metrics changed between HighBuild protocol views")
    return {
        "schema": PARTIAL_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "checkpoint": dict(authenticated_model),
        "loaded_model": {
            "model_type": metadata["model_type"],
            "epoch": metadata["epoch"],
            "fusion_mode": model.fusion_mode,
            "fine_semantic_classes": 0
            if model.fine_semantic_head is None
            else model.fine_semantic_classes,
            "building_probability_source": runtime["building_probability_source"],
        },
        "sample_counts": {
            "highbuild": len(highbuild_records),
            "open_canopy": len(open_records),
        },
        "input_identity": {
            "ordered_sample_ids_sha256": canonical_json_sha256(scene_order),
            "model_input_tensors_sha256": model_input_digest.hexdigest(),
            "supervision_tensors_sha256": supervision_digest.hexdigest(),
            "coco_annotations_sha256": coco_digest.hexdigest(),
        },
        "protocols": results,
        "open_canopy_forest_metrics": results[protocol_names[0]]["open_canopy"],
        "runtime": {
            "device": str(device),
            "precision": precision,
            "highbuild_patch_size": highbuild_patch,
            "open_canopy_patch_size": open_patch,
            "radiometric_policy": data["radiometric_policy"],
            "relative_prior_policy": data["relative_prior_policy"],
            "open_canopy_relative_prior": "enriched_cached_stored_01",
        },
    }


def _artifact_paths(
    *,
    plan_path: Path,
    plan: Mapping[str, Any],
    model_auth: Mapping[str, Any],
    data_auth: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {
        "plan": _check_file(plan_path, None, "plan"),
        "comparison_report": _check_file(
            plan["protocol"]["comparison_report"],
            plan["protocol"]["comparison_report_sha256"],
            "GAMUS comparison report",
        ),
        "highbuild_source_index": dict(data_auth["source_index"]),
        "open_canopy_manifest": dict(data_auth["open_canopy_identity"]),
    }
    for name, identity in model_auth.items():
        artifacts[f"checkpoint:{name}"] = {
            key: identity[key] for key in ("path", "sha256", "size_bytes")
        }
        if identity.get("saved_config"):
            artifacts[f"saved_config:{name}"] = dict(identity["saved_config"])
    for name, identities in data_auth["protocol_identities"].items():
        artifacts[f"manifest:{name}"] = dict(identities["manifest"])
        artifacts[f"contract:{name}"] = dict(identities["contract_report"])
    return artifacts


def _rehash_artifacts(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, identity in artifacts.items():
        path = Path(str(identity["path"]))
        result[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def _pointer_identity(plan: Mapping[str, Any]) -> dict[str, Any]:
    pointer = _resolve(plan["protocol"]["live_pointer"])
    pointer_identity = _check_file(
        pointer,
        plan["protocol"]["live_pointer_sha256"],
        "live application pointer",
    )
    target = _pointer_target(pointer)
    target_identity = _check_file(target, None, "live application checkpoint")
    protected = plan["models"]["protected"]
    if target.resolve() != _resolve(protected["checkpoint"]):
        raise CorrectedTripletError(
            "Live pointer does not resolve to the frozen protected checkpoint"
        )
    if target_identity["sha256"] != protected["checkpoint_sha256"]:
        raise CorrectedTripletError("Live target is not the protected checkpoint bytes")
    return {"pointer": pointer_identity, "target": target_identity}


def _partial_identity(
    *,
    plan_sha256: str,
    model_name: str,
    checkpoint_sha256: str,
    device: str,
    precision: str,
    highbuild_limit: int | None,
    open_canopy_limit: int | None,
) -> dict[str, Any]:
    return {
        "plan_sha256": plan_sha256,
        "model_name": model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "device": device,
        "precision": precision,
        "highbuild_limit": highbuild_limit,
        "open_canopy_limit": open_canopy_limit,
    }


def _load_reusable_partial(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    partial = _load_json(path)
    if partial.get("schema") != PARTIAL_SCHEMA:
        raise CorrectedTripletError(f"Unsupported partial report: {path}")
    if partial.get("run_identity") != expected:
        raise CorrectedTripletError(
            f"Partial report run identity differs; do not reuse {path}"
        )
    return partial


def build_report(
    *,
    plan: Mapping[str, Any],
    plan_identity: Mapping[str, Any],
    model_auth: Mapping[str, Any],
    data_auth: Mapping[str, Any],
    partials: Mapping[str, Mapping[str, Any]],
    artifacts_before: Mapping[str, Any],
    artifacts_after: Mapping[str, Any],
    pointer_before: Mapping[str, Any],
    pointer_after: Mapping[str, Any],
    complete_manifest: bool,
) -> dict[str, Any]:
    input_identities = [partials[name]["input_identity"] for name in MODEL_NAMES]
    if any(identity != input_identities[0] for identity in input_identities[1:]):
        raise RuntimeError("Models did not consume identical inputs and supervision")
    primary = str(plan["protocol"]["primary_height_protocol"])
    models: dict[str, Any] = {}
    for name in MODEL_NAMES:
        partial = partials[name]
        models[name] = {
            "checkpoint": partial["checkpoint"],
            "loaded_model": partial["loaded_model"],
            "metrics": partial["protocols"][primary]["combined"],
            "protocol_metrics": partial["protocols"],
            "open_canopy_forest_metrics": partial["open_canopy_forest_metrics"],
        }

    gates = _mapping(plan["gates"], "gates")
    comparisons: dict[str, Any] = {}
    all_pass = True
    for protocol_name in data_auth["protocol_records"]:
        protocol_result: dict[str, Any] = {}
        for reference_name, gate_name in (
            ("protected", "candidate_vs_protected_max_regression"),
            ("height_pilot", "candidate_vs_height_pilot_max_regression"),
        ):
            result = compare_candidate(
                partials["expanded_candidate"]["protocols"][protocol_name][
                    "combined"
                ],
                partials[reference_name]["protocols"][protocol_name]["combined"],
                _mapping(gates[gate_name], gate_name),
            )
            protocol_result[f"expanded_vs_{reference_name}"] = result
            all_pass &= bool(result["passes"])
        comparisons[protocol_name] = protocol_result

    protocols = [
        *data_auth["protocol_records"].keys(),
        data_auth["open_canopy_protocol"],
    ]
    return {
        "schema": REPORT_SCHEMA if complete_manifest else SUBSET_REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if complete_manifest and all_pass else (
            "failed" if complete_manifest else "development_subset"
        ),
        "complete_manifest": complete_manifest,
        "identical_protocol_for_all_models": True,
        "identical_input_identity": input_identities[0],
        "official_test_used": False,
        "split": "validation",
        "protocols": protocols,
        "primary_height_protocol": primary,
        "primary_metric_interpretation": (
            "Strict measured HighBuild pixels pooled once with OpenCanopy; the inclusive "
            "HighBuild protocol is independently scored and independently gated below."
        ),
        "evaluation_contract": {
            "highbuild_grid": "full native 1024x1024 scenes",
            "open_canopy_grid": "full native 384x384 scenes",
            "rgb": "raw",
            "relative_prior": "target-independent cached stored_01 for both suites",
            "outside_highbuild_coco_footprints": "unknown_not_ground",
            "highbuild_classification_support": "positive_annotations_only",
            "highbuild_detection_precision_and_count": "unavailable",
            "open_canopy_role": "corrected forest validation",
        },
        "plan": dict(plan_identity),
        "authenticated_artifacts_before": dict(artifacts_before),
        "authenticated_artifacts_after": dict(artifacts_after),
        "read_only_artifacts_unchanged": artifacts_before == artifacts_after,
        "live_application_before": dict(pointer_before),
        "live_application_after": dict(pointer_after),
        "live_application_pointer_changed": False,
        "models": models,
        "per_protocol_comparisons": comparisons,
        "passes_predeclared_gates": bool(complete_manifest and all_pass),
        "promotion_performed": False,
        "active_model_changed": False,
        "decision": "evaluation evidence only; explicit human review remains required",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=PROJECT_ROOT / "configs/corrected_legacy_triplet_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--model", action="append", choices=MODEL_NAMES)
    parser.add_argument("--max-highbuild-samples", type=int)
    parser.add_argument("--max-open-canopy-samples", type=int)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path | None:
    plan_path = args.plan.resolve()
    plan = load_plan(plan_path)
    plan_identity = _check_file(plan_path, None, "triplet plan")
    protocol = plan["protocol"]
    comparison_identity = _check_file(
        protocol["comparison_report"],
        protocol["comparison_report_sha256"],
        "GAMUS comparison report",
    )
    comparison_report = _load_json(Path(comparison_identity["path"]))
    if comparison_report.get("official_gamus_test_constructed") is not False:
        raise CorrectedTripletError(
            "Authenticated GAMUS comparison does not certify test exclusion"
        )
    model_auth = authenticate_models(plan, comparison_report)
    data_auth = authenticate_data(plan)
    pointer_before = _pointer_identity(plan)
    artifacts_before = _artifact_paths(
        plan_path=plan_path,
        plan=plan,
        model_auth=model_auth,
        data_auth=data_auth,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "official_test_used": False,
                    "models": {
                        name: {
                            "sha256": identity["sha256"],
                            "epoch": identity["epoch"],
                            "model_type": identity["model_type"],
                            "fusion_mode": identity["model_config"]["fusion_mode"],
                            "fine_semantic_classes": identity["model_config"].get(
                                "fine_semantic_classes", 0
                            ),
                        }
                        for name, identity in model_auth.items()
                    },
                    "sample_counts": {
                        "highbuild": len(data_auth["highbuild_records"]),
                        "open_canopy": len(data_auth["open_canopy_records"]),
                    },
                    "live_application": pointer_before,
                },
                indent=2,
            )
        )
        return None

    for name, value in (
        ("max-highbuild-samples", args.max_highbuild_samples),
        ("max-open-canopy-samples", args.max_open_canopy_samples),
    ):
        if value is not None and value <= 0:
            raise CorrectedTripletError(f"{name} must be positive")
    complete_manifest = (
        args.max_highbuild_samples is None and args.max_open_canopy_samples is None
    )
    runtime = plan["runtime"]
    device = torch.device(args.device or runtime["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CorrectedTripletError("CUDA was requested but is unavailable")
    precision = str(args.precision or runtime["precision"])
    output = (args.output.resolve() if args.output else _resolve(protocol["output"]))
    work_dir = (
        args.work_dir.resolve() if args.work_dir else _resolve(protocol["work_dir"])
    )
    protected_paths = {
        Path(identity["path"]).resolve()
        for identity in artifacts_before.values()
    }
    protected_paths.add(Path(pointer_before["pointer"]["path"]).resolve())
    protected_paths.add(Path(pointer_before["target"]["path"]).resolve())
    if output in protected_paths or work_dir in protected_paths:
        raise CorrectedTripletError("Output cannot overwrite a protected input artifact")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite final triplet report: {output}")
    work_dir.mkdir(parents=True, exist_ok=True)
    selected_models = tuple(args.model or MODEL_NAMES)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False

    partials: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        partial_path = work_dir / f"{model_name}.json"
        run_identity = _partial_identity(
            plan_sha256=plan_identity["sha256"],
            model_name=model_name,
            checkpoint_sha256=model_auth[model_name]["sha256"],
            device=str(device),
            precision=precision,
            highbuild_limit=args.max_highbuild_samples,
            open_canopy_limit=args.max_open_canopy_samples,
        )
        if partial_path.is_file():
            if not args.resume:
                raise FileExistsError(
                    f"Partial exists; pass --resume to authenticate and reuse it: {partial_path}"
                )
            partials[model_name] = _load_reusable_partial(partial_path, run_identity)
            print(f"Reused authenticated partial: {partial_path}")
            continue
        if model_name not in selected_models:
            continue
        partial = evaluate_one_model(
            model_name=model_name,
            authenticated_model=model_auth[model_name],
            data_auth=data_auth,
            plan=plan,
            device=device,
            precision=precision,
            highbuild_limit=args.max_highbuild_samples,
            open_canopy_limit=args.max_open_canopy_samples,
        )
        partial["run_identity"] = run_identity
        _atomic_json(partial_path, partial, replace=False)
        partials[model_name] = partial
        print(f"Saved resumable model partial: {partial_path}")
        del partial
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if set(partials) != set(MODEL_NAMES):
        missing = [name for name in MODEL_NAMES if name not in partials]
        print(
            "Model partials remain pending: "
            + ", ".join(missing)
            + ". Re-run with --resume to continue."
        )
        return None

    pointer_after = _pointer_identity(plan)
    artifacts_after = _rehash_artifacts(artifacts_before)
    if artifacts_before != artifacts_after:
        raise RuntimeError("An immutable input artifact changed during evaluation")
    if pointer_before != pointer_after:
        raise RuntimeError("Live application pointer or target changed during evaluation")
    report = build_report(
        plan=plan,
        plan_identity=plan_identity,
        model_auth=model_auth,
        data_auth=data_auth,
        partials=partials,
        artifacts_before=artifacts_before,
        artifacts_after=artifacts_after,
        pointer_before=pointer_before,
        pointer_after=pointer_after,
        complete_manifest=complete_manifest,
    )
    saved = _atomic_json(output, report, replace=False)
    print(
        json.dumps(
            {
                "output": str(saved),
                "schema": report["schema"],
                "status": report["status"],
                "passes_predeclared_gates": report["passes_predeclared_gates"],
                "official_test_used": False,
                "live_application_pointer_changed": False,
            },
            indent=2,
        )
    )
    return saved


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
