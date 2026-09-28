"""Train the isolated two-endpoint scene router from audited utility records.

This command does not load or modify the application checkpoint pointer.  Its
input JSONL must contain detached scene descriptors plus fallback/candidate
utilities previously measured against held-out labels. In the default mode,
targets are recomputed from utility. The explicit ``source_gamus`` mode derives
source-domain supervision from authenticated record metadata while still using
only the descriptor as model input. Any stored target field is ignored.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msr.models.routed_surface import ConservativeSceneRouter
from msr.training.scene_router import (
    BalancedSourceLandscapeSampler,
    SceneRouterDataset,
    SceneRouterRecord,
    SceneRouterTrainingConfig,
    compute_scene_router_metrics,
    read_scene_router_records,
    train_scene_router,
)
from msr.training.scene_router_record_store import (
    RECORD_STORE_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    validate_record_selection_provenance,
)


TRAINING_CONFIG_FIELDS = {
    "records",
    "output_dir",
    "record_provenance",
    "record_completion",
    "epochs",
    "batch_size",
    "hidden_features",
    "learning_rate",
    "weight_decay",
    "minimum_candidate_gain",
    "fallback_weight",
    "candidate_weight",
    "brier_weight",
    "minimum_threshold",
    "minimum_precision",
    "maximum_mean_utility_regression",
    "minimum_candidate_selections",
    "seed",
    "device",
    "target_mode",
    "minimum_gamus_coverage",
    "minimum_gamus_selections",
}

TARGET_MODES = {"utility", "source_gamus"}


class SourceGamusRouterDataset(Dataset[dict[str, object]]):
    """Derive a source-domain label while exposing descriptors as model input."""

    def __init__(self, records: Sequence[SceneRouterRecord]) -> None:
        if not records:
            raise ValueError("source_gamus router dataset requires records")
        if {record.source for record in records} - {"gamus", "legacy"}:
            raise ValueError("source_gamus labels support only GAMUS and legacy records")
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        return {
            # This is the only tensor supplied to the router network.
            "descriptor": record.descriptor.clone(),
            "candidate_target": torch.tensor(
                [float(record.source == "gamus")], dtype=torch.float32
            ),
            # Endpoint utilities are never labels in this mode. They are kept
            # solely for the held-out no-regression threshold guard.
            "fallback_utility": torch.tensor(
                [record.fallback_utility], dtype=torch.float32
            ),
            "candidate_utility": torch.tensor(
                [record.candidate_utility], dtype=torch.float32
            ),
            "candidate_gain": torch.tensor(
                [record.fallback_utility - record.candidate_utility],
                dtype=torch.float32,
            ),
            "sample_id": record.sample_id,
            "source": record.source,
            "landscape": record.landscape,
            "partition": record.partition,
        }


class BalancedSourceSampler(Sampler[int]):
    """Deterministically give GAMUS and legacy equal training exposure."""

    def __init__(self, records: Sequence[SceneRouterRecord], *, seed: int) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            groups[record.source].append(index)
        if set(groups) != {"gamus", "legacy"}:
            raise ValueError("balanced source routing requires GAMUS and legacy records")
        self.groups = {name: tuple(groups[name]) for name in sorted(groups)}
        self.samples_per_source = max(len(indices) for indices in self.groups.values())
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.samples_per_source * len(self.groups)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draws: list[int] = []
        for population in self.groups.values():
            source_draws: list[int] = []
            while len(source_draws) < self.samples_per_source:
                order = torch.randperm(len(population), generator=generator).tolist()
                source_draws.extend(population[position] for position in order)
            draws.extend(source_draws[: self.samples_per_source])
        order = torch.randperm(len(draws), generator=generator).tolist()
        return iter(draws[position] for position in order)


@dataclass(frozen=True)
class SourceGamusThresholdResult:
    threshold: float
    eligible: bool
    reason: str
    metrics: dict[str, float | int]
    evaluated_thresholds: int


def _config_defaults(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or not isinstance(
        value.get("router_training"), Mapping
    ):
        raise ValueError("router training config must contain router_training mapping")
    section = dict(value["router_training"])
    unknown = sorted(set(section) - TRAINING_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"unknown router training config fields: {unknown}")
    for key in ("records", "output_dir", "record_provenance", "record_completion"):
        if key in section and section[key] is not None:
            candidate = Path(str(section[key])).expanduser()
            section[key] = (
                candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
            ).resolve()
    return section


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    known, _ = pre_parser.parse_known_args(argv)
    defaults = _config_defaults(known.config) if known.config is not None else {}
    parser = argparse.ArgumentParser(
        description="Train only the conservative offline scene router.",
        parents=[pre_parser],
    )
    parser.add_argument("--records", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--record-provenance", type=Path)
    parser.add_argument("--record-completion", type=Path)
    parser.add_argument(
        "--allow-unprovenanced-records",
        action="store_true",
        help="Development-only escape hatch; artifacts remain ineligible for Stage-3.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-features", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--minimum-candidate-gain", type=float, default=0.02)
    parser.add_argument("--fallback-weight", type=float, default=2.0)
    parser.add_argument("--candidate-weight", type=float, default=1.0)
    parser.add_argument("--brier-weight", type=float, default=0.05)
    parser.add_argument("--minimum-threshold", type=float, default=0.75)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument(
        "--maximum-mean-utility-regression", type=float, default=0.0
    )
    parser.add_argument("--minimum-candidate-selections", type=int, default=1)
    parser.add_argument(
        "--target-mode", choices=sorted(TARGET_MODES), default="utility"
    )
    parser.add_argument("--minimum-gamus-coverage", type=float, default=0.0)
    parser.add_argument("--minimum-gamus-selections", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.target_mode not in TARGET_MODES:
        parser.error(f"--target-mode must be one of {sorted(TARGET_MODES)}")
    numeric_types = {
        "epochs": int,
        "batch_size": int,
        "hidden_features": int,
        "learning_rate": float,
        "weight_decay": float,
        "minimum_candidate_gain": float,
        "fallback_weight": float,
        "candidate_weight": float,
        "brier_weight": float,
        "minimum_threshold": float,
        "minimum_precision": float,
        "maximum_mean_utility_regression": float,
        "minimum_candidate_selections": int,
        "minimum_gamus_coverage": float,
        "minimum_gamus_selections": int,
        "seed": int,
    }
    for name, converter in numeric_types.items():
        value = getattr(args, name)
        if isinstance(value, bool):
            parser.error(f"--{name.replace('_', '-')} cannot be boolean")
        if converter is int and isinstance(value, float) and not value.is_integer():
            parser.error(f"--{name.replace('_', '-')} must be an integer")
        try:
            setattr(args, name, converter(value))
        except (TypeError, ValueError):
            parser.error(f"--{name.replace('_', '-')} has an invalid value")
    if args.records is None or args.output_dir is None:
        parser.error("--records and --output-dir are required (directly or by config)")
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_features <= 0:
        parser.error("epochs, batch size, and hidden features must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay non-negative")
    if args.fallback_weight <= 0 or args.candidate_weight <= 0:
        parser.error("router class weights must be positive")
    if args.brier_weight < 0 or args.minimum_candidate_gain < 0:
        parser.error("Brier weight and minimum candidate gain must be non-negative")
    if args.maximum_mean_utility_regression < 0:
        parser.error("maximum mean utility regression must be non-negative")
    if args.minimum_candidate_selections <= 0:
        parser.error("minimum candidate selections must be positive")
    if not 0.0 < float(args.minimum_threshold) < 1.0:
        parser.error("--minimum-threshold must be strictly within (0, 1)")
    if not 0.0 < float(args.minimum_precision) <= 1.0:
        parser.error("--minimum-precision must be within (0, 1]")
    if args.target_mode == "source_gamus":
        if args.allow_unprovenanced_records:
            parser.error("source_gamus mode forbids unprovenanced records")
        if float(args.minimum_precision) < 0.99:
            parser.error("source_gamus mode requires minimum precision >= 0.99")
        if not 0.0 < float(args.minimum_gamus_coverage) <= 1.0:
            parser.error("source_gamus minimum GAMUS coverage must be within (0, 1]")
        if int(args.minimum_gamus_selections) <= 0:
            parser.error("source_gamus minimum GAMUS selections must be positive")
        if float(args.maximum_mean_utility_regression) != 0.0:
            parser.error("source_gamus mode requires zero mean utility regression")
    return args


def validate_source_gamus_records(
    train_records: Sequence[SceneRouterRecord],
    calibration_records: Sequence[SceneRouterRecord],
    *,
    record_provenance: Mapping[str, object] | None,
    record_completion: Mapping[str, object] | None,
) -> dict[str, dict[str, int]]:
    """Fail closed unless authenticated 128-D train/val records are intact."""

    if record_provenance is None or record_completion is None:
        raise ValueError("source_gamus mode requires authenticated record evidence")
    completion_descriptor_size = record_completion.get("descriptor_size")
    if (
        isinstance(completion_descriptor_size, bool)
        or not isinstance(completion_descriptor_size, int)
        or completion_descriptor_size != 128
    ):
        raise ValueError("source_gamus mode requires authenticated 128-D descriptors")
    completion_record_count = record_completion.get("record_count")
    if (
        isinstance(completion_record_count, bool)
        or not isinstance(completion_record_count, int)
        or completion_record_count
        != len(train_records) + len(calibration_records)
    ):
        raise ValueError(
            "source_gamus records contain an unsupported or unaccounted partition"
        )
    data = record_provenance.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("source_gamus record data provenance is missing")
    included = data.get("included_source_splits")
    if not isinstance(included, list) or len(included) != 4 or set(included) != {
        "gamus/train",
        "gamus/val",
        "legacy/train",
        "legacy/val",
    }:
        raise ValueError("source_gamus records do not prove exact train/val sources")
    excluded = data.get("excluded_source_splits")
    if not isinstance(excluded, list) or len(excluded) != 2 or set(excluded) != {
        "gamus/test",
        "legacy/test",
    }:
        raise ValueError("source_gamus records do not prove test-split exclusion")

    counts: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for partition_name, records, expected_split in (
        ("train", train_records, "train"),
        ("calibration", calibration_records, "val"),
    ):
        source_counts = {"gamus": 0, "legacy": 0}
        for record in records:
            if record.descriptor.numel() != 128:
                raise ValueError("source_gamus record descriptor is not 128-D")
            if record.source not in source_counts:
                raise ValueError(f"unsupported source_gamus source: {record.source!r}")
            components = record.sample_id.split("/")
            if (
                len(components) != 3
                or components[0] != record.source
                or components[1] != expected_split
                or not components[2]
            ):
                raise ValueError(
                    "source_gamus record key disagrees with authenticated source/split: "
                    f"{record.sample_id!r}"
                )
            if record.sample_id in seen:
                raise ValueError(f"duplicate source_gamus record: {record.sample_id}")
            seen.add(record.sample_id)
            source_counts[record.source] += 1
        if any(value <= 0 for value in source_counts.values()):
            raise ValueError(
                f"source_gamus {partition_name} partition requires both sources"
            )
        counts[partition_name] = source_counts
    return counts


def _source_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    fallback_utilities: np.ndarray,
    candidate_utilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    metrics = compute_scene_router_metrics(
        probabilities,
        targets,
        threshold=threshold,
        fallback_utilities=fallback_utilities,
        candidate_utilities=candidate_utilities,
    )
    metrics.update(
        {
            "source_precision": float(metrics["precision"]),
            "source_recall": float(metrics["recall"]),
            "gamus_coverage": float(metrics["recall"]),
            "selected_gamus": int(metrics["tp"]),
            "selected_legacy": int(metrics["fp"]),
            "routed_utility_gain_vs_protected": float(
                metrics["routed_gain_vs_fallback"]
            ),
        }
    )
    return metrics


def _all_fallback_source_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    fallback_utilities: np.ndarray,
    candidate_utilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Report a genuinely disabled route even for sigmoid scores rounded to one."""

    # Use the regular implementation for all probability/calibration terms,
    # then replace every selection-dependent value with the exact no-route case.
    metrics = _source_metrics(
        probabilities,
        targets,
        fallback_utilities,
        candidate_utilities,
        threshold=1.0,
    )
    truth = np.asarray(targets, dtype=np.float64).reshape(-1).astype(bool)
    fallback = np.asarray(fallback_utilities, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_utilities, dtype=np.float64).reshape(-1)
    total = int(truth.size)
    gamus = int(np.count_nonzero(truth))
    legacy = total - gamus
    oracle = np.minimum(fallback, candidate)
    metrics.update(
        {
            "threshold": float(threshold),
            "candidate_selected": 0,
            "candidate_selection_rate": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": gamus,
            "tn": legacy,
            "accuracy": float(legacy / total),
            "balanced_accuracy": 0.5,
            "precision": 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "routed_mean_utility": float(fallback.mean()),
            "oracle_mean_utility": float(oracle.mean()),
            "routed_gain_vs_fallback": 0.0,
            "routed_regret_vs_oracle": float(fallback.mean() - oracle.mean()),
            "selected_candidate_mean_gain": 0.0,
            "source_precision": 1.0,
            "source_recall": 0.0,
            "gamus_coverage": 0.0,
            "selected_gamus": 0,
            "selected_legacy": 0,
            "routed_utility_gain_vs_protected": 0.0,
        }
    )
    return metrics


def select_source_gamus_threshold(
    probabilities: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    fallback_utilities: np.ndarray | torch.Tensor,
    candidate_utilities: np.ndarray | torch.Tensor,
    *,
    minimum_threshold: float,
    minimum_source_precision: float,
    minimum_gamus_coverage: float,
    minimum_gamus_selections: int,
) -> SourceGamusThresholdResult:
    """Maximize GAMUS coverage subject to source and utility safety guards."""

    if not 0.0 < minimum_threshold < 1.0:
        raise ValueError("source threshold floor must be within (0, 1)")
    if not 0.99 <= minimum_source_precision <= 1.0:
        raise ValueError("source precision guard must be within [0.99, 1]")
    if not 0.0 < minimum_gamus_coverage <= 1.0:
        raise ValueError("minimum GAMUS coverage must be within (0, 1]")
    if minimum_gamus_selections <= 0:
        raise ValueError("minimum GAMUS selections must be positive")
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    fallback = np.asarray(fallback_utilities, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_utilities, dtype=np.float64).reshape(-1)
    if probability.shape != target.shape:
        raise ValueError("source probabilities and labels must have matching shapes")
    if set(np.unique(target)) != {0.0, 1.0}:
        raise ValueError("source calibration requires GAMUS and legacy labels")
    gamus_count = int(np.count_nonzero(target == 1.0))
    if minimum_gamus_selections > gamus_count:
        raise ValueError("minimum GAMUS selections exceeds calibration support")
    candidate_thresholds = sorted(
        {
            float(minimum_threshold),
            *(
                float(value)
                for value in probability
                if minimum_threshold <= value < 1.0
            ),
        }
    )
    eligible: list[
        tuple[tuple[float, float, float], dict[str, float | int]]
    ] = []
    for threshold in candidate_thresholds:
        metrics = _source_metrics(
            probability,
            target,
            fallback,
            candidate,
            threshold=threshold,
        )
        if float(metrics["source_precision"]) < minimum_source_precision:
            continue
        if float(metrics["gamus_coverage"]) < minimum_gamus_coverage:
            continue
        if int(metrics["selected_gamus"]) < minimum_gamus_selections:
            continue
        if float(metrics["routed_utility_gain_vs_protected"]) < 0.0:
            continue
        # The domain objective is coverage-first. Utility gain and a higher
        # threshold break only equal-coverage ties.
        score = (
            float(metrics["gamus_coverage"]),
            float(metrics["routed_utility_gain_vs_protected"]),
            float(threshold),
        )
        eligible.append((score, metrics))
    if eligible:
        _, metrics = max(eligible, key=lambda item: item[0])
        return SourceGamusThresholdResult(
            threshold=float(metrics["threshold"]),
            eligible=True,
            reason=(
                "source precision, GAMUS coverage/selections, and routed-utility "
                "guards passed"
            ),
            metrics=metrics,
            evaluated_thresholds=len(candidate_thresholds),
        )
    fallback_threshold = float(
        np.nextafter(np.float32(1.0), np.float32(np.inf))
    )
    metrics = _all_fallback_source_metrics(
        probability,
        target,
        fallback,
        candidate,
        threshold=fallback_threshold,
    )
    return SourceGamusThresholdResult(
        threshold=fallback_threshold,
        eligible=False,
        reason="no source threshold passed; route all scenes to protected fallback",
        metrics=metrics,
        evaluated_thresholds=len(candidate_thresholds),
    )


def collect_source_gamus_validation(
    router: ConservativeSceneRouter,
    loader: DataLoader,
    *,
    device: str | torch.device,
) -> dict[str, np.ndarray]:
    """Collect source probabilities; the network receives descriptors only."""

    target_device = torch.device(device)
    router.eval()
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    fallback: list[torch.Tensor] = []
    candidate: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            descriptor = batch["descriptor"].to(target_device)
            logits = router.network(descriptor)
            probabilities.append(logits.sigmoid().float().cpu())
            targets.append(batch["candidate_target"].float().cpu())
            fallback.append(batch["fallback_utility"].float().cpu())
            candidate.append(batch["candidate_utility"].float().cpu())
    if not probabilities:
        raise ValueError("source_gamus calibration loader produced no scenes")
    return {
        "probabilities": torch.cat(probabilities).numpy(),
        "targets": torch.cat(targets).numpy(),
        "fallback_utilities": torch.cat(fallback).numpy(),
        "candidate_utilities": torch.cat(candidate).numpy(),
    }


def load_record_store_evidence(
    records_path: Path,
    *,
    provenance_path: Path | None,
    completion_path: Path | None,
    allow_unprovenanced: bool,
) -> tuple[dict | None, dict | None]:
    provenance_path = provenance_path or records_path.parent / "provenance.json"
    completion_path = completion_path or records_path.parent / "completion.json"
    if not provenance_path.is_file() or not completion_path.is_file():
        if allow_unprovenanced:
            return None, None
        raise ValueError(
            "Stage-3 router training requires record provenance.json and "
            "completion.json sidecars"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if provenance.get("schema") != RECORD_STORE_SCHEMA:
        raise ValueError("record provenance schema is unsupported")
    expected_config_hash = canonical_json_sha256(
        {
            "generation_config": provenance.get("generation_config"),
            "endpoints": provenance.get("endpoints"),
            "shared_model": provenance.get("shared_model"),
            "data": provenance.get("data"),
        }
    )
    if provenance.get("config_sha256") != expected_config_hash:
        raise ValueError("record provenance config hash does not match its contents")
    if completion.get("schema") != RECORD_STORE_SCHEMA or not completion.get("complete"):
        raise ValueError("record completion sidecar is invalid or incomplete")
    if completion.get("config_sha256") != provenance.get("config_sha256"):
        raise ValueError("record provenance and completion config hashes differ")
    validate_record_selection_provenance(
        provenance.get("data"), completed_record_count=completion.get("record_count")
    )
    if completion.get("records_sha256") != file_sha256(records_path):
        raise ValueError("record JSONL hash differs from completion sidecar")
    if not bool(provenance.get("test_splits_excluded")):
        raise ValueError("router records do not prove exclusion of test splits")
    return provenance, completion


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if str(args.device).startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    args.records = args.records.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    artifact_stem = (
        "scene_domain_router" if args.target_mode == "source_gamus" else "scene_router"
    )
    checkpoint_path = args.output_dir / f"{artifact_stem}.pt"
    report_path = args.output_dir / f"{artifact_stem}_report.json"
    existing_outputs = [
        path for path in (checkpoint_path, report_path) if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "refusing to overwrite an existing audited router artifact: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    record_provenance, record_completion = load_record_store_evidence(
        args.records,
        provenance_path=(
            args.record_provenance.expanduser().resolve()
            if args.record_provenance is not None
            else None
        ),
        completion_path=(
            args.record_completion.expanduser().resolve()
            if args.record_completion is not None
            else None
        ),
        allow_unprovenanced=args.allow_unprovenanced_records,
    )
    if record_provenance is not None:
        recorded_gain = float(
            record_provenance["generation_config"]["utility"][
                "minimum_candidate_gain"
            ]
        )
        if recorded_gain != float(args.minimum_candidate_gain):
            raise ValueError(
                "training minimum_candidate_gain differs from record provenance"
            )
    records = read_scene_router_records(args.records)
    if record_completion is not None and int(record_completion["record_count"]) != len(records):
        raise ValueError("record count differs from completion sidecar")
    train_records = [record for record in records if record.partition == "train"]
    calibration_records = [
        record for record in records if record.partition in {"calibration", "validation"}
    ]
    if not train_records:
        raise ValueError("records contain no 'train' partition")
    if not calibration_records:
        raise ValueError(
            "records require a separate 'calibration' or 'validation' partition"
        )
    descriptor_size = train_records[0].descriptor.numel()
    if descriptor_size % 2:
        raise ValueError(
            "descriptor size must be 2 x feature_channels (scene mean + std)"
        )
    if any(record.descriptor.numel() != descriptor_size for record in records):
        raise ValueError("all partitions must use the same descriptor size")

    source_partition_counts: dict[str, dict[str, int]] | None = None
    if args.target_mode == "source_gamus":
        source_partition_counts = validate_source_gamus_records(
            train_records,
            calibration_records,
            record_provenance=record_provenance,
            record_completion=record_completion,
        )
        train_dataset = SourceGamusRouterDataset(train_records)
        calibration_dataset = SourceGamusRouterDataset(calibration_records)
        sampler: Sampler[int] = BalancedSourceSampler(
            train_records, seed=args.seed
        )
    else:
        train_dataset = SceneRouterDataset(
            train_records, minimum_candidate_gain=args.minimum_candidate_gain
        )
        calibration_dataset = SceneRouterDataset(
            calibration_records, minimum_candidate_gain=args.minimum_candidate_gain
        )
        sampler = BalancedSourceLandscapeSampler(train_records, seed=args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    router = ConservativeSceneRouter(
        descriptor_size // 2,
        hidden_features=args.hidden_features,
        decision_threshold=args.minimum_threshold,
    )
    training_config = SceneRouterTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        candidate_weight=args.candidate_weight,
        fallback_weight=args.fallback_weight,
        brier_weight=args.brier_weight,
        minimum_threshold=args.minimum_threshold,
        minimum_precision=args.minimum_precision,
        maximum_mean_utility_regression=args.maximum_mean_utility_regression,
        minimum_candidate_selections=args.minimum_candidate_selections,
    )
    result = train_scene_router(
        router,
        train_loader,
        calibration_loader,
        config=training_config,
        device=args.device,
    )
    threshold = result.threshold
    if args.target_mode == "source_gamus":
        arrays = collect_source_gamus_validation(
            router, calibration_loader, device=args.device
        )
        threshold = select_source_gamus_threshold(
            arrays["probabilities"],
            arrays["targets"],
            arrays["fallback_utilities"],
            arrays["candidate_utilities"],
            minimum_threshold=args.minimum_threshold,
            minimum_source_precision=args.minimum_precision,
            minimum_gamus_coverage=args.minimum_gamus_coverage,
            minimum_gamus_selections=args.minimum_gamus_selections,
        )
        router.decision_threshold.fill_(threshold.threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # This artifact contains the router only.  The frozen endpoints remain in
    # their original checkpoint files and the live showcase pointer is untouched.
    record_provenance_hash = (
        canonical_json_sha256(record_provenance)
        if record_provenance is not None
        else None
    )
    record_completion_hash = (
        canonical_json_sha256(record_completion)
        if record_completion is not None
        else None
    )
    launch_config = {
        "records": str(args.records),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_features": args.hidden_features,
        "minimum_candidate_gain": args.minimum_candidate_gain,
        "target_mode": args.target_mode,
        "minimum_source_precision": (
            args.minimum_precision if args.target_mode == "source_gamus" else None
        ),
        "minimum_gamus_coverage": (
            args.minimum_gamus_coverage
            if args.target_mode == "source_gamus"
            else None
        ),
        "minimum_gamus_selections": (
            args.minimum_gamus_selections
            if args.target_mode == "source_gamus"
            else None
        ),
        "seed": args.seed,
        "device": args.device,
        "config_path": (
            str(args.config.expanduser().resolve()) if args.config is not None else None
        ),
        "config_sha256": (
            file_sha256(args.config.expanduser().resolve())
            if args.config is not None
            else None
        ),
        "trainer_sha256": file_sha256(Path(__file__).resolve()),
    }
    if args.target_mode == "source_gamus":
        artifact_schema = "msr.stage3b_scene_domain_router.v1"
        report_schema = "msr.stage3b_scene_domain_router_report.v1"
        artifact_type = "offline_scene_domain_router_only"
        target_provenance = (
            "candidate target is one iff authenticated record.source is GAMUS; "
            "source creates supervision and balanced sampling only, while the "
            "router network receives the 128-D scene descriptor alone; held-out "
            "endpoint utilities are used only for the no-regression threshold guard"
        )
    else:
        artifact_schema = "msr.stage3_scene_router.v1"
        report_schema = "msr.stage3_scene_router_report.v1"
        artifact_type = "offline_scene_router_only"
        target_provenance = (
            "candidate iff held-out fallback utility - candidate utility exceeds "
            "minimum_candidate_gain; source and landscape are balancing metadata only"
        )
    source_threshold_guards = (
        {
            "minimum_source_precision": args.minimum_precision,
            "minimum_gamus_coverage": args.minimum_gamus_coverage,
            "minimum_gamus_selections": args.minimum_gamus_selections,
            "minimum_routed_utility_gain_vs_protected": 0.0,
            "selection_priority": [
                "maximum_gamus_coverage",
                "maximum_routed_utility_gain_vs_protected",
                "highest_threshold",
            ],
        }
        if args.target_mode == "source_gamus"
        else None
    )
    checkpoint_payload = {
        "artifact_schema": artifact_schema,
        "artifact_type": artifact_type,
        "target_mode": args.target_mode,
        "target_provenance": target_provenance,
        "source_threshold_guards": source_threshold_guards,
        "router_state_dict": {
            name: value.detach().cpu() for name, value in router.state_dict().items()
        },
        "feature_channels": descriptor_size // 2,
        "hidden_features": args.hidden_features,
        "decision_threshold": threshold.threshold,
        "threshold_eligible": threshold.eligible,
        "minimum_candidate_gain": args.minimum_candidate_gain,
        "training_config": asdict(training_config),
        "training_launch": launch_config,
        "record_provenance": record_provenance,
        "record_completion": record_completion,
        "record_provenance_canonical_sha256": record_provenance_hash,
        "record_completion_canonical_sha256": record_completion_hash,
        "stage3_eligible_record_provenance": record_provenance is not None,
        "stage3b_eligible_record_provenance": (
            args.target_mode == "source_gamus" and record_provenance is not None
        ),
    }
    _atomic_torch_save(checkpoint_payload, checkpoint_path)
    router_artifact_sha256 = file_sha256(checkpoint_path)
    if isinstance(sampler, BalancedSourceSampler):
        balanced_train_strata = {
            source: len(indices) for source, indices in sampler.groups.items()
        }
        samples_per_stratum = sampler.samples_per_source
    else:
        balanced_train_strata = {
            f"{source}/{landscape}": len(indices)
            for (source, landscape), indices in sampler.groups.items()
        }
        samples_per_stratum = sampler.samples_per_stratum
    report = {
        "artifact_schema": report_schema,
        "artifact_type": artifact_type.replace("_only", "_report"),
        "target_mode": args.target_mode,
        "router_artifact_path": str(checkpoint_path.resolve()),
        "router_artifact_sha256": router_artifact_sha256,
        "training_config": asdict(training_config),
        "training_launch": launch_config,
        "record_provenance": record_provenance,
        "record_completion": record_completion,
        "record_provenance_canonical_sha256": record_provenance_hash,
        "record_completion_canonical_sha256": record_completion_hash,
        "records": {
            "train": len(train_records),
            "calibration": len(calibration_records),
        },
        "target_provenance": target_provenance,
        "minimum_candidate_gain": args.minimum_candidate_gain,
        "source_partition_counts": source_partition_counts,
        "source_threshold_guards": source_threshold_guards,
        "balanced_train_strata": balanced_train_strata,
        "samples_per_stratum_per_epoch": samples_per_stratum,
        "best_validation_bce": result.best_validation_bce,
        "threshold": {
            "value": threshold.threshold,
            "eligible": threshold.eligible,
            "reason": threshold.reason,
            "metrics": threshold.metrics,
            "evaluated_thresholds": threshold.evaluated_thresholds,
        },
        "history": list(result.history),
        "live_application_pointer_changed": False,
    }
    _atomic_json_save(report, report_path)
    print(f"Router checkpoint: {checkpoint_path}")
    print(f"Audit report: {report_path}")
    print(
        "Eligible for later endpoint/app validation: "
        f"{threshold.eligible} ({threshold.reason})"
    )


if __name__ == "__main__":
    main()
