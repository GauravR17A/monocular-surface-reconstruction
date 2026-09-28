"""Read-only, one-line-per-epoch watcher for the GAMUS head-only v2 run.

This utility reads only the experiment metrics, the frozen run configuration,
the v1 comparison report, and the orchestration log.  It never opens a model
checkpoint, changes a process, or writes any file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "multidomain_surface_gamus_six_class_head_only_v2.yaml"
)
DEFAULT_LOG = PROJECT_ROOT / "outputs" / "orchestration" / "gamus_six_class_head_only_v2.log"
EXPERIMENT_GLOB = "*_multidomain_surface_gamus_six_class_head_only_v2"
CLASS_NAMES = (
    "ground",
    "buildings",
    "water",
    "roads",
    "low_vegetation",
    "trees",
)
RETAINED_CLASS_NAMES = ("ground", "buildings", "low_vegetation", "trees")
DISPLAY_NAMES = {
    "ground": "G",
    "buildings": "B",
    "water": "W",
    "roads": "R",
    "low_vegetation": "LV",
    "trees": "T",
}


class MetricFormatError(ValueError):
    """Raised when a completed metrics record is incomplete or malformed."""


def _mapping(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetricFormatError(f"{role} must be an object")
    return value


def _number(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricFormatError(f"{role} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MetricFormatError(f"{role} must be finite")
    return result


def _boolean(value: object, role: str) -> bool:
    if not isinstance(value, bool):
        raise MetricFormatError(f"{role} must be boolean")
    return value


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_acceptance_inputs(
    config_path: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    """Load the predeclared targets and v1 per-class F1 reference."""

    config = _mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")), "watcher config"
    )
    protocol = _mapping(config.get("protocol"), "protocol")
    raw_targets = _mapping(
        protocol.get("classification_acceptance"), "classification acceptance"
    )
    target_names = (
        "macro_f1_min",
        "water_recall_min",
        "water_f1_min",
        "road_f1_min",
        "max_f1_drop_other_classes",
    )
    targets = {
        name: _number(raw_targets.get(name), f"classification target {name}")
        for name in target_names
    }

    report_path = _resolve_project_path(
        str(raw_targets.get("comparison_report", ""))
    )
    if not report_path.is_file():
        raise FileNotFoundError(f"comparison report is missing: {report_path}")
    report = _mapping(
        json.loads(report_path.read_text(encoding="utf-8")), "comparison report"
    )
    baseline = _mapping(
        report.get("expanded_classification"), "expanded classification baseline"
    )
    per_class = _mapping(baseline.get("per_class"), "baseline per-class metrics")
    baseline_f1 = {
        name: _number(
            _mapping(per_class.get(name), f"baseline class {name}").get("f1"),
            f"baseline {name} F1",
        )
        for name in RETAINED_CLASS_NAMES
    }
    return targets, baseline_f1


def summarize_epoch(
    record: Mapping[str, Any],
    targets: Mapping[str, float],
    baseline_f1: Mapping[str, float],
) -> dict[str, Any]:
    """Extract metrics and evaluate the frozen height/classification gates."""

    epoch_value = record.get("epoch")
    if isinstance(epoch_value, bool) or not isinstance(epoch_value, int) or epoch_value < 1:
        raise MetricFormatError("epoch must be a positive integer")

    metrics = _mapping(record.get("validation_metrics"), "validation metrics")
    six = _mapping(metrics.get("six_class_identification"), "six-class metrics")
    per_class = _mapping(six.get("per_class"), "six-class per-class metrics")
    class_values: dict[str, dict[str, float]] = {}
    for name in CLASS_NAMES:
        values = _mapping(per_class.get(name), f"class {name}")
        class_values[name] = {
            "precision": _number(values.get("precision"), f"{name} precision"),
            "recall": _number(values.get("recall"), f"{name} recall"),
            "f1": _number(values.get("f1"), f"{name} F1"),
        }

    macro_f1 = _number(six.get("macro_f1"), "macro F1")
    height_checks = {
        "validation": _boolean(
            metrics.get("passes_validation_guards"), "validation height guard"
        ),
        "initial": _boolean(
            metrics.get("passes_initial_checkpoint_guard"),
            "initial-checkpoint height guard",
        ),
        "protected": _boolean(
            metrics.get("passes_protected_base_guard"),
            "protected-base height guard",
        ),
    }
    height_passes = all(height_checks.values())

    classification_checks = {
        "macro": macro_f1 >= targets["macro_f1_min"],
        "waterR": class_values["water"]["recall"]
        >= targets["water_recall_min"],
        "waterF1": class_values["water"]["f1"] >= targets["water_f1_min"],
        "road": class_values["roads"]["f1"] >= targets["road_f1_min"],
    }
    max_drop = targets["max_f1_drop_other_classes"]
    for name in RETAINED_CLASS_NAMES:
        classification_checks[DISPLAY_NAMES[name]] = (
            class_values[name]["f1"] >= baseline_f1[name] - max_drop
        )

    failed_targets = [
        name for name, passed in classification_checks.items() if not passed
    ]
    if not height_passes:
        failed_targets.append("height")
    return {
        "epoch": epoch_value,
        "macro_f1": macro_f1,
        "classes": class_values,
        "height_passes": height_passes,
        "height_checks": height_checks,
        "classification_checks": classification_checks,
        "target_passes": not failed_targets,
        "failed_targets": failed_targets,
    }


def format_epoch_line(summary: Mapping[str, Any]) -> str:
    """Format a completed epoch into one stable, compact status line."""

    classes = _mapping(summary.get("classes"), "summary classes")
    water = _mapping(classes.get("water"), "summary water")

    def f1(name: str) -> float:
        return _number(
            _mapping(classes.get(name), f"summary {name}").get("f1"),
            f"summary {name} F1",
        )

    height = "PASS" if bool(summary.get("height_passes")) else "FAIL"
    target = "PASS" if bool(summary.get("target_passes")) else "FAIL"
    if target == "FAIL":
        failures = summary.get("failed_targets", [])
        if isinstance(failures, list) and failures:
            target += "[" + ",".join(str(value) for value in failures) + "]"
    return (
        f"E{int(summary['epoch']):02d} mF1={float(summary['macro_f1']):.3f} | "
        f"W(P/R/F1)={float(water['precision']):.3f}/"
        f"{float(water['recall']):.3f}/{float(water['f1']):.3f} "
        f"R={f1('roads'):.3f} | G={f1('ground'):.3f} "
        f"B={f1('buildings'):.3f} LV={f1('low_vegetation'):.3f} "
        f"T={f1('trees'):.3f} | H={height} TARGET={target}"
    )


def read_completed_records(metrics_path: Path) -> list[Mapping[str, Any]]:
    """Read complete JSONL records, tolerating only a partial final write."""

    raw = metrics_path.read_text(encoding="utf-8")
    lines = raw.splitlines(keepends=True)
    records: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            is_partial_tail = index == len(lines) - 1 and not line.endswith(("\n", "\r"))
            if is_partial_tail:
                continue
            raise MetricFormatError(
                f"invalid completed JSON record at {metrics_path}:{index + 1}"
            )
        records.append(_mapping(value, f"metrics record {index + 1}"))
    return records


def newest_experiment(experiments_root: Path) -> Path | None:
    matches = sorted(
        (path for path in experiments_root.glob(EXPERIMENT_GLOB) if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    return matches[0] if matches else None


def _run_has_ended(log_path: Path) -> bool:
    if not log_path.is_file():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-32_768:]
    last_start = max(tail.rfind(" START "), tail.rfind(" RESUME "))
    session = tail[last_start:] if last_start >= 0 else tail
    return (
        "COMPLETE head-only correction" in session
        or "FAILED " in session
        or "SAFETY FAILURE" in session
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--once", action="store_true", help="print only the latest epoch")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive and finite")
    config_path = _resolve_project_path(args.config)
    targets, baseline_f1 = load_acceptance_inputs(config_path)
    explicit_experiment = (
        _resolve_project_path(args.experiment) if args.experiment is not None else None
    )
    seen_epochs: set[int] = set()

    try:
        while True:
            experiment = explicit_experiment or newest_experiment(PROJECT_ROOT / "experiments")
            metrics_path = experiment / "metrics.jsonl" if experiment else None
            records = (
                read_completed_records(metrics_path)
                if metrics_path is not None and metrics_path.is_file()
                else []
            )
            summaries_by_epoch = {
                int(record["epoch"]): summarize_epoch(record, targets, baseline_f1)
                for record in records
            }
            if args.once:
                if summaries_by_epoch:
                    latest = summaries_by_epoch[max(summaries_by_epoch)]
                    print(format_epoch_line(latest), flush=True)
                else:
                    print("WAIT no completed epoch", flush=True)
                return

            for epoch in sorted(summaries_by_epoch):
                if epoch not in seen_epochs:
                    print(format_epoch_line(summaries_by_epoch[epoch]), flush=True)
                    seen_epochs.add(epoch)
            if _run_has_ended(DEFAULT_LOG):
                return
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
