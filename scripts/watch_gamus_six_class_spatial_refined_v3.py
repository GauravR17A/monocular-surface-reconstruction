"""Read-only compact watcher for the GAMUS spatial-refined v3 run.

The parsing, gate evaluation, and one-line formatting are deliberately reused
from the tested head-only v2 watcher.  This thin profile changes only the
configuration, experiment glob, orchestration log, and terminal log markers.
It never opens a checkpoint, changes a process, or writes a file.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "multidomain_surface_gamus_six_class_spatial_refined_v3.yaml"
)
DEFAULT_LOG = (
    PROJECT_ROOT
    / "outputs"
    / "orchestration"
    / "gamus_six_class_spatial_refined_v3.log"
)
EXPERIMENT_GLOB = "*_multidomain_surface_gamus_six_class_spatial_refined_v3"
BASE_WATCHER = Path(__file__).with_name("watch_gamus_six_class_head_only_v2.py")


def _load_base_watcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_msr_gamus_six_class_watcher_base", BASE_WATCHER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the shared watcher: {BASE_WATCHER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_watcher()


def _run_has_ended(log_path: Path) -> bool:
    """Return true only for a terminal marker in the latest v3 run session."""

    if not log_path.is_file():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="replace")[-32_768:]
    last_start = max(tail.rfind(" START "), tail.rfind(" RESUME "))
    session = tail[last_start:] if last_start >= 0 else tail
    return (
        "COMPLETE spatial-refined v3" in session
        or "PREFLIGHT COMPLETE; no training was started" in session
        or "FAILED " in session
        or "SAFETY FAILURE" in session
    )


def _configure_base() -> None:
    """Apply the v3 read profile to the shared watcher in this process only."""

    _BASE.DEFAULT_CONFIG = DEFAULT_CONFIG
    _BASE.DEFAULT_LOG = DEFAULT_LOG
    _BASE.EXPERIMENT_GLOB = EXPERIMENT_GLOB
    _BASE._run_has_ended = _run_has_ended


_configure_base()

# Re-export the tested, read-only parsing surface for focused tests and callers.
MetricFormatError = _BASE.MetricFormatError
load_acceptance_inputs = _BASE.load_acceptance_inputs
summarize_epoch = _BASE.summarize_epoch
format_epoch_line = _BASE.format_epoch_line
read_completed_records = _BASE.read_completed_records
newest_experiment = _BASE.newest_experiment


def main() -> None:
    _configure_base()
    _BASE.main()


if __name__ == "__main__":
    main()
