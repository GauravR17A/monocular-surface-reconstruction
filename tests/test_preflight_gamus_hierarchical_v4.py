from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from msr.evaluation.classification_metrics import compute_multiclass_metrics


ROOT = Path(__file__).parents[1]
PREFLIGHT_PATH = ROOT / "scripts" / "preflight_gamus_hierarchical_v4.py"
BUILDER_PATH = ROOT / "scripts" / "build_gamus_hierarchical_v4_config.py"
COMPARATOR_PATH = (
    ROOT
    / "configs"
    / "multidomain_surface_gamus_six_class_spatial_refined_geographic_v1.yaml"
)
BASELINE_AUDIT_PATH = ROOT / "outputs" / "evaluation" / "gamus_spatial_refined_geographic_v1" / "checkpoint_best_landscape_audit.json"
BASELINE_REPLAY_PATH = ROOT / "outputs" / "evaluation" / "gamus_dcphl_independent_replay_v1" / "v3_baseline_report.json"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _module("v4_preflight_test", PREFLIGHT_PATH)
BUILDER = _module("v4_builder_test", BUILDER_PATH)
CLASS_NAMES = PREFLIGHT.CLASS_NAMES


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline_metrics(diagonal: int = 60) -> dict:
    matrix = np.full((6, 6), (100 - diagonal) // 5, dtype=np.int64)
    np.fill_diagonal(matrix, diagonal)
    return {
        "six_class_identification": compute_multiclass_metrics(matrix, CLASS_NAMES),
        "road_boundary_quality": {"f1": 0.57},
        "water_shadow_proxy": {
            "false_water_rate_on_dark_non_water": 0.12,
        },
    }


def _baseline_audit(metrics: dict) -> dict:
    checkpoint = (
        ROOT
        / "experiments"
        / "20260829T203146Z_multidomain_surface_v2_guarded_vegetation"
        / "checkpoint_best_guarded_vegetation.pt"
    )
    return {
        "schema": PREFLIGHT.BASELINE_AUDIT_SCHEMA,
        "candidate_config": {"sha256": _sha(COMPARATOR_PATH)},
        "candidate": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
        "recipe_audit": {"passes": True},
        "state_audit": {"passes": True},
        "six_class_metric_audit": {"passes": True},
        "candidate_metrics": metrics,
        "nyc_evaluation_performed": False,
        "official_test_used_by_this_comparator": False,
        "official_test_global_status": "previously_consumed_forbidden_for_reuse",
        "promotion_performed": False,
        "app_pointer_verification": {"passes": True},
    }


def _candidate(tmp_path: Path) -> tuple[dict, dict, dict, dict, Path, Path]:
    comparator = yaml.safe_load(COMPARATOR_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_AUDIT_PATH.read_text(encoding="utf-8"))
    replay = json.loads(BASELINE_REPLAY_PATH.read_text(encoding="utf-8"))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    replay_path = tmp_path / "baseline_replay.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    candidate = BUILDER.build_config(
        comparator,
        baseline,
        replay,
        comparator_path=COMPARATOR_PATH,
        baseline_audit_path=baseline_path,
        baseline_replay_path=replay_path,
    )
    return candidate, comparator, baseline, replay, baseline_path, replay_path


def test_preflight_accepts_builder_output_as_exact_paired_experiment(
    tmp_path: Path,
) -> None:
    candidate, comparator, baseline, replay, baseline_path, replay_path = _candidate(tmp_path)

    result = PREFLIGHT.validate_v4_config(
        candidate,
        comparator,
        baseline,
        replay,
        comparator_path=COMPARATOR_PATH,
        baseline_replay_path=replay_path,
        comparator_sha256=_sha(COMPARATOR_PATH),
        baseline_sha256=_sha(baseline_path),
        baseline_replay_sha256=_sha(replay_path),
    )

    assert result["passes"] is True
    assert result["same_data"] is True
    assert result["same_seed_schedule_optimizer_and_selection"] is True
    assert result["height_output_identity_required_before_holdout"] is True
    assert result["official_test_used_by_v4"] is False
    assert result["official_test_global_status"] == (
        "previously_consumed_forbidden_for_reuse"
    )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("training", "epochs"), 9, "schedule/optimizer/losses"),
        (("training", "fine_semantic_vegetation_split_weight"), 0.4, "must equal"),
        (("data", "patch_size"), 256, "byte-identical"),
        (("model", "fusion_mode"), "legacy", "outside the classifier head"),
        (
            ("protocol", "official_test_policy"),
            "never_constructed_or_discovered",
            "forbid reuse",
        ),
    ],
)
def test_preflight_rejects_unfair_or_unsafe_drift(
    tmp_path: Path, path: tuple[str, str], value: object, message: str
) -> None:
    candidate, comparator, baseline, replay, baseline_path, replay_path = _candidate(tmp_path)
    candidate = deepcopy(candidate)
    candidate[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=message):
        PREFLIGHT.validate_v4_config(
            candidate,
            comparator,
            baseline,
            replay,
            comparator_path=COMPARATOR_PATH,
            baseline_replay_path=replay_path,
            comparator_sha256=_sha(COMPARATOR_PATH),
            baseline_sha256=_sha(baseline_path),
            baseline_replay_sha256=_sha(replay_path),
        )


def test_thresholds_are_fixed_from_all_six_baseline_classes(tmp_path: Path) -> None:
    candidate, comparator, baseline, replay, baseline_path, replay_path = _candidate(tmp_path)
    expected = PREFLIGHT.derive_fixed_gates(replay["v3"]["overall"])
    assert candidate["protocol"]["classification_acceptance"] == expected

    candidate["protocol"]["classification_acceptance"]["buildings_f1_min"] -= 0.01
    with pytest.raises(ValueError, match="must be frozen"):
        PREFLIGHT.validate_v4_config(
            candidate,
            comparator,
            baseline,
            replay,
            comparator_path=COMPARATOR_PATH,
            baseline_replay_path=replay_path,
            comparator_sha256=_sha(COMPARATOR_PATH),
            baseline_sha256=_sha(baseline_path),
            baseline_replay_sha256=_sha(replay_path),
        )


def test_preflight_rejects_hand_edited_six_class_baseline(tmp_path: Path) -> None:
    candidate, comparator, baseline, replay, baseline_path, replay_path = _candidate(tmp_path)
    replay = deepcopy(replay)
    replay["v3"]["overall"]["six_class_identification"]["macro_f1"] = 0.99

    with pytest.raises(ValueError, match="inconsistent|mean of six classes"):
        PREFLIGHT.validate_v4_config(
            candidate,
            comparator,
            baseline,
            replay,
            comparator_path=COMPARATOR_PATH,
            baseline_replay_path=replay_path,
            comparator_sha256=_sha(COMPARATOR_PATH),
            baseline_sha256=_sha(baseline_path),
            baseline_replay_sha256=_sha(replay_path),
        )
