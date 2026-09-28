from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from msr.evaluation.classification_metrics import compute_multiclass_metrics


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "audit_gamus_spatial_refined_geographic_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("geographic_v3_audit_test", SCRIPT)
assert SPEC and SPEC.loader
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)
CONFIG = (
    ROOT
    / "configs"
    / "multidomain_surface_gamus_six_class_spatial_refined_geographic_v1.yaml"
)


def _config() -> dict:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _six_class_metrics(diagonal: int = 70) -> dict:
    matrix = np.full((6, 6), (100 - diagonal) // 5, dtype=np.int64)
    np.fill_diagonal(matrix, diagonal)
    return {
        "six_class_identification": compute_multiclass_metrics(
            matrix, AUDITOR.CLASS_NAMES
        )
    }


def test_exact_geographic_comparator_recipe_passes_without_nyc_imagery() -> None:
    result = AUDITOR.validate_geographic_comparator_recipe(
        _config(), project_root=ROOT
    )

    assert result["passes"] is True
    assert result["classifier_training_cities"] == ["DC", "PHL"]
    assert result["development_selection_cities"] == ["DC", "PHL"]
    assert result["nyc_imagery_opened"] is False
    assert result["official_test_used_by_this_comparator"] is False
    assert result["official_test_global_status"] == (
        "previously_consumed_forbidden_for_reuse"
    )
    assert result["promotion_performed"] is False


@pytest.mark.parametrize(
    ("key", "bad_value", "message"),
    [
        ("locked_holdout_interpretation", "system_unseen", "must not be described"),
        ("external_system_unseen_geography", "NYC", "must remain pending"),
        ("official_test_policy", "discover", "must forbid"),
        ("auto_promotion", True, "must remain disabled"),
    ],
)
def test_geographic_comparator_rejects_truth_or_safety_drift(
    key: str, bad_value: object, message: str
) -> None:
    config = deepcopy(_config())
    config["protocol"][key] = bad_value

    with pytest.raises(ValueError, match=message):
        AUDITOR.validate_geographic_comparator_recipe(config, project_root=ROOT)


def test_six_class_scores_are_recomputed_from_exact_6x6_matrix() -> None:
    metrics = _six_class_metrics(70)
    result = AUDITOR.recompute_six_class_metrics(metrics)

    assert result["passes"] is True
    assert result["macro"]["f1"] == pytest.approx(0.70)
    assert set(result["per_class"]) == set(AUDITOR.CLASS_NAMES)


def test_five_group_or_hand_edited_macro_cannot_pass() -> None:
    metrics = _six_class_metrics(55)
    metrics["six_class_identification"]["macro"]["f1"] = 0.95
    metrics["six_class_identification"]["macro_f1"] = 0.95

    with pytest.raises(ValueError, match="mean of six classes"):
        AUDITOR.recompute_six_class_metrics(metrics)


def test_auditor_source_forbids_holdout_evaluation_and_promotion() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "nyc_evaluation_performed\": False" in source
    assert "official_test_used_by_this_comparator\": False" in source
    assert '"previously_consumed_forbidden_for_reuse"' in source
    assert "promotion_performed\": False" in source
    assert "GamusSurfaceDataset" not in source
    assert "evaluate-locked" not in source
    assert "select_showcase_checkpoint" not in source
