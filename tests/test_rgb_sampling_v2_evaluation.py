from copy import deepcopy

import numpy as np
import pytest

from msr.evaluation.classification_metrics import compute_multiclass_metrics
from msr.evaluation.rgb_sampling_v2 import compare_epoch, make_floors
from msr.evaluation.rgb_segmentation import CLASS_NAMES


def _group(matrix):
    return {
        "six_class_identification": compute_multiclass_metrics(matrix, CLASS_NAMES),
        "road_boundary_quality": {"class_index": 3, "tolerance_pixels": 2,
                                  "reference_boundary_pixels": 1000, "f1": 0.6},
        "water_dark_pixel_proxy": {"is_shadow_proxy_not_ground_truth": True,
                                   "dark_non_water_pixels": 10000,
                                   "false_water_rate_on_dark_non_water": 0.005},
    }


def _evaluation(lowveg_gain=0, phl_building_loss=0):
    dc = np.eye(6, dtype=np.int64) * 10000
    dc[0, 0], dc[0, 4] = 7000, 3000
    dc[4, 0], dc[4, 4] = 9000 - lowveg_gain, 1000 + lowveg_gain
    phl = np.eye(6, dtype=np.int64) * 10000
    phl[1, 1] -= phl_building_loss
    phl[1, 2] += phl_building_loss
    result = {
        "validation_native_size": 1024, "evaluated_sample_count": 859,
        "evaluated_samples_by_city": {"DC": 359, "PHL": 500},
        "evaluated_ids_sha256": "a" * 64, "evaluated_ordered_ids_sha256": "a" * 64,
        "evaluated_ids_by_city_sha256": {"DC": "b" * 64, "PHL": "c" * 64},
        "reference_grid_binding_sha256": "d" * 64,
        "overall": _group(dc + phl), "by_city": {"DC": _group(dc), "PHL": _group(phl)},
    }
    result["overall"]["road_boundary_quality"]["reference_boundary_pixels"] = 2000
    result["overall"]["water_dark_pixel_proxy"]["dark_non_water_pixels"] = 20000
    return result


def _config(**overrides):
    return {"comparison_epoch": 4, "control_epoch": 4, **overrides}


def test_floors_cover_every_class_and_city_and_clip_to_valid_range():
    evaluation = _evaluation()
    overall, cities = make_floors(evaluation)
    assert len(overall) == 9 and set(cities) == {"DC", "PHL"}
    assert sum(len(values) for values in cities.values()) + len(overall) == 27
    assert cities["DC"]["low_vegetation_f1_min"] == pytest.approx(1 / 7 - 0.01)
    assert overall["six_class_macro_f1_min"] == pytest.approx(evaluation["overall"]["six_class_identification"]["macro_f1"] - .005)
    assert overall["road_boundary_f1_min"] == pytest.approx(.59)
    assert overall["dark_non_water_false_water_rate_max"] == pytest.approx(.006)
    overall, _ = make_floors(evaluation, {"class_f1": 1, "dark_non_water_false_water_rate": 1})
    assert overall["ground_f1_min"] == 0 and overall["dark_non_water_false_water_rate_max"] == 1


def test_no_control_never_claims_experiment_win_but_reports_safety_and_benefit():
    result = compare_epoch(_evaluation(3000), _evaluation(), {})
    assert result["safe_vs_v1"] and result["qualified_vs_v1"]
    assert not result["experiment_win"] and not result["pairing"]["passes"]
    assert result["safety_vs_control"] is None and result["benefit_vs_control"] is None
    assert "candidate_minus_v3" not in result["safety_vs_v1"]
    assert "comparable_to_fixed_v3" not in result["safety_vs_v1"]
    assert result["safety_vs_v1"]["reference"] == "frozen_rgb_v1"


def test_same_final_epoch_improvement_over_both_references_passes_without_promotion():
    result = compare_epoch(_evaluation(3000), _evaluation(), _config(), _evaluation(1000))
    assert result["experiment_win"] and result["passes"]
    assert result["safety_vs_control"]["passes"] and result["benefit_vs_control"]["passes"]
    assert len(result["safety_vs_v1"]["checks"]) == 27
    assert not result["promotion_eligible"] and not result["promotion_performed"]


def test_safety_is_separate_from_meaningful_benefit_and_matched_control_win():
    minimal = compare_epoch(_evaluation(100), _evaluation(), _config(), _evaluation())
    assert minimal["safe_vs_v1"] and not minimal["benefit_vs_v1"]["passes"]
    tied = compare_epoch(_evaluation(3000), _evaluation(), _config(), _evaluation(3000))
    assert tied["qualified_vs_v1"] and tied["safety_vs_control"]["passes"]
    assert not tied["benefit_vs_control"]["passes"] and not tied["experiment_win"]


def test_target_gain_cannot_hide_another_city_class_regression():
    result = compare_epoch(_evaluation(3000, phl_building_loss=1000), _evaluation(), _config(), _evaluation(1000))
    assert result["benefit_vs_v1"]["passes"]
    assert not result["safe_vs_v1"] and not result["experiment_win"]
    assert any("PHL.buildings_f1_min" in failure for failure in result["safety_vs_v1"]["failed_checks"])


def test_control_regression_is_guarded_independently_of_v1_regression():
    candidate, control = _evaluation(3000), _evaluation(1000)
    candidate["by_city"]["PHL"]["road_boundary_quality"]["f1"] = .599
    control["by_city"]["PHL"]["road_boundary_quality"]["f1"] = .62
    result = compare_epoch(candidate, _evaluation(), _config(), control)
    assert result["qualified_vs_v1"] and result["benefit_vs_control"]["passes"]
    assert not result["safety_vs_control"]["passes"] and not result["experiment_win"]


@pytest.mark.parametrize("changed", ["order", "grid", "count", "support", "fabricated_metric", "city_sum", "native", "missing_hash"])
def test_incomparable_or_fabricated_evidence_fails_closed(changed):
    candidate, baseline = _evaluation(3000), _evaluation()
    if changed == "order":
        candidate["evaluated_ordered_ids_sha256"] = "f" * 64
    elif changed == "grid":
        candidate["reference_grid_binding_sha256"] = "f" * 64
    elif changed == "count":
        candidate["evaluated_sample_count"] -= 1
    elif changed == "support":
        matrix = np.array(candidate["by_city"]["DC"]["six_class_identification"]["confusion_matrix"])
        matrix[0, 0] += 1
        candidate["by_city"]["DC"] = _group(matrix)
        total = matrix + np.array(candidate["by_city"]["PHL"]["six_class_identification"]["confusion_matrix"])
        candidate["overall"]["six_class_identification"] = compute_multiclass_metrics(total, CLASS_NAMES)
    elif changed == "fabricated_metric":
        candidate["by_city"]["DC"]["six_class_identification"]["per_class"]["low_vegetation"]["f1"] = .99
    elif changed == "city_sum":
        candidate["overall"]["six_class_identification"] = deepcopy(baseline["overall"]["six_class_identification"])
    elif changed == "native":
        candidate["validation_native_size"] = 384
    else:
        del candidate["reference_grid_binding_sha256"]
    result = compare_epoch(candidate, baseline, _config(), _evaluation(1000))
    assert not result["safe_vs_v1"] and not result["safety_vs_v1"]["comparable"]
    assert not result["benefit_vs_v1"]["passes"] and not result["experiment_win"]


@pytest.mark.parametrize("config", [_config(control_epoch=3), {}, _config(comparison_epoch=3, control_epoch=3)])
def test_missing_mismatched_or_nonfinal_epoch_cannot_claim_primary_win(config):
    result = compare_epoch(_evaluation(3000), _evaluation(), config, _evaluation(1000))
    assert result["qualified_vs_v1"] and not result["pairing"]["passes"]
    assert not result["experiment_win"]


def test_same_epoch_exploratory_result_never_becomes_primary_win():
    result = compare_epoch(_evaluation(3000), _evaluation(),
                           _config(comparison_kind="exploratory", comparison_epoch=2, control_epoch=2), _evaluation(1000))
    assert result["matched_comparison_passes"] and not result["experiment_win"]


def test_v1_joint_gain_is_strict_while_control_joint_gain_is_inclusive():
    baseline, candidate, control = _evaluation(), _evaluation(3000), _evaluation(1000)
    preliminary = compare_epoch(candidate, baseline, _config(), control)
    v1_joint = preliminary["benefit_vs_v1"]["checks"][1]["value"]
    control_joint = preliminary["benefit_vs_control"]["checks"][1]["value"]
    result = compare_epoch(candidate, baseline, _config(
        benefit={"dc_ground_low_vegetation_mean_f1_gain": v1_joint},
        control_benefit={"dc_ground_low_vegetation_mean_f1_gain": control_joint}), control)
    assert not result["benefit_vs_v1"]["passes"]
    assert result["benefit_vs_control"]["passes"]


@pytest.mark.parametrize("settings", [{"class_f1": -1}, {"macro_f1": float("nan")}, {"class_f1": True}, {"typo": .01}])
def test_invalid_tolerances_cannot_silently_weaken_guards(settings):
    with pytest.raises(ValueError):
        make_floors(_evaluation(), settings)


def test_gate_does_not_mutate_input_evidence():
    baseline, candidate, control = _evaluation(), _evaluation(3000), _evaluation(1000)
    before = deepcopy((baseline, candidate, control))
    compare_epoch(candidate, baseline, _config(), control)
    assert (baseline, candidate, control) == before
