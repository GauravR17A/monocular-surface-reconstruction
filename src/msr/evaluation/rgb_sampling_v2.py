"""Frozen-V1 and matched-control gates for the RGB crop-sampling experiment.

Scores and tolerances are fractions, not percentage points. The primary causal
comparison is candidate versus control at continuation epoch four. The caller
must supply ``comparison_epoch`` and ``control_epoch`` when a control is passed;
different epochs fail closed. An exploratory comparison must explicitly set
``comparison_kind="exploratory"`` and cannot report a primary experiment win.

No function reads data, loads models, changes checkpoints, or promotes a model.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

import numpy as np

from .rgb_segmentation import CLASS_NAMES, _validate_group, classification_acceptance


DEFAULT_TOLERANCES = {
    "class_f1": 0.01,
    "macro_f1": 0.005,
    "road_boundary_f1": 0.01,
    "dark_non_water_false_water_rate": 0.001,
}
DEFAULT_BENEFIT = {
    "dc_low_vegetation_f1_gain": 0.05,
    "dc_ground_low_vegetation_mean_f1_gain": 0.025,
}
DEFAULT_CONTROL_BENEFIT = {
    "dc_low_vegetation_f1_gain": 0.02,
    "dc_ground_low_vegetation_mean_f1_gain": 0.01,
}
_CITIES = ("DC", "PHL")
_IDENTITY_KEYS = (
    "evaluated_sample_count", "evaluated_samples_by_city",
    "evaluated_ids_sha256", "evaluated_ordered_ids_sha256",
    "evaluated_ids_by_city_sha256", "reference_grid_binding_sha256",
    "validation_native_size",
)


def _settings(supplied: Mapping[str, Any] | None, defaults: Mapping[str, float]) -> dict[str, float]:
    supplied = {} if supplied is None else supplied
    if not isinstance(supplied, Mapping) or set(supplied) - set(defaults):
        raise ValueError(f"Expected only these setting keys: {sorted(defaults)}")
    result = dict(defaults)
    for key, value in supplied.items():
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a finite fraction in [0,1]")
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a finite fraction in [0,1]")
        result[key] = value
    return result


def _validate_evaluation(evaluation: Mapping[str, Any], role: str) -> None:
    if evaluation.get("validation_native_size") != 1024:
        raise ValueError(f"{role}: full-native 1024px validation is required")
    if set(evaluation["by_city"]) != set(_CITIES):
        raise ValueError(f"{role}: both DC and PHL are required")
    counts = evaluation["evaluated_samples_by_city"]
    if set(counts) != set(_CITIES) or any(type(n) is not int or n <= 0 for n in counts.values()):
        raise ValueError(f"{role}: invalid city sample counts")
    if type(evaluation["evaluated_sample_count"]) is not int or sum(counts.values()) != evaluation["evaluated_sample_count"]:
        raise ValueError(f"{role}: city sample counts do not sum to the overall count")
    digests = [evaluation[key] for key in (
        "evaluated_ids_sha256", "evaluated_ordered_ids_sha256", "reference_grid_binding_sha256")]
    city_digests = evaluation["evaluated_ids_by_city_sha256"]
    if set(city_digests) != set(_CITIES):
        raise ValueError(f"{role}: missing city ID hashes")
    digests.extend(city_digests.values())
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in digests):
        raise ValueError(f"{role}: missing or malformed evaluation identity hash")
    overall = evaluation["overall"]
    _validate_group(overall, f"{role}.overall")
    matrices = []
    for city in _CITIES:
        group = evaluation["by_city"][city]
        _validate_group(group, f"{role}.{city}")
        matrices.append(np.asarray(group["six_class_identification"]["confusion_matrix"], dtype=np.int64))
    if not np.array_equal(np.asarray(overall["six_class_identification"]["confusion_matrix"]), sum(matrices)):
        raise ValueError(f"{role}: overall confusion matrix does not equal the sum of city matrices")


def make_floors(v1_evaluation: Mapping[str, Any], tolerancesconfig: Mapping[str, Any] | None = None
                ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return all 27 guards, anchored to the supplied full-native reference.

    ``tolerancesconfig`` is a direct mapping with the DEFAULT_TOLERANCES keys;
    omitted keys use defaults. This also accepts a matched control as reference.
    The reference's reported multiclass metrics are recomputed from its matrix.
    """
    tolerances = _settings(tolerancesconfig, DEFAULT_TOLERANCES)
    _validate_evaluation(v1_evaluation, "reference")

    def group_floors(group):
        six = group["six_class_identification"]
        return {
            **{f"{name}_f1_min": max(0.0, six["per_class"][name]["f1"] - tolerances["class_f1"])
               for name in CLASS_NAMES},
            "six_class_macro_f1_min": max(0.0, six["macro_f1"] - tolerances["macro_f1"]),
            "road_boundary_f1_min": max(0.0, group["road_boundary_quality"]["f1"] - tolerances["road_boundary_f1"]),
            "dark_non_water_false_water_rate_max": min(1.0, group["water_dark_pixel_proxy"]["false_water_rate_on_dark_non_water"]
                                                        + tolerances["dark_non_water_false_water_rate"]),
        }

    return group_floors(v1_evaluation["overall"]), {
        city: group_floors(v1_evaluation["by_city"][city]) for city in _CITIES}


def _safety(candidate, reference, tolerances, label):
    try:
        _validate_evaluation(candidate, "candidate")
        _validate_evaluation(reference, label)
        for key in _IDENTITY_KEYS:
            if candidate[key] != reference[key]:
                raise ValueError(f"Candidate/{label} {key} differs")
        overall, cities = make_floors(reference, tolerances)
        gate = classification_acceptance(candidate, reference, overall, cities)
        # The reused helper's historical V3 field names must not mislabel V1 or
        # the new continuation control in the experiment report.
        gate["reference"] = label
        gate["comparable"] = gate.pop("comparable_to_fixed_v3")
        gate["candidate_minus_reference"] = gate.pop("candidate_minus_v3")
        gate["scope"] = "DC/PHL development safety gate; no production promotion"
        return gate
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return {"passes": False, "eligible": False, "comparable": False, "reference": label,
                "failed_checks": [f"Incomparable or invalid evaluation: {error}"], "checks": [],
                "candidate_minus_reference": {}, "promotion_performed": False,
                "scope": "DC/PHL development safety gate; no production promotion"}


def _benefit(candidate, reference, limits, comparable, label):
    if not comparable:
        return {"passes": False, "reference": label, "checks": [],
                "failed_checks": ["Benefit cannot be assessed on incomparable evidence"]}
    current = candidate["by_city"]["DC"]["six_class_identification"]["per_class"]
    previous = reference["by_city"]["DC"]["six_class_identification"]["per_class"]
    gains = {
        "dc_low_vegetation_f1_gain": current["low_vegetation"]["f1"] - previous["low_vegetation"]["f1"],
        "dc_ground_low_vegetation_mean_f1_gain": sum(
            current[name]["f1"] - previous[name]["f1"] for name in ("ground", "low_vegetation")) / 2,
    }
    checks = []
    for name, limit in limits.items():
        strict = label == "frozen_rgb_v1" and name == "dc_ground_low_vegetation_mean_f1_gain"
        checks.append({"metric": name, "value": gains[name], "threshold": limit,
                       "comparison": ">" if strict else ">=",
                       "passes": gains[name] > limit if strict else gains[name] >= limit - 1e-12})
    return {"passes": all(check["passes"] for check in checks), "reference": label,
            "checks": checks, "failed_checks": [check["metric"] for check in checks if not check["passes"]]}


def compare_epoch(candidate_eval: Mapping[str, Any], v1_eval: Mapping[str, Any],
                  config: Mapping[str, Any], control_eval: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Separate safety, target improvement, and the final matched-control win.

    Config keys: ``tolerances``, ``benefit``, ``control_benefit``,
    ``comparison_epoch``, ``control_epoch``, ``primary_epoch`` (default 4), and
    ``comparison_kind`` (``primary`` by default; or ``exploratory``).
    Same-epoch pairing is mandatory even for exploratory comparisons. Primary
    wins additionally require both epochs to equal the declared primary epoch.
    No control means the causal comparison is unavailable, never a win.
    """
    tolerances = _settings(config.get("tolerances"), DEFAULT_TOLERANCES)
    benefit_limits = _settings(config.get("benefit"), DEFAULT_BENEFIT)
    control_limits = _settings(config.get("control_benefit"), DEFAULT_CONTROL_BENEFIT)
    kind = config.get("comparison_kind", "primary")
    if kind not in {"primary", "exploratory"}:
        raise ValueError("comparison_kind must be primary or exploratory")
    primary_epoch = config.get("primary_epoch", 4)
    if type(primary_epoch) is not int or primary_epoch <= 0:
        raise ValueError("primary_epoch must be a positive integer")
    safety_v1 = _safety(candidate_eval, v1_eval, tolerances, "frozen_rgb_v1")
    benefit_v1 = _benefit(candidate_eval, v1_eval, benefit_limits, safety_v1["comparable"], "frozen_rgb_v1")
    safety_control = benefit_control = None
    pair_failures = []
    candidate_epoch, control_epoch = config.get("comparison_epoch"), config.get("control_epoch")
    if control_eval is not None:
        safety_control = _safety(candidate_eval, control_eval, tolerances, "matched_continuation_control")
        benefit_control = _benefit(candidate_eval, control_eval, control_limits,
                                   safety_control["comparable"], "matched_continuation_control")
        if (type(candidate_epoch) is not int or candidate_epoch <= 0
                or type(control_epoch) is not int or control_epoch <= 0):
            pair_failures.append("Explicit positive comparison_epoch and control_epoch are required")
        elif candidate_epoch != control_epoch:
            pair_failures.append("Candidate and control must be compared at the same continuation epoch")
        if kind == "primary" and (candidate_epoch != primary_epoch or control_epoch != primary_epoch):
            pair_failures.append(f"Primary comparison requires both arms at continuation epoch {primary_epoch}")
    else:
        pair_failures.append("Matched continuation control is unavailable")
    pair_passes = control_eval is not None and not pair_failures
    qualified_v1 = safety_v1["passes"] and benefit_v1["passes"]
    matched_win = bool(qualified_v1 and pair_passes and safety_control["passes"] and benefit_control["passes"])
    experiment_win = matched_win and kind == "primary"
    return {
        "schema": "msr.rgb_sampling_v2_epoch_comparison.v1",
        "metric_units": "fractions", "comparison_kind": kind,
        "comparison_epoch": candidate_epoch, "control_epoch": control_epoch, "primary_epoch": primary_epoch,
        "tolerances": tolerances, "benefit_thresholds": benefit_limits, "control_benefit_thresholds": control_limits,
        "safety_vs_v1": safety_v1, "benefit_vs_v1": benefit_v1,
        "safety_vs_control": safety_control, "benefit_vs_control": benefit_control,
        "pairing": {"passes": pair_passes, "failed_checks": pair_failures},
        "safe_vs_v1": safety_v1["passes"], "qualified_vs_v1": qualified_v1,
        "matched_comparison_passes": matched_win, "experiment_win": experiment_win,
        "passes": experiment_win, "development_only": True,
        "promotion_eligible": False, "promotion_performed": False,
    }
