"""Predeclared safety and benefit criteria; no automatic app promotion."""
from __future__ import annotations
import math
from msr.models.rgb_segmenter import CLASS_NAMES

def compare(candidate, baseline, limits):
    checks = {}
    deltas = {}
    for domain in ("gamus","oem"):
        current, previous = candidate[domain], baseline[domain]
        if current["evaluated_ordered_ids_sha256"] != previous["evaluated_ordered_ids_sha256"] or current["reference_grid_binding_sha256"] != previous["reference_grid_binding_sha256"]:
            raise ValueError("Candidate and baseline evaluated different images or reference grids")
        if set(current["by_city"]) != set(previous["by_city"]):
            raise ValueError("Region groups changed")
        for region in ("overall",*current["by_city"]):
            a = current["overall"] if region=="overall" else current["by_city"][region]
            b = previous["overall"] if region=="overall" else previous["by_city"][region]
            score, old = a["six_class_identification"], b["six_class_identification"]
            for cls in CLASS_NAMES:
                # A category absent in this reference region has no recall/F1 evidence.
                support = old["per_class"][cls].get("support_pixels", old["per_class"][cls].get("support",0))
                if not support:
                    continue
                delta = score["per_class"][cls]["f1"]-old["per_class"][cls]["f1"]
                key = f"{domain}/{region}/{cls}_f1"
                deltas[key] = delta
                checks[key] = math.isfinite(delta) and delta >= -limits[f"{domain}_class_f1_max_drop"]
            delta = score["macro_f1"]-old["macro_f1"]
            deltas[f"{domain}/{region}/macro_f1"] = delta
            if domain=="gamus":
                checks[f"{domain}/{region}/macro_f1"] = math.isfinite(delta) and delta >= -limits["gamus_macro_f1_max_drop"]
            # Helpers expose the same quality payloads for both sources.
            road_delta = a["road_boundary_quality"]["f1"]-b["road_boundary_quality"]["f1"]
            checks[f"{domain}/{region}/road_boundary"] = road_delta >= -limits["road_boundary_f1_max_drop"]
            water_delta = a["water_dark_pixel_proxy"]["false_water_rate_on_dark_non_water"]-b["water_dark_pixel_proxy"]["false_water_rate_on_dark_non_water"]
            checks[f"{domain}/{region}/dark_false_water"] = water_delta <= limits["dark_non_water_false_water_rate_max_increase"]
    regions = sorted(candidate["oem"]["by_city"])
    region_delta = sum(deltas[f"oem/{r}/macro_f1"] for r in regions)/len(regions)
    weak_delta = sum(deltas[f"oem/overall/{c}_f1"] for c in ("ground","roads","low_vegetation"))/3
    benefit = region_delta >= limits["oem_macro_f1_min_gain"] and weak_delta >= limits["oem_weak_class_mean_f1_min_gain"]
    return {"safety_pass":all(checks.values()),"benefit_pass":benefit,"eligible_development_candidate":all(checks.values()) and benefit,
            "failed_checks":[k for k,v in checks.items() if not v],"checks":checks,"deltas":deltas,
            "oem_region_macro_f1_gain":region_delta,"oem_weak_class_mean_f1_gain":weak_delta,"app_promotion":False}
