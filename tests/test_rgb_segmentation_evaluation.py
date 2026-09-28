from copy import deepcopy

import numpy as np
import pytest
import torch

from msr.evaluation.rgb_segmentation import CLASS_NAMES, classification_acceptance, evaluate_rgb_segmentation


class Perfect(torch.nn.Module):
    def forward(self, image):
        target = (image[:, 0] * 5).round().long()
        return {"logits": torch.nn.functional.one_hot(target, 6).permute(0, 3, 1, 2).float() * 10}


def _batch():
    labels = torch.tensor([[[0, 1, 2], [3, 4, 5], [0, 1, 2]]] * 2)
    return {"image": (labels[:, None].float() / 5).repeat(1, 3, 1, 1),
            "labels": labels, "image_valid_mask": torch.ones_like(labels, dtype=torch.bool),
            "classification_valid_mask": torch.ones_like(labels, dtype=torch.bool),
            "dark_pixel_proxy_mask": labels == 0, "sample_id": ["DC_val", "PHL_val"], "city": ["DC", "PHL"]}


def _evaluate(batch=None):
    return evaluate_rgb_segmentation(Perfect(), [_batch() if batch is None else batch], "cpu",
                                      expected_ids=["DC_val", "PHL_val"], native_size=3)


def _floors():
    return {"six_class_macro_f1_min": .6, "road_boundary_f1_min": .2,
            "dark_non_water_false_water_rate_max": .1,
            **{f"{name}_f1_min": .4 for name in CLASS_NAMES}}


def test_full_sixclass_metrics_native_and_per_city():
    result = _evaluate()
    assert result["evaluated_sample_count"] == 2
    assert result["overall"]["six_class_identification"]["macro_f1"] == 1
    assert result["overall"]["six_class_identification"]["total_valid_pixels"] == 18
    assert result["overall"]["road_boundary_quality"]["tolerance_pixels"] == 2
    assert result["by_city"]["DC"]["six_class_identification"]["total_valid_pixels"] == 9
    assert result["overall"]["water_dark_pixel_proxy"]["is_shadow_proxy_not_ground_truth"]
    assert not result["height_evaluated"]


def test_independent_masks_and_callback_and_restore_training():
    batch = _batch()
    batch["image_valid_mask"][0, 0, 0] = False
    batch["dark_pixel_proxy_mask"][0, 0, 0] = False
    batch["classification_valid_mask"][1, 0, 1] = False
    model = Perfect().train()
    updates = []
    result = evaluate_rgb_segmentation(model, [batch], "cpu", expected_ids=batch["sample_id"], native_size=3, progress_callback=updates.append)
    assert result["overall"]["six_class_identification"]["total_valid_pixels"] == 16
    assert model.training
    assert updates == [{"completed_batches": 1, "total_batches": 1}]


def test_wrong_order_duplicate_and_nonnative_rejected():
    batch = _batch()
    with pytest.raises(ValueError, match="fixed order"):
        evaluate_rgb_segmentation(Perfect(), [batch], "cpu", expected_ids=["PHL_val", "DC_val"], native_size=3)
    with pytest.raises(ValueError, match="exactly once"):
        evaluate_rgb_segmentation(Perfect(), [batch, batch], "cpu", expected_ids=batch["sample_id"], native_size=3)
    with pytest.raises(ValueError, match="full-native"):
        evaluate_rgb_segmentation(Perfect(), [batch], "cpu", expected_ids=batch["sample_id"])


def test_bad_logits_fail_instead_of_changing_support():
    class Bad(Perfect):
        def forward(self, image):
            out = super().forward(image)
            out["logits"][0, 0, 0, 0] = float("nan")
            return out
    with pytest.raises(FloatingPointError):
        evaluate_rgb_segmentation(Bad(), [_batch()], "cpu", expected_ids=["DC_val", "PHL_val"], native_size=3)


def test_acceptance_has_all_class_and_city_guards():
    result = _evaluate()
    floors = _floors()
    accepted = classification_acceptance(result, {"v3": result}, floors, {city: floors for city in ["DC", "PHL"]})
    assert accepted["passes"] and accepted["comparable_to_fixed_v3"]
    assert len(accepted["checks"]) == 27
    assert not accepted["promotion_performed"]
    missing = dict(floors)
    del missing["ground_f1_min"]
    refused = classification_acceptance(result, result, missing, {city: floors for city in ["DC", "PHL"]})
    assert not refused["passes"] and not refused["comparable_to_fixed_v3"]


@pytest.mark.parametrize("change", ["support", "ids", "road", "dark", "city", "fabricated_metric"])
def test_acceptance_rejects_incomparable_or_fabricated_evidence(change):
    result = _evaluate()
    other = deepcopy(result)
    if change == "support":
        other["overall"]["six_class_identification"]["per_class"]["ground"]["support_pixels"] += 1
    elif change == "ids":
        other["evaluated_ids_sha256"] = "wrong"
    elif change == "road":
        other["overall"]["road_boundary_quality"]["reference_boundary_pixels"] += 1
    elif change == "dark":
        other["overall"]["water_dark_pixel_proxy"]["dark_non_water_pixels"] += 1
    elif change == "city":
        del other["by_city"]["DC"]
    elif change == "fabricated_metric":
        other["overall"]["six_class_identification"]["macro_f1"] = .999
    floors = _floors()
    refused = classification_acceptance(other, result, floors, {city: floors for city in ["DC", "PHL"]})
    assert not refused["passes"] and not refused["comparable_to_fixed_v3"]
