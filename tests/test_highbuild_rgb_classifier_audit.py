import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("highbuild_rgb_audit", Path(__file__).resolve().parents[1] / "scripts/audit_highbuild_rgb_classifier.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_unknown_background_never_becomes_false_positive_or_accuracy():
    footprint = np.zeros((12, 12), bool)
    footprint[2:10, 2:10] = True
    pred = np.ones((12, 12), np.uint8)  # Deliberately predicts building everywhere.
    result = audit.summarize_building_support(pred, footprint, np.ones_like(footprint))
    assert result["building_pixel_recall"] == 1
    assert result["building_precision"] is None
    assert result["six_class_macro_f1"] is None
    assert result["interior_reference_pixels"] == 16


def test_pooled_recall_and_missing_support():
    mask = np.ones((10, 10), bool)
    a = audit.summarize_building_support(np.ones((10, 10), np.uint8), mask, mask)
    b = audit.summarize_building_support(np.zeros((2, 2), np.uint8), mask[:2, :2], mask[:2, :2])
    pooled = audit.aggregate([a, b])
    assert pooled["building_pixel_recall"] == pytest.approx(100/104)
    assert pooled["building_confusion_fraction"]["ground"] == pytest.approx(4/104)
    empty = audit.summarize_building_support(np.zeros((10, 10), np.uint8), ~mask, mask)
    assert empty["building_pixel_recall"] is None


def test_optical_validity_is_independent_and_invalid_class_rejected():
    footprint = np.ones((10, 10), bool)
    valid = footprint.copy(); valid[:5] = False
    pred = np.ones((10, 10), np.uint8); pred[:5] = 255
    assert audit.summarize_building_support(pred, footprint, valid)["building_reference_pixels"] == 50
    with pytest.raises(ValueError):
        audit.summarize_building_support(pred, footprint, footprint)


def test_object_coverage_is_not_claimed_as_count_accuracy():
    result = audit.summarize_objects([{"building_fraction": .9}, {"building_fraction": .05}])
    assert result["fraction_at_least_80_percent_covered"] == .5
    assert result["fraction_less_than_10_percent_covered"] == .5
    assert "not matched-object" in result["note"]
