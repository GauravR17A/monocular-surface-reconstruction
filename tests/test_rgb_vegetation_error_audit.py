import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("vegetation_audit", Path(__file__).resolve().parents[1] / "scripts/audit_rgb_vegetation_errors.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_pooled_counts_not_mean_of_image_scores():
    a = np.zeros((6, 6), dtype=np.int64)
    b = a.copy()
    a[4, 4] = 1
    b[4, 0] = 9
    rows = [{"sample_id": str(i), "six_class_identification": {"confusion_matrix": m.tolist()}} for i,m in enumerate([a,b])]
    result = audit.summarize(rows)
    assert result["recall"] == .1
    assert result["f1"] == pytest.approx(2/11)
    assert result["highest_missed_pixel_cases"][0]["sample_id"] == "1"
    with pytest.raises(ValueError):
        audit.summarize([rows[0], rows[0]])


def test_thin_and_interior_are_separate_and_unknown_not_zero():
    reference = np.zeros((11, 11), dtype=np.uint8)
    reference[1:10, 1:10] = 4
    prediction = reference.copy()
    prediction[1, 1:10] = 255
    result = audit.structure(reference, prediction)
    assert result["erosion_radius_pixels"]["1"]["reference_interior_pixels"] == 49
    assert result["erosion_radius_pixels"]["1"]["interior_recall"] == 1
    assert result["erosion_radius_pixels"]["1"]["edge_recall"] < 1
    assert result["erosion_radius_pixels"]["8"]["interior_recall"] is None


def test_absent_reference_is_not_claimed_perfect():
    empty = np.zeros((5,5), dtype=np.uint8)
    assert audit.structure(empty, empty)["erosion_radius_pixels"]["1"]["edge_recall"] is None
