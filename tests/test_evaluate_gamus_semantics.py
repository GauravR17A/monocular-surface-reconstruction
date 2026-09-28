from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_gamus_semantics.py"
SPEC = importlib.util.spec_from_file_location("evaluate_gamus_semantics_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_semantic_metrics_include_all_three_classes_and_iou() -> None:
    confusion = np.array(
        [
            [5, 1, 0],
            [1, 3, 0],
            [0, 1, 4],
        ],
        dtype=np.int64,
    )

    result = MODULE.semantic_metrics(confusion)

    assert result["total_valid_pixels"] == 15
    assert np.isclose(result["accuracy"], 12 / 15)
    assert np.isclose(result["per_class"]["ground"]["f1"], 5 / 6)
    assert np.isclose(result["per_class"]["building"]["precision"], 3 / 5)
    assert np.isclose(result["per_class"]["building"]["recall"], 3 / 4)
    assert np.isclose(result["per_class"]["building"]["iou"], 3 / 6)
    assert np.isclose(result["per_class"]["vegetation"]["iou"], 4 / 5)
    assert result["confusion_matrix"] == confusion.tolist()


def test_comparison_reports_positive_improvement_in_percentage_points() -> None:
    baseline = MODULE.semantic_metrics(
        np.array([[5, 1, 0], [1, 4, 0], [0, 1, 3]], dtype=np.int64)
    )
    candidate = MODULE.semantic_metrics(
        np.array([[6, 0, 0], [0, 5, 0], [0, 0, 4]], dtype=np.int64)
    )

    comparison = MODULE.compare_semantic_metrics(baseline, candidate)

    assert comparison["accuracy_percentage_points"] > 0
    assert comparison["macro"]["f1"]["percentage_points"] > 0
    assert comparison["per_class"]["building"]["iou"]["percentage_points"] > 0
    assert comparison["relative_error_reduction"] > 0


def test_semantic_metrics_reject_wrong_matrix_shape() -> None:
    with np.testing.assert_raises(ValueError):
        MODULE.semantic_metrics(np.zeros((2, 2), dtype=np.int64))


def test_pointer_snapshot_accepts_utf8_bom_without_rewriting(tmp_path: Path) -> None:
    pointer = tmp_path / "showcase_checkpoint.txt"
    original = b"\xef\xbb\xbfC:\\models\\protected.pt\r\n"
    pointer.write_bytes(original)

    snapshot = MODULE._pointer_snapshot(pointer)

    assert snapshot["target"] == r"C:\models\protected.pt"
    assert pointer.read_bytes() == original
