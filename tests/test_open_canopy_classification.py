import numpy as np
import pytest

from msr.data.open_canopy_classification import (
    IGN_LIDAR_HD_CLASS_NAMES,
    classification_histogram,
    preserve_classification_ids,
    rich_vegetation_target,
)


def test_preserves_original_ids_and_marks_only_invalid_as_nodata() -> None:
    values = np.asarray([[1, 2, 3], [4, 5, 67]], dtype=np.float32)
    valid = np.asarray([[True, False, True], [True, True, True]])

    preserved = preserve_classification_ids(values, valid)

    np.testing.assert_array_equal(
        preserved, np.asarray([[1, 255, 3], [4, 5, 67]], dtype=np.uint8)
    )
    assert classification_histogram(preserved) == {
        "1": 1,
        "3": 1,
        "4": 1,
        "5": 1,
        "67": 1,
    }


@pytest.mark.parametrize("bad", [1.5, -1.0, 255.0, np.nan])
def test_malformed_source_valid_ids_fail_closed(bad: float) -> None:
    with pytest.raises(ValueError):
        preserve_classification_ids(np.asarray([[bad]]), np.asarray([[True]]))


def test_rich_vegetation_target_retains_low_medium_high() -> None:
    values = np.asarray([[2, 3, 4, 5, 6, 9, 255]], dtype=np.uint8)

    target, valid = rich_vegetation_target(values)

    np.testing.assert_array_equal(valid, [[False, True, True, True, False, False, False]])
    np.testing.assert_array_equal(target, [[0, 0, 1, 2, 0, 0, 0]])
    assert IGN_LIDAR_HD_CLASS_NAMES[3] == "low_vegetation"
    assert IGN_LIDAR_HD_CLASS_NAMES[4] == "medium_vegetation"
    assert IGN_LIDAR_HD_CLASS_NAMES[5] == "high_vegetation"
    assert IGN_LIDAR_HD_CLASS_NAMES[65] == "artifact"
