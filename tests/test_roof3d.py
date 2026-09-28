import numpy as np
import pytest

from msr.data.roof3d import decode_roof_mask, roof_instance_targets


def rle(mask):
    values = np.asarray(mask).ravel(order="F")
    counts, state, count = [], False, 0
    for value in values:
        if bool(value) != state:
            counts.append(count)
            state = bool(value)
            count = 0
        count += 1
    counts.append(count)
    return {"size": list(mask.shape), "counts": counts}


def annotation(mask, identifier=1):
    return {"id": identifier, "iscrowd": 0, "segmentation": [rle(mask)]}


def test_rle_order_and_list_wrapper():
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    np.testing.assert_array_equal(decode_roof_mask([rle(mask)], mask.shape), mask)
    np.testing.assert_array_equal(decode_roof_mask(rle(mask), mask.shape), mask)


@pytest.mark.parametrize("counts", [[-1, 7], [2, 2], "compressed", [True, 5], [1.2, 4.8]])
def test_invalid_rle_is_not_silently_background(counts):
    with pytest.raises(ValueError):
        decode_roof_mask({"size": [2, 3], "counts": counts}, (2, 3))


def test_dimensions_must_match():
    with pytest.raises(ValueError):
        decode_roof_mask({"size": [3, 2], "counts": [6]}, (2, 3))


def test_unknown_background_and_missing_height_are_not_zero_targets():
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 3:9] = True
    out = roof_instance_targets([annotation(mask)], np.ones_like(mask))
    np.testing.assert_array_equal(out["classification_valid"], mask)
    assert not out["height_valid"].any()
    assert out["boundary"][3, 5]
    assert not out["boundary"][5, 5]
    assert not out["boundary_valid"][0, 0]


def test_overlaps_and_invalid_image_pixels_are_ignored():
    a = np.zeros((10, 10), dtype=bool)
    b = a.copy()
    a[2:7, 2:7] = True
    b[5:9, 5:9] = True
    valid = np.ones_like(a)
    valid[3, 3] = False
    out = roof_instance_targets([annotation(a), annotation(b, 2)], valid)
    assert not out["classification_valid"][5:7, 5:7].any()
    assert not out["classification_valid"][3, 3]
    assert not out["boundary_valid"][3, 4]
    assert out["ambiguous_overlap"].sum() == 4


def test_duplicate_ids_fail_instead_of_merging_roof_parts():
    ann = annotation(np.ones((5, 5), dtype=bool))
    with pytest.raises(ValueError, match="Duplicate"):
        roof_instance_targets([ann, ann], np.ones((5, 5), dtype=bool))
