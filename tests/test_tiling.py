import numpy as np

from msr.inference.tiling import blend_weight, tile_starts


def test_tile_starts_cover_far_edge_without_duplicate():
    starts = tile_starts(length=1000, tile_size=384, overlap=96)

    assert starts[0] == 0
    assert starts[-1] == 616
    assert len(starts) == len(set(starts))


def test_small_image_uses_one_tile():
    assert tile_starts(length=200, tile_size=384, overlap=96) == [0]


def test_blend_weight_is_positive_and_center_weighted():
    weight = blend_weight(64, 64)

    assert weight.shape == (64, 64)
    assert np.all(weight > 0)
    assert weight[32, 32] > weight[0, 0]

