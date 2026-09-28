import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from types import SimpleNamespace
from affine import Affine

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("oem_evaluation", SCRIPTS / "evaluate_christchurch_rgb_v1.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_frozen_crosswalk_and_separate_agriculture():
    original = np.arange(9, dtype=np.uint8)
    assert module.remap(original).tolist() == [255, 0, 4, 0, 3, 5, 2, 255, 1]
    assert module.remap(original, True).tolist() == [255, 0, 4, 0, 3, 5, 2, 4, 1]
    assert original.tolist() == list(range(9))


@pytest.mark.parametrize("values", [np.array([9]), np.array([-1]), np.array([1.5])])
def test_unknown_label_encoding_is_not_silently_converted(values):
    with pytest.raises(ValueError):
        module.remap(values)


def test_native_matrix_retains_agriculture_and_distinct_ground_sources():
    source = np.arange(9, dtype=np.uint8).reshape(3, 3)
    prediction = np.zeros((3, 3), dtype=np.uint8)
    valid = np.ones_like(source, dtype=bool)
    matrix = module.native_confusion(prediction, source, valid)
    assert matrix.shape == (8, 6)
    assert matrix[:, 0].tolist() == [1]*8
    valid[2, 1] = False
    assert module.native_confusion(prediction, source, valid)[6].sum() == 0


def test_coverage_is_separate_from_perfect_prediction():
    tile = np.eye(6, dtype=np.int64)*10000
    assert module.summarize(tile, [tile])["comprehensive_six_class_coverage"] is False
    assert module.summarize(tile*5, [tile]*5)["comprehensive_six_class_coverage"] is True
    empty = np.zeros((6, 6), dtype=np.int64)
    result = module.summarize(empty, [empty])
    assert result["macro_f1"] == 0
    assert result["per_class"]["trees"]["reference_class_present"] is False


def test_bootstrap_uses_whole_tiles_and_is_reproducible():
    tile = np.eye(6, dtype=np.int64)*10
    first = module.bootstrap([tile, tile], resamples=20)
    assert first == module.bootstrap([tile, tile], resamples=20)
    assert first["macro_f1_95_percentile"] == [1, 1]
    assert first["macro_iou_95_percentile"] == [1, 1]
    with pytest.raises(ValueError):
        module.bootstrap([])


@pytest.mark.parametrize("change", [{"count": 4}, {"dtypes": ("uint16",)*3}, {"shape": (2048, 2048)}])
def test_bad_raster_headers_fail_before_pixel_decode(change):
    image = SimpleNamespace(count=3, dtypes=("uint8",)*3, shape=(1024, 1024), crs=None, transform=Affine.identity())
    label = SimpleNamespace(count=1, shape=(1024, 1024), crs=None, transform=Affine.identity())
    module.validate_raster_headers(image, label)
    for key, value in change.items():
        setattr(image, key, value)
    with pytest.raises(RuntimeError):
        module.validate_raster_headers(image, label)
