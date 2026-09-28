import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("oem_all_six", SCRIPTS / "stage_oem_all_six_scenes.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_all_classes_must_have_independent_reference_support():
    labels = np.repeat(np.array([1, 8, 6, 4, 2, 5], np.uint8), 256).reshape(32, 48)
    valid = np.ones_like(labels, bool)
    assert module.label_support(labels, valid) == ([256]*6, True)
    valid[0, 0] = False
    assert module.label_support(labels, valid)[1] is False


def test_agriculture_and_unknown_do_not_manufacture_low_vegetation():
    labels = np.repeat(np.array([1, 8, 6, 4, 7, 5], np.uint8), 256).reshape(32, 48)
    counts, accepted = module.label_support(labels, np.ones_like(labels, bool))
    assert counts[4] == 0 and not accepted
    with pytest.raises(ValueError):
        module.label_support(np.full((2, 2), 9), np.ones((2, 2), bool))


@pytest.mark.parametrize("name", ["../dhaka_1.tif", "OpenEarthMap_wo_xBD/dhaka/labels/../bad.tif", "OpenEarthMap_wo_xBD/dhaka/labels/other_1.tif"])
def test_unsafe_member_rejected(name):
    with pytest.raises(ValueError):
        module.member_parts(name)


def test_supported_member():
    assert module.member_parts("OpenEarthMap_wo_xBD/dhaka/images/dhaka_1.tif")[-1] == "dhaka_1.tif"
