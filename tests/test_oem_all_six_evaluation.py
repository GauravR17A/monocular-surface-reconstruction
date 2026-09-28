import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("oem_mixed_eval", SCRIPTS / "evaluate_oem_all_six_scenes.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exhaustive_labels_penalize_invented_buildings():
    target = np.array([[0,1,2],[3,4,5]], np.uint8)
    pred = np.ones_like(target)
    matrix = module.independent_confusion(pred, target, np.ones_like(target, bool))
    score = module.independent_scores(matrix)
    assert score["per_class"]["buildings"]["recall"] == 1
    assert score["per_class"]["buildings"]["precision"] == pytest.approx(1/6)
    assert score["macro_f1"] < .05


def test_dataset_text_correction_preserves_metrics():
    value = {"nested":{"description":"GAMUS has no shadow class", "value":.02}}
    assert module.oem_descriptions(value) == {"nested":{"description":"OpenEarthMap has no shadow class", "value":.02}}
    assert value["nested"]["description"].startswith("GAMUS")


def test_all_six_perfect_prediction_is_distinct_from_invalid_pixels():
    target = np.array([[0,1,2],[3,4,5]], np.uint8)
    valid = np.ones_like(target, bool); valid[0,0] = False
    pred = target.copy(); pred[0,0] = 5
    score = module.independent_scores(module.independent_confusion(pred, target, valid))
    assert score["accuracy"] == 1
    assert score["total_valid_pixels"] == 5
    assert score["per_class"]["ground"]["support_pixels"] == 0
