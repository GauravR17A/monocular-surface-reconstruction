from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('semantic_expert_runner',ROOT/'scripts/train_semantic_expert_height.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_recipe_preserves_scope_and_guards():
    config=json.loads((ROOT/'configs/semantic_expert_height_v1.json').read_text())
    parent=json.loads((ROOT/'configs/height_band_sampling_v1.json').read_text())
    runner.validate_recipe(config,parent)
    for key,value in [('epochs',8),('automatic_promotion',True),('test_used',True),
                      ('gamus_height_evaluated',True),('sampler','height_band_balanced'),('loss','low_surface')]:
        bad=deepcopy(config);bad[key]=value
        with pytest.raises(ValueError): runner.validate_recipe(bad,parent)
    bad=deepcopy(config);bad['guards']['rmse_m']=.5
    with pytest.raises(ValueError,match='safety'):runner.validate_recipe(bad,parent)


def test_training_uses_actual_arm_and_identical_original_plans():
    # Guard a particularly costly regression: accidentally hardcoding uniform
    # as in the preceding sampler-only experiment would negate this experiment.
    source=(ROOT/'scripts/train_semantic_expert_height.py').read_text()
    assert "plan=plans['control'][str(epoch)]" in source
    assert 'output=common.forward(model,batch,arm)' in source
    assert "common.forward=lambda" not in source
    assert 'model.protected.parameters()' in source
