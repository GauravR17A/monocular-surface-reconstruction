import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('prior_replay',Path(__file__).resolve().parents[1]/'scripts/audit_training_prior_replay.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_replay_pass_is_numerical_not_correlation_alone():
    array=np.arange(16,dtype=np.float32).reshape(4,4)/16
    valid=np.ones_like(array,dtype=bool)
    assert module.compare_prior(array,array,valid)['replay_pass']
    assert not module.compare_prior(array,array+.1,valid)['replay_pass']
    with pytest.raises(ValueError): module.compare_prior(array,array,np.zeros_like(valid))


def test_selection_is_fixed_by_city_and_id_not_quality():
    rows=[{'target_kind':'building_height','region':city,'sample_id':f'{city}{i}'} for city in ['b','a'] for i in [3,2,1]]
    assert [r['sample_id'] for r in module.select_train_records(rows)]==['a1','a2','b1','b2']
