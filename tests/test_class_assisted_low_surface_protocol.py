import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('low_surface_protocol',ROOT/'scripts/train_class_assisted_height_low_surface.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config():
    return json.loads((ROOT/'configs/class_assisted_height_low_surface_v3.json').read_text())


def test_loss_only_scope_keeps_original_gates_and_training_budget():
    new=config()
    old=json.loads((ROOT/'configs/class_assisted_height_development_v1.json').read_text())
    module.validate_scope(new,old)
    assert new['initialization']=='fresh_protected_same_seed_both_arms'
    assert new['automatic_promotion'] is False


@pytest.mark.parametrize('field,value',[('epochs',3),('automatic_promotion',True),('test_used',True),
                                      ('gamus_height_evaluated',True),('initialization','candidate')])
def test_unapproved_scope_changes_fail_closed(field,value):
    new=config(); old=copy.deepcopy(new)
    new[field]=value
    with pytest.raises(ValueError):
        module.validate_scope(new,old)


def test_loosening_gates_rejected():
    new=config(); old=copy.deepcopy(new)
    new['guards']['rmse_m']=5
    with pytest.raises(ValueError,match='guards'):
        module.validate_scope(new,old)


def test_resume_rejects_source_change(tmp_path):
    new=config(); new['output_root']=str(tmp_path)
    run=tmp_path/'run'; run.mkdir()
    (run/'binding.json').write_text(json.dumps({'code':{'loss':'old'}}))
    with pytest.raises(ValueError,match='binding mismatch'):
        module.assert_resume(run,new,{'code':{'loss':'new'}})
