import importlib.util
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
from msr.evaluation.metrics import StreamingRegressionMetrics

spec=importlib.util.spec_from_file_location('paired_dev',Path(__file__).resolve().parents[1]/'scripts/train_class_assisted_height.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_epoch_pairing_covers_all_training_records_and_reproduces_crops():
    records=[SimpleNamespace(sample_id=f'u{i}',landscape='urban') for i in range(450)]+[SimpleNamespace(sample_id=f'f{i}',landscape='forest') for i in range(600)]
    a=module.epoch_plan(records,1,42); b=module.epoch_plan(records,1,42)
    assert a==b and len(a)==600
    assert len({p[0].sample_id for p in a})==450
    assert len({p[1].sample_id for p in a})==600
    assert len({p[2] for p in a})==600
    assert module.epoch_plan(records,2,42)!=a


def test_evaluation_never_invents_missing_ground_or_water_height_support():
    acc=defaultdict(StreamingRegressionMetrics)
    target=np.full((5,5),25,dtype=np.float32); valid=np.zeros((5,5),dtype=bool); valid[1:4,1:4]=True
    domain=np.ones((5,5),dtype=np.int64); probs=np.zeros((6,5,5)); probs[1]=1
    module.add_groups(acc,target-2,target,valid,domain,'highbuild','city',probs,domain)
    assert acc['highbuild/domain/building'].count==9
    assert acc['highbuild/domain/ground'].count==0
    assert acc['highbuild/tall/building'].count==9
    assert acc['highbuild/boundary/building'].count==8
    assert acc['highbuild/class_geometry_disagreement'].count==0
    assert acc['highbuild/overall'].compute()['rmse_m']==2
