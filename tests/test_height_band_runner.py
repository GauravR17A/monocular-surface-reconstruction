from collections import defaultdict
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from msr.evaluation.metrics import StreamingRegressionMetrics

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('band_runner',ROOT/'scripts/train_height_band_sampling.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def test_new_groups_preserve_all_original_groups_and_add_short_medium():
    height=np.array([[10.,25.,1.,8.,20.]])
    domain=np.array([[1,1,2,2,2]])
    valid=np.ones_like(height,dtype=bool)
    probabilities=np.ones((6,1,5))/6
    original=defaultdict(StreamingRegressionMetrics)
    expanded=defaultdict(StreamingRegressionMetrics)
    args=(height-1,height,valid,domain,'suite','region',probabilities,domain)
    runner.dev.add_groups(original,*args)
    runner.extended_groups(runner.dev.add_groups)(expanded,*args)
    for key,metrics in original.items():
        assert expanded[key].count==metrics.count
        if metrics.count:
            assert expanded[key].compute()==metrics.compute()
    assert expanded['suite/height_band/short_building'].count==1
    assert expanded['suite/height_band/medium_vegetation'].count==1


def test_pinned_crop_never_masks_other_valid_heights():
    sample={'image':torch.zeros(3,384,384),'height':torch.full((1,384,384),30.),
            'regression_mask':torch.ones(1,384,384,dtype=torch.bool),
            'domain_target':torch.ones(384,384,dtype=torch.long)}
    sample['height'][0,0,0]=3
    exposure=runner.count_bands(sample['height'][0].numpy(),sample['regression_mask'][0].numpy(),sample['domain_target'].numpy())
    draw={'crop':{'row':0,'col':0,'size':384},'exposure':exposure,'supported_pixels':384*384}
    crop=runner.crop_sample(sample,draw)
    assert crop['height'][0,0,0]==3
    assert crop['regression_mask'].all()
    draw['supported_pixels']-=1
    with pytest.raises(ValueError,match='support changed'):
        runner.crop_sample(sample,draw)
