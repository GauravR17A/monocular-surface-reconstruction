import numpy as np
import pytest
from msr.training.height_band_sampling import height_masks,choose_crop,balanced_source_plan


def test_bands_do_not_invent_missing_ground_or_vegetation_labels():
    target=np.array([[0,2,2.1,15,19,20,np.nan]])
    valid=np.array([[True]*6+[False]])
    domain=np.array([[0,2,2,2,1,1,99]])
    masks=height_masks(target,valid,domain)
    assert all(int(m.sum())==1 for m in masks.values())
    assert not any(m[0,-1] for m in masks.values())


def test_crop_anchor_is_supported_but_does_not_change_labels():
    target=np.zeros((512,512)); domain=np.zeros_like(target,dtype=int); valid=np.zeros_like(target,dtype=bool)
    target[450,450]=30; domain[450,450]=1; valid[450,450]=True
    original=target.copy()
    crop=choose_crop(target,valid,domain,'tall_building',12)
    assert crop['row']<=450<crop['row']+384 and crop['col']<=450<crop['col']+384
    np.testing.assert_array_equal(target,original)
    with pytest.raises(ValueError,match='no height support'):
        choose_crop(target,valid,domain,'short_building',12)


def test_two_epoch_coverage_reproducibility_and_exact_band_budget():
    items=[{'sample_id':str(i),'supported_pixels':1000,
            'counts':{'ground':200,'short_vegetation':100,'medium_vegetation':300,'tall_vegetation':400}} for i in range(600)]
    first=balanced_source_plan(items,'forest',1,42)
    assert first==balanced_source_plan(items,'forest',1,42)
    second=balanced_source_plan(items,'forest',2,42)
    assert len(first)==600
    assert len({r['sample_id'] for r in first+second})==600
    assert sum(r['band']=='coverage' for r in first)==300
    for band in ('ground','short_vegetation','medium_vegetation','tall_vegetation'):
        assert sum(r['band']==band for r in first)==75


def test_missing_height_band_fails_closed():
    items=[{'sample_id':'one','supported_pixels':1000,'counts':{'short_building':1000,'tall_building':0}}]
    with pytest.raises(ValueError,match='eligible training support'):
        balanced_source_plan(items,'urban',1,42)
