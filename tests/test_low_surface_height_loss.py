import pytest
import torch
from msr.training.low_surface_height_loss import LowSurfaceProtectedHeightLoss
from msr.training.balanced_height_loss import SourceBalancedHeightLoss


def case(predicted, original, target, domain, valid=None):
    p = torch.tensor([[[predicted]]], dtype=torch.float32, requires_grad=True)
    r = torch.tensor([[[original]]], dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[[target]]], dtype=torch.float32)
    return {'height':p,'uncorrected_height':r}, {'height':y,
        'regression_mask':torch.tensor([[[valid if valid is not None else [True]*len(target)]]]),
        'domain_target':torch.tensor([[domain]]), 'source':['legacy'], 'landscape':['forest']}


def test_missing_labels_have_no_loss_or_gradient_and_baseline_detached():
    out,batch = case([5,99], [2,float('nan')], [1,float('nan')], [2,99], [True,False])
    loss,_ = LowSurfaceProtectedHeightLoss()(out,batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert out['height'].grad[0,0,0,1] == 0
    assert out['uncorrected_height'].grad is None


def test_regret_zero_for_improvement_positive_for_worse():
    loss = LowSurfaceProtectedHeightLoss()
    better,batch = case([1], [4], [0], [0])
    worse,_ = case([5], [4], [0], [0])
    assert loss(better,batch)[1]['low_surface_regret'] == 0
    assert loss(worse,batch)[1]['low_surface_regret'] > 0


def test_reference_height_not_zero_is_respected():
    out,batch = case([0], [1.5], [2], [2])
    loss,components = LowSurfaceProtectedHeightLoss()(out,batch)
    loss.backward()
    assert components['low_surface_regret'] > 0
    assert out['height'].grad.item() < 0  # increase prediction toward real 2m label


def test_disabled_terms_exactly_reproduce_original_loss():
    out,batch = case([5,7], [3,6], [0,1], [0,2])
    actual,_ = LowSurfaceProtectedHeightLoss(short_supervision_weight=0, low_surface_regret_weight=0)(out,batch)
    expected,_ = SourceBalancedHeightLoss()(out,batch)
    torch.testing.assert_close(actual,expected)


def test_short_group_not_diluted_by_more_tall_pixels():
    out,batch = case([4,12], [3,8], [1,10], [2,2])
    out2,batch2 = case([4]+[12]*20, [3]+[8]*20, [1]+[10]*20, [2]*21)
    loss=LowSurfaceProtectedHeightLoss()
    c1,c2 = loss(out,batch)[1],loss(out2,batch2)[1]
    torch.testing.assert_close(c1['short_vegetation_supervision'],c2['short_vegetation_supervision'])
    torch.testing.assert_close(c1['low_surface_regret'],c2['low_surface_regret'])


def test_all_invalid_finite_zero():
    out,batch = case([float('nan')], [float('nan')], [float('nan')], [99], [False])
    loss,_ = LowSurfaceProtectedHeightLoss()(out,batch)
    loss.backward()
    assert loss.item() == 0 and out['height'].grad.item() == 0


def test_finite_baseline_required_on_labelled_low_surfaces_only():
    out,batch = case([1], [float('nan')], [0], [0])
    with pytest.raises(ValueError, match='Nonfinite frozen'):
        LowSurfaceProtectedHeightLoss()(out,batch)


def test_unverified_sources_rejected():
    out,batch = case([1], [1], [0], [0])
    batch['source']=['gamus']
    with pytest.raises(ValueError, match='verified legacy'):
        LowSurfaceProtectedHeightLoss()(out,batch)
