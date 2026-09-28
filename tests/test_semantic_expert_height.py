from copy import deepcopy
import pytest
import torch
from msr.models.semantic_expert_height import SemanticExpertHeightNet, mix_expert_corrections
from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.height_net import HeightNet


def test_native_soft_router_has_correct_class_effect_and_ignores_invalid():
    logits = torch.zeros(1,6,2,3,requires_grad=True)
    with torch.no_grad(): logits[:,1] = .1; logits[:,5] = -.2
    p = torch.zeros_like(logits); p[:,1] = .75; p[:,5] = .25
    p.requires_grad_()
    valid = torch.ones(1,1,2,3,dtype=torch.bool); valid[:,:,-1,-1] = False
    out = mix_expert_corrections(logits,p,valid,arm='predicted',maximum_correction_m=80)
    expected = 80*(.75*torch.tanh(torch.tensor(.1))+.25*torch.tanh(torch.tensor(-.2)))
    assert torch.allclose(out[0,0,0,0],expected)
    assert out[0,0,-1,-1] == 0
    out.sum().backward()
    assert p.grad is None
    assert logits.grad[:,1].abs().sum() > 0 and logits.grad[:,5].abs().sum() > 0
    assert logits.grad[:,0].abs().sum() == 0
    uniform = mix_expert_corrections(logits,p,valid,arm='uniform',maximum_correction_m=80)
    permuted = mix_expert_corrections(logits,p.flip(1),valid,arm='uniform',maximum_correction_m=80)
    assert torch.equal(uniform,permuted)


def test_zero_start_protected_semantics_and_actual_class_routing_after_learning():
    torch.manual_seed(4)
    base = DomainGatedSurfaceNet(HeightNet('resnet18',pretrained=False,decoder_channels=(32,16,8,4,4)),
                                 hidden_channels=8,fine_semantic_classes=6)
    model = SemanticExpertHeightNet(base,hidden_channels=8).eval()
    frozen = deepcopy(base.state_dict())
    image = torch.randn(2,3,35,37); prior = torch.rand(2,1,35,37)
    p = torch.zeros(2,6,35,37); p[0,1] = 1; p[1,5] = 1
    valid = torch.ones(2,1,35,37,dtype=torch.bool)
    with torch.no_grad():
        old = base(image,prior)
        for arm in ('uniform','predicted'):
            assert torch.equal(model(image,prior,probabilities=p,image_valid=valid,arm=arm)['height'],old['height'])
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=1e-4)
    target = old['height'].clone(); target[0] += 3; target[1] += 1
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        out = model(image,prior,probabilities=p,image_valid=valid,arm='predicted')
        (out['height']-target).square().mean().backward(); optimizer.step()
    assert not base.training
    assert all(v.grad is None for v in base.parameters())
    assert all(torch.equal(v,frozen[k]) for k,v in base.state_dict().items())
    with torch.no_grad():
        actual = model(image,prior,probabilities=p,image_valid=valid,arm='predicted')
        neutral = model(image,prior,probabilities=p,image_valid=valid,arm='uniform')
    assert not torch.equal(actual['height'],neutral['height'])
    for key in ('domain_logits','building_logits','vegetation_logits','fine_semantic_logits'):
        assert torch.equal(actual[key],old[key])


def test_water_and_roads_not_forced_to_zero_and_bad_probabilities_rejected():
    logits = torch.ones(1,6,2,2)*.1
    valid = torch.ones(1,1,2,2,dtype=torch.bool)
    for cls in (2,3):
        p = torch.zeros_like(logits); p[:,cls] = 1
        assert (mix_expert_corrections(logits,p,valid,arm='predicted',maximum_correction_m=80)>0).all()
    with pytest.raises(ValueError,match='sum to one'):
        mix_expert_corrections(logits,p*.5,valid,arm='predicted',maximum_correction_m=80)
