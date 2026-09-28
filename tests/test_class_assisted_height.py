from copy import deepcopy
import pytest
import torch
from msr.models.class_assisted_height import ClassAssistedHeightNet, conditioning_channels
from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.height_net import HeightNet


def model():
    base=DomainGatedSurfaceNet(HeightNet('resnet18',pretrained=False,decoder_channels=(32,16,8,4,4)),hidden_channels=8,fine_semantic_classes=6)
    return ClassAssistedHeightNet(base,hidden_channels=8)


def test_uniform_ignores_predictions_and_invalid_pixels_are_not_classes():
    p=torch.rand(1,6,3,4).softmax(1).requires_grad_()
    mask=torch.ones(1,1,3,4,dtype=torch.bool); mask[:,:,0,0]=False
    a=conditioning_channels(p,mask,arm='uniform')
    b=conditioning_channels(p.flip(1),mask,arm='uniform')
    assert torch.equal(a,b)
    assert not a.requires_grad
    assert (a[:,:,0,0]==0).all()
    assert torch.equal(a[:,6:],mask.float())
    assert not torch.equal(conditioning_channels(p,mask,arm='predicted'),a)


def test_zero_init_and_class_outputs_survive_learning():
    m=model().eval(); frozen=deepcopy(m.protected.state_dict())
    image=torch.randn(2,3,35,37); prior=torch.rand(2,1,35,37)
    p=torch.rand(2,6,35,37).softmax(1); mask=torch.ones(2,1,35,37,dtype=torch.bool)
    with torch.no_grad():
        original=m.protected(image,prior)
        for arm in ['uniform','predicted']:
            out=m(image,prior,probabilities=p,image_valid=mask,arm=arm)
            assert torch.equal(out['height'],original['height'])
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    m.train(); assert not m.protected.training
    for _ in range(2):
        opt.zero_grad(); out=m(image,prior,probabilities=p,image_valid=mask,arm='predicted')
        (out['height']-(original['height']+3)).square().mean().backward(); opt.step()
    assert next(m.deepest.parameters()).grad.abs().sum()>0
    assert all(p.grad is None for p in m.protected.parameters())
    with torch.no_grad(): out=m(image,prior,probabilities=p,image_valid=mask,arm='predicted')
    for key in ['domain_logits','building_logits','vegetation_logits','fine_semantic_logits']:
        assert torch.equal(out[key],original[key])
    assert all(torch.equal(v,frozen[k]) for k,v in m.protected.state_dict().items())


def test_probability_and_alignment_contracts_fail_closed():
    m=model(); image=torch.rand(1,3,32,32); prior=torch.rand(1,1,32,32)
    mask=torch.ones(1,1,32,32,dtype=torch.bool); p=torch.ones(1,6,32,32)/6
    with pytest.raises(ValueError,match='sum to one'):
        conditioning_channels(p*2,mask,arm='predicted')
    with pytest.raises(ValueError,match='Independent boolean'):
        conditioning_channels(p,mask.float(),arm='predicted')
    with pytest.raises(ValueError,match='grid mismatch'):
        m(image,prior,probabilities=p[:,:,:16],image_valid=mask,arm='predicted')
    with pytest.raises(ValueError,match='arm must'):
        conditioning_channels(p,mask,arm='reference')
