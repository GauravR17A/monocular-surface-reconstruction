import importlib.util
from pathlib import Path
import torch
import pytest

spec=importlib.util.spec_from_file_location('proof',Path(__file__).resolve().parents[1]/'scripts/class_assisted_height_proof.py')
proof=importlib.util.module_from_spec(spec); spec.loader.exec_module(proof)


def test_state_digest_includes_scalar_buffers_and_is_order_independent():
    state={'b':torch.tensor(3),'a':torch.ones(2,3)}
    assert proof.state_digest(state)==proof.state_digest(dict(reversed(list(state.items()))))
    assert proof.state_digest(state)!=proof.state_digest({**state,'b':torch.tensor(4)})


def test_identical_aligned_crop_keeps_all_masks_and_classifier_channels():
    image=torch.arange(36).reshape(1,6,6).repeat(3,1,1).float()
    height=torch.zeros(1,6,6); height[0,5,5]=30
    valid=height>0
    sample={'image':image,'height':height,'regression_mask':valid,'image_valid_mask':torch.ones_like(valid),
            'class_probabilities':image[:1].repeat(6,1,1),'sample_id':'training_example'}
    crop,meta=proof.crop_sample(sample,'tall',4,42)
    assert meta['row']==meta['col']==2
    assert crop['height'][0,3,3]==30
    assert torch.equal(crop['image'][0],crop['class_probabilities'][0])
    assert crop['regression_mask'].sum()==1
    assert crop['image_valid_mask'].sum()==16
    assert sample['image'].shape==(3,6,6)
    with pytest.raises(ValueError,match='lacks supervised'):
        proof.crop_sample(sample,'short',4,42)
