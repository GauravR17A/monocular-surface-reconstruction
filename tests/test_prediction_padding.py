import numpy as np
import pytest
import torch

from msr.inference.predict import predict_height
from msr.models.domain_surface_net import DomainGatedSurfaceNet


class ContextModel(DomainGatedSurfaceNet):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.shapes = []

    def forward(self, image, prior):
        self.shapes.append(tuple(image.shape[-2:]))
        assert prior.shape[-2:] == image.shape[-2:]
        # Context-dependent output exposes the effect of unnecessary padding.
        return {"height": prior + image.mean()}


def prediction(shape, policy=None):
    model = ContextModel()
    kwargs = {} if policy is None else {"padding_policy": policy}
    valid = np.ones(shape, dtype=bool)
    valid[-1, -1] = False
    result = predict_height(np.full((3, *shape), 150, dtype=np.uint8), model=model, device="cpu", relative_prior=np.full(shape, 3.0, np.float32), valid_mask=valid, amp=False, **kwargs)
    return model.shapes, result.height_map


def test_minimal_preserves_native_stride_aligned_small_scene():
    fixed_shapes, fixed = prediction((384, 384))
    explicit_shapes, explicit = prediction((384, 384), "fixed_tile")
    minimal_shapes, minimal = prediction((384, 384), "minimal")
    assert fixed_shapes == explicit_shapes == [(512, 512)]
    assert minimal_shapes == [(384, 384)]
    np.testing.assert_array_equal(fixed, explicit)
    assert not np.allclose(fixed[:-1], minimal[:-1])
    assert minimal.shape == (384, 384)
    assert np.isnan(minimal[-1, -1])


def test_minimal_rounds_rectangular_partial_dimension_only_to_stride32():
    shapes, output = prediction((353, 1024), "minimal")
    assert len(shapes) == 3
    assert set(shapes) == {(384, 512)}
    assert output.shape == (353, 1024)
    assert np.isfinite(output[:-1]).all()


def test_large_full_tiles_are_identical_between_policies():
    fixed_shapes, fixed = prediction((1024, 1024), "fixed_tile")
    minimal_shapes, minimal = prediction((1024, 1024), "minimal")
    assert fixed_shapes == minimal_shapes == [(512, 512)] * 9
    np.testing.assert_array_equal(fixed, minimal)


def test_unknown_padding_policy_fails_before_inference():
    with pytest.raises(ValueError, match="padding_policy"):
        prediction((32, 32), "guess")
