from __future__ import annotations

import pytest
import torch
from transformers import SegformerConfig

from msr.models.rgb_segmenter import CLASS_NAMES, RgbSegformer


def tiny_config():
    return SegformerConfig(hidden_sizes=[8, 16, 24, 32], depths=[1, 1, 1, 1],
                           num_attention_heads=[1, 2, 3, 4], sr_ratios=[8, 4, 2, 1],
                           decoder_hidden_size=16, num_channels=3)


def test_model_returns_native_six_class_logits_and_no_height():
    model = RgbSegformer(tiny_config()).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 3, 64, 96))
    assert set(output) == {"logits"}
    assert output["logits"].shape == (1, 6, 64, 96)
    assert torch.isfinite(output["logits"]).all()
    assert model.network.config.id2label == dict(enumerate(CLASS_NAMES))


def test_encoder_and_decoder_receive_gradients_and_parameter_groups_partition():
    model = RgbSegformer(tiny_config()).train()
    output = model(torch.rand(2, 3, 64, 64))["logits"]
    torch.nn.functional.cross_entropy(output, torch.randint(6, (2, 64, 64))).backward()
    assert all(p.requires_grad for p in model.parameters())
    for group in (model.network.segformer, model.network.decode_head):
        assert sum(float(p.grad.abs().sum()) for p in group.parameters() if p.grad is not None) > 0
    groups = model.parameter_groups(6e-5, 6e-4, 0.01)
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))


def test_checkpoint_architecture_roundtrip_and_normalization_buffers():
    model = RgbSegformer(tiny_config()).eval()
    restored = RgbSegformer.from_architecture(model.architecture).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(image)["logits"], restored(image)["logits"])
    assert torch.allclose(model.rgb_mean.flatten(), torch.tensor([0.485, 0.456, 0.406]))
    assert torch.allclose(model.rgb_std.flatten(), torch.tensor([0.229, 0.224, 0.225]))


@pytest.mark.parametrize("image", [torch.zeros(3, 64, 64), torch.zeros(1, 4, 64, 64), torch.zeros(1, 3, 64, 64, dtype=torch.uint8)])
def test_rejects_non_rgb_contract(image):
    with pytest.raises(ValueError, match="floating RGB"):
        RgbSegformer(tiny_config())(image)


def test_no_remote_model_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):
        RgbSegformer.from_local_pretrained(tmp_path / "missing")
