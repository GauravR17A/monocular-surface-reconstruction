from copy import deepcopy

import pytest
import torch

from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.height_net import HeightNet
from msr.models.residual_height import ResidualHeightSurfaceNet


def _protected() -> DomainGatedSurfaceNet:
    return DomainGatedSurfaceNet(
        HeightNet(
            "resnet18", pretrained=False, decoder_channels=(32, 16, 8, 4, 4)
        ),
        hidden_channels=8,
        fine_semantic_classes=6,
    ).eval()


def test_feature_access_keeps_original_predictions_identical() -> None:
    protected = _protected()
    image = torch.randn(1, 3, 35, 37)
    prior = torch.rand(1, 1, 35, 37)
    with torch.no_grad():
        normal = protected(image, prior)
        with_features = protected(image, prior, return_features=True)
    for key in normal:
        assert torch.equal(normal[key], with_features[key]), key
    assert len(with_features["encoder_features"]) == 5


def test_residual_starts_at_baseline_and_preserves_semantics_after_training() -> None:
    protected = _protected()
    before = deepcopy(protected.state_dict())
    model = ResidualHeightSurfaceNet(protected, hidden_channels=8)
    image = torch.randn(1, 3, 35, 37)
    prior = torch.rand(1, 1, 35, 37)
    with torch.no_grad():
        original = protected(image, prior)
        initial = model(image, prior)
    assert torch.equal(original["height"], initial["height"])
    assert torch.count_nonzero(initial["height_correction"]) == 0
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    model.train()
    assert not protected.training
    for _ in range(2):
        optimizer.zero_grad()
        output = model(image, prior)
        (output["height"] - (original["height"] + 3)).square().mean().backward()
        optimizer.step()
    assert model.correction_head.weight.grad.abs().sum() > 0
    assert next(model.deepest.parameters()).grad.abs().sum() > 0
    assert all(p.grad is None for p in protected.parameters())
    with torch.no_grad():
        after = model(image, prior)
    assert not torch.equal(initial["height"], after["height"])
    assert (after["height"] >= 0).all()
    for key in ("building_logits", "vegetation_logits", "fine_semantic_logits"):
        assert torch.equal(original[key], after[key]), key
    for key, value in protected.state_dict().items():
        assert torch.equal(value, before[key]), key


def test_refiner_requires_prior_and_valid_bound() -> None:
    protected = _protected()
    with pytest.raises(ValueError, match="finite and positive"):
        ResidualHeightSurfaceNet(protected, maximum_correction_m=float("nan"))
    model = ResidualHeightSurfaceNet(protected, hidden_channels=8)
    with pytest.raises(ValueError, match="requires the cached"):
        model(torch.rand(1, 3, 32, 32))
