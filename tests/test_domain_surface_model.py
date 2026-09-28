import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from msr.models.domain_surface_net import (
    DomainGatedSurfaceNet,
    HierarchicalVegetationFineSemanticHead,
)
from msr.models.height_net import HeightNet
from msr.training.losses import (
    FrozenDomainHeadTeacher,
    MultiDomainSurfaceLoss,
    Stage2SemanticRepairLoss,
)


def _base_model() -> HeightNet:
    return HeightNet(
        backbone="resnet18",
        pretrained=False,
        decoder_channels=(64, 32, 16, 8, 4),
        auxiliary_building_head=True,
    )


def test_domain_gates_are_exclusive_and_outputs_are_nonnegative():
    model = DomainGatedSurfaceNet(_base_model(), hidden_channels=8, freeze_base=True)
    image = torch.randn(1, 3, 33, 35)
    prior = torch.rand(1, 1, 33, 35)

    with torch.inference_mode():
        output = model(image, prior)

    assert output["height"].shape == (1, 1, 33, 35)
    assert output["domain_logits"].shape == (1, 3, 33, 35)
    assert output["adapter_features"].shape == (1, 8, 33, 35)
    assert output["prior_domain_logits"].shape == output["domain_logits"].shape
    assert output["protected_building_logits"].shape == (1, 1, 33, 35)
    torch.testing.assert_close(
        output["domain_probabilities"].sum(dim=1),
        torch.ones((1, 33, 35)),
    )
    assert torch.all(output["height"] >= 0)
    assert torch.allclose(
        output["refinement_strength"], torch.full_like(output["refinement_strength"], 0.08)
    )
    expected = output["base_height"] + output["refinement_strength"] * (
        output["gated_height"] - output["base_height"]
    )
    torch.testing.assert_close(output["height"], expected.clamp_min(0.0))
    assert all(not parameter.requires_grad for parameter in model.base_model.parameters())


def test_pilot_adapter_supports_48_hidden_channels() -> None:
    model = DomainGatedSurfaceNet(_base_model(), hidden_channels=48, freeze_base=True)
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32))
    assert output["height"].shape == (1, 1, 32, 32)


def test_optional_six_class_head_keeps_legacy_state_and_height_compatible() -> None:
    legacy = DomainGatedSurfaceNet(_base_model(), hidden_channels=8, freeze_base=True)
    expanded = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fine_semantic_classes=6,
        freeze_base=True,
    )
    legacy_state = legacy.state_dict()
    assert not any(name.startswith("fine_semantic_head.") for name in legacy_state)
    incompatible = expanded.load_legacy_compatible_state_dict(legacy_state)
    assert set(incompatible.missing_keys) == {
        "fine_semantic_head.weight",
        "fine_semantic_head.bias",
    }
    image = torch.randn(1, 3, 32, 32)
    prior = torch.rand(1, 1, 32, 32)
    legacy.eval()
    expanded.eval()
    with torch.inference_mode():
        old_output = legacy(image, prior)
        new_output = expanded(image, prior)

    assert "fine_semantic_logits" not in old_output
    assert new_output["fine_semantic_logits"].shape == (1, 6, 32, 32)
    torch.testing.assert_close(
        new_output["fine_semantic_probabilities"],
        torch.full((1, 6, 32, 32), 1.0 / 6.0),
    )
    torch.testing.assert_close(old_output["height"], new_output["height"])


def test_default_six_class_head_preserves_linear_checkpoint_keys() -> None:
    model = DomainGatedSurfaceNet(
        _base_model(), hidden_channels=8, fine_semantic_classes=6, freeze_base=True
    )

    assert model.fine_semantic_head_type == "linear"
    assert {
        name for name in model.state_dict() if name.startswith("fine_semantic_head.")
    } == {"fine_semantic_head.weight", "fine_semantic_head.bias"}


def test_refined_six_class_head_is_prefix_isolated_and_height_compatible() -> None:
    torch.manual_seed(41)
    legacy = DomainGatedSurfaceNet(_base_model(), hidden_channels=8, freeze_base=True)
    refined = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fine_semantic_classes=6,
        fine_semantic_head_type="spatial_refined",
        freeze_base=True,
    )
    legacy_state = legacy.state_dict()
    incompatible = refined.load_legacy_compatible_state_dict(legacy_state)
    refined_head_keys = {
        name
        for name in refined.state_dict()
        if name.startswith("fine_semantic_head.")
    }

    assert refined.fine_semantic_head_type == "spatial_refined"
    assert len(refined_head_keys) > 2
    assert set(incompatible.missing_keys) == refined_head_keys
    assert all(name.startswith("fine_semantic_head.") for name in refined_head_keys)

    image = torch.randn(1, 3, 31, 35)
    prior = torch.rand(1, 1, 31, 35)
    legacy.eval()
    refined.eval()
    with torch.inference_mode():
        legacy_output = legacy(image, prior)
        refined_output = refined(image, prior)

    torch.testing.assert_close(refined_output["height"], legacy_output["height"])
    torch.testing.assert_close(
        refined_output["fine_semantic_probabilities"],
        torch.full((1, 6, 31, 35), 1.0 / 6.0),
    )


def test_refined_six_class_head_rejects_linear_expanded_checkpoint() -> None:
    linear = DomainGatedSurfaceNet(
        _base_model(), hidden_channels=8, fine_semantic_classes=6, freeze_base=True
    )
    refined = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fine_semantic_classes=6,
        fine_semantic_head_type="spatial_refined",
        freeze_base=True,
    )

    with pytest.raises(RuntimeError, match="fine_semantic_head"):
        refined.load_legacy_compatible_state_dict(linear.state_dict())


def test_refined_six_class_head_rejects_partial_checkpoint() -> None:
    model = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fine_semantic_classes=6,
        fine_semantic_head_type="spatial_refined",
        freeze_base=True,
    )
    truncated_state = model.state_dict()
    truncated_state.pop("fine_semantic_head.classifier.bias")

    with pytest.raises(RuntimeError, match="fine_semantic_head.classifier.bias"):
        model.load_legacy_compatible_state_dict(truncated_state)


def test_hierarchical_six_class_head_softly_splits_vegetation_mass() -> None:
    torch.manual_seed(43)
    head = HierarchicalVegetationFineSemanticHead(channels=64)
    assert sum(parameter.numel() for parameter in head.parameters()) == 1_094
    initial = head.forward_with_components(torch.zeros(1, 64, 1, 1))
    torch.testing.assert_close(
        initial["probabilities"],
        torch.full((1, 6, 1, 1), 1.0 / 6.0),
    )
    components = head.forward_with_components(torch.randn(2, 64, 7, 9))

    probabilities = components["probabilities"]
    coarse_probabilities = components["coarse_probabilities"]
    tree_given_vegetation = components["vegetation_split_logits"].sigmoid()
    assert components["logits"].shape == (2, 6, 7, 9)
    assert components["coarse_logits"].shape == (2, 5, 7, 9)
    assert components["vegetation_split_logits"].shape == (2, 1, 7, 9)
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones((2, 7, 9)),
    )
    torch.testing.assert_close(
        probabilities[:, 4:5] + probabilities[:, 5:6],
        coarse_probabilities[:, 4:5],
    )
    torch.testing.assert_close(
        probabilities[:, 5:6],
        coarse_probabilities[:, 4:5] * tree_given_vegetation,
    )


def test_hierarchical_six_class_head_is_prefix_isolated_from_height() -> None:
    torch.manual_seed(47)
    legacy = DomainGatedSurfaceNet(_base_model(), hidden_channels=8, freeze_base=True)
    hierarchical = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fine_semantic_classes=6,
        fine_semantic_head_type="hierarchical_vegetation",
        freeze_base=True,
    )
    legacy_state = legacy.state_dict()
    incompatible = hierarchical.load_legacy_compatible_state_dict(legacy_state)
    head_keys = {
        name
        for name in hierarchical.state_dict()
        if name.startswith("fine_semantic_head.")
    }

    assert set(incompatible.missing_keys) == head_keys
    assert head_keys
    assert all(name.startswith("fine_semantic_head.") for name in head_keys)
    assert sum(
        parameter.numel() for parameter in hierarchical.fine_semantic_head.parameters()
    ) == 142
    for name, tensor in legacy_state.items():
        assert torch.equal(hierarchical.state_dict()[name], tensor), name

    image = torch.randn(1, 3, 31, 35)
    prior = torch.rand(1, 1, 31, 35)
    legacy.eval()
    hierarchical.eval()
    with torch.inference_mode():
        legacy_output = legacy(image, prior)
        hierarchical_output = hierarchical(image, prior)

    for name, tensor in legacy_output.items():
        assert torch.equal(hierarchical_output[name], tensor), name
    assert hierarchical_output["fine_semantic_logits"].shape == (1, 6, 31, 35)
    assert hierarchical_output["fine_semantic_coarse_logits"].shape == (
        1,
        5,
        31,
        35,
    )
    assert hierarchical_output[
        "fine_semantic_vegetation_split_logits"
    ].shape == (1, 1, 31, 35)


def test_hierarchical_head_preserves_spatial_v3_training_rng_stream() -> None:
    """The paired heads must leave sampling/augmentation RNG in the same state."""

    torch.manual_seed(20260913)
    DomainGatedSurfaceNet(
        torch.nn.Identity(),
        hidden_channels=64,
        fine_semantic_classes=6,
        fine_semantic_head_type="spatial_refined",
        freeze_base=True,
    )
    spatial_rng_state = torch.get_rng_state().clone()
    spatial_following_draw = torch.rand(32)

    torch.manual_seed(20260913)
    DomainGatedSurfaceNet(
        torch.nn.Identity(),
        hidden_channels=64,
        fine_semantic_classes=6,
        fine_semantic_head_type="hierarchical_vegetation",
        freeze_base=True,
    )
    hierarchical_rng_state = torch.get_rng_state().clone()
    hierarchical_following_draw = torch.rand(32)

    assert torch.equal(hierarchical_rng_state, spatial_rng_state)
    assert torch.equal(hierarchical_following_draw, spatial_following_draw)

    torch.manual_seed(20260913)
    DomainGatedSurfaceNet(
        torch.nn.Identity(),
        hidden_channels=64,
        fine_semantic_classes=6,
        fine_semantic_head_type="spatial_refined",
        freeze_base=True,
    )
    spatial_samples = list(
        WeightedRandomSampler(torch.ones(11), 32, replacement=True)
    )

    torch.manual_seed(20260913)
    DomainGatedSurfaceNet(
        torch.nn.Identity(),
        hidden_channels=64,
        fine_semantic_classes=6,
        fine_semantic_head_type="hierarchical_vegetation",
        freeze_base=True,
    )
    hierarchical_samples = list(
        WeightedRandomSampler(torch.ones(11), 32, replacement=True)
    )
    assert hierarchical_samples == spatial_samples


def test_unknown_six_class_head_type_fails_closed() -> None:
    with pytest.raises(ValueError, match="fine_semantic_head_type"):
        DomainGatedSurfaceNet(
            _base_model(),
            hidden_channels=8,
            fine_semantic_classes=6,
            fine_semantic_head_type="mystery",
            freeze_base=True,
        )


def test_legacy_compatible_load_remains_strict_for_height_tensors() -> None:
    model = DomainGatedSurfaceNet(
        _base_model(), hidden_channels=8, fine_semantic_classes=6, freeze_base=True
    )
    legacy_state = DomainGatedSurfaceNet(
        _base_model(), hidden_channels=8, freeze_base=True
    ).state_dict()
    legacy_state.pop("domain_head.weight")

    with pytest.raises(RuntimeError, match="domain_head.weight"):
        model.load_legacy_compatible_state_dict(legacy_state)


def test_legacy_compatible_load_rejects_a_partial_six_class_head() -> None:
    model = DomainGatedSurfaceNet(
        _base_model(), hidden_channels=8, fine_semantic_classes=6, freeze_base=True
    )
    truncated_state = model.state_dict()
    truncated_state.pop("fine_semantic_head.bias")

    with pytest.raises(RuntimeError, match="fine_semantic_head.bias"):
        model.load_legacy_compatible_state_dict(truncated_state)


def test_protected_fusion_keeps_confident_building_pixels_near_base() -> None:
    model = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fusion_mode="protected_vegetation",
        building_protection_power=2.0,
        freeze_base=True,
    )
    with torch.no_grad():
        assert model.base_model.building_head is not None
        model.base_model.building_head.weight.zero_()
        model.base_model.building_head.bias.fill_(10.0)
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32), torch.rand(1, 1, 32, 32))

    assert output["building_protection"].mean() > 0.99
    assert output["effective_refinement_strength"].mean() < 1e-3
    torch.testing.assert_close(
        output["height"], output["base_height"], atol=1e-3, rtol=0
    )


def test_calibrated_fusion_exposes_bounded_height_aware_building_gate() -> None:
    model = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fusion_mode="calibrated_surface",
        building_fusion_min_height_m=10.0,
        building_fusion_temperature_m=2.0,
        building_fusion_score_threshold=0.5,
        building_fusion_score_temperature=0.05,
        freeze_base=True,
    )
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32), torch.rand(1, 1, 32, 32))

    assert output["height"].shape == output["base_height"].shape
    assert torch.all(output["height"] >= 0)
    assert torch.all(output["building_fusion_gate"] >= 0)
    assert torch.all(output["building_fusion_gate"] <= 1)


def test_vegetation_fusion_temperature_only_calibrates_height_gate() -> None:
    model = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fusion_mode="protected_vegetation",
        vegetation_fusion_temperature=0.8,
        freeze_base=True,
    )
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 32, 32), torch.rand(1, 1, 32, 32))

    raw_vegetation = output["domain_probabilities"][:, 2:3]
    calibrated = output["vegetation_fusion_probability"]
    assert torch.all(calibrated >= 0)
    assert torch.all(calibrated <= 1)
    assert not torch.allclose(calibrated, raw_vegetation)


def test_high_confidence_vegetation_expert_fusion_is_bounded() -> None:
    baseline = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fusion_mode="protected_vegetation",
        vegetation_expert_fusion_threshold=0.85,
        vegetation_expert_fusion_strength=0.0,
        freeze_base=True,
    )
    fused = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fusion_mode="protected_vegetation",
        vegetation_expert_fusion_threshold=0.85,
        vegetation_expert_fusion_strength=0.1,
        freeze_base=True,
    )
    fused.load_state_dict(baseline.state_dict())
    with torch.no_grad():
        baseline.domain_head.bias.copy_(torch.tensor((0.0, 0.0, 10.0)))
        fused.domain_head.bias.copy_(baseline.domain_head.bias)
    image = torch.randn(1, 3, 32, 32)
    prior = torch.rand(1, 1, 32, 32)
    with torch.inference_mode():
        original = baseline(image, prior)
        corrected = fused(image, prior)

    assert torch.all(corrected["vegetation_expert_fusion_gate"] == 1)
    expected = original["height"] + 0.1 * (
        original["canopy_height"] - original["height"]
    )
    torch.testing.assert_close(corrected["height"], expected.clamp_min(0.0))


def test_multidomain_loss_backpropagates_into_adapter_heads():
    model = DomainGatedSurfaceNet(_base_model(), hidden_channels=8, freeze_base=True)
    image = torch.randn(1, 3, 32, 32)
    prior = torch.rand(1, 1, 32, 32)
    output = model(image, prior)
    target = torch.zeros(1, 1, 32, 32)
    domain = torch.zeros(1, 32, 32, dtype=torch.long)
    domain[:, :10] = 1
    domain[:, 10:20] = 2
    target[:, :, :10] = 12.0
    target[:, :, 10:20] = 15.0
    valid = torch.ones_like(target, dtype=torch.bool)

    total, components = MultiDomainSurfaceLoss()(output, target, valid, domain, valid[:, 0])
    total.backward()

    assert torch.isfinite(total)
    assert components["building_height"].item() > 0
    assert components["canopy_height"].item() > 0
    assert model.domain_head.weight.grad is not None
    assert model.canopy_height_head.weight.grad is not None
    assert "refinement" in components
    assert all(parameter.grad is None for parameter in model.base_model.parameters())


def test_multidomain_loss_supports_direct_fused_class_and_tall_guards() -> None:
    model = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fusion_mode="protected_vegetation",
        freeze_base=True,
    )
    output = model(torch.randn(1, 3, 32, 32), torch.rand(1, 1, 32, 32))
    target = torch.zeros(1, 1, 32, 32)
    domain = torch.zeros(1, 32, 32, dtype=torch.long)
    domain[:, :10] = 1
    domain[:, 10:20] = 2
    target[:, :, :10] = 12.0
    target[:, :, 10:20] = 25.0
    valid = torch.ones_like(target, dtype=torch.bool)
    criterion = MultiDomainSurfaceLoss(
        canopy_mse_weight=0.05,
        tall_canopy_weight=1.0,
        tall_canopy_threshold_m=15.0,
        building_mse_weight=0.05,
        tall_building_weight=1.0,
        tall_building_threshold_m=10.0,
        fused_building_weight=1.0,
        fused_building_mse_weight=0.05,
        fused_vegetation_weight=1.0,
        fused_tall_building_weight=1.0,
        fused_tall_vegetation_weight=1.0,
        building_distillation_weight=0.25,
    )

    total, components = criterion(output, target, valid, domain, valid[:, 0])

    assert torch.isfinite(total)
    assert components["canopy_mse"].item() > 0
    assert components["tall_canopy_height"].item() > 0
    assert components["building_mse"].item() > 0
    assert components["tall_building_height"].item() > 0
    assert components["fused_building_height"].item() > 0
    assert components["fused_building_mse"].item() > 0
    assert components["fused_vegetation_height"].item() > 0
    assert components["fused_tall_building_height"].item() > 0
    assert components["fused_tall_vegetation_height"].item() > 0
    assert components["building_distillation"].item() >= 0


def test_six_class_loss_uses_classification_mask_not_height_mask() -> None:
    model = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fine_semantic_classes=6,
        freeze_base=True,
    )
    model.eval()
    output = model(torch.randn(1, 3, 32, 32), torch.rand(1, 1, 32, 32))
    height = torch.zeros(1, 1, 32, 32)
    regression_valid = torch.zeros_like(height, dtype=torch.bool)
    domain = torch.zeros(1, 32, 32, dtype=torch.long)
    domain_valid = torch.ones(1, 32, 32, dtype=torch.bool)
    fine_target = torch.full((1, 32, 32), 255, dtype=torch.long)
    fine_target[:, :16] = 2  # water: identification only
    fine_target[:, 16:] = 3  # road: identification only
    classification_valid = fine_target != 255
    criterion = MultiDomainSurfaceLoss(
        height_weight=0.0,
        semantic_weight=0.0,
        fine_semantic_weight=1.0,
        building_weight=0.0,
        canopy_weight=0.0,
        ground_suppression_weight=0.0,
    )

    total, components = criterion(
        output,
        height,
        regression_valid,
        domain,
        domain_valid,
        fine_class_target=fine_target,
        classification_valid_mask=classification_valid,
        image_valid_mask=torch.ones(1, 1, 32, 32, dtype=torch.bool),
    )
    total.backward()

    assert components["fine_semantic"].item() == pytest.approx(
        torch.log(torch.tensor(6.0)).item()
    )
    assert model.fine_semantic_head is not None
    assert model.fine_semantic_head.weight.grad is not None
    assert torch.count_nonzero(model.fine_semantic_head.weight.grad) > 0


def test_six_class_focal_loss_downweights_easy_pixels() -> None:
    model = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fine_semantic_classes=6,
        freeze_base=True,
    )
    model.eval()
    output = model(torch.randn(1, 3, 32, 32), torch.rand(1, 1, 32, 32))
    output["fine_semantic_logits"] = torch.full((1, 6, 32, 32), -4.0)
    output["fine_semantic_logits"][:, 2] = 4.0
    target = torch.full((1, 32, 32), 2, dtype=torch.long)
    common = {
        "height_weight": 0.0,
        "semantic_weight": 0.0,
        "fine_semantic_weight": 1.0,
        "building_weight": 0.0,
        "canopy_weight": 0.0,
        "ground_suppression_weight": 0.0,
    }
    ordinary = MultiDomainSurfaceLoss(**common)
    focal = MultiDomainSurfaceLoss(**common, fine_semantic_focal_gamma=2.0)
    args = (
        output,
        torch.zeros(1, 1, 32, 32),
        torch.zeros(1, 1, 32, 32, dtype=torch.bool),
        torch.zeros(1, 32, 32, dtype=torch.long),
        torch.ones(1, 32, 32, dtype=torch.bool),
    )
    kwargs = {
        "fine_class_target": target,
        "classification_valid_mask": torch.ones_like(target, dtype=torch.bool),
        "image_valid_mask": torch.ones(1, 1, 32, 32, dtype=torch.bool),
    }

    ordinary_loss, _ = ordinary(*args, **kwargs)
    focal_loss, _ = focal(*args, **kwargs)

    assert focal_loss.item() < ordinary_loss.item()


def test_six_class_focal_gamma_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="fine_semantic_focal_gamma"):
        MultiDomainSurfaceLoss(fine_semantic_focal_gamma=-0.1)


def test_stage2_mixed_replay_updates_only_domain_head() -> None:
    model = DomainGatedSurfaceNet(
        _base_model(),
        hidden_channels=8,
        fusion_mode="protected_vegetation",
        freeze_base=True,
    )
    teacher = FrozenDomainHeadTeacher(model.domain_head)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.domain_head.parameters():
        parameter.requires_grad_(True)

    output = model(
        torch.randn(2, 3, 32, 32),
        torch.rand(2, 1, 32, 32),
    )
    teacher_logits = teacher(
        output["adapter_features"], output["prior_domain_logits"]
    )
    domain = torch.zeros(2, 32, 32, dtype=torch.long)
    domain[0, :16] = 1
    domain[0, 16:] = 2
    domain[1, :16] = 1
    valid = torch.ones_like(domain, dtype=torch.bool)
    target = torch.zeros(2, 1, 32, 32)
    loss, _ = Stage2SemanticRepairLoss()(
        output,
        domain,
        valid,
        ["gamus", "legacy"],
        batch_landscapes=["mixed", "urban"],
        teacher_domain_logits=teacher_logits,
        target=target,
        regression_mask=valid[:, None],
    )
    loss.backward()

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    gradient_names = {
        name for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    assert trainable_names == {"domain_head.weight", "domain_head.bias"}
    assert gradient_names == trainable_names
    assert all(parameter.grad is None for parameter in teacher.parameters())
