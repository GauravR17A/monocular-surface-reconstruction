from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.routed_surface import (
    ConservativeSceneRouter,
    FrozenDualSurfaceHeadRouter,
    FrozenSurfaceHeadPack,
    RoutedDomainGatedSurfaceNet,
    select_scene_endpoint,
)


class _TinyBase(nn.Module):
    """Fast stand-in exposing the same outputs as the protected height net."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "height": F.softplus(image[:, :1] * self.scale),
            "building_logits": image[:, 1:2] * self.scale,
        }


def _surface_model() -> DomainGatedSurfaceNet:
    model = DomainGatedSurfaceNet(
        _TinyBase(),
        hidden_channels=8,
        maximum_building_residual_m=17.0,
        freeze_base=True,
    )
    generator = torch.Generator().manual_seed(20260912)
    with torch.no_grad():
        for head in (
            model.domain_head,
            model.building_residual_head,
            model.canopy_height_head,
        ):
            head.weight.copy_(
                torch.rand(head.weight.shape, generator=generator) * 0.08 - 0.04
            )
            head.bias.copy_(
                torch.rand(head.bias.shape, generator=generator) * 0.4 - 0.2
            )
    return model


def test_frozen_head_pack_reproduces_existing_model_endpoint_exactly() -> None:
    model = _surface_model().eval()
    pack = FrozenSurfaceHeadPack.from_model(model)
    image = torch.randn(2, 3, 19, 23)
    prior = torch.rand(2, 1, 19, 23)

    with torch.inference_mode():
        reference = model(image, prior)
        endpoint = pack(
            reference["adapter_features"],
            reference["prior_domain_logits"],
            reference["base_height"],
        )

    for name in (
        "domain_logits",
        "domain_probabilities",
        "building_height",
        "canopy_height",
        "building_logits",
        "vegetation_logits",
    ):
        torch.testing.assert_close(endpoint[name], reference[name], rtol=0, atol=0)
    # A negative raw residual can be clipped when the non-negative building
    # height is formed, so it cannot always be recovered by subtracting the
    # base height.  Verify the actual forward relationship instead.
    torch.testing.assert_close(
        endpoint["building_height"],
        torch.clamp_min(
            reference["base_height"] + endpoint["building_residual"], 0.0
        ),
        rtol=0,
        atol=0,
    )
    assert not pack.training
    assert all(not parameter.requires_grad for parameter in pack.parameters())


def test_frozen_head_pack_reproduces_complete_protected_fusion_exactly() -> None:
    model = _surface_model().eval()
    model.fusion_mode = "protected_vegetation"
    model.building_protection_power = 2.5
    model.vegetation_fusion_temperature = 0.8
    model.vegetation_expert_fusion_threshold = 0.55
    model.vegetation_expert_fusion_strength = 0.1
    pack = FrozenSurfaceHeadPack.from_model(model)
    image = torch.randn(2, 3, 19, 23)
    prior = torch.rand(2, 1, 19, 23)

    with torch.inference_mode():
        reference = model(image, prior)
        endpoint = pack.forward_fused(
            reference["adapter_features"],
            reference["prior_domain_logits"],
            reference["base_height"],
            protected_building_logits=reference["protected_building_logits"],
            refinement_logits=reference["refinement_logits"],
            log_variance=reference["log_variance"],
        )

    for name in (
        "height",
        "domain_logits",
        "domain_probabilities",
        "building_height",
        "canopy_height",
        "gated_height",
        "refinement_strength",
        "effective_refinement_strength",
        "building_protection",
        "building_fusion_gate",
        "vegetation_fusion_probability",
        "vegetation_expert_fusion_gate",
        "refinement_logits",
        "building_logits",
        "protected_building_logits",
        "vegetation_logits",
        "log_variance",
        "base_height",
    ):
        torch.testing.assert_close(endpoint[name], reference[name], rtol=0, atol=0)


def test_frozen_head_pack_is_an_independent_immutable_snapshot() -> None:
    model = _surface_model()
    pack = FrozenSurfaceHeadPack.from_model(model)
    features = torch.randn(1, 8, 5, 7)
    prior_logits = torch.randn(1, 3, 5, 7)
    base_height = torch.rand(1, 1, 5, 7)
    with torch.inference_mode():
        before = pack(features, prior_logits, base_height)
    with torch.no_grad():
        model.domain_head.bias.add_(100.0)
        model.building_residual_head.bias.add_(100.0)
        model.canopy_height_head.bias.add_(100.0)
    pack.train(True)
    with torch.inference_mode():
        after = pack(features, prior_logits, base_height)

    assert not pack.training
    for name in before:
        torch.testing.assert_close(after[name], before[name], rtol=0, atol=0)


def test_conservative_scene_router_defaults_to_protected_fallback() -> None:
    router = ConservativeSceneRouter(
        8,
        hidden_features=4,
        decision_threshold=0.9,
        initial_candidate_probability=0.01,
    )
    route = router(torch.randn(3, 8, 11, 13))

    torch.testing.assert_close(
        route["candidate_probability"],
        torch.full((3, 1), 0.01),
        atol=1e-7,
        rtol=0,
    )
    assert route["candidate_selected"].shape == (3, 1)
    assert not bool(torch.any(route["candidate_selected"]))
    assert route["scene_descriptor"].shape == (3, 16)


def test_scene_router_keeps_threshold_decision_in_float32_under_autocast() -> None:
    router = ConservativeSceneRouter(
        1,
        hidden_features=2,
        decision_threshold=0.9002,
        initial_candidate_probability=0.01,
    )
    final_layer = router.network[-1]
    assert isinstance(final_layer, nn.Linear)
    with torch.no_grad():
        final_layer.weight.zero_()
        final_layer.bias.fill_(
            torch.logit(torch.tensor(0.8984, dtype=torch.float32)).item()
        )

    # In bf16, both 0.8984 and 0.9002 round to 0.8984375.  Comparing in that
    # dtype would incorrectly select the candidate.  fp16 is included to keep
    # the same precision contract for both supported autocast modes.
    for autocast_dtype in (torch.bfloat16, torch.float16):
        with torch.autocast(device_type="cpu", dtype=autocast_dtype):
            route = router(torch.ones(2, 1, 4, 4))

        assert route["scene_descriptor"].dtype == torch.float32
        assert route["candidate_logit"].dtype == torch.float32
        assert route["candidate_probability"].dtype == torch.float32
        assert bool(torch.all(route["candidate_probability"] < 0.9002))
        assert not bool(torch.any(route["candidate_selected"]))


def test_dual_surface_router_retains_fallback_near_threshold_under_autocast() -> None:
    fallback_model = _surface_model()
    candidate_model = deepcopy(fallback_model)
    with torch.no_grad():
        candidate_model.domain_head.bias.add_(torch.tensor((1.0, 2.0, 3.0)))
    fallback_pack = FrozenSurfaceHeadPack.from_model(fallback_model)
    candidate_pack = FrozenSurfaceHeadPack.from_model(candidate_model)
    router = ConservativeSceneRouter(
        8,
        hidden_features=4,
        decision_threshold=0.9002,
        initial_candidate_probability=0.01,
    )
    final_layer = router.network[-1]
    assert isinstance(final_layer, nn.Linear)
    with torch.no_grad():
        final_layer.weight.zero_()
        final_layer.bias.fill_(
            torch.logit(torch.tensor(0.8984, dtype=torch.float32)).item()
        )
    routed = FrozenDualSurfaceHeadRouter(fallback_pack, candidate_pack, router)
    features = torch.randn(2, 8, 5, 7)
    prior_logits = torch.randn(2, 3, 5, 7)
    base_height = torch.rand(2, 1, 5, 7)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        expected = fallback_pack(features, prior_logits, base_height)
        output = routed(features, prior_logits, base_height)

    assert output["route_candidate_probability"].dtype == torch.float32
    assert not bool(torch.any(output["route_candidate_selected"]))
    for name in expected:
        torch.testing.assert_close(output[name], expected[name], rtol=0, atol=0)


def test_scene_router_mask_excludes_invalid_padding_from_descriptor() -> None:
    router = ConservativeSceneRouter(1, hidden_features=2)
    features = torch.tensor([[[[1.0, 3.0], [1000.0, 1000.0]]]])
    valid = torch.tensor([[[True, True], [False, False]]])
    descriptor = router(features, valid)["scene_descriptor"]

    torch.testing.assert_close(
        descriptor,
        torch.tensor([[2.0, 1.0]]),
        atol=1e-7,
        rtol=0,
    )


def test_scene_endpoint_selection_is_exact_and_uniform_per_image() -> None:
    fallback = {
        "domain_logits": torch.zeros(2, 3, 4, 5),
        "building_height": torch.zeros(2, 1, 4, 5),
    }
    candidate = {name: torch.ones_like(value) for name, value in fallback.items()}
    selected = select_scene_endpoint(
        fallback,
        candidate,
        torch.tensor([[False], [True]]),
    )

    for value in selected.values():
        assert torch.count_nonzero(value[0]) == 0
        assert torch.all(value[1] == 1)


def test_dual_router_preserves_both_frozen_endpoints_and_only_router_trains() -> None:
    fallback_model = _surface_model()
    candidate_model = deepcopy(fallback_model)
    with torch.no_grad():
        candidate_model.domain_head.bias.add_(torch.tensor((0.4, -0.2, 0.8)))
        candidate_model.building_residual_head.bias.add_(0.5)
        candidate_model.canopy_height_head.bias.add_(1.0)
    fallback_pack = FrozenSurfaceHeadPack.from_model(fallback_model)
    candidate_pack = FrozenSurfaceHeadPack.from_model(candidate_model)
    router = ConservativeSceneRouter(8, hidden_features=4)
    routed = FrozenDualSurfaceHeadRouter(fallback_pack, candidate_pack, router)
    routed.train()

    features = torch.randn(2, 8, 9, 9)
    prior_logits = torch.randn(2, 3, 9, 9)
    base_height = torch.rand(2, 1, 9, 9)
    fallback = fallback_pack(features, prior_logits, base_height)
    candidate = candidate_pack(features, prior_logits, base_height)
    protected_result = routed(features, prior_logits, base_height)
    for name in fallback:
        torch.testing.assert_close(
            protected_result[name], fallback[name], rtol=0, atol=0
        )

    final_layer = router.network[-1]
    assert isinstance(final_layer, nn.Linear)
    with torch.no_grad():
        final_layer.weight.zero_()
        final_layer.bias.fill_(20.0)
    candidate_result = routed(features, prior_logits, base_height)
    for name in candidate:
        torch.testing.assert_close(
            candidate_result[name], candidate[name], rtol=0, atol=0
        )

    trainable = {
        name for name, parameter in routed.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("router.") for name in trainable)
    assert not routed.fallback.training
    assert not routed.candidate.training

    routed.zero_grad(set_to_none=True)
    train_features = torch.randn(2, 8, 9, 9, requires_grad=True)
    route_output = routed(train_features, prior_logits, base_height)
    F.binary_cross_entropy_with_logits(
        route_output["route_candidate_logit"], torch.ones(2, 1)
    ).backward()
    assert train_features.grad is None
    assert all(parameter.grad is None for parameter in routed.fallback.parameters())
    assert all(parameter.grad is None for parameter in routed.candidate.parameters())
    assert any(parameter.grad is not None for parameter in routed.router.parameters())


def test_whole_scene_router_exactly_recovers_protected_and_candidate_models() -> None:
    protected_model = _surface_model().eval()
    protected_model.fusion_mode = "protected_vegetation"
    protected_model.building_protection_power = 2.5
    protected_model.vegetation_fusion_temperature = 0.8
    protected_model.vegetation_expert_fusion_threshold = 0.85
    protected_model.vegetation_expert_fusion_strength = 0.1
    candidate_model = deepcopy(protected_model).eval()
    candidate_model.fusion_mode = "calibrated_surface"
    candidate_model.building_fusion_min_height_m = 0.2
    candidate_model.building_fusion_temperature_m = 1.3
    candidate_model.building_fusion_score_threshold = 0.2
    candidate_model.building_fusion_score_temperature = 0.07
    candidate_model.building_fusion_strength = 0.65
    with torch.no_grad():
        candidate_model.domain_head.bias.add_(torch.tensor((0.4, -0.2, 0.8)))
        candidate_model.building_residual_head.bias.add_(0.5)
        candidate_model.canopy_height_head.bias.add_(1.0)

    protected_pack = FrozenSurfaceHeadPack.from_model(protected_model)
    candidate_pack = FrozenSurfaceHeadPack.from_model(candidate_model)
    scene_router = ConservativeSceneRouter(8, hidden_features=4)
    routed_heads = FrozenDualSurfaceHeadRouter(
        protected_pack, candidate_pack, scene_router
    )
    model = RoutedDomainGatedSurfaceNet(protected_model, routed_heads).eval()
    image = torch.randn(2, 3, 17, 21)
    prior = torch.rand(2, 1, 17, 21)

    with torch.inference_mode():
        protected_reference = protected_model(image, prior)
        candidate_reference = candidate_model(image, prior)
        protected_output = model(image, prior)

    for name, reference in protected_reference.items():
        torch.testing.assert_close(
            protected_output[name], reference, rtol=0, atol=0
        )
    assert not bool(torch.any(protected_output["route_candidate_selected"]))

    final_layer = scene_router.network[-1]
    assert isinstance(final_layer, nn.Linear)
    with torch.no_grad():
        final_layer.weight.zero_()
        final_layer.bias.fill_(20.0)
    with torch.inference_mode():
        candidate_output = model(image, prior)

    for name, reference in candidate_reference.items():
        torch.testing.assert_close(candidate_output[name], reference, rtol=0, atol=0)
    assert bool(torch.all(candidate_output["route_candidate_selected"]))
    assert all(
        name.startswith("endpoint_router.router.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
