from __future__ import annotations

import inspect
import math

import pytest
import torch

from msr.training.stage4c_semantic_features import (
    STAGE4C_QUANTILES,
    STAGE4C_SEMANTIC_DESCRIPTOR_SIZE,
    STAGE4C_SEMANTIC_FEATURE_SCHEMA,
    Stage4CSemanticFeatureExtractor,
    stage4c_semantic_descriptor_size,
    stage4c_semantic_feature_names,
)


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def _normalized_rgb(unit_rgb: torch.Tensor) -> torch.Tensor:
    return (unit_rgb - IMAGENET_MEAN.to(unit_rgb)) / IMAGENET_STD.to(unit_rgb)


def _inputs(
    *,
    batch: int = 2,
    adapter_channels: int = 64,
    height: int = 4,
    width: int = 6,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(4107)
    adapter = torch.randn(
        batch,
        adapter_channels,
        height,
        width,
        generator=generator,
    )
    rgb_unit = torch.rand(batch, 3, height, width, generator=generator)
    rgb = _normalized_rgb(rgb_unit)
    prior = torch.rand(batch, 1, height, width, generator=generator)
    protected_logits = torch.randn(
        batch, 3, height, width, generator=generator
    )
    candidate_logits = torch.randn(
        batch, 3, height, width, generator=generator
    )
    protected_height = torch.rand(
        batch, 1, height, width, generator=generator
    ) * 20.0
    candidate_height = protected_height + torch.randn(
        batch, 1, height, width, generator=generator
    )
    return tuple(
        value.to(dtype=dtype)
        for value in (
            adapter,
            rgb,
            prior,
            protected_logits,
            candidate_logits,
            protected_height,
            candidate_height,
        )
    )


def _feature_map(
    extractor: Stage4CSemanticFeatureExtractor,
    descriptor: torch.Tensor,
) -> dict[str, float]:
    return dict(zip(extractor.feature_names, descriptor.tolist()))


def test_canonical_layout_is_fixed_unique_and_label_free() -> None:
    names = stage4c_semantic_feature_names()

    assert STAGE4C_SEMANTIC_FEATURE_SCHEMA.endswith(".v1")
    assert len(names) == len(set(names)) == 254
    assert STAGE4C_SEMANTIC_DESCRIPTOR_SIZE == 254
    assert stage4c_semantic_descriptor_size() == 254
    assert stage4c_semantic_descriptor_size(5) == 136
    assert names[:2] == ("adapter_mean_c0", "adapter_mean_c1")
    assert names[-1] == "height_delta_q90"
    assert not any(
        forbidden in name
        for name in names
        for forbidden in ("source", "city", "group", "reference", "target", "valid_mask")
    )

    parameters = set(inspect.signature(Stage4CSemanticFeatureExtractor.forward).parameters)
    assert parameters == {
        "self",
        "adapter_features",
        "rgb",
        "relative_prior",
        "protected_domain_logits",
        "candidate_domain_logits",
        "protected_height",
        "candidate_height",
    }


def test_batch_output_is_deterministic_and_matches_individual_scenes() -> None:
    extractor = Stage4CSemanticFeatureExtractor()
    inputs = _inputs()

    first = extractor(*inputs)
    second = extractor(*inputs)
    individual = torch.stack(
        [extractor(*(value[index] for value in inputs)) for index in range(2)]
    )

    assert first.shape == (2, 254)
    assert first.dtype == torch.float32
    assert torch.all(torch.isfinite(first))
    assert torch.equal(first, second)
    # Batched and per-scene softmax kernels may differ by a final float32 ULP,
    # but the descriptor must not acquire any batch-composition dependence.
    assert torch.allclose(first, individual, atol=1.0e-7, rtol=1.0e-7)
    assert extractor(*[value[0] for value in inputs]).shape == (254,)


def test_known_global_quantiles_probability_entropy_and_height_features() -> None:
    adapter = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 5.0], [5.0, 5.0]]]]
    )
    rgb_unit = torch.tensor([0.2, 0.6, 0.1]).view(1, 3, 1, 1).expand(1, 3, 2, 2)
    rgb = _normalized_rgb(rgb_unit)
    prior = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])
    logits = torch.zeros(1, 3, 2, 2)
    protected_height = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    candidate_height = protected_height + 2.0
    extractor = Stage4CSemanticFeatureExtractor(adapter_channels=2)

    descriptor = extractor(
        adapter,
        rgb,
        prior,
        logits,
        logits,
        protected_height,
        candidate_height,
    )[0]
    values = _feature_map(extractor, descriptor)

    assert extractor.descriptor_size == 130
    assert values["adapter_mean_c0"] == pytest.approx(2.5)
    assert values["adapter_std_c0"] == pytest.approx(math.sqrt(1.25))
    assert values["adapter_mean_c1"] == pytest.approx(5.0)
    assert values["adapter_std_c1"] == pytest.approx(0.0)
    assert values["rgb_mean_red"] == pytest.approx(0.2, abs=1.0e-6)
    assert values["rgb_mean_green"] == pytest.approx(0.6, abs=1.0e-6)
    assert values["rgb_mean_blue"] == pytest.approx(0.1, abs=1.0e-6)
    assert values["exg_q50"] == pytest.approx(0.9, abs=1.0e-6)
    assert values["vari_q50"] == pytest.approx(4.0 / 7.0, abs=1.0e-6)
    expected_prior = torch.quantile(prior.flatten(), torch.tensor(STAGE4C_QUANTILES))
    for quantile, expected in zip(STAGE4C_QUANTILES, expected_prior):
        suffix = f"q{round(quantile * 100):02d}"
        assert values[f"relative_prior_{suffix}"] == pytest.approx(float(expected))
    for class_name in ("ground", "building", "vegetation"):
        assert values[f"protected_prob_mean_{class_name}"] == pytest.approx(1.0 / 3.0)
        assert values[f"protected_prob_std_{class_name}"] == pytest.approx(0.0)
        assert values[f"protected_hard_fraction_{class_name}"] == pytest.approx(
            1.0 if class_name == "ground" else 0.0
        )
        for row in range(2):
            for column in range(2):
                assert values[
                    f"protected_pyramid_r{row}c{column}_{class_name}"
                ] == pytest.approx(1.0 / 3.0)
    assert values["protected_entropy_mean"] == pytest.approx(1.0)
    assert values["protected_entropy_std"] == pytest.approx(0.0)
    assert values[
        "hard_disagreement_protected_ground_candidate_ground"
    ] == pytest.approx(1.0)
    assert sum(
        value for name, value in values.items() if name.startswith("hard_disagreement_")
    ) == pytest.approx(1.0)
    for quantile in STAGE4C_QUANTILES:
        assert values[f"height_delta_q{round(quantile * 100):02d}"] == pytest.approx(2.0)


def test_hard_disagreement_and_spatial_pyramid_have_declared_orientation() -> None:
    inputs = list(_inputs(batch=1, adapter_channels=1, height=2, width=2))
    protected_classes = torch.tensor([[[0, 0], [1, 2]]])
    candidate_classes = torch.tensor([[[0, 1], [2, 2]]])

    def decisive_logits(classes: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, 3, 2, 2), -20.0)
        return logits.scatter_(1, classes[:, None], 20.0)

    inputs[3] = decisive_logits(protected_classes)
    inputs[4] = decisive_logits(candidate_classes)
    extractor = Stage4CSemanticFeatureExtractor(adapter_channels=1)
    values = _feature_map(extractor, extractor(*inputs)[0])

    expected_pairs = {
        ("ground", "ground"),
        ("ground", "building"),
        ("building", "vegetation"),
        ("vegetation", "vegetation"),
    }
    for protected in ("ground", "building", "vegetation"):
        for candidate in ("ground", "building", "vegetation"):
            assert values[
                f"hard_disagreement_protected_{protected}_candidate_{candidate}"
            ] == pytest.approx(0.25 if (protected, candidate) in expected_pairs else 0.0)
    assert values["protected_hard_fraction_ground"] == pytest.approx(0.5)
    assert values["candidate_hard_fraction_vegetation"] == pytest.approx(0.5)
    assert values["protected_pyramid_r0c0_ground"] == pytest.approx(1.0)
    assert values["protected_pyramid_r1c0_building"] == pytest.approx(1.0)
    assert values["candidate_pyramid_r0c1_building"] == pytest.approx(1.0)
    assert values["candidate_pyramid_r1c1_vegetation"] == pytest.approx(1.0)


def test_vari_uses_a_finite_neutral_value_when_denominator_is_undefined() -> None:
    inputs = list(_inputs(batch=1, adapter_channels=1, height=2, width=2))
    # G + R - B = 0 while G - R is non-zero.
    rgb_unit = torch.tensor((0.5, 0.25, 0.75)).view(1, 3, 1, 1).expand(1, 3, 2, 2)
    inputs[1] = _normalized_rgb(rgb_unit)
    extractor = Stage4CSemanticFeatureExtractor(adapter_channels=1)

    values = _feature_map(extractor, extractor(*inputs)[0])

    for quantile in STAGE4C_QUANTILES:
        assert values[f"vari_q{round(quantile * 100):02d}"] == pytest.approx(0.0)


def test_mixed_precision_inputs_are_computed_as_finite_float32() -> None:
    extractor = Stage4CSemanticFeatureExtractor()
    float_inputs = _inputs(batch=1)
    half_inputs = tuple(value.half() for value in float_inputs)

    output = extractor(*half_inputs)

    assert output.dtype == torch.float32
    assert output.shape == (1, 254)
    assert torch.all(torch.isfinite(output))
    # Half-precision input quantization can move metric-height quantiles by a
    # few millimetres; extraction itself is still performed in float32.
    assert torch.allclose(output, extractor(*float_inputs), atol=1.0e-2, rtol=3.0e-3)


@pytest.mark.parametrize(
    ("index", "replacement", "message"),
    [
        (0, torch.zeros(1, 63, 4, 6), "adapter_features must have 64 channels"),
        (1, torch.zeros(1, 4, 4, 6), "rgb must have 3 channels"),
        (2, torch.zeros(1, 2, 4, 6), "relative_prior must have 1 channels"),
        (3, torch.zeros(1, 2, 4, 6), "protected_domain_logits must have 3 channels"),
        (5, torch.zeros(1, 2, 4, 6), "protected_height must have 1 channels"),
        (6, torch.zeros(1, 1, 3, 6), "must match adapter_features"),
    ],
)
def test_shape_contract_fails_closed(
    index: int,
    replacement: torch.Tensor,
    message: str,
) -> None:
    inputs = list(_inputs(batch=1))
    inputs[index] = replacement

    with pytest.raises(ValueError, match=message):
        Stage4CSemanticFeatureExtractor()(*inputs)


@pytest.mark.parametrize("index", range(7))
def test_every_input_rejects_non_finite_values(index: int) -> None:
    inputs = list(_inputs(batch=1))
    inputs[index] = inputs[index].clone()
    inputs[index].flatten()[0] = torch.nan

    with pytest.raises(ValueError, match="only finite values"):
        Stage4CSemanticFeatureExtractor()(*inputs)


def test_rank_and_schema_configuration_fail_closed() -> None:
    inputs = list(_inputs(batch=1))
    inputs[0] = inputs[0][0]
    with pytest.raises(ValueError, match="all be unbatched.*or all be batched"):
        Stage4CSemanticFeatureExtractor()(*inputs)

    with pytest.raises(ValueError, match="schema-fixed"):
        Stage4CSemanticFeatureExtractor(quantiles=(0.5,))
    with pytest.raises(ValueError, match="positive"):
        Stage4CSemanticFeatureExtractor(adapter_channels=0)
    with pytest.raises(ValueError, match="positive"):
        stage4c_semantic_feature_names(0)
