from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from msr.evaluation.routed_validation import (
    evaluate_dense_surface_model,
    evaluate_validation_guards,
)
from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.routed_surface import (
    ConservativeSceneRouter,
    FrozenDualSurfaceHeadRouter,
    FrozenSurfaceHeadPack,
    RoutedDomainGatedSurfaceNet,
)


class _TinyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "height": F.softplus(image[:, :1] * self.scale),
            "building_logits": image[:, 1:2] * self.scale,
        }


def _surface_model() -> DomainGatedSurfaceNet:
    return DomainGatedSurfaceNet(
        _TinyBase(),
        hidden_channels=4,
        fusion_mode="protected_vegetation",
        vegetation_fusion_temperature=0.8,
        vegetation_expert_fusion_threshold=0.85,
        vegetation_expert_fusion_strength=0.1,
        freeze_base=True,
    )


def _samples() -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for index, landscape in enumerate(("urban", "forest")):
        domain = torch.tensor(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 2, 2],
                [2, 2, 2, 2],
            ],
            dtype=torch.long,
        )
        samples.append(
            {
                "image": torch.full((3, 4, 4), 0.1 * (index + 1)),
                "height": torch.ones(1, 4, 4),
                "valid_mask": torch.ones(1, 4, 4, dtype=torch.bool),
                "regression_mask": torch.ones(1, 4, 4, dtype=torch.bool),
                "domain_target": domain,
                "domain_valid_mask": torch.ones(4, 4, dtype=torch.bool),
                "relative_prior": torch.zeros(1, 4, 4),
                "sample_id": f"scene-{index}",
                "region": f"region-{index}",
                "landscape": landscape,
            }
        )
    return samples


def test_dense_routed_metrics_include_endpoint_counts_and_probabilities() -> None:
    protected = _surface_model()
    candidate = deepcopy(protected)
    with torch.no_grad():
        candidate.domain_head.bias.add_(torch.tensor((0.0, 0.0, 1.0)))
    scene_router = ConservativeSceneRouter(4, hidden_features=2)
    final = scene_router.network[-1]
    assert isinstance(final, nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(20.0)
    routed = RoutedDomainGatedSurfaceNet(
        protected,
        FrozenDualSurfaceHeadRouter(
            FrozenSurfaceHeadPack.from_model(protected),
            FrozenSurfaceHeadPack.from_model(candidate),
            scene_router,
        ),
    )

    with patch.object(routed, "forward", wraps=routed.forward) as forward_spy:
        metrics = evaluate_dense_surface_model(
            routed,
            DataLoader(_samples(), batch_size=2),
        )

    assert all("valid_mask" not in call.kwargs for call in forward_spy.call_args_list)

    assert metrics["pixel_count"] == 32
    assert set(metrics["domains"]) == {"ground", "building", "vegetation"}
    assert set(metrics["landscapes"]) == {"urban", "forest"}
    assert set(metrics["regions"]) == {"region-0", "region-1"}
    assert set(metrics["experts"]) == {"building", "vegetation"}
    assert "building_expert_rmse_m" in metrics
    assert "canopy_expert_rmse_m" in metrics
    assert metrics["semantic_identification"]["total_valid_pixels"] == 32
    assert "f0_5" in metrics["building_router"]
    routing = metrics["scene_routing"]
    assert routing["scene_count"] == 2
    assert routing["protected_selected"] == 0
    assert routing["gamus_stage1_selected"] == 2
    assert routing["candidate_selection_rate"] == pytest.approx(1.0)
    assert routing["candidate_probability"]["mean"] > 0.999
    assert [row["sample_id"] for row in routing["per_scene"]] == [
        "scene-0",
        "scene-1",
    ]


def test_both_suite_guards_use_like_for_like_protected_baselines() -> None:
    protected = {
        "gamus": {
            "rmse_m": 7.0,
            "semantic_identification": {"macro_f1": 0.70},
        },
        "legacy": {
            "rmse_m": 6.0,
            "semantic_identification": {"macro_f1": 0.63},
        },
    }
    current = {
        "gamus": {
            "rmse_m": 7.01,
            "semantic_identification": {"macro_f1": 0.75},
        },
        "legacy": {
            "rmse_m": 6.01,
            "semantic_identification": {"macro_f1": 0.64},
        },
    }
    guards = {
        "gamus": {
            "semantic_identification.macro_f1": {"min": 0.74},
            "rmse_m": {"max_regression": 0.02, "baseline": "initial"},
        },
        "legacy": {
            "semantic_identification.macro_f1": {"min": 0.622},
            "rmse_m": {"max_regression": 0.02, "baseline": "initial"},
        },
    }

    report, passes = evaluate_validation_guards(current, protected, guards)

    assert passes
    assert report["passes"]
    assert set(report["suites"]) == {"gamus", "legacy"}
    assert report["suites"]["gamus"]["guards"]["rmse_m"][
        "protected_reference"
    ] == pytest.approx(7.0)


def test_guard_failure_or_missing_suite_fails_closed() -> None:
    protected = {"gamus": {"rmse_m": 7.0}}
    current = {"gamus": {"rmse_m": 7.1}}
    report, passes = evaluate_validation_guards(
        current,
        protected,
        {"gamus": {"rmse_m": {"max_regression": 0.02, "baseline": "initial"}}},
    )
    assert not passes
    assert not report["suites"]["gamus"]["passes"]

    with pytest.raises(KeyError, match="legacy.*no routed metrics"):
        evaluate_validation_guards(
            current,
            protected,
            {"legacy": {"rmse_m": {"max": 8.0}}},
        )
