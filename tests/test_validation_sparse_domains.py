"""Source-specific evaluation must tolerate missing classes, not missing truth."""

import importlib.util
from pathlib import Path

import pytest
import torch

from msr.training.losses import MultiDomainSurfaceLoss


SPEC = importlib.util.spec_from_file_location(
    "validate_sparse_domains", Path(__file__).parents[1] / "scripts/train_multidomain.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConstantSurface(torch.nn.Module):
    def forward(self, image, relative_prior):
        height = torch.full_like(image[:, :1], 7.0)
        logits = torch.zeros_like(image)
        zeros = torch.zeros_like(height)
        return {
            "height": height,
            "base_height": height,
            "building_height": height,
            "canopy_height": height,
            "domain_logits": logits,
            "domain_probabilities": logits.softmax(dim=1),
            "protected_building_logits": zeros,
            "refinement_logits": zeros,
            "refinement_strength": zeros,
        }


def _batch(domain, valid=True, region="supported"):
    domain_target = torch.full((1, 4, 4), domain, dtype=torch.long)
    height = torch.full((1, 1, 4, 4), 10.0)
    return {
        "image": torch.zeros(1, 3, 4, 4),
        "relative_prior": torch.zeros_like(height),
        "height": height,
        "regression_mask": torch.full_like(height, valid, dtype=torch.bool),
        "domain_target": domain_target,
        "domain_valid_mask": torch.ones_like(domain_target, dtype=torch.bool),
        "landscape": ["urban" if domain == 1 else "forest"],
        "region": [region],
    }


@pytest.mark.parametrize("domain,name,missing,scalar", [
    (1, "building", "vegetation", "canopy_expert_rmse_m"),
    (2, "vegetation", "building", "building_expert_rmse_m"),
])
def test_single_domain_suite_reports_only_supported_expert(domain, name, missing, scalar):
    _, metrics = MODULE.validate(ConstantSurface(), [_batch(domain)],
                                 MultiDomainSurfaceLoss(), torch.device("cpu"), precision="fp32")
    assert metrics["pixel_count"] == 16
    assert metrics["rmse_m"] == pytest.approx(3.0)
    assert set(metrics["domains"]) == {name}
    assert set(metrics["experts"]) == {name}
    assert metrics["unavailable_expert_metrics"] == {missing: "no_valid_reference_pixels"}
    assert scalar not in metrics


def test_ground_only_has_no_expert_score_but_valid_height_result():
    _, metrics = MODULE.validate(ConstantSurface(), [_batch(0)],
                                 MultiDomainSurfaceLoss(), torch.device("cpu"), precision="fp32")
    assert metrics["rmse_m"] == pytest.approx(3.0)
    assert metrics["experts"] == {}
    assert set(metrics["unavailable_expert_metrics"]) == {"building", "vegetation"}


def test_unsupported_region_is_excluded_without_inventing_zero_height():
    _, metrics = MODULE.validate(ConstantSurface(), [_batch(1, False, "unknown"), _batch(1)],
                                 MultiDomainSurfaceLoss(), torch.device("cpu"), precision="fp32")
    assert metrics["pixel_count"] == 16
    assert set(metrics["regions"]) == {"supported"}
    assert metrics["rmse_m"] == pytest.approx(3.0)


def test_wholly_unlabelled_evaluation_still_fails_closed():
    with pytest.raises(ValueError, match="No valid pixels"):
        MODULE.validate(ConstantSurface(), [_batch(1, False)],
                        MultiDomainSurfaceLoss(), torch.device("cpu"), precision="fp32")
