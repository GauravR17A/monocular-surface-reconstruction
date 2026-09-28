from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import torch

from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.height_net import HeightNet
from msr.models.stage3_endpoints import (
    Stage3CheckpointValidationError,
    load_stage3_endpoint_packs,
)


def _state() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260912)
    return {
        "base_model.encoder.weight": torch.rand(4, 3, 3, 3, generator=generator),
        "adapter.block.weight": torch.rand(8, 6, 3, 3, generator=generator),
        "refinement_strength_head.weight": torch.rand(
            1, 8, 1, 1, generator=generator
        ),
        "domain_head.weight": torch.rand(3, 8, 1, 1, generator=generator),
        "domain_head.bias": torch.rand(3, generator=generator),
        "building_residual_head.weight": torch.rand(
            1, 8, 1, 1, generator=generator
        ),
        "building_residual_head.bias": torch.rand(1, generator=generator),
        "canopy_height_head.weight": torch.rand(1, 8, 1, 1, generator=generator),
        "canopy_height_head.bias": torch.rand(1, generator=generator),
    }


def _payload(
    state: dict[str, torch.Tensor],
    *,
    epoch: int,
    initial_checkpoint: str | None = None,
) -> dict:
    model_config = {
        "base_checkpoint": "experiments/base/checkpoint.pt",
        "hidden_channels": 8,
        "maximum_building_residual_m": 17.0,
        "initial_canopy_height_m": 8.0,
        "initial_refinement_strength": 0.08,
        "fusion_mode": "protected_vegetation",
        "building_protection_power": 2.5,
        "building_fusion_min_height_m": 7.0,
        "building_fusion_temperature_m": 1.25,
        "building_fusion_score_threshold": 0.35,
        "building_fusion_score_temperature": 0.08,
        "building_fusion_strength": 0.65,
        "vegetation_fusion_temperature": 0.8,
        "vegetation_expert_fusion_threshold": 0.85,
        "vegetation_expert_fusion_strength": 0.1,
    }
    if initial_checkpoint is not None:
        model_config["initial_checkpoint"] = initial_checkpoint
    return {
        "model_type": "domain_gated_surface_v2",
        "epoch": epoch,
        "model": state,
        "config": {"model": model_config},
    }


def _checkpoint_pair(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    protected_state = _state()
    gamus_state = deepcopy(protected_state)
    with torch.no_grad():
        gamus_state["domain_head.weight"].add_(0.25)
        gamus_state["domain_head.bias"].sub_(0.1)
        gamus_state["building_residual_head.weight"].add_(0.5)
        gamus_state["building_residual_head.bias"].add_(0.2)
        gamus_state["canopy_height_head.weight"].sub_(0.3)
        gamus_state["canopy_height_head.bias"].add_(0.4)
    protected_payload = _payload(protected_state, epoch=20)
    gamus_payload = _payload(
        gamus_state,
        epoch=8,
        initial_checkpoint="experiments/protected/checkpoint.pt",
    )
    protected_path = tmp_path / "protected.pt"
    gamus_path = tmp_path / "gamus.pt"
    torch.save(protected_payload, protected_path)
    torch.save(gamus_payload, gamus_path)
    return protected_path, gamus_path, protected_payload, gamus_payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_surface_checkpoint_pair(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    DomainGatedSurfaceNet,
    DomainGatedSurfaceNet,
]:
    torch.manual_seed(20260912)
    backbone = "mobilenetv3_small_050"
    decoder_channels = (32, 24, 16, 8, 4)
    base_model = HeightNet(
        backbone=backbone,
        pretrained=False,
        decoder_channels=decoder_channels,
        auxiliary_building_head=True,
    )
    base_path = tmp_path / "base.pt"
    torch.save(
        {
            "model": deepcopy(base_model.state_dict()),
            "config": {
                "model": {
                    "backbone": backbone,
                    "pretrained": False,
                    "decoder_channels": list(decoder_channels),
                    "auxiliary_building_head": True,
                }
            },
        },
        base_path,
    )
    protected_model = DomainGatedSurfaceNet(
        deepcopy(base_model),
        hidden_channels=8,
        maximum_building_residual_m=17.0,
        initial_canopy_height_m=8.0,
        initial_refinement_strength=0.08,
        fusion_mode="protected_vegetation",
        building_protection_power=2.5,
        building_fusion_min_height_m=7.0,
        building_fusion_temperature_m=1.25,
        building_fusion_score_threshold=0.35,
        building_fusion_score_temperature=0.08,
        building_fusion_strength=0.65,
        vegetation_fusion_temperature=0.8,
        vegetation_expert_fusion_threshold=0.85,
        vegetation_expert_fusion_strength=0.1,
        freeze_base=True,
    ).eval()
    candidate_model = deepcopy(protected_model).eval()
    with torch.no_grad():
        for head in (
            candidate_model.domain_head,
            candidate_model.building_residual_head,
            candidate_model.canopy_height_head,
        ):
            head.weight.add_(0.03)
            head.bias.sub_(0.02)

    protected_payload = _payload(
        deepcopy(protected_model.state_dict()),
        epoch=20,
    )
    candidate_payload = _payload(
        deepcopy(candidate_model.state_dict()),
        epoch=8,
        initial_checkpoint=str(tmp_path / "protected.pt"),
    )
    protected_payload["config"]["model"]["base_checkpoint"] = str(base_path)
    candidate_payload["config"]["model"]["base_checkpoint"] = str(base_path)
    protected_path = tmp_path / "protected.pt"
    candidate_path = tmp_path / "gamus.pt"
    torch.save(protected_payload, protected_path)
    torch.save(candidate_payload, candidate_path)
    return (
        protected_path,
        candidate_path,
        base_path,
        protected_model,
        candidate_model,
    )


def test_reconstructed_shared_model_and_packs_exactly_reproduce_sources(
    tmp_path: Path,
) -> None:
    (
        protected_path,
        gamus_path,
        base_path,
        protected_model,
        gamus_model,
    ) = _full_surface_checkpoint_pair(tmp_path)
    endpoints = load_stage3_endpoint_packs(
        protected_path,
        gamus_path,
        expected_protected_sha256=_sha256(protected_path),
        expected_gamus_stage1_sha256=_sha256(gamus_path),
        expected_base_sha256=_sha256(base_path),
    )
    shared_model = endpoints.require_shared_model()
    assert not shared_model.training
    assert all(not parameter.requires_grad for parameter in shared_model.parameters())
    assert endpoints.diagnostics.base_checkpoint is not None
    assert endpoints.diagnostics.base_checkpoint.sha256 == _sha256(base_path)

    image = torch.randn(2, 3, 64, 64)
    relative_prior = torch.rand(2, 1, 64, 64)
    complete_names = (
        "height",
        "gated_height",
        "refinement_strength",
        "effective_refinement_strength",
        "building_protection",
        "building_fusion_gate",
        "vegetation_fusion_probability",
        "vegetation_expert_fusion_gate",
        "domain_logits",
        "domain_probabilities",
        "building_height",
        "canopy_height",
        "building_logits",
        "protected_building_logits",
        "vegetation_logits",
        "refinement_logits",
        "log_variance",
        "base_height",
    )
    with torch.inference_mode():
        protected_reference = protected_model(image, relative_prior)
        gamus_reference = gamus_model(image, relative_prior)
        reconstructed = shared_model(image, relative_prior)
        protected_endpoint = endpoints.protected.forward_fused(
            protected_reference["adapter_features"],
            protected_reference["prior_domain_logits"],
            protected_reference["base_height"],
            protected_building_logits=protected_reference[
                "protected_building_logits"
            ],
            refinement_logits=protected_reference["refinement_logits"],
            log_variance=protected_reference["log_variance"],
        )
        gamus_endpoint = endpoints.gamus_stage1.forward_fused(
            gamus_reference["adapter_features"],
            gamus_reference["prior_domain_logits"],
            gamus_reference["base_height"],
            protected_building_logits=gamus_reference["protected_building_logits"],
            refinement_logits=gamus_reference["refinement_logits"],
            log_variance=gamus_reference["log_variance"],
        )

    for name in protected_reference:
        torch.testing.assert_close(
            reconstructed[name], protected_reference[name], rtol=0, atol=0
        )
    for endpoint, reference in (
        (protected_endpoint, protected_reference),
        (gamus_endpoint, gamus_reference),
    ):
        for name in complete_names:
            torch.testing.assert_close(endpoint[name], reference[name], rtol=0, atol=0)


def test_loads_validated_independent_frozen_head_packs(tmp_path: Path) -> None:
    protected_path, gamus_path, protected_payload, gamus_payload = _checkpoint_pair(
        tmp_path
    )

    endpoints = load_stage3_endpoint_packs(
        protected_path,
        gamus_path,
        expected_protected_sha256=_sha256(protected_path),
        expected_gamus_stage1_sha256=_sha256(gamus_path),
        reconstruct_shared_model=False,
    )

    assert not endpoints.protected.training
    assert not endpoints.gamus_stage1.training
    assert endpoints.shared_model is None
    assert endpoints.diagnostics.base_checkpoint is None
    assert all(not value.requires_grad for value in endpoints.protected.parameters())
    assert all(not value.requires_grad for value in endpoints.gamus_stage1.parameters())
    torch.testing.assert_close(
        endpoints.protected.domain_head.weight,
        protected_payload["model"]["domain_head.weight"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        endpoints.gamus_stage1.canopy_height_head.bias,
        gamus_payload["model"]["canopy_height_head.bias"],
        rtol=0,
        atol=0,
    )
    assert endpoints.protected.domain_head.weight.data_ptr() != protected_payload[
        "model"
    ]["domain_head.weight"].data_ptr()
    assert endpoints.diagnostics.protected.sha256 == _sha256(protected_path)
    assert endpoints.diagnostics.gamus_stage1.sha256 == _sha256(gamus_path)
    assert endpoints.diagnostics.compatibility.shared_tensor_count == 3
    assert len(endpoints.diagnostics.compatibility.differing_head_tensors) == 6
    assert not endpoints.diagnostics.compatibility.identical_head_tensors
    assert len(endpoints.diagnostics.compatibility.shared_state_sha256) == 64
    assert endpoints.diagnostics.as_dict()["protected"]["epoch"] == 20
    assert endpoints.protected.fusion_mode == "protected_vegetation"
    assert endpoints.protected.building_protection_power == 2.5
    assert endpoints.protected.building_fusion_min_height_m == 7.0
    assert endpoints.protected.building_fusion_temperature_m == 1.25
    assert endpoints.protected.building_fusion_score_threshold == 0.35
    assert endpoints.protected.building_fusion_score_temperature == 0.08
    assert endpoints.protected.building_fusion_strength == 0.65
    assert endpoints.protected.vegetation_fusion_temperature == 0.8
    assert endpoints.protected.vegetation_expert_fusion_threshold == 0.85
    assert endpoints.protected.vegetation_expert_fusion_strength == 0.1


def test_rejects_any_difference_outside_allowed_heads(tmp_path: Path) -> None:
    protected_path, gamus_path, _, gamus_payload = _checkpoint_pair(tmp_path)
    gamus_payload["model"]["adapter.block.weight"][0, 0, 0, 0] += 1.0
    torch.save(gamus_payload, gamus_path)

    with pytest.raises(
        Stage3CheckpointValidationError,
        match="outside the three permitted heads.*adapter.block.weight",
    ):
        load_stage3_endpoint_packs(protected_path, gamus_path)


def test_bit_identity_rejects_positive_vs_negative_zero(tmp_path: Path) -> None:
    protected_path, gamus_path, protected_payload, gamus_payload = _checkpoint_pair(
        tmp_path
    )
    protected_zero = torch.tensor(0.0)
    gamus_negative_zero = torch.tensor(-0.0)
    assert protected_zero == gamus_negative_zero
    protected_payload["model"]["shared_zero"] = protected_zero
    gamus_payload["model"]["shared_zero"] = gamus_negative_zero
    torch.save(protected_payload, protected_path)
    torch.save(gamus_payload, gamus_path)

    with pytest.raises(
        Stage3CheckpointValidationError, match="outside the three permitted heads"
    ):
        load_stage3_endpoint_packs(protected_path, gamus_path)


def test_rejects_model_architecture_config_difference(tmp_path: Path) -> None:
    protected_path, gamus_path, _, gamus_payload = _checkpoint_pair(tmp_path)
    gamus_payload["config"]["model"]["hidden_channels"] = 16
    torch.save(gamus_payload, gamus_path)

    with pytest.raises(
        Stage3CheckpointValidationError,
        match="architecture configs differ.*hidden_channels",
    ):
        load_stage3_endpoint_packs(protected_path, gamus_path)


def test_rejects_state_architecture_or_unexpected_head_tensor(tmp_path: Path) -> None:
    protected_path, gamus_path, _, gamus_payload = _checkpoint_pair(tmp_path)
    gamus_payload["model"]["domain_head.extra"] = torch.zeros(1)
    torch.save(gamus_payload, gamus_path)

    with pytest.raises(
        Stage3CheckpointValidationError, match="exactly the required three heads"
    ):
        load_stage3_endpoint_packs(protected_path, gamus_path)


def test_rejects_hash_mismatch_before_returning_packs(tmp_path: Path) -> None:
    protected_path, gamus_path, _, _ = _checkpoint_pair(tmp_path)

    with pytest.raises(Stage3CheckpointValidationError, match="SHA-256 mismatch"):
        load_stage3_endpoint_packs(
            protected_path,
            gamus_path,
            expected_protected_sha256="0" * 64,
        )


def test_rejects_nonfinite_head_weights(tmp_path: Path) -> None:
    protected_path, gamus_path, _, gamus_payload = _checkpoint_pair(tmp_path)
    gamus_payload["model"]["canopy_height_head.bias"][0] = float("nan")
    torch.save(gamus_payload, gamus_path)

    with pytest.raises(Stage3CheckpointValidationError, match="non-finite"):
        load_stage3_endpoint_packs(protected_path, gamus_path)


def test_rejects_invalid_fusion_policy_before_returning_packs(tmp_path: Path) -> None:
    protected_path, gamus_path, protected_payload, gamus_payload = _checkpoint_pair(
        tmp_path
    )
    protected_payload["config"]["model"]["vegetation_expert_fusion_strength"] = 1.1
    gamus_payload["config"]["model"]["vegetation_expert_fusion_strength"] = 1.1
    torch.save(protected_payload, protected_path)
    torch.save(gamus_payload, gamus_path)

    with pytest.raises(
        Stage3CheckpointValidationError,
        match="vegetation_expert_fusion_strength must be within",
    ):
        load_stage3_endpoint_packs(
            protected_path,
            gamus_path,
            reconstruct_shared_model=False,
        )
