from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.height_net import HeightNet


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "audit_height_output_identity.py"
SPEC = importlib.util.spec_from_file_location("height_output_identity_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_config(base_checkpoint: Path) -> dict:
    return {
        "base_checkpoint": str(base_checkpoint),
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


def _checkpoint_pair(tmp_path: Path) -> tuple[Path, Path]:
    torch.manual_seed(20260913)
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

    common = _model_config(base_path)
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
    protected_path = tmp_path / "protected.pt"
    protected_payload = {
        "model_type": "domain_gated_surface_v2",
        "epoch": 20,
        "model": deepcopy(protected_model.state_dict()),
        "config": {
            "data": {"rgb_scale": 255.0},
            "model": deepcopy(common),
        },
    }
    torch.save(protected_payload, protected_path)

    candidate_model = DomainGatedSurfaceNet(
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
        fine_semantic_classes=6,
        fine_semantic_head_type="hierarchical_vegetation",
        freeze_base=True,
    ).eval()
    candidate_model.load_legacy_compatible_state_dict(protected_payload["model"])
    with torch.no_grad():
        candidate_model.fine_semantic_head.coarse_classifier.weight.fill_(0.01)
        candidate_model.fine_semantic_head.vegetation_split.weight.fill_(0.02)
    candidate_config = deepcopy(common)
    candidate_config.update(
        {
            "initial_checkpoint": str(protected_path),
            "fine_semantic_classes": 6,
            "fine_semantic_head_type": "hierarchical_vegetation",
        }
    )
    candidate_path = tmp_path / "candidate.pt"
    torch.save(
        {
            "model_type": "domain_gated_surface_v4_six_class",
            "epoch": 3,
            "model": deepcopy(candidate_model.state_dict()),
            "config": {
                "protocol": {
                    "protected_checkpoint_sha256": _sha256(protected_path)
                },
                "data": {
                    "rgb_scale": 255.0,
                    "validation_radiometric_policy": "raw",
                },
                "model": candidate_config,
            },
        },
        candidate_path,
    )
    return protected_path, candidate_path


def test_inherited_state_is_bit_exact_not_merely_numerically_equal() -> None:
    protected = {"shared": torch.tensor(0.0)}
    candidate = {
        "shared": torch.tensor(-0.0),
        "fine_semantic_head.weight": torch.ones(1),
    }
    assert torch.equal(protected["shared"], candidate["shared"])
    with pytest.raises(ValueError, match="changed bit-for-bit"):
        MODULE.audit_inherited_state(protected, candidate)


def test_pipeline_contract_rejects_preprocessing_or_fusion_changes(
    tmp_path: Path,
) -> None:
    base_dependency = tmp_path / "base.pt"
    base_dependency.write_bytes(b"sealed base")
    protected_payload = {
        "config": {
            "data": {"rgb_scale": 255.0},
            "model": _model_config(base_dependency),
        }
    }
    candidate_payload = deepcopy(protected_payload)
    protected = MODULE.extract_pipeline_contract(
        protected_payload,
        checkpoint_path=tmp_path / "protected.pt",
        tile_size=64,
        overlap=16,
    )
    candidate = MODULE.extract_pipeline_contract(
        candidate_payload,
        checkpoint_path=tmp_path / "candidate.pt",
        tile_size=64,
        overlap=16,
    )
    assert MODULE.compare_pipeline_contracts(protected, candidate)["passes"] is True

    candidate_payload["config"]["model"]["vegetation_expert_fusion_strength"] = 0.2
    changed_fusion = MODULE.extract_pipeline_contract(
        candidate_payload,
        checkpoint_path=tmp_path / "candidate.pt",
        tile_size=64,
        overlap=16,
    )
    with pytest.raises(ValueError, match="fusion contract differs"):
        MODULE.compare_pipeline_contracts(protected, changed_fusion)

    candidate_payload = deepcopy(protected_payload)
    candidate_payload["config"]["data"]["rgb_scale"] = 10000.0
    changed_preprocessing = MODULE.extract_pipeline_contract(
        candidate_payload,
        checkpoint_path=tmp_path / "candidate.pt",
        tile_size=64,
        overlap=16,
    )
    with pytest.raises(ValueError, match="preprocessing contract differs"):
        MODULE.compare_pipeline_contracts(protected, changed_preprocessing)


def test_output_comparison_rejects_even_a_one_bit_height_change() -> None:
    protected = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float32)
    candidate = protected.clone()
    candidate.view(torch.int32)[0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="height-path output changed"):
        MODULE._assert_tensor_identity("height", protected, candidate)


def test_full_audit_executes_preprocessing_fusion_and_output_guards(
    tmp_path: Path,
) -> None:
    protected_path, candidate_path = _checkpoint_pair(tmp_path)

    report = MODULE.build_report(
        protected_path,
        candidate_path,
        tile_size=64,
        overlap=16,
        expected_protected_sha256=_sha256(protected_path),
        expected_candidate_sha256=_sha256(candidate_path),
    )

    assert report["passes"] is True
    assert report["state_audit"]["passes"] is True
    assert report["pipeline_contract_audit"]["passes"] is True
    assert report["loaded_model_audit"]["passes"] is True
    assert report["direct_output_audit"]["passes"] is True
    assert len(report["direct_output_audit"]["outputs"]) == len(
        MODULE.SHARED_FORWARD_OUTPUTS
    )
    assert report["app_output_audit"]["passes"] is True
    assert len(report["app_output_audit"]["cases"]) == 3
    assert report["app_pointer"]["inspected"] is False
    assert report["promotion_performed"] is False
