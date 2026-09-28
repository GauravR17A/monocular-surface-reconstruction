from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import re

import pytest
import torch
import yaml

from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.height_net import HeightNet


ROOT = Path(__file__).parents[1]
AUDITOR_PATH = ROOT / "scripts" / "audit_gamus_spatial_refined_head_checkpoint.py"
AUDITOR_SPEC = importlib.util.spec_from_file_location(
    "spatial_refined_head_audit_test", AUDITOR_PATH
)
assert AUDITOR_SPEC and AUDITOR_SPEC.loader
AUDITOR = importlib.util.module_from_spec(AUDITOR_SPEC)
AUDITOR_SPEC.loader.exec_module(AUDITOR)

TRAINER_PATH = ROOT / "scripts" / "train_multidomain.py"
TRAINER_SPEC = importlib.util.spec_from_file_location(
    "spatial_refined_protocol_trainer_test", TRAINER_PATH
)
assert TRAINER_SPEC and TRAINER_SPEC.loader
TRAINER = importlib.util.module_from_spec(TRAINER_SPEC)
TRAINER_SPEC.loader.exec_module(TRAINER)

V2_CONFIG_PATH = ROOT / "configs" / "multidomain_surface_gamus_six_class_head_only_v2.yaml"
V3_CONFIG_PATH = ROOT / "configs" / "multidomain_surface_gamus_six_class_spatial_refined_v3.yaml"
RUNNER_PATH = ROOT / "scripts" / "run_gamus_six_class_spatial_refined_v3.ps1"


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_state() -> dict[str, torch.Tensor]:
    return {
        "base.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "domain_head.bias": torch.tensor([1.0, 2.0, 3.0]),
    }


def _trained_candidate_state() -> dict[str, torch.Tensor]:
    return {
        **_base_state(),
        "fine_semantic_head.spatial.weight": torch.full((2, 1, 3, 3), 0.01),
        "fine_semantic_head.normalization.weight": torch.ones(2),
        "fine_semantic_head.normalization.bias": torch.zeros(2),
        "fine_semantic_head.classifier.weight": torch.full((6, 2, 1, 1), 0.02),
        "fine_semantic_head.classifier.bias": torch.zeros(6),
    }


def test_v3_recipe_is_exact_v2_contract_plus_spatial_head() -> None:
    source = _load_yaml(V2_CONFIG_PATH)
    candidate = _load_yaml(V3_CONFIG_PATH)

    result = AUDITOR.validate_recipe_contract(
        source,
        candidate,
        source_sha256=_sha256(V2_CONFIG_PATH),
    )

    assert result["passes"] is True
    assert result["official_test_used"] is False
    assert result["auto_promotion"] is False
    assert candidate["data"] == source["data"]
    assert candidate["training"] == source["training"]
    assert candidate["evaluation"] == source["evaluation"]


@pytest.mark.parametrize("section", ["data", "training", "evaluation"])
def test_v3_recipe_rejects_any_behavioral_drift(section: str) -> None:
    source = _load_yaml(V2_CONFIG_PATH)
    candidate = _load_yaml(V3_CONFIG_PATH)
    mutated = deepcopy(candidate)
    mutated[section]["unexpected_change"] = True

    with pytest.raises(ValueError, match=f"v3 {section} contract differs"):
        AUDITOR.validate_recipe_contract(
            source,
            mutated,
            source_sha256=_sha256(V2_CONFIG_PATH),
        )


def test_spatial_head_audit_accepts_only_exact_protected_inheritance() -> None:
    result = AUDITOR.audit_states(_base_state(), _trained_candidate_state())

    assert result["passes"] is True
    assert result["inherited_equality"] == "torch.equal"
    assert result["changed_inherited_tensors"] == []
    assert set(result["allowed_trainable_tensors"]) == AUDITOR.EXPECTED_HEAD_KEYS


def test_spatial_head_audit_rejects_changed_or_unexpected_tensors() -> None:
    changed = _trained_candidate_state()
    changed["base.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="protected height/shared tensors changed"):
        AUDITOR.audit_states(_base_state(), changed)

    unexpected = _trained_candidate_state()
    unexpected["fine_semantic_head.mystery"] = torch.ones(1)
    with pytest.raises(ValueError, match="must add exactly"):
        AUDITOR.audit_states(_base_state(), unexpected)


def test_spatial_head_audit_rejects_an_untrained_projection() -> None:
    candidate = _trained_candidate_state()
    candidate["fine_semantic_head.classifier.weight"].zero_()

    with pytest.raises(ValueError, match="projection was not trained"):
        AUDITOR.audit_states(_base_state(), candidate)


def test_v3_parameter_group_makes_only_spatial_head_trainable() -> None:
    config = _load_yaml(V3_CONFIG_PATH)
    model = DomainGatedSurfaceNet(
        HeightNet(
            backbone="resnet18",
            pretrained=False,
            decoder_channels=(64, 32, 16, 8, 4),
            auxiliary_building_head=True,
        ),
        hidden_channels=8,
        fine_semantic_classes=6,
        fine_semantic_head_type="spatial_refined",
        freeze_base=True,
    )

    groups = TRAINER.configure_explicit_parameter_groups(model, config["training"])
    assert groups is not None
    assert {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    } == AUDITOR.EXPECTED_HEAD_KEYS
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith(AUDITOR.HEAD_PREFIX)
    )


def test_v3_auditor_report_never_claims_pointer_verification(tmp_path: Path) -> None:
    base_path = tmp_path / "protected.pt"
    candidate_path = tmp_path / "candidate.pt"
    torch.save({"model": _base_state(), "epoch": 1}, base_path)
    torch.save(
        {
            "model": _trained_candidate_state(),
            "epoch": 2,
            "metrics": {},
            "config": _load_yaml(V3_CONFIG_PATH),
        },
        candidate_path,
    )

    report = AUDITOR.build_report(
        base_path,
        candidate_path,
        V2_CONFIG_PATH,
    )

    assert report["state_audit"]["passes"] is True
    assert report["recipe_audit"]["passes"] is True
    assert report["app_pointer_verification"] == {
        "status": "external_wrapper_required",
        "performed_by_this_auditor": False,
    }
    assert report["promotion_performed"] is False


def test_v3_runner_pins_inputs_and_never_promotes_the_app() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    config_hash_match = re.search(
        r'\$expectedConfigSha256 = "([0-9a-f]{64})"', source
    )
    source_hash_match = re.search(
        r'\$expectedSourceConfigSha256 = "([0-9a-f]{64})"', source
    )

    assert config_hash_match is not None
    assert config_hash_match.group(1) == _sha256(V3_CONFIG_PATH)
    assert source_hash_match is not None
    assert source_hash_match.group(1) == _sha256(V2_CONFIG_PATH)
    assert "showcase_checkpoint.txt" in source
    assert "official test and app promotion are forbidden" in source
    assert "select_showcase_checkpoint" not in source
    assert "--source-recipe-config" in source
    assert "checkpoint_latest_audit.json" not in source  # name is derived safely
