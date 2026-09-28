from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import msr.evaluation.routed_artifact as routed_artifact
from msr.evaluation.routed_artifact import (
    ROUTER_ARTIFACT_SCHEMA,
    ROUTER_REPORT_SCHEMA,
    STAGE3B_ROUTER_ARTIFACT_SCHEMA,
    STAGE3B_ROUTER_REPORT_SCHEMA,
    STAGE3B_TARGET_PROVENANCE,
    RoutedArtifactValidationError,
    load_guarded_routed_surface,
)
from msr.models.domain_surface_net import DomainGatedSurfaceNet
from msr.models.routed_surface import (
    ConservativeSceneRouter,
    FrozenSurfaceHeadPack,
)
from msr.models.stage3_endpoints import (
    BaseCheckpointDiagnostics,
    EndpointCheckpointDiagnostics,
    EndpointCompatibilityDiagnostics,
    Stage3EndpointDiagnostics,
    Stage3EndpointPacks,
)
from msr.training.scene_router_record_store import (
    build_record_store_provenance,
    canonical_json_sha256,
    file_sha256,
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


def _surface() -> DomainGatedSurfaceNet:
    return DomainGatedSurfaceNet(
        _TinyBase(),
        hidden_channels=4,
        fusion_mode="protected_vegetation",
        freeze_base=True,
    ).eval()


def _endpoint(
    role: str, path: Path, digest: str, *, epoch: int
) -> EndpointCheckpointDiagnostics:
    return EndpointCheckpointDiagnostics(
        role=role,
        path=str(path.resolve()),
        sha256=digest,
        file_size_bytes=100,
        epoch=epoch,
        model_type="domain_gated_surface_v2",
        model_tensor_count=10,
        model_numel=100,
        model_state_sha256=("c" if role == "protected" else "d") * 64,
        head_state_sha256=("e" if role == "protected" else "f") * 64,
        architecture_config_sha256="1" * 64,
    )


def _bundle(tmp_path: Path) -> Stage3EndpointPacks:
    protected_path = tmp_path / "protected.pt"
    candidate_path = tmp_path / "candidate.pt"
    protected_path.write_bytes(b"protected")
    candidate_path.write_bytes(b"candidate")
    protected = _surface()
    candidate = _surface()
    with torch.no_grad():
        candidate.domain_head.bias.add_(1.0)
    base = BaseCheckpointDiagnostics(
        path=str((tmp_path / "base.pt").resolve()),
        sha256="2" * 64,
        file_size_bytes=50,
        model_type="height_net_v1",
        model_tensor_count=2,
        model_numel=2,
        model_state_sha256="3" * 64,
        architecture_config_sha256="4" * 64,
    )
    diagnostics = Stage3EndpointDiagnostics(
        protected=_endpoint("protected", protected_path, "a" * 64, epoch=20),
        gamus_stage1=_endpoint(
            "gamus_stage1", candidate_path, "b" * 64, epoch=8
        ),
        compatibility=EndpointCompatibilityDiagnostics(
            shared_tensor_count=4,
            shared_numel=20,
            shared_state_sha256="5" * 64,
            differing_head_tensors=("domain_head.bias",),
            identical_head_tensors=(),
            allowed_head_tensors=("domain_head.bias",),
            architecture_config_items=(("hidden_channels", "4"),),
        ),
        base_checkpoint=base,
    )
    return Stage3EndpointPacks(
        protected=FrozenSurfaceHeadPack.from_model(protected),
        gamus_stage1=FrozenSurfaceHeadPack.from_model(candidate),
        shared_model=protected,
        diagnostics=diagnostics,
    )


def _write_artifacts(
    tmp_path: Path,
    endpoints: Stage3EndpointPacks,
    *,
    artifact_eligible: bool = True,
    report_eligible: bool = True,
    tamper_config_hash: bool = False,
    descriptor_mask: str = "full_center_crop_all_pixels_no_reference_mask",
    legacy_prior_policy: str = "stored_01",
    descriptor_size: int = 8,
    scored_record_count: int = 4,
) -> tuple[Path, Path]:
    diagnostics = endpoints.diagnostics.as_dict()
    provenance = build_record_store_provenance(
        generation_config={
            "patch_size": 4,
            "precision": "fp32",
            "descriptor_mask": descriptor_mask,
            "legacy_relative_prior_policy": legacy_prior_policy,
            "utility": {"minimum_candidate_gain": 0.02},
        },
        endpoints={
            name: diagnostics[name]
            for name in ("protected", "gamus_stage1", "compatibility")
        },
        shared_model={
            "base_checkpoint": diagnostics["base_checkpoint"],
            "shared_state_sha256": diagnostics["compatibility"][
                "shared_state_sha256"
            ],
        },
        data={
            "splits": ["gamus/val", "legacy/val"],
            "planned_record_count": scored_record_count + 1,
            "excluded_no_reference_count": 1,
            "excluded_no_reference_record_keys": [
                "legacy/train/excluded-no-reference"
            ],
            "scored_record_count": scored_record_count,
            "planned_partition_counts": {
                "train": scored_record_count,
                "calibration": 1,
            },
            "scored_partition_counts": {
                "train": scored_record_count - 1,
                "calibration": 1,
            },
        },
    )
    if tamper_config_hash:
        provenance["generation_config"]["patch_size"] = 99
    completion = {
        "schema": provenance["schema"],
        "config_sha256": provenance["config_sha256"],
        "record_count": 4,
        "part_count": 1,
        "descriptor_size": descriptor_size,
        "parts_manifest_canonical_sha256": "7" * 64,
        "records_sha256": "6" * 64,
        "records_path": str((tmp_path / "records.jsonl").resolve()),
        "complete": True,
    }
    router = ConservativeSceneRouter(
        4,
        hidden_features=3,
        decision_threshold=0.9,
    )
    artifact = {
        "artifact_schema": ROUTER_ARTIFACT_SCHEMA,
        "artifact_type": "offline_scene_router_only",
        "router_state_dict": router.state_dict(),
        "feature_channels": 4,
        "hidden_features": 3,
        "decision_threshold": 0.9,
        "threshold_eligible": artifact_eligible,
        "minimum_candidate_gain": 0.02,
        "training_config": {
            "minimum_precision": 0.9,
            "maximum_mean_utility_regression": 0.0,
            "minimum_candidate_selections": 1,
        },
        "record_provenance": provenance,
        "record_completion": completion,
        "record_provenance_canonical_sha256": canonical_json_sha256(provenance),
        "record_completion_canonical_sha256": canonical_json_sha256(completion),
        "stage3_eligible_record_provenance": True,
    }
    artifact_path = tmp_path / "scene_router.pt"
    torch.save(artifact, artifact_path)
    report = {
        "artifact_schema": ROUTER_REPORT_SCHEMA,
        "artifact_type": "offline_scene_router_report",
        "router_artifact_path": str(artifact_path.resolve()),
        "router_artifact_sha256": file_sha256(artifact_path),
        "record_provenance": provenance,
        "record_completion": completion,
        "record_provenance_canonical_sha256": canonical_json_sha256(provenance),
        "record_completion_canonical_sha256": canonical_json_sha256(completion),
        "records": {"train": 3, "calibration": 1},
        "minimum_candidate_gain": 0.02,
        "threshold": {
            "value": 0.9,
            "eligible": report_eligible,
            "metrics": {
                "precision": 1.0,
                "routed_gain_vs_fallback": 0.1,
                "candidate_selected": 1,
            },
        },
    }
    report_path = tmp_path / "scene_router_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return artifact_path, report_path


def _write_stage3b_artifacts(
    tmp_path: Path, endpoints: Stage3EndpointPacks
) -> tuple[Path, Path, Path]:
    diagnostics = endpoints.diagnostics.as_dict()
    data = {
        "included_source_splits": [
            "gamus/train",
            "gamus/val",
            "legacy/train",
            "legacy/val",
        ],
        "excluded_source_splits": ["gamus/test", "legacy/test"],
        "planned_record_count": 5,
        "excluded_no_reference_count": 1,
        "excluded_no_reference_record_keys": ["gamus/train/excluded"],
        "scored_record_count": 4,
        "planned_partition_counts": {"train": 3, "calibration": 2},
        "scored_partition_counts": {"train": 2, "calibration": 2},
        "scored_source_split_counts": {
            "gamus/train": 1,
            "gamus/val": 1,
            "legacy/train": 1,
            "legacy/val": 1,
        },
    }
    provenance = build_record_store_provenance(
        generation_config={
            "patch_size": 4,
            "precision": "fp32",
            "descriptor_mask": "full_center_crop_all_pixels_no_reference_mask",
            "legacy_relative_prior_policy": "stored_01",
            "utility": {"minimum_candidate_gain": 0.02},
        },
        endpoints={
            name: diagnostics[name]
            for name in ("protected", "gamus_stage1", "compatibility")
        },
        shared_model={
            "base_checkpoint": diagnostics["base_checkpoint"],
            "shared_state_sha256": diagnostics["compatibility"][
                "shared_state_sha256"
            ],
        },
        data=data,
    )
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "partition": partition,
                },
                sort_keys=True,
            )
            + "\n"
            for sample_id, source, partition in (
                ("gamus/train/g", "gamus", "train"),
                ("legacy/train/l", "legacy", "train"),
                ("gamus/val/g", "gamus", "calibration"),
                ("legacy/val/l", "legacy", "calibration"),
            )
        ),
        encoding="utf-8",
    )
    parts_manifest = {
        "schema": provenance["schema"],
        "config_sha256": provenance["config_sha256"],
        "parts": [
            {
                "index": 0,
                "filename": "part-00000000-test.jsonl",
                "sha256": "9" * 64,
                "record_count": 4,
                "descriptor_size": 8,
                "keys_sha256": "8" * 64,
                "first_key": "gamus/train/g",
                "last_key": "legacy/val/l",
            }
        ],
    }
    completion = {
        "schema": provenance["schema"],
        "config_sha256": provenance["config_sha256"],
        "record_count": 4,
        "part_count": 1,
        "descriptor_size": 8,
        "parts_manifest_canonical_sha256": canonical_json_sha256(parts_manifest),
        "records_sha256": file_sha256(records_path),
        "records_path": str(records_path.resolve()),
        "complete": True,
    }
    (tmp_path / "provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    (tmp_path / "completion.json").write_text(
        json.dumps(completion), encoding="utf-8"
    )
    (tmp_path / "parts_manifest.json").write_text(
        json.dumps(parts_manifest), encoding="utf-8"
    )
    config_path = tmp_path / "stage3b.yaml"
    config_path.write_text("router_training:\n  target_mode: source_gamus\n")
    trainer_path = Path(routed_artifact.__file__).resolve().parents[3] / "scripts" / (
        "train_scene_router.py"
    )
    training_config = {
        "epochs": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "candidate_weight": 1.0,
        "fallback_weight": 1.0,
        "brier_weight": 0.05,
        "minimum_threshold": 0.5,
        "minimum_precision": 0.99,
        "maximum_mean_utility_regression": 0.0,
        "minimum_candidate_selections": 1,
    }
    training_launch = {
        "records": str(records_path.resolve()),
        "output_dir": str(tmp_path.resolve()),
        "epochs": 4,
        "batch_size": 2,
        "hidden_features": 3,
        "minimum_candidate_gain": 0.02,
        "target_mode": "source_gamus",
        "minimum_source_precision": 0.99,
        "minimum_gamus_coverage": 0.8,
        "minimum_gamus_selections": 1,
        "seed": 7,
        "device": "cpu",
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "trainer_sha256": file_sha256(trainer_path),
    }
    source_guards = {
        "minimum_source_precision": 0.99,
        "minimum_gamus_coverage": 0.8,
        "minimum_gamus_selections": 1,
        "minimum_routed_utility_gain_vs_protected": 0.0,
        "selection_priority": [
            "maximum_gamus_coverage",
            "maximum_routed_utility_gain_vs_protected",
            "highest_threshold",
        ],
    }
    router = ConservativeSceneRouter(
        4,
        hidden_features=3,
        decision_threshold=0.9,
    )
    artifact = {
        "artifact_schema": STAGE3B_ROUTER_ARTIFACT_SCHEMA,
        "artifact_type": "offline_scene_domain_router_only",
        "target_mode": "source_gamus",
        "target_provenance": STAGE3B_TARGET_PROVENANCE,
        "source_threshold_guards": source_guards,
        "router_state_dict": router.state_dict(),
        "feature_channels": 4,
        "hidden_features": 3,
        "decision_threshold": 0.9,
        "threshold_eligible": True,
        "minimum_candidate_gain": 0.02,
        "training_config": training_config,
        "training_launch": training_launch,
        "record_provenance": provenance,
        "record_completion": completion,
        "record_provenance_canonical_sha256": canonical_json_sha256(provenance),
        "record_completion_canonical_sha256": canonical_json_sha256(completion),
        "stage3_eligible_record_provenance": True,
        "stage3b_eligible_record_provenance": True,
    }
    artifact_path = tmp_path / "scene_domain_router.pt"
    torch.save(artifact, artifact_path)
    report = {
        "artifact_schema": STAGE3B_ROUTER_REPORT_SCHEMA,
        "artifact_type": "offline_scene_domain_router_report",
        "target_mode": "source_gamus",
        "target_provenance": STAGE3B_TARGET_PROVENANCE,
        "source_threshold_guards": source_guards,
        "router_artifact_path": str(artifact_path.resolve()),
        "router_artifact_sha256": file_sha256(artifact_path),
        "training_config": training_config,
        "training_launch": training_launch,
        "record_provenance": provenance,
        "record_completion": completion,
        "record_provenance_canonical_sha256": canonical_json_sha256(provenance),
        "record_completion_canonical_sha256": canonical_json_sha256(completion),
        "records": {"train": 2, "calibration": 2},
        "minimum_candidate_gain": 0.02,
        "source_partition_counts": {
            "train": {"gamus": 1, "legacy": 1},
            "calibration": {"gamus": 1, "legacy": 1},
        },
        "threshold": {
            "value": 0.9,
            "eligible": True,
            "metrics": {
                "source_precision": 1.0,
                "precision": 1.0,
                "gamus_coverage": 1.0,
                "source_recall": 1.0,
                "recall": 1.0,
                "selected_gamus": 1,
                "selected_legacy": 0,
                "candidate_selected": 1,
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 1,
                "scene_count": 2,
                "routed_utility_gain_vs_protected": 0.1,
                "routed_gain_vs_fallback": 0.1,
                "fallback_mean_utility": 0.5,
                "routed_mean_utility": 0.4,
            },
        },
    }
    report_path = tmp_path / "scene_domain_router_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return artifact_path, report_path, config_path


def _rewrite_stage3b_artifact_and_hash(
    artifact_path: Path,
    report_path: Path,
    mutate,
) -> None:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    mutate(artifact)
    torch.save(artifact, artifact_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["router_artifact_sha256"] = file_sha256(artifact_path)
    report_path.write_text(json.dumps(report), encoding="utf-8")


def test_loads_only_matching_eligible_router_and_verified_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path = _write_artifacts(tmp_path, endpoints)

    def fake_endpoint_loader(protected, candidate, **kwargs):
        assert Path(protected).resolve() == Path(
            endpoints.diagnostics.protected.path
        ).resolve()
        assert Path(candidate).resolve() == Path(
            endpoints.diagnostics.gamus_stage1.path
        ).resolve()
        assert kwargs["expected_protected_sha256"] == "a" * 64
        assert kwargs["expected_gamus_stage1_sha256"] == "b" * 64
        assert kwargs["expected_base_sha256"] == "2" * 64
        assert kwargs["reconstruct_shared_model"] is True
        return endpoints

    monkeypatch.setattr(
        routed_artifact, "load_stage3_endpoint_packs", fake_endpoint_loader
    )
    result = load_guarded_routed_surface(artifact_path, report_path)

    assert result.diagnostics.router_artifact_sha256 == file_sha256(artifact_path)
    assert result.diagnostics.record_count == 4
    assert result.diagnostics.record_config_sha256 == result.router_report[
        "record_provenance"
    ]["config_sha256"]
    assert not result.model.training
    assert all(not parameter.requires_grad for parameter in result.model.parameters())
    with torch.inference_mode():
        output = result.model(torch.zeros(1, 3, 4, 4))
    assert output["height"].shape == (1, 1, 4, 4)
    assert not bool(output["route_candidate_selected"].item())


@pytest.mark.parametrize(
    ("artifact_eligible", "report_eligible", "message"),
    (
        (False, True, "artifact is not guard-eligible"),
        (True, False, "report says the route is ineligible"),
    ),
)
def test_rejects_ineligible_artifact_or_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_eligible: bool,
    report_eligible: bool,
    message: str,
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path = _write_artifacts(
        tmp_path,
        endpoints,
        artifact_eligible=artifact_eligible,
        report_eligible=report_eligible,
    )
    monkeypatch.setattr(
        routed_artifact,
        "load_stage3_endpoint_packs",
        lambda *args, **kwargs: pytest.fail("endpoints must not load after rejection"),
    )

    with pytest.raises(RoutedArtifactValidationError, match=message):
        load_guarded_routed_surface(artifact_path, report_path)


def test_rejects_record_config_tampering_before_loading_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path = _write_artifacts(
        tmp_path, endpoints, tamper_config_hash=True
    )
    monkeypatch.setattr(
        routed_artifact,
        "load_stage3_endpoint_packs",
        lambda *args, **kwargs: pytest.fail("tampered endpoints must not load"),
    )

    with pytest.raises(RoutedArtifactValidationError, match="config hash mismatch"):
        load_guarded_routed_surface(artifact_path, report_path)


def test_rejects_completion_count_that_disagrees_with_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path = _write_artifacts(
        tmp_path, endpoints, scored_record_count=5
    )
    monkeypatch.setattr(
        routed_artifact,
        "load_stage3_endpoint_packs",
        lambda *args, **kwargs: pytest.fail("invalid records must fail before endpoints"),
    )
    with pytest.raises(
        RoutedArtifactValidationError, match="provenance scored_record_count"
    ):
        load_guarded_routed_surface(artifact_path, report_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"descriptor_mask": "dataset_valid_mask"},
            "not deployable all-pixel descriptors",
        ),
        (
            {"legacy_prior_policy": "target_crop_normalized"},
            "not deployable stored 0..1 inputs",
        ),
    ),
)
def test_rejects_train_serve_input_skew_before_loading_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, str],
    message: str,
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path = _write_artifacts(tmp_path, endpoints, **kwargs)
    monkeypatch.setattr(
        routed_artifact,
        "load_stage3_endpoint_packs",
        lambda *args, **loader_kwargs: pytest.fail(
            "skewed endpoint artifact must not load"
        ),
    )

    with pytest.raises(RoutedArtifactValidationError, match=message):
        load_guarded_routed_surface(artifact_path, report_path)


def test_rejects_router_feature_and_record_descriptor_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path = _write_artifacts(
        tmp_path, endpoints, descriptor_size=10
    )
    monkeypatch.setattr(
        routed_artifact,
        "load_stage3_endpoint_packs",
        lambda *args, **kwargs: endpoints,
    )

    with pytest.raises(
        RoutedArtifactValidationError, match="completed record descriptors"
    ):
        load_guarded_routed_surface(artifact_path, report_path)


def test_rejects_report_bound_to_different_router_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path = _write_artifacts(tmp_path, endpoints)
    with artifact_path.open("ab") as handle:
        handle.write(b"changed")
    monkeypatch.setattr(
        routed_artifact,
        "load_stage3_endpoint_packs",
        lambda *args, **kwargs: pytest.fail("mismatched endpoints must not load"),
    )

    with pytest.raises(RoutedArtifactValidationError, match="SHA-256 mismatch"):
        load_guarded_routed_surface(artifact_path, report_path)


def test_loads_matching_stage3b_source_router_with_exact_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path, _ = _write_stage3b_artifacts(tmp_path, endpoints)
    monkeypatch.setattr(
        routed_artifact,
        "load_stage3_endpoint_packs",
        lambda *args, **kwargs: endpoints,
    )

    result = load_guarded_routed_surface(artifact_path, report_path)

    assert result.diagnostics.decision_threshold == pytest.approx(0.9)
    assert result.diagnostics.record_count == 4
    assert result.router_report["target_mode"] == "source_gamus"
    assert not result.model.training
    assert all(not parameter.requires_grad for parameter in result.model.parameters())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "target_provenance",
            "candidate target came from endpoint utility",
            "target provenance mismatch",
        ),
        (
            "source_threshold_guards",
            {
                "minimum_source_precision": 0.98,
                "minimum_gamus_coverage": 0.8,
                "minimum_gamus_selections": 1,
                "minimum_routed_utility_gain_vs_protected": 0.0,
                "selection_priority": [
                    "maximum_gamus_coverage",
                    "maximum_routed_utility_gain_vs_protected",
                    "highest_threshold",
                ],
            },
            "source threshold guards mismatch",
        ),
    ),
)
def test_stage3b_rejects_report_identity_mismatch_before_endpoint_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path, _ = _write_stage3b_artifacts(tmp_path, endpoints)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        routed_artifact,
        "load_stage3_endpoint_packs",
        lambda *args, **kwargs: pytest.fail("Stage-3b identity must fail first"),
    )

    with pytest.raises(RoutedArtifactValidationError, match=message):
        load_guarded_routed_surface(artifact_path, report_path)


def test_stage3b_rejects_precision_guard_below_point_99(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path, _ = _write_stage3b_artifacts(tmp_path, endpoints)

    def weaken_artifact(artifact: dict) -> None:
        artifact["source_threshold_guards"]["minimum_source_precision"] = 0.98
        artifact["training_config"]["minimum_precision"] = 0.98
        artifact["training_launch"]["minimum_source_precision"] = 0.98

    _rewrite_stage3b_artifact_and_hash(artifact_path, report_path, weaken_artifact)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_threshold_guards"]["minimum_source_precision"] = 0.98
    report["training_config"]["minimum_precision"] = 0.98
    report["training_launch"]["minimum_source_precision"] = 0.98
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        routed_artifact,
        "load_stage3_endpoint_packs",
        lambda *args, **kwargs: pytest.fail("weak guards must fail first"),
    )

    with pytest.raises(RoutedArtifactValidationError, match="at least 0.99"):
        load_guarded_routed_surface(artifact_path, report_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("selected_legacy", 1, "counts are inconsistent"),
        ("gamus_coverage", 0.7, "precision/coverage"),
        (
            "routed_utility_gain_vs_protected",
            -0.1,
            "no-regression guard",
        ),
    ),
)
def test_stage3b_rejects_inconsistent_or_regressing_threshold_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path, _ = _write_stage3b_artifacts(tmp_path, endpoints)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["threshold"]["metrics"][field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        routed_artifact,
        "load_stage3_endpoint_packs",
        lambda *args, **kwargs: pytest.fail("invalid metrics must fail first"),
    )

    with pytest.raises(RoutedArtifactValidationError, match=message):
        load_guarded_routed_surface(artifact_path, report_path)


def test_stage3b_rejects_changed_config_or_record_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoints = _bundle(tmp_path)
    artifact_path, report_path, config_path = _write_stage3b_artifacts(
        tmp_path, endpoints
    )
    config_path.write_text("router_training:\n  target_mode: utility\n")
    monkeypatch.setattr(
        routed_artifact,
        "load_stage3_endpoint_packs",
        lambda *args, **kwargs: pytest.fail("changed evidence must fail first"),
    )
    with pytest.raises(RoutedArtifactValidationError, match="config bytes"):
        load_guarded_routed_surface(artifact_path, report_path)

    artifact_path, report_path, _ = _write_stage3b_artifacts(tmp_path, endpoints)
    completion_path = tmp_path / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["part_count"] = 2
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(RoutedArtifactValidationError, match="embedded completion"):
        load_guarded_routed_surface(artifact_path, report_path)
