from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from msr.evaluation.height_scorecard import (
    HeightScorecardError,
    assess_candidate,
    authenticate_scorecard,
    file_sha256,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _metrics(
    *,
    rmse: float = 5.0,
    mae: float = 4.0,
    bias: float = 0.2,
    correlation: float = 0.5,
    r2: float = 0.2,
    building: float = 8.0,
    vegetation: float = 6.0,
    ground: float = 3.0,
) -> dict[str, object]:
    return {
        "combined": {
            "pixel_count": 100,
            "rmse_m": rmse,
            "mae_m": mae,
            "bias_m": bias,
            "correlation": correlation,
            "r2": r2,
            "landscapes": {
                "urban": {"rmse_m": building},
                "forest": {"rmse_m": vegetation},
            },
            "domains": {
                "building": {"rmse_m": building},
                "vegetation": {"rmse_m": vegetation},
                "ground": {"rmse_m": ground},
            },
        }
    }


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    protected = tmp_path / "protected.pt"
    candidate = tmp_path / "candidate.pt"
    protected.write_bytes(b"protected checkpoint")
    candidate.write_bytes(b"candidate checkpoint")
    pointer = tmp_path / "showcase_checkpoint.txt"
    pointer.write_text(str(protected.resolve()) + "\n", encoding="utf-8")
    artifact = tmp_path / "manifest.csv"
    artifact.write_text("sample_id\none\n", encoding="utf-8")
    gamus_audit = tmp_path / "gamus_audit.json"
    gamus_audit.write_text(
        json.dumps(
            {
                "validity_mask_contract": {"status": "passed"},
                "unit_and_scale": {
                    "height_numeric_unit_status": "unverified_explicit_project_assumption"
                },
                "official_split_policy": {
                    "test": "inventory_audit_only; never constructed by the pilot trainer"
                },
            }
        ),
        encoding="utf-8",
    )

    evaluation_contract = {
        "highbuild_grid": "full native",
        "open_canopy_grid": "full native",
        "rgb": "raw",
    }
    input_identity = {
        "ordered_sample_ids_sha256": "a" * 64,
        "model_input_tensors_sha256": "b" * 64,
        "supervision_tensors_sha256": "c" * 64,
    }
    artifact_identity = {
        "manifest": {
            "path": str(artifact.resolve()),
            "sha256": file_sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        }
    }
    pointer_identity = {
        "pointer": {
            "path": str(pointer.resolve()),
            "sha256": file_sha256(pointer),
            "size_bytes": pointer.stat().st_size,
        },
        "target": {
            "path": str(protected.resolve()),
            "sha256": file_sha256(protected),
            "size_bytes": protected.stat().st_size,
        },
    }
    candidate_metrics = _metrics(
        rmse=4.8,
        mae=3.9,
        bias=0.1,
        correlation=0.51,
        r2=0.21,
        building=7.4,
        vegetation=6.1,
        ground=3.0,
    )
    report = {
        "schema": "msr.corrected_legacy_triplet.v1",
        "complete_manifest": True,
        "identical_protocol_for_all_models": True,
        "identical_input_identity": input_identity,
        "official_test_used": False,
        "read_only_artifacts_unchanged": True,
        "live_application_pointer_changed": False,
        "promotion_performed": False,
        "active_model_changed": False,
        "split": "validation",
        "primary_height_protocol": "strict",
        "protocols": ["strict", "forest"],
        "evaluation_contract": evaluation_contract,
        "authenticated_artifacts_before": artifact_identity,
        "authenticated_artifacts_after": copy.deepcopy(artifact_identity),
        "live_application_before": pointer_identity,
        "live_application_after": copy.deepcopy(pointer_identity),
        "models": {
            "protected": {
                "checkpoint": {
                    "path": str(protected.resolve()),
                    "sha256": file_sha256(protected),
                },
                "protocol_metrics": {"strict": _metrics()},
            },
            "candidate": {
                "checkpoint": {
                    "path": str(candidate.resolve()),
                    "sha256": file_sha256(candidate),
                },
                "protocol_metrics": {"strict": candidate_metrics},
            },
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    scorecard = {
        "schema": "msr.height_scorecard.v1",
        "protocol": {
            "split": "validation",
            "primary_height_protocol": "strict",
            "protocols": ["strict", "forest"],
            "evaluation_contract": evaluation_contract,
        },
        "baseline": {
            "report": str(report_path.resolve()),
            "report_sha256": file_sha256(report_path),
            "model_key": "protected",
            "checkpoint_sha256": file_sha256(protected),
            "metrics": {
                "combined.pixel_count": 100,
                "combined.rmse_m": 5.0,
                "combined.domains.building.rmse_m": 8.0,
                "combined.domains.vegetation.rmse_m": 6.0,
            },
        },
        "protected_application": {
            "pointer": str(pointer.resolve()),
            "pointer_sha256": file_sha256(pointer),
            "checkpoint": str(protected.resolve()),
            "checkpoint_sha256": file_sha256(protected),
        },
        "input_identity": input_identity,
        "data_artifact_sha256": {"manifest": file_sha256(artifact)},
        "supporting_audits": {
            "gamus_preparation": {
                "path": str(gamus_audit.resolve()),
                "sha256": file_sha256(gamus_audit),
            }
        },
        "gates": {
            "maximum_regression": {
                "combined.rmse_m": 0.15,
                "combined.mae_m": 0.10,
                "combined.domains.building.rmse_m": 0.15,
                "combined.domains.vegetation.rmse_m": 0.15,
                "combined.domains.ground.rmse_m": 0.15,
            },
            "maximum_absolute_bias_regression": {"combined.bias_m": 0.10},
            "nondecreasing_metrics": ["combined.correlation", "combined.r2"],
            "material_improvement": {
                "paths": [
                    "combined.domains.building.rmse_m",
                    "combined.domains.vegetation.rmse_m",
                ],
                "minimum_absolute_m": 0.5,
                "minimum_relative_fraction": 0.05,
                "minimum_passing_paths": 1,
            },
        },
        "external_height_evidence": {
            "status": "missing_not_yet_preregistered"
        },
    }
    scorecard_path = tmp_path / "scorecard.yaml"
    scorecard_path.write_text(yaml.safe_dump(scorecard), encoding="utf-8")
    return scorecard_path, report_path


def test_candidate_can_pass_development_but_not_external_promotion(tmp_path: Path) -> None:
    scorecard_path, report_path = _bundle(tmp_path)
    authenticated = authenticate_scorecard(scorecard_path, project_root=tmp_path)
    result = assess_candidate(
        authenticated,
        report_path,
        candidate_model_key="candidate",
        project_root=tmp_path,
    )

    assert result["same_full_scene_protocol_authenticated"] is True
    assert result["development_gate_passed"] is True
    assert result["external_height_gate_passed"] is False
    assert result["promotion_eligible"] is False
    assert result["promotion_performed"] is False
    assert result["reasons"] == ["genuinely_external_metric_height_evidence_missing"]


def test_official_test_or_changed_supervision_fails_closed(tmp_path: Path) -> None:
    scorecard_path, report_path = _bundle(tmp_path)
    authenticated = authenticate_scorecard(scorecard_path, project_root=tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    report["official_test_used"] = True
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(HeightScorecardError, match="official_test_used"):
        assess_candidate(
            authenticated,
            unsafe,
            candidate_model_key="candidate",
            project_root=tmp_path,
        )

    report["official_test_used"] = False
    report["identical_input_identity"]["supervision_tensors_sha256"] = "d" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(HeightScorecardError, match="exact frozen inputs"):
        assess_candidate(
            authenticated,
            changed,
            candidate_model_key="candidate",
            project_root=tmp_path,
        )


def test_current_pointer_or_checkpoint_tampering_is_detected(tmp_path: Path) -> None:
    scorecard_path, _ = _bundle(tmp_path)
    scorecard = yaml.safe_load(scorecard_path.read_text(encoding="utf-8"))
    pointer = Path(scorecard["protected_application"]["pointer"])
    pointer.write_text("somewhere-else.pt\n", encoding="utf-8")

    with pytest.raises(HeightScorecardError, match="pointer hash"):
        authenticate_scorecard(scorecard_path, project_root=tmp_path)


def test_real_scorecard_is_explicitly_non_test_and_external_blocked() -> None:
    scorecard = yaml.safe_load(
        (PROJECT_ROOT / "configs/height_scorecard_v1.yaml").read_text(encoding="utf-8")
    )
    assert scorecard["protocol"]["official_test_used"] is False
    assert scorecard["external_height_evidence"]["required_for_production_eligibility"] is True
    assert scorecard["external_height_evidence"]["status"] == "missing_not_yet_preregistered"
    assert scorecard["truth_contract"]["gamus"]["height_unit"] == "metre_assumed_not_publisher_verified"
