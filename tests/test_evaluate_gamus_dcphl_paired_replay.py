from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_gamus_dcphl_paired_replay.py"
SPEC = importlib.util.spec_from_file_location("_dcphl_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index(tmp_path: Path, *, with_test: bool = False) -> Path:
    def split(ids: list[str]) -> dict[str, object]:
        return {
            "approved_count": len(ids),
            "approved_sample_ids": ids,
            "semantic_eligible_sample_ids": ids,
            "height_regression_eligible_sample_ids": ids,
            "source_count": len(ids),
        }

    splits = {
        "train": split(["DC_01_01", "PHL_0001"]),
        "val": split(["DC_02_01", "PHL_0002"]),
    }
    if with_test:
        splits["test"] = split(["NYC_99999"])
    path = tmp_path / "approved.json"
    path.write_text(
        json.dumps(
            {
                "schema": "msr.gamus.approved_samples.v1",
                "dataset": "GAMUS",
                "source_inventory_metadata_sha256": "a" * 64,
                "splits": splits,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _config(tmp_path: Path, *, version: int = 3, index: Path | None = None) -> tuple[Path, dict]:
    approved = index or _index(tmp_path)
    if version == 3:
        protocol_schema = REPLAY.V3_PROTOCOL
        policy = "never_constructed_or_discovered"
        head = "spatial_refined"
    else:
        protocol_schema = REPLAY.V4_PROTOCOL
        policy = "previously_consumed_forbidden_for_reuse"
        head = "hierarchical_vegetation"
    payload = {
        "experiment": {"name": f"fake_v{version}"},
        "protocol": {
            "schema": protocol_schema,
            "development_validation_cities": ["DC", "PHL"],
            "official_test_policy": policy,
            "auto_promotion": False,
        },
        "data": {
            "dataset": "gamus",
            "root": str(tmp_path / "GAMUS"),
            "relative_prior_root": str(tmp_path / "priors"),
            "approved_index_path": str(approved),
            "approved_index_file_sha256": _sha(approved),
            "require_approved_index": True,
            "require_relative_priors": True,
            "validation_patch_size": 4,
            "rgb_scale": 255.0,
            "height_max_m": 200.0,
            "validation_radiometric_policy": "raw",
        },
        "model": {
            "fine_semantic_classes": 6,
            "fine_semantic_head_type": head,
        },
    }
    path = tmp_path / f"v{version}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path, payload


def _authenticate(path: Path, version: int) -> dict:
    return REPLAY.authenticate_replay_config(
        path,
        expected_sha256=_sha(path),
        expected_protocol=REPLAY.V3_PROTOCOL if version == 3 else REPLAY.V4_PROTOCOL,
        expected_head_type="spatial_refined" if version == 3 else "hierarchical_vegetation",
        expected_validation_count=2,
    )


def _metrics() -> dict:
    group = REPLAY._MetricGroup.create()
    target = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    prediction = np.array([[0, 1, 0], [3, 5, 5]], dtype=np.int64)
    valid = np.ones_like(target, dtype=bool)
    dark = np.array([[False, False, True], [False, True, False]])
    group.update(prediction, target, dark, valid)
    return group.compute()


def test_config_authentication_seals_exact_dcphl_validation(tmp_path: Path) -> None:
    v3_path, _ = _config(tmp_path, version=3)
    v4_path, _ = _config(tmp_path, version=4, index=tmp_path / "approved.json")
    v3 = _authenticate(v3_path, 3)
    v4 = _authenticate(v4_path, 4)
    REPLAY.assert_paired_contracts(v3, v4)
    assert v3["validation_contract"]["city_counts"] == {"DC": 1, "PHL": 1}
    assert v3["validation_contract"]["validation_count"] == 2
    assert v3["approved_index_identity"]["sha256"] == _sha(
        tmp_path / "approved.json"
    )


def test_cli_supports_safe_baseline_only_and_keeps_paired_as_default(
    tmp_path: Path,
) -> None:
    common = [
        "--v3-checkpoint",
        str(tmp_path / "v3.pt"),
        "--v3-config",
        str(tmp_path / "v3.yaml"),
        "--v3-expected-checkpoint-sha256",
        "1" * 64,
        "--v3-expected-config-sha256",
        "2" * 64,
        "--output",
        str(tmp_path / "report.json"),
    ]
    baseline = REPLAY._parse_args(["--baseline-only", *common])
    assert REPLAY._paired_mode(baseline) is False

    with pytest.raises(ValueError, match="requires all four --v4"):
        REPLAY._paired_mode(REPLAY._parse_args(common))
    paired = REPLAY._parse_args(
        [
            *common,
            "--v4-checkpoint",
            str(tmp_path / "v4.pt"),
            "--v4-config",
            str(tmp_path / "v4.yaml"),
            "--v4-expected-checkpoint-sha256",
            "3" * 64,
            "--v4-expected-config-sha256",
            "4" * 64,
        ]
    )
    assert REPLAY._paired_mode(paired) is True

    help_result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--baseline-only" in help_result.stdout
    assert "Required unless --baseline-only" in help_result.stdout


def test_config_authentication_rejects_test_key_and_nyc(tmp_path: Path) -> None:
    bad_index = _index(tmp_path, with_test=True)
    config_path, _ = _config(tmp_path, version=3, index=bad_index)
    with pytest.raises(ValueError, match="only train and val"):
        _authenticate(config_path, 3)

    payload = json.loads(bad_index.read_text(encoding="utf-8"))
    payload["splits"].pop("test")
    payload["splits"]["val"] = {
        "approved_count": 2,
        "approved_sample_ids": ["DC_02_01", "NYC_00001"],
        "semantic_eligible_sample_ids": ["DC_02_01", "NYC_00001"],
        "height_regression_eligible_sample_ids": ["DC_02_01", "NYC_00001"],
        "source_count": 2,
    }
    bad_index.write_text(json.dumps(payload), encoding="utf-8")
    config_path, _ = _config(tmp_path, version=3, index=bad_index)
    with pytest.raises(ValueError, match="forbidden sample ID"):
        _authenticate(config_path, 3)


def test_checkpoint_is_bound_to_external_config_and_exact_hash(tmp_path: Path) -> None:
    config_path, config = _config(tmp_path, version=3)
    checkpoint = tmp_path / "candidate.pt"
    torch.save(
        {
            "config": config,
            "model": {"one": torch.tensor([1.0])},
            "epoch": 4,
            "model_type": "domain_gated_surface_v4_six_class",
        },
        checkpoint,
    )
    result = REPLAY.authenticate_checkpoint(
        checkpoint, expected_sha256=_sha(checkpoint), external_config=config
    )
    assert result["epoch"] == 4
    changed = deepcopy(config)
    changed["model"]["fine_semantic_head_type"] = "linear"
    with pytest.raises(ValueError, match="embedded config differs"):
        REPLAY.authenticate_checkpoint(
            checkpoint, expected_sha256=_sha(checkpoint), external_config=changed
        )
    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        REPLAY.authenticate_checkpoint(
            checkpoint, expected_sha256="0" * 64, external_config=config
        )


def test_source_binding_hashes_all_roles_and_detects_change(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for role in ("image", "height", "class", "relative_prior"):
        path = tmp_path / f"DC_02_01_{role}.h5"
        path.write_bytes(f"{role}-bytes".encode())
        paths[role] = path
    record = SimpleNamespace(
        sample_id="DC_02_01",
        image_path=paths["image"],
        height_path=paths["height"],
        class_path=paths["class"],
        relative_prior_path=paths["relative_prior"],
    )
    first = REPLAY.build_source_binding([record])
    second = REPLAY.build_source_binding([record])
    metadata_before = REPLAY.source_metadata_snapshot([record])
    assert first == second
    assert first["total_file_count"] == 4
    paths["class"].write_bytes(b"changed")
    third = REPLAY.build_source_binding([record])
    metadata_after = REPLAY.source_metadata_snapshot([record])
    assert third["combined_content_manifest_sha256"] != first[
        "combined_content_manifest_sha256"
    ]
    assert metadata_after != metadata_before


def test_metric_validation_checks_ranges_formulas_and_city_sums() -> None:
    metrics = _metrics()
    REPLAY.validate_identification_metrics(metrics)
    bad = deepcopy(metrics)
    bad["six_class_identification"]["per_class"]["ground"]["f1"] = 1.1
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        REPLAY.validate_identification_metrics(bad)
    bad = deepcopy(metrics)
    bad["water_dark_pixel_proxy"]["all_false_water_pixels"] += 1
    with pytest.raises(ValueError, match="confusion matrix"):
        REPLAY.validate_identification_metrics(bad)

    result = {
        "overall": metrics,
        "by_city": {"DC": _metrics(), "PHL": _metrics()},
        "evaluated_sample_count": 2,
        "evaluated_samples_by_city": {"DC": 1, "PHL": 1},
    }
    with pytest.raises(ValueError, match="do not sum"):
        REPLAY.validate_replay_result(result)


class _FakeSixClassModel(torch.nn.Module):
    def forward(self, image: torch.Tensor, prior: torch.Tensor) -> dict[str, torch.Tensor]:
        del prior
        prediction = image[:, 0].long()
        logits = torch.full(
            (image.shape[0], 6, image.shape[2], image.shape[3]),
            -20.0,
            device=image.device,
        )
        logits.scatter_(1, prediction[:, None], 20.0)
        return {"fine_semantic_logits": logits}


def test_replay_recomputes_overall_and_per_city_with_fake_model(tmp_path: Path) -> None:
    prediction = torch.tensor(
        [
            [[0, 1], [2, 3]],
            [[4, 5], [2, 0]],
        ],
        dtype=torch.float32,
    )
    image = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
    image[:, 0] = prediction
    target = torch.tensor(
        [
            [[0, 1], [2, 3]],
            [[4, 5], [0, 0]],
        ],
        dtype=torch.int64,
    )
    batch = {
        "image": image,
        "relative_prior": torch.zeros((2, 1, 2, 2)),
        "fine_class_target": target,
        "classification_valid_mask": torch.ones((2, 2, 2), dtype=torch.bool),
        "image_valid_mask": torch.ones((2, 1, 2, 2), dtype=torch.bool),
        "dark_pixel_proxy_mask": torch.zeros((2, 2, 2), dtype=torch.bool),
        "sample_id": ["DC_02_01", "PHL_0002"],
    }

    def loader(_checkpoint: Path, *, device: torch.device):
        return _FakeSixClassModel().to(device), {
            "epoch": 2,
            "model_type": "fake",
            "fine_semantic_head_type": "spatial_refined",
        }

    result = REPLAY.replay_checkpoint(
        tmp_path / "unused.pt",
        [batch],
        expected_ids=["DC_02_01", "PHL_0002"],
        device=torch.device("cpu"),
        precision="fp32",
        expected_head_type="spatial_refined",
        label="fake",
        model_loader=loader,
    )
    assert result["evaluated_sample_count"] == 2
    assert result["evaluated_samples_by_city"] == {"DC": 1, "PHL": 1}
    assert result["overall"]["six_class_identification"]["total_valid_pixels"] == 8
    assert result["by_city"]["DC"]["six_class_identification"]["accuracy"] == 1.0
    assert result["by_city"]["PHL"]["six_class_identification"]["accuracy"] == 0.75


def test_atomic_writer_leaves_only_complete_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "report.json"
    REPLAY.atomic_write_json(output, {"passes": True, "value": 3})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "passes": True,
        "value": 3,
    }
    assert list(output.parent.glob("*.tmp")) == []
