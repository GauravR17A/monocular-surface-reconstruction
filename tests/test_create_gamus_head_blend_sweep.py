from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "create_gamus_head_blend_sweep.py"
SPEC = importlib.util.spec_from_file_location("create_gamus_head_blend_sweep_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload(*, offset: float = 0.0) -> dict:
    model = {
        "encoder.weight": torch.tensor([1.0, 2.0]),
        "domain_head.weight": torch.tensor([2.0 + offset]),
        "domain_head.bias": torch.tensor([3.0 + offset]),
        "building_residual_head.weight": torch.tensor([4.0 + offset]),
        "building_residual_head.bias": torch.tensor([5.0 + offset]),
        "canopy_height_head.weight": torch.tensor([6.0 + offset]),
        "canopy_height_head.bias": torch.tensor([7.0 + offset]),
    }
    return {
        "model_type": "domain_gated_surface_v2",
        "epoch": 8,
        "model": model,
        "config": {
            "model": {"hidden_channels": 64, "initial_checkpoint": "source.pt"},
            "data": {"patch_size": 384},
        },
        "metrics": {"stale": True},
        "optimizer": {"unsafe_to_resume": True},
    }


def test_source_validation_allows_only_three_head_prefixes() -> None:
    baseline = _payload()
    candidate = _payload(offset=2.0)

    result = MODULE.validate_blend_sources(baseline, candidate)

    assert result["blend_tensor_count"] == 6
    assert result["non_blend_tensor_count"] == 1
    assert result["non_blend_tensors_identical"] is True
    assert result["changed_tensor_names"] == result["blend_tensor_names"]


def test_source_validation_rejects_non_head_change() -> None:
    baseline = _payload()
    candidate = _payload(offset=2.0)
    candidate["model"]["encoder.weight"] += 1.0

    with pytest.raises(ValueError, match="outside the allowed heads"):
        MODULE.validate_blend_sources(baseline, candidate)


def test_atomic_sweep_interpolates_heads_and_keeps_backbone_exact(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    output_dir = tmp_path / "derived"
    pointer = tmp_path / "showcase_checkpoint.txt"
    baseline = _payload()
    candidate = _payload(offset=4.0)
    candidate["config"]["model"]["initial_checkpoint"] = str(baseline_path)
    torch.save(baseline, baseline_path)
    torch.save(candidate, candidate_path)
    pointer.write_bytes(b"protected.pt\r\n")

    manifest = MODULE.create_blend_sweep(
        baseline_path, candidate_path, output_dir, [0.25, 0.5, 0.75]
    )

    assert output_dir.is_dir()
    assert not list(tmp_path.glob(".derived.tmp-*"))
    assert pointer.read_bytes() == b"protected.pt\r\n"
    assert manifest["source_verification"]["non_blend_tensors_identical"] is True
    for item in manifest["outputs"]:
        payload = torch.load(
            output_dir / item["checkpoint"], map_location="cpu", weights_only=False
        )
        alpha = item["alpha"]
        assert torch.equal(
            payload["model"]["encoder.weight"], baseline["model"]["encoder.weight"]
        )
        assert torch.equal(
            payload["model"]["domain_head.weight"],
            torch.lerp(
                baseline["model"]["domain_head.weight"],
                candidate["model"]["domain_head.weight"],
                alpha,
            ),
        )
        assert payload["epoch"] is None
        assert "metrics" not in payload
        assert "optimizer" not in payload
        assert payload["checkpoint_role"] == "evaluation_only_derived_head_blend"


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.pt"
    candidate_path = tmp_path / "candidate.pt"
    output_dir = tmp_path / "existing"
    torch.save(_payload(), baseline_path)
    torch.save(_payload(offset=1.0), candidate_path)
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        MODULE.create_blend_sweep(baseline_path, candidate_path, output_dir, [0.5])

    assert marker.read_text(encoding="utf-8") == "keep"
