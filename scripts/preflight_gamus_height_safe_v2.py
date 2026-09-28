"""Fail-closed preflight for the isolated GAMUS height-safe v2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_APPROVED_INDEX_SHA256 = (
    "5320d2e97be357b1e1725d7f2d9640493522ae3d13f1a051b63de55b530c05aa"
)
EXPECTED_WARM_START_SHA256 = (
    "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
)
ALLOWED_TRAINABLE_PREFIXES = {
    "base_model.height_head.",
    "canopy_height_head.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    model = config["model"]
    training = config["training"]
    contracts = config["contracts"]
    evaluation = config["evaluation"]

    assert data["dataset"] == "gamus"
    assert not any("test" in str(key).lower() for key in data)
    assert contracts["official_test_used"] is False
    assert contracts["height_unit_status"] == "metre_assumed"
    assert float(contracts["raw_height_scale"]) == 1.0
    assert contracts["depth_anything_prior_policy"] == "fixed_cached_stored_01"
    assert set(contracts["regression_source_class_ids"]) == {1, 2, 3, 6}
    assert set(contracts["excluded_regression_source_class_ids"]) == {0, 4, 5}
    assert data["require_relative_priors"] is True
    assert data["require_approved_index"] is True
    assert data["train_radiometric_policy"] == "raw"
    assert data["validation_radiometric_policy"] == "raw"
    assert float(data["height_max_m"]) == 200.0
    assert int(data["validation_patch_size"]) == 1024
    assert "train_samples_per_epoch" not in data
    assert evaluation["promotion_eligible"] is False

    prefixes = {
        str(prefix)
        for group in training["parameter_groups"]
        for prefix in group["prefixes"]
    }
    assert prefixes == ALLOWED_TRAINABLE_PREFIXES
    forbidden = (
        "adapter.",
        "domain_head.",
        "building_residual_head.",
        "refinement_strength_head.",
        "base_model.encoder.",
        "base_model.decoder_stages.",
        "fine_semantic_head.",
    )
    assert not any(prefix.startswith(forbidden) for prefix in prefixes)

    approved_index = resolve(data["approved_index_path"])
    prior_root = resolve(data["relative_prior_root"])
    warm_start = resolve(model["initial_checkpoint"])
    base_checkpoint = resolve(model["base_checkpoint"])
    qc_report = approved_index.parent / "qc_report.json"
    for required in (approved_index, qc_report, warm_start, base_checkpoint):
        assert required.is_file(), f"missing required file: {required}"
    assert prior_root.is_dir(), f"missing relative-prior root: {prior_root}"
    assert sha256(approved_index) == EXPECTED_APPROVED_INDEX_SHA256
    assert data["approved_index_file_sha256"] == EXPECTED_APPROVED_INDEX_SHA256
    assert sha256(warm_start) == EXPECTED_WARM_START_SHA256

    quality = json.loads(qc_report.read_text(encoding="utf-8"))
    assert quality["approved_index_sha256"] == EXPECTED_APPROVED_INDEX_SHA256
    assert quality["split_ids_disjoint"] is True
    assert quality["height_unit_contract"]["status"] == "explicit_project_assumption"
    assert float(quality["height_unit_contract"]["raw_to_training_scale"]) == 1.0
    assert float(quality["policies"]["height_max_m"]) == 200.0
    assert quality["splits"]["train"]["approved_tiles"] == 5001
    assert quality["splits"]["val"]["approved_tiles"] == 859

    # Inspect only train/validation development records. Importing the trainer's
    # helper deliberately never resolves the official test directory.
    import importlib.util

    trainer_path = PROJECT_ROOT / "scripts" / "train_multidomain.py"
    spec = importlib.util.spec_from_file_location("height_safe_preflight", trainer_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    datasets = module.make_gamus_development_datasets(data["root"], data)
    module.validate_gamus_splits(
        datasets,
        require_complete_official_splits=True,
    )
    assert set(datasets) == {"train", "val"}
    assert len(datasets["train"].records) == 5001
    assert len(datasets["val"].records) == 859

    inspected: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        sample = datasets[split][0]
        image_valid = sample["image_valid_mask"].bool()
        class_valid = sample["classification_valid_mask"].bool()
        height_valid = sample["height_valid_mask"].bool()
        regression = sample["regression_mask"].bool()
        fine_target = sample["fine_class_target"]
        assert torch.all(regression <= image_valid)
        assert torch.all(regression <= height_valid)
        assert torch.all(regression[0] <= class_valid)
        # six-class indices 2 and 3 are water and roads; 255 is ignored.
        assert not torch.any(regression[0] & fine_target.eq(2))
        assert not torch.any(regression[0] & fine_target.eq(3))
        assert not torch.any(regression[0] & fine_target.eq(255))
        assert torch.all(sample["height"][~regression] == 0)
        assert torch.isfinite(sample["image"]).all()
        assert torch.isfinite(sample["relative_prior"]).all()
        inspected[split] = {
            "image_valid_pixels": int(image_valid.sum()),
            "classification_valid_pixels": int(class_valid.sum()),
            "height_valid_pixels": int(height_valid.sum()),
            "regression_pixels": int(regression.sum()),
        }

    return {
        "status": "pass",
        "official_test_used": False,
        "promotion_eligible": False,
        "height_unit_status": contracts["height_unit_status"],
        "development_records": {"train": 5001, "val": 859},
        "trainable_prefixes": sorted(prefixes),
        "approved_index_sha256": sha256(approved_index),
        "warm_start_sha256": sha256(warm_start),
        "sample_mask_checks": inspected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "multidomain_surface_gamus_height_safe_v2.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.config.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
