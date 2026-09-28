"""Fail-closed preflight for corrected GAMUS + HighBuild/OpenCanopy replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from collections import Counter

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LIVE_SHA256 = (
    "e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144"
)
EXPECTED_PREFIXES = {
    "base_model.height_head.",
    "canopy_height_head.",
    "refinement_strength_head.",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _load_trainer():
    path = PROJECT_ROOT / "scripts" / "train_multidomain.py"
    spec = importlib.util.spec_from_file_location("corrected_height_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_sample_masks(sample: dict[str, object]) -> dict[str, int]:
    image_valid = sample["image_valid_mask"].bool()
    class_valid = sample["classification_valid_mask"].bool()
    height_valid = sample["height_valid_mask"].bool()
    regression = sample["regression_mask"].bool()
    assert torch.all(regression <= image_valid)
    assert torch.all(regression <= height_valid)
    # Classification and height availability are deliberately independent.
    # In particular, corrected HighBuild/OpenCanopy records can supervise
    # metric height while carrying no native six-class GAMUS labels.
    classification_and_height = class_valid & height_valid[0]
    height = sample["height"]
    assert torch.isfinite(height).all()
    assert torch.isfinite(sample["image"]).all()
    assert torch.isfinite(sample["relative_prior"]).all()
    return {
        "image_valid_pixels": int(image_valid.sum()),
        "classification_valid_pixels": int(class_valid.sum()),
        "height_valid_pixels": int(height_valid.sum()),
        "regression_pixels": int(regression.sum()),
        "classification_and_height_pixels": int(classification_and_height.sum()),
    }


def validate(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    model = config["model"]
    training = config["training"]
    contracts = config["contracts"]
    evaluation = config["evaluation"]

    assert data["dataset"] == "mixed_replay"
    assert training["mixed_replay_objective"] == "height_supervision"
    assert contracts["official_test_used"] is False
    assert not any("test" in str(key).lower() for key in data)
    assert evaluation["promotion_eligible"] is False
    assert contracts["missing_height_policy"] == "ignored_never_zero"
    assert contracts["highbuild_height_support"] == "measured_coco_intersection_only"
    assert contracts["open_canopy_height_unit"] == "verified_metres"
    assert contracts["gamus_height_unit_status"] == "metre_assumed_not_publisher_verified"
    assert set(contracts["gamus_regression_source_class_ids"]) == {1, 2, 3, 6}
    assert set(contracts["gamus_excluded_regression_source_class_ids"]) == {
        0,
        4,
        5,
    }

    configured_prefixes = {
        str(prefix)
        for group in training["parameter_groups"]
        for prefix in group["prefixes"]
    }
    assert configured_prefixes == EXPECTED_PREFIXES
    assert not training.get("trainable_adapter_prefixes")
    assert int(training["freeze_base_epochs"]) == 0
    assert float(data["gamus_train_fraction"]) == 0.5
    assert float(data["legacy_supervised_crop_probability"]) == 1.0
    assert data["legacy_relative_prior_policy"] == "stored_01"
    assert int(data["train_samples_per_epoch"]) % int(training["batch_size"]) == 0

    identities: dict[str, str] = {}
    for path_key, hash_key in (
        ("approved_index_path", "approved_index_file_sha256"),
        ("train_manifest", "train_manifest_file_sha256"),
        ("val_manifest", "val_manifest_file_sha256"),
        ("highbuild_contract_report", "highbuild_contract_report_sha256"),
    ):
        path = resolve(data[path_key])
        assert path.is_file(), f"missing sealed input: {path}"
        digest = file_sha256(path)
        assert digest == str(data[hash_key]).lower(), f"hash mismatch: {path}"
        identities[path_key] = digest

    warm_start = resolve(model["initial_checkpoint"])
    assert warm_start.is_file()
    assert file_sha256(warm_start) == EXPECTED_LIVE_SHA256
    pointer = PROJECT_ROOT / "outputs" / "runtime" / "showcase_checkpoint.txt"
    assert pointer.is_file()
    assert resolve(pointer.read_text(encoding="utf-8-sig").strip()) == warm_start

    contract = json.loads(resolve(data["highbuild_contract_report"]).read_text())
    assert contract["schema"] == "msr.highbuild_annotation_contract.v2"
    assert contract["height_protocol"] == "strict_measured"
    assert contract["historical_artifacts_modified"] is False
    assert contract["test_role"] == "previously_inspected_regression_evidence_only_not_fresh_holdout"
    assert contract["splits"]["train"]["output_manifest_sha256"] == identities["train_manifest"]
    assert contract["splits"]["validation"]["output_manifest_sha256"] == identities["val_manifest"]

    trainer = _load_trainer()
    trainer.validate_mixed_replay_training_config(
        dataset_kind=data["dataset"],
        initial_checkpoint=model["initial_checkpoint"],
        training_config=training,
    )
    gamus = trainer.make_gamus_development_datasets(
        data["root"], data, use_train_samples_per_epoch=False
    )
    trainer.validate_gamus_splits(gamus, require_complete_official_splits=True)
    train_records = trainer.load_surface_manifest(data["train_manifest"])
    validation_records = trainer.load_surface_manifest(data["val_manifest"])
    trainer.assert_surface_regions_disjoint(train_records, validation_records)
    legacy = trainer.make_surface_dataset(
        train_records,
        data,
        training=True,
        use_train_samples_per_epoch=False,
    )
    legacy_validation = trainer.make_surface_dataset(
        validation_records,
        data,
        training=False,
        use_train_samples_per_epoch=False,
    )
    assert legacy.relative_prior_policy == "stored_01"
    assert legacy_validation.relative_prior_policy == "stored_01"
    assert legacy.supervised_crop_probability == 1.0
    assert legacy_validation.supervised_crop_probability == 0.0
    landscapes = {record.landscape for record in train_records}
    assert landscapes == {"urban", "forest"}
    target_kinds = Counter(record.target_kind for record in train_records)
    assert target_kinds == {"building_height": 450, "ndsm": 600}
    urban_index = next(i for i, record in enumerate(train_records) if record.landscape == "urban")
    forest_index = next(i for i, record in enumerate(train_records) if record.landscape == "forest")

    sample_checks = {
        "gamus": _check_sample_masks(gamus["train"][0]),
        "highbuild": _check_sample_masks(legacy[urban_index]),
        "open_canopy": _check_sample_masks(legacy[forest_index]),
    }
    assert sample_checks["gamus"]["regression_pixels"] > 0
    assert sample_checks["highbuild"]["regression_pixels"] > 0
    assert sample_checks["open_canopy"]["regression_pixels"] > 0

    return {
        "status": "pass",
        "official_test_used": False,
        "promotion_eligible": False,
        "live_checkpoint_sha256": file_sha256(warm_start),
        "sealed_input_sha256": identities,
        "records": {
            "gamus_train": len(gamus["train"]),
            "gamus_validation": len(gamus["val"]),
            "legacy_train": len(train_records),
            "legacy_validation": len(validation_records),
        },
        "sample_mask_checks": sample_checks,
        "trainable_prefixes": sorted(configured_prefixes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "multidomain_surface_corrected_height_replay_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.config.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        destination = args.output.resolve()
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite preflight: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
