import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml


ROOT = Path(__file__).parents[1]
SCRIPT_PATH = ROOT / "scripts" / "seal_gamus_v4_runtime_data.py"
REPLAY_SCRIPT_PATH = ROOT / "scripts" / "evaluate_gamus_dcphl_paired_replay.py"
SPEC = importlib.util.spec_from_file_location("v4_runtime_data_seal_test", SCRIPT_PATH)
assert SPEC and SPEC.loader
SEAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEAL)


FAKE_RUNTIME = {
    "python": {"version": "3.test"},
    "packages": {"torch": "test", "numpy": "test"},
    "pytorch": {"cuda_available": True, "cuda_build_version": "test"},
    "nvidia_driver": [{"name": "test GPU", "driver_version": "test"}],
}


@pytest.fixture(autouse=True)
def _small_expected_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SEAL,
        "EXPECTED_CITY_SPLIT_COUNTS",
        {
            "train": {"DC": 1, "PHL": 0},
            "val": {"DC": 0, "PHL": 1},
        },
    )


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path.resolve()


def _json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _h5(path: Path, values: np.ndarray, *, attrs: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("image", data=values)
        for name, value in (attrs or {}).items():
            dataset.attrs[name] = value
    return path.resolve()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    dataset = tmp_path / "GAMUS"
    priors = tmp_path / "GAMUS_relative_priors"
    split_ids = {"train": ["DC_01_01"], "val": ["PHL_02_02"]}
    for split, sample_ids in split_ids.items():
        for sample_id in sample_ids:
            image_path = _h5(
                dataset / "images" / split / f"{sample_id}_RGB.h5",
                np.zeros((1024, 1024, 3), dtype=np.uint8),
            )
            class_dtype = np.float32 if sample_id.startswith("DC_") else np.uint8
            _h5(
                dataset / "classes" / split / f"{sample_id}_CLS.h5",
                np.zeros((1024, 1024), dtype=class_dtype),
            )
            _h5(
                dataset / "heights" / split / f"{sample_id}_AGL.h5",
                np.zeros((1024, 1024), dtype=np.float32),
            )
            _h5(
                priors / split / f"{sample_id}_REL.h5",
                np.zeros((1024, 1024), dtype=np.float16),
                attrs={
                    "units": "relative_0_1",
                    "model": SEAL.EXPECTED_PRIOR_MODEL,
                    "source_rgb": str(image_path),
                },
            )
    _json(priors / "msr_prior_provenance.json", {"model": "dav2-test"})

    approved = _json(
        tmp_path / "approved.json",
        {
            "schema": SEAL.APPROVED_INDEX_SCHEMA,
            "splits": {
                split: {
                    "approved_count": len(ids),
                    "approved_sample_ids": ids,
                }
                for split, ids in split_ids.items()
            },
        },
    )
    sampling = tmp_path / "sampling.jsonl"
    sampling.write_text(
        json.dumps(
            {
                "schema": SEAL.SAMPLING_INDEX_SCHEMA,
                "sample_id": "DC_01_01",
                "city": "DC",
                "class_path": str(
                    (dataset / "classes" / "train" / "DC_01_01_CLS.h5").resolve()
                ),
                "uniform_random_crop_384": {"water_hit_probability": 0.25},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sampling = sampling.resolve()

    initial = _write(tmp_path / "protected.pt", b"protected-checkpoint")
    base = _write(tmp_path / "base.pt", b"base-checkpoint")
    comparator = _write(tmp_path / "v3.yaml", b"v3-config")
    audit = _write(tmp_path / "v3-audit.json", b"v3-audit")
    evaluator_sha256 = SEAL.file_sha256(REPLAY_SCRIPT_PATH)
    replay = _json(
        tmp_path / "v3-independent-replay.json",
        {
            "schema": SEAL.EXPECTED_INDEPENDENT_REPLAY_SCHEMA,
            "mode": "baseline_v3_only",
            "passes": True,
            "access_policy": {
                "cities_opened": ["DC", "PHL"],
                "nyc_opened": False,
                "official_test_opened_or_reused": False,
                "promotion_performed": False,
                "split_constructed": "val",
            },
            "implementation_sha256": {
                "before": {"evaluator": evaluator_sha256},
                "after": {"evaluator": evaluator_sha256},
                "unchanged_during_replay": True,
            },
        },
    )
    pointer = _write(tmp_path / "pointer.txt", str(initial).encode())
    config_path = tmp_path / "v4.yaml"
    config = {
        "protocol": {
            "schema": SEAL.EXPECTED_PROTOCOL_SCHEMA,
            "learning_cities": ["DC", "PHL"],
            "development_validation_cities": ["DC", "PHL"],
            "locked_classifier_holdout_city": "NYC",
            "official_test_policy": "previously_consumed_forbidden_for_reuse",
            "protected_checkpoint_sha256": SEAL.file_sha256(initial),
            "source_comparator_config": str(comparator),
            "source_comparator_config_sha256": SEAL.file_sha256(comparator),
            "fixed_v3_baseline_audit": str(audit),
            "fixed_v3_baseline_audit_sha256": SEAL.file_sha256(audit),
            "fixed_v3_independent_replay": str(replay),
            "fixed_v3_independent_replay_sha256": SEAL.file_sha256(replay),
            "live_pointer_file": str(pointer),
            "live_pointer_file_sha256": SEAL.file_sha256(pointer),
        },
        "data": {
            "root": str(dataset.resolve()),
            "relative_prior_root": str(priors.resolve()),
            "require_relative_priors": True,
            "approved_index_path": str(approved),
            "approved_index_file_sha256": SEAL.file_sha256(approved),
            "require_approved_index": True,
            "fine_class_sampling_index_path": str(sampling),
            "fine_class_sampling_index_sha256": SEAL.file_sha256(sampling),
            "require_complete_official_splits": True,
            "patch_size": 384,
            "validation_patch_size": 1024,
            "water_sampling_boost": 2.0,
            "preserve_city_sampling_mass": True,
            "rgb_scale": 255.0,
            "train_radiometric_policy": "raw",
            "validation_radiometric_policy": "raw",
            "height_max_m": 200.0,
        },
        "model": {
            "initial_checkpoint": str(initial),
            "base_checkpoint": str(base),
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return {
        "config": config_path.resolve(),
        "approved": approved,
        "sampling": sampling,
        "dataset": dataset.resolve(),
        "priors": priors.resolve(),
        "replay": replay,
    }


def _create(fixture: dict[str, Path], output: Path) -> dict:
    return SEAL.create_seal(
        fixture["config"],
        output,
        jobs=2,
        behavior_source_paths=[SCRIPT_PATH],
        runtime=FAKE_RUNTIME,
    )


def test_create_and_verify_hashes_exact_four_roles_without_holdout(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "seal.json"
    payload = _create(fixture, output)

    dataset = payload["contract"]["dataset"]
    assert dataset["sample_count"] == 2
    assert dataset["file_count"] == 8
    assert {item["role"] for item in dataset["files"]} == set(SEAL.DATA_ROLES)
    assert {item["city"] for item in dataset["files"]} == {"DC", "PHL"}
    assert all("NYC" not in item["path"] for item in dataset["files"])
    assert all("test" not in Path(item["path"]).parts for item in dataset["files"])
    assert payload["contract"]["policy"]["nyc_inputs_opened"] is False
    assert payload["contract"]["policy"]["official_test_inputs_opened"] is False
    replay_controls = [
        item
        for item in payload["contract"]["control_artifacts"]
        if "fixed_v3_independent_replay" in item["roles"]
    ]
    assert len(replay_controls) == 1
    assert replay_controls[0]["sha256"] == SEAL.file_sha256(fixture["replay"])

    verified = SEAL.verify_seal(output, jobs=2, runtime=FAKE_RUNTIME)
    assert verified["passes"] is True
    assert verified["file_count"] == 8


def test_default_behavior_inventory_includes_independent_replay_evaluator() -> None:
    assert "scripts/evaluate_gamus_dcphl_paired_replay.py" in SEAL.BEHAVIOR_SOURCE_PATHS
    evaluator = REPLAY_SCRIPT_PATH.resolve()
    unique_direct_sources = {
        ROOT / "src" / "msr" / "data" / "gamus_dataset.py",
        ROOT / "src" / "msr" / "evaluation" / "classification_metrics.py",
        ROOT / "src" / "msr" / "inference" / "predict.py",
        ROOT / "src" / "msr" / "models" / "domain_surface_net.py",
    }
    assert evaluator.is_file()
    assert all(
        path.resolve().relative_to(ROOT.resolve()).as_posix()
        in SEAL.BEHAVIOR_SOURCE_PATHS
        for path in unique_direct_sources
    )


def test_v4_requires_pinned_independent_replay_pair(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    for missing in (
        "fixed_v3_independent_replay",
        "fixed_v3_independent_replay_sha256",
    ):
        config = yaml.safe_load(fixture["config"].read_text(encoding="utf-8"))
        del config["protocol"][missing]
        fixture["config"].write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        with pytest.raises(SEAL.SealError, match=missing):
            SEAL.build_contract(
                fixture["config"],
                behavior_source_paths=[SCRIPT_PATH],
                runtime=FAKE_RUNTIME,
            )
        fixture = _fixture(tmp_path / missing)


def test_replay_artifact_hash_is_enforced(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["replay"].write_text("{}", encoding="utf-8")
    with pytest.raises(SEAL.SealError, match="fixed_v3_independent_replay SHA-256 mismatch"):
        SEAL.build_contract(
            fixture["config"],
            behavior_source_paths=[SCRIPT_PATH],
            runtime=FAKE_RUNTIME,
        )


def test_replay_must_bind_current_evaluator_implementation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    replay = json.loads(fixture["replay"].read_text(encoding="utf-8"))
    replay["implementation_sha256"]["after"]["evaluator"] = "0" * 64
    fixture["replay"].write_text(json.dumps(replay), encoding="utf-8")
    config = yaml.safe_load(fixture["config"].read_text(encoding="utf-8"))
    config["protocol"]["fixed_v3_independent_replay_sha256"] = SEAL.file_sha256(
        fixture["replay"]
    )
    fixture["config"].write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(SEAL.SealError, match="evaluator SHA-256"):
        SEAL.build_contract(
            fixture["config"],
            behavior_source_paths=[SCRIPT_PATH],
            runtime=FAKE_RUNTIME,
        )


def test_contract_fingerprint_is_deterministic(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    first = _create(fixture, tmp_path / "first.json")
    second = _create(fixture, tmp_path / "second.json")
    assert first["created_at_utc"] != ""
    assert first["contract_sha256"] == second["contract_sha256"]
    assert first["contract"] == second["contract"]


def test_verify_fails_on_same_size_data_content_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "seal.json"
    _create(fixture, output)
    image = fixture["dataset"] / "images" / "train" / "DC_01_01_RGB.h5"
    original_size = image.stat().st_size
    with h5py.File(image, "r+") as handle:
        handle["image"][0, 0, 0] = 1
    assert image.stat().st_size == original_size

    with pytest.raises(SEAL.SealError, match="data file content drifted"):
        SEAL.verify_seal(output, jobs=1, runtime=FAKE_RUNTIME)


def test_verify_fails_before_data_hashing_on_runtime_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "seal.json"
    _create(fixture, output)
    monkeypatch.setattr(
        SEAL,
        "hash_entries",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("data touched")),
    )
    with pytest.raises(SEAL.SealError, match="runtime identity drifted"):
        SEAL.verify_seal(output, runtime={"python": {"version": "changed"}})


def test_creation_refuses_nyc_before_resolving_any_data_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = json.loads(fixture["approved"].read_text(encoding="utf-8"))
    payload["splits"]["train"] = {
        "approved_count": 1,
        "approved_sample_ids": ["NYC_01_01"],
    }
    fixture["approved"].write_text(json.dumps(payload), encoding="utf-8")
    config = yaml.safe_load(fixture["config"].read_text(encoding="utf-8"))
    config["data"]["approved_index_file_sha256"] = SEAL.file_sha256(fixture["approved"])
    fixture["config"].write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(SEAL.SealError, match=r"forbidden/non-DC\+PHL"):
        SEAL.build_contract(
            fixture["config"],
            behavior_source_paths=[SCRIPT_PATH],
            runtime=FAKE_RUNTIME,
        )


def test_creation_refuses_any_test_split_in_approved_index(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    payload = json.loads(fixture["approved"].read_text(encoding="utf-8"))
    payload["splits"]["test"] = {
        "approved_count": 1,
        "approved_sample_ids": ["DC_09_09"],
    }
    fixture["approved"].write_text(json.dumps(payload), encoding="utf-8")
    config = yaml.safe_load(fixture["config"].read_text(encoding="utf-8"))
    config["data"]["approved_index_file_sha256"] = SEAL.file_sha256(fixture["approved"])
    fixture["config"].write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(SEAL.SealError, match="only train and val"):
        SEAL.discover_data_entries(config)


def test_sampling_index_must_exactly_match_train_ids(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["sampling"].write_text("", encoding="utf-8")
    config = yaml.safe_load(fixture["config"].read_text(encoding="utf-8"))
    config["data"]["fine_class_sampling_index_sha256"] = SEAL.file_sha256(
        fixture["sampling"]
    )
    with pytest.raises(SEAL.SealError, match="exact train-ID match"):
        SEAL.discover_data_entries(config)


def test_both_rgb_and_img_variants_fail_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _h5(
        fixture["dataset"] / "images" / "train" / "DC_01_01_IMG.h5",
        np.zeros((1024, 1024, 3), dtype=np.uint8),
    )
    config = yaml.safe_load(fixture["config"].read_text(encoding="utf-8"))
    with pytest.raises(SEAL.SealError, match="exactly one approved RGB/IMG"):
        SEAL.discover_data_entries(config)


def test_prior_must_name_its_exact_rgb_source_and_model(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    prior = fixture["priors"] / "train" / "DC_01_01_REL.h5"
    with h5py.File(prior, "r+") as handle:
        handle["image"].attrs["source_rgb"] = str(
            fixture["dataset"] / "images" / "val" / "PHL_02_02_RGB.h5"
        )
    config = yaml.safe_load(fixture["config"].read_text(encoding="utf-8"))
    entries, _ = SEAL.discover_data_entries(config)
    with pytest.raises(SEAL.SealError, match="source RGB mismatch"):
        SEAL.validate_hdf5_entries(entries, jobs=2)


def test_tampered_contract_fingerprint_fails_before_verification(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "seal.json"
    _create(fixture, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["contract"]["dataset"]["sample_count"] = 999
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SEAL.SealError, match="fingerprint is invalid"):
        SEAL.verify_seal(output, runtime=FAKE_RUNTIME)


def test_external_seal_sha256_anchor_rejects_whole_file_replacement(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "seal.json"
    _create(fixture, output)
    with pytest.raises(SEAL.SealError, match="external SHA-256 anchor"):
        SEAL.verify_seal(
            output,
            expected_seal_sha256="0" * 64,
            runtime=FAKE_RUNTIME,
        )


def test_atomic_creation_refuses_overwrite_and_leaves_no_pending_file(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "seal.json"
    _create(fixture, output)
    with pytest.raises(SEAL.SealError, match="refusing to overwrite"):
        _create(fixture, output)
    assert list(tmp_path.glob(".seal.json.pending-*.tmp")) == []


def test_estimate_reads_metadata_only_and_reports_four_files_per_sample(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    result = SEAL.estimate(fixture["config"])
    assert result["sample_count"] == 2
    assert result["file_count"] == 8
    assert result["total_bytes"] > 0
    assert set(result["hashing_seconds_estimate"]) == {
        "at_100_mib_per_second",
        "at_250_mib_per_second",
        "at_500_mib_per_second",
    }
