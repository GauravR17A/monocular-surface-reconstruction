from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from torch.utils.data import Dataset

from msr.data.surface_dataset import MultiDomainSurfaceDataset, SurfaceSampleRecord
from msr.training.scene_router_record_store import file_sha256


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_routed_surface.py"
SPEC = importlib.util.spec_from_file_location("evaluate_routed_surface_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _FakeGamusDataset(Dataset):
    opened_splits: list[str] = []
    records_by_split: dict[str, list[SimpleNamespace]] = {}

    def __init__(self, _root: Path, split: str, **_: object) -> None:
        self.opened_splits.append(split)
        self.records = self.records_by_split[split]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):  # pragma: no cover - planning never reads pixels
        raise AssertionError(f"unexpected pixel read at {index}")


def _touch(path: Path, contents: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def _gamus_record(tmp_path: Path, split: str) -> SimpleNamespace:
    prefix = tmp_path / f"gamus-{split}"
    return SimpleNamespace(
        sample_id=f"gamus-{split}-1",
        region=f"GAMUS_{split}",
        image_path=_touch(prefix.with_name(prefix.name + "-rgb.h5"), b"rgb"),
        height_path=_touch(prefix.with_name(prefix.name + "-agl.h5"), b"agl"),
        class_path=_touch(prefix.with_name(prefix.name + "-cls.h5"), b"cls"),
        relative_prior_path=_touch(
            prefix.with_name(prefix.name + "-rel.h5"), b"prior"
        ),
    )


def _legacy_record(tmp_path: Path, split: str) -> SurfaceSampleRecord:
    return SurfaceSampleRecord(
        sample_id=f"legacy-{split}-1",
        region=f"legacy-{split}-region",
        landscape="urban",
        rgb_path=_touch(tmp_path / f"legacy-{split}-rgb.tif", b"rgb"),
        surface_path=_touch(tmp_path / f"legacy-{split}-height.tif", b"height"),
        target_kind="ndsm",
        relative_prior_path=_touch(tmp_path / f"legacy-{split}-prior.tif", b"prior"),
    )


def _plan_inputs(tmp_path: Path):
    val_manifest = _touch(tmp_path / "validation.csv", b"validation manifest")
    test_manifest = _touch(tmp_path / "test.csv", b"test manifest")
    gamus_val_record = _gamus_record(tmp_path, "val")
    gamus_test_record = _gamus_record(tmp_path, "test")
    legacy_val_record = _legacy_record(tmp_path, "val")
    legacy_test_record = _legacy_record(tmp_path, "test")
    _FakeGamusDataset.records_by_split = {
        "val": [gamus_val_record],
        "test": [gamus_test_record],
    }
    _FakeGamusDataset.opened_splits = []

    generation = {
        "patch_size": 384,
        "crop": "center",
        "rgb_scale": 255.0,
        "height_max_m": 200.0,
        "building_threshold_m": 2.0,
        "radiometric_policy": "raw",
        "descriptor_mask": "full_center_crop_all_pixels_no_reference_mask",
        "legacy_relative_prior_policy": "stored_01",
        # Produce this side of the fixture directly with the record generator,
        # not with the evaluator wrappers.  The evaluator must accept the
        # generator's exact schema without translating or reinterpreting it.
        "implementation": MODULE.generator_implementation_fingerprint(),
    }
    gamus_val = _FakeGamusDataset(tmp_path, "val")
    legacy_val = MultiDomainSurfaceDataset(
        [legacy_val_record],
        patch_size=384,
        random_crop=False,
        augment=False,
        rgb_scale=255.0,
        height_max_m=200.0,
        building_threshold_m=2.0,
        radiometric_policy="raw",
    )
    provenance = {
        "generation_config": generation,
        "data": {
            "manifests": {
                "legacy_val": {
                    "path": str(val_manifest.resolve()),
                    "sha256": file_sha256(val_manifest),
                }
            },
            "datasets": [
                MODULE.generator_dataset_plan_fingerprint(
                    MODULE.GeneratorDatasetPlan(
                        source="gamus",
                        split="val",
                        partition="calibration",
                        dataset=gamus_val,
                        original_sample_ids=(gamus_val_record.sample_id,),
                    )
                ),
                MODULE.generator_dataset_plan_fingerprint(
                    MODULE.GeneratorDatasetPlan(
                        source="legacy",
                        split="val",
                        partition="calibration",
                        dataset=legacy_val,
                        original_sample_ids=(legacy_val_record.sample_id,),
                    )
                ),
            ],
        },
    }
    _FakeGamusDataset.opened_splits = []
    guard_config = {
        "data": {
            "root": str(tmp_path / "gamus"),
            "relative_prior_root": str(tmp_path / "priors"),
            "require_relative_priors": True,
            "require_complete_official_splits": False,
            "val_manifest": str(val_manifest),
            "test_manifest": str(test_manifest),
            "validation_patch_size": 384,
            "num_workers": 0,
            "rgb_scale": 255.0,
            "height_max_m": 200.0,
            "building_threshold_m": 2.0,
            "validation_radiometric_policy": "raw",
        },
        "evaluation": {
            "recompute_initial_metrics_on_current_validation": True,
            "validation_guards": {
                "gamus": {"rmse_m": {"max": 10.0}},
                "legacy": {"rmse_m": {"max": 10.0}},
            }
        },
    }
    records = {
        val_manifest.resolve(): [legacy_val_record],
        test_manifest.resolve(): [legacy_test_record],
    }
    return provenance, guard_config, records


def test_cli_requires_explicit_flag_before_test_evaluation() -> None:
    args = MODULE.parse_args(
        [
            "--router-artifact",
            "router.pt",
            "--router-report",
            "report.json",
            "--output",
            "validation.json",
        ]
    )
    assert args.include_test is False


def test_exact_generator_provenance_is_accepted_without_opening_test_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance, guard_config, records = _plan_inputs(tmp_path)
    manifest_reads: list[Path] = []

    def fake_manifest(path):
        resolved = Path(path).resolve()
        manifest_reads.append(resolved)
        return records[resolved]

    monkeypatch.setattr(MODULE, "GamusSurfaceDataset", _FakeGamusDataset)
    monkeypatch.setattr(MODULE, "load_surface_manifest", fake_manifest)
    validation, tests, guards, evidence, workers = MODULE._validation_plan(
        provenance, guard_config, include_test=False
    )

    assert set(validation) == {"gamus", "legacy"}
    assert validation["legacy"].relative_prior_policy == "stored_01"
    assert tests == {}
    assert _FakeGamusDataset.opened_splits == ["val"]
    assert manifest_reads == [Path(guard_config["data"]["val_manifest"]).resolve()]
    assert set(guards) == {"gamus", "legacy"}
    assert evidence["test_inventory"] == {}
    assert all(
        item["fingerprint_method"]
        == "ordered resolved_path+size_bytes+mtime_ns"
        for item in evidence["authenticated_validation_inventory"].values()
    )
    assert evidence["implementation"] == provenance["generation_config"][
        "implementation"
    ]
    assert "runtime" in evidence["implementation"]
    assert workers == 0


def test_explicit_test_flag_opens_both_test_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance, guard_config, records = _plan_inputs(tmp_path)
    manifest_reads: list[Path] = []

    def fake_manifest(path):
        resolved = Path(path).resolve()
        manifest_reads.append(resolved)
        return records[resolved]

    monkeypatch.setattr(MODULE, "GamusSurfaceDataset", _FakeGamusDataset)
    monkeypatch.setattr(MODULE, "load_surface_manifest", fake_manifest)
    validation, tests, _, evidence, _ = MODULE._validation_plan(
        provenance, guard_config, include_test=True
    )

    assert set(validation) == {"gamus", "legacy"}
    assert set(tests) == {"gamus", "legacy"}
    assert tests["legacy"].relative_prior_policy == "stored_01"
    assert _FakeGamusDataset.opened_splits == ["val", "test"]
    assert manifest_reads == [
        Path(guard_config["data"]["val_manifest"]).resolve(),
        Path(guard_config["data"]["test_manifest"]).resolve(),
    ]
    assert evidence["test_inventory"]["gamus"]["record_count"] == 1


def test_validation_plan_rejects_preprocessing_drift_before_opening_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance, guard_config, _ = _plan_inputs(tmp_path)
    guard_config["data"]["validation_radiometric_policy"] = "percentile"
    monkeypatch.setattr(MODULE, "GamusSurfaceDataset", _FakeGamusDataset)

    with pytest.raises(MODULE.RoutedEvaluationPlanError, match="radiometric_policy"):
        MODULE._validation_plan(provenance, guard_config, include_test=False)
    assert _FakeGamusDataset.opened_splits == []


def test_generator_inventory_or_runtime_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance, guard_config, records = _plan_inputs(tmp_path)
    val_manifest = Path(guard_config["data"]["val_manifest"]).resolve()

    def fake_manifest(path):
        return records[Path(path).resolve()]

    monkeypatch.setattr(MODULE, "GamusSurfaceDataset", _FakeGamusDataset)
    monkeypatch.setattr(MODULE, "load_surface_manifest", fake_manifest)
    legacy_rgb = records[val_manifest][0].rgb_path
    legacy_rgb.write_bytes(legacy_rgb.read_bytes() + b"changed")
    with pytest.raises(MODULE.RoutedEvaluationPlanError, match="legacy/val data differs"):
        MODULE._validation_plan(provenance, guard_config, include_test=False)

    provenance, guard_config, records = _plan_inputs(tmp_path / "runtime")
    provenance["generation_config"]["implementation"]["runtime"]["torch"] = (
        "unexpected"
    )
    with pytest.raises(
        MODULE.RoutedEvaluationPlanError, match="implementation differs"
    ):
        MODULE._validation_plan(provenance, guard_config, include_test=False)
