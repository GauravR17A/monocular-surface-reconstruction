"""Offline, fail-closed validation of the Stage-3 routed surface model.

The default command opens only the authenticated GAMUS and legacy validation
splits.  Test data is inaccessible unless ``--include-test`` is explicitly
supplied.  This evaluator is read-only with respect to model checkpoints and
the live application pointer; it never promotes a result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_stage3_scene_router_records import (  # noqa: E402
    RouterDatasetPlan as GeneratorDatasetPlan,
    dataset_plan_fingerprint as generator_dataset_plan_fingerprint,
    implementation_fingerprint as generator_implementation_fingerprint,
)

from msr.data.gamus_dataset import (  # noqa: E402
    GAMUS_OFFICIAL_SPLIT_COUNTS,
    GamusSurfaceDataset,
)
from msr.data.surface_dataset import (  # noqa: E402
    MultiDomainSurfaceDataset,
    SurfaceSampleRecord,
    assert_surface_regions_disjoint,
    load_surface_manifest,
)
from msr.evaluation.routed_artifact import (  # noqa: E402
    GuardedRoutedSurfaceBundle,
    load_guarded_routed_surface,
)
from msr.evaluation.routed_validation import (  # noqa: E402
    evaluate_dense_surface_model,
    evaluate_validation_guards,
)
from msr.training.scene_router_record_store import (  # noqa: E402
    canonical_json_sha256,
    file_sha256,
)


REPORT_SCHEMA = "msr.stage3_routed_surface_validation.v1"
DEFAULT_GUARD_CONFIG = (
    PROJECT_ROOT / "configs" / "stage3_routed_validation.yaml"
)
DEFAULT_POINTER = PROJECT_ROOT / "outputs" / "runtime" / "showcase_checkpoint.txt"
SUPPORTED_SUITES = frozenset({"gamus", "legacy"})


class RoutedEvaluationPlanError(ValueError):
    """Raised before evaluation when data/config identity is not proven."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate and evaluate the offline Stage-3 routed model on "
            "GAMUS + legacy validation without changing the application pointer."
        )
    )
    parser.add_argument("--router-artifact", type=Path, required=True)
    parser.add_argument("--router-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guard-config", type=Path, default=DEFAULT_GUARD_CONFIG)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16" if torch.cuda.is_available() else "fp32",
    )
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--include-test",
        action="store_true",
        help=(
            "Explicitly consume both held-out test splits after validation. "
            "Without this flag no test path is read or opened."
        ),
    )
    return parser.parse_args(argv)


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _require_mapping(value: object, *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoutedEvaluationPlanError(f"{role} must be a mapping")
    return value


def _load_yaml_mapping(path_like: str | Path, *, role: str) -> tuple[Path, Mapping[str, Any]]:
    path = _resolve(path_like)
    if not path.is_file():
        raise RoutedEvaluationPlanError(f"{role} is unavailable: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise RoutedEvaluationPlanError(f"Could not read {role}: {error}") from error
    return path, _require_mapping(value, role=role)


def _same_path(left: object, right: object) -> bool:
    if not isinstance(left, (str, Path)) or not isinstance(right, (str, Path)):
        return False
    return os.path.normcase(str(_resolve(left))) == os.path.normcase(str(_resolve(right)))


def current_implementation_fingerprint() -> dict[str, object]:
    """Return the generator's exact code-and-runtime identity contract."""

    try:
        return dict(generator_implementation_fingerprint())
    except (OSError, RuntimeError, ValueError) as error:
        raise RoutedEvaluationPlanError(
            f"Could not fingerprint the current router implementation: {error}"
        ) from error


def _require_current_implementation(generation: Mapping[str, Any]) -> dict[str, object]:
    expected = generation.get("implementation")
    if not isinstance(expected, Mapping):
        raise RoutedEvaluationPlanError(
            "Router provenance has no implementation fingerprint"
        )
    current = current_implementation_fingerprint()
    if canonical_json_sha256(expected) != canonical_json_sha256(current):
        raise RoutedEvaluationPlanError(
            "Current routing/data implementation differs from authenticated records"
        )
    return current


def dataset_inventory_fingerprint(
    dataset: Dataset, *, source: str, split: str, partition: str
) -> dict[str, object]:
    """Call the generator's exact ordered path/size/mtime inventory helper."""

    records = getattr(dataset, "records", None)
    if not isinstance(records, Sequence) or len(records) != len(dataset):
        raise RoutedEvaluationPlanError(
            "Evaluation dataset must expose one immutable record per scene"
        )
    plan = GeneratorDatasetPlan(
        source=source,
        split=split,
        partition=partition,
        dataset=dataset,
        original_sample_ids=tuple(str(record.sample_id) for record in records),
    )
    try:
        return dict(generator_dataset_plan_fingerprint(plan))
    except (OSError, RuntimeError, ValueError) as error:
        raise RoutedEvaluationPlanError(
            f"Could not fingerprint current {source}/{split} data: {error}"
        ) from error


def _provenance_inventory(
    provenance: Mapping[str, Any], *, source: str, split: str
) -> Mapping[str, Any]:
    data = _require_mapping(provenance.get("data"), role="record data provenance")
    inventories = data.get("datasets")
    if not isinstance(inventories, list):
        raise RoutedEvaluationPlanError(
            "Record data provenance has no dataset inventory list"
        )
    matches = [
        item
        for item in inventories
        if isinstance(item, Mapping)
        and item.get("source") == source
        and item.get("split") == split
    ]
    if len(matches) != 1:
        raise RoutedEvaluationPlanError(
            f"Expected exactly one authenticated {source}/{split} inventory"
        )
    return matches[0]


def _require_inventory_match(
    dataset: Dataset,
    provenance: Mapping[str, Any],
    *,
    source: str,
    split: str,
    partition: str,
) -> dict[str, object]:
    current = dataset_inventory_fingerprint(
        dataset, source=source, split=split, partition=partition
    )
    expected = _provenance_inventory(provenance, source=source, split=split)
    if canonical_json_sha256(current) != canonical_json_sha256(expected):
        raise RoutedEvaluationPlanError(
            f"Current {source}/{split} data differs from authenticated router records"
        )
    return current


def _legacy_dataset(
    records: list[SurfaceSampleRecord], generation: Mapping[str, Any]
) -> MultiDomainSurfaceDataset:
    return MultiDomainSurfaceDataset(
        records,
        patch_size=int(generation["patch_size"]),
        random_crop=False,
        augment=False,
        samples_per_epoch=None,
        rgb_scale=float(generation["rgb_scale"]),
        height_max_m=float(generation["height_max_m"]),
        building_threshold_m=float(generation["building_threshold_m"]),
        radiometric_policy=str(generation["radiometric_policy"]),
        relative_prior_policy=str(generation["legacy_relative_prior_policy"]),
    )


def _validation_plan(
    provenance: Mapping[str, Any],
    guard_config: Mapping[str, Any],
    *,
    include_test: bool,
) -> tuple[
    dict[str, Dataset],
    dict[str, Dataset],
    Mapping[str, Any],
    dict[str, object],
    int,
]:
    """Open authenticated validation data; touch test only under explicit opt-in."""

    generation = _require_mapping(
        provenance.get("generation_config"), role="record generation config"
    )
    data_provenance = _require_mapping(
        provenance.get("data"), role="record data provenance"
    )
    guard_data = _require_mapping(guard_config.get("data"), role="guard config data")
    evaluation = _require_mapping(
        guard_config.get("evaluation"), role="guard config evaluation"
    )
    guards = _require_mapping(
        evaluation.get("validation_guards"), role="validation guards"
    )
    if evaluation.get("recompute_initial_metrics_on_current_validation") is not True:
        raise RoutedEvaluationPlanError(
            "Hard-guard config must recompute its baseline on current validation data"
        )
    if set(guards) != SUPPORTED_SUITES:
        raise RoutedEvaluationPlanError(
            "Hard guards must define exactly the GAMUS and legacy validation suites"
        )

    expected_generation = {
        "patch_size": int(guard_data.get("validation_patch_size", -1)),
        "rgb_scale": float(guard_data.get("rgb_scale", float("nan"))),
        "height_max_m": float(guard_data.get("height_max_m", float("nan"))),
        "building_threshold_m": float(
            guard_data.get("building_threshold_m", float("nan"))
        ),
        "radiometric_policy": str(
            guard_data.get("validation_radiometric_policy", "")
        ),
    }
    for name, expected in expected_generation.items():
        if generation.get(name) != expected:
            raise RoutedEvaluationPlanError(
                f"Guard config {name} differs from authenticated record preprocessing"
            )
    if generation.get("crop") != "center":
        raise RoutedEvaluationPlanError("Authenticated validation crop is not center")
    if (
        generation.get("descriptor_mask")
        != "full_center_crop_all_pixels_no_reference_mask"
    ):
        raise RoutedEvaluationPlanError(
            "Router provenance does not use the deployable all-pixel scene descriptor"
        )
    if generation.get("legacy_relative_prior_policy") != "stored_01":
        raise RoutedEvaluationPlanError(
            "Router provenance does not use the deployable stored 0..1 legacy prior"
        )
    implementation_evidence = _require_current_implementation(generation)
    if guard_data.get("require_relative_priors") is not True:
        raise RoutedEvaluationPlanError("GAMUS validation must require relative priors")

    manifests = _require_mapping(
        data_provenance.get("manifests"), role="record manifest provenance"
    )
    legacy_val_evidence = _require_mapping(
        manifests.get("legacy_val"), role="legacy validation manifest provenance"
    )
    legacy_val_path = _resolve(guard_data.get("val_manifest", ""))
    if not _same_path(legacy_val_path, legacy_val_evidence.get("path")):
        raise RoutedEvaluationPlanError(
            "Guard-config legacy validation manifest differs from router provenance"
        )
    if file_sha256(legacy_val_path) != legacy_val_evidence.get("sha256"):
        raise RoutedEvaluationPlanError(
            "Legacy validation manifest hash differs from router provenance"
        )

    gamus_root = _resolve(guard_data.get("root", ""))
    prior_root = _resolve(guard_data.get("relative_prior_root", ""))
    gamus_val = GamusSurfaceDataset(
        gamus_root,
        "val",
        patch_size=int(generation["patch_size"]),
        random_crop=False,
        augment=False,
        samples_per_epoch=None,
        rgb_scale=float(generation["rgb_scale"]),
        height_max_m=float(generation["height_max_m"]),
        radiometric_policy=str(generation["radiometric_policy"]),
        relative_prior_root=prior_root,
        require_relative_prior=True,
    )
    if bool(guard_data.get("require_complete_official_splits", True)) and len(
        gamus_val.records
    ) != GAMUS_OFFICIAL_SPLIT_COUNTS["val"]:
        raise RoutedEvaluationPlanError(
            "Current GAMUS validation split is not the complete official split"
        )
    legacy_val_records = load_surface_manifest(legacy_val_path)
    legacy_val = _legacy_dataset(legacy_val_records, generation)
    validation = {"gamus": gamus_val, "legacy": legacy_val}
    validation_inventory = {
        "gamus": _require_inventory_match(
            gamus_val,
            provenance,
            source="gamus",
            split="val",
            partition="calibration",
        ),
        "legacy": _require_inventory_match(
            legacy_val,
            provenance,
            source="legacy",
            split="val",
            partition="calibration",
        ),
    }

    tests: dict[str, Dataset] = {}
    test_evidence: dict[str, object] = {}
    if include_test:
        # Keep every test-specific lookup inside this explicit branch.  This is
        # intentionally structured so the default evaluator cannot even open a
        # held-out test path by accident.
        legacy_test_value = guard_data.get("test_manifest")
        if not isinstance(legacy_test_value, str) or not legacy_test_value.strip():
            raise RoutedEvaluationPlanError(
                "Explicit test evaluation requires data.test_manifest"
            )
        gamus_test = GamusSurfaceDataset(
            gamus_root,
            "test",
            patch_size=int(generation["patch_size"]),
            random_crop=False,
            augment=False,
            samples_per_epoch=None,
            rgb_scale=float(generation["rgb_scale"]),
            height_max_m=float(generation["height_max_m"]),
            radiometric_policy=str(generation["radiometric_policy"]),
            relative_prior_root=prior_root,
            require_relative_prior=True,
        )
        if bool(guard_data.get("require_complete_official_splits", True)) and len(
            gamus_test.records
        ) != GAMUS_OFFICIAL_SPLIT_COUNTS["test"]:
            raise RoutedEvaluationPlanError(
                "Current GAMUS test split is not the complete official split"
            )
        gamus_val_ids = {record.sample_id for record in gamus_val.records}
        gamus_test_ids = {record.sample_id for record in gamus_test.records}
        if gamus_val_ids & gamus_test_ids:
            raise RoutedEvaluationPlanError("GAMUS validation/test scenes overlap")

        legacy_test_path = _resolve(legacy_test_value)
        legacy_test_records = load_surface_manifest(legacy_test_path)
        assert_surface_regions_disjoint(legacy_val_records, legacy_test_records)
        tests = {
            "gamus": gamus_test,
            "legacy": _legacy_dataset(legacy_test_records, generation),
        }
        test_evidence = {
            "gamus": {
                "split": "official_test",
                "record_count": len(gamus_test),
                "sample_ids_sha256": canonical_json_sha256(
                    [record.sample_id for record in gamus_test.records]
                ),
            },
            "legacy": {
                "manifest_path": str(legacy_test_path),
                "manifest_sha256": file_sha256(legacy_test_path),
                "record_count": len(legacy_test_records),
            },
        }

    workers_value = (
        guard_data.get("num_workers", 0)
        if guard_config.get("_evaluation_num_workers") is None
        else guard_config["_evaluation_num_workers"]
    )
    workers = int(workers_value)
    if workers < 0:
        raise RoutedEvaluationPlanError("num_workers cannot be negative")
    evidence = {
        "authenticated_validation_inventory": validation_inventory,
        "legacy_validation_manifest": {
            "path": str(legacy_val_path),
            "sha256": file_sha256(legacy_val_path),
        },
        "test_inventory": test_evidence,
        "preprocessing": {
            **expected_generation,
            "crop": "center",
            "require_relative_priors": True,
            "legacy_relative_prior_policy": "stored_01",
        },
        "implementation": implementation_evidence,
    }
    return validation, tests, guards, evidence, workers


def _loader(dataset: Dataset, *, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


def _evaluate_suites(
    bundle: GuardedRoutedSurfaceBundle,
    datasets: Mapping[str, Dataset],
    *,
    device: str,
    precision: str,
    workers: int,
    label: str,
) -> tuple[dict[str, object], dict[str, object]]:
    protected: dict[str, object] = {}
    routed: dict[str, object] = {}
    for suite in sorted(datasets):
        dataset = datasets[suite]
        print(f"[{label}] {suite}: protected endpoint ({len(dataset)} scenes)")
        protected[suite] = evaluate_dense_surface_model(
            bundle.model.shared_model,
            tqdm(
                _loader(dataset, workers=workers),
                desc=f"{label}/{suite}/protected",
                unit="scene",
            ),
            device=device,
            precision=precision,
        )
        print(f"[{label}] {suite}: routed model ({len(dataset)} scenes)")
        routed[suite] = evaluate_dense_surface_model(
            bundle.model,
            tqdm(
                _loader(dataset, workers=workers),
                desc=f"{label}/{suite}/routed",
                unit="scene",
            ),
            device=device,
            precision=precision,
        )
    return protected, routed


def _file_identity(path_like: str | Path) -> dict[str, object]:
    path = _resolve(path_like)
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None}
    return {
        "path": str(path),
        "exists": True,
        "sha256": file_sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _protected_file_identities(
    bundle: GuardedRoutedSurfaceBundle, pointer_path: Path
) -> dict[str, dict[str, object]]:
    endpoint = bundle.diagnostics.endpoint_diagnostics
    result = {
        "router_artifact": _file_identity(bundle.diagnostics.router_artifact_path),
        "router_report": _file_identity(bundle.diagnostics.router_report_path),
        "protected_endpoint": _file_identity(endpoint.protected.path),
        "gamus_stage1_endpoint": _file_identity(endpoint.gamus_stage1.path),
        "live_application_pointer": _file_identity(pointer_path),
    }
    if endpoint.base_checkpoint is not None:
        result["base_checkpoint"] = _file_identity(endpoint.base_checkpoint.path)
    return result


def _verify_authenticated_file_identities(
    bundle: GuardedRoutedSurfaceBundle,
    identities: Mapping[str, Mapping[str, object]],
) -> None:
    endpoint = bundle.diagnostics.endpoint_diagnostics
    expected = {
        "router_artifact": bundle.diagnostics.router_artifact_sha256,
        "router_report": bundle.diagnostics.router_report_sha256,
        "protected_endpoint": endpoint.protected.sha256,
        "gamus_stage1_endpoint": endpoint.gamus_stage1.sha256,
    }
    if endpoint.base_checkpoint is not None:
        expected["base_checkpoint"] = endpoint.base_checkpoint.sha256
    mismatches = [
        name
        for name, digest in expected.items()
        if identities.get(name, {}).get("exists") is not True
        or identities.get(name, {}).get("sha256") != digest
    ]
    if mismatches:
        raise RoutedEvaluationPlanError(
            "Authenticated model files changed after loading: "
            + ", ".join(sorted(mismatches))
        )


def _atomic_json(payload: Mapping[str, object], path_like: str | Path) -> Path:
    path = _resolve(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def run(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.num_workers is not None and args.num_workers < 0:
        raise RoutedEvaluationPlanError("--num-workers cannot be negative")
    if args.precision != "fp32" and not str(args.device).lower().startswith("cuda"):
        raise RoutedEvaluationPlanError("fp16/bf16 evaluation requires CUDA")
    guard_path, guard_config_value = _load_yaml_mapping(
        args.guard_config, role="hard-guard config"
    )
    guard_config = dict(guard_config_value)
    if args.num_workers is not None:
        guard_config["_evaluation_num_workers"] = int(args.num_workers)

    bundle = load_guarded_routed_surface(
        _resolve(args.router_artifact),
        _resolve(args.router_report),
        device=args.device,
    )
    provenance = _require_mapping(
        bundle.router_report.get("record_provenance"),
        role="authenticated router provenance",
    )
    validation, tests, guards, data_evidence, workers = _validation_plan(
        provenance, guard_config, include_test=bool(args.include_test)
    )

    before = _protected_file_identities(bundle, DEFAULT_POINTER)
    _verify_authenticated_file_identities(bundle, before)
    output_path = _resolve(args.output)
    protected_paths = {
        os.path.normcase(str(_resolve(item["path"])))
        for item in before.values()
        if isinstance(item.get("path"), str)
    }
    protected_paths.add(os.path.normcase(str(guard_path)))
    legacy_manifest = data_evidence["legacy_validation_manifest"]
    if isinstance(legacy_manifest, Mapping):
        protected_paths.add(os.path.normcase(str(_resolve(legacy_manifest["path"]))))
    test_inventory = data_evidence["test_inventory"]
    if isinstance(test_inventory, Mapping):
        legacy_test = test_inventory.get("legacy")
        if isinstance(legacy_test, Mapping) and isinstance(
            legacy_test.get("manifest_path"), str
        ):
            protected_paths.add(
                os.path.normcase(str(_resolve(legacy_test["manifest_path"])))
            )
    if os.path.normcase(str(output_path)) in protected_paths:
        raise RoutedEvaluationPlanError(
            "Evaluation output cannot overwrite an artifact, config, manifest, or "
            "application pointer"
        )
    guard_hash_before = file_sha256(guard_path)
    protected_validation, routed_validation = _evaluate_suites(
        bundle,
        validation,
        device=args.device,
        precision=args.precision,
        workers=workers,
        label="validation",
    )
    guard_results, passes_guards = evaluate_validation_guards(
        routed_validation, protected_validation, guards
    )

    test_results: dict[str, object] | None = None
    if args.include_test:
        protected_test, routed_test = _evaluate_suites(
            bundle,
            tests,
            device=args.device,
            precision=args.precision,
            workers=workers,
            label="test",
        )
        test_results = {
            "protected_reference": protected_test,
            "routed": routed_test,
            "selection_note": (
                "Held-out test results are reporting-only and were not used to "
                "choose the router, threshold, endpoints, or guards."
            ),
        }

    after = _protected_file_identities(bundle, DEFAULT_POINTER)
    if canonical_json_sha256(before) != canonical_json_sha256(after):
        raise RuntimeError(
            "A protected artifact or the live application pointer changed during "
            "read-only evaluation"
        )
    if file_sha256(guard_path) != guard_hash_before:
        raise RuntimeError("Hard-guard config changed during evaluation")
    current_validation_inventory = {
        suite: dataset_inventory_fingerprint(
            dataset, source=suite, split="val", partition="calibration"
        )
        for suite, dataset in validation.items()
    }
    if canonical_json_sha256(current_validation_inventory) != canonical_json_sha256(
        data_evidence["authenticated_validation_inventory"]
    ):
        raise RuntimeError("Validation data changed during evaluation")
    if file_sha256(_resolve(legacy_manifest["path"])) != legacy_manifest["sha256"]:
        raise RuntimeError("Legacy validation manifest changed during evaluation")
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eligible_for_later_promotion_review": bool(passes_guards),
        "promotion_performed": False,
        "live_application_pointer_changed": False,
        "router_bundle": bundle.diagnostics.as_dict(),
        "router_training_report_sha256": bundle.diagnostics.router_report_sha256,
        "guard_config": {
            "path": str(guard_path),
            "sha256": guard_hash_before,
        },
        "data_evidence": data_evidence,
        "validation": {
            "protected_reference": protected_validation,
            "routed": routed_validation,
            "hard_guards": guard_results,
        },
        "test_splits_evaluated": bool(args.include_test),
        "test": test_results,
        "read_only_identity_before": before,
        "read_only_identity_after": after,
        "runtime": {
            "device": str(args.device),
            "precision": str(args.precision),
            "batch_size": 1,
            "num_workers": workers,
        },
    }
    saved = _atomic_json(report, output_path)
    return saved, bool(passes_guards)


def main() -> None:
    args = parse_args()
    output, passes = run(args)
    print(f"Validation report: {output}")
    print(f"Both hard-guard suites passed: {passes}")
    print("Application pointer changed: False")
    if not passes:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
