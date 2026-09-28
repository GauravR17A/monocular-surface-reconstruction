from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import pytest
import yaml

from msr.data.gamus_geographic_candidate import (
    EXPECTED_SPATIAL_REFINED_EXPERIMENT_NAME,
    GAMUS_GEOGRAPHIC_CANDIDATE_SCHEMA,
    GAMUS_SAMPLING_INDEX_SCHEMA,
    GAMUS_SPATIAL_REFINED_GEOGRAPHIC_CANDIDATE_SCHEMA,
    derive_learning_sampling_index,
    validate_geographic_candidate_config,
)
from msr.data.gamus_geographic_holdout import (
    build_city_holdout_contract,
    build_holdout_approved_index,
    build_learning_approved_index,
    canonical_sha256,
    file_sha256,
    load_development_index,
    validate_holdout_contract,
)


ROOT = Path(__file__).parents[1]


def _split(ids: list[str], *, official_count: int | None = None) -> dict[str, object]:
    count = len(ids) if official_count is None else official_count
    return {
        "approved_count": len(ids),
        "approved_sample_ids": ids,
        "semantic_eligible_sample_ids": ids,
        "height_regression_eligible_sample_ids": ids,
        "source_count": count,
        "expected_official_count": count,
    }


def _development(tmp_path: Path) -> tuple[dict, dict]:
    payload = {
        "schema": "msr.gamus.approved_samples.v1",
        "dataset": "GAMUS",
        "splits": {
            "train": _split(
                ["DC_01_01", "NYC_00001", "PHL_0001"], official_count=5004
            ),
            "val": _split(["DC_02_01", "PHL_0002"], official_count=859),
            "test": _split(["NYC_99999"]),
        },
    }
    source = tmp_path / "approved.json"
    source.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    development = load_development_index(source)
    contract = build_city_holdout_contract(development, holdout_city="NYC")
    return development, contract


def _sampling_row(sample_id: str, probability: float) -> dict[str, object]:
    city = sample_id.split("_", 1)[0]
    return {
        "schema": GAMUS_SAMPLING_INDEX_SCHEMA,
        "sample_id": sample_id,
        "city": city,
        "class_path": f"D:/GAMUS/classes/train/{sample_id}_CLS.h5",
        "class_counts": {
            "ground": 1,
            "buildings": 2,
            "water": int(probability > 0),
            "roads": 3,
            "low_vegetation": 4,
            "trees": 5,
        },
        "uniform_random_crop_384": {"water_hit_probability": probability},
    }


def _write_sampling(path: Path, rows: list[dict[str, object] | str]) -> None:
    lines = [
        row if isinstance(row, str) else json.dumps(row, sort_keys=True)
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_learning_sampling_index_is_exact_deterministic_and_holdout_blind(
    tmp_path: Path,
) -> None:
    _, contract = _development(tmp_path)
    source = tmp_path / "full.jsonl"
    # This deliberately invalid NYC JSON proves the holdout payload is not
    # decoded.  Only its sample_id is used to prove complete exclusion.
    _write_sampling(
        source,
        [
            _sampling_row("PHL_0001", 0.4),
            '{"sample_id":"NYC_00001",THIS_MUST_NOT_BE_PARSED',
            _sampling_row("DC_01_01", 0.0),
        ],
    )
    first, audit = derive_learning_sampling_index(
        source,
        expected_source_sha256=file_sha256(source),
        contract=contract,
    )
    second, _ = derive_learning_sampling_index(
        source,
        expected_source_sha256=file_sha256(source),
        contract=contract,
    )
    assert first == second
    output_rows = [json.loads(line) for line in first.decode().splitlines()]
    assert [row["sample_id"] for row in output_rows] == ["DC_01_01", "PHL_0001"]
    assert audit["learning_rows"] == 2
    assert audit["excluded_geographic_holdout_rows"] == 1
    assert audit["official_test_rows_discovered_or_used"] == 0


def test_sampling_derivation_fails_on_hash_or_inventory_change(tmp_path: Path) -> None:
    _, contract = _development(tmp_path)
    source = tmp_path / "full.jsonl"
    _write_sampling(
        source,
        [
            _sampling_row("DC_01_01", 0.0),
            _sampling_row("PHL_0001", 0.2),
            _sampling_row("NYC_00001", 0.1),
        ],
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        derive_learning_sampling_index(
            source,
            expected_source_sha256="0" * 64,
            contract=contract,
        )
    source.write_text(
        source.read_text(encoding="utf-8")
        + json.dumps(_sampling_row("NYC_00002", 0.0))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside the sealed train partition"):
        derive_learning_sampling_index(
            source,
            expected_source_sha256=file_sha256(source),
            contract=contract,
        )


def test_contract_rejects_role_source_split_mismatch(tmp_path: Path) -> None:
    _, contract = _development(tmp_path)
    changed = deepcopy(contract)
    learning = next(row for row in changed["assignments"] if row["role"] == "learning")
    learning["source_split"] = "val"
    changed["assignment_sha256"] = canonical_sha256(changed["assignments"])
    with pytest.raises(ValueError, match="source split mismatch"):
        validate_holdout_contract(changed)


def test_candidate_preflight_proves_recipe_and_city_exclusion(tmp_path: Path) -> None:
    development, contract = _development(tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")
    learning_index = build_learning_approved_index(development, contract)
    learning_index_path = tmp_path / "learning.json"
    learning_index_path.write_text(
        json.dumps(learning_index, sort_keys=True), encoding="utf-8"
    )
    holdout_index = build_holdout_approved_index(development, contract)
    holdout_index_path = tmp_path / "holdout.json"
    holdout_index_path.write_text(
        json.dumps(holdout_index, sort_keys=True), encoding="utf-8"
    )
    source_sampling = tmp_path / "full.jsonl"
    _write_sampling(
        source_sampling,
        [
            _sampling_row("DC_01_01", 0.0),
            _sampling_row("NYC_00001", 0.8),
            _sampling_row("PHL_0001", 0.4),
        ],
    )
    sampling_bytes, _ = derive_learning_sampling_index(
        source_sampling,
        expected_source_sha256=file_sha256(source_sampling),
        contract=contract,
    )
    learning_sampling = tmp_path / "learning.jsonl"
    learning_sampling.write_bytes(sampling_bytes)

    protected = tmp_path / "protected.pt"
    protected.write_bytes(b"protected-state")
    pointer = tmp_path / "pointer.txt"
    pointer.write_text(str(protected), encoding="utf-8")
    model = {
        "base_checkpoint": "unused.pt",
        "initial_checkpoint": str(protected),
        "fine_semantic_classes": 6,
    }
    training = {
        "epochs": 8,
        "parameter_groups": [
            {"prefixes": ["fine_semantic_head."], "learning_rate": 0.001}
        ],
        "height_weight": 0.0,
        "semantic_weight": 0.0,
        "fine_semantic_weight": 1.0,
        "building_weight": 0.0,
        "canopy_weight": 0.0,
        "weight_decay": 0.0,
    }
    evaluation = {"primary_selection": {"suite": "gamus", "metric": "macro_f1"}}
    source_data = {
        "dataset": "gamus",
        "root": "unused",
        "approved_index_path": "full.json",
        "approved_index_file_sha256": "1" * 64,
        "fine_class_sampling_index_path": "full.jsonl",
        "fine_class_sampling_index_sha256": "2" * 64,
        "require_approved_index": True,
        "require_complete_official_splits": True,
    }
    source_recipe = {
        "experiment": {"name": "source", "seed": 7, "output_root": "experiments"},
        "protocol": {"schema": "source"},
        "data": source_data,
        "model": model,
        "training": training,
        "evaluation": evaluation,
    }
    source_recipe_path = tmp_path / "source.yaml"
    source_recipe_path.write_text(yaml.safe_dump(source_recipe), encoding="utf-8")
    candidate_data = deepcopy(source_data)
    candidate_data.update(
        {
            "approved_index_path": str(learning_index_path),
            "approved_index_file_sha256": file_sha256(learning_index_path),
            "fine_class_sampling_index_path": str(learning_sampling),
            "fine_class_sampling_index_sha256": file_sha256(learning_sampling),
        }
    )
    candidate = {
        "experiment": {
            "name": "multidomain_surface_gamus_six_class_head_only_geographic_v1",
            "seed": 7,
            "output_root": "experiments",
        },
        "protocol": {
            "schema": GAMUS_GEOGRAPHIC_CANDIDATE_SCHEMA,
            "source_recipe_config": str(source_recipe_path),
            "source_recipe_config_sha256": file_sha256(source_recipe_path),
            "holdout_contract": str(contract_path),
            "holdout_contract_sha256": file_sha256(contract_path),
            "learning_cities": ["DC", "PHL"],
            "development_validation_cities": ["DC", "PHL"],
            "locked_holdout_city": "NYC",
            "locked_holdout_approved_index": str(holdout_index_path),
            "locked_holdout_approved_index_sha256": file_sha256(holdout_index_path),
            "expected_counts": {
                "learning": 2,
                "development_validation": 2,
                "locked_geographic_holdout": 1,
            },
            "protected_checkpoint_sha256": file_sha256(protected),
            "live_pointer_file": str(pointer),
            "live_pointer_file_sha256": file_sha256(pointer),
            "official_test_policy": "never_constructed_or_discovered",
            "auto_promotion": False,
        },
        "data": candidate_data,
        "model": model,
        "training": training,
        "evaluation": evaluation,
    }
    report = validate_geographic_candidate_config(candidate, project_root=tmp_path)
    assert report["learning"]["tiles"] == 2
    assert report["epoch_selection"]["tiles"] == 2
    assert report["locked_post_freeze_evaluation"]["tiles"] == 1
    assert report["locked_post_freeze_evaluation"]["city"] == "NYC"
    assert report["locked_post_freeze_evaluation"]["opened_by_preflight"] is False
    assert report["official_test_constructed_or_discovered"] is False

    spatial = deepcopy(candidate)
    spatial["experiment"]["name"] = EXPECTED_SPATIAL_REFINED_EXPERIMENT_NAME
    spatial["protocol"]["schema"] = (
        GAMUS_SPATIAL_REFINED_GEOGRAPHIC_CANDIDATE_SCHEMA
    )
    spatial_report = validate_geographic_candidate_config(
        spatial, project_root=tmp_path
    )
    assert spatial_report["experiment"] == EXPECTED_SPATIAL_REFINED_EXPERIMENT_NAME

    leaked = deepcopy(candidate)
    leaked["data"]["approved_index_path"] = str(development["source_path"])
    leaked["data"]["approved_index_file_sha256"] = file_sha256(
        development["source_path"]
    )
    with pytest.raises(ValueError):
        validate_geographic_candidate_config(leaked, project_root=tmp_path)


def test_spatial_geographic_runner_pins_its_protocol_inputs() -> None:
    runner = (
        ROOT / "scripts" / "run_gamus_spatial_refined_geographic_v1.ps1"
    ).read_text(encoding="utf-8")
    files = {
        "expectedConfigSha256": (
            ROOT
            / "configs"
            / "multidomain_surface_gamus_six_class_spatial_refined_geographic_v1.yaml"
        ),
        "expectedEvaluationProtocolSha256": (
            ROOT / "configs" / "gamus_locked_nyc_spatial_refined_evaluation_v1.yaml"
        ),
        "expectedGeographicModuleSha256": (
            ROOT / "src" / "msr" / "data" / "gamus_geographic_candidate.py"
        ),
        "expectedLockedEvaluatorSha256": (
            ROOT / "scripts" / "evaluate_gamus_locked_nyc.py"
        ),
    }
    for variable, path in files.items():
        match = re.search(rf'\${variable} = "([0-9a-f]{{64}})"', runner)
        assert match is not None
        assert match.group(1) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "$PreflightOnly" in runner
    assert "NYC holdout remains unconsumed" in runner
    assert "official GAMUS test and app promotion are forbidden" in runner
