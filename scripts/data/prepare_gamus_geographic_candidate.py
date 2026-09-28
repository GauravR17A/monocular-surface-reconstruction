"""Prepare immutable DC+PHL learning artifacts for the NYC-blind candidate.

This is a metadata-only preparation step.  It does not open imagery, semantic
rasters, height rasters, cached priors, or anything in the official test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msr.data.gamus_geographic_candidate import (  # noqa: E402
    derive_learning_sampling_index,
)
from msr.data.gamus_geographic_holdout import (  # noqa: E402
    build_holdout_approved_index,
    file_sha256,
    load_development_index,
    load_sealed_holdout_contract,
    role_sample_ids,
)


DEFAULT_SOURCE_APPROVED_INDEX = Path(
    "D:/MSRData/GAMUS_quality_contract_v1_1/approved_samples.json"
)
DEFAULT_SOURCE_APPROVED_INDEX_SHA256 = (
    "5320d2e97be357b1e1725d7f2d9640493522ae3d13f1a051b63de55b530c05aa"
)
DEFAULT_CONTRACT = (
    PROJECT_ROOT
    / "outputs/data_audits/gamus_geographic_holdout_city_nyc_v1/contract.json"
)
DEFAULT_CONTRACT_SHA256 = (
    "a8a5b0dfaa3a6338a590a12dc7eeab74734f26b465e526b05a15833e9cf02435"
)
DEFAULT_SOURCE_SAMPLING_INDEX = (
    PROJECT_ROOT / "outputs/analysis/gamus_train_six_class_tile_index_v1.jsonl"
)
DEFAULT_SOURCE_SAMPLING_INDEX_SHA256 = (
    "afdd9178c0e744ea20e52f88999130110c3b880dd499e25ea79dff0a3862363b"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs/data_audits/gamus_head_only_geographic_candidate_v1"
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _immutable_write(path: Path, content: bytes) -> None:
    """Write once, or accept an identical previously sealed artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        raise FileExistsError(f"refusing to overwrite different sealed artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def prepare(
    *,
    source_approved_index: Path,
    expected_source_approved_index_sha256: str,
    contract_path: Path,
    expected_contract_sha256: str,
    source_sampling_index: Path,
    expected_source_sampling_index_sha256: str,
    output_root: Path,
) -> dict[str, object]:
    development = load_development_index(source_approved_index)
    if development["source_sha256"] != expected_source_approved_index_sha256:
        raise ValueError(
            "source approved-index SHA-256 mismatch: expected "
            f"{expected_source_approved_index_sha256}, found "
            f"{development['source_sha256']}"
        )
    contract = load_sealed_holdout_contract(
        contract_path,
        expected_sha256=expected_contract_sha256,
    )
    if contract.get("source_approved_index_sha256") != development["source_sha256"]:
        raise ValueError("contract and source approved index do not match")

    sampling_bytes, sampling_audit = derive_learning_sampling_index(
        source_sampling_index,
        expected_source_sha256=expected_source_sampling_index_sha256,
        contract=contract,
    )
    holdout_index = build_holdout_approved_index(development, contract)

    sampling_path = output_root / "gamus_learning_six_class_sampling_index_v1.jsonl"
    sampling_audit_path = output_root / "water_sampling_index_audit.json"
    holdout_index_path = output_root / "nyc_holdout_approved_samples.json"
    _immutable_write(sampling_path, sampling_bytes)
    _immutable_write(sampling_audit_path, _json_bytes(sampling_audit))
    _immutable_write(holdout_index_path, _json_bytes(holdout_index))

    learning_ids = role_sample_ids(contract, "learning")
    validation_ids = role_sample_ids(contract, "development_validation")
    holdout_ids = role_sample_ids(contract, "geographic_holdout")
    report = {
        "schema": "msr.gamus.geographic_candidate_preparation.v1",
        "timestamp_policy": "omitted_for_deterministic_artifact_identity",
        "status": "prepared_not_started",
        "purpose": (
            "head-only six-class candidate trained on DC+PHL, selected on the "
            "existing DC+PHL validation split, with NYC locked until post-freeze"
        ),
        "partition": {
            "learning": {"count": len(learning_ids), "cities": ["DC", "PHL"]},
            "epoch_selection": {
                "count": len(validation_ids),
                "cities": ["DC", "PHL"],
            },
            "locked_post_freeze_evaluation": {
                "count": len(holdout_ids),
                "city": "NYC",
                "opened_during_preparation": False,
            },
        },
        "preparation_data_access": {
            "source_approved_index_metadata": True,
            "sealed_sampling_index": True,
            "learning_sampling_rows_decoded": len(learning_ids),
            "holdout_sampling_rows_decoded": 0,
            "imagery_opened": False,
            "semantic_rasters_opened": False,
            "height_rasters_opened": False,
            "relative_priors_opened": False,
            "official_test_files_discovered_or_opened": False,
        },
        "authenticated_inputs": {
            "source_approved_index": {
                "path": str(source_approved_index.resolve()),
                "sha256": development["source_sha256"],
            },
            "holdout_contract": {
                "path": str(contract_path.resolve()),
                "sha256": file_sha256(contract_path),
                "assignment_sha256": contract.get("assignment_sha256"),
            },
            "source_sampling_index": {
                "path": str(source_sampling_index.resolve()),
                "sha256": file_sha256(source_sampling_index),
            },
        },
        "outputs": {
            "learning_sampling_index": _artifact(sampling_path),
            "learning_sampling_audit": _artifact(sampling_audit_path),
            "locked_nyc_approved_index": _artifact(holdout_index_path),
        },
        "training_started": False,
        "evaluation_started": False,
        "active_app_or_checkpoint_changed": False,
    }
    report_path = output_root / "preparation_report.json"
    _immutable_write(report_path, _json_bytes(report))

    artifact_index = {
        "schema": "msr.gamus.geographic_candidate_artifacts.v1",
        "contract_id": contract.get("contract_id"),
        "artifacts": {
            path.name: _artifact(path)
            for path in (
                sampling_path,
                sampling_audit_path,
                holdout_index_path,
                report_path,
            )
        },
    }
    artifact_index_path = output_root / "artifact_hashes.json"
    _immutable_write(artifact_index_path, _json_bytes(artifact_index))
    return artifact_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-approved-index", type=Path, default=DEFAULT_SOURCE_APPROVED_INDEX)
    parser.add_argument(
        "--source-approved-index-sha256",
        default=DEFAULT_SOURCE_APPROVED_INDEX_SHA256,
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--contract-sha256", default=DEFAULT_CONTRACT_SHA256)
    parser.add_argument("--source-sampling-index", type=Path, default=DEFAULT_SOURCE_SAMPLING_INDEX)
    parser.add_argument(
        "--source-sampling-index-sha256",
        default=DEFAULT_SOURCE_SAMPLING_INDEX_SHA256,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare(
        source_approved_index=args.source_approved_index.expanduser().resolve(),
        expected_source_approved_index_sha256=args.source_approved_index_sha256,
        contract_path=args.contract.expanduser().resolve(),
        expected_contract_sha256=args.contract_sha256,
        source_sampling_index=args.source_sampling_index.expanduser().resolve(),
        expected_source_sampling_index_sha256=args.source_sampling_index_sha256,
        output_root=args.output_root.expanduser().resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
