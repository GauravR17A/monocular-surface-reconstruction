"""Audit current Open-Canopy isolation and plan a stronger train holdout."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from msr.data.open_canopy_holdout import (
    SUPPORTED_ISOLATION_FIELDS,
    plan_holdout,
    split_overlap_audit,
)


SCHEMA = "msr.open_canopy.holdout_plan.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(
    provenance_report: str | Path,
    *,
    source_split: str,
    isolation_fields: list[str],
    holdout_fraction: float,
    seed: str,
) -> dict[str, Any]:
    path = Path(provenance_report).expanduser().resolve()
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if provenance.get("schema") != "msr.open_canopy.provenance.v1":
        raise ValueError("unsupported Open-Canopy provenance report schema")
    records = list(provenance.get("records", []))
    eligible = [record for record in records if record.get("split") == source_split]
    if not eligible:
        raise ValueError(f"no Open-Canopy records found for source split {source_split}")
    current_audit = split_overlap_audit(records, isolation_fields)
    plan = plan_holdout(
        eligible,
        isolation_fields,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    lookup = {record["sample_id"]: record for record in eligible}
    assignments = []
    for assignment in plan.pop("assignments"):
        source = lookup[assignment["sample_id"]]
        assignments.append(
            {
                **assignment,
                "official_split": source_split,
                "region": source["region"],
                "image_name": source["image_name"],
                "imagery_acquisition_date": source["imagery_acquisition_date"],
                "lidar_acquisition_date": source["lidar_acquisition_date"],
            }
        )
    return {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_report": str(path),
        "provenance_report_sha256": _sha256(path),
        "planning_scope": (
            f"accepted official {source_split} metadata only; official validation and "
            "locked-test pixels are not used to choose this plan"
        ),
        "current_accepted_split_isolation": current_audit,
        "plan": plan,
        "assignments": assignments,
        "limitations": [
            "The current accepted 600/120/120 manifests are region-disjoint but share "
            "many larger SPOT source mosaics across official splits.",
            "This derived holdout is valid only for the explicitly listed isolation "
            "fields; it does not prove transfer to a new sensor or country.",
            "LiDAR and imagery acquisition dates describe different sensors and may "
            "not match; date isolation is optional and must state which date is used.",
        ],
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _assignment_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provenance_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--isolate",
        action="append",
        choices=sorted(SUPPORTED_ISOLATION_FIELDS),
        dest="isolation_fields",
        help="Repeat to keep any shared identity in one partition (default: region).",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--seed", default="msr-open-canopy-holdout-v1")
    parser.add_argument("--assignment-csv", type=Path)
    parser.add_argument(
        "--require-current-splits-disjoint",
        action="store_true",
        help="Exit nonzero when existing official splits overlap on selected fields.",
    )
    args = parser.parse_args()
    fields = args.isolation_fields or ["region"]
    report = build_report(
        args.provenance_report,
        source_split=args.source_split,
        isolation_fields=fields,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    output = args.output.expanduser().resolve()
    _atomic_json(output, report)
    if args.assignment_csv:
        _assignment_csv(args.assignment_csv.expanduser().resolve(), report["assignments"])
    concise = {
        "current_accepted_split_isolation_passes": report[
            "current_accepted_split_isolation"
        ]["passes"],
        **{
            key: report["plan"][key]
            for key in (
                "isolation_fields",
                "component_count",
                "largest_component_samples",
                "learning_samples",
                "holdout_samples",
                "achieved_holdout_fraction",
            )
        },
    }
    print(json.dumps(concise, indent=2))
    if (
        args.require_current_splits_disjoint
        and not report["current_accepted_split_isolation"]["passes"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
