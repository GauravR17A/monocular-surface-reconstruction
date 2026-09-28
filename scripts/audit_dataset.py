"""Audit every pair in a manifest before any model is trained."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from msr.data.manifest import audit_pair, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--height-min-m", type=float, default=0.0)
    parser.add_argument("--height-max-m", type=float, default=1000.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    audits = [
        audit_pair(
            record,
            height_min_m=args.height_min_m,
            height_max_m=args.height_max_m,
        )
        for record in load_manifest(args.manifest)
    ]
    issues: list[dict[str, str]] = []
    for audit in audits:
        if not audit.aligned:
            issues.append({"sample_id": audit.sample_id, "issue": "unaligned RGB/height pair"})
        if audit.rgb_band_count < 3:
            issues.append({"sample_id": audit.sample_id, "issue": "RGB has fewer than 3 bands"})
        if audit.height_band_count != 1:
            issues.append({"sample_id": audit.sample_id, "issue": "height raster is not single-band"})
        if audit.valid_height_fraction <= 0:
            issues.append({"sample_id": audit.sample_id, "issue": "no valid height pixels"})
        if audit.below_height_range_fraction > 0 or audit.above_height_range_fraction > 0:
            issues.append(
                {
                    "sample_id": audit.sample_id,
                    "issue": (
                        "height values outside configured range "
                        f"(below={audit.below_height_range_fraction:.6f}, "
                        f"above={audit.above_height_range_fraction:.6f})"
                    ),
                }
            )
    report = {
        "sample_count": len(audits),
        "aligned_count": sum(audit.aligned for audit in audits),
        "issue_count": len(issues),
        "issues": issues,
        "samples": [asdict(audit) | {"aligned": audit.aligned} for audit in audits],
    }
    serialized = json.dumps(report, indent=2)
    if args.summary_only:
        print(
            json.dumps(
                {key: report[key] for key in ("sample_count", "aligned_count", "issue_count", "issues")},
                indent=2,
            )
        )
    else:
        print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if issues:
        raise SystemExit(f"Dataset audit failed with {len(issues)} issue(s)")


if __name__ == "__main__":
    main()
