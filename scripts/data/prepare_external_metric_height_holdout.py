#!/usr/bin/env python
"""Acquire, align, verify, and later seal the external NZ height holdout.

This script never imports a model and never performs inference.  The
``mark-consumed`` mode exists so a future evaluator can make consumption
explicit immediately before its first model read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from msr.evaluation.external_height_holdout import (  # noqa: E402
    ExternalHoldoutError,
    acquire_sources,
    load_contract,
    mark_consumed,
    prepare_package,
    resolve_data_root,
    seal_evaluation_plan,
    verify_contract_seal,
    verify_prepared_package,
)


DEFAULT_CONTRACT = REPOSITORY_ROOT / "configs" / "external_metric_height_nz_bop_v1.json"


class ProgressPrinter:
    def __init__(self) -> None:
        self._last_percent: dict[str, int] = {}

    def __call__(self, name: str, downloaded: int, total: int) -> None:
        percent = min(100, int(downloaded * 100 / total)) if total else 0
        previous = self._last_percent.get(name, -10)
        if percent == 100 or percent >= previous + 10:
            print(f"download {name}: {percent:3d}% ({downloaded:,}/{total:,} bytes)", flush=True)
            self._last_percent[name] = percent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("acquire-prepare", "verify", "seal-plan", "mark-consumed"),
    )
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--allowed-root", type=Path, default=Path(r"D:\MSRData"))
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--code-path", type=Path)
    parser.add_argument("--app-pointer", type=Path)
    parser.add_argument("--reason")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract_sha256 = verify_contract_seal(args.contract, args.seal)
    contract = load_contract(args.contract)
    root = resolve_data_root(contract, allowed_root=args.allowed_root)

    if args.mode == "acquire-prepare":
        acquire_sources(contract, root, progress=ProgressPrinter())
        manifest_path = prepare_package(contract, contract_sha256, root)
        package = verify_prepared_package(contract, contract_sha256, root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "status": "prepared_unconsumed",
                    "benchmark_id": contract["benchmark_id"],
                    "root": str(root),
                    "contract_sha256": contract_sha256,
                    "manifest_sha256": package.manifest_sha256,
                    "support_counts": manifest["support_counts"],
                    "inference_run": False,
                },
                indent=2,
            )
        )
        return 0

    package = verify_prepared_package(contract, contract_sha256, root)
    if args.mode == "verify":
        print(
            json.dumps(
                {
                    "status": "consumed" if package.consumed else "prepared_unconsumed",
                    "benchmark_id": contract["benchmark_id"],
                    "root": str(root),
                    "contract_sha256": package.contract_sha256,
                    "manifest_sha256": package.manifest_sha256,
                },
                indent=2,
            )
        )
        return 0

    if args.mode == "seal-plan":
        missing = [
            flag
            for flag, value in (
                ("--candidate-checkpoint", args.candidate_checkpoint),
                ("--candidate-config", args.candidate_config),
                ("--code-path", args.code_path),
            )
            if value is None
        ]
        if missing:
            raise ExternalHoldoutError(f"seal-plan requires: {', '.join(missing)}")
        plan = seal_evaluation_plan(
            package,
            contract,
            candidate_checkpoint=args.candidate_checkpoint,
            candidate_config=args.candidate_config,
            code_path=args.code_path,
            app_pointer=args.app_pointer,
        )
        print(f"sealed evaluation plan: {plan}")
        return 0

    if not args.reason:
        raise ExternalHoldoutError("mark-consumed requires --reason")
    marker = mark_consumed(package, reason=args.reason)
    print(f"holdout irreversibly marked consumed: {marker}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExternalHoldoutError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
