"""Authenticate Monocular Surface Reconstruction's height scorecard and assess one candidate.

The command is read-only unless ``--output`` is supplied.  Even then it writes
only a new decision JSON and refuses to overwrite it.  It never changes the
application pointer or any checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from msr.evaluation.height_scorecard import (
    assess_candidate,
    authenticate_scorecard,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite decision artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scorecard",
        type=Path,
        default=PROJECT_ROOT / "configs/height_scorecard_v1.yaml",
    )
    parser.add_argument("--candidate-report", type=Path)
    parser.add_argument("--candidate-model-key")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    authenticated = authenticate_scorecard(
        args.scorecard, project_root=PROJECT_ROOT
    )
    if (args.candidate_report is None) != (args.candidate_model_key is None):
        raise ValueError(
            "--candidate-report and --candidate-model-key must be supplied together"
        )
    if args.candidate_report is None:
        result: dict[str, Any] = {
            "status": "baseline_authenticated",
            "scorecard": {
                "path": str(authenticated.path),
                "sha256": authenticated.sha256,
            },
            "official_test_used": False,
            "protected_checkpoint_sha256": authenticated.config["baseline"][
                "checkpoint_sha256"
            ],
            "live_application_changed": False,
            "promotion_performed": False,
            "promotion_eligible": False,
            "external_height_status": authenticated.config[
                "external_height_evidence"
            ]["status"],
        }
    else:
        result = assess_candidate(
            authenticated,
            args.candidate_report,
            candidate_model_key=args.candidate_model_key,
            project_root=PROJECT_ROOT,
        )
    if args.output is not None:
        saved = _atomic_json(args.output, result)
        result = {**result, "saved_to": str(saved)}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
