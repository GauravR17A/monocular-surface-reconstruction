"""Fail-closed preflight for the DC+PHL-only GAMUS classifier candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msr.data.gamus_geographic_candidate import (  # noqa: E402
    validate_geographic_candidate_config,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/multidomain_surface_gamus_six_class_head_only_geographic_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("candidate config must decode to a mapping")
    report = validate_geographic_candidate_config(
        config,
        project_root=PROJECT_ROOT,
    )
    report["candidate_config"] = str(config_path)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if output.exists() and output.read_bytes() != content:
            raise FileExistsError(
                f"refusing to overwrite a different preflight report: {output}"
            )
        if not output.exists():
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
