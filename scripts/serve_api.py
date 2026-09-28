"""Start the local Monocular Surface Reconstruction GPU inference API."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from msr.api.app import InferenceRuntime, create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/release/height_model.pt"),
    )
    parser.add_argument(
        "--relative-model",
        default="models/foundation/depth-anything-v2-small-hf",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    runtime = InferenceRuntime(
        checkpoint=args.checkpoint,
        relative_model=args.relative_model,
        device=args.device,
    )
    app = create_app(runtime=runtime, results_root="outputs/web_jobs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
