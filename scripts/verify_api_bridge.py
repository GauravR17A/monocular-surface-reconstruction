"""JSON-lines transport for browser verification; opens no listening socket."""
from __future__ import annotations
import base64
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MSR_VIEWER_ORIGINS", "http://msr.test")
from fastapi.testclient import TestClient
from msr.api.app import InferenceRuntime, create_app

runtime = InferenceRuntime(checkpoint=ROOT / "models/release/height_model.pt",
                           relative_model=str(ROOT / "models/foundation/depth-anything-v2-small-hf"),
                           device=os.environ.get("MSR_TEST_DEVICE", "cuda"))
client = TestClient(create_app(runtime=runtime, results_root=ROOT / "outputs/browser-verification"))
for line in sys.stdin:
    try:
        request = json.loads(line)
        started = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            response = client.request(request["method"], request["path"],
                                      headers=request.get("headers", {}),
                                      content=base64.b64decode(request.get("body", "")))
        result = {"id": request["id"], "status": response.status_code,
                  "headers": dict(response.headers), "body": base64.b64encode(response.content).decode(),
                  "elapsed_seconds": time.perf_counter() - started}
    except Exception as error:
        result = {"id": request.get("id"), "error": str(error)}
    print(json.dumps(result), flush=True)
