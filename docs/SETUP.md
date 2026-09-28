# Installation, operation and reproduction

## Supported setup

The checked development host uses Windows x64, Python 3.10.11, Node.js 22.x and an
NVIDIA RTX 4070 Laptop GPU with about 8 GiB VRAM. Python declares support for
3.10–3.12. CPU inference is available but has not been given the same performance
claim as GPU execution. Linux/macOS commands are provided as portable equivalents,
not as claims of a completed unrelated-host installation.

Clone the public repository and work from its root. Do not copy a `.venv` from
another machine. Create the environment and install a compatible Torch pair:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
```

For a CPU-only installation, use the same Torch versions with the PyTorch CPU
wheel index instead of `cu128`, then run with `--device cpu`. On Linux/macOS,
replace `.venv/Scripts/python.exe` with `.venv/bin/python`. The direct application
dependency versions are recorded in `requirements.txt`; `pyproject.toml` describes
package extras. The frontend has an npm lockfile.

## Acquire model artifacts

Read the third-party notices before acquiring the restricted research weights.
Then run:

```powershell
.venv/Scripts/python.exe scripts/download_models.py --include-foundation
.venv/Scripts/python.exe scripts/download_models.py --include-foundation --verify-only
```

The manifest pins the packaged height/classifier SHA-256 values and a foundation
revision. Downloads use temporary files, verify size/hash and replace the target
only after verification. The downloader does not deserialize model payloads.
Training archives are not required for normal inference. They are required for
reproducing full training or historical dataset-dependent analyses.

## Start the application

In the first terminal:

```powershell
.venv/Scripts/python.exe scripts/serve_api.py --device cuda
```

In a second terminal:

```powershell
npm --prefix viewer ci
npm --prefix viewer run build
npm --prefix viewer run start -- --hostname 127.0.0.1 --port 3000
```

Open `http://127.0.0.1:3000`. Saved sample scenes work without downloading model
weights, but automatic classification and new-image reconstruction require the
API. `/api/health` reports the configured device; the first prediction still has
to load and verify the actual model path.

The default API and viewer use ports 8000 and 3000. `--port` selects the API port;
`NEXT_PUBLIC_MSR_API_URL` selects the frontend's API base. `MSR_VIEWER_ORIGINS`
configures the API's comma-separated allowed viewer origins for a deliberately
changed local port. Public hosting is outside this release's verified scope.

## Verify and inspect evidence

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest tests -q
npm --prefix viewer run test
npm --prefix viewer run typecheck
npm --prefix viewer run build
.venv/Scripts/python.exe scripts/verify_publication.py
```

The public test suite explicitly skips 26 historical checks bound to original
source hashes or unbundled experiment archives. They passed in the original
development workspace, but the renamed source package is not that exact sealed
artifact. `--run-historical-seals` attempts them only when their original
prerequisites are intentionally available; it is not required for the public
inference workflow. Numerical and application assertions are not loosened.

## Reproduce research responsibly

Historical configs and scripts preserve experiment structure, but some locators
point to historical data/experiment arrangements. Start from the corresponding
protocol, acquire the exact allowed source revision, prepare the manifests and
validate shapes, units, masks and geographic overlap. Create new source/config/data
bindings for a new run rather than editing an old seal to imply identity.

Use the declared comparator, budget, masks, selector and safety gates. Keep full
scene and app-tiling evaluation separate. Do not repurpose a consumed external test
as new validation. The reserved Bay of Plenty package is not needed for setup,
smoke tests or this release verification and remains untouched.

## Rebuild the handbook

Install the documentation dependencies and run the included builder:

```powershell
.venv/Scripts/python.exe -m pip install -e ".[docs]"
.venv/Scripts/python.exe scripts/build_documentation.py
```

The chapter order is explicit in the builder. Current chapters precede dated
history; the source inventory and evidence paths remain available. The builder
creates a contents page and PDF bookmarks from those sources. Figures use actual
recorded values or clearly labelled architecture diagrams.

## Recovery

If the viewer loads but classification fails, check the local API and classifier
artifact hash. If inference fails, check the configured device, height and
foundation files, supported input type and server log. A relative image cannot be
made geographically metric by changing a display setting. Do not select a new
training checkpoint merely to clear an error.

Local `outputs/web_jobs/` stores uploads and derived products. Back up selected
results explicitly. Stop only processes you started; do not terminate unrelated
Python/Node processes on a shared development machine.
