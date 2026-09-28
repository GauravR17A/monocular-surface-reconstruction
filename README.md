# Monocular Surface Reconstruction

**Technical Documentation — Status as of 28 September 2026**

[![Verify research package](https://github.com/GauravR17A/monocular-surface-reconstruction/actions/workflows/verify.yml/badge.svg)](https://github.com/GauravR17A/monocular-surface-reconstruction/actions/workflows/verify.yml)

A research workspace for turning one overhead RGB image into an estimated surface,
inspecting it in 3D, identifying six land-cover categories, and checking predictions
against an independent reference. The project combines a trained height model,
geospatial calibration, explicit evaluation contracts and an interactive viewer.

**Active research prototype.** Height estimates are not survey measurements. The
current classifier has documented regional regressions. Completed work, rejected
experiments and remaining validation are recorded separately.

[Documentation](docs/README.md) · [Full technical PDF](docs/Monocular_Surface_Reconstruction_Technical_Documentation.pdf) · [Results](docs/RESULTS.md) · [Release checks](docs/RELEASE.md) · [Source credits](THIRD_PARTY_NOTICES.md)

![Research workspace](docs/assets/workspace.png)

## What you can do

| Workflow | Current behavior |
| --- | --- |
| Start with a saved scene | Explore urban, sparse, hilly and forest examples without running inference. |
| Reconstruct an image | Process supported RGB imagery through a local Python API with separately acquired model artifacts. |
| Calibrate a raster | Preserve georeferencing; supply a terrain raster or elevation control points; optionally acquire public terrain. |
| Identify surface classes | Ground, buildings, water, roads, low vegetation and trees, independent of the height model. |
| Explore and inspect | Orbit, fly, walk, inspect source values, use the map and image reference, and sample an elevation profile. |
| Compare and export | Inspect reference differences and scoped metrics; download geospatial products and metadata. |

![Pipeline architecture](docs/assets/architecture.svg)

## Run the viewer

Requirements: Node.js 22.13 or later in the 22.x line and npm. The bundled sample
surfaces support a first visual inspection without Python or downloaded weights.
Automatic classification and new-image reconstruction require the API below.

```powershell
npm --prefix viewer ci
npm --prefix viewer run build
npm --prefix viewer run start -- --hostname 127.0.0.1 --port 3000
```

Open [127.0.0.1:3000](http://127.0.0.1:3000). The initial screen is neutral; choose
a reference scene to begin. The viewer uses React, TypeScript, Three.js, GeoTIFF.js
and the vinext implementation of the Next.js App Router.

## Enable image inference

Use Python 3.10–3.12. The verified development host is Windows x64 with an NVIDIA
RTX 4070 Laptop GPU. The commands below install the documented CUDA build; the
[setup guide](docs/SETUP.md) also describes CPU installation and environment limits.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e . --no-deps
.venv/Scripts/python.exe scripts/download_models.py --include-foundation
.venv/Scripts/python.exe scripts/serve_api.py --device cuda
```

Keep the viewer in a second terminal. The API defaults to `127.0.0.1:8000`. Start
with a bundled scene before uploading a new RGB image. Model artifacts are hash
checked and distributed separately from Git. Read their **non-commercial research
and evaluation restrictions** in the [notices](THIRD_PARTY_NOTICES.md).

## Verify

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest tests -q
npm --prefix viewer run test
npm --prefix viewer run typecheck
npm --prefix viewer run build
.venv/Scripts/python.exe scripts/verify_publication.py
```

Historical checks requiring exact original source seals and unbundled training
archives are reported as skipped in the public suite, not passing. The
[release report](docs/RELEASE.md) states the observed counts, environment, initial
failures, portability changes and browser checks.

## Read the evidence correctly

On the corrected full-scene development benchmark, the protected height model has
**13.24 m measured-building RMSE** and **8.36 m vegetation-only RMSE**. These describe
different masked populations; neither is a universal error bound. The bundled
showcase cases have their own narrower results.

The active V3 classifier's best-development epoch has **69.42% equal-region OEM
macro F1** and **83.20% GAMUS macro F1**. It failed some predeclared retention gates.
The older V1 classifier's **61.63% Christchurch macro F1** is a separate external
transfer experiment, not a result for V3. [Detailed results and interpretation](docs/RESULTS.md).

## Repository map

| Path | Purpose |
| --- | --- |
| `src/msr/` | Data, models, training, inference, calibration, evaluation and API |
| `viewer/` | Interactive application, sample products and frontend tests |
| `scripts/`, `configs/` | Acquisition, preparation, experiments and evaluation recipes |
| `tests/` | Numerical, model, raster, protocol and API checks |
| `docs/` | Current handbook, evidence, diagrams and dated development records |
| `model-artifacts.json` | Download locations, sizes and hashes for inference weights |

The initial public history contains the prepared source package. Local datasets,
training checkpoints, private development logs and earlier repository history are
not included. Historical protocols retain their scientific limitations; original
sealed experiments are not silently rebound to renamed public files.

No blanket open-source licence has been selected for the original application
code. Public availability does not override the separate terms of models, source
imagery and dependencies. See [rights and attribution](THIRD_PARTY_NOTICES.md).
