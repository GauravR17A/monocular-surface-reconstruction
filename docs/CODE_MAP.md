# Implementation map and maintenance guide

The Python package is `msr`; its source is under `src/msr`. The public package
uses a fresh repository history and scientific naming. This chapter describes the
implementation families so that a reviewer can trace a claim to the responsible
code and its tests without needing the private development archive.

## Data and learning

`data/` implements raster loading, normalisation and validity handling. RGB arrays
are converted to channel-first float arrays, with the selected intensity scale
and ImageNet channel normalisation. Reference validity is an independent array;
unknown height is never made valid merely because its numeric value is zero.

`models/height_net.py` defines the urban encoder/decoder; `domain_surface_net.py`
wraps the protected urban model and guarded vegetation fusion. The public loader
reconstructs the base architecture from the packaged configuration and loads all
of its tensors. Training does not occur during loading. `residual_height.py` and
the semantic/class-assisted experiment modules are research alternatives, not
the released default predictor.

The scripts directory contains preparation, training, fixed-protocol evaluation,
error analysis, checkpoint sealing and demonstration tooling. Configurations
retain experiment names and protocol choices. A config is not proof that a run
finished; consult the matching outcome record and checkpoint identity.

## Inference and calibration

`inference/predict.py` owns checkpoint construction, tiled metric prediction,
valid masks and optional semantic products. `inference/tiling.py` determines tile
starts and positive blending weights. A complete raster can involve many model
calls; average overlapping predictions only on supported pixels.

Foundation relative depth and learned height are distinct products. Calibration
code operates on explicitly supplied DEM or control-point evidence. Presentation
meshes and building solids are derived for viewing; they do not become the raw
prediction simply because their appearance is smoother.

## API boundary

`api/app.py` owns request parsing, runtime loading, serial model access, job
directories and response manifests. The endpoint writes actual output rasters
and returns URLs under `/results/`. `scripts/serve_api.py` configures Uvicorn,
device, checkpoint and output directory. The documented default binds locally.

`scripts/download_models.py` validates release sizes and SHA-256 before replacing
files. The foundation repository revision and required files are pinned in the
same manifest. Acquire artifacts before starting a production workflow; a health
response alone does not prove a model has successfully loaded.

## Viewer boundary

`viewer/app/terrain-workspace.tsx` coordinates scene state, loading, Three.js,
navigation, controls and output links. Smaller modules isolate scientific and
interaction contracts: `surface-query.ts`, `terrain-analysis.ts`,
`terrain-presentation.ts`, `six-class-layer.ts`, `six-class-request.ts`,
`image-import.ts`, `flight-navigation.ts`, `walk-navigation.ts`,
`fullscreen-navigation.ts`, `metric-grid.ts`, `selection-region.ts` and
`explore-map-math.ts`. Their Node tests execute deterministic geometry and state
logic without requiring a GPU or browser.

`landscape-evaluation.tsx` reports selected demonstration measurements.
`validation-comparison.tsx` visualises aligned prediction/reference/error rasters.
The restricted US3D imagery is excluded, and its historical row cannot request
an unavailable image comparison. The other public demonstrations remain usable.

## Safe extension points

For a new predictor, declare its input range, channel order, output units,
normalisation, valid-mask behavior and checkpoint identity before integration.
Evaluate it under the existing matched protocol; do not change the baseline while
measuring a candidate. For a new terrain workflow, establish horizontal CRS and
vertical datum compatibility and retain the original surface as a separate export.

For a new viewer effect, operate on a presentation copy. Add numerical tests only
where the change can affect units, coordinates, classification, geometry or state
transitions. Use browser checks for visible interaction and an API check for actual
generated products. Preserve a reproducible failing example before adjusting a
scientific rule.
