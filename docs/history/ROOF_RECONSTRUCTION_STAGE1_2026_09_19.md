# Roof reconstruction: stage 1, 19 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Current viewer decision — flat roofs restored

At the user's request, roof relief and experimental fitting are no longer
connected to the viewer. Buildings retain flat, optical-textured roofs and their
existing model-derived building-height estimates. Fly metadata, the Inspect
popover and selected-building analysis show **“Roof height estimation yet to be
configured”**. Raw predictions, exports, checkpoints and facade textures are
unchanged. The experimental modules, tests and prepared data remain saved but
inactive; no further roof training or downloads were started. The stage-1 notes
below describe the earlier experiment, not currently enabled viewer features.

## Status — foundational implementation, NOT a completed roof-accuracy fix

The viewer can now represent a sharp two-plane gable with lower eaves and vertical
walls instead of placing rounded relief on top of a retained flat slab. This is
an **experimental geometric fitter**, not a newly trained roof detector. It does
not yet separate attached buildings, predict arbitrary roof parts, reconstruct
hipped/complex roofs, or reliably identify rooftop rooms and dormers.

The exact user example, Strasbourg `grid_00074_z19.jpg`, still has **0 accepted
fits out of 12 existing connected footprints**: eight are edge-clipped and four
are complex/merged/undersized under the current gate. The original view remains.
This is a rejection/coverage result, not a height-accuracy score. The source
image was already inspected and is a development example, not a fresh test.

## Implemented

- `viewer/app/roof-reconstruction.ts`: bounded orientation search, robust
  two-plane fitting, spatially distributed internal fit-check samples, comparison
  against constant/tilted-plane alternatives, optical ridge support and rejection
  for insufficient/missing/ambiguous evidence. These heuristics are not calibrated
  probabilities of roof correctness. Agreement with model heights is not agreement
  with independently surveyed heights.
- Sharp mesh joins, image-projected roof UVs, separate illustrative wall UVs and
  full-height walls. Accepted replacements hide the old solid/relief and lower
  only the underlying presentation mesh under that footprint. Switching off
  restores the original geometry. No export raster is changed.
- Viewer layer **Fitted roof shapes · experimental**, off by default; disabled
  when no candidate passes. The collapsed report gives rejection reasons and a
  downloadable JSON audit. Fitted roof inspection distinguishes raw source height
  from fitted display height. Collision retains a conservative maximum envelope.
- `src/msr/data/roof3d.py`: strict decoding of the release's list-wrapped
  uncompressed COCO RLE, independent image/classification/boundary/height-validity
  masks, overlap exclusion, and cropped-border safeguards. Unknown background is
  not a negative class; missing height is never a zero-height target.
- Versioned data preparation and acquisition scripts; all new datasets on **D:**.

## Data acquired and audited

### Roof3D v2 training-source pilot

Path: `D:/MSRData/data/roof3d_training_pilot_v1`.

Official record: <https://zenodo.org/records/8300629>.
Authors' dataset repository: <https://github.com/dlrPHS/GPUB>.
Licence in the publisher's API: **CC-BY-4.0**; attribution is saved in the manifest.

- Published archive: 4,724,970,465 bytes. **Not downloaded in full.**
- Read its inventory and selectively fetched two training annotation files and
  16 RGB/DSM pairs using strict HTTP byte ranges: 44,252,029 bytes transferred.
- Selected ZIP members checked by CRC32 and length; local SHA256 recorded.
  The full-archive MD5 cannot be claimed verified with a partial download.
- The archive lists 3,337 training RGB images. Only 16 evenly spaced filenames
  were selected for this **schema/preparation pilot**, not a spatially independent
  evaluation set. No v2 validation pixels were accessed by this staging script.
- Selected examples contain 1,705 roof-plane annotations and 850 building-section
  annotations. These are patch annotation counts, not unique physical roof counts.
- Plane supervision: 1,133,803 valid labelled pixels, 123,159 valid boundary pixels;
  164,394 overlapping pixels excluded. Section supervision: 1,279,733 valid pixels,
  95,921 boundary pixels; 5,990 overlapping pixels excluded.
- RGB and DSM grids match within all 16 selected pairs, but the chips lack useful
  georeferencing/source-location tags. Numeric names alone do not identify city,
  original tile, or real versus synthetic source. Recover this provenance before
  constructing geography-level validation claims.
- Both JSON files use the generic category name `individual_building`; their
  **file role** (`annotation_plane` vs `annotation_sec`) distinguishes supervision.
  Segmentation is a list of uncompressed RLE dictionaries, not ordinary polygons.

Prepared targets:
`D:/MSRData/data/roof3d_training_pilot_v1/prepared_positive_only_v1`.
All `height_valid` masks are deliberately false pending a verified height contract.

### Roof3D v3 small reference tile

Path: `D:/MSRData/data/roof3d_reference_pilot_v1`.
Official record: <https://zenodo.org/records/10910492>.

59,299,694 bytes, all four released file MD5 values verified. This record contains
the publisher's test area; it is marked as a **development reference diagnostic**,
not training data or our untouched final holdout.

The grid audit did NOT pass exact agreement: GT has no CRS and has a slightly
different transform; RGB and DSM/DTM use differently described CRS metadata.
DTM is constant 45 in this tile, and GT values are not on the same numerical
range as the absolute DSM. Do not guess the GT's datum or subtract DTM twice.
Verify the product definition, alignment and vertical reference before metrics.

## Verification

- 35 viewer tests passed (14 new reconstruction tests plus 21 existing navigation,
  inspection, relief and facade checks).
- 21 Python tests passed (10 new Roof3D label tests and 11 range/staging regressions).
- TypeScript and targeted ESLint passed.
- Fresh isolated-browser upload of the exact Strasbourg image: normal rendering,
  expected rejection report, no page errors.
- A clearly named **synthetic** gable fixture tested the full UI replacement toggle,
  removal/restoration of the underlying slab, and Fly click inspection. It is not
  a real satellite prediction or evaluation result. Its mock API response was
  confined to the isolated test browser and removed by reload.
- Screenshots: `outputs/runtime/roof-fit-synthetic-on.png`,
  `roof-fit-synthetic-off.png`, `roof-fit-synthetic-inspection.png`, and
  `roof-fit-strasbourg-fallback.png`. Only the last is the actual user scene.

The running development browser briefly accumulated errors during an intermediate
hot reload when the builder's return signature changed. After completing both sides
of that change, a fresh session passed without page errors. Refresh an old app tab
if it retained an interrupted development render.

## Next implementation stages / release gates

1. Audit plane overlaps and annotation/image alignment visually; recover source
   geography/type and freeze a non-overlapping split. Preserve the official split
   as published evidence, not a claim that the whole existing system never saw
   those cities.
2. Train an **isolated RGB roof-part/ridge model**. Preserve production height and
   semantic checkpoints and preprocessing. Use RGB only at inference; reference
   DSM/roof labels are supervision/evaluation, never hidden inference inputs.
3. Decompose merged footprints, preserve courtyard holes, and predict roof parts.
   Extend geometry to supported hip/flat/complex roofs and vertical rooftop steps.
   A predicted ridge alone does not establish its metric height.
4. Resolve roof-level height supervision before learning local roof residuals.
   Distinguish ridge/eave height and roof-plane error from whole-building RMSE.
5. Report roof-part/boundary precision, recall and IoU; ridge-position error;
   roof-only height errors; accepted coverage and failures, separately by roof type,
   material, size and geography. Include confidently wrong fits, not only accepted
   attractive examples. Check building, forest and terrain regressions.
6. Promote only after independent reference evaluation and app regression tests.
   **No roof-model training has started and no production model was changed here.**

## Research direction

Building decomposition plus ridge detection is an established direction, but no
paper's results are claimed for our geometric baseline:

- Zhang & Aliaga, *Procedural Roof Generation From a Single Satellite Image*, 2022:
  <https://doi.org/10.1111/cgf.14472>.
- Schuegraf et al., *Roof3D*, 2023:
  <https://doi.org/10.5194/isprs-annals-X-1-W1-2023-971-2023>.

## Reproduction

```powershell
.venv/Scripts/python.exe scripts/data/prepare_roof3d_reference_pilot.py
.venv/Scripts/python.exe scripts/data/stage_roof3d_training_pilot.py
.venv/Scripts/python.exe scripts/data/prepare_roof3d_training_masks.py
.venv/Scripts/python.exe -m pytest tests/test_roof3d.py tests/test_openearthmap_archive_ranges.py tests/test_stage_christchurch_holdout.py -q
```

From `viewer`:

```powershell
node --experimental-strip-types --test tests/roof-reconstruction.test.mjs tests/roof-geometry.test.mjs tests/building-facades.test.mjs tests/flight-navigation.test.mjs tests/surface-query.test.mjs
npx tsc --noEmit
npx eslint app/roof-reconstruction.ts app/terrain-workspace.tsx app/surface-query.ts
```
