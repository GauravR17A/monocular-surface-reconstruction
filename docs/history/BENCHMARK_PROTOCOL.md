# Monocular Surface Reconstruction locked benchmark protocol

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Why this evaluator exists

`scripts/analyze_multidomain_checkpoint.py` is useful during training, but it
loads a centred 384 x 384 crop. The urban controls are 1024 x 1024, so that
path scores only 14.06% of each urban tile. It is not equivalent to an upload
through the application.

`scripts/evaluate_locked_benchmark.py` instead uses the application read limits,
recomputes the Depth Anything V2 prior, and runs the production height predictor
over the complete raster with 512 px tiles and 128 px overlap. References are
loaded only after the input grid is established and are never passed to either
model.

The report includes:

- RMSE, MAE, bias, correlation, and R2 for the raw upload;
- landscape and ground/building/vegetation breakdowns where labels permit them;
- the same metrics for deterministic stretch, JPEG, gamma, exposure, and
  channel-gain variants;
- prediction-versus-raw deltas, which expose sensor/radiometry shortcuts;
- manifest, dataset, checkpoint, and foundation-model hashes;
- train/validation overlap checks and a before/after checkpoint hash;
- source-licensing caveats.

## Two-pass locked workflow

First produce a read-only integrity report. This obtains the exact hashes without
running either neural network:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_locked_benchmark.py `
  --checkpoint experiments\20260829T203146Z_multidomain_surface_v2_guarded_vegetation\checkpoint_best_guarded_vegetation.pt `
  --manifest data\multidomain_v2\manifests_with_priors\test.csv `
  --deny-manifest data\multidomain_v2\manifests_with_priors\train.csv `
  --deny-manifest data\multidomain_v2\manifests_with_priors\validation.csv `
  --benchmark-id msr-legacy-regression-v1 `
  --benchmark-status development `
  --integrity-only `
  --output outputs\benchmarks\msr-legacy-regression-v1\integrity.json
```

Copy the four hashes from that report into the release invocation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_locked_benchmark.py `
  --checkpoint <CHECKPOINT> `
  --manifest <FROZEN_MANIFEST> `
  --deny-manifest <TRAIN_MANIFEST> `
  --deny-manifest <VALIDATION_MANIFEST> `
  --benchmark-id <RELEASE_ID> `
  --benchmark-status locked_release `
  --expected-manifest-sha256 <HASH> `
  --expected-dataset-sha256 <HASH> `
  --expected-checkpoint-sha256 <HASH> `
  --expected-relative-model-sha256 <HASH> `
  --device cuda `
  --output outputs\benchmarks\<RELEASE_ID>\report.json
```

Never tune on a `locked_release` report. Candidate selection and threshold
changes belong on validation/development data; run a new named locked release
only after those choices are frozen.

## Current evidence status

The existing combined test manifest has 200 HighBuild-derived urban controls and
120 Open-Canopy forest controls. It is geographically disjoint from the current
train and validation manifests and all 3,300 RGB/reference files have no exact
cross-split byte duplicates. It has nevertheless been evaluated repeatedly, so
it is a **legacy regression suite**, not a fresh blind result. It also contains
no true plains, hilly, or mixed-labelled scene.

A future manifest should use project-relative paths and contain
`sample_id, region, landscape, rgb_path, surface_path, target_kind, dtm_path,
building_mask_path, vegetation_mask_path, valid_mask_path, gsd_m`. A mixed scene
requires both building and vegetation masks. Hilly absolute-DSM results need a
calibration datum and a genuinely independent evaluation reference; coarse
terrain R2 must not be presented as building/tree accuracy.

### Corrected HighBuild protocols

HighBuild's `building_height_m.tif` is a sparse building-label raster, not a
complete surface nDSM. A valid benchmark row must therefore declare
`target_kind=building_height` and provide two distinct aligned masks:

- `building_mask_path` contains every official COCO building footprint and is
  semantic supervision only;
- `valid_mask_path` states which footprint pixels have height supervision under
  the named evaluation protocol.

The supported protocols are `highbuild_all_annotated_positive_v1` (measured and
estimated positive annotated heights) and
`highbuild_measured_coco_intersection_v1` (positive, non-estimated annotations
only). Outside the footprint is unknown, never zero-height ground. Reports from
these protocols must remain separate and cannot be compared directly with the
historical zero-filled HighBuild metrics. Manifest loading and the locked
benchmark now reject the ambiguous historical target contract.

## Licence and provenance guardrails

- **Open-Canopy:** Etalab Open Licence 2.0. Keep attribution, retrieval date,
  upstream revision, and per-file hashes. Existing `resolve/main` URLs are not a
  revision lock. Its 120-chip test set is already inspected. The 5 km regions
  are disjoint, but many chips across splits originate from the same larger SPOT
  acquisition, so it is not an acquisition-scene-disjoint sensor test.
- **HighBuild-derived controls:** the local city split is conservative and the
  imagery inventory is source-specific, but the separate provenance/licence
  chain for height geometry and labels is not frozen locally. Use internally;
  do not redistribute chips or call every annotation measured ground truth yet.
  The upstream metadata marks some annotations as estimated, so final reports
  should stratify measured/non-estimated and estimated subsets.
- **DFC19 / US3D:** keep the current local copy internal. The secondary
  CC-attribution note conflicts with restrictive official DFC19 redistribution
  terms. Do not publish or package these assets until written permission and
  chain-of-title are confirmed.
- **Sentinel/Copernicus/SRTM hilly evidence:** usable with the required source
  notices. Copernicus is the calibration datum and SRTM is a coarse cross-check,
  not LiDAR or object-height truth.

This document records an engineering audit, not legal advice. Benchmark code
does not grant redistribution rights to its inputs.

## Pre-registered unused HighBuild candidates

The following tiles are absent from the current 320-row combined test manifest.
They were selected without model predictions by SHA256 ordering of
`msr-lock-v1|<sample_id>`. Source rows and members are in
`data/highbuild_full/msr_splits/test.csv` and archives under
`data/highbuild_full/msr_splits/shards/test/` (physically on D: through
the junction). Extract and hash the frozen list before inference.

Dense urban controls (`num_annotations >= 10`, five per held-out city):

- Amsterdam: `grid_11377_z19`, `grid_18227_z19`, `grid_01605_z19`,
  `grid_01225_z19`, `grid_07725_z19`
- Frankfurt: `grid_00007_z19`, `grid_00040_z19`, `grid_00051_z19`,
  `grid_00047_z19`, `grid_00030_z19`
- Marseille: `grid_00091_z19`, `grid_00169_z19`, `grid_00066_z19`,
  `grid_00032_z19`, `grid_00150_z19`
- Odense: `grid_00026_z18`, `grid_00086_z18`, `grid_00023_z18`,
  `grid_00028_z18`, `grid_00048_z18`
- Vancouver: `grid_00092_z18`, `grid_00131_z18`, `grid_00140_z18`,
  `grid_00132_z18`, `grid_00081_z18`

Sparse-building controls (`num_annotations <= 3`; this is sparse urban, not a
true plains benchmark):

- Amsterdam: `grid_00326_z19`, `grid_00695_z19`, `grid_00305_z19`,
  `grid_01318_z19`, `grid_15358_z19`, `grid_01156_z19`, `grid_08700_z19`,
  `grid_14699_z19`
- Marseille: `grid_00021_z19`, `grid_00023_z19`, `grid_00155_z19`,
  `grid_00035_z19`
- Vancouver: `grid_00146_z18`, `grid_00048_z18`, `grid_00101_z18`,
  `grid_00152_z18`, `grid_00137_z18`, `grid_00126_z18`, `grid_00149_z18`,
  `grid_00150_z18`

Use the full `webdataset_key` from the source CSV when materializing the final
manifest; the short grid names above are grouped by city for readability.

No equivalent unused local set currently exists for forest, hilly object
height, plains, or mixed building-plus-vegetation scenes. The next honest lock
therefore requires new Open-Canopy test regions, independent references for the
four unvalidated hilly candidates, a true plains set, and an openly
redistributable mixed RGB+nDSM+semantic source. The existing 15 DFC19 windows
cannot fill the mixed slot: all were ranked after evaluation, they overlap along
one Omaha strip, and their redistribution rights are unresolved.
