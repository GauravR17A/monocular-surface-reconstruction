# Monocular Surface Reconstruction progress and recovery record

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Last updated: 2026-09-13 (Asia/Calcutta)

This file is the restart point if the active Codex session reaches its usage
limit. Source code is also checkpointed in Git; large datasets, model weights,
and generated outputs are deliberately ignored by Git but remain on the local
disk under the paths documented below.

## Verified environment

- Windows, Python 3.10.11
- NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB VRAM, 140 W
- NVIDIA driver 596.36
- Project environment: `.venv`
- PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128
- CUDA is available and a 2048x2048 matrix-multiplication smoke test passed
- Core test suite: 56 tests passed

Re-run verification:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Implemented

- Explicit CSV sample manifests with mandatory geographic region identifiers
- Geographic train/validation/test leakage rejection
- GeoTIFF RGB/height shape, transform, CRS, nodata, and range audit
- Raster-window PyTorch dataset with no height resampling
- Synchronized rotations/flips and validity masking
- MAE, RMSE, bias, correlation, R-squared, median, P90, and P95 errors
- Exact streaming primary metrics for large validation sets
- Seam-resistant overlapping tile layout and Hann blending
- Clean `HeightNet` with a timm multiscale encoder, metric height head, and
  auxiliary building head
- Masked Huber, gradient/edge, and building losses
- CUDA AMP training, accumulation, early stopping, JSONL history, and best/latest
  checkpoints
- Public `predict_height` result contract and GeoTIFF prediction CLI
- Isolated adapter for research-only FusedSeg-HE benchmarking

## Real smoke data

A public reviewer sample from `feifei140729/small-sample` is stored at:

`data/smoke/highbuild_small/NorthAmerica_USA_Chicago/`

Sample `grid_00034_z18` contains:

- RGB: 1024x1024, uint8, three bands
- height raster: 1024x1024, float32, range 0-230 m
- 16 building polygons with per-building heights
- no CRS/geotransform or documented GSD

The pair is pixel-aligned and useful for inference plumbing/domain-shift smoke
tests. It is not sufficient for final scientific claims because it lacks
geospatial metadata and its full-dataset license/provenance still needs review.

Audit output: `outputs/smoke_pair_audit.json`

## Multi-city reviewer training

The public HighBuild reviewer subset is stored at `data/highbuild_reviewer/`:

- 260 RGB/height pairs from 26 cities
- train: 180 samples / 18 cities
- validation: 40 samples / 4 cities
- test: 40 samples / 4 cities
- all 260 pairs passed the strict alignment/band/validity audit
- raw labels span 0-584 m

This subset declares `license: other` and is for engineering/reviewer inspection,
not final production claims. The two-epoch pilot completed successfully. Validation
RMSE improved from 7.91 m to 7.80 m and building-only RMSE from 12.95 m to 12.77 m.
The pilot proved BF16 training, metrics, and checkpoint persistence; a resume test
also passed after correcting CPU/CUDA RNG-state restoration.

The active preliminary experiment is:

`experiments/20260826T135412Z_highbuild_reviewer_train/`

At epoch 2 it reached 7.15 m all-pixel RMSE and 11.61 m building-pixel RMSE.
Check `metrics.jsonl` for newer completed epochs. Resume safely with:

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --config experiments\20260826T135412Z_highbuild_reviewer_train\config.yaml `
  --resume experiments\20260826T135412Z_highbuild_reviewer_train\checkpoint_latest.pt
```

## Full HighBuild preparation

HighBuild-1M metadata is at `data/highbuild_full/`; the resumable shard download
was started with `scripts/data/download_highbuild_full.py --include-shards`.
The hosted release contains 61,215 benchmark tiles in about 35 GB of TAR shards.

The upstream random split is not acceptable for Monocular Surface Reconstruction because cities leak
across train/validation/test. `scripts/data/prepare_highbuild_full.py` creates a
city-disjoint, conservative license-filtered split containing:

- train: 25,446 tiles / 5 cities
- validation: 1,921 tiles / 4 held-out cities
- test: 2,310 tiles / 5 held-out cities

USA, Cape Town, and conditional-license city sources are excluded from the
production split by default. `scripts/data/repack_highbuild_full.py` atomically
repackages only selected samples and is safe to restart per source shard.

## Full HighBuild training and error analysis

The full ConvNeXtV2-tiny run completed normally through epoch 33 and then
early-stopped after eight epochs without a new RMSE best. The experiment is:

`experiments/20260826T192229Z_highbuild_full_convnext_tiny/`

Protected checkpoints:

- epoch 25 best RMSE: 4.366 m RMSE, 1.979 m MAE, +1.240 m bias,
  0.593 correlation, and 0.093 R-squared
- epoch 32 best MAE: 1.951 m MAE and 4.409 m RMSE

Validation-only error analysis for the epoch-25 checkpoint is saved under:

`outputs/error_analysis/epoch25_best_rmse/`

The main findings are:

- 430.6M of 503.6M validation pixels are below 2 m in the building-height
  reference; these pixels have an essentially zero target but receive a mean
  1.729 m prediction, producing 4.072 m RMSE.
- Building pixels average 11.049 m in the reference and 9.403 m in prediction,
  with 5.808 m RMSE and -1.646 m bias.
- Tall structures are strongly compressed: the 20-40 m bin has -10.71 m bias,
  and higher bins degrade sharply.
- Validation is dominated by Copenhagen (1,618/1,921 tiles). Copenhagen reaches
  3.611 m RMSE and 0.292 R-squared, while Lyon, Munich, and Strasbourg have
  5.668-8.574 m RMSE and negative R-squared.
- The auxiliary building head peaks around 0.477 IoU / 0.646 F1 at threshold
  0.3. Hard gating barely changes overall RMSE; soft gating lowers overall RMSE
  to 3.855 m but damages building RMSE to 7.610 m, so it must not be adopted as
  the final post-processing rule.
- Two-fold cross-fitted global affine calibration reaches a diagnostic 3.699 m
  RMSE and 0.349 R-squared, but it is dominated by background pixels. Building-
  only cross-fitted calibration reaches 5.442 m RMSE and 0.273 R-squared.

The next urban refinement should use city- and height-balanced sampling,
explicit tall-building supervision, stronger segmentation/background control,
and checkpoint selection for building and cross-city metrics. The untouched
test split remains unused.

## Targeted refinement and end-to-end raster foundation

The targeted refinement experiment is running at:

`experiments/20260828T044931Z_highbuild_targeted_refinement/`

It starts from the protected epoch-25 full-data checkpoint and uses weighted
city streams, foreground/tall-guided crops, explicit foreground MSE and tall
Huber terms, and separate checkpoint selection for overall, building-only, and
cross-city macro RMSE. The source checkpoint remains untouched, so a failed
refinement cannot erase the strongest honest baseline.

The missing raster/calibration foundation has also been added:

- licensed Apache-2.0 Depth Anything V2 Small relative-depth extraction;
- explicit dimensionless rDSM output for non-georeferenced imagery;
- CRS/transform-preserving, atomic float GeoTIFF output;
- DEM reprojection onto the exact RGB grid;
- robust affine GCP calibration with outlier trimming;
- absolute DSM composition as ground terrain plus non-negative object height;
- a unified `scripts/run_pipeline.py` entry point and machine-readable metadata.

The pretrained weights are cached locally under
`models/foundation/depth-anything-v2-small-hf/` and intentionally excluded from
Git. A real non-georeferenced Chicago sample completed the relative-depth/rDSM
path successfully under `outputs/pipeline_smoke/`.

## Interactive large-raster safety

The local API now enforces a 3072 x 3072 interactive processing budget based on
pixel dimensions rather than compressed file size. Rasterio reads an overview
directly, preserves the original CRS and full geospatial bounds by updating the
affine transform, and records original/processing grids in response metadata.
Pixel-coordinate GCPs are scaled onto that overview grid; map-coordinate GCPs
continue to use the preserved transform.

The exact 34,713 x 36,360, 202.8 MB GeoTIFF that previously expanded toward
36 GB of process memory completed end-to-end in 23.96 seconds. The bounded path
peaked around 2.0 GB RAM, returned HTTP 200, preserved EPSG:31985 and identical
map bounds, and emitted a clear overview-resolution confidence warning.

The known-good local app can be restored with:

```powershell
.\scripts\start_prototype.ps1 -OpenBrowser
```

## Pretrained research benchmark

The FusedSeg-HE checkpoint and MiT encoder weights were downloaded to:

`checkpoints/benchmarks/fusedhe/`

The upstream implementation was inspected separately at:

`C:/Users/researcher/AppData/Local/Temp/msr-research/FusedHE`

Its license is research/evaluation-only and explicitly non-commercial. None of
its source code is copied into the clean Monocular Surface Reconstruction package; it is benchmarked
only through `scripts/benchmarks/run_fusedhe.py`.

The first fp16 inference completed but produced non-finite pixels over roughly
44% of the image. Its metrics are invalid and must not be reported as a model
score. The adapter now rejects any non-finite tile and defaults to fp32.

The fp32 rerun produced finite values at all 1,048,576 pixels. It is a valid
inference run but a failed cross-domain baseline on this sample:

- prediction range: 0-39.44 m; target range: 0-230 m
- all-valid pixels: RMSE 93.89 m, MAE 63.57 m, bias -59.29 m
- positive-height pixels: RMSE 123.01 m, MAE 105.79 m, bias -105.78 m

Metric records are in
`outputs/benchmarks/fusedhe/chicago_grid_00034_z18_fp32_*.json`. Do not present
these numbers as representative checkpoint quality: the checkpoint was trained
for a different sensor/domain and its scale plainly does not transfer to this
tall-building reviewer sample. Preserve it as evidence that preprocessing and
domain compatibility must be verified before model selection.

## Dataset access constraint

The official DFC2023 Track 2 archive is 3.88 GB and requires an IEEE DataPort
login plus approval as a competition participant. It cannot be downloaded from
an anonymous session. Do not silently substitute an unofficial mirror for the
final benchmark. Continue public-data smoke testing while arranging legitimate
access or selecting a clearly licensed alternative.

## Multi-domain model milestone (2026-08-28)

Implemented and verified the next-stage surface model without overwriting the
protected urban checkpoint:

- explicit aligned `dsm`/`ndsm`/`building_height` manifests with ground,
  building, and vegetation supervision;
- a shared gated adapter whose per-pixel expert weights are mutually exclusive;
- a supervised refinement gate so canopy corrections open over vegetation but
  remain conservative over protected urban surfaces;
- a frozen-base first training stage and independent guards for all-urban and
  building-only RMSE;
- landscape-, semantic-domain-, and region-level validation plus signed-error
  rasters;
- a resumable Open-Canopy range-streamer with official split preservation,
  1.5 m output, decimetre-to-metre conversion, and LiDAR-class filtering;
- a city-balanced extractor for the existing conservative HighBuild shards;
- one-time CUDA precomputation of the Depth Anything V2 relative prior;
- a durable preparation/training driver at
  `scripts/run_multidomain_pilot.ps1`.

The corrected licensed-data pilot used 168 training chips (120 urban and 48
forest), 60 validation chips, and 72 untouched test chips. Forest/urban sampling
and semantic loss are class-balanced; repeated crops and augmentation do not get
counted as new geographic scenes.

Best eligible held-out validation checkpoint: epoch 9 in
`experiments/20260828T175822Z_multidomain_surface_pilot`.

- overall RMSE: 5.67 m;
- landscape-macro RMSE: 6.25 m;
- urban RMSE: 5.21 m, 0.25 m better than the protected urban baseline on the
  identical pilot validation subset;
- forest RMSE: 7.28 m, improved from the protected baseline's 11.87 m;
- building-only RMSE: 4.34 m, a 0.17 m regression and therefore inside the
  configured 0.25 m protection guard;
- vegetation-only final RMSE: 10.11 m; canopy-expert RMSE: 8.88 m;
- vegetation routing precision/recall/F1: 52.3% / 94.6% / 67.3%;
- mean fusion strength: 10.3% ground, 18.5% building, 63.5% vegetation.

Epochs 10-13 improved forest RMSE as far as 6.83 m, but exceeded the building
regression guard and were correctly rejected. Training early-stopped at epoch
13. The protected urban checkpoint, epoch-9 best checkpoint, and epoch-13 latest
diagnostic checkpoint all remain saved. These are pilot results, not a claim of
production or industry-wide accuracy; broader geographic training data remains
necessary.

Earlier smoke results that proved the plumbing:

- three Open-Canopy chips streamed from the official release and passed raster
  alignment/unit checks;
- 14 urban samples extracted across 5 train, 4 validation, and 5 test cities;
- combined geographic leakage validation passed;
- a complete BF16 CUDA training/validation/checkpoint epoch ran on the RTX 4070;
- the initial safety blend held sampled urban regression to -0.039 m and
  building-only regression to -0.017 m relative to the protected base;
- 57 automated tests pass.

The inference stack now loads both legacy `HeightNet` and
`domain_gated_surface_v1` checkpoints. It passes the cached Depth Anything prior
into the adapter and exports building probability, vegetation probability, and
relative prediction confidence alongside nDSM/rDSM. A real CUDA/API filesystem
smoke generated all products from the new checkpoint format. The web viewer
build passes and exposes the new diagnostic rasters for download.

## research showcase integration milestone (2026-08-29)

The viewer/API now cover the remaining high-value prototype interactions while
the v2 multidomain model trains independently:

- optional reference nDSM or absolute DSM upload with grid alignment, honest
  RMSE/MAE/bias/correlation/R2 reporting, and a signed-error GeoTIFF;
- point-height plus metric local-slope inspection and scene mean slope;
- automatic public bare-earth terrain for georeferenced imagery, with source
  URLs and attribution recorded alongside the derived absolute DSM;
- manual DEM and GCP calibration retained as the accuracy-first alternatives;
- guarded showcase-checkpoint selection so only an explicitly reviewed held-out
  checkpoint replaces the protected pilot;
- PID-scoped start/stop scripts that do not terminate unrelated Python, browser,
  GPU, or training processes.

The automatic terrain path was exercised on a real Open-Canopy GeoTIFF and
successfully mosaicked and reprojected two public terrain tiles onto the exact
RGB grid. This supplies an absolute datum, not high-resolution object truth;
building/tree height quality still comes from the trained surface model.

The v2 quality run completed all 24 configured epochs. Epoch 20 was the best
guard-eligible validation checkpoint and was then evaluated once on the
untouched 320-record test manifest:

- overall final-surface RMSE 6.17 m, MAE 3.38 m, correlation 0.619, R2 0.361;
- forest RMSE 6.96 m versus 11.49 m for the protected urban-only base;
- urban RMSE 5.65 m versus 5.63 m for the base (0.02 m regression);
- vegetation final-surface RMSE 8.79 m versus 16.40 m for the base;
- dedicated canopy expert RMSE 7.04 m, correlation 0.488, R2 0.229;
- building-domain RMSE 10.64 m versus 10.68 m for the base on this harder
  geographically held-out test set;
- both urban and building regression guards passed.

The test audit is saved at
`outputs/analysis/20260829T130607Z_multidomain_surface_v2-test.json`. A real
CUDA pipeline smoke also generated rDSM, nDSM, building/vegetation rasters, and
metadata under `outputs/showcase/v2_checkpoint_smoke`. The reviewed epoch-20
checkpoint is now the explicit showcase selection; `checkpoint_latest.pt` was
not promoted.

## GAMUS specialist and guarded routing audit (2026-09-13)

The GAMUS Stage-1 specialist materially improved semantic identification on the
authenticated GAMUS validation split, but it was not safe to replace the
protected multi-domain model globally. A Stage-2 mixed fine-tune failed its
regression guards, so neither checkpoint was promoted directly.

Stage 3 therefore trained a small scene router over frozen shared-model feature
descriptors. Its purpose is to select the GAMUS specialist only for scenes that
look appropriate and otherwise fall back to the protected model. The final
record package contains 7,184 authenticated train/calibration records; test
splits were excluded while the router and threshold were chosen. Nine planned
GAMUS crops were audited as empty and excluded. Windows-safe manifest replacement
retries and full-precision route decisions were added, with fail-closed artifact
authentication for the new Stage-3b format.

The selected H128 router passed its calibration requirements at threshold
0.707040: 99.06% source precision, 98.02% GAMUS coverage (842/859), 8/280 legacy
selections, and positive routed utility gain. All 230 automated project tests
passed before final evaluation.

Dense validation also passed both predeclared hard-guard suites:

- GAMUS identification accuracy: 57.40% protected -> 73.36% routed;
- GAMUS macro F1: 0.6002 -> 0.7495;
- GAMUS vegetation F1: 0.3690 -> 0.7379;
- GAMUS height RMSE: 7.0469 m -> 7.0283 m;
- legacy height RMSE: 6.5768 m -> 6.5767 m;
- legacy macro F1: 0.6326 -> 0.6323 (inside the validation guard);
- the router selected 841/859 GAMUS and 8/280 legacy validation scenes.

After all choices were frozen, the official held-out test was evaluated once.
It confirmed strong GAMUS gains but exposed a geographic/style generalisation
failure on unseen legacy cities:

- GAMUS test identification accuracy: 55.86% -> 65.45%;
- GAMUS test macro F1: 0.5625 -> 0.6767;
- GAMUS test vegetation F1: 0.2632 -> 0.5591;
- GAMUS test height RMSE: 7.9167 m -> 7.9045 m;
- legacy test height RMSE: 6.1455 m -> 6.1418 m;
- legacy test identification accuracy: 68.50% -> 64.08%;
- legacy test macro F1: 0.6140 -> 0.5837;
- the router selected 102/200 unseen legacy urban scenes and 0/120 legacy forest
  scenes, despite selecting only 8/160 legacy urban validation scenes.

This router is therefore **not approved for application promotion**. The result
suggests it learned dataset/city appearance cues rather than a sufficiently
general rule for when the specialist is beneficial. The live application
pointer remains on the protected epoch-20 checkpoint, whose SHA-256 is
`e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144`.
The immutable final audit is
`outputs/evaluation/stage3b_routed_final_h128.json`; its SHA-256 is
`3f251627fadd6a5a1294ec80e84650073127dda978d0baaa62a0ca47ac79dc7e`.

The leakage-safe next iteration must use group-held-out cities within the
development data, train a content/utility-aware gate rather than a source-label
shortcut, and be reviewerd on a newly locked external test set. The now-exposed
official test remains a final audit record and must not be reused for threshold
or model selection, training, architecture comparison, visual case selection,
or repeat evaluation.

## Stage-4 independent routing audit (2026-09-13)

Stage 4 replaced the coupled source router with two independently guarded
endpoint-gain heads: one for metric height and one for semantic surface class.
Its authenticated record journal contains 3,347 development scenes (2,208 fit,
1,139 calibration), preserves geographic group separation, excludes both test
splits, and retains exact pixel-level SSE/confusion evidence for every frozen
endpoint. The generated artifact is valid and reproducible, but both heads were
correctly disabled; it was not promoted and the application checkpoint pointer
is unchanged.

The audit explains why this is the safe outcome:

- height is already effectively tied between endpoints: exact calibration RMSE
  is 6.9172 m protected versus 6.9008 m candidate, and only 14/1,131 scenes beat
  the required 0.25 m material-gain threshold. The predeclared minimum of 32
  high-precision selections is therefore mathematically impossible;
- semantic routing has real value on GAMUS mixed scenes (balanced error 0.3657
  to 0.2455; 775/859 materially beneficial), but the candidate regresses on
  legacy forest (0.3747 to 0.4906) and urban scenes (0.1993 to 0.2492);
- the semantic gain regressor learned the sole fit city, NYC, rather than a
  transferable rule. Its gain correlation changes from +0.711 on fit data to
  -0.269 on held-out DC/Philadelphia, and no threshold reaches the required 98%
  beneficial-selection precision while satisfying all non-regression guards.

The next isolated iteration keeps height permanently on the protected endpoint
and targets semantic routing only. It uses multiple GAMUS training cities,
group-held-out folds, conservative ensemble confidence, out-of-distribution
abstention, and a richer label-free content descriptor. Existing validation is
development evidence only. Promotion still requires a newly locked external
geographic holdout; the previously revealed official test is never reused for
tuning.

## Corrected supervision and direct-height reset (2026-09-13)

An audit found that every HighBuild row in the multidomain-v2 manifests was
tagged as a complete `ndsm`, even though the raster contains building heights
on a zero-filled, unlabelled background. This caused unsupported non-building
pixels to be learned and scored as confirmed zero-height ground. Historical
scores and checkpoints are preserved, but those urban/ground metrics are now
labelled legacy and must not be compared directly with corrected scores.

Two immutable corrected validation protocols are now recorded:

- `highbuild_all_annotated_positive_v1` scores all positive annotated
  HighBuild height support, including heights marked estimated;
- `highbuild_measured_coco_intersection_v1` scores only COCO footprint pixels
  with finite positive raster height and `is_estimated_height=false`.

The first corrected-subset pass on the protected live checkpoint reported
inclusive overall/building RMSE of 6.8203/7.6652 m and strict measured-only
overall/building RMSE of 6.8787/8.2791 m. Its paired forest/vegetation values
were 6.6615/8.2742 m. These are preserved as historical, protocol-specific
results; they are not interchangeable with the later full-native-scene audit.

The canonical corrected **full-scene** report is now
`outputs/evaluation/corrected_legacy_triplet_v1/report.json`. On the protected
model it records HighBuild strict-measured building-only RMSE **13.243 m**,
OpenCanopy vegetation-only RMSE **8.361 m**, and OpenCanopy all-valid forest
RMSE **6.707 m**. These three values use different masks and supports and must
remain separately labelled. Earlier 4.366 m overall/5.808 m building crop
scores, the preliminary corrected-subset values above, inclusive annotations,
and pooled scores remain valid history only under their own protocols; none is
a substitute for another.

Further router and head-blend work is paused. The next candidate directly
optimizes final metric height on the full QC-approved GAMUS training split while
keeping Depth Anything V2 as a fixed cached relative-depth prior. The official
training inventory contains 5,004 tiles; 5,001 pass the immutable QC contract
and three all-class-255/no-supervision tiles are quarantined. Official
validation (859 approved tiles) is development evaluation. The official test
(2,861 source tiles) was already consumed by the older Stage-3 final audit and
remains excluded from all new tuning, model selection, and repeat evaluation. A
separate dataset-QC/quarantine contract and a full-tile paired evaluator gate
the experiment. The application pointer and its checkpoint SHA-256
`e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144`
remain unchanged until a candidate wins identical-protocol validation plus a
newly locked external geographic holdout.

## Six-class GAMUS expansion (implemented 2026-09-13; refinement ongoing)

The current three-group height pilot is preserved as the comparison run. Here
“three-group” means the ground/building/vegetation **height-routing domains**;
it is not a three-class identification output. A
separate versioned candidate adds a six-class identification output for
ground, building, water, road, low vegetation, and tree while retaining the
existing ground/building/vegetation height experts and old-checkpoint loading.
Unknown/background stays ignored. Image validity, classification validity, and
height validity are now independent; water and road pixels may supervise
identification but never become fabricated zero-height targets.

The expansion also adds six-class precision/recall/IoU, targeted water-shadow
and road-boundary checks, corrected HighBuild per-building and outline
evaluation, OpenCanopy provenance preservation, unit/pixel-scale and >200 m
audits, and cached Depth Anything prior authentication. No candidate can change
the live application pointer until it beats the protected model and the
height-focused pilot on their own identical protocols, passes corrected urban
and forest regressions, and succeeds on a genuinely new external geography.
Six-class macro metrics must be recomputed across all six final classes; merging
low vegetation and trees is a separately labelled five-group diagnostic.

NYC is excluded from the new DC+Philadelphia classifier-head comparison, but
it is not system-unseen: earlier GAMUS work and the inherited HighBuild training
exposed New York data. Any NYC result must therefore be labelled
classifier-head-excluded. Detailed status and gates are in
`docs/EXPANDED_GAMUS_PLAN.md`.

OpenCanopy provenance is now indexed from the accepted manifests rather than
from every downloaded candidate folder. The audit passes for 600 train, 120
validation, and 120 locked-test samples with no accepted cross-split coarse
region overlap. It preserves imagery and LiDAR acquisition dates, EPSG:2154
location groups, source scenes/URLs, GSD, and metadata hashes in
`outputs/data_audits/open_canopy_v2_provenance_v1.json`. Rejected but cached
streaming candidates remain on disk and are explicitly excluded from counts.

## Immediate resume commands

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rasters.py `
  outputs\benchmarks\fusedhe\chicago_grid_00034_z18_fp32.tif `
  data\smoke\highbuild_small\NorthAmerica_USA_Chicago\height\grid_00034_z18.tiff `
  --scope positive-target --height-max-m 300

.\.venv\Scripts\python.exe scripts\data\download_highbuild_full.py `
  --output-root data\highbuild_full --include-shards --workers 6

.\.venv\Scripts\python.exe scripts\data\prepare_highbuild_full.py
.\.venv\Scripts\python.exe scripts\data\repack_highbuild_full.py

.\scripts\run_multidomain_pilot.ps1
```

## Usage conservation policy

Long training and download jobs should be self-logging and run locally without
continuous Codex polling. Spend agent turns on dataset correctness, failed-run
diagnosis, milestone comparisons, and decisions that can change the model.

For unattended continuation, `scripts/continue_full_pipeline.ps1` waits for any
tracked download/training PIDs, verifies the resumable download, prepares and
repackages the conservative full-data split, runs tests, then starts or resumes
the full-data experiment. Its durable log is
`outputs/orchestration/full_pipeline.log`.
