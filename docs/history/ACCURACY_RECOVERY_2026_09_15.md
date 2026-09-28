# Accuracy recovery: 15 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## What the audit actually found

The application still uses the protected 29 August model. Later experiments
have not earned a release. Changes to the app interface and evaluation tooling
are useful work, but are not evidence that height accuracy improved.

On the corrected full native scene development benchmark, the protected
model has 13.243 m measured-building RMSE and 8.361 m vegetation RMSE. The
older crop scores used different pixels and label rules. They cannot be
substituted for these scores.

The fair DC + Philadelphia classification replay records six-class macro F1
of 56.48% for spatial V3 and 55.11% for hierarchical V4. The hierarchy did
not improve the combined six-class result. V4 remains rejected.

## Why the small height runs stalled

The four-epoch height-safe control changed only 98 of 31,428,489 parameters.
Its pooled GAMUS validation RMSE moved from 7.1211 m to 7.1276 m. No epoch
passed the improvement criteria.

A diagnostic on four fixed GAMUS **training** crops found:

- The canopy/semantic adapter sees RGB, scalar relative depth, base height,
  and building probability. It has a nominal 5 by 5 pixel receptive field,
  with no direct access to the backbone's richer encoder/decoder features.
- Mean effective correction strength was 0.0266.
- The optional hard canopy gate fired on about 0.085% of the true vegetation
  pixels in these training crops.
- The final vegetation loss passed about 1.45% of its direct height gradient
  through to the canopy output, approximately 69 times weaker.
- In the protected-vegetation fusion formula, the separate building residual
  expert does not enter the final building height. Improving this expert can
  therefore leave final building predictions unchanged.

These are diagnostic observations, not validation accuracy claims. Evidence:
`outputs/diagnostics/height_learning_bottlenecks_v1/report.json`.

## Data and processing corrections

1. **HighBuild unknown pixels stay unknown.** Urban regression uses measured
   height support intersected with building footprints. Estimated heights are
   kept out of the primary metric. No height label means no height loss.
2. **Supervised urban crops.** Some buildings cover under 1% of a tile.
   The new training recipe samples a crop containing measured support.
   Validation still covers the complete fixed scene.
3. **Independent prior maps.** Cached Depth Anything V2 maps use their stored
   0-1 values, independent of where height labels happen to exist.
4. **Independent masks.** Image validity, class availability, and height
   availability remain distinct. Road and water height artifacts are excluded
   while their classification annotations are retained.
5. **Equal contributions to the loss.** Sources and object types with sparse
   labels must not be outweighed just because forest/ground have more pixels.
6. **Inference protocol audit.** Compare the full-native benchmark with the
   actual app's 512/128 tiling on the identical reference pixels. GroupNorm
   makes crop size/context matter. This audit changes inference protocol and
   precision; it is not a claim of learned model improvement.

## New experiment: residual_height_v1

### App padding audit: a measured processing issue

The paired full-native/app audit evaluated all 160 urban and 120 forest scenes
on the same 31,011,260 reference pixels. App tiling raised measured-building
RMSE from 13.243 to 13.442 m and vegetation RMSE from 8.361 to 11.882 m.

A separate controlled forest comparison then held fp16 precision fixed and
changed only small-image padding. All 120 scenes and 17,155,570 valid pixels
were retained. Minimal padding reduced vegetation RMSE from 11.882 to 8.366 m
and forest RMSE from 8.639 to 6.707 m. However, forest-ground RMSE worsened from
4.297 to 4.885 m; 89 scenes improved and 31 worsened. Therefore the optional
`padding_policy="minimal"` is tested but **not enabled globally**. The API and
backward-compatible `fixed_tile` default remain unchanged. This is an inference
correction, not a learned-model improvement. Evidence:
`outputs/evaluation/open_canopy_padding_v1/report.json`.

### Training correction

The new model retains the entire protected model as a frozen feature source.
A separate copy of its multi-scale decoder learns a correction to the final
height. Its final correction layer starts at zero, so the initial prediction
is exactly the protected prediction. The correction receives the encoder's
image features, RGB, cached relative depth, and the original predicted height.
Reference masks are used only by the loss and evaluator.

This increases learnable capacity from 98 numbers to a full spatial decoder
while keeping the classification computation frozen. A height error directly
teaches this decoder without passing through the earlier expert gates.

Training uses QC-approved GAMUS training data plus corrected measured
HighBuild and OpenCanopy replay. First perform a 48-step feasibility check on
four fixed training samples. Require at least 15% loss reduction, nonzero
decoder gradients, and unchanged protected tensors/classification. A successful
feasibility check only shows that optimization works; it is not generalisation.
Restart from the zero correction before the real run.

The initial run has up to six epochs and saves atomic resumable checkpoints.
All validation is paired with the same frozen baseline, image inputs, and masks.
Track RMSE, MAE, bias, correlation and R2 by source and object type. Keep the
full-native and app-tiling reports separately named.

## What counts as a useful result

First milestone: at least a 0.50 m or 5% reduction in building or vegetation
RMSE, while the other measured domains regress by no more than 0.15 m and
MAE/bias/correlation checks remain acceptable. The classifier must produce
the same results from the same image. Longer-term building RMSE below 5 m and
vegetation RMSE below 7 m remain targets, not promised outcomes.

The frozen full-scene scorecard and additional per-building checks still govern
release review. Candidate success on development validation alone cannot
establish new-location accuracy or justify a production switch.

The development selection score gives equal weight to HighBuild building RMSE,
OpenCanopy vegetation RMSE, and GAMUS's three-domain macro RMSE. Ground-only
improvement cannot satisfy the required material improvement. Undefined
correlation on constant targets/predictions is labelled non-comparable; it is
not rewritten as zero or as perfect agreement.

## Proposed competitive product targets (not research cutoffs)

The SAC reference README was rechecked on 15 September 2026:
project-brief-reference . It specifies 50% DSM
accuracy and 50% rendering/UX, but no numerical pass marks or winning scores.
The targets below are engineering ambitions, not promised results or published
industry certification thresholds.

| Measure | Ambitious product target |
| --- | --- |
| Measured building heights | RMSE 3-5 m; MAE at most 2-3 m |
| Canopy heights | RMSE 4-6 m; MAE at most 2-3 m |
| Short vegetation (0-5 m reference band) | MAE at most 1 m; report separately |
| Correlation / R2 | Aim for at least 0.8 / 0.7 per non-flat landscape |
| Systematic height bias | Absolute bias at most 1 m per evaluated category |
| Terrain / absolute DSM | Beat a registered DEM-only baseline on independent reference pixels; report all four landscapes and vertical datum |
| Additional six-class identification | Macro F1 at least 0.80 as a later independent target, never a replacement for height validation |
| Interactive usability | At least 30 FPS on a declared scene/mesh/device; 20 consecutive completed upload-to-export regression tests |

Use independent geography and the actual deployed inference pipeline. Publish
sample counts, resolution, height bands, per-scene spread and uncertainty of
the measured scores. Pixel counts are not independent sample counts. Report
above-ground heights separately from terrain elevation and absolute DSM.
Calibration references must be separate from evaluation references. A good
pooled R2 over a mountain scene can hide poor local building/tree reconstruction.

Reaching building RMSE below 5 m requires about 62% improvement from the current
13.243 m corrected development score. One passing optimization check or a small
pilot cannot establish that such a reduction will occur.

## Geographic evidence

GAMUS DC validation tiles are often adjacent to training tiles; prior work also
exposed NYC. OpenCanopy development regions differ, but source mosaics overlap.
These are development checks, not proof of geographic independence.

The Bay of Plenty RGB + metric LiDAR DSM/DEM package is sealed separately.
It remains unconsumed until a candidate and a comparison protocol are fixed.
Its one-shot result must not be used to select thresholds or train the model.

## Credit and compute policy

Run one GPU experiment at a time. Use persistent local logs/checkpoints and
an hourly scheduled check during training. Notify on completed epochs, a
meaningful failure, or completion. Resume analysis after training; inspect
failed outcomes before authorizing another experimental recipe. Keep source
code, data contracts, rejected experiments and production artifacts saved.

## Local run and recovery

- Launch: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_residual_height_v1.ps1`.
  A matching passing training-only feasibility report is mandatory.
- Resume: the same launcher with `-ResumeCheckpoint <absolute checkpoint_latest.pt>`.
  Resume uses the experiment's saved config and verifies protected artifacts.
- View: `powershell -NoProfile -File scripts/watch_residual_height_v1.ps1`.
  Closing the viewer does not stop training.
- Persistent log: `outputs/orchestration/residual_height_v1.log`.
- Results: `experiments/*_residual_height_v1/metrics.jsonl`, `status.json`, and
  atomic epoch/latest checkpoints. Only an eligible epoch may create
  `checkpoint_best_guarded.pt`; it does not change the app's active model.

Resume restores model, optimizer, scheduler and main-process RNG states at an
epoch boundary. Multiworker random-crop streams are statistically resumable,
not guaranteed bitwise identical to an uninterrupted run.

## Execution handoff: 15 September, 17:30 IST

- Full Python regression suite: **576 passed**, 12 dependency/georeferencing
  warnings; no test failures.
- Final sealed training-only feasibility proof:
  `outputs/diagnostics/residual_height_v1_overfit/report.json`
  (proof run ID `20260915T115808Z`; source snapshot is versioned by this ID).
  Loss fell 7.4554 to 5.3215 (28.62%) in 48 steps on four training examples.
  Every source contributed, copied-decoder gradients were nonzero, protected
  semantic outputs were exactly equal before/after, and inherited weights did
  not change. These are optimization checks, not validation accuracy.
- Fresh full run started: `experiments/20260915T115844Z_residual_height_v1`.
  Its first stage is paired full-scene baseline validation. No feasibility
  weights are reused. Up to six epochs follow, with the declared guards.
- Background launcher PID at startup: 10496. The virtual-environment Python
  launcher forwards to the actual interpreter; two related Python process
  entries do not mean two competing trainers.
- Hourly heartbeat: `msr-residual-accuracy-check`. It checks meaningful
  progress/failures and resumes the evidence-led evaluation work after training.
- RTX confirmed active during baseline validation; no app model change.
- Do not modify the source files listed in the experiment's source manifest
  while this run is active. The trainer authenticates its frozen source/config.

## Recovery: 15 September, 17:55 IST

The first run failed at 17:34 IST during baseline validation, before training an
epoch or writing any learned checkpoint. The common evaluator unconditionally
computed both expert metrics even on HighBuild-only data, which has no valid
vegetation reference pixels. This raised `No valid pixels were accumulated`.
The original experiment, source snapshot, log, and a `failure_recovery.json`
record are preserved; this failure is not a failed model-accuracy result.

The evaluator now reports unsupported experts as unavailable (not zero), skips
zero-support regional summaries, and still rejects a wholly unlabeled suite.
Regression tests cover urban-only, vegetation-only, ground-only, missing-region
support and entirely missing truth. Per-suite batch progress and terminal
failure status are now recorded by the residual trainer.

The recovery regression suite passed **584 tests**. A real CUDA validation smoke
test processed two complete native samples from each source and passed:
`outputs/diagnostics/residual_height_v1_validation_smoke/20260915T122952Z/report.json`.
It verified HighBuild's building-only support and OpenCanopy's ground/vegetation
support without requiring nonexistent class labels. Architecture, training
recipe and acceptance criteria are unchanged. A new source-bound training-only
feasibility proof is required before the fresh restart.

### Windows status-lock recovery

The second run `20260915T123138Z_residual_height_v1` finished GAMUS and HighBuild
baseline passes but failed during OpenCanopy baseline when Windows denied the
atomic replacement of a status file held open by a reader. It also trained no
epoch. Its failure summary is preserved.

Atomic JSON/text/checkpoint replacement now retries only transient Windows
sharing/access errors 5/32/33, with a bounded delay; permanent errors still
raise and preserve the prior file. A real Windows CreateFileW sharing-lock
integration test passed. The compact viewer opens status and metrics with
FileShare.ReadWrite | FileShare.Delete so it cannot block these renames.

Smoke test `20260915T123837Z` and resealed 48-step proof `20260915T123910Z` passed.
The new active run is `experiments/20260915T124056Z_residual_height_v1`, launched
at 18:10 IST. The hourly monitor follows this run; both failed attempts remain
available as evidence. Model architecture, recipe, data and guards did not
change during these runtime recoveries.

### Confirmed training handoff: 18:17 IST

All three baseline suites completed in the active `20260915T124056Z` run.
Epoch 1 was directly observed at batch 375/1500 with finite loss 5.8906; RTX
utilization was 86%, temperature 63 C. No epoch-validation improvement is yet
claimed. The final full regression suite passed **593 tests**, including the
real Windows sharing-lock test. Hourly monitoring is active and follows this
run. See `docs/RESIDUAL_HEIGHT_POST_RUN.md` for the remaining evaluator
compatibility work required before a formal candidate release decision.
