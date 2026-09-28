# Height-band sampling: fixed four-epoch matched experiment

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Predeclared 19 September 2026, before this run trains. User authorized continuing
the accuracy work. All earlier experiments and production remain immutable.

## What changes — and what does not

Test sampling, not another classifier or loss weight. Both arms use the original
source/domain-balanced Huber + 0.02 MSE loss, fresh identical protected start,
residual decoder, uniform six-channel conditioning, bf16, batch 2, accumulation
2, 384 px patches, learning rates and optimizer from the passed feasibility
recipe. No weights from failed runs are reused. The rejected low-surface penalty
is NOT used in either arm. The current comparison therefore isolates sampling;
historical low-surface-loss results remain a separate comparison, not this control.

Both arms run four epochs at 600 urban/forest pairs per epoch (1,200 optimizer
updates per arm). This is a newly declared common budget, not a silent extension
of a completed two-epoch experiment. No extra epochs or automatic promotion.

Control A reproduces the previous sampler: all 450 urban images plus 150 urban
repeats, all 600 forest images once, crops anchored on supported height pixels.

Candidate B uses 300 regular-coverage + 300 height-targeted draws per source:

- Urban: 150 shorter-building (<20 m) and 150 tall-building (>=20 m) targeted
  draws; crop anchors come from supported pixels of the requested band.
- Forest: 75 each for labelled ground, short vegetation (0<h<=2 m), medium
  vegetation (2<h<15 m) and tall vegetation (h>=15 m). These chips already have
  native 384 px size, so balancing changes chip frequency, not crop geometry.
- Targeted image probability is proportional to its fraction of supported
  pixels in that band, with at least 64 such pixels. No silent missing-band fallback.
- A deterministic coverage cycle ensures ALL 1,050 training images have been
  shown by epoch 2, and again within the epoch 3–4 cycle. Per-epoch repetition,
  unique images, regions and actual supported-pixel exposure are recorded.

Height labels choose training observations only. All valid labels within each
crop remain supervised; no band-specific loss mask is substituted. No true
height/reference class reaches model.forward. Unknown labels remain ignored,
and image/class/height validity remain independent. No road/water zero rule.

## Measured preparation evidence

The CPU-only preparation read all 1,050 TRAIN images and generated all four
epochs before training. It never read development or final-test data.

Epoch 1 actual labelled pixel observations:

| Band | Control | Balanced |
|---|---:|---:|
| Short buildings | 12,347,770 | 10,830,097 |
| Tall buildings | 1,451,884 | 4,205,520 |
| Ground | 47,710,844 | 46,158,656 |
| Short vegetation | 2,275,710 | 3,242,330 |
| Medium vegetation | 19,178,553 | 20,223,934 |
| Tall vegetation | 17,100,459 | 16,667,717 |

These are repeated training observations, not new independent labelled pixels
or guaranteed accuracy gains. The candidate has fewer distinct images within
one epoch but complete coverage by two. Training region frequencies can change;
the audit records them. This is a sampling intervention, not a city-holdout claim.

## Evaluation, gates and integrity

Recompute the protected native baseline on all 160 urban + 120 forest DEVELOPMENT
images, retaining original support and all old groups. Require old baseline
metrics/support/semantics to match. Add explicit short-building and medium-
vegetation global/region groups; apply original safety limits to these as well.
Keep original short vegetation, tall objects, boundaries and region checks.

No original gate is loosened: RMSE +0.15 m, MAE +0.10 m, absolute bias +0.10 m,
correlation/R2 decrease tolerance 1e-6. Benefit is >=0.5 m AND >=5% building OR
canopy RMSE reduction from protected, beating same-epoch control. Report safety
and benefit separately and distinguish unrestricted from eligible checkpoints.

Checkpoint each 50 optimizer updates, atomic commit after save/evaluation, exact
optimizer/RNG resume, original-source/software checks, copied immutable sample
plans and original data/cache hashes. A shared OS lock prevents overlapping
training jobs. Stop on invalid/nonfinite data, gradients, changed bindings or
unexpected frozen weights. Do not silently change batch/recipe on recovery.

No GAMUS height units are assumed, no final reserved geography consumed, no
all-image accuracy claim, no app checkpoint/pointer changes. New artifacts are
on D:. If this succeeds, per-building and real app-path checks precede release.
