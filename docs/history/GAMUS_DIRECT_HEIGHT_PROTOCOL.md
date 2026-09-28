# GAMUS direct-height experiment protocol

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: predeclared before candidate training.

## Purpose

Test whether genuine GAMUS metric-height fine-tuning improves Monocular Surface Reconstruction's
final surface prediction. This is not another identification-only router run.
The protected application checkpoint remains unchanged throughout.

## Data roles

- GAMUS official train: all 5,004 source tiles were audited; 5,001 approved
  tiles are training data and three all-class-255/no-supervision tiles are
  quarantined by the immutable QC contract.
- GAMUS official validation: all 859 usable tiles are development evaluation.
- GAMUS official test: all 2,861 source tiles were excluded from this pilot's
  training, tuning, epoch selection, thresholds, and architecture decisions.
  The older Stage-3 final audit consumed the official test once;
  it is now immutable historical evidence and is forbidden for every new V4
  decision or repeat evaluation.
- Corrected HighBuild/Open-Canopy validation: regression-safety evidence only;
  it is not merged into GAMUS validation statistics.
- A newly locked external/geographic reference set is required before an app
  promotion or an unseen-city claim. Until then, a validation win creates only
  a candidate checkpoint.

NYC does not satisfy that system-level unseen requirement. A fresh classifier
head can exclude NYC while learning on DC+Philadelphia, but the broader system
already has New York exposure through earlier GAMUS experiments and its
inherited HighBuild-trained checkpoint.

Every accepted GAMUS tile must pass the versioned QC index. The QC report
records split counts, file identities, tensor shapes, numeric ranges, semantic
classes, unit provenance/assumption, and quarantined records. Files are never
deleted by QC.

## Model and learning objective

- Warm start from the protected Monocular Surface Reconstruction checkpoint.
- Keep the cached Depth Anything V2 prior fixed for this experiment.
- Directly supervise the final `height` output against valid GAMUS above-ground
  heights; semantic identification remains an auxiliary objective.
- Allow only the late encoder/decoder, metric-height path, and required adapter
  heads to learn at conservative, separately controlled learning rates.
- Use a fresh optimizer and schedule. Do not reuse Stage-1 or Stage-2 optimizer
  state.
- Use BF16 on the RTX 4070 with a small microbatch and gradient accumulation.
  Reduce crop size only after a confirmed CUDA out-of-memory failure.
- Save each epoch and a resumable latest checkpoint. Never write to the live
  pointer automatically.

## Fixed validation gates

All comparisons use identical inputs, masks, units, crops/tiles, and metric
code for the protected baseline and candidate.

On full GAMUS validation:

- pooled RMSE improves by at least 0.20 m;
- pooled MAE improves by at least 0.10 m;
- correlation drops by no more than 0.01 and R-squared drops by no more than
  0.02;
- neither DC nor Philadelphia RMSE regresses by more than 0.10 m, and at least
  one improves by 0.20 m or more (enforced with a 0.20 m city-macro
  improvement gate plus the per-city caps);
- ground RMSE regresses by no more than 0.05 m;
- building and vegetation RMSE each regress by no more than 0.10 m, and at
  least one improves by 0.20 m or more (enforced with a 0.20 m object-domain
  macro improvement gate plus the per-domain caps);
- tall-building and tall-vegetation RMSE each regress by no more than 0.20 m;
- semantic macro F1 drops by no more than 0.02 and no class F1 drops by more
  than 0.03;
- no NaN, empty required stratum, split overlap, missing prior, unit mismatch,
  or artifact-identity mismatch is allowed.

On corrected legacy validation:

- overall, urban, forest, building, and vegetation RMSE each regress by no more
  than 0.15 m;
- corresponding MAE values regress by no more than 0.10 m;
- HighBuild results are reported separately under
  `highbuild_all_annotated_positive_v1` and
  `highbuild_measured_coco_intersection_v1`.

For later comparisons, the canonical corrected full-scene protected baselines
are 13.243 m HighBuild strict-measured building-only RMSE, 8.361 m OpenCanopy
vegetation-only RMSE, and 6.707 m OpenCanopy all-valid forest RMSE. They have
different masks and support and must never be treated as interchangeable scores.

These gates are frozen before results are seen. Failure is recorded honestly;
thresholds are not relaxed after the run.

## Decision rule

A gate-passing pilot authorizes a longer or correctly masked replay experiment.
It does not automatically replace the application model. Promotion additionally
requires a newly locked geographic holdout, visual sanity checks, API/viewer
regression tests, checkpoint hashing, and explicit human review.
