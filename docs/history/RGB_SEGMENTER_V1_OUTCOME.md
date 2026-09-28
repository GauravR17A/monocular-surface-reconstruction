# RGB-only six-class pilot: completed outcome

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Update: the separate fresh-checkpoint replay has now **passed on all 859
validation images with exactly matching counts and scores**. See
`docs/RGB_SEGMENTER_V1_TEST_REPORT.md` and
`outputs/evaluation/gamus_rgb_segmenter_v1_independent_replay/report.json`.
This is repeatability evidence, not external-geography validation or app release.

## Result and scope

Run `experiments/20260915T140116654177Z_gamus_rgb_segmenter_v1` completed
eight epochs at 2026-09-15 14:25:30 UTC. Both the best development checkpoint
and best checkpoint passing every predeclared gate are **epoch 8**.

On the fixed 859-image DC/Philadelphia development validation, six-class macro
F1 increased from **56.48% to 83.83%** (+27.34 percentage points). Macro IoU is
**72.47%**. These are pixel-identification metrics, not object-count accuracy,
height accuracy, or evidence of generalization to new geography.

| Epoch | Six-class macro F1 | All 27 checks pass? |
|---|---:|---|
| 1 | 79.71% | No: DC buildings and road boundaries |
| 2 | 80.75% | Yes |
| 3 | 82.02% | Yes |
| 4 | 82.73% | Yes |
| 5 | 83.33% | Yes |
| 6 | 83.35% | Yes |
| 7 | 83.51% | Yes |
| 8 | 83.83% | Yes |

## Best-epoch identification results

| Category | Overall F1 | DC F1 | Philadelphia F1 |
|---|---:|---:|---:|
| Ground | 77.37% | 68.17% | 82.53% |
| Buildings | 88.69% | 86.38% | 91.40% |
| Water | 91.37% | 90.01% | 92.12% |
| Roads | 81.28% | 77.56% | 88.88% |
| Low vegetation | 79.87% | **22.49%** | 87.22% |
| Trees | 84.38% | 82.86% | 85.86% |
| Macro F1 | 83.83% | 71.24% | 88.00% |

All overall class F1 scores improved over the fixed V3 replay. DC and
Philadelphia macro F1 improved by 19.95 and 32.36 percentage points,
respectively. No gate was loosened after observing this run.

## Remaining errors: saved-metric analysis, not a new evaluation

- **DC low vegetation remains weak.** F1 is 22.49%, precision 38.74%, and
  recall 15.84%; most reference low vegetation is still missed. Philadelphia's
  much stronger performance makes the pooled overall number look better.
  Passing a guard against an already weak baseline is not production readiness.
- Road-boundary F1 at the fixed two-pixel tolerance is 42.96% overall,
  36.52% in DC and 66.62% in Philadelphia. Boundaries remain appreciably less
  accurate than broad road classification, despite passing the protection gates.
- False water on dark non-water reference pixels is 0.458% overall
  (DC 0.437%, Philadelphia 0.524%). This is a dark-RGB proxy, not a measurement
  against independently annotated shadows. Total water false positives remain
  3,193,899 pixels; overall water precision is 86.45%.
- Epoch 8 is best on the declared macro-F1 criterion, not every class at once:
  water F1 peaked at 92.75% in epoch 5 versus 91.37% at epoch 8.
- No height branch was trained or changed. The separate residual-height run
  remains rejected; its results must not be blended into these percentages.

## Saved evidence and next release steps

The exact experiment directory contains epoch 1-8 checkpoints, latest,
best_development, best_guarded, metrics, final summary, config, source snapshots,
data/software binding and both feasibility reports. The new classifier has
**not** replaced the protected app model or pointer. No additional training or
holdout evaluation was started while preparing this outcome.

Read-only CPU integrity audit passed: 210 model tensors and all 615 saved
optimizer tensors are finite; scheduler/RNG states exist; checkpoint history
matches all eight metrics rows. The seven current source files, seven source
snapshots, current/saved configs and binding match. Best guarded, best
development, latest and epoch 008 checkpoints are byte-identical, SHA-256:
`4940b2d4ae8817370384ccb54da9f73aee1b6bf3614435d8c077bf3e3363d021`.
Protected checkpoint and app pointer hashes still match their original seals.
These checks prove artifact integrity, not predictive accuracy.

Next: authenticate an independent checkpoint replay of the exact benchmark;
inspect DC low-vegetation confusion and labels without tuning on sealed
holdouts; then carry out the separately planned genuinely new-geography
evaluation. App integration needs an explicitly tested RGB overlay adapter
and regression tests proving that classification cannot silently change height
outputs. Height improvement is a separate workstream.

NYC is excluded from this new classifier's training, not historically unseen
by the entire system. Official GAMUS test and external holdouts were not opened
for this completion check. The NVIDIA MiT-B0 backbone remains restricted to
non-commercial research/evaluation under its upstream licence.

The original outcome sections above summarize saved training metrics. The
subsequent independent inference replay is documented in the linked test
report; neither is a new-geography scientific validation study.
