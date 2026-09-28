# Sampling comparison: no accepted replacement

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Completed run: `D:/MSRData/experiments/height_band_sampling_v1/20260919T173241288352Z`.
All eight epochs completed. No candidate passed every safety check. Production
and classifier checkpoints/pointer remain unchanged. Final committed artifacts
and frozen protected tensors were verified in
`outputs/diagnostics/height_conditioning_response_v1/report.json`.

| Native development RMSE (m) | Protected | Control epoch 4 | Balanced epoch 4 |
|---|---:|---:|---:|
| Buildings | 13.243 | 12.795 | 13.094 |
| Canopy | 8.361 | 7.807 | 7.770 |
| Ground | 4.894 | 4.642 | 4.910 |
| Short vegetation | 5.205 | 4.936 | 5.215 |
| Tall buildings | 29.797 | 28.834 | 28.115 |
| Tall canopy | 10.486 | 9.594 | 9.387 |

Balanced epoch 4 passes the paired canopy-benefit rule but fails safety. Its
additional canopy advantage over matched control is only 0.038 m, while
building/ground/short-vegetation aggregates are worse than control. It is not a
convincing general solution. Best unrestricted checkpoint by the declared equal
building/canopy-RMSE ranking is CONTROL epoch 2; it too fails safety. Best
eligible checkpoint: NONE. Do not mix best categories from different epochs.

## Diagnostic and next experiment

A same-checkpoint intervention on the original fixed 20 TRAIN examples changed
the earlier predicted-class model's input from actual soft classes to uniform
classes. Mean absolute height change was 0.18-0.19 m on buildings and 0.22-0.60 m
across vegetation bands. Thus class inputs are not disconnected or entirely
ignored, but the earlier tiny paired development gains do not show sufficient
useful transfer. On these examples, predicted coarse vegetation covered only
26.3% of labelled short-vegetation pixels, versus 84.5% for tall vegetation.
This is a small, non-representative training diagnostic, not a segmentation
benchmark; zero label height or reference class never enters inference.

Test a different explicit mechanism: six learned height-correction outputs
mixed by frozen classifier probabilities at native resolution. Compare with
identical architecture and uniform routing. Both use original sampling and
original loss, since height-band balancing did not establish an overall gain.
This is a hypothesis, not a promised improvement; class mistakes can still hurt.
First require a training-only optimization/identity proof before development
training. Retain every original safety check, no reference routing, no GAMUS
height-unit assumption, no final-test geography, no automatic promotion.
