# Low-surface protection outcome — not promoted

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Completed 19 September 2026, 17:14:17 UTC. Run:
`D:/MSRData/experiments/class_assisted_height_low_surface_v3/20260919T170601414242Z`.
Two fresh epochs per arm; approximately 8 minutes 16 seconds total including
input authentication. Production and both frozen classifiers remain unchanged.

## Result

All rows below use identical native-resolution development support, not final
unseen tests. RMSE in metres; lower is better.

| Category | Protected | Previous class-assisted epoch 2 | New class-assisted epoch 2 |
|---|---:|---:|---:|
| Buildings | 13.243 | 12.788 | 13.076 |
| Canopy | 8.361 | 7.689 | 9.637 |
| Ground | 4.894 | 5.260 | 4.144 |
| Short vegetation, reference 0–2 m | 5.205 | 5.962 | 3.233 |
| Tall buildings, reference >=20 m | 29.797 | 28.600 | 29.041 |
| Tall canopy, reference >=15 m | 10.486 | 9.012 | 12.577 |

The new class-assisted model reduces ground RMSE by 15.32% and short-vegetation
RMSE by 37.89% from protected, but worsens canopy RMSE by 15.26%. These are error
changes, not accuracy percentages. Errors of 3.23 m on vegetation known to be
0–2 m high remain too large for precise individual-object measurement.

Ground bias improves +1.555 -> -0.031 m and ground MAE 2.896 -> 1.931 m.
Canopy bias worsens -3.517 -> -6.161 m. The new objective has exchanged the
previous upward spillover for excessive suppression of taller vegetation.
It does not solve selective geometry estimation.

The matched uniform-input epoch-2 control has building/canopy/ground/short
vegetation RMSE 13.085 / 9.619 / 4.155 / 3.316 m. Class information therefore
adds only small mixed differences: approximately -0.009 m buildings, +0.018 m
canopy, -0.011 m ground and -0.083 m short vegetation. This single paired run
does not establish robust height improvement from six-class conditioning.

All four epochs fail original safety gates. Neither candidate epoch meets the
priority benefit gate. Best unrestricted building/canopy average within the new
run is **control A epoch 2**. Best eligible checkpoint: **none**. Earlier
class-assisted epoch 2 still has better building/canopy numbers but remains
rejected for ground/short-vegetation damage. Never combine category-wise minima
from different checkpoints into a claimed single-model result.

## Additional train-only context diagnostic

On the original fixed 20 feasibility examples (10 urban + 10 forest), changing
urban input context from full 1024 px to its same-pixel 384 px training crop
changes protected building heights by 0.774 m mean absolute difference; after
excluding a 32 px crop border the difference remains 0.753 m. Forest examples
already have native 384 px size and give identical predictions. This is a
limited training diagnostic, not a population accuracy score or proof that
context is the dominant failure. It cannot explain the forest tradeoff.

## Decision and next justified work

Do not promote, prolong this recipe automatically, relax its gates, or consume
reserved geography. Keep six-class identification independent of height.

The next bounded experiment should address object-height selectivity rather
than merely raising/lowering the loss weight again:

1. Derive a training-only height-band exposure plan, retaining true labels and
   original masks, to separately cover shorter/taller measured buildings and
   shorter/taller vegetation. The saved HighBuild audit has 9,841 measured/not-
   marked-estimated outlines, overlap caveats, and missing GSD on all 450 tiles.
   Do not silently treat every outline or inferred pixel scale as authoritative.
2. Test height-band-balanced crop exposure as one intervention with fixed
   optimizer budget and matched control; retain full native validation and
   explicitly report short, medium and tall strata so aggregate gains cannot
   conceal the same failure again. Predeclare new experimental settings first.
3. Treat per-instance supervision, context-matched training and recovered
   physical pixel scale as separate tests, not simultaneous unexplained changes.
4. Only an eligible development candidate advances to per-building, app-path,
   and genuinely reserved-geography evaluation. No all-image accuracy promise.

## Saved evidence and checks

- `outputs/reports/class_assisted_height_low_surface_v3/REPORT.md`: all epochs,
  historical controls, RMSE/MAE/bias/correlation/R2 and failed-check summary.
- `outputs/reports/class_assisted_height_low_surface_v3/verification.json`:
  protected hashes, source snapshots, committed artifact checks and paired
  initialization/sample order verified. A reporting-only serialization check
  was corrected to authenticate original config bytes plus saved JSON semantics;
  no training binding, checkpoint, evaluation or gate was changed.
- `outputs/diagnostics/class_assisted_low_surfaces_v1.json`: all 280 development
  images from the preceding candidate, stratified errors and correction direction.
- `outputs/diagnostics/height_crop_context_train_v1.json`: train-only context check.
- 49 targeted tests passed across loss, masking, protocol, resume, frozen model,
  gates, data provenance, dashboard, reporting and Windows atomic replacement.

All code and checkpoints are saved. No trainer remains active. The existing
monitor remains paused; no extra recurring checks or new training were started.
