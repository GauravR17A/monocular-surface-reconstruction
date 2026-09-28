# Class-assisted height comparison: completed, not promoted

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Run: `D:/MSRData/experiments/class_assisted_height_fast_v2/20260919T164305030475Z`.
Completed 2026-09-19 16:50:51 UTC. Two epochs per arm, including the authenticated
continuation of control A epoch 1. No candidate passes both benefit and safety.

## Fixed development results

RMSE in metres, lower is better. All rows use the same native-resolution
160 HighBuild / 120 OpenCanopy development images and reference-valid pixels.

| Model | Buildings | Canopy | Ground | Short vegetation | Legacy safety |
|---|---:|---:|---:|---:|---|
| Protected app model | 13.2426 | 8.3610 | 4.8944 | 5.2049 | Baseline |
| A control, epoch 1 | 12.9630 | 8.0529 | 5.1278 | 5.5979 | Fail |
| A control, epoch 2 | 12.7829 | 7.7268 | 5.3143 | 5.9793 | Fail |
| B class-assisted, epoch 1 | 12.9553 | 8.0071 | 5.1252 | 5.6380 | Fail |
| B class-assisted, epoch 2 | 12.7883 | 7.6885 | 5.2600 | 5.9624 | Fail |

Best unrestricted mean building/canopy RMSE is B epoch 2. Best individual
building RMSE is A epoch 2. Best eligible candidate: **none**. Both B epochs
fail safety; B epoch 2 passes the separate canopy-benefit rule, not release.

B epoch 2 versus protected: building RMSE falls 3.43%, canopy 8.04%; ground
RMSE rises 7.47% and short-vegetation RMSE rises 14.55%. These are relative
changes in error, not percentages of correct predictions.

## What the matched comparison actually establishes

B epoch 2 beats A epoch 2 on canopy by only 0.0383 m (about 0.50%), while its
building RMSE is 0.0054 m worse. Most improvement versus the protected baseline
also occurs without informative six-class probabilities. One seed and two epochs
do not establish a robust, material benefit from class conditioning. Nor do they
prove that classification can never help height in a different controlled design.

## Error pattern and limitations

- Underestimated tall-object heights improve somewhat: B epoch-2 tall-building
  RMSE 29.80 -> 28.60 m; tall-canopy RMSE 10.49 -> 9.01 m. Errors remain large.
- The gain comes with excessive positive height on shorter surfaces. Ground
  bias rises from +1.56 to +2.06 m; short-vegetation bias +2.56 to +3.08 m.
  This is consistent with an insufficiently selective upward correction, not
  proof of a single architectural cause. Both arms show the same broad tradeoff.
- B epoch-2 building correlation/R2 are 0.220 / 0.006; canopy 0.450 / 0.008.
  Lower RMSE does not make this a high-accuracy reconstruction system.
- Per-region safety checks also fail. A failed-check count is not an image count
  or a statistical significance estimate; multiple metrics/groups overlap.
- No fresh final geography, GAMUS height safety, road/water height validation,
  broad hill/plain stability or cross-sensor universal accuracy was established.

## Justified next experiment, not launched

1. Inspect fixed development examples where ground/low vegetation rises alongside
   improved canopy; stratify by reference height, boundaries and class/geometry
   disagreement. Keep dataset split and input-validity contracts unchanged.
2. Define a bounded test of stronger ground/short-vegetation protection in the
   residual loss, paired against the same control. Do not hard-force roads or
   water to zero, use reference classes at inference, or hide failed categories.
3. Independently address tall-building training support and missing physical pixel
   scale before expecting a major building gain. More epochs alone are not proof
   these problems will disappear.
4. If a new development candidate passes predeclared guards, then run per-building,
   actual app-path and appropriately reserved geography checks before promotion.
   Do not loosen this completed run's rules or consume final tests while tuning.

## Saved and protected

Both final committed checkpoints and evaluation hashes were reverified after
completion. All 18 bound source files match; fixed support and protected semantic
output identity checks pass. Production height checkpoint, live pointer, V1
classifier and V3 conditioning-classifier hashes match their protected values.
The original slow run and its checkpoint/cache remain preserved. No trainer is
active. The faster run took roughly 7m46s including cache authentication and
four validations; this excludes the already-completed parent training epoch.

Readable report, full-group CSV and snapshot JSON are under
`outputs/reports/class_assisted_height/20260919T164305030475Z/`.
The run monitor is paused after completion reporting. No further training or app
promotion was started as part of this status check.
