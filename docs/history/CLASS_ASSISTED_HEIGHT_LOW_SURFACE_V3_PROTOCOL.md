# Low-surface protection: bounded matched experiment

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Authorized by the user's request to continue on 19 September 2026. This protocol
is written before training. It does not change any previous experiment's verdict.

## Evidence and hypothesis

The complete 280-image DEVELOPMENT audit of the previous class-assisted epoch 2
found that 99.13% of supported short-building pixels and 99.67% of tall-building
pixels were raised. Short buildings (<20 m) RMSE worsened 4.77 -> 4.83 m, while
tall buildings improved 29.80 -> 28.60 m. Forest ground received +0.50 m mean
correction and short vegetation +0.52 m. Of labelled short-vegetation pixels,
75.45% had larger absolute errors. Medium vegetation (2-15 m) also worsened
(5.90 -> 6.30 m), while tall canopy improved (10.49 -> 9.01 m).

Thus, the aggregate benefit concealed errors in shorter groups. These are pixel
statistics on fixed development data, not percentages of wrong images, an
independent final test, or proof of a single architectural cause.

Hypothesis: the existing vegetation loss averages short and tall vegetation;
explicit short-vegetation supervision and a per-pixel low-surface non-regression
penalty may reduce that tradeoff. This may also reduce the earlier canopy gain.
No improvement or acceptance is guaranteed.

## The one declared training change

Keep the original source/domain-balanced Huber + 0.02 MSE loss. Add:

- 0.5 times separate mean short-vegetation supervision (reference 0 < h <= 2 m).
- 1.0 times a low-surface regret penalty, averaged over present sample/group
  means for reference-labelled ground and short vegetation. For each pixel,
  regret = max(0, |prediction-reference| - |protected-reference|); penalty is
  regret + 0.02 regret^2.

The frozen prediction is a comparator, not the truth. Improving it is allowed;
making its absolute error larger costs extra loss. Actual labels are retained:
short vegetation is NOT assigned zero. Missing labels are excluded before
arithmetic. Classification, image-validity and height-validity remain distinct.
No reference labels/masks are passed to inference. No water/road zero rule.

## Controlled comparison and resources

Both arms restart from the same protected initialization and seed, not failed
candidate weights. A receives uniform probabilities; B frozen RGB-predicted
six-class probabilities. Both receive the SAME new loss. Compare each with its
historical same-epoch old-loss result, and B with its new same-epoch A.

Exactly two epochs/arm, 600 source-paired batches per epoch: all 450 corrected
HighBuild training scenes plus 150 urban repeats, and all 600 OpenCanopy scenes.
Original batch 2, accumulation 2, 384 px crops, bf16, optimizer, learning rates,
sample order and frozen model/semantics remain unchanged. Four-worker ordered
CPU prefetch and pinned transfers preserve the measured faster input path.
One GPU experiment only; no overclocking, cooling overrides or unrelated kills.

All 160 HighBuild and 120 OpenCanopy development images are evaluated at native
resolution after every epoch on the same supported pixels. Reuse only fully
authenticated probability/prior receipts. New immutable experiment on D:,
source/config/data/software binding, optimizer/RNG checkpoints every 50 updates,
and atomic commits. Resume refuses changed bindings. The existing progress
window follows the new active registration automatically.

## Acceptance and stopping

Original safety thresholds are UNCHANGED, including regional groups and
correlation/R2 checks. Candidate benefit still requires >=0.5 m AND >=5% building
or canopy RMSE reduction from protected, beating same-epoch matched control.
Report safety and benefit separately, all four epochs, and best unrestricted
versus eligible checkpoint. No silent adjustment to gates after seeing results.
No automatic additional epochs, repeated hyperparameter search, or promotion.

GAMUS height units remain unresolved. No reserved geography is consumed, no
universal accuracy claim, and no height validation claimed for roads/water.
App-path, per-building and final-geography checks remain separate prerequisites
if a candidate eventually passes. The protected production model, live pointer,
V1/V3 classifiers and all earlier runs are preserved and hashed.
