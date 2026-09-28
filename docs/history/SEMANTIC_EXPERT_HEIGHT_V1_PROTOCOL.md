# Six-class soft-routed height corrections: prospective experiment

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

19 September 2026. Prompted by completed sampler and conditioning diagnostics.
No earlier candidate is promoted; earlier experiments and protected app remain.

## Explicit hypothesis

The previous model produced one correction field from concatenated class inputs.
It used classes somewhat, but achieved only tiny additional development gains.
Test six learned correction outputs with frozen RGB class probabilities mixing
them at each native pixel. This gives classification a direct, inspectable path
to height without forcing any category to a hand-written height or using labels
at inference. Soft routing can still fail when classification is wrong. A small
training diagnostic found weak short-vegetation identification; this limitation
is not solved merely by adding the router.

Uniform-router A and predicted-six-class-router B have identical parameter
counts and initial states. Spatial features receive identical RGB, DAV2 and
protected-height context; ONLY the final mixing probabilities differ. Sum six
bounded signed corrections, then add to the frozen model's height and clamp the
result to nonnegative. Invalid imagery has no correction. Independent RGB
classifier and all protected semantic/height weights remain frozen. Semantic
labels are not height labels, missing height is ignored, roads/water are never
assigned zero by a category rule.

## Bounded training-only proof first

Reuse the authenticated fixed 20 TRAIN feasibility examples, source-bound DAV2
and classifier caches. Fresh model, 64 updates per arm, original loss/batch/LRs.
Require identical starting states/sample sequences, zero-start height identity,
unchanged protected tensors/semantic outputs, finite gradients, at least 10%
training-loss reduction per arm, and measurable actual-vs-uniform routing effect
after learning. Proof weights are discarded for the development run.
No retry or recipe change to turn a failed proof into a claimed pass.

## Development run only if proof passes

Four epochs per arm, 600 urban/forest pairs per epoch. ORIGINAL sampler in both,
not the unsuccessful height-band intervention: same sealed original plans/crops,
same original source-balanced loss, seed, batch 2, accumulation 2, learning rates,
384px patches and optimizer updates. Start both from the protected weights and
new zero correction heads. No augmentation. All 450 HighBuild +600 OpenCanopy
training images included each epoch. No training on development or final geography.

Recompute/check protected baseline over all 160 urban +120 forest native-size
DEVELOPMENT scenes, preserving exact support, original semantics, short/medium/
tall height bands, boundaries and per-region checks. Same strict safety rules as
previous run; do not loosen them after seeing results. Paired benefit requires
building OR canopy RMSE reduction >=0.5m AND >=5% from protected, beating the
same-epoch uniform-router control. Passing benefit alone is not acceptance.
Report best eligible separately from unrestricted equal building/canopy RMSE.

Hash-bound source/config/data/runtime, optimizer/RNG checkpoints each 50 updates,
atomic commit, source/input verification before and after. No automatic OOM
recipe/batch changes or competing GPU job. One compatible resume after inspection
is allowed for development, not a failed feasibility proof. No automatic app
promotion; per-building, app-path and a deliberately reserved final geography
check remain later release gates. GAMUS metric height, broad plains/hills and
road/water height accuracy remain uncertified. Six routes do not mean six newly
validated height categories.
