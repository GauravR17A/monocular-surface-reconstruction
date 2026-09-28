# GAMUS six-class v4: isolated hierarchical vegetation head

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: architecture, loss, configuration builder, and fail-closed safety
auditors are implemented and tested. V4 is not yet trained, so none of those
checks establish its accuracy. This experiment does not authorize changing the
app model or production pointer.

## Why a v4 would be justified

The completed experiments show a narrow failure rather than a general collapse:

| Candidate | Macro F1 | Water F1 | Road F1 | Low-vegetation F1 | Tree F1 |
|---|---:|---:|---:|---:|---:|
| Joint v1, epoch 4 | 0.4673 | ~0.0000 | 0.5171 | 0.5730 | 0.6211 |
| Frozen linear v2, best macro epoch 3 | 0.5055 | 0.2201 | 0.6165 | 0.5184 | 0.4668 |

At v2 epoch 3, 30.7% of true tree pixels became low vegetation and 14.5%
became water. Tree precision was 0.6153 but recall was only 0.3760. At epochs
4 and 5 tree recall fell further, to 0.2791 and 0.2945, while precision stayed
high. The classifier is therefore conservative about predicting trees; this is
not evidence that the labels contain no vegetation signal. The joint v1 run
learned the vegetation split, but its shared-feature changes caused unacceptable
height regression. V4 should recover that split without touching shared or
height tensors.

## Frozen input available to the semantic head

The optional six-class head receives the existing 64-channel `adapter_features`.
Those frozen features are computed from six input planes:

1. RGB image (3 planes),
2. cached Depth Anything V2 relative-depth prior (1 plane),
3. log-normalized protected base height (1 plane), and
4. protected building probability (1 plane).

The current linear v2 head is a 1x1 convolution over these features (390
parameters), so it makes each decision without explicit neighbourhood context.
V3 adds a depthwise 3x3 residual refinement and has 1,094 head parameters. If
that local context still does not recover both vegetation classes, the smallest
useful next change is to factor the same six-class prediction into a coarse
decision plus a dedicated vegetation subtype decision.

## Proposed architecture

Name: `hierarchical_vegetation`.

All modules and parameters must exist strictly below the one allowed prefix:

`fine_semantic_head.`

The head is:

1. `spatial`: depthwise 3x3 convolution, 64 groups, no bias.
2. `normalization`: GroupNorm with one group, followed by GELU.
3. Residual feature: `h = x + GELU(GN(spatial(x)))`.
4. `coarse_classifier`: 1x1 convolution from 64 channels to five logits:
   ground, buildings, water, roads, and combined vegetation.
5. `vegetation_split`: 1x1 convolution from 64 channels to one logit, giving
   tree versus low-vegetation probability only inside the vegetation branch.
6. Convert the five-way coarse probabilities and binary split into six aligned
   probabilities:
   - the first four probabilities are unchanged;
   - `P(low vegetation) = P(vegetation) * (1 - P(tree | vegetation))`;
   - `P(trees) = P(vegetation) * P(tree | vegetation)`.
   Their clamped log-probabilities are returned as the six logits, so the
   existing softmax/export interface remains six-class compatible.

This factorization prevents the low-vegetation/tree objective from competing
with water, roads, buildings, and ground for the same single flat decision. It
also preserves the total vegetation probability exactly while deciding how to
split it between low vegetation and trees.

### Parameter count

For 64 frozen input channels:

| Tensor group | Parameters |
|---|---:|
| Depthwise 3x3 | 64 x 3 x 3 = 576 |
| GroupNorm scale and bias | 64 + 64 = 128 |
| Five-way 1x1 classifier | 5 x 64 + 5 = 325 |
| Vegetation split 1x1 classifier | 1 x 64 + 1 = 65 |
| **Total** | **1,094** |

V4 therefore has the same parameter count as v3 and only 704 more parameters
than the v2 linear head. The improvement comes from the task structure, not a
larger shared model.

## Initialization

- Start again from the protected production height checkpoint for a clean,
  attributable comparison; do not warm-start or modify any shared tensor.
- Initialize the depthwise convolution to zero, GroupNorm scale to one and bias
  to zero. The residual path initially behaves as the unchanged frozen feature.
- Initialize both classifier weights with a tiny zero-mean normal distribution
  (standard deviation 0.001) so the spatial branch can receive gradients on the
  first optimizer step without making strongly biased predictions.
- Initialize the binary split bias to zero (equal low-vegetation/tree prior).
- Initialize the combined-vegetation coarse bias to `ln(2)` and the other four
  coarse biases to zero. With a zero split logit, this yields an exactly uniform
  six-class initial probability rather than accidentally giving each vegetation
  subtype only half the mass of every other class.

## Minimal training change

Keep v3's sealed data, raw radiometry, water-aware sampler, six-class focal loss,
optimizer, and all height loss weights at zero. Add only one auxiliary loss:

- On pixels whose valid six-class target is low vegetation or trees, apply an
  equal-weight binary cross-entropy to `vegetation_split`.
- The mask must be `image_valid AND classification_valid AND target in {4,5}`.
  It must never use `height_valid`, and ignored/background pixels remain absent.
- Suggested auxiliary weight: 0.5 relative to the existing six-class loss.
- Do not change water sampling, class weights, colour augmentation, or decision
  biases in this architecture comparison. Those would confound the result.

The equal-weight subtype objective directly addresses v2's tree under-recall,
which the current global class weights `[0.7, 0.6, 3.0, 0.7, 0.6, 0.4]` can
otherwise encourage. If a later decision-bias calibration is used, it must be
fitted only after selecting/fixing the model and only on the development
geography; it is not confidence calibration.

For the controlled architecture comparison, retain the paired spatial-V3
schedule exactly: 8 epochs, validation every epoch, minimum 4 epochs, and
early-stop patience 3. Changing the schedule would confound the comparison.

## Exact checkpoint and height-safety contract

The optimizer allowlist must contain exactly `fine_semantic_head.`. Every other
parameter must have `requires_grad=False`; no other prefix is permitted. GroupNorm
is used because it has no running statistics. No BatchNorm, stochastic shared
state, or height-dependent post-processing may be added.

After every saved candidate, audit all state entries not starting with
`fine_semantic_head.` using exact `torch.equal`, not a floating-point tolerance.
The inherited tensor key set, shapes, dtypes, and values must exactly match the
protected checkpoint. The production checkpoint file and live pointer hashes
must also match their pre-run hashes. Any mismatch makes the artifact invalid,
regardless of metrics.

Weight equality alone is insufficient. The separate height-output audit must
also prove that canonical preprocessing, tiling, fusion settings, all direct
height-path outputs, and deterministic app-level outputs are bit-identical to
the protected model. Its sealed pipeline contract is SHA-bound to the V4
configuration.

## Selection and acceptance gates

Freeze the paired DC+Philadelphia V3 development metrics before launching V4.
The configuration builder derives and seals the numeric gates from that exact
audit before training begins. Training-time height guards and macro-F1
selection remain identical to paired V3. The selected candidate is eligible
for the classifier-excluded-city check only when all of the following pass:

1. Exact inherited-tensor, height-output, pointer, and checkpoint hash audits pass.
2. Existing GAMUS height guards pass at the current 0.0001 m reporting
   tolerance; exact output equality is the stronger requirement.
3. Final six-class macro F1 is at least `max(0.60, paired_v3_macro_f1)`.
4. Ground F1 is at least `max(0.40, paired_v3_ground_f1 - 0.005)`.
5. Building F1 is at least `max(0.82, paired_v3_building_f1 - 0.005)`.
6. Water F1 is at least `max(0.45, paired_v3_water_f1 - 0.005)`.
7. Road F1 is at least `max(0.65, paired_v3_road_f1 - 0.005)`.
8. Low-vegetation F1 is at least
   `max(0.58, paired_v3_low_vegetation_f1 + 0.010)`.
9. Tree F1 is at least `max(0.62, paired_v3_tree_f1 + 0.010)`.
10. Road-boundary F1 is no more than 0.005 below paired V3, and the
    dark-non-water false-water rate is no more than 0.010 above paired V3.

Select checkpoints by final six-class macro F1, matching paired V3. The smaller
of low-vegetation and tree F1 remains a diagnostic only. Recompute every stored
score from the exact 6x6 confusion matrix; a five-group vegetation merge can
never satisfy the macro gate.

Report precision, recall, F1, and IoU for all six classes, the full confusion
matrix, water/shadow proxy, road-boundary metrics, and city/geography breakdown.
Only a fully eligible frozen candidate may receive one NYC check. NYC is
excluded from this classifier head's fitting and selection, but it is not
system-unseen because earlier GAMUS and inherited HighBuild work exposed New
York data. The official GAMUS test was also consumed by an older experiment and
is forbidden for reuse. A separately sourced, preregistered geography remains
required for a genuine system-level generalisation claim.

## Risks and stop rule

- The frozen adapter may simply not encode enough texture/scale information to
  separate grass from tree canopy across cities. A head cannot recover absent
  information.
- Water oversampling may still pull dark trees toward water. The hard
  water/shadow gate prevents hiding that trade-off behind macro F1.
- The hierarchy can improve the vegetation split while worsening the earlier
  combined-vegetation decision; the per-class and macro gates prevent this.
- Raw RGB radiometry can shift geographically, so DC+Philadelphia success and
  the classifier-excluded NYC check do not replace a new external geography.
- GAMUS labels identify land cover, not vegetation height. Success here must
  never be presented as proof of canopy-height accuracy.

If v4 cannot pass both vegetation F1 gates, do not unfreeze the production
adapter or height trunk. The next scientifically defensible escalation is a
separate semantic-only sidecar encoder whose entire state remains under an
isolated prefix; it should be evaluated as a new experiment rather than risking
the working height model.
