# Monocular Surface Reconstruction expanded GAMUS plan

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: protocol predeclared before training; implementation and run status are
updated below without changing the frozen comparison rules.

## Non-negotiable baselines

- The production pointer remains unchanged. Its protected checkpoint SHA-256 is
  `e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144`.
- The existing three height-routing-group (ground/building/vegetation)
  direct-height pilot remains the height-focused comparison. It is not
  redefined after results are seen.
- The expanded model is a separate candidate. It cannot auto-promote.
- GAMUS train uses the full QC-approved split: 5,001 of 5,004 official tiles.
  All 859 approved validation tiles stay separate. The official GAMUS test was
  consumed once by the older Stage-3 final audit and is permanently excluded
  from all new training, tuning, threshold selection, model selection, and
  repeat evaluation.
- NYC is excluded from the new DC+Philadelphia classifier heads, but it is not
  system-unseen: earlier GAMUS experiments and the inherited HighBuild-trained
  checkpoint exposed New York data. A separate external geography remains the
  final generalisation holdout.

## Work and status

| Requirement | Current state | Next proof required |
|---|---|---|
| Full usable GAMUS data | Implemented: immutable QC index, 5,001 train and 859 validation | Treat the already-consumed official test as immutable historical evidence; never reuse it |
| Three-group metric-height machinery | Implemented and checkpoint-compatible; frozen comparison early-stopped after epoch 4 and is hash-sealed. “Three-group” means height-routing domains (ground/building/vegetation), not a three-class identification score. | Keep its raw result as the control; it is not promotion-eligible |
| Six GAMUS categories | Implemented and evaluated as a separate head: ground, building, water, road, low vegetation, tree; the isolated candidate early-stopped after epoch 4 | Improve water recall and ground/building/tall-canopy guards before any app exposure |
| Ignore background/unknown | Implemented in grouped and six-class losses/metrics with ignore index 255 | Verify full-validation support counts |
| Independent validity masks | Implemented and audited: explicit image-valid, class-valid, and height-valid tensors; regression intersects all required evidence | Keep colour transforms tied only to image validity |
| Water/road identification without unreliable height | Implemented and tested: both are classification-only and excluded from regression; v1 development metrics exposed a near-total water collapse | Compare V4 with paired V3 on the same DC+Philadelphia split, then report water/road metrics on a newly reserved external geography |
| Height protection | Production checkpoint and pointer protection implemented; the corrected legacy triplet completed and rejected both experimental height systems versus the protected model | Finish the independent tensor audit for the active head-only correction; do not promote automatically |
| Corrected HighBuild supervision | Implemented and run: two immutable pixel protocols plus per-building height, outline/boundary, support, and measured/estimated slices across 160 validation scenes | Improve annotated-building recall and obtain exhaustive negative/instance labels before claiming precision or count accuracy |
| OpenCanopy provenance and richer vegetation labels | Provenance implemented: accepted 600/120/120 manifests have authenticated dates, regions, source scenes and URLs; code and a smoke sample preserve upstream low/medium/high vegetation IDs; a 477/123 region+source-scene holdout plan is sealed | Run the full sidecar backfill and use the stronger holdout before richer-class training |
| GAMUS unit/pixel-scale audit | Implemented in `gamus_preparation_v1`: nominal 0.33 m pixels documented; local georeferencing absent; AGL metres remain an explicit unverified assumption | Obtain publisher/source-product confirmation before claiming absolute unit proof |
| DAV2 prior authentication | Implemented: all 8,720 approved pairs passed; current bytes are hash-locked; five train/validation samples reproduced exactly | Future caches must embed source/model/config hashes at creation time |
| Six-class evaluation | Implemented and reported on development validation: per-class precision/recall/IoU/F1, exact 6 x 6 confusion matrix, road boundaries, and water-vs-dark proxy | Recompute macro scores from all six final classes; do not substitute the easier five-group score formed by merging low vegetation and trees |
| Reviewer/app outputs | Six-map tiled inference support exists, but no GAMUS candidate is enabled in the active app | After validation: wire a six-class overlay, height summaries, coverage, and scale-aware area |
| Confidence | Not calibrated | Do not show probability-of-correctness until calibrated on held-out geography |
| Flood and snow modules | Future, explicitly separate | Flood needs terrain/hydraulics; snow cover needs labels; snow depth needs metric supervision |

## Common preparation contract

Both experiments must use the same approved IDs, radiometry, crop policy,
cached relative priors, masks, height units, and validation code. Each sample
has three independent masks:

1. `image_valid_mask`: pixels with usable optical input;
2. `classification_valid_mask`: pixels with a supported class in 1..6;
3. `height_valid_mask`: finite, physically accepted metric AGL labels.

The regression mask is `height_valid_mask` intersected only with ground,
building, low-vegetation, and tree classes. Water and road remain valid for
classification and invalid for height regression. Missing/ignored labels never
become zero-height evidence. Colour augmentation stays disabled until this
contract is verified.

The current six-epoch development protocol visits every one of the 5,001
approved training tiles once per epoch, using one deterministic-random 384 px
crop from each tile. It does not claim that every source pixel is seen in six
epochs. Validation processes all 859 approved 1024 px tiles.

## Experiment order

1. Complete unit, pixel-scale, >200 m, prior-integrity, and validity audits.
2. Run the unchanged three-height-routing-group pilot and freeze its artifacts.
3. Initialize a separate six-class candidate from the same protected checkpoint.
4. Train it with the same height objective plus a six-class auxiliary loss.
5. Evaluate the protected model, height pilot, and expanded candidate with
   identical masks and inputs.
6. Run corrected HighBuild per-instance/boundary checks and OpenCanopy/forest
   regression checks. **Complete: the candidate failed against the protected
   model, so promotion remains blocked.**
7. Run the isolated six-class-head-only correction and verify bit-for-bit that
   every inherited height/shared tensor remains equal to the protected model.
8. Train paired V3 and V4 classifier heads on the same sealed DC+Philadelphia
   learning partition. A one-time NYC evaluation may measure transfer by those
   new heads, but must be labelled classifier-head-excluded rather than
   system-unseen.
9. Evaluate a separately reserved external geography that has not influenced
   the shared model, height pipeline, classifier, thresholds, or case selection.
10. Only after all classification, height, external-domain and genuinely new
   geography gates pass should app exposure be considered.

## Required reports

Height is reported in metres as RMSE, MAE, bias, correlation and R-squared for
overall, city, ground, building, vegetation, tall building and tall vegetation.
The expanded candidate must not exceed the predeclared regression limits used
by the height pilot, and it must be compared directly with the height pilot,
not merely with historical scores.

Identification is reported independently for every one of the six categories:
precision, recall, IoU, F1, support and the full confusion matrix. Water-shadow
confusion and road boundary quality receive named slices. Empty or unsupported
classes are reported as unavailable, never silently treated as perfect.
The six-class macro values are recomputed from the exact final 6 x 6 matrix.
Combining low vegetation and trees creates a five-group analysis and must be
labelled separately; it is never a replacement for six-class macro F1.

HighBuild is evaluated per building as well as per pixel: matched-instance
height MAE/RMSE/bias, outline/boundary score, detection/count precision and
recall, and separate measured versus estimated-height annotations. Square-metre
coverage is emitted only when CRS/pixel scale is authenticated.

## Promotion rule

The expanded candidate result is complete but **not promotion-eligible**. The
live model changes only after corrected HighBuild/OpenCanopy and genuinely new
external-geography evidence is supplied, every required guard passes, and a
human explicitly approves the change. NYC alone cannot satisfy that requirement.

## Frozen height-focused result

The control early-stopped after epoch 4. Its best raw validation result was
RMSE 4.471 m, MAE 2.646 m, bias +0.320 m, correlation 0.781 and R-squared
0.608. Domain RMSE was 2.497 m ground, 4.542 m building and 5.037 m
vegetation; tall-building and tall-vegetation RMSE were 12.393 m and 10.367 m.
It is not promotion-eligible: ground RMSE regressed beyond the frozen limit and
building identification F1 was 0.631 versus the 0.815 protected baseline. The
exact epoch-4 result remains reproducible through `checkpoint_latest.pt`; the
production pointer was verified byte-identical after completion.

## Expanded six-class result

The isolated candidate early-stopped after epoch 4. Its GAMUS development
result was RMSE 4.442 m, MAE 2.604 m, bias +0.166 m, correlation 0.783 and
R-squared 0.613. Against the height-focused control it improved overall RMSE by
0.029 m, ground RMSE by 0.113 m and vegetation RMSE by 0.030 m; building RMSE
regressed by 0.020 m and tall-vegetation RMSE regressed by 0.286 m. The latter
exceeds the frozen 0.20 m cap.

Six-class macro F1/IoU were 0.467/0.330. Per-class F1 was 0.402 ground, 0.690
buildings, effectively 0.000 water, 0.517 roads, 0.573 low vegetation and 0.621
trees. Road-boundary F1 was 0.257. The near-zero water recall is a clear failed
category, and the low water-vs-dark false-positive proxy is not evidence of
good water detection because the model almost never predicts water.

The comparison report is saved at
`outputs/evaluation/gamus_six_class_comparison_v1/report.json`. A strict
reporting check initially rejected harmless independent-CUDA accumulation drift
(maximum 9.9e-6); it was corrected to a recorded 1e-4 fail-closed numerical
tolerance, covered by 375 passing tests, without resuming training.

## Current comparison limitations

The official GAMUS train and validation IDs do not overlap, but both cover DC
and Philadelphia. In DC, 350 of 359 validation tiles are directly adjacent to
a training tile in the four-neighbour grid, and all 359 are adjacent under an
eight-neighbour test. Current validation is therefore a useful development
comparison, not the final genuinely new-geography claim. NYC exclusion is also
only a classifier-head check because other inherited system components have New
York exposure.

The production baseline is evaluated with its deployed `protected_vegetation`
fusion while both experimental candidates use the same `legacy` fusion. This
is correct for deciding whether an entire candidate system is safer than the
live system, and the height-pilot versus six-class ablation remains controlled.
It does mean that protected-versus-height-pilot gains cannot be attributed to
fine-tuning alone without an additional untrained legacy-fusion baseline.

## Corrected HighBuild/OpenCanopy triplet result

The full corrected comparison is complete at
`outputs/evaluation/corrected_legacy_triplet_v1/report.json`. Its `status` is
`failed` because the acceptance gates failed, not because execution crashed.
All three models used identical full-scene inputs and supervision. This report
did not use official GAMUS test data; the older Stage-3 final audit had already
consumed that test, so it is forbidden for reuse. The report is complete and the
live pointer/checkpoint stayed unchanged.

Under the primary strict-measured protocol, pooled protected/height-pilot/joint-
candidate RMSE was 10.161/15.460/15.463 m. OpenCanopy all-valid forest RMSE was
6.707/16.919/16.910 m, OpenCanopy vegetation-only RMSE was
8.361/10.893/10.802 m, and corrected HighBuild strict-measured building-only
RMSE was 13.243/13.435/13.457 m. The joint candidate was approximately neutral versus
the height pilot but failed every protected-system aggregate/forest guard and
also exceeded the protected HighBuild building-RMSE allowance by 0.064 m beyond
the permitted +0.15 m regression.

The protected model's canonical corrected full-scene baselines are therefore
13.243 m for HighBuild building-only, 8.361 m for OpenCanopy vegetation-only,
and 6.707 m for OpenCanopy all-valid forest. These scopes have different masks
and supports and are not interchangeable. Historical crop-based, zero-filled,
inclusive-annotation, or pooled metrics remain useful only under their original
labels.

The evaluator also produced per-building and boundary evidence. The protected
model matched 400 of 2,581 rasterizable annotations versus 282 for the joint
candidate; annotated recall was 0.155 versus 0.109, boundary F1 0.488 versus
0.456, and matched-outline IoU 0.738 versus 0.709. Precision and count error
remain deliberately unavailable because HighBuild supplies positive outlines,
not exhaustive negative/instance truth.

## Isolated classification correction result

`configs/multidomain_surface_gamus_six_class_head_only_v2.yaml` completed after
epoch 6 with epoch 3 as the best development checkpoint. It was initialized
from the protected checkpoint, permitted only `fine_semantic_head.` to train,
set every height and three-height-routing-group loss to zero, and used a hash-sealed
water-aware sampler plus class-weighted focal loss.

The independent audit at
`outputs/evaluation/gamus_six_class_head_only_v2/checkpoint_best_audit.json`
proved exact equality for all 240 inherited tensors (31,428,489 values). Best
macro F1 was 0.506; class F1 was 0.379 ground, 0.833 buildings, 0.220 water,
0.617 roads, 0.518 low vegetation and 0.467 trees. Ground, low-vegetation and
tree retention failed, so it was not promoted.

An external additive-bias development calibration then improved full
DC+Philadelphia macro F1 to 0.547 and class F1 to 0.413/0.842/0.324/0.622/
0.522/0.557 in the same class order. This changes argmax decisions only, not
height or probability calibration. It still misses the original
low-vegetation/tree retention targets and is not unseen-geography evidence.

The sealed spatial-refined head comparison completed after epoch 6. Epoch 2 was
best at macro F1/IoU 0.554/0.400, with class F1 0.409/0.840/0.386/0.630/
0.483/0.575 for ground/buildings/water/roads/low vegetation/trees. Its exact
audit proves all 240 inherited tensors and 31,428,489 inherited values stayed
bitwise equal to the protected model. It improves the isolated classifier but
still fails the original low-vegetation and tree retention gates, so it remains
inactive.

A decision-only calibration protocol is sealed for that best checkpoint. It
can adjust only six external argmax offsets, not weights, heights or confidence,
and requires a no-regression sampled gate before one authoritative full
DC+Philadelphia pass. A spatial DC+Philadelphia-only learning recipe has also
passed preflight for 3,837 learning tiles, 859 epoch-selection tiles and one
  locked 1,164-tile NYC evaluation; no NYC imagery was opened by preflight. That
  run provides a fair classifier-head comparison and, at most, a one-shot
  classifier-excluded NYC result. It cannot support a system-unseen claim or
  replace the separate external-geography evidence required for app promotion.

## OpenCanopy provenance result

`outputs/data_audits/open_canopy_v2_provenance_v1.json` passes with no accepted
cross-split region overlap. It indexes 600 training, 120 validation, and 120
locked-test samples, including official source identity, EPSG:2154 location,
SPOT imagery date, LiDAR acquisition date, source URLs, GSD and metadata-file
hash. Extra materialized candidates left behind during streaming selection are
reported separately and are not treated as accepted training/evaluation rows.
The accepted splits do share many large SPOT source mosaics: 57 train/validation,
49 train/test and 44 validation/test source identities overlap. Therefore this
is geographic-region isolation, not source-acquisition isolation. A deterministic
region-plus-source-scene plan now produces 477 learning and 123 held-out samples
and is the recommended stronger comparison. Exact upstream classification IDs
can also be written to separate, versioned sidecars without changing the current
height masks; low, medium and high vegetation are LiDAR height strata, not tree
species. A real 384-pixel smoke sample passed. Full backfill and training remain
pending so the current OpenCanopy artifacts stay untouched.

## HighBuild instance-contract result

The per-building evaluator verified all 160 corrected validation tiles and
2,588 COCO annotations. Seven valid outlines smaller than the raster pixel grid
are disclosed and excluded rather than fabricated; 1,133 annotations without
the optional estimated flag are also disclosed under the frozen contract.
Matched-building height and outline metrics are now supported. Because the
current HighBuild background is unknown rather than exhaustively labelled
non-building, detection precision and count error deliberately remain
unavailable until an independent exhaustive mask or instance reference exists.
