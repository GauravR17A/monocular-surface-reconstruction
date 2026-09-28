# GAMUS six-class experiment status

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Evidence snapshot: 2026-09-13. The live application still points to the
protected height checkpoint. No GAMUS experiment has been promoted.

## Current decision

- The height-focused pilot is complete and sealed. It is not eligible for
  promotion because its ground-height and building-identification guards fail.
- The first joint six-class candidate is also complete. It learned five useful
  categories, but water recall collapsed to effectively zero and its
  tall-vegetation guard failed.
- The corrected full-scene HighBuild/OpenCanopy triplet benchmark completed
  successfully as an evaluation job, but its final decision is **failed**: the
  height pilot and joint six-class candidate are materially worse than the
  protected model on the corrected forest evidence. This is an acceptance-gate
  failure, not a crashed benchmark.
- The isolated linear six-class-head correction completed and was rejected for
  promotion. Its best checkpoint was epoch 3; all 240 inherited tensors
  (31,428,489 values) were exactly equal to the protected model, but ground,
  low-vegetation and tree retention gates failed.
- A development-only additive decision calibration materially improved the
  same frozen-height output, but still missed the original low-vegetation and
  tree retention thresholds. It is not probability calibration and has not
  been exposed in the app.
- The sealed spatial-refined six-class head completed after epoch 6. Epoch 2
  is the best raw development checkpoint at macro F1 0.554, but the original
  low-vegetation and tree retention gates still fail. Exact inherited-state
  and recipe audits pass; it is not promoted.
- The sealed DC+Philadelphia decision-bias check completed. It improved the
  frozen classifier's development trade-off without changing weights or height
  output, but still failed the low-vegetation/tree acceptance floors and remains
  inactive.
- The application model must remain unchanged. The fair next comparison trains
  V3 and V4 classifier heads on the same DC+Philadelphia split. NYC may then be
  described only as excluded from those new heads' training—not as system-unseen.
  A new external geography, corrected height checks, category gates, and
  explicit human approval are still required before exposure.

## Requirement-by-requirement audit

| Expanded requirement | Status | Repository evidence and honest limitation |
|---|---|---|
| Six GAMUS categories in a separate output | Implemented and evaluated | `DomainGatedSurfaceNet.fine_semantic_head` outputs ground, buildings, water, roads, low vegetation and trees while old checkpoints remain loadable. The v1 report is `outputs/evaluation/gamus_six_class_comparison_v1/report.json`. |
| Preserve three-group height machinery | Implemented and bitwise verified | Here “three-group” means the existing ground/building/vegetation **height-routing domains**, not a three-class replacement for the six-class identifier. The additive identifier still reports all six classes. Both v2 and v3 audits proved exact equality for all 240 inherited tensors/31,428,489 values. V3 trained only its five `fine_semantic_head.*` tensors and kept every height loss at zero. |
| Independent image/classification/height masks | Implemented and audited | `GAMUSDataset` emits `image_valid_mask`, `classification_valid_mask`, `height_valid_mask` and a separate `regression_mask`. Water/roads may teach class identity but are excluded from height regression; missing height never becomes zero. |
| Keep the production height checkpoint safe | Implemented; candidate gates failed | The live pointer and protected checkpoint were hash-identical before/after the corrected triplet, completed v2 run and decision-calibration evaluation. No experimental result has been promoted. |
| HighBuild individual outlines, heights and boundaries | Implemented for available annotations | `src/msr/evaluation/highbuild_instances.py` reports per-building matches, height error, outline IoU and boundary scores, separating measured and estimated annotations. The corrected benchmark covered 160 validation tiles and 2,588 source annotations. |
| Building counting as a separately validated feature | Implemented fail-closed; current evidence unavailable | The evaluator has a distinct per-scene counting section (MAE/RMSE, exact-count rate, over/undercount totals and MAPE when valid). Current positive-only labels cannot support honest precision/count error, so those fields remain null until an exhaustive independent instance reference exists. |
| OpenCanopy dates, locations and source provenance | Implemented | `outputs/data_audits/open_canopy_v2_provenance_v1.json` authenticates 600/120/120 accepted samples with region, source image, imagery date, LiDAR date, URL, GSD and metadata hash. It also discloses source-mosaic overlap across the present splits. |
| Richer vegetation classes | Partial | Low/medium/high upstream vegetation strata can be preserved in versioned sidecars without changing height masks. Full sidecar backfill, training and held-out evaluation remain pending; these are LiDAR height strata, not species labels. |
| GAMUS units, pixel scale and >200 m outliers | Audited with one unresolved proof | Nominal 0.33 m pixels and all >200 m values are inventoried in `outputs/data_audits/gamus_preparation_v1/`. Local HDF5 lacks CRS/transform and the publisher has not independently confirmed the raw AGL unit, so results remain metre-assumed rather than absolute-unit-proven. |
| DAV2 cache/source integrity | Implemented for current cache | All 8,720 approved RGB/prior pairs were hash-locked and five samples reproduced exactly. Historical generation did not embed hashes at creation, so the audit states that limitation rather than inventing provenance. |
| Per-class precision, recall, IoU and F1 | Implemented and reported | Metrics cover all six classes plus a confusion matrix. Road boundaries have a tolerance-aware metric. The dark-pixel check is explicitly only a water/shadow-confusion proxy because GAMUS has no shadow ground truth. |
| Geographic generalisation | Classifier-head exclusion prepared; genuinely new geography pending | A predeclared NYC partition and DC+Philadelphia learning index exist in `outputs/data_audits/gamus_geographic_holdout_city_nyc_v1/`. The spatial recipe passed a 3,837-train/859-validation/1,164-NYC preflight without opening NYC for this comparison. NYC is not system-unseen: older GAMUS work and the inherited HighBuild-trained system already exposed New York data. The official GAMUS test was consumed by the older Stage-3 audit and is forbidden for reuse. Final evidence must come from a separately reserved external geography. |
| Six-class overlays and summaries in the app | Validated-export path implemented; app exposure pending | An inactive fail-closed exporter can write aligned class IDs 0..5/nodata 255, separate validity masks and allowed summaries only from a SHA-bound fully accepted candidate. The active app checkpoint has no six-class head and no overlay is enabled. |
| Calibrated confidence | Pending | Additive argmax decision biases were fitted and evaluated, but this is explicitly not probability/confidence calibration. Existing confidence-like outputs must not be presented as probability of correctness. |
| Flood and snow extensions | Explicit future modules, not current claims | Water identification can support flood work, but a flood simulator still needs terrain and hydraulic inputs. Snow cover needs suitable class labels; snow depth needs separate metric supervision. Infrared and before/after change detection also require new preparation. |

## Completed corrected legacy triplet

The hash-sealed report is
`outputs/evaluation/corrected_legacy_triplet_v1/report.json` (SHA-256
`0977bc90e034746df74dc6241c6676f15a6ea5f36dc0b422008a75c6767ab351`).
It used identical raw RGB, cached target-independent relative priors, masks and
full native grids for all three models: 160 corrected HighBuild scenes and 120
OpenCanopy scenes. This triplet job did not use the official GAMUS test; that
test had already been consumed by the older Stage-3 final audit and cannot be
reused by V4.

Primary strict-measured pooled height result:

| Model | Pooled RMSE (m) | Pooled MAE (m) | Correlation | R-squared | OpenCanopy all-valid forest RMSE (m) | OpenCanopy vegetation-only RMSE (m) | HighBuild strict-measured building-only RMSE (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Protected live model | 10.161 | 4.911 | 0.482 | 0.203 | 6.707 | 8.361 | 13.243 |
| Height-focused pilot | 15.460 | 10.857 | -0.139 | -0.846 | 16.919 | 10.893 | 13.435 |
| Joint six-class candidate | 15.463 | 10.837 | -0.150 | -0.847 | 16.910 | 10.802 | 13.457 |

These are the canonical corrected full-scene baselines. The three protected
values answer different questions and are not interchangeable: **13.243 m** is
building-only error on strict measured HighBuild pixels, **8.361 m** is
vegetation-only error on valid OpenCanopy vegetation pixels, and **6.707 m** is
all-valid OpenCanopy forest-scene error including lower/ground pixels. Earlier
crop, zero-filled, inclusive-annotation, or mixed pooled scores must retain their
own protocol labels and must not be compared as if they measured the same task.

The joint candidate passes the small non-regression deltas versus the already
weak height pilot, but fails the predeclared comparison with the protected
model: pooled RMSE is +5.302 m, OpenCanopy/forest RMSE is +10.202 m, vegetation
RMSE is +2.441 m and corrected HighBuild building RMSE is +0.214 m.

HighBuild instance evidence reinforces the no-promotion decision. The protected
model matched 400/2,581 rasterizable reference buildings (annotated recall
0.155), while the joint candidate matched 282/2,581 (0.109). Mean boundary F1
was 0.488 versus 0.456 and mean matched-outline IoU was 0.738 versus 0.709. The
candidate's lower RMSE on only its matched subset must not be treated as an
overall win because it matched substantially fewer buildings.

## Six-class development result

The joint candidate's GAMUS development result was RMSE 4.442 m, MAE 2.604 m,
correlation 0.783 and R-squared 0.613. Six-class macro F1/IoU were 0.467/0.330.
“Six-class macro” always means the arithmetic mean over the final six classes;
merging low vegetation and trees into one vegetation class produces a different
five-group statistic and cannot be reported as the six-class score.
Per-class F1 was 0.402 ground, 0.690 buildings, effectively 0.000 water, 0.517
roads, 0.573 low vegetation and 0.621 trees; road-boundary F1 was 0.257.

The completed linear head-only correction addressed the concrete imbalance
behind the water failure (water is only 3.70% of training pixels) with
class-weighted focal loss and a hash-sealed water-aware sampler. Its best epoch
3 achieved macro F1/IoU 0.506/0.362. Per-class F1 was 0.379 ground, 0.833
buildings, 0.220 water, 0.617 roads, 0.518 low vegetation and 0.467 trees;
road-boundary F1 was 0.270. The checkpoint audit passed exact equality for all
inherited height/shared tensors, but the frozen classification acceptance gate
failed ground, low-vegetation and tree retention. It was not promoted.

A one-time full DC+Philadelphia development evaluation of external additive
decision biases improved macro F1/IoU from 0.506/0.362 to 0.547/0.395 and
accuracy from 0.543 to 0.578. Calibrated per-class F1 was 0.413 ground, 0.842
buildings, 0.324 water, 0.622 roads, 0.522 low vegetation and 0.557 trees;
road-boundary F1 was 0.273. The dark-non-water false-water rate fell from 0.371
to 0.126. This is a development-set class-decision adjustment, not probability
calibration or unseen-geography evidence. It still misses the original
low-vegetation and tree retention targets and remains inactive.

The spatial-refined comparison kept the same data, sampler, losses, seed and
gates while replacing only the isolated six-class head with a 1,094-parameter
local-context head. It early-stopped after epoch 6 with epoch 2 best: macro
F1/IoU 0.554/0.400 and accuracy 0.581. Per-class F1 was 0.409 ground, 0.840
buildings, 0.386 water, 0.630 roads, 0.483 low vegetation and 0.575 trees;
road-boundary F1 was 0.287 and the dark-non-water false-water rate was 0.101.
All height metrics were identical to the protected initialization and all
31,428,489 inherited values passed exact equality. The raw candidate still
misses the original low-vegetation (0.568) and tree (0.616) retention floors,
so it remains inactive. Evidence is sealed at
`outputs/evaluation/gamus_six_class_spatial_refined_v3/checkpoint_best_landscape_audit.json`.
