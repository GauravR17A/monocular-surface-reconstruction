# Six-class scanning and height-training audit — 19 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Shipped in the running app

Choose a scene/upload, click **Identify six classes**, then Explore → Fly or Walk.
Aim at a surface for two seconds or click to scan immediately. The primary
scanner label can now be Ground, Buildings, Water, Roads, Low vegetation or Trees.
Uncheck **Show on 3D surface** to keep the original photograph; classification
remains available to the scanner. Changing scenes clears the previous labels.

The scanner and Inspect panel distinguish classification source from height
source. An explicit disagreement warning appears when the RGB class conflicts
with the protected three-group height geometry. Classification never gates,
rescales, zeroes or replaces heights, collisions or exported rasters.

- nDSM: estimated surface height above local ground, not altitude.
- Building solid: protected model's building-object height summary.
- Calibrated DSM: surface elevation in its datum, not object height.
- Relative rDSM: dimensionless score, no metre claim.
- Water: surface value only; never water depth and not validated water height.
- Roads/ground: not automatically zero or sea-level elevation.

V3 epoch 1 remains explicitly **experimental**, because its development safety
gates failed. It has not become the production height model. A six-class label
does not establish class-specific height accuracy.

## Verification

- 89 viewer unit tests passed, including all six scanner labels, no-data,
  class disagreements, hidden overlays, relative units and DSM/object-height distinction.
- 25 targeted backend/audit tests passed; TypeScript and edited-file ESLint passed.
- Browser: actual CUDA classification of the Urban reference; colors turned off;
  Fly two-second aim and real mouse click both produce metadata; Walk aim produces
  the building class and 6.4 m object estimate. These are UI examples, not accuracy scores.
- Forest scene switch removed old class data; no browser errors recorded.
- Screenshot evidence: `outputs/runtime/six_class_scanner_click_20260919.png`
  and `outputs/runtime/six_class_scanner_walk_20260919.png`.
- Production checkpoint and live pointer SHA-256 values remain unchanged:
  `e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144`
  and `a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9`.

Next.js/React review kept the scanner client-side, retained asynchronous API
inference, preserved effect cleanup and kept display visibility independent of
the semantic data. Browser verification tested the actual workflow, not just rendering.

## New training-only evidence

CPU audit of the pinned corrected training manifest inspected 450 HighBuild
tiles from five cities and 9,841 measured-flag building annotations. No new
validation, official test or reserved external images were consumed.

| Check | Finding |
|---|---|
| Corrected semantic/height masks vs COCO contract | All 450 match |
| Cached-prior grid, finite values, [0,1] range | All 450 pass |
| Supported training height pixels above 200 m | 0 |
| Supported pixels with height >20 m | 11.73% |
| Training annotations with height >20 m | 626 / 9,841 (6.36%) |
| Explicit GSD in these training rows | Missing in all 450 |
| Prior source-image + model-weight hash tags | Missing in all 450 |
| Annotation/raster footprint agreement below 95% | 363 instances flagged |
| Annotation vs footprint median differs by >1 m | 13 instances flagged |

The large-disagreement review shows five fully overlapped footprints, three
partly overlapped footprints whose exclusive-support median matches their own
annotation, and five tiny supports of 1–7 pixels. This does **not** justify
removing labels or declaring them corrupt. Overlap-aware per-building support
is necessary before instance-height loss/evaluation.

To check actual prior reproducibility, two lexicographically selected training
images per city were replayed through the saved DAV2 model on CUDA. **All ten
matched their cached priors exactly** (maximum absolute difference 0). The
report records RGB, cache, model weights and preprocessing hashes. This is a
sampled reproducibility check, not certification of every cache or an accuracy test.

Evidence:

- `outputs/diagnostics/height_train_support_20260919/report.json`
- `outputs/diagnostics/height_train_support_20260919/measured_instances.csv`
- `outputs/diagnostics/height_train_support_20260919/overlap_review.json`
- `outputs/diagnostics/height_training_prior_replay_20260919/report.json`

## Decision

Do not start another long six-class height run simply because identification
looks better. The earlier residual-height pilot improved some aggregate errors
but failed ground/tall-object protections; the saved development analysis found
89.31% of baseline building squared error in references >=20 m.

Keep its evidence and the protected baseline. The next proposal is a **matched,
bounded class-conditioning experiment**, defined in
`CLASS_ASSISTED_HEIGHT_PILOT_PROTOCOL.md`. It is not trained or promoted yet.
Resolve the specified preparation gates, run a train-only feasibility check,
then a paired development comparison. No new height-accuracy improvement is
claimed from this turn's UI and data-audit work.
