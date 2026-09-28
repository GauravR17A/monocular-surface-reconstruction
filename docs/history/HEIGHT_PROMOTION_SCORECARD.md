# Monocular Surface Reconstruction metric-height scorecard v1

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: **authenticated development baseline; production promotion is blocked
until genuinely external metric-height evidence exists.**

This scorecard makes every future height candidate answer one question under
one unchanged protocol: *is it materially better than the model currently used
by the app, without sacrificing another landscape?* It does not train, select,
copy, rename, or activate a checkpoint.

## Protected application identity

- Checkpoint:
  `experiments/20260829T203146Z_multidomain_surface_v2_guarded_vegetation/checkpoint_best_guarded_vegetation.pt`
- Checkpoint SHA-256:
  `e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144`
- Live pointer SHA-256:
  `a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9`
- Canonical full-scene report SHA-256:
  `0977bc90e034746df74dc6241c6676f15a6ea5f36dc0b422008a75c6767ab351`

The scorecard authenticates both the recorded identities and the current bytes
on disk. A changed pointer, checkpoint, manifest, contract, input tensor,
supervision tensor, sample order, or COCO payload fails closed.
The YAML is also checked against `configs/height_scorecard_v1.sha256`; replacing
this version requires an explicit new scorecard and seal rather than silently
changing the comparison rules.

## Canonical full-scene baseline

These are validation results, not a fresh blind test. HighBuild is evaluated on
complete 1024 x 1024 images using the strict measured-height mask. OpenCanopy
is evaluated on complete 384 x 384 images. Both use the same raw-RGB and cached
Depth Anything V2 prior policy.

| Scope | Pixels / objects | RMSE | MAE | Correlation | R2 |
|---|---:|---:|---:|---:|---:|
| Pooled strict HighBuild + OpenCanopy | 31,011,260 px | 10.16 m | 4.91 m | 0.482 | 0.203 |
| HighBuild measured building pixels | 13,855,690 px | 13.24 m | 5.36 m | 0.186 | -0.066 |
| OpenCanopy forest pixels | 17,155,570 px | 6.71 m | 4.55 m | 0.659 | 0.422 |
| OpenCanopy vegetation-only pixels | 7,852,823 px | 8.36 m | 6.51 m | 0.411 | -0.174 |
| OpenCanopy ground pixels | 9,302,747 px | 4.89 m | 2.90 m | 0.502 | -0.088 |
| Matched measured buildings | 328 objects | 11.15 m | 4.42 m | 0.244 | 0.037 |

The pixel and object building scores are different valid questions. The first
weights every eligible building pixel; the second takes one median prediction
per matched building. Neither may be relabelled as the other.

## What the labels actually mean

### HighBuild

- `building_height_m.tif` is a sparse building-height label, **not** a complete
  nDSM.
- The building mask contains all official COCO building outlines and supplies
  semantic positives.
- The strict validity mask contains only finite positive height pixels whose
  annotations are not marked `is_estimated_height=true`.
- Pixels outside official footprints are unknown. They are never scored as
  zero-height ground.
- Estimated heights remain available only in the separately named inclusive
  protocol. They are excluded from the primary baseline.
- Height values are used as metres according to the source annotation and
  `building_height_m` encoding. Pixel area is not claimed because projected
  scale is not independently authenticated for every tile.

### OpenCanopy

- LiDAR-derived canopy values were converted from stored uint16 decimetres with
  `value x 0.1`, and the output rasters declare metres.
- RGB resolution is 1.5 m per pixel.
- Image validity, height validity, and vegetation membership are separate.
- The current validation is region-disjoint, but many larger SPOT acquisitions
  overlap the learning splits. It is therefore not source-scene independent.

### GAMUS

- The loader keeps independent `image_valid_mask`,
  `classification_valid_mask`, and `height_valid_mask`; missing height never
  becomes zero.
- Water, road, background, and unknown pixels can teach identification while
  remaining outside height regression.
- The paper documents nominal 0.33 m pixels, but local HDF5 files have no
  per-tile CRS, transform, or pixel-scale metadata.
- Local files also lack a numeric AGL unit declaration. Raw AGL x 1.0 may be
  used only as an explicitly **metre-assumed** development target, not as
  publisher-verified metric truth.
- Values above the present 200 m cutoff are rare and span several semantic
  classes, so they stay excluded pending source-level resolution.
- The official GAMUS test has already been consumed by an older experiment. It
  is legacy evidence and may never be used for further tuning or selection.

## Candidate gate

A report is comparable only when it certifies all of the following:

1. Validation-only, complete full scenes; no official test.
2. Exact sample order, model inputs, supervision tensors, COCO data, manifests,
   mask contracts, radiometry and relative-prior policy.
3. Protected baseline replayed in the same invocation.
4. Checkpoints and the app pointer unchanged before and after evaluation.
5. No overwrite or automatic promotion.

The development gate then requires:

- no RMSE regression above 0.15 m in pooled, urban, forest, building,
  vegetation, ground, or per-building scopes;
- no MAE regression above 0.10 m;
- no absolute-bias regression above 0.10 m;
- correlation and R2 do not decrease;
- outline IoU, boundary F1, and annotated-building recall do not decrease; and
- at least one of building, vegetation, or ground RMSE improves by either
  0.5 m or 5%, while all other guards still pass.

Passing those rules means **development candidate**, not production winner.
Production eligibility additionally requires a preregistered, one-shot test on
new geography with independent metric-height truth.

## External-geography status

No current artifact satisfies that final requirement:

- HighBuild test and OpenCanopy locked test have already been inspected and are
  legacy regression suites.
- OpenCanopy validation is geographically separated at region level but shares
  source mosaics with learning data.
- NYC was excluded from the newest classifier's training, but earlier GAMUS
  work and the inherited HighBuild model have NYC exposure.
- Christchurch OpenEarthMap is only a preregistered six-class segmentation
  design. It is not downloaded and contains no height truth.

The next scorecard version must be created *before inference* on a genuinely new
location with a frozen manifest, independent DSM/nDSM reference, fixed
crosswalks and masks, and a no-tuning/no-reselection rule.

## Commands

Authenticate the baseline and current app identity:

```powershell
.\.venv\Scripts\python.exe scripts\check_height_candidate_promotion.py
```

Assess a completed same-protocol report without activating its checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\check_height_candidate_promotion.py `
  --candidate-report outputs\evaluation\<run>\report.json `
  --candidate-model-key <model-key> `
  --output outputs\evaluation\<run>\height_gate_decision.json
```

The output path must be new. The checker refuses to overwrite an existing
decision and never writes `outputs/runtime/showcase_checkpoint.txt`.
