# Scientific contracts

These contracts define what the application's quantities and evidence mean.
They take precedence over the visual appearance of a scene.

## Quantities and units

| Quantity | Meaning | What it does not establish |
| --- | --- | --- |
| Relative depth / rDSM | A dimensionless ordering or relief score inferred from one image. | Absolute metres, a surveyed height or an elevation datum. |
| nDSM / object height | Estimated surface height above local ground. | Absolute elevation or hidden ground under an object. |
| DEM / DTM input | A supplied terrain-elevation reference with its own grid and datum. | Fine object height or guaranteed bare-earth quality for every provider product. |
| Absolute DSM | Aligned terrain elevation plus nonnegative inferred object height. | Accuracy beyond the terrain and inference sources used to construct it. |
| GCP-calibrated surface | A scene-level affine mapping fitted from supplied control elevations. | Independent validation on those same fitting points. |
| Semantic class | A predicted RGB category from the six-class classifier. | Water depth, vegetation species, a verified outline or correctness probability. |
| Presentation height | A mesh-oriented height field adjusted for readable geometry. | An additional measurement or a better numerical prediction. |
| Signed error | Prediction minus eligible reference, in matched units. | Error outside the evaluated support or against an incompatible datum. |

## Validity is task-specific

Image validity, semantic-label availability and height-label availability are
separate masks. A usable optical pixel can lack a class label or a height label.
That missing label must not be replaced with zero. Semantic road or water pixels
can teach identification while invalid height artifacts at the same locations are
excluded from height regression.

HighBuild primary height evaluation uses measured-height support intersected with
eligible annotated buildings. Estimated annotations are separated. Unknown
background is not evidence for zero-height ground, and a positive-only building
inventory cannot support exhaustive counting precision.

OpenCanopy supervision respects source validity and LiDAR class limitations.
Source acquisition dates and mosaic overlap matter even where tile IDs are
different. Derived strata such as low/medium/high vegetation are height groups,
not botanical labels.

## Spatial alignment

Paired RGB, target and masks must match shape and their declared spatial grid, or
an explicit alignment operation must produce a new derived raster. Reprojection
must retain a record of interpolation and support. An array with no CRS cannot be
silently assigned an authoritative map scale from its filename.

The interactive reader can reduce resolution while preserving full geographic
bounds through an adjusted affine transform. Pixel-coordinate control points must
be transformed to that processing grid. Geographic-coordinate control points
remain interpreted in the declared spatial reference. A resampled raster does not
gain information from its finer output pixel spacing.

## Evaluation definitions

For N eligible pairs, let e_i = prediction_i - reference_i.

- MAE = sum(abs(e_i)) / N.
- RMSE = sqrt(sum(e_i * e_i) / N).
- Bias = sum(e_i) / N; positive bias means overestimation.
- R-squared = 1 - sum(e_i * e_i) / sum((reference_i - mean_reference)^2).
- Correlation describes association, not absence of scale or offset error.
- Precision, recall, F1 and IoU come from class-specific confusion counts.
- Six-class macro scores average the six final categories equally. A merged
  five-group vegetation statistic is a different result.

Degenerate cases need explicit treatment rather than an invented finite score.
Pooled pixel metrics weight pixels; per-scene means weight scenes; equal-region
scores weight regions. Those aggregation rules can rank the same candidates
differently. A published selector must not be changed after seeing results.

## Data separation and claim scope

Training diagnostics, reused development validation and a previously untouched
external test are distinct evidence. A city excluded from a newly trained head
can still have been seen by the inherited system. Unknown upstream pretraining
coverage prevents claims of absolute global novelty.

Consumed holdouts remain consumed even after a poor result or a process crash.
New model selection in response to an external test needs another reserved test
for a fresh final claim. A prepared/downloaded holdout is not a measured result.

## Visualization and provenance

Source arrays support measurements, slope/profile calculations and exports.
Smoothing, facades, vertical exaggeration and illustrative solids support display.
Source/presentation separation is tested because a plausible 3D view can otherwise
make an inaccurate field look authoritative.

Each reported metric should identify the model/checkpoint, input contract,
geography, split, reference type, validity support, grid, aggregation rule and
evaluation date. Hashes identify bytes; they do not themselves establish label
quality, licensing, independence or scientific validity.
