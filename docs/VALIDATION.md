# Validation protocol and evidence interpretation

## Separate four kinds of evidence

Research evaluation measures a named model against independent reference data.
Application verification checks whether a implemented workflow behaves correctly.
Publication verification checks that a portable release preserves identities,
notices, links and model values. Presentation inspection checks that the visible
experience communicates those products clearly. Passing one does not imply the
others have passed.

## Metric definitions

For valid paired prediction p and reference y, error e = p - y. MAE is the mean of
|e|, RMSE is sqrt(mean(e²)), and signed bias is mean(e). R-squared is
1 - sum(e²)/sum((y - mean(y))²), undefined when the reference variance is zero.
Pearson correlation measures centered linear association, not metre accuracy.
Report support counts, masks, aggregation and units alongside every score.

Pooled RMSE is the square root of pooled squared error per valid pixel; it is not
the mean of per-scene RMSE values. Pixel-weighted and equal-scene summaries answer
different questions. Object metrics require an explicit matching rule and
aggregation within each building. An object-only score omits unmatched objects
and therefore cannot replace detection or all-eligible pixel evaluation.

Semantic scores begin from a confusion matrix with a documented ontology. Class
precision is TP/(TP+FP), recall TP/(TP+FN), F1 2TP/(2TP+FP+FN), and IoU
TP/(TP+FP+FN). Macro averages weight classes equally; region-balanced averages
also prevent a large or easy city from dominating the selector. Missing classes
and nodata need declared treatment. V1 and V3 results cannot be transferred across
checkpoint identities.

## Reference isolation

Reference nDSM/DSM is accepted for evaluation only after the inference input is
fixed. It must not alter inference, select a visually favourable crop, fit a
post-hoc scale/offset, or repair the prediction. A terrain DEM used as an absolute
datum cannot simultaneously be presented as an independent validation source for
that same calibration. The hilly demonstration therefore uses separate SRTM for
its disclosed coarse comparison.

## Matched height protocol

Freeze model hashes, full scene lists, preprocessing, tiling, precision, units,
masks and aggregation before comparing candidates. HighBuild measured-building
support excludes estimated annotations and unknown background. OpenCanopy splits
vegetation, ground and low vegetation using its fixed LiDAR-derived supports.
Record tallest-height bands, bias, centered spread, agreement and safety guards.

Development subsets are allowed for fast iteration, but must be labelled as such.
The paired control receives the same budget and non-semantic inputs. A candidate
is eligible only when the registered selection and retention requirements hold;
an improved average cannot erase a failed safety support. Rejected experiments
remain valuable records of what was learned.

## Geographic and temporal dependence

City exclusion for a new head is not necessarily complete-system novelty. Adjacent
tiles, overlapping mosaics and prior tests can create dependence even when file
names differ. Record image date, LiDAR date, provider and geographic overlap.
Tile bootstrap intervals describe within-support uncertainty; they do not prove
global geographic generalisation. Label-enriched challenges are diagnostic rather
than random population samples.

## Reproducibility ladder

The public package supports source inspection, deterministic logic tests, saved
scene inspection and inference with identified release weights. It also includes
historical protocols/configurations needed to reconstruct research workflows after
lawful data acquisition. Full training replay additionally requires original data
revisions, caches, sampled indices, hardware/software settings and run archives.
The release does not claim bitwise training replay from a small inference bundle.

Release tensor equality and demo-raster equality are recorded under `docs/evidence`.
The Python suite reports historical archive dependencies explicitly. No sealed
holdout is consumed merely to verify packaging.
