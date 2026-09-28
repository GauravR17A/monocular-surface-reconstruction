# HighBuild per-building evaluation

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

This evaluator adds object-level evidence without changing the existing pixel
height metrics or the corrected HighBuild supervision artifacts.

## What is scored

- Every individual COCO building outline is retained as a separate reference
  object.
- Predicted objects are optimally matched to references at a declared IoU
  threshold (default `0.5`).
- Matched outlines report IoU and boundary precision, recall and F1 at a
  declared pixel tolerance (default `2 px`).
- Matched buildings with eligible height labels report instance-level height
  RMSE, MAE and signed bias. The predicted height is the median dense model
  prediction over the reference outline and valid height support.
- Measured and `is_estimated_height=true` annotations are reported separately.
  The strict protocol excludes estimated heights from regression; the inclusive
  protocol includes them only in a separate visible slice and combined total.
  Some upstream cities omit the flag; to remain byte-compatible with the
  corrected contract those records are treated as "not marked estimated" and
  their count is disclosed as `implicit_not_estimated_annotations`.
- The original valid-pixel height metrics are included alongside the new
  object metrics.
- Building counting has its own `counting` section and is never inferred from
  pixel accuracy. With exhaustive instance truth it reports per-scene count
  MAE/RMSE, median absolute error, exact-count rate, overcount/undercount scene
  totals and percentage error. Per-scene absolute metrics prevent an
  overcount in one tile from cancelling an undercount in another.
- Valid COCO polygons smaller than the raster grid can cover zero pixel centres.
  They are not fabricated or silently counted as misses; the report exposes
  them as `unrasterizable_annotations_excluded`.

The evaluator accepts either a building probability/binary raster or a positive
integer instance-label raster. A binary semantic mask is split into connected
components. Adjacent buildings can therefore merge; use a separately validated
instance postprocessor and `prediction_instance_path` before claiming reliable
building counts.

## Honest limitation of the current corrected masks

The current corrected HighBuild contract says that COCO polygons are known
building positives and pixels outside them are unknown. It does **not** certify
every outside pixel as non-building. Consequently:

- annotated-building recall and matched outline quality are valid;
- detection precision and count error are deliberately unavailable;
- a model prediction outside an annotation is not silently called a false
  positive.

Precision, F1 and all count-accuracy metrics are enabled only when every prediction row
supplies an independent `classification_valid_mask_path` that certifies
exhaustive annotations in that region. This is distinct from image validity and
height-label validity.

Without that independent exhaustive reference, the report still discloses the
number of supplied positive outlines and predicted components for auditing, but
all comparative count values remain `null`. This is intentionally labelled
`unavailable_positive_annotations_are_not_exhaustive`.

Square-metre coverage is emitted only when every evaluated raster has a
projected CRS with authenticated linear units. A filename, zoom level or
unverified `gsd_m` value is never used to invent area.

## Prediction manifest

Required columns:

- `sample_id`
- `prediction_height_path`
- exactly one of `prediction_building_path` or `prediction_instance_path`

Optional columns:

- `image_valid_mask_path`
- `classification_valid_mask_path`
- `classification_validity_contract` (must equal
  `exhaustive_building_annotations` whenever the classification mask is used)

Relative paths are resolved against the prediction manifest directory. All
rasters must use the exact corrected HighBuild shape, transform and CRS.
When `image_valid_mask_path` is absent, image validity comes from the source
RGB raster's own band masks—not from class labels or height availability.

## Validation command

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_highbuild_instances.py `
  --manifest data\multidomain_v2\highbuild_measured_coco_intersection_v1\manifests\validation.csv `
  --contract-report data\multidomain_v2\highbuild_measured_coco_intersection_v1\contract_report.json `
  --source-index data\highbuild_full\msr_splits\validation.csv `
  --split-root data\highbuild_full\msr_splits `
  --prediction-manifest outputs\evaluation\candidate_predictions\manifest.csv `
  --split validation `
  --output outputs\evaluation\candidate_highbuild_instances.json
```

The CLI verifies the immutable corrected-manifest hash in the contract report,
checks exact grid alignment, records hashes for every evaluated raster and COCO
payload, and refuses to overwrite an existing report. It requires complete
manifest coverage unless `--allow-subset` explicitly labels a development run.

The historical HighBuild `test` split has already been inspected in this
project and remains legacy regression evidence, not a genuinely unseen final
holdout.
