# Offline six-class export

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Status and safety boundary

This is an **offline review tool**, not an application feature. It does not
change `outputs/runtime/showcase_checkpoint.txt`, the protected production
checkpoint, an API route, or the viewer. A candidate cannot be exported merely
because training finished: the command fails closed unless a separate JSON
report authenticates the exact checkpoint SHA-256 and records all required
validation gates as passed.

The accepted classes and raster IDs are fixed:

| ID | Class |
|---:|---|
| 0 | ground |
| 1 | buildings |
| 2 | water |
| 3 | roads |
| 4 | low_vegetation |
| 5 | trees |
| 255 | invalid / unclassified |

No probability map or confidence percentage is exported. The class raster is
only the categorical `argmax` decision. This is not presented as a calibrated
probability of correctness.

## Required validation contract

The independent evaluator/reviewer must create a report with this exact
minimum contract only after the candidate has passed the applicable checks:

```json
{
  "schema": "msr.validated_six_class_candidate.v1",
  "candidate_checkpoint": {
    "path": "C:/absolute/path/to/candidate.pt",
    "sha256": "exact checkpoint SHA-256"
  },
  "offline_export": {
    "eligible": true,
    "six_class_validation_passed": true,
    "height_regression_guards_passed": true,
    "unseen_geography_evaluated": true
  }
}
```

The checkpoint path and hash are checked again before inference. A normal
training log, an incomplete evaluator output, a failed gate, or a mismatched
checkpoint is rejected. Offline export eligibility does not promote the model
into the app.

For this legacy schema, `unseen_geography_evaluated` means a genuinely new
external geography that did not influence any shared model, height pipeline,
classifier head, threshold, or case selection. NYC excluded only from a fresh
DC+Philadelphia head does not satisfy it because earlier GAMUS experiments and
the inherited HighBuild-trained system have New York exposure. The official
GAMUS test also cannot satisfy it: Stage 3 already consumed that test, so it is
forbidden for reuse.

## Command

```powershell
.\.venv\Scripts\python.exe scripts\export_validated_six_class.py `
  D:\input\scene.tif `
  C:\path\to\validated_candidate.pt `
  C:\path\to\validated_candidate_report.json `
  outputs\offline_six_class\scene_01
```

Optional exact-grid sidecars are supported:

```text
--image-valid-mask image_valid.tif
--classification-valid-mask classification_valid.tif
--height-valid-mask height_valid.tif
--relative-prior relative_depth.tif
```

Sidecars must already match the RGB raster's dimensions, transform, and CRS;
the exporter never silently resamples them. If no classification-valid mask is
provided, a separate copy of image validity is used because dense
classification exists wherever RGB inference is valid. If no height-valid mask
is provided, finite predicted height on image-valid pixels is used. The chosen
origin of every mask is recorded in the summary.

## Output bundle

The destination must not already exist. It is staged and then published as a
single new directory containing:

- `six_class_ids.tif`: aligned `uint8` class IDs, nodata/invalid `255`;
- `height_m.tif`: predicted height, with invalid height pixels kept as `NaN`;
- `image_valid_mask.tif`, `classification_valid_mask.tif`, and
  `height_valid_mask.tif`: three separate 0/1 masks;
- `summary.json`: checkpoint/validation identities, artifact hashes, per-class
  pixel counts and shares, area availability, and height summaries.

Height summaries are produced only for buildings, low vegetation, and trees,
after intersecting image validity, classification validity, the selected class,
and height validity. Missing heights are excluded rather than converted to
zero. Water and road heights are never reported.

Square-metre coverage is reported only for a projected CRS with a finite,
positive linear pixel area. For PNG/JPG, missing CRS, geographic/angular CRS,
or invalid transforms, every class receives `coverage_m2: null` and the summary
states the reason. This prevents approximate or invented area claims.
