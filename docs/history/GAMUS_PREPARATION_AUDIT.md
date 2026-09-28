# GAMUS preparation audit v1

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: **passed for the height-focused comparison pilot only**. This is not
permission to promote a checkpoint or to tune on the official test split.

Historical clarification: this preparation audit handled the official GAMUS
test as inventory-only and did not score it. Separately, the older Stage-3 final
evaluation consumed that test once. It is therefore no longer an untouched holdout
and is forbidden for V4 tuning, selection, or repeat evaluation.

## What was verified

- The immutable GAMUS quality index still has its declared SHA-256 and keeps
  5,001 approved training tiles, 859 validation tiles, and 2,860 inventory-only
  test tiles. The official splits remain disjoint.
- Every approved RGB tile was paired with its cached Depth Anything V2 prior.
  All 8,720 pairs passed grid, numeric range, recorded model, recorded source
  path, and unit-tag checks. Current RGB and prior file bytes are locked by
  per-file SHA-256 in the identity manifest.
- Five deterministic samples spanning every city present in train/validation
  were regenerated with the locally locked DAV2 model. All 5,242,880 float16
  output values matched the cache exactly. This audit did not recompute the
  official test split; the separate Stage-3 consumption noted above is
  preserved as historical final-audit evidence.
- The loader exposes separate image-, classification-, and height-validity
  masks. Real tiles contain thousands of pixels where the image is usable but
  the height label is not, and those pixels remain excluded from regression.
  Colour transforms now use image validity rather than height availability.

## Units and geographic scale

- The GAMUS paper documents a nominal RGB/nDSM pixel resolution of **0.33 m**
  and 1024 x 1024 tiles (about 337.92 m wide nominally).
- The local HDF5 files do not embed pixel scale, CRS, transform, or geolocation.
  The 0.33 m value is therefore dataset-level documentation, not per-file
  geospatial proof. Precise square-metre output still requires trusted input
  georeferencing.
- The files and local README do not declare the numeric AGL unit. The official
  loader reads AGL without rescaling, but that does not prove the unit. For the
  comparison pilot only, Monocular Surface Reconstruction keeps the explicit assumption that raw
  AGL x 1.0 is metres. Results must be described as metre-assumed until the
  publisher or original source products confirm it.

## Values above the 200 cutoff

| Split | Pixels above 200 | Affected tiles | Fraction of nonnegative finite labels | Maximum |
|---|---:|---:|---:|---:|
| Train | 3,253 | 15 / 5,001 | 0.0000640% | 392.60 |
| Validation | 43 | 9 / 859 | 0.00000482% | 272.59 |
| Official test (inventory only) | 135,892 | 8 / 2,860 | 0.004705% | 304.77 |

Some values above 200 belong to ground, road, low-vegetation, or tree labels,
not only tall buildings. That is strong evidence that blindly raising the
global cutoff would admit label artefacts. The comparison pilot therefore
keeps the predeclared 200 cutoff, while the exact tile/class/city inventory is
preserved for later source-level review.

## Honest DAV2 provenance limit

The historical cache stores model name, source path and `relative_0_1` units,
but it did not store source/model/config content hashes when each prior was
generated. The v1 audit does not invent that missing history. It locks the
current RGB/prior/model/code bytes and adds independent deterministic
recomputation evidence. Future cache generation should write those hashes at
creation time.

## Artifacts

- Machine report: `outputs/data_audits/gamus_preparation_v1/audit_report.json`
- Per-pair DAV2 identity manifest:
  `outputs/data_audits/gamus_preparation_v1/dav2_prior_identity.jsonl`
- Height-outlier records:
  `outputs/data_audits/gamus_preparation_v1/height_outliers.jsonl`
- Exact artifact hashes:
  `outputs/data_audits/gamus_preparation_v1/artifact_hashes.json`

The report, manifests, loader checks, and tests were created without modifying
the GAMUS files, starting training, or changing the app's live checkpoint.
