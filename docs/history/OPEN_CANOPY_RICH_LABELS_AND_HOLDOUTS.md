# OpenCanopy richer-label and holdout contract

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: metadata and planning implementation complete; class backfill and model
training intentionally not started.

## What is already preserved

- Every accepted OpenCanopy record is linked back to its official 1 km cell,
  EPSG:2154 bounds, 1.5 m ground sampling distance, larger SPOT source image,
  imagery acquisition date, LiDAR acquisition date, source URLs, licence, and
  local `source.json` hash.
- The accepted subset remains 600 training, 120 validation, and 120 locked-test
  chips. The official test role is unchanged.
- The three accepted splits have no shared 5 km region identifiers.
- Existing canopy heights, binary vegetation masks, validity masks, and CSV
  manifests are untouched.

The authenticated metadata inventory is
`outputs/data_audits/open_canopy_v2_provenance_v1.json`.

## Rich LiDAR classification path

The optional sidecar preserves source-valid upstream byte identifiers exactly
and uses 255 only for source-invalid pixels. Known IGN LiDAR HD meanings are:

| ID | Meaning | Candidate use |
|---:|---|---|
| 0 | never classified / unused | preserve and ignore |
| 1 | unclassified | ignore |
| 2 | ground | context/audit, not a vegetation subclass |
| 3 | low vegetation | richer vegetation target |
| 4 | medium vegetation | richer vegetation target |
| 5 | high vegetation | richer vegetation target |
| 6 | building | exclude from the forest expert |
| 9 | water | context/audit, not a vegetation subclass |
| 17 | bridge deck | preserve if present; unsupported in the present expert |
| 64 | permanent above-ground object | preserve if present; unsupported |
| 65 | artifact | preserve if present; ignore |
| 66 | synthetic point | preserve if present; unsupported |
| 67 | miscellaneous built object | preserve if present; unsupported |

These are LiDAR classification/height-stratum labels, not tree species. The
three vegetation categories therefore support a low/medium/high vegetation
study; they do not justify claims about species, health, snow, or land-cover
types absent from the source labels.

The resumable backfill utility is
`scripts/data/backfill_open_canopy_classification.py`. It reads only the
selected remote windows and writes to a separate versioned root such as
`D:/MSRData/data/open_canopy_v2_classification_v1`. It refuses to write
inside the prepared binary subset. Train and validation are the default;
including locked test requires a separate explicit switch. Each output raster
gets an aligned-RGB hash, source URL, class histogram, grid metadata, and its
own SHA256 sidecar.

A one-chip train-only smoke artifact has been streamed to
`D:/MSRData/data/open_canopy_v2_classification_smoke_v1`. Its aligned
384 x 384 class raster passed its hash/alignment checks and contained IDs 0, 2,
3, 4, 5, and 65. In particular, it recovered 124 low-, 512 medium-, and 146,026
high-vegetation pixels. This verifies the preservation path, not class balance
or model quality. The full train/validation backfill has not been downloaded,
and no richer-label model has been trained. The smoke raster SHA256 is
`e9575207e7202d2f1c236802fedfa61a431d20f88ecb08947cdbeb637fa48f3e`.

## Holdout findings

The present split is geographically separated at the existing coarse-region
level, but it is not independent by source acquisition. Larger SPOT mosaics
shared across accepted splits are:

| Pair | shared 5 km regions | shared SPOT source mosaics |
|---|---:|---:|
| train / validation | 0 | 57 |
| train / locked test | 0 | 49 |
| validation / locked test | 0 | 44 |

Therefore the current forest results may be called region-disjoint, but not
source-scene-disjoint or acquisition-disjoint.

`scripts/data/plan_open_canopy_holdout.py` builds a deterministic holdout from
accepted official-training metadata only. It groups samples transitively, so a
source mosaic that links multiple regions cannot be split across learning and
holdout. It can enforce any combination of region, source image, imagery date,
and LiDAR date, and it fails closed when required metadata is missing.

Generated versioned plans:

| Plan | constraints | components | largest component | learning / holdout |
|---|---|---:|---:|---:|
| geographic v1 | region | 507 | 7 | 480 / 120 |
| scene-geographic v1 | region + source image | 64 | 20 | 477 / 123 |
| acquisition stress v1 | region + source image + both dates | 5 | 459 | 486 / 114 |

The scene-geographic plan is the recommended stronger development holdout.
The all-acquisition plan technically passes its requested isolation check, but
459 of 600 records collapse into one connected component. That makes it a
useful stress test, not a well-balanced primary benchmark.

Artifacts and hashes:

- `outputs/data_audits/open_canopy_v2_holdout_geographic_v1.json` —
  `c930deb86905f52c4fe9eb0178c1741368022a04dd0264cd2055414515127b08`
- `outputs/data_audits/open_canopy_v2_holdout_scene_geographic_v1.json` —
  `0ec239e2e7de0a33b7e0a77505ef0d4b8aafe2276cfccebfc040ad5ff8938bab`
- `outputs/data_audits/open_canopy_v2_holdout_scene_acquisition_v1.json` —
  `8b7e909983479c4d5f7ac4cf11389f433781bc7fe7c8fed881ed17e038b46f5d`

## Remaining work before using richer classes

1. Pin an upstream OpenCanopy revision instead of relying on mutable
   `resolve/main` URLs, then backfill train and validation sidecars.
2. Audit the observed class histogram, spatial coverage, RGB/class alignment,
   class-specific label sparsity, and the imagery-versus-LiDAR time gap.
3. Decide whether low/medium/high vegetation is sufficiently visible at 1.5 m
   RGB resolution to justify a new head. Keep the existing binary canopy path
   as the protected comparison.
4. Train a separate candidate and report each vegetation class independently.
   Do not promote it unless OpenCanopy, GAMUS, corrected HighBuild, and unseen
   geography height guards all pass.
5. Open the locked test labels only once the experiment and thresholds are
   frozen. Metadata-only inventory does not authorize tuning on test pixels.

## Source limitations

- OpenCanopy's dataset card explicitly describes 1.5 m VHR imagery,
  LiDAR-derived canopy height, and a classification raster used to mask class 1.
- The detailed class names above come from the IGN LiDAR HD classification
  specification used by those source rasters.
- Dates are provenance fields, not proof of simultaneous acquisition. Seasonal
  and real land-cover change between SPOT and LiDAR dates can create honest
  label disagreement.
- A France-only source-scene holdout does not prove transfer to a new country,
  sensor, resolution, or season.

Official references:

- https://huggingface.co/datasets/AI4Forest/Open-Canopy
- https://geoservices.ign.fr/sites/default/files/2024-11/Fiche_produit_Nuages-de-points-LiDAR.pdf
