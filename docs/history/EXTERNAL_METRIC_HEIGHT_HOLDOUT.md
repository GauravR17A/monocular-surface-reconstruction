# External metric-height holdout: Bay of Plenty v1

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: **downloaded, publisher-hash verified, aligned, and sealed unconsumed;
no model inference or score has been run**.

This package closes the missing external-geography evidence path without
reusing GAMUS official test data or the known DFC19 showcase.  It is a single,
fixed New Zealand scene for one post-freeze metric-height evaluation.  It must
never be used for training, validation, calibration, threshold choice,
checkpoint choice, fusion choice, or showcase selection.

## Why this source

The source is the official Toitu Te Whenua Land Information New Zealand public
AWS archive:

- New Zealand Imagery registry:
  https://registry.opendata.aws/nz-imagery/
- New Zealand Elevation registry:
  https://registry.opendata.aws/nz-elevation/
- LINZ attribution requirements:
  https://www.linz.govt.nz/products-services/data/licensing-and-using-data/attributing-elevation-or-aerial-imagery-data

The registry describes open aerial imagery and 1 m LiDAR-derived DSM/DEM
products with STAC metadata.  Each fixed collection says `CC-BY-4.0`; its STAC
provider block identifies BOPLASS as licensor, AAM NZ as producer, and LINZ as
host/processor.

Required attribution:

> Sourced from the LINZ Data Service and licensed by BOPLASS for re-use under
> the Creative Commons Attribution 4.0 International licence.

## Fixed scene (chosen before pixel inspection)

Selection was mechanical, not visual: the first lexicographic RGB STAC item in
the fixed 2018-2019 Bay of Plenty urban collection whose complete footprint is
contained by same-season DSM and DEM coverage.

| Role | Official collection / item | Capture range | Grid |
|---|---|---|---|
| RGB orthophoto | `bay-of-plenty_2018-2019_0.1m`, `BC36_1000_2314` | 2018-07-11 to 2019-01-16 | EPSG:2193, 0.1 m |
| LiDAR DSM | `bay-of-plenty_2018-2019/dsm_1m`, `BC36_10000_0302` | 2018-11-30 to 2019-04-29 | EPSG:2193, 1 m |
| LiDAR DEM | `bay-of-plenty_2018-2019/dem_1m`, `BC36_10000_0302` | 2018-11-30 to 2019-04-29 | EPSG:2193, 1 m |

The official capture ranges overlap, but per-pixel dates are unavailable.
Real change between the optical and LiDAR acquisitions can therefore contribute
to measured error and must be disclosed.

Exact URLs, collection IDs, item bounds, publisher SHA-256 multihashes, file
sizes, licence, attribution, CRS, GSD, and dates are frozen in
`configs/external_metric_height_nz_bop_v1.json`.  Its repository seal is:

`ddf68cdf382366450d29ed433ddb07a737c2fe07581e03b7c32e290a68eb456c`

## Fixed reference and masks

The metric object-height reference is:

`reference nDSM = LiDAR DSM in metres - paired LiDAR DEM in metres`

Because DSM and DEM are paired products from the same survey, subtraction
removes their shared absolute vertical datum.  The GeoTIFFs expose EPSG:2193 as
their horizontal CRS but do not embed a compound vertical CRS, so an absolute
elevation-datum claim is intentionally not made.

Before any model output existed, three supports were fixed:

1. `all_valid`: valid RGB + finite DSM/DEM + nonnegative DSM-minus-DEM.
2. `object_gt2m`: `all_valid` and reference nDSM strictly above 2 m.
3. `tall_object_gt5m`: `all_valid` and reference nDSM strictly above 5 m.

Only nodata, nonfinite values, and negative DSM-minus-DEM artifacts are
excluded.  There is no upper-height cutoff.  The object supports prevent the
large amount of flat ground from hiding structural failures.

## Preparation and one-shot use

Acquire, align to the fixed 1 m grid, derive nDSM/masks, and verify hashes:

```powershell
.\.venv\Scripts\python.exe scripts\data\prepare_external_metric_height_holdout.py acquire-prepare
.\.venv\Scripts\python.exe scripts\data\prepare_external_metric_height_holdout.py verify
```

The compact package lives only at:

`D:\MSRData\external_holdouts\nz_bop_2018_2019_bc36_2314_v1`

Preparation imports no model and runs no inference.  It creates
`SEALED_MANIFEST.json`, hashes every raw and derived file, and writes an
explicit `NO_TRAIN_OR_TUNE.txt` marker.

Acquisition verification on 2026-09-15 produced:

| Check | Sealed result |
|---|---:|
| Raw official files | 11 files, 97,123,077 bytes |
| Prepared aligned rasters | 7 files, 3,658,293 bytes |
| All-valid support | 326,925 pixels |
| Object support, reference nDSM >2 m | 133,567 pixels |
| Tall-object support, reference nDSM >5 m | 124,217 pixels |
| Negative reference artifacts excluded | 18,675 pixels |
| Package-manifest SHA-256 | `6cff2c60939e8ea5809569bcaf6216dc589728b106420c5ebd0383788fdf7e1b` |

These are preregistered mask-support counts, not model results.  Neither an
`EVALUATION_PLAN.json` nor a `CONSUMED.json` exists yet.

Local audit order is also preserved: the final contract was sealed at
11:10:08 UTC, the first raw source file was created at 11:16:22 UTC, and the
prepared manifest was sealed at 11:18:52 UTC.  Thus the scene, masks, and
thresholds were fixed before the first raster was downloaded or decoded.

Only after the candidate, configuration, code, and app pointer are final may a
maintainer create `EVALUATION_PLAN.json` with `seal-plan`.  The evaluator must
then create `CONSUMED.json` *before* its first model read.  A crash still counts
as consumption; any exceptional rerun must be disclosed.  Complete all-valid,
object-only, and tall-object metrics must be published regardless of outcome.

## Geography claim limit

A repository inventory on 2026-09-15 found no Bay of Plenty or New Zealand
samples in known Monocular Surface Reconstruction training/evaluation manifests.  A Christchurch
OpenEarthMap holdout exists only as an unacquired design and remains untouched.
The permitted wording is:

**"One-shot transfer to a preregistered Bay of Plenty metric-height scene, new
to known Monocular Surface Reconstruction project data, using an independently hosted LINZ/BOPLASS
orthophoto and LiDAR reference."**

This does not prove worldwide generalisation or absence from unknown upstream
foundation-model pretraining.
