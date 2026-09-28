# All-six-class mixed-scene test — completed

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Data and selection

Downloaded 12 new OpenEarthMap v1 RGB/reference-label pairs: four from
Duesseldorf (Germany), four from Dhaka (Bangladesh), four from Chiang Mai
(Thailand). Each selected reference image contains all six mapped classes,
with at least 256 valid labelled pixels per class. Combined RGB/label validity
confirmed that every selected image still meets that criterion.

The 12 scenes were chosen by ascending SHA256(filename) within each region,
taking the first four with all-six reference support. Nineteen candidate label
files were inspected. This selection was locked before RGB decoding or model
inference; predictions were not used to choose images. All 12 selected results
are published, including failures. This is a deliberately label-enriched
mixed-scene challenge, not random city sampling or a global-accuracy estimate.

Only about **39.7 MB** of archive ranges were transferred, not the whole 9.1 GB
archive. All selected ZIP entries passed CRC32 and length checks; local source
SHA256 hashes are recorded. The full archive's published MD5 was NOT verified.

Data, original images, labels, predictions and the comparison gallery are saved
at `D:/MSRData/evaluation/oem_all_six_20260919_v1/`.

Official sources: [OpenEarthMap v1](https://zenodo.org/records/7223446) and
[source attribution](https://open-earth-map.org/attribution.html).
Duesseldorf: GeoNRW / North Rhine-Westphalia, DL-DE-BY-2.0. Dhaka: AIGEO Center,
CC BY 4.0. Chiang Mai: UR Field Lab Chiang Mai, CC BY 4.0.

## Model and reference interpretation

The same frozen RGB SegFormer V1 guarded epoch 8 was used, SHA256
`4940b2d4ae8817370384ccb54da9f73aee1b6bf3614435d8c077bf3e3363d021`.
Inputs were native-resolution raw RGB/255 with internal ImageNet normalization;
CUDA bf16 inference, argmax, no augmentation, calibration or postprocessing.
No training or app-model replacement occurred. This classifier was trained on
DC/Philadelphia, not these three regions; that is not proof about all upstream
pretraining data. These downloaded scenes and inspected candidate labels are
now consumed diagnostics, not an untouched future final test.

Fixed OEM-to-model mapping: bareland + developed space -> ground; building ->
building; water -> water; road -> road; rangeland -> low vegetation; tree -> tree.
Unknown and agriculture are ignored in the primary result. Agriculture can
include woody crops, so its optional low-vegetation mapping is reported only as
a separate sensitivity result. The native 8-by-6 confusion table is preserved.
Overall 10.53% of pixels were ignored. These labels are semantic references,
**not LiDAR and not height measurements**.

## Results

Pooled confusion counts across all 12 images give:

| Category | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| Ground | 28.0% | 57.4% | 37.7% | 23.2% |
| Buildings | 80.3% | 72.1% | 76.0% | 61.3% |
| Water | 93.1% | 69.7% | 79.7% | 66.3% |
| Roads | 32.8% | 40.7% | 36.3% | 22.2% |
| Low vegetation | 36.4% | 13.4% | 19.6% | 10.9% |
| Trees | 76.6% | 73.1% | 74.8% | 59.7% |

- Six-class macro F1: **54.02%**.
- Macro IoU: **40.60%**.
- Pixel accuracy: **58.46%**. This is influenced by common classes and is not
  interchangeable with macro F1.
- Two-pixel road-boundary F1: **18.11%**.
- Dark non-water reference pixels called water: **0.746%**. This is a dark-RGB
  proxy, not accuracy against independently labelled shadows.
- Every category occurs in all 12 tiles and exceeds 10,000 pooled reference pixels.
  This ensures support for computing the scores, not broad representativeness.

Per-region pooled macro F1: Duesseldorf **55.95%**, Dhaka **40.35%**, Chiang Mai
**53.20%**. The global 54.02% comes from pooled per-class confusion counts, not
the arithmetic mean of region or image F1 scores.

## Plain-language verdict

Water, buildings and trees are the stronger categories on this set, but none is
perfect. Grass/low vegetation, ground and roads are unreliable. Only 13.4% of
reference low-vegetation pixels are found; around 52.4% are called ground and
23.9% trees. Nearly 47.5% of road reference pixels are called ground.

Visual checks of the first preselected scene from each region confirm useful
large-area recognition but blurred/merged roof boundaries and missing narrow
roads/grass. Duesseldorf_19 illustrates why pixel accuracy alone can mislead:
81.8% pixel accuracy but only 36.2% macro F1, because a large wooded area is found
while several smaller categories are weak.

Black padding may receive predictions when a source TIFF has no optical nodata
declaration. Such regions with unknown reference labels are excluded from the
reported scores, not treated as ground or water truth. Predictions are not
secretly corrected using reference labels for display.

The earlier HighBuild result was 79.2% known-building recall with unknown
background, not six-class F1. It is not comparable to this 54.0% macro F1.
The weights have not worsened during this test: harder/different imagery,
reference definitions and a different metric reveal transfer weaknesses.

## Next improvement priority

Prioritize independently reviewed multi-region grass/ground/road training
examples with consistent class mapping and geographic splits; then matched
scale/appearance augmentation and detail-preserving experiments. Preserve
water/building/tree checks and the production height model. Do not claim a
reliable improvement from more epochs alone or tune on these scenes while
continuing to describe them as an untouched test. New final locations are needed
after any model changes motivated by this diagnostic.

## Files and verification

- `results/index.html`: all original/reference/prediction comparisons, class
  table and per-image results; browser checked, all 12 comparison images load.
- `images/*.tif`: original downloaded RGB files.
- `candidate_labels/<region>/*.tif`: original class labels (IDs 0-8, not heights).
- `results/scenes/<id>/original.png`: convenient native-resolution RGB for testing.
- `results/scenes/<id>/reference_ids.png`, `prediction_ids.png`: mapped class IDs.
- `results/report.json`, `per_image.jsonl`, `seal.json`: measurements/provenance.
- `selection_contract.json`, `manifest.json`, receipts and evaluation markers:
  source integrity and pre-inference selection history.

23 focused selection, range-download, mapping and metric tests passed.
Production pointer, protected height checkpoint and classifier weights remained
hash-identical. GPU inference finished and released its allocation. No new
training process or deployment was started.
