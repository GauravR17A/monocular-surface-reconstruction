# RGB segmenter V1: completed Christchurch transfer test

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Completed 2026-09-15. The frozen epoch-eight RGB-only classifier was evaluated
once on all 49 publicly labelled Christchurch OpenEarthMap pairs at native
resolution. No Christchurch training, calibration, checkpoint selection, or
reference input to inference occurred. The other 21 regional images lack
released labels and were explicitly inventoried before any pixels were decoded.

## Results

| Class | Precision | Recall | F1 | IoU |
|---|---:|---:|---:|---:|
| Ground | 22.63% | 27.38% | 24.78% | 14.14% |
| Buildings | 58.69% | 95.06% | 72.57% | 56.95% |
| Water | 95.45% | 80.57% | 87.38% | 77.59% |
| Roads | 49.86% | 82.91% | 62.27% | 45.21% |
| Low vegetation | 68.74% | 44.15% | 53.77% | 36.77% |
| Trees | 79.38% | 61.07% | 69.03% | 52.70% |

- Six-class macro F1: **61.63%**; macro IoU: **47.23%**.
- Whole-tile bootstrap macro F1 interval: **58.43-63.83%** (2,000 resamples).
  This is descriptive within-city sampling uncertainty. Spatial dependence
  between neighbouring tiles and global geographic uncertainty are not captured.
- All six classes pass the predeclared coverage check: at least five tiles and
  10,000 labelled pixels each. This is coverage, not a quality acceptance gate.
- Two-pixel road-boundary F1: **23.07%**. Roads are often identified but their
  boundaries are not reliably precise.
- Dark non-water pixels predicted as water: **0.554%**. This is a dark-RGB proxy,
  not labelled-shadow accuracy.
- Agriculture-as-low-vegetation sensitivity macro F1: **61.41%**; it does not
  replace the primary result. Primary ignored pixel share: **6.37%**.
- Combined-vegetation diagnostic F1: **75.60%**. Combining categories is not an
  improvement in the six-class score and must never replace it.

## What it means

Development F1 was 83.83%, but that was reused DC/Philadelphia validation.
The 61.63% result measures transfer to a different city and annotation system.
It is not evidence that the model weights degraded, and there is no older-model
Christchurch comparison establishing an external improvement percentage.

Building recall is high, but precision is only 58.69%: many non-building pixels
are being called buildings. Ground and low vegetation are clear weaknesses.
Water transfers considerably better. This is useful experimental performance,
not a broadly reliable six-category production claim.

This city is new to the known project inventories, not guaranteed absent from
unknown upstream pretraining. The OEM-to-GAMUS ontology mapping is explicit;
source-label differences may contribute, but do not excuse incorrect outputs.
Christchurch is now consumed. Any future model adjusted in response to these
findings needs another genuinely untouched final location.

## Safety, integrity and reproducibility

- Candidate SHA-256:
  `4940b2d4ae8817370384ccb54da9f73aee1b6bf3614435d8c077bf3e3363d021`.
- Source/config/software/data identities were sealed before the consumption
  marker and first pixel decode. Whole-file identities were checked again after
  evaluation. No evaluation retry was needed.
- 42 targeted tests passed before the run, including crosswalk, masking-related
  confusion handling, native source matrix, coverage, bootstrap, header checks,
  download range refusal, cache position, and development replay tests.
- All 98 selected TIFF files passed archive CRC32 and length verification, then
  were SHA-256 recorded. About 173 MB of network payload including range-cache
  overhead was used; extracted files total 205.75 MB. The full 9.1 GB archive
  was not downloaded and its published MD5 was NOT verified.
- Protected production checkpoint and app pointer remain hash-identical. No
  height pipeline change or height evaluation was performed. This test measures
  identification only; OpenEarthMap does not supply height truth.
- GPU inference has finished; no new training was launched.

### Report wording erratum (numerical results unchanged)

The reused dark-pixel metric helper embeds the word "GAMUS" in its descriptive
strings. In this external run the actual reference arrays came exclusively from
OpenEarthMap, through the frozen crosswalk. Read that field as "OpenEarthMap
reference class is not water" and "OpenEarthMap has no labelled shadow class".
The exact counts and rate above are unchanged. The sealed evaluator and original
machine report are preserved instead of silently rewriting them after testing.

## Saved evidence and next step

- `outputs/evaluation/christchurch_rgb_segmenter_v1/report.json`
- `outputs/evaluation/christchurch_rgb_segmenter_v1/seal.json`
- `outputs/evaluation/christchurch_rgb_segmenter_v1/per_image_metrics.jsonl`
- `outputs/evaluation/christchurch_rgb_segmenter_v1/predictions/`
- `D:/MSRData/external_holdouts/oem_christchurch_rgb_v1/CONSUMED.json`
- `D:/MSRData/external_holdouts/oem_christchurch_rgb_v1/manifest.json`

The staging status/manifest describe the earlier acquisition stage; CONSUMED.json
and the evaluator's completed status are authoritative about evaluation use.

Next: audit training-only city/crop coverage, then a controlled small-vegetation
sampling experiment against an equal-budget baseline. Protect all six class
development guards and the production height pipeline. Do not automatically
promote this candidate or start an expensive run without defining that experiment.

Dataset and imagery attribution: OpenEarthMap v1,
https://zenodo.org/records/7223446 ; AIRS / LINZ, CC BY 4.0,
https://open-earth-map.org/attribution.html .
