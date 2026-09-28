# Multi-region RGB classification V3

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Purpose and boundaries

Train a standalone six-category classifier after the frozen V1 generalisation audits. The recent 12-scene challenge scored 54.02% macro F1, with particularly weak low vegetation (19.60%) and roads (36.30%). Those are **diagnostic results**, not worldwide accuracy. A prior crop-sampling V2 did not establish a safe benefit over its matched control.

This experiment changes the training-data/augmentation package, not model architecture. It is not an ablation establishing which individual change caused improvement. The production height checkpoint, live pointer, app and earlier experiments remain untouched. No accuracy gain is guaranteed.

## Frozen comparison

- Start: RGB SegFormer V1, epoch 8, `experiments/20260915T140116654177Z_gamus_rgb_segmenter_v1/checkpoint_best_guarded.pt`.
- SHA-256: `4940b2d4ae8817370384ccb54da9f73aee1b6bf3614435d8c077bf3e3363d021`.
- Independent GAMUS validation replay: 859 full-native DC/PHL tiles, macro F1 83.83%. These are reused development results, not untouched geography.
- Evaluate the exact same V1 checkpoint on the new 50-image OEM development split before any new training updates.

## Data and geography

All added downloads are under `D:/MSRData/training/oem_multiregion_v3`.

| Role | Source | Tiles |
| --- | --- | ---: |
| Training replay | GAMUS DC/Philadelphia | 3,837 |
| Added training | OEM Aachen, Chisinau, Accra, Kitsap (30 each), Bogota (5), Niamey (9) | 134 |
| Existing development validation | GAMUS DC/Philadelphia | 859 |
| New-region development validation | OEM Melbourne and Vienna (25 each) | 50 |
| Reserved for later final evaluation | OEM Rosario and Monrovia | Not downloaded or inferred in this run |

New paired images are chosen by filename SHA-256 ordering, not model performance or a favourable class-content filter. Entire OEM regions are separated. Reference/RGB alignment, source CRC and SHA-256, class IDs, nonempty labels, duplicate train/validation image content and pooled six-class coverage are checked before training. The selected source files total approximately **756 MB**, not the full 9 GB archive; range requests add network overhead.

Christchurch, Duesseldorf, Dhaka and Chiang Mai are excluded from this run's training and development validation. Their earlier results are preserved as already-consumed diagnostics. Do not relabel them as untouched final tests. The new development regions are excluded from this continuation's training; upstream foundation-model pretraining geography is not comprehensively known.

## Inputs, labels and missing data

The six classes remain ground, buildings, water, roads, low vegetation and trees. RGB is raw byte imagery scaled to 0–1, with ImageNet normalisation inside the same model. No height, AGL or DAV2 maps are model inputs here.

OEM mapping: bareland/developed space -> ground; rangeland -> low vegetation; road/tree/water/building retain their identities. Agriculture and unknown -> ignored (255), **not ground or zero height**. This mapping is approximate; OEM and GAMUS annotation definitions are not identical.

Image-validity and classification-validity masks remain independent. Joint validity determines classification loss. Spatial transforms are identical across image, labels and masks; labels/masks use nearest-neighbour resampling. Colour changes do not infer or erase label validity. The dark-pixel water diagnostic remains a proxy, not a labelled shadow test.

## Training and resource limits

Maximum eight epochs, with early stopping after at least four and three epochs without improved OEM region-macro F1. Every epoch visits all 3,837 GAMUS images once and adds 1,663 equal-city OEM draws. Random 256–512 px source windows are resampled to 384 px; aligned rotations/flips, modest brightness/contrast/saturation variation and occasional blur augment training only.

Batch 4, two data workers, RTX CUDA bf16. Encoder learning rate 1.5e-5; decoder 1.5e-4; focal supervision and clipping. A 24-update mixed-source overfit/gradient sanity check must reduce its training-only loss by at least 10%; those proof weights are discarded before the real run. This check demonstrates that optimisation works, not predictive accuracy.

## Predeclared acceptance

- Evaluate all 859 GAMUS and 50 OEM development images at native resolution after every epoch, with fixed references and no test-time augmentation.
- Protect GAMUS per-class F1 within 1 percentage point, pooled and per city; macro F1 within 0.5 points.
- Protect OEM per-class F1 within 2 points, pooled and per validation region, for classes with reference support.
- Protect road-boundary F1 within 2 points and dark-non-water false-water rate within +0.5 points in each evaluated group.
- Require at least +3 points in OEM region-averaged six-class macro F1, and +5 points in the mean pooled ground/road/low-vegetation F1.
- Save best unrestricted development and best safety-passing checkpoints separately. Safety passing is **not** the same as demonstrated improvement.
- No automatic app promotion. After completion, review errors and HighBuild positive-outline coverage as an additional transfer check. Freeze an accepted candidate before consuming reserved final geography. Classification gains do not establish height gains.

## Saved state and recovery

Code/config/data identities are sealed per experiment. Each epoch saves model, optimiser, scheduler, RNG and metrics using an epoch checkpoint and atomic commit. Latest/best files are recoverable aliases. The original V1 checkpoint and height pointer are hash-checked before/throughout the run.

Start: `scripts/run_rgb_multiregion_v3.ps1`. Resume: the same script with `-Resume <exact experiment directory>`. Do not bypass integrity checks, run duplicate trainers or silently change the recipe on CUDA OOM. Incomplete downloads resume from verified receipts.

Compact status: `scripts/watch_rgb_multiregion_v3.ps1`. Training registration: `outputs/orchestration/rgb_multiregion_v3_active.json`.

## Sources and licensing

- [OpenEarthMap paper](https://arxiv.org/abs/2210.10732), Xia et al., WACV 2023.
- [Published dataset](https://zenodo.org/records/7223446) and [official regional attribution](https://open-earth-map.org/attribution.html).
- Aachen: GeoNRW/North Rhine-Westphalia, DL-DE-BY-2.0. Chisinau: HTCD/Lightcyphers, CC BY 4.0. Accra/Niamey: Open Cities AI/GFDRR, CC BY 4.0. Kitsap: USGS, public domain. Bogota: MaptimeBogota, CC BY 4.0. Melbourne: City of Melbourne, CC BY 4.0. Vienna: City of Vienna, public domain.
- Existing MiT-B0 pretrained weight terms remain research/evaluation restrictions; this experiment does not grant commercial deployment rights.
