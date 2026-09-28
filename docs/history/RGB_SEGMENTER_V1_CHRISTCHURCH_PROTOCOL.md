# RGB segmenter V1: Christchurch external-transfer protocol

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Recorded 2026-09-15, before acquiring or decoding selected image/label files.
Status: acquisition permitted; evaluator implementation/seal required before
pixel decoding. No external scores exist at the time of writing.

## Explicit amendment, not retroactive preregistration

The earlier `OPENEARTHMAP_CHRISTCHURCH_EXTERNAL_HOLDOUT.md` reserved this city
for V3/V4. That document remains unchanged. This new protocol evaluates the
completed independent RGB segmenter V1 instead. It is a locally recorded
pre-evaluation commitment, not an externally timestamped preregistration.

Candidate: epoch 8 of `20260915T140116654177Z_gamus_rgb_segmenter_v1`.
Checkpoint SHA-256:
`4940b2d4ae8817370384ccb54da9f73aee1b6bf3614435d8c077bf3e3363d021`.
No candidate reselection, calibration, training, or class-threshold tuning on
Christchurch. There is no paired V3/V4 comparison in this protocol.

## Source, inclusion and integrity

- OpenEarthMap v1, https://zenodo.org/records/7223446 .
- Official archive size: 9,099,481,727 bytes; published whole-archive MD5:
  `64155d1dc9d3b68536063f79878e1a67`.
- Include EVERY `christchurch/images/*.tif` with its exact same-stem
  `christchurch/labels/*.tif`, independent of published training/test membership.
  An orphan label or unexpected format stops preparation; no selective omission
  of any publicly labelled pair.
- Imagery attribution: AIRS / LINZ, CC BY 4.0;
  https://open-earth-map.org/attribution.html . Retain source attribution.
- Store under `D:/MSRData/external_holdouts/oem_christchurch_rgb_v1`.
- Download official ZIP byte ranges only, with strict HTTP 206 and Content-Range
  checks. Verify selected entries using ZIP CRC32 and lengths, then record SHA-256
  for every extracted file and a sorted complete manifest.
- This selective method does NOT verify the published whole-archive MD5. CRC32
  detects accidental corruption, not adversarial tampering. Per-file SHA-256
  records retrieved identity; it is not a publisher-provided authenticity proof.
  HTTPS official source and frozen metadata are additional safeguards.
- ZIP decompression is allowed during staging; TIFF pixel decoding, thumbnails,
  histograms, and model evaluation are prohibited until the evaluator is sealed.

## Fixed ontology

Pre-pixel metadata clarification: the official archive inventory contains 70
Christchurch images but only 49 labels, each paired with an image. Therefore
evaluate all 49 publicly labelled pairs and record the other 21 image stems as
unscorable (no released label). No image pixels or scores informed this decision.
The first staging attempt stopped at the strict pairing check before extracting
any files. The amended check rejects orphan labels and duplicate members but
allows explicitly inventoried unlabelled images to remain unscored.

OEM IDs map to six-class indices as follows:
`0 -> ignore255, 1 -> ground0, 2 -> low_vegetation4, 3 -> ground0,
4 -> roads3, 5 -> trees5, 6 -> water2, 7 -> ignore255, 8 -> buildings1`.
Agriculture is ambiguous (including orchards). Report a separate sensitivity
table mapping 7 to low vegetation, never replacing the primary result.
Report the native OEM 8-by-6 table as well, so merged classes cannot hide errors.

## Fixed inference and consumption

Use the sealed V1 model wrapper and strict checkpoint loading; raw uint8 RGB
divided by 255, unchanged wrapper ImageNet normalization, native image grid,
batch one, CUDA bf16, bilinear logits upsampling as already implemented, argmax,
no test-time augmentation or per-scene normalization. No resizing, tuning,
height inputs, reference inputs, DEMs, or class-dependent fusion. Unsupported
radiometry/dimensions stop the run and must be disclosed, not silently adapted.
RGB validity and label validity remain separate; exclude raster no-data masks
and OEM 0/7 only in the primary evaluation. Do not assume black RGB is invalid.

Before first TIFF pixel decode, save model/config/code/software/file identities
and this protocol's hash, then create `CONSUMED.json`. Record any execution
failure and retry under the same unchanged contract. Do not claim a rerun was
the first look. All complete pairs must finish before publishing a headline.

## Required output and claim limits

- Exact pooled confusion matrix; per-class P/R/F1/IoU, tile and pixel support;
  macro scores with absent classes explicitly marked, not called perfect.
- Whole-tile bootstrap (seed 20260915, 2,000 resamples) confidence intervals,
  labelled as within-this-city sampling uncertainty, not global uncertainty.
- Native OEM 8-by-6 table, ignored share, agriculture sensitivity,
  low-vegetation/tree confusion and diagnostic combined-vegetation F1.
- Two-pixel road-boundary scores and dark-RGB water false-positive proxy;
  explicitly not labelled-shadow accuracy.
- Call it comprehensive six-class coverage only if each mapped class appears
  in at least five tiles and 10,000 reference pixels. Insufficient coverage is
  reported without swapping locations to obtain a better score.
- Publish the complete aggregate report before inspecting visual examples.
  Preview order: SHA-256 of UTF-8 tile stem, ascending; first six.
- No app promotion from this result alone. No height-accuracy claim: OEM has no
  height ground truth. Retain protected production checkpoint and pointer hashes.

Permitted claim: transfer to Christchurch, new to the known project inventories,
under this fixed crosswalk. Not proof of global generalisation, absence from
unknown upstream pretraining, or performance on every satellite sensor.
