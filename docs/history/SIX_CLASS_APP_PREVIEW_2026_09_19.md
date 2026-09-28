# Six-class application preview — 19 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## What is available

Open http://localhost:3000, upload an RGB image (or choose a reference scene), then select **Identify six classes** under **Scene intelligence** in the right panel.

- Pinned V3 epoch-1 RGB model identifies ground, buildings, water, roads, low vegetation and trees.
- A separate color overlay is projected onto the existing 3D surface and building solids. Class filtering dims other classes; it does not flatten or raise geometry.
- The miniature classification map, pixel coverage, label GeoTIFF and provenance JSON are available. Unknown image pixels remain 255/transparent and are excluded from coverage.
- Projected map areas are reported only for a projected CRS with known units. Ordinary JPEGs and bundled JPEG demos report pixels, not invented square metres.
- Inspection and flight readouts can show an additional **RGB preview class**, separately from the protected height model's classification and height.
- Disable **Show on 3D surface** to restore the previous appearance. Three-group height-isolation controls are disabled while this overlay is visible, to prevent confusing two different features.
- Switching scenes clears the previous classification. Requests use the original uploaded RGB, not a stretched display texture. Bundled demos use their explicitly identified display RGB.

## Boundaries

This is an opt-in experimental classifier, **not a new production height model**. No default checkpoint/pointer promotion, retraining, height correction, mesh-mask replacement, collision change, object-count claim or calibrated confidence claim occurred.

The preview currently accepts 8-bit RGB only. It uses native-resolution tiles up to 1024 pixels, 128-pixel overlaps and mean probabilities, with the app's existing 3072-pixel processing cap. Larger scenes therefore use a tiled preview recipe; development scores are not measured accuracy for arbitrary uploads. High-bit-depth imagery requires a separately validated classification radiometry pipeline. The hilly 10-metre demo is explicitly outside the high-resolution training domain.

## Fresh HighBuild comparison

Same 160 corrected development images, same native CUDA/bfloat16 inference, four cities, all supplied building outlines. These positive-only annotations support building-pixel recall/coverage, **not precision, F1, counting accuracy or height accuracy**.

| Diagnostic | V1 | V3 epoch 1 |
| --- | ---: | ---: |
| Annotated building-pixel recall | 79.21% | 82.60% |
| Building interior recall | 79.85% | 83.64% |
| Building boundary recall | 73.08% | 72.52% |
| Copenhagen recall | 88.43% | 86.73% |
| Lyon recall | 87.72% | 87.01% |
| Strasbourg recall | 71.93% | 78.24% |
| Munich recall | 73.86% | 80.50% |
| Outlines at least 80% covered | 64.59% | 69.82% |
| Outlines less than 10% covered | 8.95% | 6.32% |

Result: useful overall gains, **not universal improvement**. Existing GAMUS ground/low-vegetation and road-boundary regression failures still block default promotion. These HighBuild results are development evidence; reserved final geographies remain unconsumed.

Full audit: `outputs/evaluation/highbuild_rgb_classifier_v3_preview_20260919/report.json`; complete image gallery in the same directory, `index.html`. Earlier evidence is preserved.

## Verification

- 40 targeted Python tests passed, covering API, preview tiling, validity, projected area, source-path restrictions, exports, HighBuild audit, existing prediction and mesh behavior.
- 77 viewer tests passed; TypeScript and edited-file ESLint passed.
- Real browser → API → CUDA → class-raster → 3D overlay exercised on urban and forest demos and an uploaded 1024×1024 GeoTIFF.
- Class filters work; unknown pixels stay transparent; scene changes clear the overlay; inspection shows the new class separately. Browser showed no framework error overlay or console error logs during these flows.
- All original files in uploaded job `f0163ba3aed24b19b28eec01bf9627b7` remained byte-identical after classification. Re-uploading the exact same source into job `1bc8a7a5a6624755aa7907ec23735f61` produced **identical pixel values** in raw heights, presentation heights and three-group labels (maximum difference 0).
- Protected height checkpoint SHA256 remains `e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144`; pointer SHA256 remains `a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9`.
- Classifier SHA256: `10fbb39c3d916695627e93d49f2115be18ecd163352b4daa96ba691c0090470c`; verified before loading.

## Next accuracy work

Fix the measured city/class/boundary regressions on development data using the same baseline and preregistered acceptance rules. Do not compensate by quietly lowering acceptance thresholds. Freeze a candidate before one reserved-geography evaluation. Only an accepted classification release should become the default; any semantic-guided height candidate requires its own height-regression evaluation.
