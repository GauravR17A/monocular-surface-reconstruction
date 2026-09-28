# HighBuild classification: what works and what needs improvement

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## What was actually tested

The frozen RGB-only SegFormer six-class V1, guarded epoch 8, was run on all
160 native 1024 x 1024 images in the existing corrected HighBuild validation
subset: 40 each from Copenhagen, Lyon, Strasbourg and Munich. This is not
the entire HighBuild dataset. These European cities were excluded from this
standalone classifier's DC/Philadelphia training, but this is already-used
project development/regression data, not an untouched final system benchmark.

The six classes are ground, buildings, water, roads, low vegetation and trees.
The classifier is separate from the active application's protected height
pipeline. No app model, checkpoint, dataset label or height output was changed.
No training was performed.

## Measured results

| City | Images | Annotated-building pixel recall |
|---|---:|---:|
| Copenhagen | 40 | 88.43% |
| Lyon | 40 | 87.72% |
| Strasbourg | 40 | 71.93% |
| Munich | 40 | 73.86% |
| Pooled | 160 | **79.21%** |

This means 14,748,689 of 18,619,411 supplied building-footprint pixels were
classified as building. It does NOT mean 79.21% overall accuracy. In particular,
it does not penalize invented buildings outside supplied footprints: HighBuild
does not certify those unlabelled pixels as non-buildings. Precision, F1, IoU,
building-count accuracy and the other five categories' accuracy are therefore
deliberately unavailable in this audit.

Of the known building pixels, 12.89% were called road and 6.75% ground.
Building-interior recall was 79.85%; the two-pixel inner edge-band recall was
73.08%. Interior errors remain substantial, so this is not just an edge problem.

For 2,581 annotations with rasterizable, image-valid footprint pixels:

- 79.58% had at least half their footprint labelled building.
- 64.59% had at least 80% labelled building.
- 8.95% had less than 10% labelled building.
- Small footprints (<256 pixels; 178 annotations) averaged only **52.86%**
  building coverage; **39.89%** of those small annotations were nearly missed
  (<10% coverage).

These are footprint coverage diagnostics, not instance detection/count scores.
Seven of the contract's 2,588 supplied annotations contribute no rasterized
image-valid pixels and are absent from those object-coverage denominators.
Size bins are pixels, not square metres; geographic scale was not authenticated.

## What the pictures show (qualitative, not independent six-class ground truth)

Eight of 16 previews selected before inference were inspected across all four
cities. Every one of the 160 predictions remains in the gallery, including failures.

- Many large roofs are recognized, but predictions also spill onto adjoining
  streets/courtyards. For example, Copenhagen grid_00469_z18 has 92.7% building
  recall yet visibly overpredicts building regions. High recall alone hides this.
- In Copenhagen grid_00003_z18, an obvious sports field is mostly called ground
  instead of low vegetation. Its 95.2% building recall says nothing about that
  failure because only the building outlines are scored.
- In Lyon grid_00000_z19, substantial visible canopy is labelled ground, although
  several building roofs are found. Tree-versus-ground transfer is inconsistent.
- Munich grid_00112_z19 identifies much of a wooded area as trees but breaks
  some of it into ground and recognizes only 21.4% of the supplied building pixels.
- Copenhagen grid_01943_z18 identifies some open water but also calls a large
  bright industrial ground area a building. There is no measured water accuracy.
- Strasbourg and Munich examples show fragmented boundaries and roof/road
  confusion, matching the measured building-error distribution.

Conclusion: the classifier has learned useful recognition, but it is **not yet
reliable enough to automatically control all app geometry/heights on HighBuild**.
This does not prove a single cause. Geographic appearance, season/shadows,
image scale, class definitions and limited detail are hypotheses to test.

## Improvement plan, in priority order

1. **Get the right evidence and supervision.** Keep these 160 images as a fixed
   development regression suite. Obtain independently reviewed six-class labels
   on representative HighBuild-like imagery, with roofs, asphalt courtyards,
   shadows, grass and small trees all included. Split by source location/scene,
   not neighbouring crops. Label background as unknown where it is not verified.
   HighBuild outlines alone cannot teach or score the other five classes, and
   positive-only building training alone cannot reliably teach false positives.
2. **Broaden the learning geography.** Mix the existing GAMUS training data with
   appropriately mapped, licensed multi-region semantic labels. OpenEarthMap is
   a candidate, not yet downloaded for this step: it includes imagery from 97
   regions in 44 countries, with eight classes requiring an explicit six-class
   mapping. Do not reuse consumed Christchurch as a fresh final test. Reserve
   other entire locations before training. Start with a bounded, audited subset.
3. **Test scale/appearance robustness before a large run.** Current V1 code uses
   384-pixel crops and rotations/flips, but no color/scale augmentation. Compare
   a matched continuation with controlled scale, brightness/contrast/color,
   JPEG/blur perturbations, while keeping label alignment and independent image
   validity. Verify actual source pixel scales when available; filenames are not
   trusted ground sampling distances. Any gain is a hypothesis until measured.
4. **Then test finer detail.** Small footprints fail disproportionately. Compare
   a detail-preserving decoder or boundary-supervised training separately from
   the data/augmentation change. Roof-versus-road and grass-versus-ground need
   semantic correction too; merely sharpening the existing masks is insufficient.
5. **Keep six-class and height safety separate.** On exhaustive labels measure
   every class's precision, recall, F1 and IoU, plus road/roof boundaries, small
   objects and water/shadow mistakes. Use a fixed baseline and predeclared city
   regression limits. Leave the production height pipeline frozen; classification
   can become an overlay only after validation, not silently alter heights.

Do not simply extend the old sampling run: the controlled RGB sampling V2
experiment failed its primary benefit and safety gates. Its exploratory epoch 2
is preserved but is not an established improvement over its matched control.

Research basis (supports directions, not guaranteed Monocular Surface Reconstruction gains):

- [OpenEarthMap benchmark](https://arxiv.org/abs/2210.10732): multi-region semantic
  coverage and explicit geographic/domain-generalization evaluation.
- [SegFormer](https://arxiv.org/abs/2105.15203): multiscale encoder architecture;
  architecture properties alone do not remove appearance/domain shifts.
- [AerialFormer](https://arxiv.org/abs/2306.06842): aerial imagery's small-object
  challenges and high-resolution, local/global feature aggregation.

## Artifacts and checks

- `outputs/evaluation/highbuild_rgb_classifier_20260919_v1/index.html`: all 160
  original/prediction pairs, fixed city/sample order, no hidden failure cases.
- Same directory: `report.json`, `protocol.json`, `per_image.jsonl`,
  `per_object.json`, raw six-class ID PNGs, colored masks and 16 comparison panels.
- `scripts/audit_highbuild_rgb_classifier.py`: reproducible, non-overwriting run.
- Four focused audit tests passed. Known checkpoint hash authenticated before
  loading; classification source authenticated; corrected manifest/mask/COCO
  consistency checked; production pointer/checkpoint hashes unchanged afterward.
- Inference plus artifact generation completed in about 51 seconds on the RTX
  4070. The evaluation process exited and released its GPU allocation.

Checkpoint SHA-256: `4940b2d4ae8817370384ccb54da9f73aee1b6bf3614435d8c077bf3e3363d021`.
