# Predicted rooftop relief — 19 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## What was fixed

The semantic viewer used one percentile-derived height per connected building footprint. This intentionally clean display discarded smaller raised regions in the raw prediction. The new **Predicted roof relief** layer retains supported raised regions above that existing summary roof. The original optical image is projected onto both the base and the raised geometry; illustrative facade textures remain on the outer walls.

The source is the existing raw metric nDSM, not image brightness, invented roof templates, a reference raster, or a new model. Export rasters, model checkpoints and evaluation metrics are unchanged. This is a rendering improvement, not a claim of better predictive accuracy.

## Safeguards and implementation

- Bounded regular-grid samples inside each existing footprint. Semantic building labels (or the existing 0.3 building-probability threshold for legacy scenes) gate support.
- A 3x3 spatial median rejects isolated height spikes. At least nine supported samples and 65% supported coverage are required.
- At least five samples / 1% of supported samples must exceed the summary roof by the larger of 0.5 m or 7% of its height. These are conservative display heuristics, not calibrated correctness probabilities.
- Lower edge estimates do not pull down the existing solid. Detail is clipped inside the footprint and meets the retained roof at unsupported boundaries. Relative and absolute-DSM input scenes do not silently become metric rooftop nDSM detail.
- Regular-grid triangles avoid long thin triangles across irregular footprints. New roof geometry is bounded to roughly 30,000 sampling points per scene, with per-building budgets of 160–1,400 and a minimum spacing of two source pixels.
- The checkbox toggles the extra roof geometry; existing building shapes and wall textures remain available. Raycasts ignore disabled roof meshes.
- Fly scanning identifies a raised-roof hit as **ROOF POINT · ABOVE GROUND** and reads its raw source pixel. Displayed relief can differ from the raw value due to the baseline floor, median filtering and interpolation. The parent building ID is unchanged, so small aim movements do not restart dwell.
- Collision uses the conservative maximum of the base solid and the roof-detail bounds. It prevents passage through raised roofs, but may keep the camera above lower parts of the same building.

## Exact user-image check

Input: `data/highbuild_reviewer/images/Europe_France_Strasbourg/grid_00074_z19.jpg`.

The existing production model generated job `b3c8c9df2b6d4783b5cde2a6fcde648e` through the normal upload API. It contained **12 building components**; **11** had enough supported raised-height variation to enable this display layer. These are component/rendering counts, not validated building counts or eleven proven rooftop reconstructions.

The same saved inference response was reused in the isolated browser test while editing the viewer, avoiding repeated GPU inference. On/off screenshots used the identical prediction. A Fly click on raised geometry reported a raw roof-pixel estimate (22.0 m in the sampled view) rather than repeating the whole-building summary. That point has not been checked against surveyed truth.

Screenshots: `outputs/runtime/strasbourg-roofs-detailed.png`, `strasbourg-roofs-flat.png`, and `strasbourg-roof-inspect.png`.

## Verification

- 21 targeted viewer tests passed, including coherent rooftop-block retention, isolated-spike rejection, raw-data immutability, unknown/vegetation/NoData handling, relative-input rejection, low-edge protection, concave-footprint clipping, facade/roof UV preservation and navigation regressions.
- TypeScript and targeted ESLint passed.
- Exact-image browser upload/render, relief toggle and Fly roof-pixel inspection checked; no browser page errors reported.

## What remains unresolved

This is not a dormer, chimney, HVAC-unit or rooftop-room detector. It cannot recover shapes missing in the height prediction, resolve all roof slopes, separate touching buildings, restore courtyard holes missing from existing footprints, or infer hidden walls from one overhead image. Model errors can still produce incorrect relief. It also deliberately does not reconstruct lower roof tiers beneath the existing summary height. Fine rooftop accuracy needs appropriate labels, better-resolution predictions and independent validation.
