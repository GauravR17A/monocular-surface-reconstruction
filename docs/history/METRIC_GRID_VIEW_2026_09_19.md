# Metric grid view — 19 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Neutral startup

The app starts and reloads with a static neutral grid and **“Upload an image to see the 3D simulation”**. No reference is selected and no demo image/height raster or Three.js canvas is loaded automatically. Urban, Sparse, Hilly and Forest remain explicit reference buttons. A standalone height map loaded before any photograph uses a neutral texture and grid view, never an unrelated urban image.

## User-facing behaviour

- The **METRIC GRID** button in the 3D viewport, or **Layers → Metric grid · no photo**, reveals the rendered height surface without optical or façade textures.
- A regular horizontal grid drapes across the surface and flat building tops. Horizontal height bands run across slopes and building walls; these are elevation intervals, **not detected floors**.
- The legend covers the displayed surface plus building tops, states the horizontal cell size and height-band interval, and distinguishes height above ground, absolute elevation and relative units.
- Ordinary images without verified map scale use **pixel** spacing, even if their height estimates are in metres. Projected metric/foot-based raster metadata or explicit metre GSD can establish horizontal scale. Degrees and unreferenced TIFF transforms cannot.
- **PHOTO VIEW** restores the optical surface and original building materials. Fly, Inspect, collisions, the roof-configuration notice and exports remain available. Cosmetic tree proxies are hidden only in grid view.
- Photo/grid changes use a reversible 650 ms eased material blend, including fading façade bump/roughness detail. Geometry and camera position do not change. Reduced-motion preferences bypass the animation.

## Import reveal

While a new image is processed, a neutral blank grid and the real elapsed-status text cover the previous scene. The old 3D renderer pauses during inference. No fabricated model percentage or intermediate detection is displayed.

Once the actual image, height products and footprints are ready, a 3.65-second presentation sequence shows **grid → building shapes → height rise → optical wrapping**. This temporarily scales the displayed mesh vertically, then returns exactly to its original display scale. Predictions and raw metadata never change. The card explicitly says the prediction is ready and this is a visual reveal, not live detection.

**Skip animation** finishes immediately. Reduced-motion users go straight to the finished scene. Fly and inspection are unavailable during the temporary height rise and are restored afterward. Reference-comparison opening is deferred until the reveal ends. Failed inference removes the waiting view and retains the previous usable scene. Replacing a texture does not retrigger the import reveal.

## Integrity and implementation

`viewer/app/metric-grid.ts` supplies bounded grid intervals, scale validation and a texture-free display material. Switching only changes materials/visibility; it does not change mesh positions, raw rasters, predictions, building heights, checkpoints or exports. Colour/contour coordinates account for the current vertical exaggeration without changing the interval labels. Inactive materials are retained for switching and disposed on scene teardown.

The view is a presentation surface, not a bare-earth DEM or an additional model prediction. Smoothing/visual aids can differ from raw height values; use Inspect and exports for those values. Roofs remain flat and roof geometry is unconfigured. Grid axes follow the raster, not an assumed north direction. Horizontal extents are between first and last pixel centres.

## Checks

- 47 focused tests passed: scale handling, material transitions, reversible/reduced-motion timing, reveal stages/skip, navigation, inspection, façades and preserved roof modules.
- TypeScript and focused ESLint checks.
- Browser checks cover real-image uploads, optical/grid switching, inspection and hilly absolute-elevation and relative-raster views.
- The Strasbourg example completed every reveal stage without browser errors. Its inspected building read 14.4 m in both photo and grid views, with the same roof-status notice.
- Observed the photo/grid transition across 36 distinct rendered intermediate frames. A real import showed processing, grid, footprints, heights, texture and completion in order; controls re-enabled afterward.
- Confirmed Skip during the height-rise stage, reduced-motion import without the reveal, and recovery from a simulated failed network request in an isolated browser (previous scene retained, upload re-enabled, no uncaught errors).
- Manali's known-scale DSM displayed 500 m horizontal cells and absolute-elevation bands; a relative Chicago raster displayed pixel cells and dimensionless heights, not metre claims.
