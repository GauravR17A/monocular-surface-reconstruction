# Inspect class isolation, faster Walk and stable import grid

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## What changed

- **Walk**: normal pace is now 4.3 m/s (previously 1.4); hold G for 3×. This is game-style exploration, not a travel-time estimate. Starts/stops are more responsive. Footstep bob now includes small vertical dip, side sway, pitch and roll; sprint cadence is limited so boost does not cause frantic head shaking. Bob stops with actual motion and respects the existing toggle and reduced-motion preference. The nominal standing eye height remains 1.80 m.
- **Inspect → Class explorer**: Surface, Buildings and Vegetation show pixel coverage and can be selected individually. The photo transitions to the metric grid, existing geometry flattens, and only the selected class rises. Cyan/amber/green tint and emissive edge lighting highlight the chosen class. Unselected geometry is display-flattened, not re-predicted as zero. All heights restores the complete scene. Compact controls work in fullscreen and at widths where the analysis rail is hidden.
- **Import grid**: the empty-state and processing grids now have identical top, height, opacity and perspective. The loading overlay fades out over 650 ms when a reconstructed scene is ready, instead of jumping directly into a differently projected grid. Reduced motion skips the crossfade/reveal.

## Data integrity and limits

Class isolation changes only the presentation mesh, building display scale and materials. Original raster data, collision-height buffers, checkpoints and exports are not modified. Leaving Inspect restores geometry before Fly/Walk or sightline measurements. Selecting Photo view restores all heights. Point inspection continues to report original heights, not the temporary animation height; clicks during the rise animation are ignored.

The categories match the viewer's resolved three-class map and visible building footprints. If semantic labels are absent, the existing RGB-assisted vegetation presentation is explicitly labelled. Unknown/missing-height pixels are not assigned a fabricated named class. Coverage is the fraction of valid-height pixels in the displayed classification map, not a building count or confidence score. Six-class identification is not claimed. Surfaces with near-zero predicted relief remain flat; highlighting does not invent heights.

## Verification

- TypeScript and focused ESLint checks passed; no browser runtime or shader errors recorded.
- 63 targeted automated tests: walking, flight, per-pixel inspector, class animation/isolation/coverage, metric materials, import reveal, façades and preserved roof utilities.
- Live Urban reference: three category selectors, flatten/rise phases, rapid switching and All heights restore; final weight vectors were exactly one active class, or all ones on reset.
- Live Strasbourg upload: resolved three-class output with 38.2% surface / 52.5% building footprint / 9.3% vegetation for that result. Building inspection retained the original 14.4 m value in isolation view.
- Live import instrumentation: default, processing and fade-out grids all retained top 128.797 px, height 312.797 px and the identical CSS perspective transform in the test viewport; fade-out opacity was observed below 1 before removal. Import finished normally.
- Live hilly walking: approximately 6.19 m normal movement then 18.35 m boosted movement in comparable intervals. The previous run gave approximately 2.06 m / 6.01 m, respectively. Surface constraints and frame timing affect actual distance. Step bob eased back to the nominal 1.80 m eye offset at rest.
- Live fullscreen: compact selector successfully isolated Vegetation. At 980 px viewport width the analysis rail was hidden, compact controls remained visible, and All heights remained usable.

Implementation: `viewer/app/class-focus.ts`, `surface-query.ts`, `metric-grid.ts`, `walk-navigation.ts`, `terrain-workspace.tsx`, and `globals.css`. Tests: `viewer/tests/class-focus.test.mjs` plus updates to metric-grid/walking tests.
