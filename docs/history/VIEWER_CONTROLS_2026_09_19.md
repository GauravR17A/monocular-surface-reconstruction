# Viewer controls update — 19 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Changes

- Sightline accepts one observer/target pair and then disarms selection. Reset starts another measurement. Clicking Sightline again, Escape, switching modes or loading another scene clears its graphics/state.
- Fly keeps mouse look, WASD/arrows, Space up, Shift down and G for 3x speed. Normalized movement avoids faster diagonal flight.
- Collision uses the displayed terrain triangles and building footprints, with swept substeps and clearance. Gentle slopes raise the camera; walls/cliffs block or allow sliding. Space climbs over obstacles. It follows vertical exaggeration and safely lifts a camera initially embedded by an orbit view.
- Fly only: aiming the centre crosshair at the same building or continuous surface class for two seconds reveals a fixed scanner HUD. Left-click reveals it immediately. Small mouse movements within the target/class preserve the dwell and update values; changing targets, aiming at the sky, or leaving Fly clears it. No confidence percentages are presented.
- Readouts sample full-resolution source pixels rather than downsampled mesh-triangle averages. NoData remains unavailable. Unknown class pixels remain unclassified; RGB assistance does not override an existing classification map.
- The flight HUD includes slope when metric pixel scale exists. DSMs are labelled surface elevation (ASL), not tree/building height. Relative products retain dimensionless units. Building solids report their model-derived object summary; vegetation retains RGB-assisted/model basis. Collision feedback prompts Space to climb when movement is blocked.
- Building roofs retain optical projection, including corrected box-roof UVs. Walls now use three coordinated procedural facade palettes, plaster grain, inset-window patterns, panel joints, floor bands and soft ground-contact shading. Their colors are gently matched to each roof. These are illustrative architectural patterns, not observed facade photographs, detected windows or measured floors; the UI explicitly says so.
- Wall textures follow each wall's true tangent, including diagonal outlines and rectangular scene scaling. Optical roof UVs and all vertex positions are preserved. Color, bump and roughness maps are shared per palette and disposed on scene replacement; no extra geometry or external texture downloads are required.

## Checks

- `npx tsc --noEmit`: passed.
- `npx eslint app/terrain-workspace.tsx app/flight-navigation.ts app/surface-query.ts app/building-facades.ts`: passed.
- `node --experimental-strip-types --test tests/flight-navigation.test.mjs tests/surface-query.test.mjs tests/building-facades.test.mjs`: 14 tests passed (collision/navigation, source-pixel values, class identity during movement, NoData, raster alignment, unknown labels, metric slope, facade tiling and geometry/roof preservation).
- Facade browser checks: urban overview and closer orbit screenshots inspected; new textures render, optical roofs remain intact, disclosure is present, and no page errors were reported.
- Browser: urban, forest and hilly scenes loaded. The HUD remained empty before dwell, appeared despite repeated small mouse movements, switched to CLICK SCAN on an immediate click, and stayed visible during further jitter. Sky aiming cleared the target; leaving Fly removed the HUD. The hill readout explicitly used ASL. Sightline completed and graphics cleared on mode switch. No browser page errors observed.
- Existing development-server log warns about multiple concurrent renderers; no associated browser failure observed during these checks.

Model checkpoints, exported prediction rasters, evaluation scores and backend inference were not changed. Collision protects navigation through the reconstructed scene; it is not evidence that the estimated geometry matches the real world. Experimental decorative tree objects are not physical colliders; their underlying terrain/canopy heightfield is.
