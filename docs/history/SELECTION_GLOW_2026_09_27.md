# Active inspection boundary

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

The active Inspect / Fly / Walk readout now has a mint boundary with a soft halo. It uses three thin screen-space strokes, depth testing and a short fade-in (instant with reduced motion), not a full-screen bloom pass.

## Behavior

- Inspect click pins the selected boundary with the existing metadata. Pointer movement alone does not unpin it.
- Selecting a different object replaces the old boundary; outlines do not accumulate.
- Explore's two-second aim and click scanner share the same selection/readout source. Moving off the target, changing mode, exiting Explore, starting an import or changing scenes clears the highlight with the active inspection.
- Building solids use their actual transformed mesh edges, including nonrectangular footprints. Vertical display exaggeration remains aligned with the selected solid.
- Raster-only surfaces use the connected region under the selected sample. Disconnected patches of the same class are different dwell targets. Adjacent trees are not falsely claimed to be separate tree instances.
- Region outlines use a bounded, at-most-512-pixel display grid and follow the displayed terrain triangles. Holes and ignored labels stay separate. Missing or undersized regions receive a small local sample locator, not a fabricated object boundary.
- Photo/metric switching preserves the selected boundary and numeric height. The rendering overlay does not change model weights, source heights, collisions or exports.

## Verification

- 142 viewer tests passed, including seven new tests covering disconnected regions, holes, ignored labels, bounded sampling, local locators, terrain draping, transformed building edges, resource cleanup and noninterfering picking.
- TypeScript and ESLint checks passed for the changed TypeScript files.
- Browser verification passed: two different building IDs, connected surface selection, persistence, photo/metric switching with identical readout, mode clearing, empty-space clearing, Fly and Walk dwell/click selection, looking away into empty sky, and scene switching.
- No page errors were captured during the final browser verification.
- This was a display/interaction change, not a new model accuracy evaluation.

Evidence: `outputs/runtime/selection_glow/verification.json` and screenshots in the same folder. Reproducible browser harness: `scripts/verify_selection_glow.mjs`, connected only to a dedicated agent-browser session.

Files: `viewer/app/selection-region.ts`, `viewer/app/selection-glow.ts`, `viewer/app/terrain-workspace.tsx`, `viewer/tests/selection-glow.test.mjs`.
