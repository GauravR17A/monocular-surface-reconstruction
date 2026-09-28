# Compact scene details and original-image zoom

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Delivered

- Outside fullscreen, height information and map scale are collapsed by default, with compact **Height info** / **Map scale** buttons at the bottom of the 3D view.
- Click a button to reveal its panel; click again or use its close button to dismiss. Only one opens at a time. New scenes and navigation mode changes reset the panels.
- Fullscreen keeps the original expanded readouts. Leaving fullscreen collapses them. Heights, source rasters and the live perspective scale calculation are unchanged.
- Clicking the **Original image** thumbnail opens an accessible modal with the current scene's RGB preview.
- Zoom from fit to 8× with buttons, the mouse wheel or +/-; drag to pan within the image edges; **Fit image** or 0 restores the complete image.
- Escape closes the image first, without also leaving fullscreen or clearing an underlying inspection. The native dialog provides modal focus handling; a close button is also available.
- Image dimensions refer to the displayed RGB preview. Zoom does not add source detail or change any height/model output.

## Verification

- TypeScript and targeted ESLint passed.
- All 146 viewer tests passed, including four new tests covering pan bounds, aspect ratio, zoom anchor and limits.
- Browser checks passed for windowed toggles, panel exclusivity, live ruler, image identity, zoom/pan/reset/keyboard handling, fullscreen Escape ordering, scene changes and 390px layout. No page errors captured.
- Evidence: `outputs/runtime/viewport_details_zoom/verification.json` and adjacent screenshots.
- Repeatable browser harness: `scripts/verify_viewport_details_zoom.mjs` using a dedicated agent-browser CDP session.
