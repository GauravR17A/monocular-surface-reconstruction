# Explore mapping — 27 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Delivered

- Fly and Walk replace the original-image reference card with a live, heading-up minimap of the same source image.
- A high-contrast glowing player arrow, direction cone and subtle pulse show horizontal position and facing direction. Reduced-motion preferences disable the pulse.
- The image scrolls and rotates with the camera; Walk uses closer context than Fly. Coverage expands with travel speed and remains bounded by the scene footprint. Changes in coverage ease smoothly.
- M toggles a native modal covering the complete browser viewport, including inside the app's fullscreen mode. The whole image is visible with the player arrow and a dashed outline of the minimap coverage.
- Movement, jumping, look and scanner updates pause while the full map is open. M or the close button resumes Explore at the same location. Esc closes the map without also exiting Explore; click the scene to recapture the mouse if needed.
- Outside-image positions are explicitly marked in amber, with the arrow pinned at the map edge.
- Leaving Explore restores the original-image reference and existing enlargement controls.

## Position and scale

The arrow uses camera X/Z in Fly and foot X/Z in Walk, not the crosshair's intersection with a roof or hill. Walk head bob, altitude and display exaggeration cannot move the horizontal marker.

Both positive horizontal pixel scales are required for metre labels. Otherwise the map uses pixel distances and explicitly says the geographic scale is unknown. Full-map orientation is image-up, not assumed north. No additional height inference or geolocation is fabricated.

Typical known-scale local coverage starts at 180 m for Walk and 420 m for Fly, with a lower limit of 24 source cells on coarse imagery and up to four seconds of horizontal travel visible. Unknown-scale coverage uses a bounded fraction of the image footprint. Small scenes cap coverage near their complete footprint.

## Performance and implementation

The React performance review kept transient camera poses in refs. Small canvas views render at a bounded rate (approximately 30 Hz), cap pixel density at 2, skip drawing in hidden tabs, and clean up animation frames and ResizeObservers. No new dependencies, backend changes, classifier changes or height-model changes were needed.

Implementation: `viewer/app/explore-map.tsx`, `viewer/app/explore-map-math.ts`, and the existing terrain workspace/styles.

## Verification

- All 155 viewer unit tests pass, including 9 new map geometry/scale tests.
- TypeScript no-emit check passes.
- Targeted ESLint check covers the changed TypeScript/React implementation.
- Dedicated browser verification passes: Orbit shortcut isolation; image-to-camera registration; look rotation and movement tracking; complete M-screen coverage; paused navigation; M-repeat protection; close-button behavior; restored original reference; Walk context; paused jumping; Esc priority inside fullscreen; Urban, Sparse, Hilly and Forest scene mapping; and full-map layout at 390 px width.
- No page exceptions in the successful integration run.
- Evidence: `outputs/runtime/explore_map/verification.json` and adjacent screenshots. Repeatable harness: `scripts/verify_explore_map.mjs` using the dedicated agent-browser session's CDP endpoint.

The mapping feature reuses each scene's RGB preview, including the backend-produced preview for TIFF uploads. This verification exercised the four existing landscape demonstrations; it did not start a new upload/inference job or alter saved predictions.
