# Explore: Fly and Walk

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

**Follow-up update:** The movement settings below describe the initial implementation. Normal pace is now 4.3 m/s with a game-style step bob (up to 4.5 cm downward dip), quicker acceleration/deceleration and the same G boost. See `INSPECT_CLASSES_AND_WALK_POLISH_2026_09_19.md` for current settings and verification.

## Use

Load an image or select a landscape reference. Choose **Explore**, then **Fly** or **Walk**.

| Control | Fly | Walk |
| --- | --- | --- |
| Mouse | Look around | Look around |
| WASD / arrow keys | Move in the viewing direction | Move along the surface; looking up does not lift you |
| Space / Shift | Rise / descend | No vertical movement |
| Hold G while moving | 3× speed | 3× pace |
| Aim for 2 seconds / left-click | Surface scanner | Same surface scanner |
| Esc | Exit to Orbit | Exit and restore the pre-walk Orbit view |

Walk uses a 1.80 m eye offset above the reconstructed surface for metre-valued scenes. Optional gentle head motion follows actual distance travelled, eases to rest, and is disabled by the system's reduced-motion preference. It can also be turned off in the Explore chooser.

## Movement and collision

- Foot movement uses camera yaw only, normalised diagonal input, and gradual acceleration/deceleration.
- Map-scaled scenes use 1.4 m/s walking and 4.2 m/s with G.
- The feet follow the same triangle-interpolated surface used in the visible mesh.
- The body stays within the image boundary. Missing height pixels are not treated as zero ground.
- Building footprints and building-labelled pixels block entry. Tall classified canopy is not assumed to be walkable ground.
- Substepped movement guards against tunnelling through walls. Sliding allows movement alongside obstacles.
- Slopes over 40 degrees and abrupt body-sized height discontinuities are rejected. This is a conservative navigation aid, not a route-safety system.
- Entry searches for an open starting point near the orbit target. If none is found, Walk reports that and suggests Fly rather than placing the camera on a roof.
- Close clipping adapts to human scale, including large hilly scenes. Walk uses a 65-degree field of view; the original camera settings are restored on exit.

## Honest limitations

The scene is a predicted/DEM-assisted surface, **not a surveyed pedestrian environment**. Walk does not recover ground underneath tree canopies, interiors, bridges, stairs, or unseen façades. Collision quality depends on predicted classes and geometry. A noisy reconstructed surface can block travel; use Fly to cross areas that cannot support walking.

If horizontal scale is unknown, pace is explicitly labelled approximate. A dimensionless relative-height input gets a human-scale preview, not a metre-scale claim. Display exaggeration still applies; no model heights or exports are changed. The small head bob adds at most 2.4 cm to the nominal eye offset.

## Preserved features

Neutral startup, manual reference selection, import reveal, metric-grid transition, first-person target dwell/click scanning, flat roofs and pending-roof-height message, and existing Fly collisions/controls remain intact. Model checkpoints and inference were not modified.

## Verification

- TypeScript and focused ESLint checks passed.
- 56 targeted tests passed, including 9 new walking tests: pitch-independent input, diagonal speed, slopes in both directions, cliffs/drops, sprint wall collision/sliding, NoData/borders, safe spawning, anisotropic horizontal scaling, head motion/reduced motion, invalid scale and bounded large deltas.
- Live browser: neutral startup; Urban → Explore → Walk; eye offset 1.80 m at rest; Space/Shift leave position unchanged; 2-second aim and click both show the building scanner; Esc restores the original orbit camera.
- Live hilly reference: ground height changed while walking and the camera retained its eye offset. Normal movement covered about 2.06 m and boosted movement about 6.01 m in comparable intervals (frame timing/acceleration apply).
- Live Fly regression: Space raised the camera about 12.60 scene units, and Space+G about 37.78 in comparable intervals; mouse-look and pointer lock worked.
- Live metric-grid Walk: scanner showed CLICK SCAN, and with walking motion disabled the eye offset stayed exactly 1.80 m during movement. No browser runtime errors were recorded.

Screenshots: `outputs/runtime/explore-options.png`, `walk-urban.png`, `walk-hilly.png`, `walk-metric-grid.png`.

Primary implementation: `viewer/app/walk-navigation.ts`, `viewer/app/terrain-workspace.tsx`, `viewer/app/globals.css`; pure regression tests: `viewer/tests/walk-navigation.test.mjs`.
