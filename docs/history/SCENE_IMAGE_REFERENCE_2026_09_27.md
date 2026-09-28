# Original-image reference in the 3D viewport

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

The upper-right **Original image** card displays the RGB texture paired with
the active 3D scene. It is a 2D input reference, not a prediction, classification
overlay, second inference, or accuracy claim.

## Behaviour

- Uses `sceneInput.textureUrl` for reference scenes, imported imagery and replacement textures.
- TIFF/GeoTIFF and other server-decoded formats use the same browser-readable RGB preview returned by inference and used on the mesh.
- Shows the complete image with preserved aspect ratio (no crop), plus the current scene label.
- Collapsible with an accessible button; a new image resets the card to expanded.
- Remains in colour during metric, elevation and classification views.
- Shrinks during Inspect/Explore; the reference and height readout share a vertical dock, avoiding overlap.
- Remains inside native or window fullscreen. Narrow views use a smaller preview.
- Hidden in the neutral empty workspace and while a new scene is processing/revealing; failed image loads have an explicit placeholder.
- Source heights, inference, exports, mesh geometry and selection/glow logic are unchanged.

## Verification

- Existing 142 viewer tests passed.
- TypeScript and targeted ESLint passed.
- Dedicated browser: all four Urban/Sparse/Hilly/Forest references; collapse and scene reset; metric view; Inspect; Fly; fullscreen; layout at 390/820/1280/1600 px.
- Real PNG, JPG and TIFF uploads: each displayed image URL matched its own API response's `texture_url`. Old-image previews were absent during processing.
- No browser page errors captured.

Repeatable browser check: `scripts/verify_scene_image_reference.mjs`.
Evidence and screenshots: `outputs/runtime/scene_image_reference/verification.json`.
