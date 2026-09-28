# Monocular Surface Reconstruction research Prototype Runbook

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Pre-demo preparation

1. Keep the laptop connected to power and select its performance GPU mode.
2. Run `scripts/preflight_demo.ps1` from the repository root.
3. Select only a reviewed held-out checkpoint with
   `scripts/select_showcase_checkpoint.ps1`.
4. Build the viewer with `npm --prefix viewer run build`.
5. Start the stable local build with
   `scripts/start_prototype.ps1 -Production -OpenBrowser`.

The launcher records its API and viewer PIDs under `outputs/runtime`. Use
`scripts/stop_prototype.ps1` to stop only those recorded process trees.

## Recommended reviewer flow

1. Open the included sample to demonstrate orbit, fly, optical texture, height
   colours, wireframe, point height, and local slope without relying on network
   access.
2. Upload a normal JPG/PNG and explain that its output is a dimensionless rDSM;
   no absolute elevation is claimed because the image has no spatial datum.
3. Upload a GeoTIFF and attach a known terrain DEM, GCP CSV, or enable the public
   coarse-terrain fallback. Download the resulting absolute DSM and metadata.
4. If reference LiDAR/nDSM is available, attach it before inference and show the
   aligned RMSE, MAE, bias, correlation, R2, and signed-error raster.
5. Use raw exported height for numerical claims. Vertical exaggeration and the
   experimental tree-object layer are visualization controls, not extra model
   accuracy.

## Product interpretation

- `height_above_ground_m.tif`: predicted object/surface height above local
  ground (nDSM).
- `rdsm_relative.tif`: relative surface relief for non-georeferenced imagery.
- `dsm_absolute_m.tif`: terrain elevation plus non-negative predicted surface
  height, produced only when an absolute terrain datum exists.
- `gcp_calibrated_surface_m.tif`: relative geometry affinely calibrated from
  supplied elevation control points.
- `validation_signed_error_m.tif`: prediction minus reference in metres.
- `metadata.json`: processing grid, checkpoint, calibration, public-terrain
  attribution, reference validation, and warnings.

## Honest limitations to state

Single-view RGB cannot guarantee survey-grade height everywhere. Fine roofs,
small bushes, shadowed canopy, unseen ground under dense forest, and domains far
from training data remain the hardest cases. Public coarse terrain supplies the
absolute ground datum but does not replace LiDAR object truth. The prototype is
an integrated, measurable estimation system—not a substitute for certified
survey products.

## Recovery

- API error: inspect `outputs/runtime/api_stderr.log`.
- Viewer error: inspect `outputs/runtime/viewer_stderr.log`.
- Clean restart: run `scripts/stop_prototype.ps1`, then the production launcher.
- Failed new model: leave the protected pilot selected; never point the demo at
  `checkpoint_latest.pt` merely because it is newest.
