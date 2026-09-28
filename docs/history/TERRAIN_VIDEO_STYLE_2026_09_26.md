# Terrain presentation matched toward the supplied video

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

The request concerns continuous, rounded slopes and valley floors across imported TIFF regions. The supplied recording shows a different height-generation pipeline and 1.5x display Z. Its precise elevation field is unavailable, so an exact geometric match or an elevation-accuracy ranking cannot be established from the video.

## Final behavior

- **Smooth / Soft is the default** for compatible terrain rasters. Fine and Balanced remain available; Source restores the original rendered heights.
- The terrain reference uses symmetric Gaussian reconstruction with 2, 6 or 10 source-cell standard deviation, bounded to 360 m physical support. Independent axis spacing handles rectangular pixels. Positive normalized weights avoid overshooting contributing terrain samples.
- The old per-vertex displacement limiter was removed: it retained sharp residuals and could make stronger smoothing locally rougher. Supplied ground references are reconstructed continuously instead of treating steep slopes as structural edges.
- Surface-minus-ground object heights are added back unchanged. Object demos keep their established mesh and display scale. Fine-resolution standalone DSMs below 5 m without a ground reference retain their geometry to protect buildings; standalone coarse rasters protect abrupt breaks. Missing data remains missing.
- Imported absolute terrain now defaults to a gentle **1.5x** display multiplier, matching the visible setting in the reference video. The earlier adaptive 4-12x boost made forests and low-relief areas look disproportionately steep. Truly flat/unknown surfaces receive no automatic boost. True scale and the manual Z slider remain available.
- The implementation is shared by terrain imports, with no filename, location or region-specific logic. Unknown-scale/relative imagery cannot be treated as measured metric terrain. TIFF is a container: image-only TIFFs still need inferred heights or a terrain reference.
- Inspection, slope analysis, profile CSV and exported elevations continue to use the original source data. Mesh boundaries and collision sampling follow the displayed surface. Lighting is unchanged by this revision.

## Actual downloaded Joshimath TIFF

Input SHA256: `989d7ec01634bed96e211ebdc47d395871533a238b20b7314389e670b1a9b6f7`.

The final fresh import is job `2f49ba86b9ed45a5a5679317f3427a14`. It retains all 1024 x 1024 source samples in the mesh (1,048,576 vertices). Same-camera screenshots use identical lighting and 1.5x Z.

| Geometry | Pixel-scale roughness RMS | Display shift RMS | Maximum display shift |
| --- | ---: | ---: | ---: |
| Source | 5.115 m | 0 m | 0 m |
| Fine | 2.231 m | 12.76 m | 118.57 m |
| Balanced | 0.909 m | 49.05 m | 293.09 m |
| Soft (default) | 0.544 m | 85.87 m | 375.68 m |

Roughness is the RMS of a one-pixel Gaussian high-pass residual, excluding a 20-pixel border. The default reduces this diagnostic by **89.4%**; it is a presentation metric, not a ground-truth error. Source relief is 5563.05 m; Soft displayed relief is 5337.21 m. This deliberately removes local terrain detail to achieve the requested rounded appearance. Measurements and exports do not use these adjusted heights.

## Verification

135 viewer tests pass, including continuous smoothing without retained sharp residuals, broad hill preservation, no overshoot, nodata, object residuals, source immutability and gentle automatic relief. TypeScript, targeted ESLint and the production build pass. The build retains its existing large-chunk/static-route-classification notices.

The browser test imported all eight downloaded regional RGB GeoTIFFs through the API, then a standalone float elevation GeoTIFF, and finally the original Joshimath TIFF. Every regional import selected Soft automatically and used 1.5x display Z:

| Region | Source unchanged | Display adjustment RMS |
| --- | --- | ---: |
| Urban, New Delhi | Yes | 2.39 m |
| Plains, Punjab | Yes | 1.28 m |
| Forest, Dandeli | Yes | 6.10 m |
| Hills, Munnar | Yes | 11.11 m |
| Desert, Thar | Yes | 1.40 m |
| Coast, Goa | Yes | 2.75 m |
| Wetlands, Sundarbans | Yes | 1.06 m |
| Rocky, Ladakh | Yes | 11.69 m |

Urban/Sparse/Forest object demos bypass terrain smoothing, with the previous Urban 0.36 vertical scale and 1.7x display multiplier intact. Source/Smooth and strength changes retain camera and source elevation range. All visual layers retain source scale. Browser and shader checks report no errors.

The original TIFF's uploaded bytes are identical to the download, its exported DSM matches the earlier source exactly, and its inspected height matches the source raster. All 744 profile samples agree with source interpolation within 0.043 m, including rounded CSV coordinates. Manual display exaggeration leaves the profile CSV unchanged.

Evidence: `outputs/runtime/video-match-review/` contains per-region screenshots, `final-verification.json`, `regional-source-check.json`, `final-source-check.json`, `geometry-metrics.json`, `same-camera-comparison.json`, `source-close.png`, `soft-close.png`, and the orbit preview. Reproduce the browser run with `scripts/verify_terrain_presentation.mjs <agent-browser.exe> <original.tif> <regional.tif> --all-regions`.

This covers the downloaded regional set and supported terrain/height import paths. It does not establish identical appearance for every possible TIFF encoding, unknown scale, source quality or missing elevation field.
