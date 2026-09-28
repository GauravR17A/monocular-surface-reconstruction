# Smoothness and accuracy review — 26 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Evidence reviewed

- User recording: `D:/Downloads/Recording 2026-09-26 141802.mp4`, 157 frames, approximately 5.21 seconds, 1252 × 812. The full frame sequence was decoded and inspected at half-second intervals, including the last frame. Its visible label reads **Absolute DSM terrain · DA3MONO-LARGE prior · rendered LOD 0 · 1.5× display Z**.
- Our same downloaded Joshimath RGB GeoTIFF and generated job `e87ecbe2eebb4ae3a77767ff04ebb070`. Its active presentation multiplier was 1×, not the 12× automatic boost used on some other low-relief regional imports.
- Our source and rendered-mesh sampling code. The initial investigation left rendering unchanged. The subsequent visual improvements below change presentation only; prediction and source elevation values remain unchanged.

## Current implementation

The stronger reconstruction and consistent display scale supersede the earlier limited filter. See [Terrain video-style implementation and verification](TERRAIN_VIDEO_STYLE_2026_09_26.md) for current defaults, quantitative changes and all-region browser evidence.

The earlier filter capped each vertex displacement and retained a sharp source residual. Its Soft setting could even increase local high-frequency roughness. The replacement removes that limiter, reconstructs a continuous terrain reference and preserves object-height residuals separately. The brief lighting experiment was removed; subsequent work concerns geometry and display scale.


## Findings

The recorded scene looks smoother, especially around its lower valley surfaces. The label establishes a different depth prior and display multiplier; it does not expose its DEM source, smoothing parameters, mesh dimensions, calibration, exported heights, or validation error. LOD 0 alone does not establish a particular mesh resolution. The same RGB TIFF does not guarantee the same height field.

Our surface on this file is dominated by the terrain reference, not learned building or vegetation heights. It spans 1139.491–6702.545 m, while added object heights range from 0.060 to 6.793 m, with a median of 0.412 m. A one-pixel Gaussian high-pass diagnostic gives RMS amplitudes of 5.106 m for the terrain reference, 5.106 m for the final DSM, and 0.030 m for added object heights. For source locations with gradient slope below 5°, the corresponding amplitudes are 3.327 m, 3.327 m, and 0.024 m. These are roughness diagnostics, not ground-truth errors: genuine terrain features also contribute to a high-pass residual.

Before this change, the browser limited the 1024 × 1024 raster to 512 × 512 mesh cells (513 × 513 vertices). Vertex spacing was approximately 65.79 × 65.46 m, versus source sample spacing of 32.92 × 32.76 m. Reconstructing that mesh's bilinear vertex sampling and triangular faces at all source centres produced these differences from the source DSM:

| Metric | Difference |
| --- | ---: |
| RMS, all source centres | 4.986 m |
| 95th percentile absolute difference | 10.320 m |
| Maximum absolute difference | 75.006 m |
| RMS where source slope is below 5° | 3.153 m |

This is a measurable approximation in the displayed geometry, not a demonstrated error against actual ground. The raw exports and inspection samples retain the original raster values. Finite mesh resolution can cause visible facets, but finer geometry alone cannot prove which small terrain variations are genuine.

A HEAD request for the reference tile `https://s3.amazonaws.com/elevation-tiles-prod/geotiff/12/2953/1681.tif` returned `x-amz-meta-x-imagery-sources: srtm/N30E079.tif`. This identifies SRTM for that sampled tile, not necessarily every tile in the mosaic. [Mapzen's source documentation](https://github.com/tilezen/joerd/blob/master/docs/data-sources.md) describes mixed sources and oversampling. [USGS explains that SRTM elevations can contain canopy/building effects](https://www.usgs.gov/special-topics/significant-topographic-changes-in-the-united-states/science/accuracy-assessment). Consequently, treating every public terrain tile as bare earth and adding object heights can double-count surface features; the existing "bare-earth" assumption is not generally justified.

[DA3's official model description](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-zoo) identifies DA3MONO-LARGE as relative monocular depth. A downstream pipeline may calibrate it to a DEM, but the recording alone does not show whether or how that was done. A larger model and smoother rendering do not establish better geographic elevation accuracy.

## What should improve next

1. Extend the bounded source-resolution mesh to adaptive level of detail for much larger rasters, with filtered reduction. The current implementation improves the tested 1024-pixel imports; it does not claim unlimited resolution or universal source quality.
2. Track whether an elevation source is a bare-earth DTM, surface DSM, or unspecified mixture. Only add ground-relative building/canopy heights to an appropriate ground reference. Retain provider, resolution, date, datum and uncertainty metadata.
3. Use conservative, terrain-aware artifact handling only when justified. A smoother preview may be useful, but must not silently replace analytical elevations or erase real ridgelines, buildings, and vegetation.
4. Compare both exported height products against independent, aligned reference elevations (survey/LiDAR where available, or a documented independent DEM comparison). Match CRS, vertical datum, resolution, extent and product type; report bias, MAE and RMSE separately for flatter terrain, hills and surface objects. Agreement with the DEM used to generate a result is not independent validation.

The recording supports a visual-quality comparison. A defensible accuracy ranking requires the other pipeline's height output and an independent reference. No accuracy improvement from smoothing or a DA3 model swap has been established here.

Local analysis artifacts: `outputs/runtime/smoothness-review/contact-sheet.jpg`, `roughness-analysis.json`, `mesh-sampling-analysis.json`, and `mesh-fidelity.png`.
