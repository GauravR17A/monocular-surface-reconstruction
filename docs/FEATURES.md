# Current feature register

Status refers to the source package as of 28 September 2026. Automated checks
support specific contracts; they do not establish every possible input or device.

| Feature | Implementation and evidence | Current boundary |
| --- | --- | --- |
| RGB reconstruction | `api/app.py`, `inference/predict.py`; prediction and API tests | Model artifacts and supported decoding are required; domain shift remains. |
| Bounded GeoTIFF reading | `io/raster.py`; raster/import tests | Interactive processing can downsample; original detail is not recovered. |
| Relative geometry | `inference/relative_depth.py`; normalization tests | Dimensionless until a justified metric mapping exists. |
| Terrain alignment and absolute DSM | `geospatial/calibration.py`; calibration tests | Datum/source quality determines what the composition means. |
| GCP calibration | `geospatial/gcps.py`; GCP and affine-fit tests | Fit residual is not independent validation. |
| Optional public terrain | `geospatial/public_terrain.py`; terrain tests | Network-dependent, coarse and not guaranteed bare-earth truth. |
| Reference comparison | `evaluation/raster_validation.py`, viewer comparison | Requires a compatible independent reference and explicit support. |
| Six-class identification | `api/classification.py`, `inference/rgb_preview.py`, viewer class helpers | 8-bit RGB; V3 regression failures disclosed. |
| Class coverage and export | `coverage_report`, classification raster outputs | Areas require supported projected scale; pixel share otherwise. |
| Orbit, fly and walk | Viewer navigation helpers and tests | Mouse/keyboard-oriented research workspace; broad device coverage is unverified. |
| Point inspection and scanning | `inspection-readout.ts`, `surface-query.ts`; navigation records | Source/class/geometry disagreement is possible and displayed. |
| Class focus and selection glow | `class-focus.ts`, `six-class-layer.ts`, `selection-glow.ts` | Highlighting does not alter physical measurements. |
| Navigation map | `explore-map.tsx`, `explore-map-math.ts` | Scene-relative navigation aid; bounded update rate. |
| Original image reference | `image-reference-view.ts`; image-reference tests | Side/top context is the input image, not an independent measurement. |
| Metric grid and scene scale | `metric-grid.ts`, `scene-scale.ts` | Requires justified metric scale; display exaggeration remains visible. |
| Surface profile and CSV | `terrain-analysis.ts`, `terrain-profile.tsx` | Missing samples stay missing; unknown scale cannot produce metre distances. |
| Source/smoothed terrain | `terrain-presentation.ts`; source-preservation tests | Smoothness is a presentation change, not an accuracy score. |
| Building solids and facades | `inference/viewer_mesh.py`, `building-facades.ts` | Illustrative geometry; fine roof configuration remains unestablished. |
| Roof fitting | `roof-reconstruction.ts`, roof dataset/evaluation modules | Experimental and inactive in the current viewer. |
| Height/semantic candidate gates | `evaluation/height_scorecard.py` and protocol scripts | Historical sealed identities require their original artifacts. |
| Local evidence artifacts | GeoTIFF/JSON/CSV outputs and metadata | Local storage has no automatic durable backup or expiry policy. |

Implementation paths without a viewer prefix are under `src/msr/`. The current
release report identifies which checks ran on the public package. The historical
appendix preserves additional dated UI checks and known harness limitations.
