# Terrain imports and measured profiles — 26 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

The shared import and rendering paths now preserve terrain relief instead of normalizing every mountain scene into a fixed-height mesh. These changes apply across imported scenes; no filename, place or demo identifier selects special terrain behavior.

## Import and rendering behavior

- Georeferenced RGB TIFF imports request a public terrain reference by default. The decision uses GeoTIFF CRS and extent metadata. Uploaded DEMs or GCPs take priority. Users can explicitly disable automatic calibration.
- Ordinary imagery without map coordinates retains the existing surface-height workflow. TIFF is a container format, not evidence of an elevation datum. Missing terrain or scale is labeled; metric terrain cannot be recovered from format alone.
- Automatic lookup keeps the preferred source zoom for small scenes and chooses a coarser source for larger extents within the tile budget. The output metadata records actual source URLs, zoom and tile count. Unsupported extents or download failures show an error; users can crop or supply a DEM.
- A calibrated DSM is rendered directly. The above-ground height product is no longer substituted for a mountainous absolute surface. Standalone height-map imports remain available.
- Metric meshes use a consistent scale across horizontal distance and elevation, with independent X/Y pixel spacing. Z × 1 gives source proportions; exaggeration changes display only. Web Mercator output metadata now accounts for local ground scale instead of treating projected map metres as ground metres.
- Bilinear mesh sampling retains continuous terrain. Nodata remains missing and is excluded from triangles rather than becoming a zero-height cliff.
- Optical, elevation, shaded-relief, slope, contour and metric-grid views share the same geometry. Fit-terrain, top-down and reset-exaggeration controls frame the actual extent and relief.

## Measurements

Point inspection reports source surface elevation, the available aligned terrain reference, their difference, local slope and raster pixel. The elevation legend shows the source range and relief. Profile mode accepts two clicks and displays a measured cross-section with distance, endpoint elevations, elevation change, ascent/descent and CSV export. Profiles remain available when the right sidebar is hidden. Nodata gaps are retained, and measurements do not change with visual exaggeration.

The public terrain reference is coarse and can contain source artifacts or vegetation/structures. Its datum and the predicted object heights determine the resulting DSM's accuracy. This work does not establish survey accuracy or improve the existing semantic classifier. Horizontal measurements use local centre-grid spacing; they are not a full geodesic measurement across arbitrary continental extents.

## Regional presentation update

Eight additional region TIFFs were prepared in `Downloads/Monocular Surface Reconstruction_Region_TIFFs`: New Delhi urban, Punjab plains, Dandeli forest, Munnar hills, Thar desert, Goa coast, Sundarbans wetlands and Ladakh rocky mountains. These are RGB GeoTIFF mosaics of public EOX Sentinel-2 cloudless 2024 map imagery, with source URLs, hashes and previews. They are not elevation ground truth.

Testing exposed weak visual depth in low-relief scenes at true scale. Georeferenced absolute DSMs select a clearly labeled automatic display exaggeration from the central 96% of source heights and map extent, bounded to 1–12×. A constant raster never receives invented relief. The physical mesh scale and all source samples remain unchanged. Users can select True scale, adjust the slider or return to Auto relief. Valid outer edges of absolute DSMs extend down to the scene minimum to make terrain volume visible; these edges do not represent measured subsurface geology. An Elevation colours shortcut makes the source elevation ramp accessible directly from the legend.

Unreferenced imagery retains the original urban/object presentation: `min(0.36, 12 / analysisRange)` world units per predicted metre, with a labeled 1.7× display multiplier. Both vegetation and building solids use that same scale. The complete analysis range determines the scale, rather than a display mesh that may have removed buildings. This corrects the inflated object-height regression without changing predicted heights. The original closer camera is retained for these scenes, with portrait fitting; georeferenced terrain keeps extent-based framing. Absolute terrain sidewalls are not added to nDSM object scenes. TIFFs without a CRS and usable transform receive the same unreferenced treatment as ordinary images.

The viewport toolbar now uses content-sized rows and container queries, keeping navigation, relief controls and actions inside their available width. Mobile header controls stay in layout instead of overlaying the toolbar; sidebar scrollbars use the application's dark palette.

## Verification on the user's downloaded TIFF

Final browser import used `C:\Users\researcher\Downloads\joshimath_s2cloudless_2024.tif`, with automatic terrain left at its default setting. The uploaded bytes match the original file:

`989d7ec01634bed96e211ebdc47d395871533a238b20b7314389e670b1a9b6f7`

- Source and generated rasters: 1024 × 1024, EPSG:3857, original transform retained.
- Ground spacing: 32.924868805 m × 32.761287028 m; independently checked using transformed geographic coordinates.
- Rendered surface range: 1139.491–6702.545 m; relief: 5563.054 m.
- DSM equals terrain reference plus predicted height to within 0.000245 m of float32 rounding.
- Inspected pixel (561, 565): surface 3349.638672 m, terrain reference 3349.218750 m, difference 0.419922 m. Displayed values agree after rounding.
- Profile: 812 samples over 26,673.780 m. Independent bilinear resampling agrees within 0.041 m, including the CSV's six-decimal UV rounding. CSV download completed and all 37,067 bytes matched the displayed profile export.
- Browser verified unchanged measurements at Z × 4, all terrain styles, contour switching, top-down framing and a 900 × 760 compact viewport. No browser errors were reported.
- TypeScript and targeted ESLint checks passed; 120 viewer tests passed after the presentation update; the prior 18 relevant backend tests remain applicable to the unchanged backend. Production build passed, with the existing large-bundle warning and vinext route-classification notice.

Reproducible checks:

```powershell
node scripts/verify_terrain_analysis.mjs <agent-browser-executable> C:/Users/researcher/Downloads/joshimath_s2cloudless_2024.tif
./.venv/Scripts/python.exe scripts/verify_terrain_source.py
```

Screenshots, downloaded CSV and machine-readable browser/source verification reports are under `outputs/runtime/terrain-final/` (local runtime artifacts). Browser download verification uses the Chrome protocol with a Windows-native directory because the agent-browser download helper canceled its downloads on this host.

## File drop, clipboard and visible scale follow-up

The main image import supports file selection, dragging files anywhere in the workspace, Ctrl+V / Command+V image paste, and a Paste image button. All routes use one validator and the existing inference/calibration pipeline. Supported RGB formats are TIFF/GeoTIFF, PNG, JPEG, WebP, BMP and JPEG 2000. Multiple supplied images open an explicit chooser; importing is limited to one active scene at a time. Unsupported, empty and over-512 MB files receive feedback before inference. Normal text-field paste is unaffected. Clipboard denial or text-only content has fallback guidance, and drop feedback also works in fullscreen.

Original file drops preserve GeoTIFF bytes and metadata. Clipboard image pixels commonly lack the original map coordinates; the UI therefore recommends dropping the original GeoTIFF for geospatial work. The API handles ordinary grayscale/palette PNG images as RGB, while single-band elevation TIFFs remain the separate **Load existing height map** workflow. JP2 can request terrain conditionally after the backend checks its CRS; manual DEM/GCP inputs keep priority.

The previous decorative, unnumbered scale bar is replaced with a numeric ruler that updates when the camera moves or zooms. It measures horizontal distance on a plane through the view centre, using metres when ground scale is known and pixels otherwise. The adjacent readout includes X/Y extents, calibrated grid spacing, the full rendered Z range including building solids, and display exaggeration. Photo and analytical views use the same calibrated floor grid. Unknown map scale is stated rather than invented.

Follow-up validation: 125 viewer tests, 31 relevant backend tests, TypeScript, targeted ESLint and production build pass. Twenty end-to-end browser checks include real OS file-drop events, actual clipboard PNG paste by keyboard and button, all supported formats, multiple-file selection, busy import handling, fullscreen, and ten widths from 320–1920 px. Additional checks cover clipboard-denial feedback, mobile camera-button separation and real wheel zoom; the ruler agrees with an independent camera projection within 0.5%.

The final imported source was again the original downloaded Joshimath TIFF, now through drag-and-drop. Job `e87ecbe2eebb4ae3a77767ff04ebb070` preserves its SHA-256, grid and CRS, the 1139.491–6702.545 m surface range, and terrain-plus-height composition within 0.000245 m. Reports and screenshots are in `outputs/runtime/import-review/`, including `verification.json`, `final-ui-checks.json`, `source-check.json`, `05-urban-final.png` and `09-final-verified-joshimath.png`.

```powershell
./.venv/Scripts/python.exe scripts/data/create_import_test_fixtures.py
node scripts/verify_image_imports.mjs <agent-browser-executable> C:/Users/researcher/Downloads/joshimath_s2cloudless_2024.tif
```

## Reference video

Reviewed the full available transcript and storyboard frames spanning the 4:49 reference demo. YouTube playback itself returned an error, so uninterrupted audiovisual playback was unavailable. The implemented changes address visible terrain relief, explicit elevation/depth context and measured profiles; this is not a claim of complete feature parity with that demo.
