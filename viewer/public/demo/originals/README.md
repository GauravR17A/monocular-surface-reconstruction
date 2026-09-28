# Monocular Surface Reconstruction demonstration source images

Open `00_OPEN_original_inputs_gallery.html` for a clean, presentation-ready view of the three original optical inputs.

## 1. Urban — Copenhagen

- Original RGB used by the built-in Urban demonstration: `01_URBAN_Copenhagen_original_RGB.jpg`
- Purpose: demonstrate building-heavy urban surface reconstruction and mixed-scene semantic routing.
- Output type shown in Monocular Surface Reconstruction: predicted height-above-ground nDSM.

## 2. Sparse — Amsterdam rural fringe

- Original RGB: `02_SPARSE_Amsterdam_original_RGB.jpg`
- Sample ID: `Europe_Netherlands_Amsterdam__grid_01361_z19`
- Split: held-out HighBuild test chip.
- Independent reference height raster: `02_SPARSE_Amsterdam_reference_nDSM_m.tif`
- Selection rule: chosen using reference building coverage only; predictions were not inspected during scene selection.
- Honest demonstration metrics: RMSE 1.48 m, MAE 0.71 m, correlation 0.873, R² 0.752.

## 3. Hilly — Manali, Himachal Pradesh

- Presentation preview: `03_HILLY_Manali_original_RGB_preview.jpg`
- Full georeferenced optical input: `03_HILLY_Manali_original_georeferenced_RGB.tif`
- Terrain calibration datum: `03_HILLY_Manali_Copernicus_GLO30_terrain_datum_m.tif`
- Optical source: Sentinel-2 L2A item `S2B_43SFR_20251209_0_L2A`, acquired 2025-12-09.
- CRS and grid: EPSG:32643 at 10 m GSD.
- Terrain relief: approximately 1,543 m between the 5th and 95th percentiles.

Important: the Copernicus DEM is used to anchor the Manali result to an absolute terrain datum. It is not presented as an independent reference for an RMSE claim.

## Suggested reviewer narration

“These are the three untouched optical inputs used by our built-in demonstrations. Copenhagen demonstrates dense urban reconstruction, Amsterdam demonstrates held-out sparse-scene validation against an independent height raster, and Manali demonstrates georeferenced hilly terrain calibrated to an absolute elevation datum.”
