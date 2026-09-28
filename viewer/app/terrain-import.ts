/** Import policy depends on raster metadata, never a filename, place or demo id. */
export function hasTerrainGeoreference(keys: Record<string, unknown> | null, bounds: readonly number[] | null) {
  if (!keys || !bounds || bounds.length < 4 || !bounds.every(Number.isFinite)) return false;
  const model = Number(keys.GTModelTypeGeoKey);
  const crs = Number(model === 1 ? keys.ProjectedCSTypeGeoKey : keys.GeographicTypeGeoKey);
  return (model === 1 || model === 2) && Number.isFinite(crs) && crs > 0
    && bounds[2] > bounds[0] && bounds[3] > bounds[1];
}

export function terrainCalibrationSource(options: { georeferenced: boolean; automatic: boolean; dem: boolean; gcps: boolean }) {
  if (options.dem) return 'dem';
  if (options.gcps) return 'gcps';
  return options.automatic && options.georeferenced ? 'public' : 'none';
}
