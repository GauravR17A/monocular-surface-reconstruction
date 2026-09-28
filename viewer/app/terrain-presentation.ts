import type { SurfaceRaster } from './surface-query';

/** Keep source-resolution terrain where it fits the GPU budget, with bounded
 * storage for larger/elongated rasters. Object scenes retain their existing mesh. */
export function terrainMeshResolution(width: number, height: number, terrain: boolean) {
  if (!terrain) return { segmentsX: Math.min(width - 1, 512), segmentsY: Math.min(height - 1, 512) };
  const scale = Math.min(1, Math.sqrt(1024 * 1024 / (width * height)), 2048 / Math.max(width, height));
  return { segmentsX: Math.max(1, Math.floor(width * scale) - 1), segmentsY: Math.max(1, Math.floor(height * scale) - 1) };
}

/** Reconstruct continuous terrain for a natural DISPLAY preview. A symmetric
 * Gaussian support rounds raster-scale ridges without flattening whole hills. Never use
 * this copy for elevations, profiles, exports or slope measurements.
 * When a matching terrain reference exists, preserve the object-height residual
 * exactly. Fine-resolution DSMs without that reference retain their geometry. */
export function terrainPresentationSurface(raster: SurfaceRaster, ground?: SurfaceRaster | null, strength = 3) {
  const { width, height, units, gsdX, gsdY } = raster.stats;
  const source = raster.values;
  if (units !== 'm' || !gsdX || !gsdY || !Number.isFinite(gsdX + gsdY) || Math.min(gsdX, gsdY) <= 0 || width < 3 || height < 3) {
    return { values: source, maximumAdjustment: 0, rmsAdjustment: 0 };
  }
  const matchingGround = ground?.stats.units === 'm' && ground.stats.width === width && ground.stats.height === height
    && ground.stats.gsdX === gsdX && ground.stats.gsdY === gsdY;
  const base = matchingGround ? ground.values : source;
  const spacing = Math.min(gsdX, gsdY);
  if (spacing < 5 && !matchingGround) return { values: source, maximumAdjustment: 0, rmsAdjustment: 0 };
  const level = Math.max(1, Math.min(3, Number.isFinite(strength) ? strength : 3));
  // Radius follows map spacing, not image size or location. Separate axes keep
  // physical support consistent for rasters with rectangular pixels.
  const supportM = Math.min(360, spacing * (level * 4 - 2));
  let filtered = base;
  for (const [stride, size, gsd] of [[1, width, gsdX], [width, height, gsdY]]) {
    const sigma = supportM / gsd;
    const radius = Math.min(30, Math.ceil(3 * sigma));
    const spatial = Array.from({ length: radius + 1 }, (_, step) => Math.exp(-step * step / (2 * sigma * sigma)));
    const next = filtered.slice();
    for (let row = 1; row < height - 1; row++) for (let col = 1; col < width - 1; col++) {
      const index = row * width + col, centre = filtered[index];
      if (!Number.isFinite(centre)) continue;
      const coordinate = stride === 1 ? col : row;
      let sum = centre, weights = 1;
      let previousLeft = centre, previousRight = centre;
      for (let step = 1; step <= Math.min(radius, coordinate, size - 1 - coordinate); step++) {
        const left = filtered[index - step * stride], right = filtered[index + step * stride];
        // Missing-data gaps always stop support. Standalone surface rasters
        // additionally protect abrupt structural edges. A supplied terrain
        // reference can be reconstructed continuously; its object residual is
        // added back below instead of confusing steep slopes with structures.
        // Pairing keeps constant slopes planar instead of pulling them uphill.
        if (!Number.isFinite(left) || !Number.isFinite(right)
          || (!matchingGround && (Math.abs(left - previousLeft) > gsd * 3 || Math.abs(right - previousRight) > gsd * 3))) break;
        sum += (left + right) * spatial[step]; weights += 2 * spatial[step];
        previousLeft = left; previousRight = right;
      }
      next[index] = sum / weights;
    }
    filtered = next;
  }
  const values = source.slice();
  let maximumAdjustment = 0, squared = 0, count = 0;
  for (let row = 1; row < height - 1; row++) for (let col = 1; col < width - 1; col++) {
    const index = row * width + col, centre = source[index];
    if (!Number.isFinite(centre) || !Number.isFinite(base[index])) continue;
    // Do not blend a capped displacement back into the jagged input: doing so
    // reintroduces its spikes. Positive Gaussian weights keep terrain within
    // its contributing sample range. Object heights remain an exact residual.
    values[index] = centre + filtered[index] - base[index];
    const adjustment = values[index] - centre;
    maximumAdjustment = Math.max(maximumAdjustment, Math.abs(adjustment));
    squared += adjustment * adjustment; count++;
  }
  return { values, maximumAdjustment, rmsAdjustment: count ? Math.sqrt(squared / count) : 0 };
}
