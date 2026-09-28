import type { SurfaceRaster } from './surface-query';

export type TerrainPoint = { u: number; v: number };
export type ProfileSample = TerrainPoint & { distance: number; surface: number | null; ground: number | null };
export type TerrainProfileData = {
  samples: ProfileSample[]; distance: number; distanceUnit: 'm' | 'px'; heightUnit: 'm' | 'relative';
  minimum: number | null; maximum: number | null; delta: number | null; ascent: number; descent: number; gaps: boolean;
};
type TerrainStats = SurfaceRaster['stats'] & { minimum: number; maximum: number };
const positive = (n: number | null | undefined): n is number => typeof n === 'number' && Number.isFinite(n) && n > 0;

/** One world metre has the same size in X, Y and Z. Never normalize mountain relief to a fixed height. */
export function terrainDimensions(stats: TerrainStats, worldWidth = 112, analysisStats: TerrainStats = stats) {
  const metric = stats.units === 'm' && positive(stats.gsdX) && positive(stats.gsdY);
  const extentX = Math.max(1, stats.width - 1) * (metric ? stats.gsdX! : 1);
  const extentY = Math.max(1, stats.height - 1) * (metric ? stats.gsdY! : 1);
  return { width: worldWidth, depth: worldWidth * extentY / extentX, metric,
    // Presentation meshes can exclude buildings. Use the complete analysis range
    // and the original cap for unreferenced imagery, shared by vegetation and solids.
    verticalScale: metric ? worldWidth / extentX : analysisStats.units === 'relative'
      ? 4.2 / Math.max(1e-6, analysisStats.maximum - analysisStats.minimum)
      : Math.min(.36, 12 / Math.max(1e-6, analysisStats.maximum - analysisStats.minimum)) };
}

/** Gentle default relief, shared by terrain regions. Never inflate low-relief
 * plains/forests to a fixed mountain-like screen height. Manual Z remains available. */
export function automaticTerrainExaggeration(raster: SurfaceRaster) {
  const { units, gsdX, gsdY } = raster.stats;
  if (units !== 'm' || !positive(gsdX) || !positive(gsdY)) return 1;
  const sample: number[] = [];
  const step = Math.max(1, Math.ceil(raster.values.length / 4096));
  for (let i = 0; i < raster.values.length; i += step) {
    if (Number.isFinite(raster.values[i])) sample.push(raster.values[i]);
  }
  if (sample.length < 2) return 1;
  sample.sort((a, b) => a - b);
  // Ignore isolated extrema when choosing the presentation, never when rendering.
  const relief = sample[Math.floor((sample.length - 1) * .98)] - sample[Math.floor((sample.length - 1) * .02)];
  if (relief <= .01) return 1; // A truly flat raster must remain flat.
  return 1.5;
}

/** Bilinear sampling preserves continuous ridges without inventing values across nodata. */
export function interpolateSurface(raster: SurfaceRaster, u: number, v: number): number | null {
  if (![u, v].every(Number.isFinite) || u < 0 || v < 0 || u > 1 || v > 1) return null;
  const { width, height } = raster.stats;
  const x = u * (width - 1), y = v * (height - 1), left = Math.floor(x), top = Math.floor(y);
  const right = Math.min(width - 1, left + 1), bottom = Math.min(height - 1, top + 1);
  const dx = x - left, dy = y - top;
  const terms = [[top * width + left, (1 - dx) * (1 - dy)], [top * width + right, dx * (1 - dy)],
    [bottom * width + left, (1 - dx) * dy], [bottom * width + right, dx * dy]];
  let result = 0;
  for (const [index, weight] of terms) {
    if (weight <= 1e-12) continue;
    const value = raster.values[index];
    if (!Number.isFinite(value)) return null;
    result += value * weight;
  }
  return result;
}

export function surfaceSlope(raster: SurfaceRaster, u: number, v: number): number | null {
  const { width, height, units, gsdX, gsdY } = raster.stats;
  if (units !== 'm' || !positive(gsdX) || !positive(gsdY) || width < 2 || height < 2 || ![u,v].every(Number.isFinite) || u < 0 || v < 0 || u > 1 || v > 1) return null;
  const x = Math.round(u * (width - 1)), y = Math.round(v * (height - 1));
  if (!Number.isFinite(raster.values[y * width + x])) return null;
  const l = Math.max(0, x - 1), r = Math.min(width - 1, x + 1), t = Math.max(0, y - 1), b = Math.min(height - 1, y + 1);
  const [left, right, top, bottom] = [y * width + l, y * width + r, t * width + x, b * width + x].map(i => raster.values[i]);
  if (![left, right, top, bottom].every(Number.isFinite)) return null;
  return Math.atan(Math.hypot((right - left) / ((r - l) * gsdX), (bottom - top) / ((b - t) * gsdY))) * 180 / Math.PI;
}

/** Measurements always use unexaggerated source elevations and map spacing. */
export function sampleTerrainProfile(surface: SurfaceRaster, ground: SurfaceRaster | null, a: TerrainPoint, b: TerrainPoint): TerrainProfileData {
  const { width, height, units, gsdX, gsdY } = surface.stats;
  const metric = positive(gsdX) && positive(gsdY);
  const dx = (b.u - a.u) * (width - 1), dy = (b.v - a.v) * (height - 1);
  const distance = Math.hypot(dx * (metric ? gsdX! : 1), dy * (metric ? gsdY! : 1));
  const count = Math.min(2048, Math.max(2, Math.ceil(Math.hypot(dx, dy)) + 1));
  const samples: ProfileSample[] = [];
  let ascent = 0, descent = 0, previous: number | null = null, minimum = Infinity, maximum = -Infinity;
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1), u = a.u + (b.u - a.u) * t, v = a.v + (b.v - a.v) * t;
    const value = interpolateSurface(surface, u, v);
    samples.push({ u, v, distance: distance * t, surface: value, ground: ground ? interpolateSurface(ground, u, v) : null });
    if (value !== null) {
      minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
      if (previous !== null) { ascent += Math.max(0, value - previous); descent += Math.max(0, previous - value); }
    }
    previous = value;
  }
  const first = samples[0].surface, last = samples.at(-1)!.surface;
  return { samples, distance, distanceUnit: metric ? 'm' : 'px', heightUnit: units,
    minimum: Number.isFinite(minimum) ? minimum : null, maximum: Number.isFinite(maximum) ? maximum : null,
    delta: first === null || last === null ? null : last - first, ascent, descent, gaps: samples.some(p => p.surface === null) };
}

export function profileCsv(profile: TerrainProfileData) {
  const suffix = profile.heightUnit === 'm' ? 'm' : 'relative';
  return [`distance_${profile.distanceUnit},surface_${suffix},terrain_reference_${suffix},image_u,image_v`,
    ...profile.samples.map(p => [p.distance.toFixed(3), p.surface?.toFixed(3) ?? '', p.ground?.toFixed(3) ?? '', p.u.toFixed(6), p.v.toFixed(6)].join(','))].join('\n');
}
