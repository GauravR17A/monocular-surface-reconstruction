/** Point readouts use source pixels, not downsampled mesh triangle averages. */
export type SurfaceRaster = { values: Float32Array; stats: { width: number; height: number; units: 'm' | 'relative'; gsdX?: number | null; gsdY?: number | null } };
export type ClassificationStatus = 'idle' | 'waiting' | 'loading' | 'ready' | 'unsupported' | 'error';
export type InspectionData = {
  experimentalClass?: string | null;
  classificationStatus?: ClassificationStatus;
  sampleUv?: readonly [number, number];
  kind: 'building' | 'vegetation' | 'surface';
  rawValue: number;
  displayValue: number;
  buildingProbability: number | null;
  vegetationProbability: number | null;
  confidence: number | null;
  rgbVegetation: boolean;
  slopeDegrees: number | null;
  terrainElevation?: number | null;
  aboveGroundHeight?: number | null;
  objectHeight?: boolean;
  roofPoint?: boolean;
  roofReconstruction?: boolean;
  classBasis?: string;
  targetKey?: string;
};
export type SurfaceQuery = {
  raw: SurfaceRaster; display: SurfaceRaster;
  ground?: SurfaceRaster | null;
  building: SurfaceRaster | null; vegetation: SurfaceRaster | null;
  semantic: SurfaceRaster | null; confidence: SurfaceRaster | null;
  rgba: Uint8ClampedArray | null; textureWidth: number; textureHeight: number;
};

// u,v are top-left image coordinates. Different raster resolutions are sampled
// at the same normalized location; no-data stays unavailable rather than zero.
export function rasterValue(raster: SurfaceRaster | null, u: number, v: number): number | null {
  if (!raster || !Number.isFinite(u) || !Number.isFinite(v) || u < 0 || v < 0 || u > 1 || v > 1) return null;
  const { width, height } = raster.stats;
  const value = raster.values[Math.round(v * (height - 1)) * width + Math.round(u * (width - 1))];
  return Number.isFinite(value) ? value : null;
}

export function querySurface(query: SurfaceQuery, u: number, v: number): InspectionData | null {
  const rawValue = rasterValue(query.raw, u, v);
  if (rawValue === null) return null;
  const offset = (Math.round(v * (query.textureHeight - 1)) * query.textureWidth + Math.round(u * (query.textureWidth - 1))) * 4;
  const [red, green, blue] = query.rgba ? query.rgba.subarray(offset, offset + 3) : [0,0,0];
  const rgbVegetation = green > 33 && green > red * 1.035 && green > blue * 1.025 && 2 * green - red - blue > 9;
  const vegetationProbability = rasterValue(query.vegetation, u, v);
  const semantic = rasterValue(query.semantic, u, v);
  const knownClass = semantic !== null && [0,1,2].includes(semantic);
  const kind = knownClass ? semantic === 1 ? 'building' : semantic === 2 ? 'vegetation' : 'surface'
    : !query.semantic && ((vegetationProbability ?? 0) >= .35 || rgbVegetation) ? 'vegetation' : 'surface';
  const { width, height, units, gsdX, gsdY } = query.raw.stats;
  const col = Math.round(u * (width - 1)), row = Math.round(v * (height - 1));
  let slopeDegrees = null;
  if (units === 'm' && gsdX && gsdY && col > 0 && col < width - 1 && row > 0 && row < height - 1) {
    const left = query.raw.values[row * width + col - 1], right = query.raw.values[row * width + col + 1];
    const top = query.raw.values[(row - 1) * width + col], bottom = query.raw.values[(row + 1) * width + col];
    if ([left,right,top,bottom].every(Number.isFinite)) slopeDegrees = Math.atan(Math.hypot((right - left) / (2 * gsdX), (bottom - top) / (2 * gsdY))) * 180 / Math.PI;
  }
  const terrainElevation = query.raw.stats.units === 'm' && query.ground?.stats.units === 'm' ? rasterValue(query.ground, u, v) : null;
  return { kind, sampleUv: [u, v], terrainElevation, aboveGroundHeight: terrainElevation === null ? null : rawValue - terrainElevation,
    targetKey: `raster:${kind}:${query.semantic && !knownClass ? 'unknown' : 'known'}`, rawValue, displayValue: rasterValue(query.display,u,v) ?? rawValue,
    buildingProbability: rasterValue(query.building,u,v), vegetationProbability,
    confidence: rasterValue(query.confidence,u,v), rgbVegetation, slopeDegrees,
    classBasis: query.semantic ? knownClass ? 'Resolved surface class' : 'Unclassified pixel' : undefined };
}

/** Same classification rules as the point inspector, without calculating per-pixel slopes. */
export function surfaceClassCode(query: SurfaceQuery, u: number, v: number): 0 | 1 | 2 | 3 {
  if (rasterValue(query.raw, u, v) === null) return 3;
  const semantic = rasterValue(query.semantic, u, v);
  if (query.semantic) return semantic === 0 || semantic === 1 || semantic === 2 ? semantic : 3;
  const offset = (Math.round(v * (query.textureHeight - 1)) * query.textureWidth + Math.round(u * (query.textureWidth - 1))) * 4;
  const [r, g, b] = query.rgba ? query.rgba.subarray(offset, offset + 3) : [0, 0, 0];
  const green = g > 33 && g > r * 1.035 && g > b * 1.025 && 2 * g - r - b > 9;
  return (rasterValue(query.vegetation, u, v) ?? 0) >= .35 || green ? 2 : 0;
}
