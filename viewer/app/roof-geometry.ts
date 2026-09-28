import * as THREE from 'three';
import type { SurfaceRaster } from './surface-query';

export type RoofFootprint = [number, number][];
export type RoofSource = { height: SurfaceRaster; semantic: SurfaceRaster | null; probability: SurfaceRaster | null };
export type RoofDetail = { geometry: THREE.BufferGeometry; minimum: number; maximum: number; supportedSamples: number; sampledPoints: number };

export function insideRoof(u: number, v: number, outline: RoofFootprint) {
  let inside = false;
  for (let i = 0, j = outline.length - 1; i < outline.length; j = i++) {
    const [x, y] = outline[i], [px, py] = outline[j];
    if ((y > v) !== (py > v) && u < (px - x) * (v - y) / (py - y) + x) inside = !inside;
  }
  return inside;
}

function rasterAt(raster: SurfaceRaster, u: number, v: number) {
  return raster.values[Math.round(v * (raster.stats.height - 1)) * raster.stats.width + Math.round(u * (raster.stats.width - 1))];
}

/** A small spatial median rejects single-pixel spikes, not coherent rooftop patches. */
export function supportedRoofHeight(source: RoofSource, outline: RoofFootprint, u: number, v: number): number | null {
  const { width, height } = source.height.stats;
  const cx = Math.round(u * (width - 1)), cy = Math.round(v * (height - 1));
  const values: number[] = [];
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const x = cx + dx, y = cy + dy;
    if (x < 0 || y < 0 || x >= width || y >= height) continue;
    const pu = x / (width - 1), pv = y / (height - 1);
    if (!insideRoof(pu, pv, outline)) continue;
    if (source.semantic && rasterAt(source.semantic, pu, pv) !== 1) continue;
    if (!source.semantic && source.probability && !(rasterAt(source.probability, pu, pv) >= .3)) continue;
    const value = source.height.values[y * width + x];
    if (Number.isFinite(value) && value >= 0) values.push(value);
  }
  if (values.length < 3) return null;
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)];
}

/**
 * Build a bounded regular-grid roof overlay, clipped inside the footprint.
 * Preserves supported elevations ABOVE the existing summary roof, not invented
 * RGB geometry. Low edge predictions are not allowed to collapse the building.
 * The original raster is never modified. Unsupported scenes retain the old solid.
 */
export function buildRoofDetail(
  outline: RoofFootprint, source: RoofSource,
  sceneWidth: number, sceneDepth: number, verticalScale: number,
  baseM: number, summaryHeightM: number, pointBudget = 1400,
): RoofDetail | null {
  if (source.height.stats.units !== 'm' || source.height.stats.width < 3 || source.height.stats.height < 3 || outline.length < 3) return null;
  if (outline.some(([u, v]) => !Number.isFinite(u) || !Number.isFinite(v) || u < 0 || v < 0 || u > 1 || v > 1)) return null;
  const contour = outline.map(([u, v]) => new THREE.Vector2(u, v));
  if (contour.at(-1)!.equals(contour[0])) contour.pop();
  if (contour.length < 3) return null;
  const footprint = contour.map(p => [p.x, p.y] as [number, number]);
  const minU = Math.min(...contour.map(p => p.x)), maxU = Math.max(...contour.map(p => p.x));
  const minV = Math.min(...contour.map(p => p.y)), maxV = Math.max(...contour.map(p => p.y));
  const { width, height } = source.height.stats;
  const pixelWidth = (maxU - minU) * (width - 1), pixelHeight = (maxV - minV) * (height - 1);
  const step = Math.max(2, Math.ceil(Math.sqrt(pixelWidth * pixelHeight / Math.max(16, pointBudget))));
  const supported: number[] = [], positions: number[] = [], uvs: number[] = [];
  const heights: number[] = [], valid: boolean[] = [];
  const points: [number, number][] = [];
  const xs: number[] = [], ys: number[] = [];
  for (let x = Math.ceil(minU * (width - 1)) + 1; x < maxU * (width - 1) - 1; x += step) xs.push(x);
  for (let y = Math.ceil(minV * (height - 1)) + 1; y < maxV * (height - 1) - 1; y += step) ys.push(y);
  let insideCount = 0;
  for (const y of ys) {
    for (const x of xs) {
      const u = x / (width - 1), v = y / (height - 1);
      const inside = insideRoof(u, v, footprint);
      const value = inside ? supportedRoofHeight(source, footprint, u, v) : null;
      if (inside) insideCount++;
      const roofHeight = Math.max(summaryHeightM, value ?? summaryHeightM);
      points.push([u, v]); heights.push(roofHeight); valid.push(inside && value !== null);
      // Tiny depth offset only prevents z-fighting with the retained summary roof.
      positions.push((u - .5) * sceneWidth, (baseM + roofHeight) * verticalScale + .002, (v - .5) * sceneDepth);
      uvs.push(u, 1 - v);
      if (value !== null) supported.push(value);
    }
  }
  if (supported.length < 9 || supported.length < insideCount * .65) return null;
  supported.sort((a, b) => a - b);
  // Do not turn negligible numeric noise into detailed geometry.
  const minimumRelief = Math.max(.5, summaryHeightM * .07);
  const elevated = supported.filter(value => value >= summaryHeightM + minimumRelief);
  if (elevated.length < Math.max(5, Math.ceil(supported.length * .01))) return null;
  // Meet the retained flat roof at unsupported edges instead of leaving a
  // floating lip or extrapolating an upper structure outside its footprint.
  for (let row = 0; row < ys.length; row++) for (let col = 0; col < xs.length; col++) {
    const i = row * xs.length + col;
    if (row === 0 || col === 0 || row === ys.length - 1 || col === xs.length - 1
      || !valid[i-1] || !valid[i+1] || !valid[i-xs.length] || !valid[i+xs.length]) {
      heights[i] = summaryHeightM;
      positions[i*3+1] = (baseM + summaryHeightM) * verticalScale + .002;
    }
  }
  const indices: number[] = [];
  for (let row = 0; row < ys.length - 1; row++) for (let col = 0; col < xs.length - 1; col++) {
    const a = row * xs.length + col, b = a + 1, c = a + xs.length, d = c + 1;
    for (const triangle of [[a,c,b],[b,c,d]]) {
      if (!triangle.every(i => valid[i])) continue;
      if (!triangle.some(i => heights[i] > summaryHeightM + .02)) continue;
      const [pa,pb,pc] = triangle.map(i => points[i]);
      if (![[pa,pb],[pb,pc],[pc,pa]].every(([p,q]) => insideRoof((p[0]+q[0])/2,(p[1]+q[1])/2,footprint))) continue;
      if (!insideRoof((pa[0]+pb[0]+pc[0])/3,(pa[1]+pb[1]+pc[1])/3,footprint)) continue;
      indices.push(...triangle);
    }
  }
  if (!indices.length) return null;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.addGroup(0, indices.length, 0);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  return { geometry, minimum: Math.min(...heights), maximum: Math.max(...heights), supportedSamples: supported.length, sampledPoints: insideCount };
}
