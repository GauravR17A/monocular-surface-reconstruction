import * as THREE from 'three';
import type { RoofFootprint, RoofSource } from './roof-geometry';

/** Experimental model-based geometry. Fit residuals are NOT survey accuracy. */
export type RoofImage = { rgba: Uint8ClampedArray; width: number; height: number };
type Point = [number, number];
type Sample = { u: number; v: number; x: number; y: number; z: number; check: boolean };
export type RoofFrame = {
  angle: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  width: number;
  height: number;
};
export type RoofFit = {
  kind: 'gable';
  frame: RoofFrame;
  ridge: number;
  coefficients: number[];
  minimum: number;
  maximum: number;
  sampleCount: number;
  checkMaeM: number;
  flatMaeM: number;
  planeMaeM: number;
  imageEdgeSupport: number;
};
export type RoofFitResult = { fit: RoofFit | null; reason: string };

function inside(u: number, v: number, p: RoofFootprint) {
  let yes = false;
  for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
    const [x, y] = p[i],
      [a, b] = p[j];
    if (y > v !== b > v && u < ((a - x) * (v - y)) / (b - y) + x) yes = !yes;
  }
  return yes;
}
function quantile(values: number[], q: number) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(q * (sorted.length - 1))];
}
function area(p: Point[]) {
  return (
    Math.abs(p.reduce((s, [x, y], i) => s + x * p[(i + 1) % p.length][1] - y * p[(i + 1) % p.length][0], 0)) /
    2
  );
}
function framePoint(frame: RoofFrame, x: number, y: number): Point {
  const a = frame.minX + x * (frame.maxX - frame.minX),
    b = frame.minY + y * (frame.maxY - frame.minY);
  return [
    (a * Math.cos(frame.angle) - b * Math.sin(frame.angle)) / frame.width,
    (a * Math.sin(frame.angle) + b * Math.cos(frame.angle)) / frame.height,
  ];
}
export function roofCoordinates(frame: RoofFrame, u: number, v: number): Point {
  const px = u * frame.width,
    py = v * frame.height;
  return [
    (px * Math.cos(frame.angle) + py * Math.sin(frame.angle) - frame.minX) / (frame.maxX - frame.minX),
    (-px * Math.sin(frame.angle) + py * Math.cos(frame.angle) - frame.minY) / (frame.maxY - frame.minY),
  ];
}
function frameFor(outline: RoofFootprint, width: number, height: number): RoofFrame {
  let best: RoofFrame | null = null,
    bestArea = Infinity;
  // Bounded minimum-area orientation search, independent of roof colour.
  for (let k = 0; k < 36; k++) {
    const angle = (k * Math.PI) / 72,
      c = Math.cos(angle),
      s = Math.sin(angle);
    const xs = outline.map(([u, v]) => u * width * c + v * height * s),
      ys = outline.map(([u, v]) => -u * width * s + v * height * c);
    const minX = Math.min(...xs),
      maxX = Math.max(...xs),
      minY = Math.min(...ys),
      maxY = Math.max(...ys);
    const size = (maxX - minX) * (maxY - minY);
    if (size < bestArea) {
      bestArea = size;
      best = { angle, minX, maxX, minY, maxY, width, height };
    }
  }
  return best!;
}
function basis(x: number, y: number, ridge: number) {
  return [1, y - 0.5, Math.min(0, x - ridge), Math.max(0, x - ridge)];
}
function dot(a: number[], b: number[]) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}
function solve(matrix: number[][], rhs: number[]): number[] | null {
  const n = rhs.length,
    a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    if (Math.abs(a[pivot][col]) < 1e-8) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    for (let j = col; j <= n; j++) a[col][j] /= divisor;
    for (let row = 0; row < n; row++)
      if (row !== col) {
        const factor = a[row][col];
        for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
      }
  }
  return a.map((row) => row[n]);
}
function regress(samples: Sample[], feature: (p: Sample) => number[]) {
  const rows = samples.map(feature),
    size = rows[0].length;
  let coefficients: number[] | null = null;
  // Huber reweighting: a handful of predicted chimneys/spikes must not set pitch.
  let weights = samples.map(() => 1);
  for (let iteration = 0; iteration < 3; iteration++) {
    const a = Array.from({ length: size }, () => new Array<number>(size).fill(0)),
      b = new Array<number>(size).fill(0);
    rows.forEach((row, k) => {
      for (let i = 0; i < size; i++) {
        b[i] += weights[k] * row[i] * samples[k].z;
        for (let j = 0; j < size; j++) a[i][j] += weights[k] * row[i] * row[j];
      }
    });
    coefficients = solve(a, b);
    if (!coefficients) return null;
    const errors = rows.map((row, k) => Math.abs(dot(row, coefficients!) - samples[k].z));
    const scale = Math.max(0.2, quantile(errors, 0.5) * 1.5);
    weights = errors.map((error) => Math.min(1, scale / Math.max(error, 1e-6)));
  }
  return coefficients;
}
function mae(samples: Sample[], predict: (p: Sample) => number) {
  return samples.reduce((s, p) => s + Math.abs(p.z - predict(p)), 0) / samples.length;
}
function enclosedGap(support: boolean[], nx: number, ny: number) {
  const seen = new Set<number>(),
    queue: number[] = [];
  for (let i = 0; i < support.length; i++)
    if (!support[i] && (i < nx || i >= nx * (ny - 1) || i % nx === 0 || i % nx === nx - 1)) {
      seen.add(i);
      queue.push(i);
    }
  for (let q = 0; q < queue.length; q++) {
    const i = queue[q];
    for (const j of [i - nx, i + nx, ...(i % nx > 0 ? [i - 1] : []), ...(i % nx < nx - 1 ? [i + 1] : [])]) {
      if (j >= 0 && j < support.length && !support[j] && !seen.has(j)) {
        seen.add(j);
        queue.push(j);
      }
    }
  }
  return support.reduce((n, valid, i) => n + (!valid && !seen.has(i) ? 1 : 0), 0) >= 4;
}
function at(source: RoofSource, u: number, v: number) {
  const rasterValue = (r: NonNullable<RoofSource['semantic']>) =>
    r.values[Math.round(v * (r.stats.height - 1)) * r.stats.width + Math.round(u * (r.stats.width - 1))];
  if (
    source.semantic
      ? rasterValue(source.semantic) !== 1
      : !source.probability || !(rasterValue(source.probability) >= 0.6)
  )
    return null;
  const value = rasterValue(source.height);
  return Number.isFinite(value) && value > 0 ? value : null;
}
function imageSupport(image: RoofImage, frame: RoofFrame, ridge: number) {
  const sample = (x: number, y: number) => {
    const [u, v] = framePoint(frame, x, y);
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    const k = (Math.round(v * (image.height - 1)) * image.width + Math.round(u * (image.width - 1))) * 4;
    if (image.rgba[k + 3] === 0) return null;
    return 0.2126 * image.rgba[k] + 0.7152 * image.rgba[k + 1] + 0.0722 * image.rgba[k + 2];
  };
  const dx = Math.max(1.5 / (frame.maxX - frame.minX), 0.015);
  const edge = (x: number, y: number) => {
    const a = sample(x - dx, y),
      b = sample(x + dx, y);
    return a === null || b === null ? null : Math.abs(a - b);
  };
  let valid = 0,
    supported = 0;
  for (let i = 0; i < 24; i++) {
    const y = 0.12 + (i * 0.76) / 23,
      centre = edge(ridge, y),
      left = edge(ridge - 0.15, y),
      right = edge(ridge + 0.15, y);
    if (centre === null || left === null || right === null) continue;
    valid++;
    if (centre >= 8 && centre >= (1.3 * (left + right)) / 2) supported++;
  }
  return valid >= 18 ? supported / valid : 0;
}

/**
 * First roof-reconstruction stage: simple, well-supported gables only.
 * No colour-to-shape rule, arbitrary pitch, or reference label at inference.
 * Rejection is intentional for merged footprints, flat/noisy predictions,
 * missing semantic support and roofs whose optical ridge is ambiguous.
 */
export function fitRoof(
  outline: RoofFootprint,
  source: RoofSource,
  image: RoofImage | null,
  summaryHeightM: number,
): RoofFitResult {
  const reject = (reason: string): RoofFitResult => ({ fit: null, reason });
  const { width, height, units } = source.height.stats;
  if (units !== 'm' || width < 16 || height < 16 || !Number.isFinite(summaryHeightM) || summaryHeightM <= 0)
    return reject('metric height evidence unavailable');
  if (!image || image.width < 16 || image.height < 16 || image.rgba.length !== image.width * image.height * 4)
    return reject('RGB evidence unavailable');
  if (
    outline.length < 3 ||
    outline.some((p) => p.some((v) => !Number.isFinite(v) || v <= 0.002 || v >= 0.998))
  )
    return reject('incomplete or edge-clipped footprint');
  const initial = frameFor(outline, width - 1, height - 1);
  const boxArea = (initial.maxX - initial.minX) * (initial.maxY - initial.minY);
  const fill = (area(outline) * (width - 1) * (height - 1)) / boxArea;
  if (fill < 0.82 || Math.min(initial.maxX - initial.minX, initial.maxY - initial.minY) < 16)
    return reject('complex, merged or undersized footprint');
  let best: RoofFit | null = null,
    hadHeightFit = false;
  for (let swap = 0; swap < 2; swap++) {
    const frame = swap
      ? {
          ...initial,
          angle: initial.angle + Math.PI / 2,
          minX: initial.minY,
          maxX: initial.maxY,
          minY: -initial.maxX,
          maxY: -initial.minX,
        }
      : initial;
    const nx = Math.min(28, Math.floor((frame.maxX - frame.minX) / 2)),
      ny = Math.min(36, Math.floor((frame.maxY - frame.minY) / 2));
    const samples: Sample[] = [];
    let count = 0,
      supported = 0;
    const support = new Array<boolean>(nx * ny).fill(false);
    for (let iy = 0; iy < ny; iy++)
      for (let ix = 0; ix < nx; ix++) {
        const x = 0.08 + (0.84 * (ix + 0.5)) / nx,
          y = 0.08 + (0.84 * (iy + 0.5)) / ny;
        const [u, v] = framePoint(frame, x, y);
        if (!inside(u, v, outline)) continue;
        count++;
        const z = at(source, u, v);
        if (z === null) continue;
        support[iy * nx + ix] = true;
        supported++;
        samples.push({ u, v, x, y, z, check: (Math.floor(ix / 3) + Math.floor(iy / 3)) % 3 === 0 });
      }
    // Courtyards/unknown labels must not silently become a solid roof.
    if (
      count < nx * ny * 0.82 ||
      supported < count * 0.9 ||
      samples.length < 80 ||
      enclosedGap(support, nx, ny)
    )
      continue;
    const zs = samples.map((p) => p.z),
      lo = quantile(zs, 0.05),
      hi = quantile(zs, 0.95);
    if (hi - lo < 1.2) continue;
    const train = samples.filter((p) => !p.check),
      check = samples.filter((p) => p.check);
    if (check.length < 25) continue;
    const flat = quantile(
        train.map((p) => p.z),
        0.5,
      ),
      plane = regress(train, (p) => [1, p.x, p.y]);
    if (!plane) continue;
    const flatMaeM = mae(check, () => flat),
      planeMaeM = mae(check, (p) => dot([1, p.x, p.y], plane));
    for (const ridge of [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]) {
      const coefficients = regress(train, (p) => basis(p.x, p.y, ridge));
      if (!coefficients) continue;
      const predict = (x: number, y: number) => dot(basis(x, y, ridge), coefficients);
      const riseLeft = coefficients[2] * ridge,
        riseRight = -coefficients[3] * (1 - ridge);
      const rise = Math.min(riseLeft, riseRight),
        largestRise = Math.max(riseLeft, riseRight);
      if (
        rise < 0.8 ||
        largestRise > Math.min(12, summaryHeightM * 0.8) ||
        largestRise / rise > 2.5 ||
        Math.abs(coefficients[1]) > rise * 0.65
      )
        continue;
      const extremes = [
        predict(0, 0),
        predict(0, 1),
        predict(1, 0),
        predict(1, 1),
        predict(ridge, 0),
        predict(ridge, 1),
      ];
      const minimum = Math.min(...extremes),
        maximum = Math.max(...extremes);
      if (minimum < Math.max(0.5, lo - (hi - lo) * 0.35) || maximum > hi + Math.max(0.6, (hi - lo) * 0.2))
        continue;
      const checkMaeM = mae(check, (p) => predict(p.x, p.y));
      if (
        checkMaeM > Math.min(1.5, (hi - lo) * 0.2) ||
        checkMaeM > flatMaeM * 0.7 ||
        checkMaeM > planeMaeM * 0.75
      )
        continue;
      // Require the rise/fall on each side in multiple along-ridge bands.
      let bands = 0;
      for (let band = 0; band < 3; band++) {
        const rows = samples.filter((p) => Math.floor(p.y * 3) === band);
        const centre = rows.filter((p) => Math.abs(p.x - ridge) < 0.1).map((p) => p.z);
        const left = rows.filter((p) => p.x < 0.23).map((p) => p.z),
          right = rows.filter((p) => p.x > 0.77).map((p) => p.z);
        if (
          centre.length >= 3 &&
          left.length >= 3 &&
          right.length >= 3 &&
          quantile(centre, 0.5) - Math.max(quantile(left, 0.5), quantile(right, 0.5)) > 0.45
        )
          bands++;
      }
      if (bands < 3) continue;
      hadHeightFit = true;
      const imageEdgeSupport = imageSupport(image, frame, ridge);
      if (imageEdgeSupport < 0.45) continue;
      if (!best || checkMaeM < best.checkMaeM)
        best = {
          kind: 'gable',
          frame,
          ridge,
          coefficients,
          minimum,
          maximum,
          sampleCount: samples.length,
          checkMaeM,
          flatMaeM,
          planeMaeM,
          imageEdgeSupport,
        };
    }
  }
  return best
    ? { fit: best, reason: 'experimental RGB-supported two-plane fit' }
    : reject(
        hadHeightFit
          ? 'height shape lacks a clear RGB ridge'
          : 'insufficient coherent two-sided height evidence',
      );
}

export function fittedRoofHeight(fit: RoofFit, u: number, v: number) {
  const [x, y] = roofCoordinates(fit.frame, u, v);
  return dot(basis(x, y, fit.ridge), fit.coefficients);
}

/** Clip each existing footprint triangle at the ridge; never fill concavities. */
export function buildFittedRoof(
  outline: RoofFootprint,
  fit: RoofFit,
  sceneWidth: number,
  sceneDepth: number,
  verticalScale: number,
  baseM: number,
) {
  const contour = outline.map((p) => new THREE.Vector2(...p));
  if (contour[0].distanceTo(contour[contour.length - 1]) < 1e-8) contour.pop();
  const faces = THREE.ShapeUtils.triangulateShape(contour, []);
  const positions: number[] = [],
    uvs: number[] = [];
  const push = (points: [number, number, number][]) => {
    for (const [u, v, h] of points) {
      positions.push((u - 0.5) * sceneWidth, (baseM + h) * verticalScale + 0.002, (v - 0.5) * sceneDepth);
      uvs.push(u, 1 - v);
    }
  };
  const clip = (poly: Point[], side: number) => {
    const out: Point[] = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i],
        b = poly[(i + 1) % poly.length],
        da = (roofCoordinates(fit.frame, ...a)[0] - fit.ridge) * side,
        db = (roofCoordinates(fit.frame, ...b)[0] - fit.ridge) * side;
      if (da >= -1e-10) out.push(a);
      if ((da > 0 && db < 0) || (da < 0 && db > 0)) {
        const t = da / (da - db);
        out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
      }
    }
    return out;
  };
  for (const face of faces)
    for (const side of [-1, 1]) {
      const polygon = clip(
        face.map((i) => [contour[i].x, contour[i].y]),
        side,
      );
      for (let i = 1; i < polygon.length - 1; i++) {
        const tri = [polygon[0], polygon[i], polygon[i + 1]];
        const cross =
          (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) -
          (tri[1][1] - tri[0][1]) * (tri[2][0] - tri[0][0]);
        if (Math.abs(cross) < 1e-12) continue;
        if (cross > 0) [tri[1], tri[2]] = [tri[2], tri[1]];
        push(tri.map(([u, v]) => [u, v, fittedRoofHeight(fit, u, v)]));
      }
    }
  const roofVertices = positions.length / 3;
  const signed = contour.reduce(
    (s, p, i) => s + p.x * contour[(i + 1) % contour.length].y - p.y * contour[(i + 1) % contour.length].x,
    0,
  );
  for (let i = 0; i < contour.length; i++) {
    const a: Point = [contour[i].x, contour[i].y],
      b: Point = [contour[(i + 1) % contour.length].x, contour[(i + 1) % contour.length].y];
    const xa = roofCoordinates(fit.frame, ...a)[0] - fit.ridge,
      xb = roofCoordinates(fit.frame, ...b)[0] - fit.ridge;
    const line: Point[] = [a];
    if (xa * xb < 0) {
      const t = xa / (xa - xb);
      line.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
    line.push(b);
    for (let j = 0; j < line.length - 1; j++) {
      const p = line[j],
        q = line[j + 1],
        hp = fittedRoofHeight(fit, ...p),
        hq = fittedRoofHeight(fit, ...q);
      const triangles = [
        [...p, -0.1],
        [...p, hp],
        [...q, hq],
        [...p, -0.1],
        [...q, hq],
        [...q, -0.1],
      ] as [number, number, number][];
      if (signed < 0) {
        [triangles[1], triangles[2]] = [triangles[2], triangles[1]];
        [triangles[4], triangles[5]] = [triangles[5], triangles[4]];
      }
      push(triangles);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.addGroup(0, roofVertices, 0);
  geometry.addGroup(roofVertices, positions.length / 3 - roofVertices, 1);
  // Non-indexed vertices keep the ridge and wall/roof joins truly sharp.
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  const position = geometry.getAttribute('position'),
    normal = geometry.getAttribute('normal'),
    uv = geometry.getAttribute('uv');
  const colors = new Float32Array(position.count * 3).fill(1);
  const tileWidth = Math.max(1.25, verticalScale * 6),
    rows = Math.max(1, fit.maximum / 3.4);
  for (let i = roofVertices; i < position.count; i++) {
    const fraction = Math.max(
      0,
      Math.min(1, (position.getY(i) - baseM * verticalScale) / (fit.maximum * verticalScale)),
    );
    uv.setXY(
      i,
      (-normal.getZ(i) * position.getX(i) + normal.getX(i) * position.getZ(i)) / tileWidth,
      fraction * rows,
    );
    colors.set([0.76 + 0.24 * fraction, 0.76 + 0.24 * fraction, 0.76 + 0.24 * fraction], i * 3);
  }
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  return geometry;
}
