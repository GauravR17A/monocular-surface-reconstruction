/** Connected display regions, not inferred object instances or measurement masks. */
export type SelectionRegionIndex = ReturnType<typeof createSelectionRegions>;

export function createSelectionRegions(sourceWidth: number, sourceHeight: number,
  labelAt: (u: number, v: number) => number, maxDimension = 512) {
  if (![sourceWidth, sourceHeight, maxDimension].every(n => Number.isInteger(n) && n > 0)) throw new Error('Invalid selection grid');
  const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale)), height = Math.max(1, Math.round(sourceHeight * scale));
  const labels = new Int16Array(width * height);
  const regions = new Int32Array(labels.length);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const code = labelAt((x + .5) / width, (y + .5) / height);
    labels[y * width + x] = Number.isInteger(code) && code >= 0 && code < 255 ? code : -1;
  }
  const queue = new Int32Array(labels.length);
  let count = 0;
  for (let seed = 0; seed < labels.length; seed++) {
    if (labels[seed] < 0 || regions[seed]) continue;
    const id = ++count, code = labels[seed];
    let head = 0, tail = 1;
    queue[0] = seed; regions[seed] = id;
    const visit = (i: number) => {
      if (!regions[i] && labels[i] === code) { regions[i] = id; queue[tail++] = i; }
    };
    while (head < tail) {
      const i = queue[head++], x = i % width;
      if (x > 0) visit(i - 1);
      if (x < width - 1) visit(i + 1);
      if (i >= width) visit(i - width);
      if (i < labels.length - width) visit(i + width);
    }
  }
  // Bound both the segmentation work and cached contour memory independently of upload size.
  const contours = new Map<number, Float32Array>();
  return {
    width, height, count,
    at(u: number, v: number, expectedLabel?: number): number | null {
      if (![u, v].every(Number.isFinite) || u < 0 || v < 0 || u > 1 || v > 1) return null;
      const i = Math.min(height - 1, Math.floor(v * height)) * width + Math.min(width - 1, Math.floor(u * width));
      return regions[i] && (expectedLabel === undefined || labels[i] === expectedLabel) ? regions[i] : null;
    },
    boundary(id: number): Float32Array {
      if (!Number.isInteger(id) || id <= 0 || id > count) return new Float32Array();
      const cached = contours.get(id);
      if (cached) return cached;
      const edges: number[] = [];
      const edge = (x: number, y: number, x2: number, y2: number) => edges.push(x / width, y / height, x2 / width, y2 / height);
      for (let i = 0; i < regions.length; i++) if (regions[i] === id) {
        const x = i % width, y = Math.floor(i / width);
        if (y === 0 || regions[i - width] !== id) edge(x, y, x + 1, y);
        if (x === width - 1 || regions[i + 1] !== id) edge(x + 1, y, x + 1, y + 1);
        if (y === height - 1 || regions[i + width] !== id) edge(x + 1, y + 1, x, y + 1);
        if (x === 0 || regions[i - 1] !== id) edge(x, y + 1, x, y);
      }
      const result = new Float32Array(edges);
      if (contours.size >= 8) contours.delete(contours.keys().next().value!);
      contours.set(id, result);
      return result;
    },
  };
}

/** A missing/too-small region gets a local sample locator, never a made-up object outline. */
export function sampleLocator(u: number, v: number, radiusU: number, radiusV: number) {
  const edges: number[] = [];
  const clamp = (n: number) => Math.max(0, Math.min(1, n));
  for (let i = 0; i < 40; i++) {
    const a = i / 40 * Math.PI * 2, b = (i + 1) / 40 * Math.PI * 2;
    edges.push(clamp(u + Math.cos(a) * radiusU), clamp(v + Math.sin(a) * radiusV),
      clamp(u + Math.cos(b) * radiusU), clamp(v + Math.sin(b) * radiusV));
  }
  return new Float32Array(edges);
}
