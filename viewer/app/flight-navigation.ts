/** Collision is against the displayed reconstruction, not surveyed real-world geometry. */
export type Position3 = { x: number; y: number; z: number };
export type FlightObstacle = { footprint: [number, number][]; top: number };
export const FLIGHT_CLEARANCE = 0.65;

export function touchesFootprint(x: number, z: number, polygon: [number, number][], radius: number) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [ax, az] = polygon[j], [bx, bz] = polygon[i];
    if ((az > z) !== (bz > z) && x < (bx - ax) * (z - az) / (bz - az) + ax) inside = !inside;
    const dx = bx - ax, dz = bz - az;
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / (dx * dx + dz * dz || 1)));
    if (Math.hypot(x - ax - t * dx, z - az - t * dz) <= radius) return true;
  }
  return inside;
}

/** Sample the same two triangles used by THREE.PlaneGeometry, not a smoothed raster. */
export function gridSurfaceHeight(heights: Float32Array, cols: number, rows: number, width: number, depth: number, x: number, z: number) {
  const u = (x / width + 0.5) * (cols - 1), v = (z / depth + 0.5) * (rows - 1);
  if (u < 0 || v < 0 || u > cols - 1 || v > rows - 1) return 0;
  const c = Math.min(cols - 2, Math.floor(u)), r = Math.min(rows - 2, Math.floor(v));
  const fx = u - c, fz = v - r;
  const a = heights[r * cols + c], b = heights[(r + 1) * cols + c];
  const d = heights[r * cols + c + 1], e = heights[(r + 1) * cols + c + 1];
  return fx + fz <= 1 ? a + fx * (d - a) + fz * (b - a) : e + (1 - fx) * (b - e) + (1 - fz) * (d - e);
}

export function moveFlight(position: Position3, movement: Position3, ground: (x: number, z: number) => number, obstacles: FlightObstacle[], exaggeration: number) {
  const p = { ...position }, radius = FLIGHT_CLEARANCE;
  const terrainFloor = (x: number, z: number) => Math.max(...[[0, 0], [radius, 0], [-radius, 0], [0, radius], [0, -radius]].map(([dx, dz]) => ground(x + dx, z + dz))) + radius;
  const solidFloor = (x: number, z: number) => obstacles.reduce((floor, obstacle) => touchesFootprint(x, z, obstacle.footprint, radius) ? Math.max(floor, obstacle.top * exaggeration + radius) : floor, terrainFloor(x, z));
  // Recover safely when entering Fly from an orbit camera inside a hill/roof,
  // or when the user raises vertical exaggeration while flying.
  p.y = Math.max(p.y, solidFloor(p.x, p.z));
  const steps = Math.max(1, Math.ceil(Math.hypot(movement.x, movement.y, movement.z) / (radius * 0.45)));
  const step = { x: movement.x / steps, y: movement.y / steps, z: movement.z / steps };
  for (let i = 0; i < steps; i++) {
    p.y = Math.max(p.y + step.y, solidFloor(p.x, p.z));
    const advance = (dx: number, dz: number) => {
      const x = p.x + dx, z = p.z + dz;
      if (obstacles.some(o => o.top * exaggeration + radius > p.y + 1e-6 && touchesFootprint(x, z, o.footprint, radius))) return false;
      const floor = terrainFloor(x, z);
      // Follow gentle hills, but do not teleport up cliffs or raster buildings.
      if (floor - p.y > Math.hypot(dx, dz) * 1.3 + 1e-6) return false;
      p.x = x; p.z = z; p.y = Math.max(p.y, floor);
      return true;
    };
    if (!advance(step.x, step.z)) { advance(step.x, 0); advance(0, step.z); }
  }
  return p;
}
