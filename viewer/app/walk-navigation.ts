/** Grounded navigation on the displayed reconstruction, not a surveyed pedestrian route. */
export const WALK_EYE_METRES = 1.8;
// Brisk, game-style exploration pace, not a pedestrian travel-time estimate.
export const WALK_SPEED_MPS = 4.3;
export const WALK_RADIUS_METRES = 0.28;
export const WALK_JUMP_HEIGHT_METRES = 1.25;
const WALK_GRAVITY_MPS2 = 18;
export type WalkPoint = { x: number; z: number; y: number };
export type WalkMotion = { position: WalkPoint; velocityY: number; grounded: boolean; safePosition: WalkPoint };
export type WalkWorld = {
  width: number;
  depth: number;
  /** Scene units per metre (or an explicitly approximate human-scale preview). */
  scaleX: number;
  scaleZ: number;
  scaleY: number;
  /** Unexaggerated rendered ground. Missing/forbidden pixels return null, never zero. */
  ground: (x: number, z: number) => number | null;
  blocked: (x: number, z: number, radius: number) => boolean;
};

function validWorld(world: WalkWorld) {
  return [world.width, world.depth, world.scaleX, world.scaleY, world.scaleZ].every(v => Number.isFinite(v) && v > 0);
}

/** Keep the entire body inside the image and out of solids, canopy and NoData. */
function bodyGround(world: WalkWorld, x: number, z: number): number[] | null {
  if (!validWorld(world)) return null;
  const rx = WALK_RADIUS_METRES * world.scaleX, rz = WALK_RADIUS_METRES * world.scaleZ;
  if (Math.abs(x) + rx >= world.width / 2 || Math.abs(z) + rz >= world.depth / 2) return null;
  if (world.blocked(x, z, Math.max(rx, rz))) return null;
  const heights = [[0, 0], [rx, 0], [-rx, 0], [0, rz], [0, -rz]].map(([dx, dz]) => world.ground(x + dx, z + dz));
  if (heights.some(h => h === null || !Number.isFinite(h))) return null;
  return heights as number[];
}

export function walkGround(world: WalkWorld, x: number, z: number): number | null {
  const samples = bodyGround(world, x, z);
  if (!samples) return null;
  // A body-sized cliff or steep slope is not a safe standing point.
  if ((Math.max(...samples) - Math.min(...samples)) / world.scaleY > 0.55) return null;
  return samples[0];
}

/** Find an open starting location; never place a walker on a roof just to make entry succeed. */
export function findWalkSpawn(world: WalkWorld, anchor: { x: number; z: number }): WalkPoint | null {
  if (!validWorld(world)) return null;
  let best: WalkPoint | null = null, bestScore = Infinity;
  const consider = (x: number, z: number) => {
    const score = Math.hypot((x - anchor.x) / world.scaleX, (z - anchor.z) / world.scaleZ);
    if (score >= bestScore) return;
    const y = walkGround(world, x, z);
    if (y !== null) { best = { x, y, z }; bestScore = score; }
  };
  consider(anchor.x, anchor.z);
  if (best) return best;
  // Fine local search finds narrow streets; a bounded scene-wide search handles an orbit over a roof.
  for (let ring = 1; ring <= 12; ring++) for (let i = 0; i < 24; i++) {
    const angle = i * Math.PI / 12;
    consider(anchor.x + Math.cos(angle) * ring * world.scaleX, anchor.z + Math.sin(angle) * ring * world.scaleZ);
  }
  for (let r = 0; r < 61; r++) for (let c = 0; c < 61; c++) {
    consider((c / 60 - 0.5) * world.width * 0.98, (r / 60 - 0.5) * world.depth * 0.98);
  }
  return best;
}

/** Horizontal movement in metres; camera pitch and vertical-flight keys cannot lift the walker. */
export function walkIntent(yaw: number, forward: number, right: number) {
  const length = Math.max(1, Math.hypot(forward, right));
  return { x: (-Math.sin(yaw) * forward + Math.cos(yaw) * right) / length,
    z: (-Math.cos(yaw) * forward - Math.sin(yaw) * right) / length };
}

export function moveWalk(position: WalkPoint, movementMetres: { x: number; z: number }, world: WalkWorld) {
  const p = { ...position };
  if (!validWorld(world) || ![movementMetres.x, movementMetres.z].every(Number.isFinite)) return { position: p, distance: 0, blocked: true };
  const requested = Math.hypot(movementMetres.x, movementMetres.z);
  // Bounded subdivision prevents sprinting through thin walls or off cliffs.
  const travel = Math.min(requested, 5), ratio = requested ? travel / requested : 0;
  const steps = Math.max(1, Math.ceil(travel / 0.08));
  const dx = movementMetres.x * ratio * world.scaleX / steps, dz = movementMetres.z * ratio * world.scaleZ / steps;
  let blocked = requested > 5, distance = 0;
  const startHeight = walkGround(world, p.x, p.z);
  if (startHeight === null) return { position: p, distance, blocked: true };
  p.y = startHeight;
  const advance = (x: number, z: number) => {
    const horizontal = Math.hypot(x / world.scaleX, z / world.scaleZ);
    if (horizontal < 1e-10) return false;
    const y = walkGround(world, p.x + x, p.z + z);
    if (y === null || Math.abs(y - p.y) / world.scaleY > horizontal * Math.tan(40 * Math.PI / 180) + 1e-6) return false;
    p.x += x; p.z += z; p.y = y; distance += horizontal;
    return true;
  };
  for (let i = 0; i < steps && requested > 0; i++) {
    if (!advance(dx, dz)) { blocked = true; advance(dx, 0); advance(0, dz); }
  }
  return { position: p, distance, blocked };
}

export function createWalkMotion(position: WalkPoint): WalkMotion {
  return { position: { ...position }, safePosition: { ...position }, velocityY: 0, grounded: true };
}

/** Space is an edge-triggered jump, not a held lift/flight input. Physics stays in unexaggerated metres. */
export function stepWalk(state: WalkMotion, velocity: { x: number; z: number }, elapsed: number, world: WalkWorld, jumpPressed: boolean) {
  if (!validWorld(world) || ![elapsed, velocity.x, velocity.z, state.position.x, state.position.y, state.position.z, state.velocityY].every(Number.isFinite) || elapsed <= 0) {
    return { ...state, distance: 0, blocked: true };
  }
  const dt = Math.min(elapsed, 0.1);
  if (state.grounded && !jumpPressed) {
    const result = moveWalk(state.position, { x: velocity.x * dt, z: velocity.z * dt }, world);
    return { ...createWalkMotion(result.position), distance: result.distance, blocked: result.blocked };
  }
  const p = { ...state.position }, safe = { ...state.safePosition };
  let velocityY = state.grounded ? Math.sqrt(2 * WALK_GRAVITY_MPS2 * WALK_JUMP_HEIGHT_METRES) : state.velocityY;
  let distance = 0, blocked = elapsed > dt;
  if (state.grounded && walkGround(world, p.x, p.z) === null) return { ...state, distance: 0, blocked: true };
  const speed = Math.hypot(velocity.x, velocity.z);
  const travelScale = Math.min(1, 5 / Math.max(speed * dt, 1e-10));
  const steps = Math.max(1, Math.ceil(dt * 120), Math.ceil(speed * travelScale * dt / 0.08));
  const tick = dt / steps;
  const dx = velocity.x * travelScale * tick * world.scaleX, dz = velocity.z * travelScale * tick * world.scaleZ;
  for (let i = 0; i < steps; i++) {
    const current = bodyGround(world, p.x, p.z);
    if (!current) return { ...createWalkMotion(safe), distance, blocked: true };
    const nextY = p.y + (velocityY * tick - 0.5 * WALK_GRAVITY_MPS2 * tick * tick) * world.scaleY;
    velocityY -= WALK_GRAVITY_MPS2 * tick;
    p.y = nextY;
    // Land rather than penetrating the surface. Unsafe ledges recover to the last safe stance.
    if (velocityY <= 0 && p.y <= Math.max(...current)) {
      const floor = walkGround(world, p.x, p.z);
      if (floor === null) return { ...createWalkMotion(safe), distance, blocked: true };
      p.y = floor;
      const rest = moveWalk(p, { x: velocity.x * travelScale * tick * (steps - i), z: velocity.z * travelScale * tick * (steps - i) }, world);
      return { ...createWalkMotion(rest.position), distance: distance + rest.distance, blocked: blocked || rest.blocked };
    }
    const advance = (x: number, z: number) => {
      if (Math.hypot(x, z) < 1e-12) return true;
      const surface = bodyGround(world, p.x + x, p.z + z);
      // Even airborne, buildings, tall canopy, missing data and map borders stay solid/forbidden.
      if (!surface || Math.max(...surface) > p.y + 1e-8) return false;
      p.x += x; p.z += z;
      distance += Math.hypot(x / world.scaleX, z / world.scaleZ);
      return true;
    };
    if (!advance(dx, dz)) { blocked = true; advance(dx, 0); advance(0, dz); }
  }
  return { position: p, safePosition: safe, velocityY, grounded: false, distance, blocked };
}

/** Game-style footfall bob with gentle lateral sway; no motion when pushing into a wall. */
export function walkHeadMotion(distance: number, strength: number, reducedMotion: boolean) {
  if (reducedMotion || strength <= 0) return { lift: 0, roll: 0, sway: 0, pitch: 0 };
  const phase = distance / 3.2 * Math.PI * 2;
  const amount = Math.max(0, Math.min(1, strength));
  return { lift: -Math.abs(Math.sin(phase)) * 0.045 * amount,
    roll: Math.sin(phase) * 0.005 * amount, sway: Math.sin(phase) * 0.025 * amount,
    pitch: Math.cos(phase * 2) * 0.004 * amount };
}
