export type MapScene = { width: number; height: number; gsdX: number | null; gsdY: number | null };
export type MapFrame = { u: number; v: number; du: number; dv: number; mode: 'fly' | 'walk' };
export type MapSpace = { width: number; height: number; unit: 'm' | 'px'; resolution: number };
export type MapWindow = { x: number; y: number; span: number; heading: number; outside: boolean };
const clamp = (n: number, a: number, b: number) => Math.max(a, Math.min(b, n));
const positive = (n: number | null): n is number => n !== null && Number.isFinite(n) && n > 0;

export function mapSpace(scene: MapScene): MapSpace {
  const metric = positive(scene.gsdX) && positive(scene.gsdY);
  return { width: Math.max(1, scene.width - 1) * (metric ? scene.gsdX! : 1),
    height: Math.max(1, scene.height - 1) * (metric ? scene.gsdY! : 1), unit: metric ? 'm' : 'px',
    resolution: metric ? Math.max(scene.gsdX!, scene.gsdY!) : 1 };
}

// Scene plane X increases toward image-right; Z increases toward image-bottom.
// Yaw zero looks toward image-top. Pitch/height and display exaggeration do not
// move the map marker: this is horizontal position, not the aim-ray hit point.
export function mapFrame(x: number, z: number, width: number, depth: number, yaw: number, mode: MapFrame['mode']): MapFrame {
  return { u: x / width + .5, v: z / depth + .5, du: -Math.sin(yaw) / width, dv: -Math.cos(yaw) / depth, mode };
}

export function localMapWindow(space: MapSpace, pose: MapFrame, speed = 0): MapWindow {
  const diagonal = Math.hypot(space.width, space.height);
  const preferred = space.unit === 'm'
    ? Math.max(pose.mode === 'walk' ? 180 : 420, space.resolution * 24)
    : clamp(Math.max(space.width, space.height) * (pose.mode === 'walk' ? .28 : .45), 128, pose.mode === 'walk' ? 1024 : 2048);
  // Four seconds of travel remain visible at speed; don't zoom past the whole
  // footprint or amplify coarse source pixels beyond a usable local context.
  const span = Math.min(diagonal * 1.08, Math.max(preferred, Math.max(0, speed) * 4));
  const heading = Math.atan2(pose.du * space.width, -pose.dv * space.height);
  const outside = pose.u < 0 || pose.u > 1 || pose.v < 0 || pose.v > 1;
  const lookAhead = outside ? 0 : span * .12;
  return { x: clamp(pose.u * space.width + Math.sin(heading) * lookAhead, 0, space.width),
    y: clamp(pose.v * space.height - Math.cos(heading) * lookAhead, 0, space.height), span, heading, outside };
}

export function mapScreenPoint(space: MapSpace, pose: MapFrame, window: MapWindow, width: number, height: number, full: boolean) {
  const scale = full ? Math.min((width - 32) / space.width, (height - 32) / space.height) : Math.min(width, height) / window.span;
  const centreX = full ? space.width / 2 : window.x, centreY = full ? space.height / 2 : window.y;
  const angle = full ? 0 : -window.heading;
  const x = ((full ? clamp(pose.u, 0, 1) : pose.u) * space.width - centreX) * scale;
  const y = ((full ? clamp(pose.v, 0, 1) : pose.v) * space.height - centreY) * scale;
  const rawX = width / 2 + x * Math.cos(angle) - y * Math.sin(angle);
  const rawY = height / 2 + x * Math.sin(angle) + y * Math.cos(angle);
  return { x: clamp(rawX, 13, width - 13), y: clamp(rawY, 13, height - 13), scale, centreX, centreY, angle,
    edge: rawX < 13 || rawX > width - 13 || rawY < 13 || rawY > height - 13 || window.outside };
}

export function mapDistanceLabel(value: number, unit: MapSpace['unit']) {
  if (unit === 'm' && value >= 1000) return `${(value / 1000).toFixed(1)} km`;
  return `${Math.round(value).toLocaleString('en-US')} ${unit}`;
}
