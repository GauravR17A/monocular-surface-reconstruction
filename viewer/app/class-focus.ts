export type FocusClass = 0 | 1 | 2;
export type ClassWeights = [number, number, number, number];
export const CLASS_NAMES = ['Surface', 'Buildings', 'Vegetation'] as const;
export const ALL_CLASS_HEIGHTS: ClassWeights = [1, 1, 1, 1];
export const UNKNOWN_CLASS = 3;
const ease = (t: number) => { const x = Math.max(0, Math.min(1, t)); return x * x * (3 - 2 * x); };

/** Flatten first, then raise just the chosen class. Unknown never becomes a named class. */
export function classFocusFrame(from: ClassWeights, selected: FocusClass | null, seconds: number, reducedMotion = false): { weights: ClassWeights; done: boolean } {
  const duration = selected === null ? .65 : 1.05;
  const t = reducedMotion ? duration : Math.max(0, seconds);
  const target: ClassWeights = selected === null ? [...ALL_CLASS_HEIGHTS] : [0, 0, 0, 0];
  if (selected !== null) target[selected] = 1;
  if (t >= duration) return { weights: target, done: true };
  if (selected === null) return { weights: from.map((v, i) => v + (target[i] - v) * ease(t / duration)) as ClassWeights, done: false };
  if (t < .32) return { weights: from.map(v => v * (1 - ease(t / .32))) as ClassWeights, done: false };
  return { weights: target.map(v => v * ease((t - .32) / .73)) as ClassWeights, done: false };
}

/** Only the presentation vertex buffer is changed; source and collision heights remain immutable. */
export function focusedVertexHeights(source: Float32Array, classes: Uint8Array, weights: ClassWeights, positions: Float32Array) {
  if (source.length !== classes.length || positions.length !== source.length * 3) throw new Error('Class focus geometry dimensions do not match');
  for (let i = 0; i < source.length; i++) positions[i * 3 + 1] = source[i] * (weights[classes[i]] ?? weights[UNKNOWN_CLASS]);
}

export function classCoverage(classes: Uint8Array, valid: Uint8Array) {
  if (classes.length !== valid.length) throw new Error('Class coverage dimensions do not match');
  const counts: ClassWeights = [0, 0, 0, 0];
  let total = 0;
  for (let i = 0; i < classes.length; i++) if (valid[i]) { counts[classes[i] <= 2 ? classes[i] : UNKNOWN_CLASS]++; total++; }
  return { counts, total, percentages: counts.map(n => total ? n / total * 100 : 0) as ClassWeights };
}
