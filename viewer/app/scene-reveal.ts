export const SCENE_REVEAL_DURATION = 3.65;
export type RevealPhase = 'grid' | 'footprints' | 'heights' | 'texture' | 'complete';
const ease = (value: number) => {
  const t = Math.min(1, Math.max(0, value));
  return t * t * (3 - 2 * t);
};

/** Presentation of a completed result, never a proxy for model progress. */
export function sceneRevealAt(elapsedSeconds: number, skip = false) {
  const t = skip ? SCENE_REVEAL_DURATION : Math.max(0, elapsedSeconds);
  const phase: RevealPhase = t < .5 ? 'grid' : t < 1.05 ? 'footprints'
    : t < 2.75 ? 'heights' : t < SCENE_REVEAL_DURATION ? 'texture' : 'complete';
  return {
    phase,
    progress: Math.min(1, t / SCENE_REVEAL_DURATION),
    // Nonzero scale keeps matrices invertible. No vertex or prediction is edited.
    scaleY: .001 + .999 * ease((t - 1.05) / 1.7),
    metricMix: 1 - ease((t - 2.75) / .9),
    showBuildings: t >= .5,
  };
}

export const REVEAL_LABELS: Record<RevealPhase, string> = {
  grid: 'Laying out the grid',
  footprints: 'Revealing building shapes',
  heights: 'Raising the predicted heights',
  texture: 'Wrapping the optical image',
  complete: 'Interactive mesh ready',
};
