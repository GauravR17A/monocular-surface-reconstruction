/** A readable 1/2/5 scale length that fits the sampled screen span. */
export function sceneScaleLength(span: number) {
  if (!Number.isFinite(span) || span <= 0) return null;
  const power = 10 ** Math.floor(Math.log10(span));
  const scaled = span / power;
  return (scaled >= 5 ? 5 : scaled >= 2 ? 2 : 1) * power;
}
