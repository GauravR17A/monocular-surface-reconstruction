import type { ClassificationStatus, InspectionData } from './surface-query';

/** Independent RGB labels: never convert these codes into height/physics masks. */
export const SIX_CLASS_NAMES = ['Ground', 'Buildings', 'Water', 'Roads', 'Low vegetation', 'Trees'] as const;
export const SIX_CLASS_COLORS = [[186, 161, 127], [255, 158, 72], [61, 166, 255], [193, 200, 218], [174, 224, 91], [30, 178, 124]] as const;
export type SixClassLayer = { labels: Uint8Array; width: number; height: number; selected: number | null; visible?: boolean };

/** Hiding the colors must not discard identification used by the scanner. */
export function sixClassOverlayVisible(layer: SixClassLayer | null): boolean {
  return Boolean(layer && layer.visible !== false);
}

export function classLayerPixels(layer: SixClassLayer) {
  if (layer.labels.length !== layer.width * layer.height) throw new Error('Classification grid size mismatch');
  const pixels = new Uint8Array(layer.labels.length * 4);
  for (let i = 0; i < layer.labels.length; i++) {
    const code = layer.labels[i];
    if (code > 5) continue;
    const chosen = layer.selected === null || layer.selected === code;
    pixels.set(chosen ? SIX_CLASS_COLORS[code] : [20, 32, 39], i * 4);
    pixels[i * 4 + 3] = 255;
  }
  return pixels;
}

export function sixClassAt(layer: SixClassLayer | null, u: number, v: number): string | null {
  if (!layer || !Number.isFinite(u) || !Number.isFinite(v) || u < 0 || u > 1 || v < 0 || v > 1) return null;
  const code = layer.labels[Math.min(layer.height - 1, Math.floor(v * layer.height)) * layer.width + Math.min(layer.width - 1, Math.floor(u * layer.width))];
  return SIX_CLASS_NAMES[code] ?? null;
}

/** Pair aligned outputs for display only. Never gate or modify a height. */
export function withSixClass(data: InspectionData, layer: SixClassLayer | null, u: number, v: number, status?: ClassificationStatus): InspectionData {
  const label = layer ? sixClassAt(layer, u, v) : status ? null : undefined;
  return { ...data, sampleUv: [u, v], experimentalClass: label, classificationStatus: layer ? 'ready' : status,
    targetKey: `${data.targetKey ?? data.kind}|rgb:${label === undefined ? 'off' : label ?? 'unknown'}` };
}

export function classificationSource(textureUrl: string, demoId: string | null) {
  // Replaced textures/local height maps must not accidentally classify an old demo.
  const match = /^http:\/\/(?:127\.0\.0\.1|localhost):8000\/results\/([a-f0-9]{32})\/texture\.jpg$/.exec(textureUrl);
  if (match) return { job_id: match[1] };
  const demos: Record<string, string> = { urban: '/demo/copenhagen_rgb.jpg', sparse: '/demo/landscapes/sparse/texture.jpg',
    hilly: '/demo/landscapes/hilly/texture.jpg', forest: '/demo/forest_validation/satellite_rgb.jpg' };
  if (demoId && demos[demoId] === textureUrl.split('?')[0]) return { demo_id: demoId };
  return null;
}
