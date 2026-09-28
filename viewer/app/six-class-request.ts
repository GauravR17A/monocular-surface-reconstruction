import type { classificationSource } from './six-class-layer';
import type { ClassificationStatus } from './surface-query';

export type ClassificationSource = NonNullable<ReturnType<typeof classificationSource>>;
export type SixClassResult = {
  width: number; height: number; labels: Uint8Array; valid_pixels: number; ignored_pixels: number;
  classes: { id: number; name: string; color: number[]; pixels: number; coverage_percent: number; area_m2: number | null }[];
  raster_url: string; metadata_url: string; labels_url: string; resampled: boolean;
  source_job_id: string | null; source_demo_id: string | null;
};
export type SixClassState = { sceneKey: string; status: ClassificationStatus; result: SixClassResult | null; message?: string };
export type SixClassView = { selected: number | null; visible: boolean };
export const DEFAULT_CLASS_VIEW: SixClassView = { selected: null, visible: false };

export function stateForScene(state: SixClassState | null, sceneKey: string): SixClassState {
  // Discard old labels synchronously, before the new scene's effects run.
  return state?.sceneKey === sceneKey ? state : { sceneKey, status: sceneKey ? 'waiting' : 'idle', result: null };
}

export class ClassificationError extends Error {
  readonly status: 'unsupported' | 'error';
  constructor(message: string, status: 'unsupported' | 'error' = 'error') {
    super(message); this.name = 'ClassificationError'; this.status = status;
  }
}

const CLASS_IDS = ['ground', 'buildings', 'water', 'roads', 'low_vegetation', 'trees'];
const artifactPath = /^\/results\/classification_[a-f0-9]{32}\/(classes\.(bin|tif)|metadata\.json)$/;

async function requestClassification(apiBase: string, source: ClassificationSource, fetcher: typeof fetch): Promise<SixClassResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300_000);
  try {
    const form = new FormData();
    for (const [key, value] of Object.entries(source)) if (value) form.set(key, value);
    const response = await fetcher(`${apiBase}/api/classify`, { method: 'POST', body: form, signal: controller.signal });
    const raw = await response.json();
    if (!raw || typeof raw !== 'object') throw new ClassificationError('Invalid identification response.');
    const payload = raw as Omit<SixClassResult, 'labels'> & { detail?: string; height_pipeline_changed?: boolean };
    if (!response.ok) throw new ClassificationError(typeof payload.detail === 'string' ? payload.detail : 'Identification failed. Retry without reconstructing the scene.', response.status === 422 ? 'unsupported' : 'error');
    if (payload.source_job_id !== ('job_id' in source ? source.job_id : null)
      || payload.source_demo_id !== ('demo_id' in source ? source.demo_id : null)) throw new ClassificationError('Identification does not match the current source image.');
    if (!Number.isInteger(payload.width) || !Number.isInteger(payload.height) || payload.width < 1 || payload.height < 1
      || payload.width > 3072 || payload.height > 3072 || payload.height_pipeline_changed !== false
      || !Array.isArray(payload.classes) || payload.classes.length !== 6
      || ![payload.labels_url, payload.raster_url, payload.metadata_url].every(path => typeof path === 'string' && artifactPath.test(path))
      || !payload.labels_url.endsWith('/classes.bin') || !payload.raster_url.endsWith('/classes.tif') || !payload.metadata_url.endsWith('/metadata.json')
      || new Set([payload.labels_url, payload.raster_url, payload.metadata_url].map(path => path.split('/')[2])).size !== 1) throw new ClassificationError('Invalid identification response.');
    const classRows = payload.classes as SixClassResult['classes'];
    if (classRows.some((row, index) => row.id !== index || row.name !== CLASS_IDS[index]
      || !Number.isInteger(row.pixels) || row.pixels < 0 || !Number.isFinite(row.coverage_percent)
      || row.coverage_percent < 0 || row.coverage_percent > 100 || !(row.area_m2 === null || Number.isFinite(row.area_m2) && row.area_m2 >= 0))) throw new ClassificationError('Invalid category statistics.');
    const labelsResponse = await fetcher(`${apiBase}${payload.labels_url}`, { signal: controller.signal });
    if (!labelsResponse.ok) throw new ClassificationError('Identification map could not be loaded.');
    const labels = new Uint8Array(await labelsResponse.arrayBuffer());
    const counts = new Array<number>(6).fill(0); let ignored = 0;
    for (const code of labels) { if (code < 6) counts[code]++; else if (code === 255) ignored++; else throw new ClassificationError('Invalid identification grid.'); }
    if (labels.length !== payload.width * payload.height || payload.ignored_pixels !== ignored
      || payload.valid_pixels !== labels.length - ignored || classRows.some((row, i) => row.pixels !== counts[i]
        || Math.abs(row.coverage_percent - (payload.valid_pixels ? counts[i] / payload.valid_pixels * 100 : 0)) > .001)) throw new ClassificationError('Identification grid and coverage do not agree.');
    return { ...payload, labels };
  } catch (error) {
    if (controller.signal.aborted) throw new ClassificationError('Identification timed out. Heights remain available; retry when the engine is ready.');
    if (error instanceof ClassificationError) throw error;
    throw new ClassificationError('Identification service unavailable. Heights are unchanged. Check the local engine and retry.');
  } finally { clearTimeout(timeout); }
}

/** Coalesce repeated mounts/visits. Scene changes unsubscribe their UI but
 * cannot cancel a GPU job already on the server. Keep four completed maps;
 * failed requests are never cached. */
export function createClassificationLoader(fetcher: typeof fetch = fetch) {
  const pending = new Map<string, Promise<SixClassResult>>();
  const completed = new Map<string, SixClassResult>();
  return function load(apiBase: string, source: ClassificationSource): Promise<SixClassResult> {
    const key = JSON.stringify([apiBase, source]);
    const cached = completed.get(key);
    if (cached) return Promise.resolve(cached);
    const active = pending.get(key);
    if (active) return active;
    const promise = requestClassification(apiBase, source, fetcher).then(result => {
      completed.set(key, result);
      if (completed.size > 4) completed.delete(completed.keys().next().value!);
      return result;
    }).finally(() => pending.delete(key));
    pending.set(key, promise);
    return promise;
  };
}
export const loadSceneClassification = createClassificationLoader();
