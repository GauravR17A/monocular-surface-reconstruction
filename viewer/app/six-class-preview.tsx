'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { classificationSource, classLayerPixels, SIX_CLASS_NAMES, SIX_CLASS_COLORS } from './six-class-layer';
import { ClassificationError, loadSceneClassification, type SixClassState, type SixClassView } from './six-class-request';

export function SixClassPreview({ apiBase, textureUrl, demoId, ready, online, state, view, onState, onView }: {
  apiBase: string; textureUrl: string; demoId: string | null; ready: boolean; online: boolean;
  state: SixClassState; view: SixClassView;
  onState: (state: SixClassState) => void; onView: (view: SixClassView) => void;
}) {
  const [attempt, setAttempt] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const source = useMemo(() => classificationSource(textureUrl, demoId), [textureUrl, demoId]);
  const result = state.result;

  useEffect(() => {
    if (!textureUrl || !ready || !online) return;
    let current = true;
    // Avoid enqueuing GPU work during transient mounts or rapid scene switching.
    const start = window.setTimeout(() => {
      if (!source) {
        onState({ sceneKey: textureUrl, status: 'unsupported', result: null, message: 'This texture has no matching original RGB source. Upload the original image to identify it.' });
        return;
      }
      onState({ sceneKey: textureUrl, status: 'loading', result: null });
      void loadSceneClassification(apiBase, source).then(result => {
        if (current) onState({ sceneKey: textureUrl, status: 'ready', result });
      }, error => {
        if (current) onState({ sceneKey: textureUrl, status: error instanceof ClassificationError ? error.status : 'error', result: null,
          message: error instanceof Error ? error.message : 'Identification unavailable.' });
      });
    }, 250);
    return () => { current = false; window.clearTimeout(start); };
  }, [apiBase, textureUrl, source, ready, online, attempt, onState]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!result || !canvas) return;
    canvas.width = result.width; canvas.height = result.height;
    const pixels = new Uint8ClampedArray(classLayerPixels({ ...result, selected: view.selected }));
    canvas.getContext('2d')?.putImageData(new ImageData(pixels, result.width, result.height), 0, 0);
  }, [result, view.selected]);

  const busy = state.status === 'loading';
  const message = !textureUrl ? 'Upload an image or choose a reference scene.'
    : state.message ?? (busy ? 'Identifying six classes · GPU queue / inference…'
      : state.status === 'ready' ? 'Six-class identification ready. Heights unchanged.'
      : !online ? 'Waiting for the local engine. Identification starts automatically when it reconnects.'
      : 'Identification starts automatically when the 3D scene is ready.');

  return <section className="six-class-panel" aria-label="Six-class Class Explorer" data-classification-status={state.status}>
    <div className="six-class-heading"><span>CLASS EXPLORER</span><b>{result ? '6 CATEGORIES' : 'AUTOMATIC'}</b></div>
    <h3>Understand your scene.</h3>
    <p role="status" aria-live="polite">{message}</p>
    {busy && <progress className="six-class-progress" aria-label="Identifying scene categories" />}
    {state.status === 'error' && <button type="button" className="six-class-run" disabled={!ready || !online} onClick={() => setAttempt(value => value + 1)}>Retry identification</button>}
    {state.status === 'unsupported' && <p className="six-class-warning">Height reconstruction is still available. Use an 8-bit RGB image for six-class identification.</p>}
    <label className="six-class-toggle"><input type="checkbox" disabled={!result} checked={view.visible} onChange={event => onView({ ...view, visible: event.target.checked })} /> Show class colors</label>
    {result && <div className="six-class-map"><canvas ref={canvasRef} aria-label="Six-class map; colors follow the category legend" /></div>}
    <button type="button" className="six-class-all" disabled={!result} aria-pressed={view.selected === null} onClick={() => onView({ ...view, selected: null })}>All six classes</button>
    <div className="six-class-choices">{SIX_CLASS_NAMES.map((name, id) => {
      const item = result?.classes[id];
      return <button type="button" key={id} aria-label={`Highlight ${name}`} aria-pressed={view.selected === id} disabled={!item || item.pixels === 0}
        onClick={() => onView({ selected: id, visible: true })}>
        <i style={{ background: `rgb(${SIX_CLASS_COLORS[id].join(',')})` }} /><span>{name}<small>{!item ? 'Waiting for identification' : !item.pixels ? 'Not detected in this scene'
          : item.area_m2 === null ? 'pixel coverage' : `${item.area_m2.toLocaleString(undefined, { maximumFractionDigits: 0 })} m² map area`}</small></span>
        <b>{item ? `${item.coverage_percent.toFixed(1)}%` : '—'}</b>
      </button>;
    })}</div>
    <small>Class filters highlight and dim colors only; all heights stay intact. Inspect, Fly and Walk use these labels even with colors off.</small>
    {result && <>
      <small>{result.valid_pixels.toLocaleString()} valid pixels · {result.ignored_pixels.toLocaleString()} ignored{result.resampled ? ' · resized input' : ''}. Coverage is not confidence or an object count.</small>
      <div className="six-class-downloads"><a href={`${apiBase}${result.raster_url}`} download>Six-class GeoTIFF ↗</a><a href={`${apiBase}${result.metadata_url}`} target="_blank" rel="noreferrer">Provenance ↗</a></div>
    </>}
    {demoId === 'hilly' && <small className="six-class-warning">This 10 m scene is outside the classifier’s high-resolution training domain. Small buildings and roads may be unresolved.</small>}
    <details><summary>Model details & accuracy limits</summary><p>Identification: RGB V3, epoch 1. Development results improved on some datasets, but class/city and road-boundary regression checks failed. This integration is not an accuracy certification. Height estimates, building geometry and collisions still come from the separate protected height pipeline.</p></details>
  </section>;
}
