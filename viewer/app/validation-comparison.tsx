'use client';

import { useEffect, useRef, useState } from 'react';
import { fromArrayBuffer } from 'geotiff';

export type ReferenceValidationMetrics = {
  pixel_count: number;
  rmse_m: number;
  mae_m: number;
  bias_m: number;
  correlation: number | null;
  r2: number | null;
  reference_kind: 'ndsm' | 'dsm';
  prediction_product: string;
  alignment: string;
};

type ValidationComparisonProps = {
  open: boolean;
  onClose: () => void;
  textureUrl: string;
  predictionUrl: string;
  referenceUrl: string;
  errorUrl: string;
  metrics: ReferenceValidationMetrics;
  title: string;
  provenance: string;
  heldOut: boolean;
};

type RasterData = {
  values: Float32Array;
  width: number;
  height: number;
};

const HEIGHT_STOPS = [
  [0, [7, 24, 43]],
  [0.28, [16, 94, 116]],
  [0.56, [49, 191, 146]],
  [0.78, [185, 231, 91]],
  [1, [255, 210, 79]],
] as const;

function sampleFinite(values: Float32Array, absolute = false) {
  const step = Math.max(1, Math.floor(values.length / 120_000));
  const sampled: number[] = [];
  for (let index = 0; index < values.length; index += step) {
    const value = values[index];
    if (Number.isFinite(value)) sampled.push(absolute ? Math.abs(value) : value);
  }
  sampled.sort((left, right) => left - right);
  return sampled;
}

function percentile(sorted: number[], amount: number) {
  if (!sorted.length) return 0;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * amount)));
  return sorted[index];
}

async function readRaster(url: string): Promise<RasterData> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load validation raster (${response.status}).`);
  const tiff = await fromArrayBuffer(await response.arrayBuffer());
  const image = await tiff.getImage();
  const width = image.getWidth();
  const height = image.getHeight();
  const samples = image.getSamplesPerPixel();
  const raw = await image.readRasters({ interleave: true }) as unknown as ArrayLike<number>;
  const noData = image.getGDALNoData();
  const values = new Float32Array(width * height);
  for (let pixel = 0; pixel < values.length; pixel += 1) {
    const value = Number(raw[pixel * samples]);
    values[pixel] = Number.isFinite(value) && (noData === null || value !== noData)
      ? value
      : Number.NaN;
  }
  return { values, width, height };
}

function heightColour(amount: number) {
  const normalized = Math.min(1, Math.max(0, amount));
  for (let index = 1; index < HEIGHT_STOPS.length; index += 1) {
    if (normalized <= HEIGHT_STOPS[index][0]) {
      const [leftAt, left] = HEIGHT_STOPS[index - 1];
      const [rightAt, right] = HEIGHT_STOPS[index];
      const mix = (normalized - leftAt) / Math.max(rightAt - leftAt, 1e-6);
      return left.map((value, channel) => Math.round(value + (right[channel] - value) * mix));
    }
  }
  return [...HEIGHT_STOPS[HEIGHT_STOPS.length - 1][1]];
}

function errorColour(value: number, maximum: number) {
  const normalized = Math.min(1, Math.abs(value) / Math.max(maximum, 1e-6));
  const neutral = [230, 239, 235];
  const target = value < 0 ? [47, 128, 237] : [238, 89, 81];
  return neutral.map((channel, index) => Math.round(channel + (target[index] - channel) * normalized));
}

function drawRaster(
  canvas: HTMLCanvasElement,
  raster: RasterData,
  low: number,
  high: number,
  mode: 'height' | 'error',
) {
  const maximumWidth = 620;
  const scale = Math.min(1, maximumWidth / raster.width);
  const width = Math.max(1, Math.round(raster.width * scale));
  const height = Math.max(1, Math.round(raster.height * scale));
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) return;
  const image = context.createImageData(width, height);
  for (let row = 0; row < height; row += 1) {
    const sourceRow = Math.min(raster.height - 1, Math.round(row / Math.max(height - 1, 1) * (raster.height - 1)));
    for (let col = 0; col < width; col += 1) {
      const sourceCol = Math.min(raster.width - 1, Math.round(col / Math.max(width - 1, 1) * (raster.width - 1)));
      const value = raster.values[sourceRow * raster.width + sourceCol];
      const offset = (row * width + col) * 4;
      if (!Number.isFinite(value)) {
        image.data.set([7, 18, 24, 255], offset);
        continue;
      }
      const colour = mode === 'height'
        ? heightColour((value - low) / Math.max(high - low, 1e-6))
        : errorColour(value, high);
      image.data.set([colour[0], colour[1], colour[2], 255], offset);
    }
  }
  context.putImageData(image, 0, 0);
}

function drawOptical(canvas: HTMLCanvasElement, url: string) {
  return new Promise<void>((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      canvas.getContext('2d')?.drawImage(image, 0, 0);
      resolve();
    };
    image.onerror = () => reject(new Error('Could not load the satellite preview.'));
    image.src = url;
  });
}

export function ValidationComparison({
  open,
  onClose,
  textureUrl,
  predictionUrl,
  referenceUrl,
  errorUrl,
  metrics,
  title,
  provenance,
  heldOut,
}: ValidationComparisonProps) {
  const opticalRef = useRef<HTMLCanvasElement>(null);
  const predictionRef = useRef<HTMLCanvasElement>(null);
  const referenceRef = useRef<HTMLCanvasElement>(null);
  const errorRef = useRef<HTMLCanvasElement>(null);
  const [state, setState] = useState<'loading' | 'ready' | 'error'>('loading');

  useEffect(() => {
    if (!open) return;
    let active = true;
    Promise.resolve().then(() => active && setState('loading'));
    Promise.all([
      readRaster(predictionUrl),
      readRaster(referenceUrl),
      readRaster(errorUrl),
      opticalRef.current ? drawOptical(opticalRef.current, textureUrl) : Promise.resolve(),
    ]).then(([prediction, reference, error]) => {
      if (!active || !predictionRef.current || !referenceRef.current || !errorRef.current) return;
      const heightSamples = sampleFinite(prediction.values).concat(sampleFinite(reference.values));
      heightSamples.sort((left, right) => left - right);
      const low = Math.min(0, percentile(heightSamples, 0.02));
      const high = Math.max(low + 1, percentile(heightSamples, 0.98));
      const errorLimit = Math.max(1, percentile(sampleFinite(error.values, true), 0.95));
      drawRaster(predictionRef.current, prediction, low, high, 'height');
      drawRaster(referenceRef.current, reference, low, high, 'height');
      drawRaster(errorRef.current, error, -errorLimit, errorLimit, 'error');
      setState('ready');
    }).catch(() => active && setState('error'));
    return () => { active = false; };
  }, [open, textureUrl, predictionUrl, referenceUrl, errorUrl]);

  if (!open) return null;
  return (
    <div className="validation-overlay" role="dialog" aria-modal="true" aria-label="Independent reference comparison">
      <section className="validation-board">
        <header className="validation-board-header">
          <div><p className="eyebrow">Reviewer evidence · dense height validation</p><h2>{title}</h2><p>{provenance}</p></div>
          <div className="validation-board-actions"><span className="evidence-badge">{heldOut ? 'HELD-OUT TEST' : 'UPLOADED REFERENCE'}</span><button onClick={onClose} type="button" aria-label="Close validation comparison">×</button></div>
        </header>

        <div className="validation-proof-strip">
          <div><span>RMSE</span><strong>{metrics.rmse_m.toFixed(2)} m</strong><small>penalises large errors</small></div>
          <div><span>MAE</span><strong>{metrics.mae_m.toFixed(2)} m</strong><small>typical absolute error</small></div>
          <div><span>Correlation</span><strong>{metrics.correlation === null ? '—' : metrics.correlation.toFixed(3)}</strong><small>shape agreement</small></div>
          <div><span>R²</span><strong>{metrics.r2 === null ? '—' : metrics.r2.toFixed(3)}</strong><small>explained variation</small></div>
          <div><span>Bias</span><strong>{metrics.bias_m > 0 ? '+' : ''}{metrics.bias_m.toFixed(2)} m</strong><small>prediction − reference</small></div>
        </div>

        <div className={`validation-quad ${state}`}>
          <article><div className="comparison-label"><span>01</span><div><strong>Satellite RGB</strong><small>single optical input</small></div></div><canvas ref={opticalRef} role="img" aria-label="Satellite RGB input" /></article>
          <article><div className="comparison-label"><span>02</span><div><strong>Model prediction</strong><small>{metrics.reference_kind === 'dsm' ? 'predicted absolute surface DSM · metres' : 'predicted height-above-ground nDSM · metres'}</small></div></div><canvas ref={predictionRef} role="img" aria-label="Predicted height map" /></article>
          <article><div className="comparison-label"><span>03</span><div><strong>{metrics.reference_kind === 'dsm' ? 'Independent DEM reference' : 'LiDAR reference'}</strong><small>{metrics.reference_kind === 'dsm' ? 'separate coarse terrain elevation source' : 'independent LiDAR-derived height label'}</small></div></div><canvas ref={referenceRef} role="img" aria-label="Independent reference height map" /></article>
          <article><div className="comparison-label"><span>04</span><div><strong>Signed error</strong><small>blue = under · red = over</small></div></div><canvas ref={errorRef} role="img" aria-label="Signed prediction error map" /></article>
          {state === 'loading' && <p className="validation-loading">Rendering aligned evidence…</p>}
          {state === 'error' && <p className="validation-loading error">Evidence rasters could not be rendered.</p>}
        </div>

        <footer className="validation-board-footer">
          <p><strong>{metrics.pixel_count.toLocaleString()}</strong> aligned pixels · {metrics.alignment.replaceAll('_', ' ')} · same height grid and units</p>
          <p>These are honest model outputs. The independent reference is used only for evaluation, never as an inference input.</p>
        </footer>
      </section>
    </div>
  );
}
