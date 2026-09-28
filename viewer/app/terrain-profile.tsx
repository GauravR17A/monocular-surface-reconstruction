'use client';

import { useMemo, useState } from 'react';
import { profileCsv, type TerrainProfileData } from './terrain-analysis';

export function TerrainProfile({ profile, picking, onReset, onClose }: {
  profile: TerrainProfileData | null; picking: boolean; onReset: () => void; onClose: () => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const csvUrl = useMemo(() => profile ? `data:text/csv;charset=utf-8,${encodeURIComponent(profileCsv(profile))}` : null, [profile]);
  const values = profile?.samples.flatMap(p => [p.surface, p.ground].filter((n): n is number => n !== null)) ?? [];
  const minimum = values.length ? Math.min(...values) : 0, maximum = values.length ? Math.max(...values) : 1;
  const range = Math.max(.01, maximum - minimum), samples = profile?.samples ?? [];
  const x = (i: number) => 40 + i / Math.max(1, samples.length - 1) * 520;
  const y = (value: number) => 106 - (value - minimum) / range * 82;
  function path(key: 'surface' | 'ground') {
    let drawing = false;
    return samples.map((sample, i) => {
      const value = sample[key];
      if (value === null) { drawing = false; return ''; }
      const command = `${drawing ? 'L' : 'M'}${x(i).toFixed(2)},${y(value).toFixed(2)}`;
      drawing = true; return command;
    }).join(' ');
  }
  const unit = profile?.heightUnit === 'm' ? 'm' : 'rel.';
  const active = hover === null ? null : samples[Math.min(hover, samples.length - 1)];
  return <section className="terrain-profile-panel" aria-label="Measured terrain profile">
    <div className="terrain-profile-heading"><div><span>SURFACE CROSS-SECTION</span><strong>{profile ? 'A → B · source elevations' : picking ? 'Point A set. Click point B.' : 'Click two points on the terrain'}</strong></div><div>
      {csvUrl ? <a href={csvUrl} download="msr-surface-profile.csv">CSV ↓</a> : <button type="button" disabled>CSV ↓</button>}<button type="button" onClick={onReset}>Reset</button><button type="button" aria-label="Close surface profile" onClick={onClose}>×</button>
    </div></div>
    {profile ? <>
      <svg viewBox="0 0 580 130" role="img" aria-label={`Surface profile, ${profile.distance.toFixed(1)} ${profile.distanceUnit}, elevation from ${profile.minimum?.toFixed(1) ?? 'unknown'} to ${profile.maximum?.toFixed(1) ?? 'unknown'} ${unit}`}
        onPointerMove={event => { const matrix = event.currentTarget.getScreenCTM(); if (!matrix) return; const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse()); setHover(Math.round(Math.max(0, Math.min(1, (point.x - 40) / 520)) * (samples.length - 1))); }} onPointerLeave={() => setHover(null)}>
        {[0, .5, 1].map(t => <g key={t}><line x1="40" x2="560" y1={24 + 82 * t} y2={24 + 82 * t} stroke="#27453f" strokeDasharray="3 5" /><text x="0" y={28 + 82 * t}>{(maximum - range * t).toFixed(range > 10 ? 0 : 2)}</text></g>)}
        <text x="40" y="13">Elevation ({unit})</text>
        <path d={path('ground')} fill="none" stroke="#ffc857" strokeWidth="1.5" strokeDasharray="4 3" />
        <path d={path('surface')} fill="none" stroke="#45f3c4" strokeWidth="2" />
        <text x="40" y="124">A · 0</text><text x="560" y="124" textAnchor="end">B · {profile.distance.toFixed(1)} {profile.distanceUnit}</text>
        {active && <line x1={x(hover!)} x2={x(hover!)} y1="20" y2="108" stroke="#e9fff8" opacity=".7" />}
      </svg>
      <div className="terrain-profile-numbers"><span>Distance <b>{profile.distance.toFixed(1)} {profile.distanceUnit}</b></span><span>A / B <b>{samples[0]?.surface?.toFixed(1) ?? '—'} / {samples.at(-1)?.surface?.toFixed(1) ?? '—'} {unit}</b></span><span>Δ elevation <b>{profile.delta?.toFixed(1) ?? '—'} {unit}</b></span><span>Ascent / descent <b>{profile.ascent.toFixed(1)} / {profile.descent.toFixed(1)} {unit}</b></span></div>
      <p>{active ? `${active.distance.toFixed(1)} ${profile.distanceUnit} · surface ${active.surface?.toFixed(1) ?? 'nodata'} ${unit}${active.ground !== null ? ` · terrain reference ${active.ground.toFixed(1)} ${unit}` : ''}` : `Green: surface${samples.some(p => p.ground !== null) ? ' · amber: terrain reference' : ''} · sampled raster, independent of display exaggeration`}{profile.gaps ? ' · Gaps: nodata; ascent/descent exclude gaps.' : ''}</p>
    </> : <p>Measures horizontal distance and elevation change from the source raster. Drag to orbit; click to place A and B.</p>}
  </section>;
}
