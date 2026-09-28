'use client';

import { useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { localMapWindow, mapDistanceLabel, mapScreenPoint, mapSpace, type MapFrame, type MapScene, type MapSpace, type MapWindow } from './explore-map-math';

function MapCanvas({ image, space, feed, localWindow, full = false }: { image: HTMLImageElement; space: MapSpace; feed: RefObject<MapFrame | null>; localWindow: RefObject<MapWindow | null>; full?: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const captionRef = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current, caption = captionRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !caption) return;
    let id = 0, previousTime = 0, previousPose: MapFrame | null = null, speed = 0, visibleSpan = 0;
    let cssWidth = 1, cssHeight = 1;
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    const resize = new ResizeObserver(entries => { cssWidth = entries[0].contentRect.width; cssHeight = entries[0].contentRect.height; });
    resize.observe(canvas);
    const draw = (now: number) => {
      id = requestAnimationFrame(draw);
      if (now - previousTime < 32 || document.hidden || cssWidth < 32 || cssHeight < 32) return;
      const pose = feed.current;
      if (!pose) return;
      const dt = Math.min(.15, (now - previousTime) / 1000 || .032);
      const instantaneous = previousPose ? Math.hypot((pose.u - previousPose.u) * space.width, (pose.v - previousPose.v) * space.height) / dt : 0;
      speed += (instantaneous - speed) * (1 - Math.exp(-dt * 3));
      const target = localMapWindow(space, pose, speed);
      visibleSpan = visibleSpan ? visibleSpan + (target.span - visibleSpan) * (1 - Math.exp(-dt * 3)) : target.span;
      const window: MapWindow = full && localWindow.current ? localWindow.current : { ...target, span: visibleSpan };
      if (!full) localWindow.current = window;
      const w = cssWidth, h = cssHeight, dpr = Math.min(devicePixelRatio || 1, 2);
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#06130f'; ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = '#25453766'; ctx.lineWidth = 1;
      for (let i = 0; i < Math.max(w, h); i += 24) { ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, h); ctx.moveTo(0, i); ctx.lineTo(w, i); ctx.stroke(); }
      const point = mapScreenPoint(space, pose, window, w, h, full);
      ctx.save(); ctx.translate(w / 2, h / 2); ctx.rotate(point.angle); ctx.scale(point.scale, point.scale); ctx.translate(-point.centreX, -point.centreY);
      ctx.drawImage(image, 0, 0, space.width, space.height);
      ctx.lineWidth = 1.5 / point.scale; ctx.strokeStyle = '#aeeed3'; ctx.strokeRect(0, 0, space.width, space.height);
      if (full) {
        // Show the currently tracked local window on the overview, without
        // suggesting that the crop is new data or a second reconstruction.
        ctx.save(); ctx.translate(window.x, window.y); ctx.rotate(window.heading);
        ctx.fillStyle = '#45f3c41a'; ctx.strokeStyle = '#65ffccbb'; ctx.setLineDash([6 / point.scale, 4 / point.scale]);
        ctx.fillRect(-window.span / 2, -window.span / 2, window.span, window.span);
        ctx.strokeRect(-window.span / 2, -window.span / 2, window.span, window.span); ctx.restore();
      }
      ctx.restore();
      ctx.save(); ctx.translate(point.x, point.y);
      const pulse = reducedMotion.matches ? 0 : (Math.sin(now / 520) + 1) / 2;
      ctx.beginPath(); ctx.arc(0, 0, 17 + pulse * 4, 0, Math.PI * 2);
      ctx.lineWidth = 1.2; ctx.strokeStyle = window.outside ? '#ffc36d77' : '#79ffce77'; ctx.stroke();
      ctx.rotate(full ? window.heading : 0);
      ctx.fillStyle = window.outside ? '#ffc36d33' : '#70ffd33b';
      ctx.beginPath(); ctx.moveTo(0, 0); ctx.arc(0, 0, full ? 40 : 32, -Math.PI / 2 - .43, -Math.PI / 2 + .43); ctx.closePath(); ctx.fill();
      ctx.shadowColor = window.outside ? '#ffb94b' : '#45ffc4'; ctx.shadowBlur = 14; ctx.lineWidth = 3; ctx.strokeStyle = '#052b20'; ctx.fillStyle = window.outside ? '#ffd084' : '#82ffda';
      ctx.beginPath(); ctx.moveTo(0, -12); ctx.lineTo(8, 9); ctx.lineTo(0, 5); ctx.lineTo(-8, 9); ctx.closePath(); ctx.stroke(); ctx.fill();
      ctx.shadowBlur = 0; ctx.fillStyle = '#f1fff9'; ctx.beginPath(); ctx.moveTo(0, -7); ctx.lineTo(3, 4); ctx.lineTo(0, 2); ctx.lineTo(-3, 4); ctx.closePath(); ctx.fill(); ctx.restore();
      ctx.fillStyle = '#061a12df'; ctx.fillRect(7, 7, full ? 128 : 100, 21);
      ctx.fillStyle = '#d0efdf'; ctx.font = '9px monospace'; ctx.fillText(full ? '↑ IMAGE TOP' : '↑ FACING DIRECTION', 13, 21);
      const distance = window.span / 4;
      if (!full) { ctx.strokeStyle = '#d9fff0'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(10, h - 19); ctx.lineTo(10 + w / 4, h - 19); ctx.stroke();
        ctx.font = '9px monospace'; ctx.fillStyle = '#03100ce8'; ctx.fillRect(6, h - 17, 91, 15); ctx.fillStyle = '#d9fff0'; ctx.fillText(mapDistanceLabel(distance, space.unit), 10, h - 6); }
      caption.textContent = window.outside ? 'Outside image · marker at edge' : full
        ? `${mapDistanceLabel(space.width, space.unit)} × ${mapDistanceLabel(space.height, space.unit)} · arrow = you`
        : `${mapDistanceLabel(window.span, space.unit)} across · following you`;
      canvas.dataset.mapFrame = JSON.stringify({ ...pose, heading: window.heading, span: window.span, centreX: window.x, centreY: window.y,
        markerX: point.x, markerY: point.y, width: w, height: h, outside: window.outside, unit: space.unit, full });
      previousPose = pose; previousTime = now;
    };
    id = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(id); resize.disconnect(); delete canvas.dataset.mapFrame; };
  }, [image, space, feed, localWindow, full]);
  return <><canvas ref={canvasRef} className={full ? 'explore-map-canvas full' : 'explore-map-canvas'} aria-label={full ? 'Full image map with your position and direction' : 'Live local map with your position and direction'} /><span ref={captionRef} className="explore-map-caption">Locating you…</span></>;
}

export function ExploreMap({ src, label, scene, feed, open, onToggle, onClose }: { src: string; label: string; scene: MapScene;
  feed: RefObject<MapFrame | null>; open: boolean; onToggle: () => void; onClose: () => void }) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [failed, setFailed] = useState(false);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const localWindow = useRef<MapWindow | null>(null);
  const space = useMemo(() => mapSpace(scene), [scene]);
  useEffect(() => {
    let active = true;
    const source = new window.Image();
    source.onload = () => { if (active) setImage(source); };
    source.onerror = () => { if (active) setFailed(true); };
    source.src = src;
    return () => { active = false; source.onload = null; source.onerror = null; };
  }, [src]);
  useEffect(() => {
    const dialog = dialogRef.current;
    if (open && !dialog?.open) dialog?.showModal();
    if (!open && dialog?.open) dialog.close();
    return () => { if (dialog?.open) dialog.close(); };
  }, [open]);
  const placeholder = <span className="explore-map-placeholder" role="status">{failed ? 'Map image unavailable' : 'Loading map…'}</span>;
  return <>
    <section className="explore-minimap" aria-label="Explore minimap">
      <div className="explore-map-heading"><span><i />LIVE MAP</span><kbd>M</kbd></div>
      <button type="button" className="explore-map-open" aria-label="Open full map (M)" onClick={onToggle}>
        {image ? <MapCanvas image={image} space={space} feed={feed} localWindow={localWindow} /> : placeholder}
      </button>
      <small>{space.unit === 'm' ? 'Map-scaled view' : 'Pixel scale · distances unknown'} · M: full map</small>
    </section>
    <dialog ref={dialogRef} className="explore-map-dialog" aria-labelledby="explore-map-title" onCancel={event => { event.preventDefault(); onClose(); }}
      onClick={event => { if (event.target === event.currentTarget) onClose(); }}>
      <div className="explore-full-map-shell">
        <header><div><span>EXPLORE / POSITION MAP</span><h2 id="explore-map-title">Your position</h2><p title={label}>{label}</p></div>
          <button type="button" onClick={onClose} aria-label="Close full map (M)" autoFocus>×</button></header>
        <div className="explore-full-map-body">{image && open ? <MapCanvas image={image} space={space} feed={feed} localWindow={localWindow} full /> : placeholder}</div>
        <footer><span>Image-up · not necessarily north · dashed box = local map</span><strong>Movement paused · M / Esc to return</strong></footer>
      </div>
    </dialog>
  </>;
}
