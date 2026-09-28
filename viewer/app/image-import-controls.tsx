'use client';

import { useEffect, useRef, useState, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { IMAGE_ACCEPT, IMAGE_FORMATS, clipboardImageFile, clipboardImageType, filesFromTransfer, prepareImageImport } from './image-import';

type Props = {
  inputRef: RefObject<HTMLInputElement | null>;
  disabled: boolean;
  onImport: (file: File) => void;
};

export function ImageImportControls({ inputRef, disabled, onImport }: Props) {
  const [dragging, setDragging] = useState(false);
  const [choices, setChoices] = useState<File[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const [readingClipboard, setReadingClipboard] = useState(false);
  const [portal, setPortal] = useState<Element | null>(null);
  const handlers = useRef({ disabled, onImport });
  const mounted = useRef(false);
  const accepting = useRef(false);
  const dragDepth = useRef(0);
  const choiceFocus = useRef<HTMLButtonElement>(null);

  useEffect(() => { handlers.current = { disabled, onImport }; }, [disabled, onImport]);
  useEffect(() => { if (!disabled) accepting.current = false; }, [disabled]);
  useEffect(() => { if (choices.length) choiceFocus.current?.focus(); }, [choices]);

  function start(file: File) {
    if (handlers.current.disabled || accepting.current) {
      setNotice('An image is still processing. Wait for the scene to finish, then import again.');
      return;
    }
    accepting.current = true;
    setChoices([]); setNotice(null);
    handlers.current.onImport(file);
  }

  function receive(files: File[]) {
    if (handlers.current.disabled || accepting.current) {
      setNotice('An image is still processing. Wait for the scene to finish, then import again.');
      return;
    }
    const valid: File[] = [], errors: string[] = [];
    for (const file of files) {
      try { valid.push(prepareImageImport(file)); }
      catch (reason) { errors.push(reason instanceof Error ? reason.message : String(reason)); }
    }
    setChoices([]);
    if (valid.length === 1 && !errors.length) start(valid[0]);
    else {
      setChoices(valid);
      setNotice(errors.slice(0, 3).join(' ') || (!valid.length ? 'No image file was supplied. Drop an image file or paste a copied image.' : null));
    }
  }

  // These listeners only handle file transfers. Normal text paste and canvas
  // orbit/drag gestures keep their existing behavior.
  useEffect(() => {
    mounted.current = true;
    const updatePortal = () => setPortal(document.fullscreenElement ?? document.body);
    updatePortal();
    const fileDrag = (event: DragEvent) => Array.from(event.dataTransfer?.types ?? []).includes('Files');
    const clear = () => { dragDepth.current = 0; setDragging(false); };
    const enter = (event: DragEvent) => {
      if (!fileDrag(event)) return;
      event.preventDefault(); dragDepth.current++; setDragging(true);
      if (handlers.current.disabled || accepting.current) setNotice('An image is still processing. Wait for the scene to finish, then import again.');
    };
    const over = (event: DragEvent) => {
      if (!fileDrag(event)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = handlers.current.disabled ? 'none' : 'copy';
    };
    const leave = (event: DragEvent) => {
      if (!fileDrag(event)) return;
      dragDepth.current = Math.max(0, dragDepth.current - 1);
      if (!dragDepth.current || !event.relatedTarget) clear();
    };
    const drop = (event: DragEvent) => {
      if (!fileDrag(event)) return;
      event.preventDefault(); clear(); receive(filesFromTransfer(event.dataTransfer));
    };
    const paste = (event: ClipboardEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      if (target?.closest('textarea, [contenteditable="true"], [role="textbox"], input:not([type="range"]):not([type="checkbox"]):not([type="radio"]):not([type="file"])')) return;
      const files = filesFromTransfer(event.clipboardData);
      if (!files.length) return;
      event.preventDefault(); receive(files);
    };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') { clear(); setChoices([]); setNotice(null); } };
    document.addEventListener('dragenter', enter);
    document.addEventListener('dragover', over);
    document.addEventListener('dragleave', leave);
    document.addEventListener('drop', drop);
    document.addEventListener('paste', paste);
    document.addEventListener('keydown', escape);
    document.addEventListener('fullscreenchange', updatePortal);
    window.addEventListener('blur', clear);
    return () => {
      mounted.current = false;
      document.removeEventListener('dragenter', enter);
      document.removeEventListener('dragover', over);
      document.removeEventListener('dragleave', leave);
      document.removeEventListener('drop', drop);
      document.removeEventListener('paste', paste);
      document.removeEventListener('keydown', escape);
      document.removeEventListener('fullscreenchange', updatePortal);
      window.removeEventListener('blur', clear);
    };
    // receive/start read changing props through handlers, so global listeners stay stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function pasteImage() {
    setReadingClipboard(true);
    try {
      if (!navigator.clipboard?.read) throw new Error('Clipboard reading is unavailable.');
      const items = await navigator.clipboard.read();
      const files: File[] = [];
      for (const item of items) {
        const type = clipboardImageType(item.types);
        if (type) files.push(clipboardImageFile(await item.getType(type), files.length));
      }
      if (!mounted.current) return;
      if (!files.length) setNotice('No image in the clipboard. Copy an image or screenshot, then paste again. For a file copied in Explorer, try Ctrl+V or drag the original file here.');
      else receive(files);
    } catch {
      if (mounted.current) setNotice('Clipboard access is unavailable or was denied. Press Ctrl+V / ⌘V with a copied image, or drop the original image file here.');
    } finally {
      if (mounted.current) setReadingClipboard(false);
    }
  }

  return <>
    <div className="image-import-actions">
      <div><button className="primary-button" disabled={disabled} onClick={() => inputRef.current?.click()} type="button">{disabled ? 'Processing…' : 'Open image'}</button>
        <button className="paste-image-button" disabled={disabled || readingClipboard} onClick={() => void pasteImage()} type="button">{readingClipboard ? 'Reading…' : 'Paste image'}</button></div>
      <small>Drop anywhere · Ctrl+V / ⌘V</small>
      <input ref={inputRef} hidden multiple accept={IMAGE_ACCEPT} aria-label="Import images" type="file" onChange={event => { const files = Array.from(event.target.files ?? []); event.target.value = ''; if (files.length) receive(files); }} />
    </div>
    {portal && createPortal(<>
      {dragging && <div className="image-drop-overlay" role="status"><div><span>↓</span><strong>{disabled ? 'An image is still processing' : 'Drop your image to build a scene'}</strong><p>{IMAGE_FORMATS}</p><small>Drop the original GeoTIFF to preserve its map coordinates.</small></div></div>}
      {(notice || choices.length > 0) && <section className="image-import-feedback" aria-label="Image import" role="region">
        <button className="image-import-dismiss" aria-label="Dismiss import message" onClick={() => { setChoices([]); setNotice(null); }} type="button">×</button>
        {notice && <p role="alert">{notice}</p>}
        {choices.length > 0 && <><strong>Choose one image to view</strong><p>Each image opens its own scene.</p><div className="image-import-choices">{choices.map((file, index) => <button ref={index === 0 ? choiceFocus : undefined} key={`${index}-${file.name}`} disabled={disabled} type="button" onClick={() => start(file)}><span>{file.name}</span><small>{(file.size / 1024 / 1024).toFixed(1)} MB</small></button>)}</div></>}
      </section>}
    </>, portal)}
  </>;
}
