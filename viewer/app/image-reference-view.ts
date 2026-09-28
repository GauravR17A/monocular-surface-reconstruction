export type ImageView = { zoom: number; x: number; y: number };
export type ImageViewport = { width: number; height: number; imageWidth: number; imageHeight: number };
export const FIT_IMAGE_VIEW: ImageView = { zoom: 1, x: 0, y: 0 };

export function clampImageView(view: ImageView, bounds: ImageViewport): ImageView {
  const zoom = Math.max(1, Math.min(8, view.zoom));
  if (bounds.width <= 0 || bounds.height <= 0 || bounds.imageWidth <= 0 || bounds.imageHeight <= 0) return { zoom, x: 0, y: 0 };
  const fit = Math.min(bounds.width / bounds.imageWidth, bounds.height / bounds.imageHeight);
  const limitX = Math.max(0, (bounds.imageWidth * fit * zoom - bounds.width) / 2);
  const limitY = Math.max(0, (bounds.imageHeight * fit * zoom - bounds.height) / 2);
  return { zoom, x: limitX ? Math.max(-limitX, Math.min(limitX, view.x)) : 0, y: limitY ? Math.max(-limitY, Math.min(limitY, view.y)) : 0 };
}

// Keep the point under the cursor stationary when zooming, unless an image edge
// needs clamping. Zoom is relative to fit-to-window, not an accuracy/resolution gain.
export function zoomImageView(view: ImageView, zoom: number, bounds: ImageViewport, anchor = { x: 0, y: 0 }): ImageView {
  const nextZoom = Math.max(1, Math.min(8, zoom));
  const ratio = nextZoom / view.zoom;
  return clampImageView({ zoom: nextZoom, x: anchor.x - (anchor.x - view.x) * ratio, y: anchor.y - (anchor.y - view.y) * ratio }, bounds);
}
