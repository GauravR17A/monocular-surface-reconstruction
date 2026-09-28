import test from 'node:test';
import assert from 'node:assert/strict';
import { clampImageView, FIT_IMAGE_VIEW, zoomImageView } from '../app/image-reference-view.ts';

const bounds = { width: 800, height: 600, imageWidth: 1000, imageHeight: 1000 };
test('Fit always centres the full image, preserving letterboxing', () => {
  assert.deepEqual(clampImageView({ zoom: 1, x: 50, y: -50 }, bounds), FIT_IMAGE_VIEW);
  assert.deepEqual(clampImageView({ zoom: 1, x: 999, y: 999 }, { ...bounds, imageWidth: 4000 }), FIT_IMAGE_VIEW);
});
test('Pan is constrained to real image edges after zoom, not its square wrapper', () => {
  assert.deepEqual(clampImageView({ zoom: 2, x: 999, y: -999 }, bounds), { zoom: 2, x: 200, y: -300 });
  assert.deepEqual(clampImageView({ zoom: 2, x: 999, y: 999 }, { ...bounds, imageWidth: 4000 }), { zoom: 2, x: 400, y: 0 });
});
test('Zoom keeps cursor anchor stable within pan limits', () => {
  assert.deepEqual(zoomImageView(FIT_IMAGE_VIEW, 2, bounds, { x: 100, y: 80 }), { zoom: 2, x: -100, y: -80 });
});
test('Zoom range, fit reset and not-yet-loaded images are safe', () => {
  assert.equal(zoomImageView(FIT_IMAGE_VIEW, 100, bounds).zoom, 8);
  assert.deepEqual(zoomImageView({ zoom: 3, x: 200, y: 100 }, .1, bounds), FIT_IMAGE_VIEW);
  assert.deepEqual(clampImageView({ zoom: 2, x: 5, y: 5 }, { ...bounds, imageWidth: 0 }), { zoom: 2, x: 0, y: 0 });
});
