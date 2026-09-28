import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ALL_CLASS_HEIGHTS, classFocusFrame, classCoverage, focusedVertexHeights } from '../app/class-focus.ts';
import { surfaceClassCode, querySurface } from '../app/surface-query.ts';

test('every selected class flattens first, rises alone and keeps unknown flat', () => {
  for (const selected of [0, 1, 2]) {
    assert.deepEqual(classFocusFrame(ALL_CLASS_HEIGHTS, selected, 0).weights, [1, 1, 1, 1]);
    assert.deepEqual(classFocusFrame(ALL_CLASS_HEIGHTS, selected, .32).weights, [0, 0, 0, 0]);
    const result = classFocusFrame(ALL_CLASS_HEIGHTS, selected, 1.1);
    assert.ok(result.done);
    result.weights.forEach((w, c) => assert.equal(w, c === selected ? 1 : 0));
  }
});
test('rapid switching starts from the current geometry, reset restores every class', () => {
  const halfway = classFocusFrame(ALL_CLASS_HEIGHTS, 1, .7).weights;
  assert.deepEqual(classFocusFrame(halfway, 2, 0).weights, halfway);
  assert.deepEqual(classFocusFrame(halfway, null, .65).weights, ALL_CLASS_HEIGHTS);
  assert.deepEqual(classFocusFrame(halfway, 2, 0, true).weights, [0, 0, 1, 0]);
  assert.deepEqual(classFocusFrame(halfway, null, 0, true).weights, ALL_CLASS_HEIGHTS);
});
test('animation remains bounded and finishes identically at different refresh rates', () => {
  for (const fps of [30, 60, 144]) {
    for (let i = 0; i < fps * 2; i++) {
      const f = classFocusFrame(ALL_CLASS_HEIGHTS, 2, i / fps);
      assert.ok(f.weights.every(v => v >= 0 && v <= 1));
    }
    assert.deepEqual(classFocusFrame(ALL_CLASS_HEIGHTS, 2, 2).weights, [0, 0, 1, 0]);
  }
});
test('display isolation cannot modify source/collision heights, xz positions or raw metrics', () => {
  const heights = new Float32Array([4, 9, 15, 3]), original = heights.slice();
  const classes = new Uint8Array([0, 1, 2, 3]);
  const positions = new Float32Array([1,4,2, 3,9,4, 5,15,6, 7,3,8]);
  focusedVertexHeights(heights, classes, [0, 1, 0, 0], positions);
  assert.deepEqual([...positions], [1,0,2, 3,9,4, 5,0,6, 7,0,8]);
  assert.deepEqual(heights, original);
  focusedVertexHeights(heights, classes, ALL_CLASS_HEIGHTS, positions);
  assert.deepEqual([...positions], [1,4,2, 3,9,4, 5,15,6, 7,3,8]);
});
test('coverage is pixel coverage and unknown/no-data never turn into a named class', () => {
  const c = classCoverage(new Uint8Array([0,1,1,2,3,255]), new Uint8Array([1,1,1,1,1,0]));
  assert.deepEqual(c.counts, [1,2,1,1]); assert.equal(c.total, 5);
  assert.deepEqual(c.percentages, [20,40,20,20]);
  assert.equal(classCoverage(new Uint8Array([0]), new Uint8Array([0])).total, 0);
});
test('class resolver agrees with inspector for labels, RGB assist, unknowns and missing heights', () => {
  const raster = values => ({ values: new Float32Array(values), stats: { width: 2, height: 2, units: 'm' } });
  const q = { raw: raster([3,8,9,4]), display: raster([3,8,9,4]), semantic: raster([0,1,2,255]),
    vegetation: null, building: null, confidence: null, rgba: null, textureWidth: 2, textureHeight: 2 };
  for (const [u,v,code] of [[0,0,0],[1,0,1],[0,1,2],[1,1,3]]) {
    assert.equal(surfaceClassCode(q,u,v), code);
    if (code < 3) assert.equal(querySurface(q,u,v).kind, ['surface','building','vegetation'][code]);
  }
  q.semantic = null; q.rgba = new Uint8ClampedArray([20,100,20,255, 80,80,80,255, 80,80,80,255, 80,80,80,255]);
  assert.equal(surfaceClassCode(q,0,0),2); assert.equal(querySurface(q,0,0).kind,'vegetation');
  q.raw.values[0] = NaN; assert.equal(surfaceClassCode(q,0,0),3);
});
