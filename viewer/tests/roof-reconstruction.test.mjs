import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { fitRoof, buildFittedRoof, fittedRoofHeight } from '../app/roof-reconstruction.ts';

const outline = [
  [0.2, 0.1],
  [0.8, 0.1],
  [0.8, 0.9],
  [0.2, 0.9],
];
function fixture(fn = (x) => 15 - 10 * Math.abs(x - 0.5), ridge = true, size = 101) {
  const values = new Float32Array(size * size),
    rgba = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      values[y * size + x] = fn(x / (size - 1), y / (size - 1));
      const colour = ridge ? (x < (size - 1) * 0.5 ? 80 : 150) : 100;
      rgba.set([colour, colour * 0.8, colour * 0.65, 255], (y * size + x) * 4);
    }
  const stats = { width: size, height: size, units: 'm' };
  return {
    source: {
      height: { values, stats },
      semantic: { values: new Float32Array(size * size).fill(1), stats },
      probability: null,
    },
    image: { rgba, width: size, height: size },
  };
}
const fit = (f, polygon = outline) => fitRoof(polygon, f.source, f.image, 14);

test('recovers two roof planes, sharp ridge and lower eaves without changing inputs', () => {
  const f = fixture(),
    before = f.source.height.values.slice(),
    result = fit(f);
  assert.ok(result.fit, result.reason);
  assert.equal(result.fit.kind, 'gable');
  assert.ok(result.fit.checkMaeM < 0.05);
  assert.ok(Math.abs(fittedRoofHeight(result.fit, 0.5, 0.5) - 15) < 0.05);
  assert.ok(Math.abs(fittedRoofHeight(result.fit, 0.2, 0.5) - 12) < 0.1);
  assert.deepEqual(f.source.height.values, before);
  const geometry = buildFittedRoof(outline, result.fit, 100, 100, 0.4, 2);
  const mesh = new THREE.Mesh(geometry, [new THREE.MeshBasicMaterial(), new THREE.MeshBasicMaterial()]);
  mesh.updateMatrixWorld();
  for (const u of [0.25, 0.45, 0.55, 0.75]) {
    const ray = new THREE.Raycaster(new THREE.Vector3((u - 0.5) * 100, 50, 0), new THREE.Vector3(0, -1, 0));
    const hit = ray.intersectObject(mesh)[0];
    assert.ok(hit);
    assert.equal(hit.face.materialIndex, 0);
    assert.ok(hit.face.normal.y > 0);
    assert.ok(Math.abs(hit.point.y - ((2 + 15 - 10 * Math.abs(u - 0.5)) * 0.4 + 0.002)) < 0.02);
    assert.ok(Math.abs(hit.uv.x - u) < 1e-5);
    assert.ok(Math.abs(hit.uv.y - 0.5) < 1e-5);
  }
  const normals = geometry.attributes.normal;
  const left = new THREE.Vector3().fromBufferAttribute(normals, 0);
  assert.ok(
    Array.from({ length: geometry.groups[0].count }, (_, i) => normals.getX(i)).some((x) => x * left.x < 0),
    'separate normals across ridge',
  );
  // The roof has been replaced, not floored to the old 14 m summary.
  assert.ok(geometry.boundingBox.min.y < 2 * 0.4);
  geometry.dispose();
  mesh.material.forEach((m) => m.dispose());
});

test('brown or black colour alone never manufactures roof pitch', () => {
  for (const colour of [12, 80, 150]) {
    const f = fixture(() => 14);
    for (let i = 0; i < f.image.rgba.length; i += 4)
      f.image.rgba.set([colour, colour * 0.6, colour * 0.4], i);
    assert.equal(fit(f).fit, null);
  }
});
test('a tilted plane is not misclassified as a gable', () =>
  assert.equal(fit(fixture((x) => 10 + x * 8)).fit, null));
test('height ridge without optical evidence is rejected', () =>
  assert.equal(fit(fixture(undefined, false)).fit, null));
test('an optical stripe on a flat roof is rejected', () => assert.equal(fit(fixture(() => 14)).fit, null));
test('relative, unclassified, no-data and image-less inputs fail closed', () => {
  for (const variant of ['relative', 'semantic', 'nan', 'image']) {
    const f = fixture();
    if (variant === 'relative') f.source.height.stats.units = 'relative';
    if (variant === 'semantic') f.source.semantic.values.fill(255);
    if (variant === 'nan') f.source.height.values.fill(NaN);
    if (variant === 'image') f.image = null;
    assert.equal(fit(f).fit, null, variant);
  }
});
test('complex, clipped and small footprints need roof-part decomposition first', () => {
  const f = fixture();
  for (const poly of [
    [
      [0.2, 0.1],
      [0.8, 0.1],
      [0.8, 0.3],
      [0.4, 0.3],
      [0.4, 0.9],
      [0.2, 0.9],
    ],
    [
      [0, 0.1],
      [0.8, 0.1],
      [0.8, 0.9],
      [0, 0.9],
    ],
    [
      [0.4, 0.4],
      [0.5, 0.4],
      [0.5, 0.5],
      [0.4, 0.5],
    ],
  ])
    assert.equal(fit(f, poly).fit, null);
});
test('a courtyard with missing building support cannot become a full roof', () => {
  const f = fixture();
  for (let y = 30; y < 70; y++) for (let x = 30; x < 70; x++) f.source.semantic.values[y * 101 + x] = 0;
  assert.equal(fit(f).fit, null);
});
test('isolated spikes on otherwise flat roofs do not create pitch', () => {
  const f = fixture(() => 12);
  for (let i = 0; i < 30; i++) f.source.height.values[(i * 137) % 10201] = 35;
  assert.equal(fit(f).fit, null);
});
test('rotation preserves a supported ridge', () => {
  const size = 121,
    angle = 0.35,
    c = Math.cos(angle),
    s = Math.sin(angle);
  const poly = outline.map(([u, v]) => [
    (u - 0.5) * c - (v - 0.5) * s + 0.5,
    (u - 0.5) * s + (v - 0.5) * c + 0.5,
  ]);
  const f = fixture((u, v) => 15 - 10 * Math.abs((u - 0.5) * c + (v - 0.5) * s), true, size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      const colour = (x / (size - 1) - 0.5) * c + (y / (size - 1) - 0.5) * s < 0 ? 80 : 150;
      f.image.rgba.set([colour, colour, colour, 255], (y * size + x) * 4);
    }
  const result = fit(f, poly);
  assert.ok(result.fit, result.reason);
  assert.ok(result.fit.checkMaeM < 0.2);
});
test('a small interior courtyard is rejected even with over 90 percent support', () => {
  const f = fixture();
  for (let y = 44; y <= 56; y++) for (let x = 44; x <= 56; x++) f.source.semantic.values[y * 101 + x] = 0;
  assert.equal(fit(f).fit, null);
});
test('non-square raster coordinates keep a correct ridge height', () => {
  const width = 101,
    height = 161,
    values = new Float32Array(width * height),
    rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      values[y * width + x] = 15 - 10 * Math.abs(x / (width - 1) - 0.5);
      const c = x < 50 ? 70 : 150;
      rgba.set([c, c, c, 255], (y * width + x) * 4);
    }
  const stats = { width, height, units: 'm' };
  const result = fit({
    source: {
      height: { values, stats },
      semantic: { values: new Float32Array(width * height).fill(1), stats },
      probability: null,
    },
    image: { rgba, width, height },
  });
  assert.ok(result.fit, result.reason);
  assert.ok(Math.abs(fittedRoofHeight(result.fit, 0.5, 0.5) - 15) < 0.1);
});
test('roof colour changes do not change fitted heights when the ridge evidence is retained', () => {
  const brown = fixture(),
    black = fixture();
  for (let i = 0; i < black.image.rgba.length; i += 4)
    for (let c = 0; c < 3; c++) black.image.rgba[i + c] = Math.round(black.image.rgba[i + c] * 0.4);
  assert.deepEqual(fit(brown).fit.coefficients, fit(black).fit.coefficients);
});
test('walls face outwards and use separate UVs without stretching roof texture', () => {
  const f = fixture(),
    result = fit(f);
  assert.ok(result.fit);
  const geometry = buildFittedRoof(outline, result.fit, 100, 100, 0.4, 2);
  const mesh = new THREE.Mesh(geometry, [new THREE.MeshBasicMaterial(), new THREE.MeshBasicMaterial()]);
  mesh.updateMatrixWorld();
  const ray = new THREE.Raycaster(new THREE.Vector3(0, 3, -60), new THREE.Vector3(0, 0, 1));
  const hit = ray.intersectObject(mesh)[0];
  assert.ok(hit);
  assert.equal(hit.face.materialIndex, 1);
  assert.ok(hit.face.normal.z < 0);
  assert.ok(geometry.attributes.color);
  geometry.dispose();
  mesh.material.forEach((m) => m.dispose());
});
