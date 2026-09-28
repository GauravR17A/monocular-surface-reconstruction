import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { advanceMetricBlend, metricPixelScale, metricGridSpec, niceGridStep, createMetricGridMaterial, createMetricGridFloor } from '../app/metric-grid.ts';

const stats = { width: 1025, height: 513, minimum: 0, maximum: 40, gsdX: .5, gsdY: .5, units: 'm' };

test('metric grid uses known horizontal scale and bounded nice intervals', () => {
  const spec = metricGridSpec(stats, 112, 56);
  assert.equal(spec.horizontalUnit, 'm');
  assert.equal(spec.extentX, 512);
  assert.equal(spec.extentY, 256);
  assert.equal(spec.spacing, 50);
  assert.equal(spec.heightInterval, 5);
  assert.equal(spec.worldStepX, spec.worldStepZ);
});

test('missing or invalid horizontal scale is pixels, even with metric heights', () => {
  for (const gsdX of [null, 0, -1, NaN, Infinity]) {
    const spec = metricGridSpec({ ...stats, gsdX }, 112, 56);
    assert.equal(spec.horizontalUnit, 'px');
    assert.equal(spec.heightUnit, 'm');
    assert.equal(spec.extentX, 1024);
    assert.equal(spec.extentY, 512);
  }
});

test('relative heights never acquire metre labels from georeferencing', () => {
  const spec = metricGridSpec({ ...stats, minimum: 0, maximum: 1, units: 'relative' }, 112, 56);
  assert.equal(spec.heightUnit, 'relative');
  assert.equal(spec.heightInterval, .1);
});

test('degrees and unreferenced native pixel transforms cannot masquerade as metres', () => {
  assert.equal(metricPixelScale(NaN, 1), null);
  assert.equal(metricPixelScale(NaN, .00001, 2, 9001), null);
  assert.equal(metricPixelScale(NaN, 1, 1), null);
  assert.equal(metricPixelScale(.7, .00001, 2), .7);
  assert.equal(metricPixelScale(NaN, 2, 1, 9001), 2);
  assert.equal(metricPixelScale(NaN, 2, 1, 9002), .6096);
});

test('flat surfaces and huge terrain ranges have finite intervals and bounded grid', () => {
  for (const maximum of [0, .001, 3200, 1e6]) {
    const spec = metricGridSpec({ ...stats, maximum }, 112, 56);
    assert.ok(Number.isFinite(spec.heightInterval) && spec.heightInterval > 0);
    const floor = createMetricGridFloor(spec);
    assert.ok(floor.geometry.attributes.position.count <= 68);
    floor.geometry.dispose(); floor.material.dispose();
  }
  assert.equal(niceGridStep(NaN), 1);
});

test('grid material has no photos, updates height scale without rebuilding or mutating input', () => {
  const spec = metricGridSpec(stats, 112, 56), before = structuredClone(spec);
  const grid = createMetricGridMaterial(spec, .3);
  assert.equal(grid.material.map, null);
  assert.equal(grid.material.bumpMap, null);
  assert.equal(grid.material.vertexColors, false);
  const shader = { uniforms: {}, vertexShader: '#include <begin_vertex>', fragmentShader: '#include <color_fragment>' };
  grid.material.onBeforeCompile(shader);
  assert.ok(shader.fragmentShader.includes('dwLine'));
  assert.ok(shader.fragmentShader.includes('mix(diffuseColor.rgb, dwGridColour, dwMetricBlend)'));
  grid.setExaggeration(2);
  assert.equal(shader.uniforms.dwHeightScale.value, .6);
  assert.deepEqual(spec, before);
  grid.material.dispose();
});

test('absolute terrain and building solids can share a ramp without mixing their height origins', () => {
  const spec = metricGridSpec({ ...stats, minimum: 1500, maximum: 3200 }, 112, 56);
  const surface = createMetricGridMaterial(spec, .01, 1500);
  const buildings = createMetricGridMaterial(spec, .01, 0);
  const shaders = [surface, buildings].map(grid => {
    const shader = { uniforms: {}, vertexShader: '#include <begin_vertex>', fragmentShader: '#include <color_fragment>' };
    grid.material.onBeforeCompile(shader);
    return shader;
  });
  assert.equal(shaders[0].uniforms.dwHeightOrigin.value, 1500);
  assert.equal(shaders[1].uniforms.dwHeightOrigin.value, 0);
  assert.equal(shaders[0].uniforms.dwHeightMinimum.value, shaders[1].uniforms.dwHeightMinimum.value);
  assert.equal(spec.heightInterval, 200);
  surface.material.dispose(); buildings.material.dispose();
});

test('transition is bounded, reversible, frame-rate independent and respects reduced motion', () => {
  assert.equal(advanceMetricBlend(0, 1, .325), .5);
  assert.equal(advanceMetricBlend(.5, 0, .325), 0);
  assert.equal(advanceMetricBlend(.7, 1, 2), 1);
  assert.equal(advanceMetricBlend(.2, 0, 2), 0);
  assert.equal(advanceMetricBlend(0, 1, .001, true), 1);
  assert.equal(advanceMetricBlend(.8, 0, .001, true), 0);
  assert.equal(advanceMetricBlend(.2, 1, -1), .2);
  for (const fps of [30, 60, 144]) {
    let blend = 0;
    for (let frame = 0; frame < fps; frame++) blend = advanceMetricBlend(blend, 1, 1 / fps);
    assert.equal(blend, 1);
  }
});

test('optical material and facade detail survive a round trip without geometry changes', () => {
  const photo = new THREE.Texture(), bump = new THREE.Texture();
  const material = new THREE.MeshStandardMaterial({ map: photo, bumpMap: bump, bumpScale: .018 });
  const controller = createMetricGridMaterial(metricGridSpec(stats, 112, 56), .3, 0, material);
  assert.equal(controller.material, material);
  const shader = { uniforms: {}, vertexShader: '#include <begin_vertex>', fragmentShader: '#include <color_fragment>\n#include <roughnessmap_fragment>' };
  material.onBeforeCompile(shader);
  for (const blend of [0, .4, 1, .6, 0]) {
    controller.setBlend(blend);
    assert.equal(shader.uniforms.dwMetricBlend.value, blend);
    assert.equal(material.map, photo);
    assert.equal(material.bumpMap, bump);
    assert.equal(material.bumpScale, .018 * (1 - blend));
  }
  material.dispose(); photo.dispose(); bump.dispose();
});

test('class highlight uses nearest raster classes or solid-building class, and resets cleanly', () => {
  const c = createMetricGridMaterial(metricGridSpec(stats, 112, 56), .3, 0, undefined, 1);
  const shader = { uniforms: {}, vertexShader: '#include <begin_vertex>', fragmentShader: '#include <color_fragment>\n#include <emissivemap_fragment>' };
  c.material.onBeforeCompile(shader);
  const map = new THREE.DataTexture(new Uint8Array([0,1,2,3]),2,2,THREE.RedFormat);
  c.setClassFocus(map,1,1);
  assert.equal(shader.uniforms.dwClassMap.value,map); assert.equal(shader.uniforms.dwObjectClass.value,1);
  assert.equal(shader.uniforms.dwFocusAmount.value,1);
  assert.ok(shader.fragmentShader.includes('totalEmissiveRadiance += dwAccent'));
  c.setClassFocus(map,-1,1); assert.equal(shader.uniforms.dwFocusAmount.value,0);
  map.dispose(); c.material.dispose();
});

