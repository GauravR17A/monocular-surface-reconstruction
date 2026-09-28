import test from 'node:test';
import assert from 'node:assert/strict';
import { classLayerPixels, classificationSource, sixClassAt } from '../app/six-class-layer.ts';
import { createMetricGridMaterial, metricGridSpec } from '../app/metric-grid.ts';

test('six-class colors retain top-left orientation and ignored transparency', () => {
  const layer = { labels: new Uint8Array([0, 1, 2, 3, 4, 255]), width: 3, height: 2, selected: null };
  const copy = layer.labels.slice();
  const pixels = classLayerPixels(layer);
  assert.equal(pixels[23], 0);
  assert.equal(sixClassAt(layer, 0, 0), 'Ground');
  assert.equal(sixClassAt(layer, 1, 0), 'Water');
  assert.equal(sixClassAt(layer, 0, 1), 'Roads');
  assert.equal(sixClassAt(layer, 1, 1), null);
  const selected = classLayerPixels({ ...layer, selected: 2 });
  assert.deepEqual(Array.from(selected.slice(0, 3)), [20, 32, 39]);
  assert.deepEqual(selected.slice(8, 12), pixels.slice(8, 12));
  assert.deepEqual(layer.labels, copy);
});

test('only current app jobs and explicit bundled demos can be classified', () => {
  assert.deepEqual(classificationSource(`http://127.0.0.1:8000/results/${'a'.repeat(32)}/texture.jpg`, null), { job_id: 'a'.repeat(32) });
  assert.deepEqual(classificationSource('/demo/copenhagen_rgb.jpg', 'urban'), { demo_id: 'urban' });
  assert.equal(classificationSource('blob:http://localhost/new-texture', 'urban'), null);
  assert.equal(classificationSource('https://foreign/texture.jpg', null), null);
  assert.equal(classificationSource('/demo/forest_validation/satellite_rgb.jpg', 'urban'), null);
  assert.deepEqual(classificationSource('/demo/landscapes/hilly/texture.jpg?v=2', 'hilly'), {demo_id:'hilly'});
});

test('preview changes only fragment shading; no vertex displacement or height input', () => {
  const spec = metricGridSpec({ width: 4, height: 4, minimum: 0, maximum: 20, gsdX: null, gsdY: null, units: 'm' }, 112, 112);
  const controller = createMetricGridMaterial(spec, .3);
  const shader = { uniforms: {}, vertexShader: '#include <begin_vertex>', fragmentShader: '#include <color_fragment>\n#include <emissivemap_fragment>\n#include <roughnessmap_fragment>' };
  controller.material.onBeforeCompile(shader, null);
  const vertex = shader.vertexShader;
  controller.setSixClassOverlay({ fake: true }, .82);
  assert.equal(shader.uniforms.dwSixBlend.value, .82);
  assert.equal(shader.vertexShader, vertex);
  assert.ok(!shader.vertexShader.includes('dwSix'));
  controller.setSixClassOverlay(null, 1);
  assert.equal(shader.uniforms.dwSixBlend.value, 0);
  controller.material.dispose();
});
