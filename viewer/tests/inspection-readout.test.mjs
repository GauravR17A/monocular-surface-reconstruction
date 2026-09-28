import test from 'node:test';
import assert from 'node:assert/strict';
import { inspectionReadout } from '../app/inspection-readout.ts';
import { sixClassOverlayVisible, withSixClass } from '../app/six-class-layer.ts';

const base = { kind: 'surface', rawValue: 1.25, displayValue: 3.5, buildingProbability: null,
  vegetationProbability: null, confidence: null, rgbVegetation: false, slopeDegrees: 12, targetKey: 'raster:surface:known' };
const layer = { labels: new Uint8Array([0,1,2,3,4,5,255]), width: 7, height: 1, selected: null, visible: false };

for (const [code, label, kind] of [[0,'Ground','surface'],[1,'Buildings','building'],[2,'Water','surface'],[3,'Roads','surface'],[4,'Low vegetation','vegetation'],[5,'Trees','vegetation']]) {
  test(`${label} remains inspectable with colors off; heights unchanged`, () => {
    const original = { ...base, kind };
    const data = withSixClass(original, layer, (code + .5)/7, .5);
    const readout = inspectionReadout(data, 'm', 'ndsm');
    assert.equal(sixClassOverlayVisible(layer), false);
    assert.equal(readout.label, label);
    assert.equal(readout.value, original.rawValue);
    assert.equal(data.displayValue, original.displayValue);
    assert.equal(readout.disagreement, false);
    assert.equal(readout.classSource, 'RGB six-class V3');
    assert.equal(original.experimentalClass, undefined);
  });
}
test('class changes reset dwell identity but motion within a category does not', () => {
  assert.equal(withSixClass(base, layer, .61, 0).targetKey, withSixClass(base, layer, .69, 0).targetKey);
  assert.notEqual(withSixClass(base, layer, .1, 0).targetKey, withSixClass(base, layer, .5, 0).targetKey);
});
test('water is neither depth nor a forced zero', () => {
  const result = inspectionReadout(withSixClass(base, layer, 2.5/7, 0), 'm', 'ndsm');
  assert.equal(result.value, 1.25);
  assert.match(result.notes.join(' '), /never water depth/);
});
test('tree label cannot silently rename a protected building object height', () => {
  const data = withSixClass({ ...base, kind:'building', objectHeight: true }, layer, 5.5/7, 0);
  const result = inspectionReadout(data, 'm', 'ndsm');
  assert.equal(result.label, 'Trees');
  assert.equal(result.measurement, 'Building height above ground');
  assert.equal(result.disagreement, true);
  assert.match(result.notes.join(' '), /Models disagree/);
});
test('absolute elevation is not height above ground; building summary is not ASL', () => {
  const terrain = inspectionReadout({ ...base, rawValue:1500 }, 'm', 'dsm');
  assert.equal(terrain.measurement, 'Surface elevation (datum)');
  assert.match(terrain.notes.join(' '), /separate ground reference/);
  assert.equal(inspectionReadout({ ...base, kind:'building', objectHeight:true }, 'm', 'dsm').measurement, 'Building height above ground');
});
test('relative and unavailable values never become metres or zero', () => {
  for (const [units, product] of [['relative','ndsm'], ['m','relative']]) {
    assert.equal(inspectionReadout(base, units, product).units, 'relative');
  }
  assert.equal(inspectionReadout({ ...base, rawValue:NaN }, 'm', 'ndsm').value, null);
});
test('ignored label is unknown; no classifier retains legacy labels', () => {
  assert.equal(inspectionReadout(withSixClass(base, layer, 1, 0), 'm', 'ndsm').label, 'Unclassified');
  assert.equal(inspectionReadout(withSixClass(base, null, 1, 0), 'm', 'ndsm').label, 'Surface');
  assert.equal(sixClassOverlayVisible(null), false);
  assert.equal(sixClassOverlayVisible({ ...layer, visible:true }), true);
});

for (const status of ['waiting','loading','error','unsupported']) {
  test(`${status}: inspector never silently substitutes the old building label`, () => {
    const original = {...base, kind:'building', objectHeight:true, rawValue:12.25};
    const result = inspectionReadout(withSixClass(original, null, .5, .5, status), 'm', 'ndsm');
    assert.equal(result.label, ['waiting','loading'].includes(status) ? 'Identifying…' : 'Classification unavailable');
    assert.equal(result.value, 12.25);
    assert.equal(result.measurement, 'Building height above ground');
    assert.equal(result.disagreement, false);
  });
}
