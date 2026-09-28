import { test } from 'node:test';
import assert from 'node:assert/strict';
import { hasTerrainGeoreference, terrainCalibrationSource } from '../app/terrain-import.ts';
test('any projected or geographic TIFF with a spatial grid can receive terrain', () => {
  for (const code of [3857,32643,32644,32755,26918]) assert.equal(hasTerrainGeoreference({GTModelTypeGeoKey:1,ProjectedCSTypeGeoKey:code},[10,20,110,220]),true);
  assert.equal(hasTerrainGeoreference({GTModelTypeGeoKey:2,GeographicTypeGeoKey:4326},[79,30,80,31]),true);
});
test('unreferenced, geocentric and invalid grids never trigger a terrain lookup', () => {
  for (const [keys,bounds] of [[null,null],[{},[0,0,100,100]],[{GTModelTypeGeoKey:1},[0,0,100,100]],[{GTModelTypeGeoKey:3,GeographicTypeGeoKey:4326},[0,0,100,100]],[{GTModelTypeGeoKey:2,GeographicTypeGeoKey:4326},[0,0,NaN,100]]]) assert.equal(hasTerrainGeoreference(keys,bounds),false);
});
test('automatic terrain applies to every georeferenced import while manual evidence and opt-out win', () => {
  const defaults={georeferenced:true,automatic:true,dem:false,gcps:false};
  assert.equal(terrainCalibrationSource(defaults),'public');
  assert.equal(terrainCalibrationSource({...defaults,dem:true}),'dem');
  assert.equal(terrainCalibrationSource({...defaults,gcps:true}),'gcps');
  assert.equal(terrainCalibrationSource({...defaults,automatic:false}),'none');
  assert.equal(terrainCalibrationSource({...defaults,georeferenced:false}),'none');
});
