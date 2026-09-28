import { test } from 'node:test';
import assert from 'node:assert/strict';
import { querySurface, rasterValue } from '../app/surface-query.ts';
const raster=(values,width=2,height=2,extra={})=>({ values: new Float32Array(values),stats:{width,height,units:'m',...extra}});
const base=()=>({raw:raster([0,20,2,4]),display:raster([0,99,8,9]),building:null,vegetation:null,semantic:null,confidence:null,rgba:null,textureWidth:2,textureHeight:2});

test('absolute elevation and difference above an aligned terrain reference stay separate',()=>{
  const result=querySurface({...base(),raw:raster([2030,2040,2050,2060]),ground:raster([2000,2000,2000,NaN])},1,0);
  assert.equal(result.rawValue,2040); assert.equal(result.terrainElevation,2000); assert.equal(result.aboveGroundHeight,40);
  const missing=querySurface({...base(),ground:raster([2000,2000,2000,NaN])},1,1);
  assert.equal(missing.aboveGroundHeight,null);
});
test('exact roof pixel instead of a mesh triangle average; raw remains unexaggerated',()=>{
  const result=querySurface({...base(),semantic:raster([0,1,0,2])},1,0);
  assert.equal(result.rawValue,20); assert.equal(result.displayValue,99); assert.equal(result.kind,'building');
});
test('moving within a raster class preserves dwell identity while values update',()=>{
  const query={...base(),semantic:raster([2,2,0,1])};
  const first=querySurface(query,0,0), next=querySurface(query,1,0);
  assert.equal(first.targetKey,next.targetKey);
  assert.notEqual(first.rawValue,next.rawValue);
  assert.notEqual(first.targetKey,querySurface(query,0,1).targetKey);
  assert.notEqual(first.targetKey,querySurface(query,1,1).targetKey);
});
test('NoData and out-of-range coordinates never become a zero-height claim',()=>{
  assert.equal(querySurface({...base(),raw:raster([NaN,1,2,3])},0,0),null);
  assert.equal(rasterValue(base().raw,NaN,0),null);
  assert.equal(rasterValue(base().raw,1.1,0),null);
});
test('resolutions share normalized coordinates and image Y is not inverted',()=>{
  const query={...base(),semantic:raster([2],1,1)};
  assert.equal(querySurface(query,0,1).rawValue,2);
  assert.equal(querySurface(query,0,1).kind,'vegetation');
});
test('unknown labels stay unknown even over green RGB; legacy greenery is disclosed',()=>{
  const rgba=new Uint8ClampedArray([20,150,20,255]);
  const query={...base(),rgba,textureWidth:1,textureHeight:1};
  assert.equal(querySurface(query,0,0).kind,'vegetation');
  const result=querySurface({...query,semantic:raster([255],1,1)},0,0);
  assert.equal(result.classBasis,'Unclassified pixel'); assert.equal(result.kind,'surface');
});
test('slope uses metric spacing and is unavailable for relative data',()=>{
  const raw=raster([0,1,2,0,1,2,0,1,2],3,3,{gsdX:1,gsdY:1});
  assert.equal(querySurface({...base(),raw},.5,.5).slopeDegrees,45);
  raw.stats.units='relative';
  assert.equal(querySurface({...base(),raw},.5,.5).slopeDegrees,null);
});
