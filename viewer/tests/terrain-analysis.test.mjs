import { test } from 'node:test';
import assert from 'node:assert/strict';
import { terrainDimensions, automaticTerrainExaggeration, interpolateSurface, surfaceSlope, sampleTerrainProfile, profileCsv } from '../app/terrain-analysis.ts';
const raster = (values, width, height, stats = {}) => ({ values: new Float32Array(values), stats: { width, height, units: 'm', gsdX: 10, gsdY: 10, minimum: 100, maximum: 200, ...stats } });

test('hilly terrain keeps metre proportions instead of normalizing relief to 12 world units', () => {
  const result = terrainDimensions(raster([], 768, 768, { minimum: 1728, maximum: 3821 }).stats);
  assert.equal(result.metric, true);
  assert.ok(Math.abs(result.verticalScale - 112 / 7670) < 1e-12);
  assert.ok((3821 - 1728) * result.verticalScale > 30);
  assert.equal(result.verticalScale, terrainDimensions(raster([], 768, 768, { maximum: 1800 }).stats).verticalScale);
});
test('non-square pixels preserve the same scale in all three axes', () => {
  const result = terrainDimensions(raster([], 11, 21, { gsdX: 2, gsdY: 3 }).stats, 100);
  assert.equal(result.depth, 300); assert.equal(result.verticalScale, 5);
  assert.equal(terrainDimensions(raster([], 11, 21, { gsdX: null }).stats).metric, false);
  assert.equal(terrainDimensions(raster([], 11, 21, { units: 'relative' }).stats).metric, false);
});

test('urban display meshes retain the original bounded scale shared with building solids', () => {
  const display = raster([], 512, 512, { minimum: 0, maximum: 8.6, gsdX: null, gsdY: null }).stats;
  const analysis = { ...display, maximum: 42 };
  assert.equal(terrainDimensions(display, 112, analysis).verticalScale, 12 / 42);
  assert.equal(terrainDimensions(display).verticalScale, .36);
  assert.equal(terrainDimensions({ ...display, gsdX: 10, gsdY: 10 },112,analysis).verticalScale,112/5110);
});
test('bilinear interpolation recovers a plane and does not bridge nodata', () => {
  const plane = raster([0, 20, 30, 50], 2, 2);
  assert.equal(interpolateSurface(plane, .5, .5), 25);
  plane.values[3] = NaN;
  assert.equal(interpolateSurface(plane, .5, .5), null);
  assert.equal(interpolateSurface(plane, 0, 0), 0);
  assert.equal(interpolateSurface(plane, -.1, .5), null);
});
test('a planar profile has known physical distance, elevations, slope and signed change', () => {
  const plane = raster([100,110,120,100,110,120,100,110,120], 3, 3);
  const profile = sampleTerrainProfile(plane, raster(Array(9).fill(95),3,3), {u:0,v:.5}, {u:1,v:.5});
  assert.equal(profile.distance,20); assert.equal(profile.distanceUnit,'m');
  assert.equal(profile.delta,20); assert.equal(profile.ascent,20); assert.equal(profile.descent,0);
  assert.equal(profile.samples[1].surface,110); assert.equal(profile.samples[1].ground,95);
  assert.equal(surfaceSlope(plane,.5,.5),45);
  assert.equal(sampleTerrainProfile(plane,null,{u:1,v:.5},{u:0,v:.5}).delta,-20);
  assert.match(profileCsv(profile),/distance_m,surface_m,terrain_reference_m/);
});
test('missing data creates a gap and never a spurious elevation gain across it', () => {
  const profile = sampleTerrainProfile(raster([100,NaN,140],3,1),null,{u:0,v:0},{u:1,v:0});
  assert.equal(profile.gaps,true); assert.equal(profile.ascent,0);
  assert.equal(profile.samples[1].surface,null); assert.match(profileCsv(profile),/10\.000,,,/);
});
test('relative values and unknown horizontal scale cannot become metre claims', () => {
  const plane = raster([0,.5,1],3,1,{units:'relative',gsdX:null,gsdY:null});
  const profile = sampleTerrainProfile(plane,null,{u:0,v:0},{u:1,v:0});
  assert.equal(profile.heightUnit,'relative'); assert.equal(profile.distanceUnit,'px');
  assert.equal(surfaceSlope(plane,.5,0),null); assert.match(profileCsv(profile),/surface_relative/);
});

test('automatic terrain view uses gentle consistent relief, without turning plains into mountains', () => {
  const plain = raster(Array.from({length:1000}, (_,i)=>230+i/1000*23), 1000, 1000);
  const original = plain.values.slice(), dimensions = terrainDimensions(plain.stats);
  assert.equal(automaticTerrainExaggeration(plain),1.5);
  assert.deepEqual(plain.values,original);
  assert.deepEqual(terrainDimensions(plain.stats),dimensions);
  const mountain = raster(Array.from({length:1000},(_,i)=>1100+i*6),1000,1000,{gsdX:33,gsdY:33});
  assert.equal(automaticTerrainExaggeration(mountain),1.5);
});

test('automatic relief never invents variation in flat or unavailable elevations', () => {
  assert.equal(automaticTerrainExaggeration(raster(Array(100).fill(200),10,10)),1);
  assert.equal(automaticTerrainExaggeration(raster([NaN,NaN],2,1)),1);
  assert.equal(automaticTerrainExaggeration(raster([0,1],2,1,{units:'relative'})),1);
  assert.equal(automaticTerrainExaggeration(raster([100,101],2,1,{gsdX:null})),1);
});
