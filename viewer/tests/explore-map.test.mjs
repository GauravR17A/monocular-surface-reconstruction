import test from 'node:test';
import assert from 'node:assert/strict';
import { mapFrame, mapSpace, localMapWindow, mapScreenPoint, mapDistanceLabel } from '../app/explore-map-math.ts';
const near = (a,b) => assert.ok(Math.abs(a-b) < 1e-8, `${a} != ${b}`);
const scene = { width: 1001, height: 501, gsdX: 2, gsdY: 2 };
test('Map geometry uses both valid horizontal scales, never height units', () => {
  assert.deepEqual(mapSpace(scene), { width: 2000, height: 1000, unit: 'm', resolution: 2 });
  assert.deepEqual(mapSpace({ ...scene, gsdX: null }), { width: 1000, height: 500, unit: 'px', resolution: 1 });
  assert.equal(mapSpace({ ...scene, gsdY: NaN }).unit, 'px');
  assert.equal(mapSpace({ ...scene, gsdY: -1 }).unit, 'px');
});
test('World-plane position is registered with image corners and non-square extents', () => {
  const corner = mapFrame(-56, -28, 112, 56, 0, 'fly');
  assert.equal(corner.u, 0); assert.equal(corner.v, 0);
  const other = mapFrame(56, 28, 112, 56, 0, 'walk');
  assert.equal(other.u, 1); assert.equal(other.v, 1);
  const middle = mapFrame(0, 0, 112, 56, 0, 'fly');
  assert.equal(middle.u, .5); assert.equal(middle.v, .5);
});
test('Yaw follows actual view direction, not an assumed geographic north', () => {
  const space = mapSpace(scene);
  near(localMapWindow(space, mapFrame(0,0,112,56,0,'fly')).heading, 0);
  near(localMapWindow(space, mapFrame(0,0,112,56,-Math.PI/2,'fly')).heading, Math.PI/2);
  near(localMapWindow(space, mapFrame(0,0,112,56,Math.PI/2,'fly')).heading, -Math.PI/2);
});
test('Local tracking chooses useful metre spans and expands at speed, never beyond footprint context', () => {
  const space = mapSpace(scene), pose = mapFrame(0,0,112,56,0,'fly');
  assert.equal(localMapWindow(space,pose).span,420);
  assert.equal(localMapWindow(space,{...pose,mode:'walk'}).span,180);
  assert.equal(localMapWindow(space,pose,200).span,800);
  near(localMapWindow(space,pose,1e8).span,Math.hypot(space.width,space.height)*1.08);
  const small = mapSpace({width:51,height:51,gsdX:1,gsdY:1});
  assert.ok(localMapWindow(small,pose).span < 80);
});
test('Coarse metric rasters keep context; unreferenced images use proportional pixels', () => {
  const pose = mapFrame(0,0,112,112,0,'fly');
  assert.equal(localMapWindow(mapSpace({width:1001,height:1001,gsdX:30,gsdY:30}),pose).span,720);
  const space = mapSpace({width:1025,height:1025,gsdX:null,gsdY:null});
  near(localMapWindow(space,pose).span,460.8);
  assert.match(mapDistanceLabel(localMapWindow(space,pose).span,space.unit), /px$/);
});
test('Heading-up marker keeps a little more forward context while camera turns', () => {
  const space = mapSpace(scene);
  for (const yaw of [0,.7,Math.PI,-2]) {
    const pose = mapFrame(0,0,112,56,yaw,'fly'), window = localMapWindow(space,pose);
    const point = mapScreenPoint(space,pose,window,184,184,false);
    near(point.x,92); near(point.y,92 + 184 * .12);
  }
});
test('Outside-image positions are explicit; overview marker pins to footprint edge', () => {
  const space = mapSpace(scene), pose = mapFrame(200,-70,112,56,0,'fly'), window = localMapWindow(space,pose);
  assert.equal(window.outside,true);
  assert.ok(window.x >= 0 && window.x <= space.width && window.y >= 0 && window.y <= space.height);
  for (const full of [false,true]) {
    const point = mapScreenPoint(space,pose,window,500,400,full);
    assert.equal(point.edge,true); assert.ok(point.x >= 13 && point.x <= 487 && point.y >= 13 && point.y <= 387);
  }
  assert.ok(pose.u > 1); // true position is retained, not rewritten to an in-bounds location
});
test('Full map fits rectangular extent without cropping or rotating the image', () => {
  const space = mapSpace(scene), pose = mapFrame(0,0,112,56,.7,'fly');
  const point = mapScreenPoint(space,pose,localMapWindow(space,pose),800,600,true);
  near(point.x,400); near(point.y,300); assert.equal(point.angle,0);
  assert.ok(point.scale * space.width <= 800 && point.scale * space.height <= 600);
});
test('Distance labels do not invent metres for pixel-only scenes', () => {
  assert.equal(mapDistanceLabel(1500,'m'),'1.5 km');
  assert.equal(mapDistanceLabel(1500,'px'),'1,500 px');
});
