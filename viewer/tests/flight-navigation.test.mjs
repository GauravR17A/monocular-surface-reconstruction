import { test } from 'node:test';
import assert from 'node:assert/strict';
import { moveFlight, gridSurfaceHeight, touchesFootprint } from '../app/flight-navigation.ts';
const box = { footprint: [[0,0],[2,0],[2,2],[0,2]], top: 5 };
test('boost cannot tunnel through even a narrow wall', () => {
  const p = moveFlight({x:-3,y:2,z:1},{x:10,y:0,z:0},()=>0,[box],1);
  assert.ok(p.x < -0.6); assert.equal(p.y,2);
});
test('can fly above roof but cannot descend through it', () => {
  const p = moveFlight({x:1,y:8,z:1},{x:0,y:-15,z:0},()=>0,[box],1);
  assert.ok(p.y >= 5.65);
  assert.ok(moveFlight({x:-3,y:8,z:1},{x:8,y:0,z:0},()=>0,[box],1).x>2);
});
test('exaggeration and entry inside a building use displayed geometry', () => {
  assert.ok(moveFlight({x:1,y:1,z:1},{x:0,y:0,z:0},()=>0,[box],2).y>=10.65);
});
test('gentle terrain following and steep cliff blocking', () => {
  assert.ok(moveFlight({x:0,y:1,z:0},{x:5,y:0,z:0},x=>Math.max(0,x*.3),[],1).x>4.9);
  assert.ok(moveFlight({x:-2,y:1,z:0},{x:6,y:0,z:0},x=>x>=0?10:0,[],1).x<0);
});
test('sliding, polygon boundaries and triangle interpolation', () => {
  const p=moveFlight({x:-1,y:2,z:.5},{x:2,y:0,z:1},()=>0,[box],1);
  assert.ok(p.x<0); assert.ok(p.z>1);
  assert.ok(touchesFootprint(-.1,1,box.footprint,.2));
  assert.equal(gridSurfaceHeight(new Float32Array([0,0,0,10]),2,2,2,2,0,0),0);
  assert.equal(gridSurfaceHeight(new Float32Array([0,0,0,10]),2,2,2,2,.5,.5),5);
});
