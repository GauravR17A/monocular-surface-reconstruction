import { test } from 'node:test';
import assert from 'node:assert/strict';
import { WALK_EYE_METRES, WALK_SPEED_MPS, WALK_JUMP_HEIGHT_METRES, createWalkMotion, stepWalk, walkGround, findWalkSpawn, walkIntent, moveWalk, walkHeadMotion } from '../app/walk-navigation.ts';
const world = (extra = {}) => ({ width: 40, depth: 40, scaleX: 1, scaleY: 1, scaleZ: 1, ground: () => 0, blocked: () => false, ...extra });
test('eye height is human-scale; horizontal input is yaw-based and diagonal speed is bounded', () => {
  assert.equal(WALK_EYE_METRES, 1.8);
  assert.equal(WALK_SPEED_MPS, 4.3);
  assert.deepEqual(walkIntent(0, 1, 0), { x: 0, z: -1 });
  assert.ok(Math.abs(walkIntent(Math.PI / 2, 1, 0).x + 1) < 1e-9);
  assert.ok(Math.abs(Math.hypot(...Object.values(walkIntent(0, 1, 1))) - 1) < 1e-9);
});
test('walker follows gentle hills both up and down without flight', () => {
  const w = world({ ground: x => x * 0.3 });
  const up = moveWalk({ x: 0, y: 50, z: 0 }, { x: 3, z: 0 }, w);
  assert.ok(Math.abs(up.position.y - .9) < 1e-8);
  const down = moveWalk(up.position, { x: -4, z: 0 }, w);
  assert.ok(Math.abs(down.position.y + .3) < 1e-8);
});
test('steps, cliffs and steep drops are blocked in both directions', () => {
  for (const high of [6, -6]) {
    const p = moveWalk({ x: -1, y: 0, z: 0 }, { x: 4, z: 0 }, world({ ground: x => x > 0 ? high : 0 }));
    assert.ok(p.position.x < 0); assert.equal(p.position.y, 0); assert.ok(p.blocked);
  }
});
test('sprint cannot tunnel through a thin building; walker slides alongside walls', () => {
  const w = world({ blocked: (x, z, r) => x + r >= 0 && x - r <= .05 && z < 10 });
  const p = moveWalk({ x: -1, y: 0, z: 0 }, { x: 4, z: 1 }, w);
  assert.ok(p.position.x < -.27); assert.ok(p.position.z > .9); assert.ok(p.blocked);
});
test('NoData, forbidden surfaces and map borders are not zero-height ground', () => {
  assert.equal(walkGround(world({ ground: () => null }), 0, 0), null);
  assert.equal(walkGround(world({ ground: () => NaN }), 0, 0), null);
  assert.equal(walkGround(world(), 19.9, 0), null);
  assert.equal(findWalkSpawn(world({ blocked: () => true }), { x: 0, z: 0 }), null);
  assert.ok(moveWalk({ x: 19, y: 0, z: 0 }, { x: 3, z: 0 }, world()).position.x < 19.72);
});
test('spawn searches for safe ground instead of a rooftop', () => {
  const w = world({ blocked: (x, z) => Math.abs(x) < 3 && Math.abs(z) < 3 });
  const p = findWalkSpawn(w, { x: 0, z: 0 });
  assert.ok(p); assert.ok(Math.abs(p.x) >= 3 || Math.abs(p.z) >= 3); assert.equal(p.y, 0);
});
test('horizontal pixel scales and vertical scale remain independent', () => {
  const w = world({ scaleX: .2, scaleZ: .4, scaleY: .01, ground: x => x / .2 * .2 * .01 });
  const p = moveWalk({ x: 0, y: 0, z: 0 }, { x: 1, z: 1 }, w);
  assert.ok(Math.abs(p.position.x - .2) < 1e-9); assert.ok(Math.abs(p.position.z - .4) < 1e-9);
  assert.ok(Math.abs(p.position.y - .002) < 1e-9);
});
test('head motion is subtle, stops at rest and respects reduced motion', () => {
  assert.deepEqual(walkHeadMotion(1, 1, true), { lift: 0, roll: 0, sway: 0, pitch: 0 });
  assert.equal(walkHeadMotion(1, 0, false).lift, 0);
  for (let d = 0; d < 3; d += .01) {
    const h = walkHeadMotion(d, 1, false);
    assert.ok(h.lift <= 0 && h.lift >= -.045); assert.ok(Math.abs(h.roll) <= .005);
    assert.ok(Math.abs(h.sway) <= .025); assert.ok(Math.abs(h.pitch) <= .004);
  }
});
test('invalid scale fails safely and enormous deltas are bounded', () => {
  assert.equal(findWalkSpawn(world({ scaleX: 0 }), { x: 0, z: 0 }), null);
  assert.equal(walkGround(world({ scaleY: Infinity }), 0, 0), null);
  assert.ok(moveWalk({ x: 0, y: 0, z: 0 }, { x: 1e8, z: 0 }, world()).distance < 5.01);
});

test('Space creates one finite jump arc and returns to standing height', () => {
  let state = createWalkMotion({ x: 0, y: 0, z: 0 }), maximum = 0;
  for (let frame = 0; frame < 150; frame++) {
    state = stepWalk(state, { x: 0, z: 0 }, 1 / 120, world(), frame === 0);
    maximum = Math.max(maximum, state.position.y);
    assert.ok(state.position.y >= 0);
  }
  assert.ok(Math.abs(maximum - WALK_JUMP_HEIGHT_METRES) < .005);
  assert.equal(state.grounded, true); assert.equal(state.position.y, 0); assert.equal(state.velocityY, 0);
});

test('airborne Space presses cannot double-jump or increase the apex', () => {
  let state = createWalkMotion({ x: 0, y: 0, z: 0 }), maximum = 0;
  for (let frame = 0; frame < 70; frame++) {
    state = stepWalk(state, { x: 0, z: 0 }, 1 / 120, world(), true);
    maximum = Math.max(maximum, state.position.y);
  }
  assert.ok(maximum <= WALK_JUMP_HEIGHT_METRES + .001);
});

test('jump clears a low terrain obstacle which grounded walking cannot cross', () => {
  const w = world({ ground: x => x > 0 && x < .4 ? .5 : 0 });
  assert.ok(moveWalk({ x: -1, y: 0, z: 0 }, { x: 3, z: 0 }, w).position.x < 0);
  let state = createWalkMotion({ x: -1, y: 0, z: 0 });
  for (let frame = 0; frame < 100; frame++) state = stepWalk(state, { x: 4.3, z: 0 }, 1 / 120, w, frame === 0);
  assert.ok(state.position.x > 2); assert.equal(state.position.y, 0); assert.ok(state.grounded);
});

test('jump lands on a reachable raised surface and remains able to walk', () => {
  const w = world({ ground: x => x > 0 ? .5 : 0 });
  let state = createWalkMotion({ x: -1, y: 0, z: 0 });
  for (let frame = 0; frame < 120; frame++) state = stepWalk(state, { x: 4.3, z: 0 }, 1 / 120, w, frame === 0);
  assert.ok(state.position.x > 2); assert.equal(state.position.y, .5); assert.ok(state.grounded);
});

test('jump/sprint cannot pass through a building, high wall or missing ground', () => {
  for (const w of [world({ blocked: (x, z, r) => x + r >= 0 && x-r < .05 }), world({ ground: x => x > 0 ? 8 : 0 }), world({ ground: x => x > 0 ? null : 0 })]) {
    let state = createWalkMotion({ x: -1, y: 0, z: 0 });
    for (let frame = 0; frame < 120; frame++) state = stepWalk(state, { x: 12.9, z: 0 }, 1 / 120, w, frame === 0);
    assert.ok(state.position.x < 0); assert.equal(state.position.y, 0); assert.ok(state.grounded);
  }
});

test('jump height uses vertical metres independently of horizontal scale or frame rate', () => {
  for (const rate of [30,60,144]) {
    let state = createWalkMotion({ x: 0, y: 0, z: 0 }), maximum = 0;
    const w = world({ scaleX: .2, scaleY: .01, scaleZ: .4 });
    for (let frame = 0; frame < rate; frame++) {
      state = stepWalk(state, { x: 0, z: 0 }, 1 / rate, w, frame === 0);
      maximum = Math.max(maximum, state.position.y / w.scaleY);
    }
    assert.ok(Math.abs(maximum-WALK_JUMP_HEIGHT_METRES) < .01);
    assert.ok(state.grounded); assert.equal(state.position.y,0);
  }
});
