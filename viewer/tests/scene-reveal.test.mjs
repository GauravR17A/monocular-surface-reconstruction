import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sceneRevealAt, SCENE_REVEAL_DURATION } from '../app/scene-reveal.ts';

test('reveal proceeds from grid to shapes, heights, texture and complete', () => {
  assert.deepEqual([0, .6, 1.5, 3, 4].map(t => sceneRevealAt(t).phase),
    ['grid', 'footprints', 'heights', 'texture', 'complete']);
  assert.equal(sceneRevealAt(0).showBuildings, false);
  assert.equal(sceneRevealAt(.6).showBuildings, true);
  assert.equal(sceneRevealAt(.6).scaleY, .001);
  assert.equal(sceneRevealAt(2.75).scaleY, 1);
  assert.equal(sceneRevealAt(2.75).metricMix, 1);
  assert.equal(sceneRevealAt(SCENE_REVEAL_DURATION).metricMix, 0);
});

test('skip and reduced-motion path land on the identical complete result', () => {
  assert.deepEqual(sceneRevealAt(.01, true), sceneRevealAt(10));
  assert.equal(sceneRevealAt(0, true).progress, 1);
  assert.equal(sceneRevealAt(0, true).scaleY, 1);
});

test('rise is continuous, bounded and monotonic with no changes to measured data', () => {
  let previous = sceneRevealAt(0);
  for (let t = 0; t < 5; t += .01) {
    const next = sceneRevealAt(t);
    assert.ok(next.scaleY >= previous.scaleY && next.scaleY <= 1);
    assert.ok(next.metricMix <= previous.metricMix && next.metricMix >= 0);
    assert.ok(next.progress >= previous.progress && next.progress <= 1);
    assert.ok(next.scaleY - previous.scaleY < .01);
    previous = next;
  }
  assert.deepEqual(sceneRevealAt(-5), sceneRevealAt(0));
});
