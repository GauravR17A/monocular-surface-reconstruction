// Dedicated agent-browser CDP only; never attaches to a user's existing browser.
// node scripts/verify_explore_map.mjs <playwright-package-dir> <cdp-url>
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
const { chromium } = createRequire(import.meta.url)(process.argv[2]);
const browser = await chromium.connectOverCDP(process.argv[3]);
const page = browser.contexts()[0].pages().find(p => p.url().startsWith('http://localhost:3000'));
const output = path.resolve(import.meta.dirname, '../outputs/runtime/explore_map');
await mkdir(output, { recursive: true });
const report = { checks: [], errors: [], passed: false };
page.on('pageerror', error => report.errors.push(error.message));
const check = (name, evidence = true) => { report.checks.push({ name, evidence }); console.log('PASS', name); };
const pause = ms => page.waitForTimeout(ms);
const shot = name => page.screenshot({ path: path.join(output, `${name}.png`) });
const stage = page.locator('.terrain-canvas'), canvas = page.locator('.terrain-canvas canvas');
const mini = page.locator('.explore-minimap canvas');
const dialog = page.getByRole('dialog', { name: 'Your position', exact: true });
const nav = async () => JSON.parse(await stage.getAttribute('data-navigation'));
const frame = async () => JSON.parse(await mini.getAttribute('data-map-frame'));
const closeEnough = (a,b,epsilon = .015) => assert.ok(Math.abs(a-b) < epsilon, `${a} not close to ${b}`);
async function ready() { await page.getByRole('button', { name: 'Enlarge original image', exact: true }).waitFor({ timeout: 120000 }); }
async function enter(mode) {
  await page.getByRole('button', { name: /^explore/i }).click();
  await page.getByRole('button', { name: mode === 'fly' ? /01 \/ FLY/ : /02 \/ WALK/ }).click();
  await page.waitForFunction(() => Boolean(document.querySelector('.explore-minimap canvas')?.dataset.mapFrame), undefined, { timeout: 20000 });
  await pause(400);
}
async function orbit() {
  await page.keyboard.press('Escape');
  if (await page.locator('.flight-hud').count()) await page.evaluate(() => document.exitPointerLock());
  await page.waitForFunction(() => !document.querySelector('.flight-hud'));
}
async function fullScreenMap() {
  await dialog.waitFor();
  await page.waitForFunction(() => Boolean(document.querySelector('.explore-map-canvas.full')?.dataset.mapFrame));
  const size = await page.evaluate(() => ({ width: innerWidth, height: innerHeight }));
  const rect = await dialog.boundingBox();
  closeEnough(rect.x,0); closeEnough(rect.y,0); closeEnough(rect.width,size.width,1); closeEnough(rect.height,size.height,1);
  assert.equal(await page.locator('.flight-hud').count(), 1);
  assert.equal(await page.evaluate(() => document.pointerLockElement === null), true);
}
try {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.reload();
  // The SSR shell can expose buttons before their client handlers are hydrated.
  await page.waitForFunction(() => document.querySelector('.run-strip strong')?.textContent === 'ENGINE READY');
  await page.getByRole('button', { name: 'Load Urban demonstration', exact: true }).click(); await ready();
  await page.keyboard.press('m'); assert.equal(await dialog.isVisible(), false);
  check('M does not hijack Orbit or the original image viewer');
  await page.getByRole('button', { name: 'Top down', exact: true }).click(); await pause(400);
  await enter('fly');
  assert.equal(await page.locator('.scene-image-reference').count(), 0);
  const initial = await frame(), dimensions = JSON.parse(await stage.getAttribute('data-terrain-scale'));
  const pose = await nav();
  closeEnough(initial.u, pose.x / dimensions.width + .5);
  closeEnough(initial.v, pose.z / dimensions.depth + .5);
  assert.equal(initial.unit, 'px');
  check('Fly replaces source preview with correctly registered pixel-scale minimap', initial);
  const targetYaw = .6, targetPitch = -.25;
  await canvas.dispatchEvent('mousemove', { movementX: (pose.yaw - targetYaw) / .0022, movementY: (pose.pitch - targetPitch) / .0022, bubbles: true });
  await pause(450);
  const turned = await frame(); assert.ok(Math.abs(turned.heading - initial.heading) > .2);
  closeEnough(turned.u, initial.u); closeEnough(turned.v, initial.v);
  check('Looking rotates the heading-up map without moving the horizontal position');
  await page.keyboard.down('w'); await pause(700); await page.keyboard.up('w'); await pause(400);
  const moved = await frame(), movedPose = await nav();
  assert.ok(Math.hypot(moved.u - turned.u, moved.v - turned.v) > .015);
  closeEnough(moved.u, movedPose.x / dimensions.width + .5);
  closeEnough(moved.v, movedPose.z / dimensions.depth + .5);
  await shot('fly-live-map');
  check('Actual flight motion scrolls the local map; position agrees with 3D camera', moved);
  await page.keyboard.press('m'); await fullScreenMap();
  await pause(400); const paused = await nav();
  await page.keyboard.down('w'); await pause(500); await page.keyboard.up('w');
  await page.mouse.move(250, 250); await pause(400);
  const still = await nav();
  for (const key of ['x','y','z','yaw','pitch']) closeEnough(still[key], paused[key], 1e-6);
  await shot('full-screen-map');
  check('M map takes over the entire screen, frees cursor and freezes movement/look');
  await page.keyboard.down('m'); await pause(100); await page.keyboard.down('m'); await pause(100); await page.keyboard.up('m');
  assert.equal(await dialog.isVisible(), false); assert.equal(await page.locator('.flight-hud').count(),1);
  await pause(400); const resumed = await nav();
  for (const key of ['x','z','yaw','pitch']) closeEnough(resumed[key], paused[key], 1e-5);
  check('M closes map and resumes same Explore pose; key repeat does not bounce the map');
  await page.keyboard.press('m'); await fullScreenMap();
  await page.getByRole('button', { name: 'Close full map (M)', exact: true }).click();
  assert.equal(await dialog.isVisible(),false);
  await orbit(); await ready();
  assert.equal(await page.locator('.explore-minimap').count(),0);
  check('Map close button resumes Explore; leaving Explore restores the original-image reference');

  await page.getByRole('button', { name: 'FULLSCREEN', exact: true }).click(); await pause(200);
  await enter('walk');
  const walkStart = await nav(), walkMap = await frame();
  closeEnough(walkMap.u, walkStart.x / dimensions.width + .5, .02);
  assert.equal(walkMap.mode,'walk'); assert.ok(walkMap.span < initial.span);
  await page.keyboard.down('w'); await pause(600); await page.keyboard.up('w'); await pause(400);
  await shot('walk-minimap');
  await page.keyboard.press('m'); await fullScreenMap();
  await pause(400); const pausedWalk = await nav();
  // Space on the auto-focused close button intentionally activates it (native
  // keyboard accessibility). Focus the dialog to test paused movement instead.
  await dialog.evaluate(element => element.focus());
  assert.equal(await dialog.evaluate(element => document.activeElement === element), true);
  await page.keyboard.down('Space'); await pause(300); await page.keyboard.up('Space'); await pause(300);
  const stillWalk = await nav();
  for (const key of ['x','y','z']) closeEnough(pausedWalk[key],stillWalk[key],1e-6);
  await page.keyboard.press('Escape'); await pause(100);
  assert.equal(await dialog.isVisible(),false); assert.equal(await page.locator('.flight-hud').count(),1);
  assert.notEqual(await page.locator('.viewport').getAttribute('data-fullscreen-kind'),'off');
  check('Walk uses tighter map context; map pauses jumping; Escape closes only the map in fullscreen');
  await orbit();
  assert.notEqual(await page.locator('.viewport').getAttribute('data-fullscreen-kind'),'off');
  await page.getByRole('button', { name: 'EXIT FULLSCREEN', exact: true }).click();
  for (const name of ['Sparse','Hilly','Forest']) {
    await page.getByRole('button', { name: `Load ${name} demonstration`, exact: true }).click(); await ready();
    await page.getByRole('button', { name: 'Top down', exact: true }).click(); await pause(400);
    await enter('fly');
    const value = await frame(); assert.ok(Number.isFinite(value.span) && value.span > 0);
    if (name === 'Hilly') assert.equal(value.unit,'m');
    await shot(`${name.toLowerCase()}-minimap`);
    check(`${name} scene gets its own adaptive map and correct scale units`, value);
    if (name === 'Hilly') {
      await page.setViewportSize({ width: 390, height: 844 });
      await page.keyboard.press('m'); await fullScreenMap(); await shot('mobile-full-map');
      await page.keyboard.press('m'); await page.setViewportSize({ width: 1600, height: 1000 });
      check('M map also covers the full screen at 390px width');
    }
    await orbit();
  }
  assert.deepEqual(report.errors, []); report.passed = true;
} catch (error) {
  report.failure = error.stack; console.error(error); process.exitCode = 1;
  await shot('failure').catch(() => {});
} finally {
  await writeFile(path.join(output, 'verification.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
