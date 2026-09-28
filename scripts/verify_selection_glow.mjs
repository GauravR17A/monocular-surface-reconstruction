// Dedicated agent-browser session only; does not control the user's tab.
// node scripts/verify_selection_glow.mjs <playwright-package-dir> <cdp-url>
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
const { chromium } = createRequire(import.meta.url)(process.argv[2]);
const browser = await chromium.connectOverCDP(process.argv[3]);
const page = browser.contexts()[0].pages().find(p => p.url().startsWith('http://localhost:3000'));
if (!page) throw new Error('Open Monocular Surface Reconstruction in the dedicated browser first');
const output = path.resolve(import.meta.dirname, '../outputs/runtime/selection_glow');
await mkdir(output, { recursive: true });
const report = { assertions: [], errors: [], passed: false };
const check = (name, evidence = true) => { report.assertions.push({ name, evidence }); console.log('PASS', name); };
page.on('pageerror', error => report.errors.push(error.message));
const canvas = page.locator('.terrain-canvas canvas');
const glow = async () => {
  const value = await page.locator('.terrain-canvas').getAttribute('data-selection-glow');
  return value && value !== 'off' ? JSON.parse(value) : null;
};
const pause = ms => page.waitForTimeout(ms);
const selectMode = name => page.getByRole('button', { name: new RegExp(`^${name}$`, 'i') }).click();
try {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.waitForFunction(() => document.querySelector('.six-class-panel')?.dataset.classificationStatus === 'ready', undefined, { timeout: 120000 });
  await page.getByRole('button', { name: 'Top down', exact: true }).click(); await pause(500);
  await selectMode('inspect');
  const box = await canvas.boundingBox();
  const coordinates = [.35,.45,.55,.65,.75].flatMap(y => [.35,.45,.55,.65,.75].map(x => [x,y]));
  const unobscured = async (x, y) => page.evaluate(({ x, y }) => document.elementFromPoint(x, y) === document.querySelector('.terrain-canvas canvas'), { x: box.x + box.width * x, y: box.y + box.height * y });
  const found = new Map(); let regionFound = false;
  for (const [x, y] of coordinates) {
    if (!await unobscured(x,y)) continue;
    await canvas.click({ position: { x: box.width * x, y: box.height * y } }); await pause(140);
    const selected = await glow();
    if (!selected) continue;
    assert.equal(await page.locator('.inspect-popover').count(), 1);
    if (selected.kind === 'building' && !found.has(selected.identity)) {
      found.set(selected.identity, { x, y, selected });
      await page.screenshot({ path: path.join(output, `building-${found.size}.png`) });
    }
    if (selected.kind === 'region' && !regionFound) {
      regionFound = true;
      await page.screenshot({ path: path.join(output, 'surface-region.png') });
    }
    if (found.size >= 2 && regionFound) break;
  }
  assert.ok(found.size >= 2, 'inspect two distinct buildings');
  assert.ok(regionFound, 'inspect a connected surface region');
  check('Click selection follows two different building IDs and connected terrain regions', [...found.values()]);
  const first = [...found.values()][0];
  await canvas.click({ position: { x: box.width * first.x, y: box.height * first.y } }); await pause(150);
  const before = await glow();
  const height = await page.locator('.inspect-popover div').first().innerText();
  await page.mouse.move(box.x + 15, box.y + 15); await pause(500);
  assert.equal((await glow()).identity, before.identity);
  check('Pinned Inspect outline persists when the pointer leaves the object');
  await page.getByRole('button', { name: 'Toggle metric grid view', exact: true }).click(); await pause(650);
  assert.equal((await glow()).identity, before.identity);
  assert.equal(await page.locator('.inspect-popover div').first().innerText(), height);
  await page.screenshot({ path: path.join(output, 'metric-selection.png') });
  check('Photo → metric transition retains selected object and unchanged height');
  await page.getByRole('button', { name: 'Toggle metric grid view', exact: true }).click();
  await selectMode('orbit'); await pause(150); assert.equal(await glow(), null);
  check('Mode switch clears the outline');
  await selectMode('inspect');
  await canvas.click({ position: { x: box.width * first.x, y: box.height * first.y } }); await pause(150);
  assert.ok(await glow());
  for (const [x,y] of [[.5,.06],[.15,.3],[.1,.45],[.5,.12]]) {
    if (!await unobscured(x,y)) continue;
    await canvas.click({ position: { x: box.width*x, y: box.height*y } }); await pause(150);
    if (!await glow()) break;
  }
  assert.equal(await glow(), null); assert.equal(await page.locator('.inspect-popover').count(), 0);
  check('Clicking empty space clears metadata and glow together');

  await page.getByRole('button', { name: 'Fit terrain', exact: true }).click(); await pause(500);
  for (const mode of ['FLY', 'WALK']) {
    await page.getByRole('button', { name: /^explore/i }).click();
    await page.getByRole('button', { name: mode === 'FLY' ? /01 \/ FLY/ : /02 \/ WALK/ }).click();
    await pause(500);
    const pose = JSON.parse(await page.locator('.terrain-canvas').getAttribute('data-navigation'));
    const yaw = mode === 'FLY' ? Math.atan2(pose.x, pose.z) : pose.yaw;
    const pitch = mode === 'FLY' ? -Math.atan2(pose.y - 3, Math.hypot(pose.x, pose.z)) : -.65;
    await canvas.dispatchEvent('mousemove', { movementX: (pose.yaw - yaw) / .0022, movementY: (pose.pitch - pitch) / .0022, bubbles: true });
    await page.waitForFunction(() => document.querySelector('.flight-hud')?.innerText.includes('AIM SCAN')
      && document.querySelector('.terrain-canvas')?.dataset.selectionGlow?.startsWith('{'), undefined, { timeout: 7000 });
    assert.ok(await glow()); check(`${mode}: two-second aim shows metadata and boundary`, await glow());
    await canvas.dispatchEvent('pointerup', { button: 0, bubbles: true }); await pause(200);
    assert.match(await page.locator('.flight-hud').innerText(), /CLICK SCAN/);
    assert.ok(await glow());
    await page.screenshot({ path: path.join(output, `${mode.toLowerCase()}-selection.png`) });
    check(`${mode}: click scanning keeps the current boundary`);
    // Look up into empty sky: selection and readout must disappear together.
    const current = JSON.parse(await page.locator('.terrain-canvas').getAttribute('data-navigation'));
    await canvas.dispatchEvent('mousemove', { movementX: 0, movementY: (current.pitch - 1.3) / .0022, bubbles: true });
    await pause(450); assert.equal(await glow(), null);
    assert.match(await page.locator('.flight-hud').innerText(), /STANDBY/);
    check(`${mode}: moving aim off the surface clears the boundary`);
    await page.keyboard.press('Escape');
    if (await page.locator('.flight-hud').count()) await page.evaluate(() => document.exitPointerLock());
    await page.waitForFunction(() => !document.querySelector('.flight-hud'));
    assert.equal(await glow(), null);
  }
  await page.getByRole('button', { name: 'Load Forest demonstration', exact: true }).click();
  await pause(250); assert.equal(await glow(), null);
  check('Changing scenes does not retain the previous selection');
  assert.deepEqual(report.errors, []); report.passed = true;
} catch (error) {
  report.failure = error.stack; console.error(error); process.exitCode = 1;
  await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {});
} finally {
  await writeFile(path.join(output, 'verification.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
