// Dedicated agent-browser CDP session; never attaches to the user's browser.
// node scripts/verify_scene_image_reference.mjs <playwright-package-dir> <cdp-url>
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
const { chromium } = createRequire(import.meta.url)(process.argv[2]);
const browser = await chromium.connectOverCDP(process.argv[3]);
const page = browser.contexts()[0].pages().find(p => p.url().startsWith('http://localhost:3000'));
if (!page) throw new Error('Open the dedicated agent-browser session first');
const root = path.resolve(import.meta.dirname, '..');
const output = path.join(root, 'outputs/runtime/scene_image_reference');
await mkdir(output, { recursive: true });
const report = { assertions: [], pageErrors: [], passed: false };
page.on('pageerror', error => report.pageErrors.push(error.message));
const check = (name, evidence = true) => { report.assertions.push({ name, evidence }); console.log('PASS', name); };
const reference = page.getByRole('region', { name: 'Original input image reference', exact: true });
const referenceImage = reference.locator('img');
const waitReady = () => page.waitForFunction(() => {
  const img = document.querySelector('.scene-image-reference img');
  return img?.complete && img.naturalWidth > 0 && document.querySelector('.scene-label')?.textContent.includes('INTERACTIVE MESH READY')
    && !document.querySelector('.scene-reveal-card') && !document.querySelector('.import-waiting-stage');
}, undefined, { timeout: 180000 });
const imageEvidence = () => referenceImage.evaluate(img => ({ src: img.src, width: img.naturalWidth, height: img.naturalHeight, fit: getComputedStyle(img).objectFit }));
const overlap = (a, b) => a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
const screenshot = name => page.screenshot({ path: path.join(output, `${name}.png`) });
try {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.reload();
  await page.getByRole('region', { name: 'Empty 3D workspace' }).waitFor();
  assert.equal(await reference.count(), 0);
  check('Neutral empty workspace has no unrelated input preview');
  const expected = { Urban: '/demo/copenhagen_rgb.jpg', Sparse: '/demo/landscapes/sparse/texture.jpg', Hilly: '/demo/landscapes/hilly/texture.jpg', Forest: '/demo/forest_validation/satellite_rgb.jpg' };
  for (const [name, src] of Object.entries(expected)) {
    await page.getByRole('button', { name: `Load ${name} demonstration`, exact: true }).click();
    await waitReady();
    const img = await imageEvidence();
    assert.equal(new URL(img.src).pathname, src);
    assert.equal(img.fit, 'contain');
    assert.ok(await reference.locator('.scene-image-toggle').getAttribute('aria-expanded') === 'true');
    await screenshot(name.toLowerCase());
    await reference.locator('.scene-image-toggle').click();
    assert.equal(await referenceImage.isVisible(), false);
    check(`${name}: correct full-image reference, no crop; collapse works`, img);
    // The next scene must reopen and load its own reference rather than retaining the previous one.
  }
  await page.getByRole('button', { name: 'Load Urban demonstration', exact: true }).click();
  await waitReady();
  const urbanImage = await imageEvidence();
  const canvasState = () => page.locator('.terrain-canvas').evaluate(el => ({ mesh: el.dataset.terrainMesh, scale: el.dataset.terrainScale }));
  const before = await canvasState();
  await page.getByRole('button', { name: 'Toggle metric grid view', exact: true }).click();
  assert.deepEqual(await imageEvidence(), urbanImage);
  assert.deepEqual(await canvasState(), before);
  await screenshot('metric');
  check('Metric view retains original RGB; no changes to mesh or height scale');
  await page.getByRole('button', { name: 'Toggle metric grid view', exact: true }).click();
  await page.getByRole('button', { name: 'Top down', exact: true }).click();
  await page.waitForTimeout(500);
  await page.getByRole('button', { name: /^inspect$/i }).click();
  const canvas = page.locator('.terrain-canvas canvas');
  const canvasBox = await canvas.boundingBox();
  for (const fraction of [.5, .4, .6]) {
    await canvas.click({ position: { x: canvasBox.width * fraction, y: canvasBox.height * .55 } });
    if (await page.locator('.inspect-popover').count()) break;
  }
  await page.locator('.inspect-popover').waitFor();
  const inspectionText = await page.locator('.inspect-popover').innerText();
  assert.equal(overlap(await reference.boundingBox(), await page.locator('.inspect-popover').boundingBox()), false);
  await reference.locator('.scene-image-toggle').click();
  assert.equal(await page.locator('.inspect-popover').innerText(), inspectionText);
  await reference.locator('.scene-image-toggle').click();
  await screenshot('inspect');
  check('Reference and inspection do not overlap; reference controls do not select/deselect terrain');
  await page.getByRole('button', { name: /^orbit$/i }).click();
  await page.getByRole('button', { name: 'Fit terrain', exact: true }).click();
  await page.getByRole('button', { name: /^explore/i }).click();
  await page.getByRole('button', { name: /01 \/ FLY/ }).click();
  await page.locator('.flight-hud').waitFor();
  const minimap = page.getByRole('region', { name: 'Explore minimap', exact: true });
  await minimap.waitFor();
  assert.equal(overlap(await minimap.boundingBox(), await page.locator('.flight-hud').boundingBox()), false);
  assert.equal(await reference.count(), 0);
  await screenshot('fly');
  await page.keyboard.press('Escape');
  if (await page.locator('.flight-hud').count()) await page.evaluate(() => document.exitPointerLock());
  await page.waitForFunction(() => !document.querySelector('.flight-hud'));
  await waitReady();
  assert.deepEqual(await imageEvidence(), urbanImage);
  check('Fly-through uses a live minimap above the scanner; Orbit restores the unchanged original reference');
  await page.getByRole('button', { name: 'FULLSCREEN', exact: true }).click();
  await page.waitForTimeout(400);
  const fullscreen = await reference.evaluate(el => ({ contained: document.querySelector('.viewport')?.contains(el), kind: document.querySelector('.viewport')?.dataset.fullscreenKind }));
  assert.equal(fullscreen.contained, true); assert.notEqual(fullscreen.kind, 'off');
  await screenshot('fullscreen');
  await page.getByRole('button', { name: 'EXIT FULLSCREEN', exact: true }).click();
  check('Reference remains inside native/window fullscreen', fullscreen);
  for (const width of [390, 820, 1280, 1600]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.waitForTimeout(200);
    const rect = await reference.boundingBox();
    const camera = await page.locator('.terrain-camera-tools').boundingBox();
    const scale = await page.locator('.scene-scale').boundingBox();
    const bounds = await page.locator('.terrain-canvas-wrap').boundingBox();
    assert.ok(rect.x >= bounds.x && rect.x + rect.width <= bounds.x + bounds.width + 1);
    assert.ok(rect.y >= bounds.y && rect.y + rect.height <= bounds.y + bounds.height + 1);
    assert.equal(overlap(rect, camera), false, `Camera controls covered at ${width}`);
    if (scale) assert.equal(overlap(rect, scale), false, `Scale covered at ${width}`);
    if (width === 390) await screenshot('mobile');
  }
  check('Preview stays inside 3D viewport and avoids camera/scale controls at 390/820/1280/1600px');
  await page.setViewportSize({ width: 1600, height: 1000 });
  for (const suffix of ['png', 'jpg', 'tif']) {
    const responsePromise = page.waitForResponse(res => res.url().includes('/api/predict') && res.request().method() === 'POST', { timeout: 180000 });
    await page.locator('input[aria-label="Import images"]').setInputFiles(path.join(root, `outputs/runtime/import-review/urban-import.${suffix}`));
    await page.waitForFunction(() => !document.querySelector('.scene-image-reference'), undefined, { timeout: 10000 });
    const response = await responsePromise;
    assert.ok(response.ok(), `${suffix} prediction failed`);
    const payload = await response.json();
    await waitReady();
    const img = await imageEvidence();
    assert.equal(new URL(img.src).pathname, payload.texture_url);
    assert.ok((await reference.innerText()).includes(`urban-import.${suffix}`));
    if (suffix === 'tif') await screenshot('tiff-upload');
    check(`${suffix.toUpperCase()} real upload → API texture → matching 2D reference; stale image hidden while processing`, img);
  }
  assert.deepEqual(report.pageErrors, []);
  report.passed = true;
} catch (error) {
  report.failure = error.stack; console.error(error); process.exitCode = 1;
  await screenshot('failure').catch(() => {});
} finally {
  await writeFile(path.join(output, 'verification.json'), JSON.stringify(report, null, 2));
  await browser.close();
}
