// Connects to a dedicated agent-browser verification session, never the user's browser.
// Usage: node scripts/verify_six_class_integration.mjs <playwright-package-dir> <cdp-url> <ui|upload|navigation>
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
const { chromium } = createRequire(import.meta.url)(process.argv[2]);
const phase = process.argv[4] ?? 'ui';
const root = path.resolve(import.meta.dirname, '..');
const output = path.join(root, 'outputs/runtime/six_class_day1');
await mkdir(output, {recursive:true});
const browser = await chromium.connectOverCDP(process.argv[3]);
const context = browser.contexts()[0];
const page = context.pages().find(p => p.url().startsWith('http://localhost:3000')) ?? await context.newPage();
page.setDefaultTimeout(20_000);
const errors = []; const requests = []; const evidence = {phase, assertions:[], errors};
page.on('pageerror', error => errors.push(error.message));
page.on('request', request => { if (request.url().includes('/api/') && request.method() === 'POST') requests.push({url:request.url(),body:request.postData()}); });
const panel = page.locator('.six-class-panel');
const check = (name, data = true) => { evidence.assertions.push({name, data}); console.log('PASS', name); };
const ready = () => page.waitForFunction(() => document.querySelector('.six-class-panel')?.dataset.classificationStatus === 'ready', undefined, {timeout:120_000});
const status = wanted => page.waitForFunction(wanted => document.querySelector('.six-class-panel')?.dataset.classificationStatus === wanted, wanted, {timeout:120_000});
const count = () => requests.filter(r=>r.url.endsWith('/api/classify')).length;
const geometry = () => page.locator('.terrain-canvas').getAttribute('data-class-focus');
async function hydrated() { await page.waitForFunction(()=>document.querySelector('.run-strip strong')?.textContent==='ENGINE READY'); }
async function scene(name) { await hydrated(); await page.getByRole('button', {name:`Load ${name} demonstration`,exact:true}).click(); await ready(); }
async function hashes(dir) {
  const entries = await readdir(dir, {withFileTypes:true}); const result={};
  for (const entry of entries) if (entry.isFile()) result[entry.name] = createHash('sha256').update(await readFile(path.join(dir,entry.name))).digest('hex');
  return result;
}
async function inspectPoint() {
  await page.getByRole('button',{name:/^inspect$/i}).click();
  const canvas=page.locator('.terrain-canvas canvas'); const box=await canvas.boundingBox();
  for (const [x,y] of [[.6,.65],[.55,.5],[.5,.5]]) {
    await canvas.click({position:{x:box.width*x,y:box.height*y}});
    if (await page.locator('.inspect-popover').count()) return;
  }
  throw new Error('Could not inspect a visible surface');
}
try {
  if (phase === 'ui') {
    await page.goto('http://localhost:3000/');
    assert.equal(await panel.getAttribute('data-classification-status'),'idle');
    assert.equal(await panel.getByRole('button',{name:/^Highlight /}).count(),6);
    assert.equal(count(),0);
    check('Neutral grid: six named categories, no inference before a scene is selected');
    await scene('Urban'); assert.equal(count(),1);
    assert.equal(await panel.getByRole('checkbox').isChecked(),false);
    check('Urban automatically classified once; photo view preserved');
    const original=await geometry();
    for (const name of ['Ground','Buildings','Water','Roads','Low vegetation','Trees']) {
      const button=panel.getByRole('button',{name:`Highlight ${name}`,exact:true});
      if (await button.isDisabled()) { check(`${name}: not-detected row remains visible`); continue; }
      await button.click();
      assert.equal(await button.getAttribute('aria-pressed'),'true');
      await page.waitForFunction(()=>document.querySelector('.terrain-canvas')?.dataset.sixClassOverlay?.startsWith('six-class'));
      assert.equal(await geometry(),original);
    }
    await panel.getByRole('checkbox').uncheck();
    await inspectPoint();
    const text=await page.locator('.inspect-popover').innerText();
    assert.match(text,/RGB six-class V3/); assert.match(text,/Protected height model/);
    const value=await page.locator('.inspect-popover div').first().innerText();
    await panel.getByRole('button',{name:'Highlight Buildings',exact:true}).click();
    assert.equal(await page.locator('.inspect-popover div').first().innerText(),value);
    assert.equal(count(),1); check('All available filters preserve geometry, pinned heights and inference count', {value});
    await page.screenshot({path:path.join(output,'urban-inspection.png')});

    let release; const held=new Promise(resolve=>{release=resolve;});
    await page.route('**/api/classify',async route=>{await held; await route.continue();});
    await page.getByRole('button',{name:'Load Forest demonstration',exact:true}).click();
    await status('loading');
    assert.equal(await panel.locator('canvas').count(),0);
    assert.equal(await panel.getByRole('button',{name:'Highlight Trees'}).isDisabled(),true);
    assert.equal(await page.locator('.inspect-popover').count(),0);
    check('Changing scene clears old labels, filter and pinned inspection before new results arrive');
    release(); await ready(); await page.unroute('**/api/classify');
    await scene('Urban'); assert.equal(count(),2);
    check('Revisiting Urban uses the cached source result, not another GPU job');

    await page.goto('http://localhost:3000/');
    await page.route('**/api/classify',route=>route.fulfill({status:500,contentType:'application/json',body:JSON.stringify({detail:'Verification: temporary classifier failure'})}));
    await hydrated();
    await page.getByRole('button',{name:'Load Urban demonstration',exact:true}).click();
    await status('error'); await inspectPoint();
    assert.match(await page.locator('.inspect-popover').innerText(),/Classification unavailable/i);
    assert.match(await page.locator('.inspect-popover').innerText(),/Protected height model/);
    const failedHeight=await page.locator('.inspect-popover div').first().innerText();
    check('Simulated classifier failure keeps heights and explicitly withholds classification');
    await page.screenshot({path:path.join(output,'classifier-failure.png')});
    await page.unroute('**/api/classify');
    await panel.getByRole('button',{name:'Retry identification'}).click(); await ready();
    assert.equal(await page.locator('.inspect-popover div').first().innerText(),failedHeight);
    check('Retry restores labels without rebuilding heights');

    await page.goto('http://localhost:3000/');
    await page.route('**/api/classify',route=>route.fulfill({status:422,contentType:'application/json',body:JSON.stringify({detail:'Verification: classifier requires 8-bit RGB'})}));
    await hydrated();
    await page.getByRole('button',{name:'Load Urban demonstration',exact:true}).click(); await status('unsupported');
    assert.match(await panel.innerText(),/8-bit RGB/);
    assert.equal(await panel.locator('canvas').count(),0);
    check('Simulated unsupported response is explicit; no fabricated classification map');
    await page.unroute('**/api/classify');
    await page.goto('http://localhost:3000/'); await scene('Hilly');
    assert.match(await panel.innerText(),/outside the classifier/);
    check('Hilly reference retains its out-of-domain notice');
  } else if (phase === 'upload') {
    await page.goto('http://localhost:3000/');
    let job; let before; let after;
    await page.route('**/api/classify',async route=>{
      const match=route.request().postData()?.match(/name="job_id"\r\n\r\n([a-f0-9]{32})/);
      assert.ok(match,'classification must use the completed uploaded job'); job=match[1];
      before=await hashes(path.join(root,'outputs/web_jobs',job));
      await route.continue();
    });
    const image='D:/MSRData/evaluation/oem_all_six_20260919_v1/results/scenes/duesseldorf_65/original.png';
    await hydrated();
    await page.locator('input[accept^=".jpg"]').setInputFiles(image); await ready();
    after=await hashes(path.join(root,'outputs/web_jobs',job));
    assert.deepEqual(after,before);
    assert.equal(count(),1); assert.equal(requests.filter(r=>r.url.endsWith('/api/predict')).length,1);
    evidence.job=job; evidence.artifactHashes=after;
    check('Real PNG upload → protected CUDA heights → automatic CUDA six-class identification');
    check('Every original job artifact remains byte-identical across classification',Object.keys(after));
    for (const name of ['Ground','Buildings','Water','Roads','Low vegetation','Trees']) {
      const button=panel.getByRole('button',{name:`Highlight ${name}`,exact:true});
      assert.equal(await button.isEnabled(),true); await button.click();
      assert.equal(await button.getAttribute('aria-pressed'),'true');
    }
    check('All six class filters work on a previously used diagnostic scene (not new accuracy evidence)');
    await panel.getByRole('checkbox').uncheck();
    await page.screenshot({path:path.join(output,'six-class-upload.png')});
    await page.unroute('**/api/classify');
  } else if (phase === 'navigation') {
    await page.goto('http://localhost:3000/'); await scene('Urban');
    for (const mode of ['FLY','WALK']) {
      await page.getByRole('button',{name:/^explore/i}).click();
      await page.getByRole('button',{name: mode==='FLY' ? /01 \/ FLY/ : /02 \/ WALK/}).click();
      const canvas=page.locator('.terrain-canvas canvas');
      await page.waitForFunction(mode=>JSON.parse(document.querySelector('.terrain-canvas')?.dataset.navigation ?? '{}').mode===mode.toLowerCase(),mode);
      // Aim at the visible scene centre in flight, or at nearby ground in walk.
      // Fixed mouse deltas are unreliable after automation moves a locked pointer.
      // Let native pointer-capture movement settle before reading the pose.
      await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      const pose=JSON.parse(await page.locator('.terrain-canvas').getAttribute('data-navigation'));
      const yaw=mode==='FLY' ? Math.atan2(pose.x,pose.z) : pose.yaw;
      const pitch=mode==='FLY' ? -Math.atan2(pose.y-3,Math.hypot(pose.x,pose.z)) : -.65;
      await canvas.dispatchEvent('mousemove',{movementX:(pose.yaw-yaw)/.0022,movementY:(pose.pitch-pitch)/.0022,bubbles:true});
      await page.evaluate(()=>window.dispatchEvent(new Event('blur')));
      await page.waitForFunction(()=>document.querySelector('.flight-hud')?.innerText.includes('RGB six-class V3') && document.querySelector('.flight-hud')?.innerText.includes('AIM SCAN'));
      const aim=await page.locator('.flight-hud').innerText(); assert.match(aim,/AIM SCAN/);
      // A locator click moves the pointer first, which rotates a locked camera.
      // Press at its current position to keep the crosshair on the same target.
      await canvas.dispatchEvent('pointerdown',{button:0,bubbles:true});
      await page.waitForFunction(()=>document.querySelector('.flight-hud')?.innerText.includes('CLICK SCAN'));
      const click=await page.locator('.flight-hud').innerText(); assert.match(click,/CLICK SCAN/);
      check(`${mode}: six-class label and independent height source through dwell and pointer-event handlers`,{aim,click});
      await page.screenshot({path:path.join(output,`${mode.toLowerCase()}-scanner.png`)});
      await page.keyboard.press('Escape');
      // Exit pointer capture on browsers where Escape is handled natively.
      if (await page.locator('.flight-hud').count()) await page.evaluate(()=>document.exitPointerLock());
      await page.waitForFunction(()=>!document.querySelector('.flight-hud'));
    }
    assert.equal(count(),1); check('Changing navigation modes never reruns classification');
  }
  assert.deepEqual(errors,[]);
  check('No page errors during verification');
  evidence.passed=true;
} catch(error) {
  evidence.passed=false; evidence.failure=error.stack;
  await page.screenshot({path:path.join(output,`${phase}-failure.png`)}).catch(()=>{});
  console.error(error); process.exitCode=1;
} finally {
  evidence.requests=requests.map(r=>({url:r.url}));
  await writeFile(path.join(output,`${phase}-verification.json`),JSON.stringify(evidence,null,2));
  await browser.close();
}
