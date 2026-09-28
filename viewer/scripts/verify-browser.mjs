/** Exercise the production viewer and real API without a listening application server.
 * Build first. Set MSR_PYTHON and optionally MSR_BROWSER_CDP; see PERFORMANCE.md.
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
const viewer = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const root = path.dirname(viewer);
process.chdir(viewer);
const { default: handle } = await import('../dist/server/index.js');
const evidenceDir = path.join(root, 'docs/evidence');
const assetDir = path.join(root, 'docs/assets');
await mkdir(assetDir, { recursive: true });
const child = spawn(process.env.MSR_PYTHON || 'python', [path.join(root, 'scripts/verify_api_bridge.py')], { cwd: root, env: process.env, windowsHide: true, stdio: ['pipe','pipe','pipe'] });
const pending = new Map();
let nextId = 0;
child.stderr.on('data', b => process.stderr.write(b));
createInterface({ input: child.stdout }).on('line', line => {
  try { const data = JSON.parse(line); const p = pending.get(data.id); if (p) { pending.delete(data.id); if (data.error) p.reject(new Error(data.error)); else p.resolve(data); } }
  catch { process.stderr.write(line + '\n'); }
});
child.on('exit', code => { for (const p of pending.values()) p.reject(new Error(`API bridge exited ${code}`)); pending.clear(); });
function api(method, url, headers={}, body=Buffer.alloc(0)) {
  const id = ++nextId;
  return new Promise((resolve, reject) => { pending.set(id,{resolve,reject}); child.stdin.write(JSON.stringify({ id, method, path: url, headers, body: body.toString('base64') })+'\n'); });
}
const connected = Boolean(process.env.MSR_BROWSER_CDP);
const browser = connected ? await chromium.connectOverCDP(process.env.MSR_BROWSER_CDP) : await chromium.launch({headless:true});
const context = connected ? browser.contexts()[0] : await browser.newContext();
const page = await context.newPage();
await page.setViewportSize({width:1600,height:1000});
const report = { status_date:'2026-09-28', transport:'Playwright route interception; production request handler + FastAPI TestClient; no application listener', checks:[], requests:[], console_errors:[], page_errors:[] };
page.on('pageerror', err => report.page_errors.push(String(err)));
page.on('console', msg => { if (msg.type()==='error') report.console_errors.push(msg.text()); });
const types = {'.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png','.ico':'image/x-icon','.tif':'image/tiff','.json':'application/json','.woff2':'font/woff2'};
await page.route('**/*', async route => {
  const req=route.request(), url=new URL(req.url());
  try {
    if (url.origin==='http://127.0.0.1:8000') {
      const headers={...req.headers()}; delete headers['content-length'];
      const r=await api(req.method(),url.pathname+url.search,headers,req.postDataBuffer()||Buffer.alloc(0));
      report.requests.push({ method:req.method(), path:url.pathname.replace(/[a-f0-9]{32}/g,'JOB'), status:r.status, seconds:r.elapsed_seconds });
      delete r.headers['content-encoding']; delete r.headers['content-length'];
      return route.fulfill({status:r.status,headers:r.headers,body:Buffer.from(r.body,'base64')});
    }
    if (url.origin==='http://msr.test') {
      const relative=decodeURIComponent(url.pathname).replace(/^\/+/, '');
      for (const base of [path.join(viewer,'dist/client'),path.join(viewer,'public')]) {
        const file=path.resolve(base,relative);
        if (!file.startsWith(base+path.sep)) continue;
        try { if ((await stat(file)).isFile()) return route.fulfill({status:200,contentType:types[path.extname(file)]||'application/octet-stream',body:await readFile(file)}); } catch {}
      }
      const response=await handle(new Request(req.url(),{method:req.method(),headers:req.headers()}));
      return route.fulfill({status:response.status,headers:Object.fromEntries(response.headers),body:Buffer.from(await response.arrayBuffer())});
    }
    if (url.protocol==='data:' || url.protocol==='blob:') return route.continue();
    throw new Error(`Unexpected external request: ${url.origin}`);
  } catch(err) { report.page_errors.push(String(err)); await route.abort(); }
});
const check = (name,details={}) => { report.checks.push({name,passed:true,...details}); console.log('PASS '+name); };
async function screenshot(name) { await page.screenshot({path:path.join(assetDir,name),fullPage:false}); }
try {
  const health=await api('GET','/api/health'); assert.equal(health.status,200); check('API health',JSON.parse(Buffer.from(health.body,'base64')));
  await page.goto('http://msr.test/',{waitUntil:'networkidle'});
  await page.getByRole('heading',{name:'Monocular Surface Reconstruction',exact:true}).waitFor();
  check('Production SSR, hydration and scientific identity');
  await page.getByRole('button',{name:'Load Urban demonstration',exact:true}).click();
  await page.waitForFunction(() => document.querySelector('canvas')?.width > 0);
  await page.waitForTimeout(6000);
  await screenshot('workspace.png');
  check('Urban demo and WebGL canvas');
  console.log((await page.locator('body').innerText()).slice(-7000));
  await page.getByRole('button',{name:'Load Hilly demonstration',exact:true}).click();
  await page.waitForTimeout(6000); await screenshot('hilly-workspace.png');
  check('Hilly georeferenced DSM demo');
  await page.getByRole('button',{name:'Load Forest demonstration',exact:true}).click();
  await page.waitForTimeout(5000); check('Forest demo');
  const upload=path.join(viewer,'public/demo/copenhagen_rgb.jpg');
  const prediction=page.waitForResponse(r=>r.url().includes('/api/predict') && r.request().method()==='POST',{timeout:240000});
  await page.getByLabel('Import images',{exact:true}).setInputFiles(upload);
  const result=await prediction; assert.equal(result.status(),200);
  const predictionBody=await result.json();
  check('Real RGB upload and protected model reconstruction',{response_keys:Object.keys(predictionBody)});
  await page.waitForTimeout(6000); await screenshot('upload-workspace.png');
  const text=await page.locator('body').innerText();
  await writeFile(path.join(root,'outputs/browser-verification/viewer-text.txt'),text);
  assert(report.requests.some(r=>r.path.includes('classif')&&r.status===200),'Real classification response missing');
  assert(report.requests.some(r=>r.path.startsWith('/results/')&&r.status===200),'Generated products missing');
  check('Real six-category classification and generated result retrieval');
  const bad=await api('POST','/api/predict'); assert.equal(bad.status,422); check('Missing upload rejected with 422');
  await page.setViewportSize({width:390,height:844}); await page.waitForTimeout(600); await screenshot('mobile-workspace.png');
  const dimensions=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
  assert(dimensions.scroll<=dimensions.width+2,'Mobile horizontal overflow'); check('Mobile layout width',dimensions);
  assert.equal(report.page_errors.length,0); assert.equal(report.console_errors.length,0);
  check('No browser errors'); report.passed=true;
} catch(err) { report.passed=false;report.failure=String(err); console.error(err); await screenshot('verification-failure.png');process.exitCode=1; }
finally { await writeFile(path.join(evidenceDir,'browser-verification.json'),JSON.stringify(report,null,2)+'\n'); child.stdin.end(); child.kill(); if(!connected) await browser.close(); else await browser.close(); }
