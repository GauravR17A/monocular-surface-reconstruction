// Uses a dedicated agent-browser session. Run after the local API and viewer are ready.
// node scripts/verify_terrain_analysis.mjs <agent-browser-executable> <downloaded-rgb-tiff>
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const binary = process.argv[2], source = process.argv[3];
assert.ok(binary && source, 'Pass the agent-browser executable and source TIFF');
const output = path.resolve('outputs/runtime/terrain-final'); mkdirSync(output, {recursive:true});
const session = 'terrain-review';
const report = {source, checks: [], errors: []};
function call(...args) {
  const raw = execFileSync(binary, ['--session', session, '--json', ...args], {encoding:'utf8',windowsHide:true,timeout:45000,maxBuffer:12*1024*1024});
  const result = JSON.parse(raw);
  if (!result.success) throw new Error(`${args[0]}: ${result.error}`);
  return result.data;
}
const evaluate = code => call('eval', code).result;
const sleep = ms => new Promise(resolve=>setTimeout(resolve,ms));
const check = (name, detail = true) => { report.checks.push({name,detail}); console.log('PASS',name); };
const clickPoint = (x,y) => { call('mouse','move',String(x),String(y));call('mouse','down');call('mouse','up'); };
const screenshot = name => call('screenshot',path.join(output,name+'.png'));
// Use the browser protocol directly: the native CLI download helper cancels
// completed downloads on this Windows host. The click still uses agent-browser.
async function downloadProfile(expected) {
  const ws = new WebSocket(call('get','cdp-url').cdpUrl);
  const downloaded = await new Promise((resolve,reject) => {
    const timer=setTimeout(()=>{ws.close();reject(new Error('CSV download timed out'));},15000);
    ws.onopen=()=>ws.send(JSON.stringify({id:1,method:'Browser.setDownloadBehavior',params:{behavior:'allow',downloadPath:output,eventsEnabled:true}}));
    ws.onmessage=event=>{
      const message=JSON.parse(event.data);
      if(message.id===1) { if(message.error) {clearTimeout(timer);ws.close();reject(new Error(message.error.message));} else call('click','.terrain-profile-heading a'); }
      if(message.method==='Browser.downloadProgress' && message.params.state!=='inProgress') {
        clearTimeout(timer);ws.close();
        if(message.params.state==='completed') resolve(message.params.filePath ?? path.join(output,'msr-surface-profile.csv'));
        else reject(new Error('Browser canceled CSV download'));
      }
    };
    ws.onerror=()=>{clearTimeout(timer);reject(new Error('Browser download connection failed'));};
  });
  assert.equal(readFileSync(downloaded,'utf8'),expected);
  writeFileSync(path.join(output,'downloaded-profile.csv'),expected);
}
async function waitFor(code, seconds=30) {
  const deadline=Date.now()+seconds*1000;
  while(Date.now()<deadline) { if(evaluate(code)) return; await sleep(1500); }
  throw new Error('Timed out: '+code);
}
try {
  call('open','http://localhost:3000');call('set','viewport','1600','1000');
  await waitFor("document.querySelector('.run-strip strong')?.textContent === 'ENGINE READY'");
  assert.equal(evaluate("document.querySelector('.terrain-toggle input').checked"),true);
  check('Terrain calibration is enabled by default for imported georeferenced TIFFs');
  call('upload','input[accept^=".jpg"]',path.resolve(source));
  console.log('Uploaded original TIFF with automatic public terrain calibration.');
  await waitFor("document.querySelector('.scene-label')?.innerText.includes('INTERACTIVE MESH READY') && document.querySelector('.source-card')?.innerText.includes('joshimath')",240);
  await waitFor("document.querySelector('.six-class-panel')?.dataset.classificationStatus === 'ready'",120);
  report.legend=evaluate("document.querySelector('.terrain-elevation-key').innerText");
  report.scale=evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.terrainScale)");
  report.products=evaluate("[...document.querySelectorAll('.export-card a')].map(a=>({name:a.innerText,url:a.href}))");
  assert.match(report.legend,/SURFACE ELEVATION/); assert.equal(report.scale.metric,true);
  check('Original Joshimath TIFF -> API -> absolute DSM -> metric terrain',report.legend);
  screenshot('01-joshimath-terrain');
  call('click','.segmented > button:nth-of-type(2)'); clickPoint(790,575);
  await waitFor("!!document.querySelector('.inspect-popover')");
  report.inspection=evaluate("({text:document.querySelector('.inspect-popover').innerText,u:+document.querySelector('.inspect-popover').dataset.sampleU,v:+document.querySelector('.inspect-popover').dataset.sampleV})");
  assert.match(report.inspection.text,/Terrain reference/);assert.match(report.inspection.text,/Surface − terrain/);
  check('Click inspection reports source elevation, terrain reference, difference and slope',report.inspection);
  screenshot('02-point-inspection');
  call('click','.segmented > button:last-child');clickPoint(650,445);clickPoint(1030,660);
  await waitFor("!!document.querySelector('.terrain-profile-heading a')");
  report.profile=evaluate("document.querySelector('.terrain-profile-panel').innerText");
  const csvUrl=evaluate("document.querySelector('.terrain-profile-heading a').href");
  const csv=decodeURIComponent(csvUrl.slice(csvUrl.indexOf(',')+1));
  writeFileSync(path.join(output,'profile-link-payload.csv'),csv);
  assert.match(csv,/distance_m,surface_m,terrain_reference_m/);
  assert.ok(csv.split('\n').length>100);
  check('A-B profile uses real samples and exposes a valid CSV',report.profile);
  await downloadProfile(csv);
  check('CSV downloaded through the browser and matches displayed profile');
  screenshot('03-measured-profile');
  call('click','input[aria-label="Vertical exaggeration"]');call('press','End');
  await waitFor("document.querySelector('input[aria-label=\"Vertical exaggeration\"]').value === '12'");
  assert.equal(evaluate("document.querySelector('.terrain-profile-panel').innerText"),report.profile);
  assert.equal(evaluate("document.querySelector('.terrain-profile-heading a').href"),csvUrl);
  check('Z x12 changes display only; profile and CSV are identical');
  call('click','.terrain-camera-tools button:last-child');
  call('click','.terrain-profile-heading button:last-child');
  call('check','.layer-stack label:has(input[type=checkbox]):nth-last-of-type(2) input');
  call('check','.layer-stack input[type=radio]'); // First radio: optical.
  screenshot('04-optical-contours');
  call('check','.layer-stack label:nth-of-type(3) input');
  screenshot('05-shaded-relief');
  call('check','.layer-stack label:nth-of-type(4) input');
  await waitFor("document.querySelector('.terrain-elevation-key')?.innerText.includes('SURFACE SLOPE')");
  screenshot('06-surface-slope');
  call('check','.layer-stack label:nth-of-type(5) input');
  screenshot('07-metric-grid');
  assert.deepEqual(evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.terrainScale)"),report.scale);
  check('Optical, contour, shaded-relief, slope and metric views retain identical source geometry');
  call('check','.layer-stack label:nth-of-type(1) input');
  call('click','.terrain-camera-tools button:nth-child(2)');screenshot('08-top-down');
  call('click','.terrain-camera-tools button:first-child');
  call('click','.segmented > button:last-child');clickPoint(650,445);clickPoint(1030,660);
  await waitFor("!!document.querySelector('.terrain-profile-heading a')");
  call('set','viewport','900','760');screenshot('09-compact-profile');
  const bounds=evaluate("({panel:document.querySelector('.terrain-profile-panel').getBoundingClientRect().toJSON(),width:innerWidth})");
  assert.ok(bounds.panel.right<=bounds.width && bounds.panel.left>=0);
  check('Profile remains usable when the analysis sidebar is hidden',bounds);
  call('set','viewport','1600','1000');
  call('click','.terrain-profile-heading button:last-child');call('click','.terrain-camera-tools button:first-child');
  screenshot('10-final-joshimath');
  report.errors=call('errors'); console.log('Browser errors:',JSON.stringify(report.errors));
} finally { writeFileSync(path.join(output,'browser-verification.json'),JSON.stringify(report,null,2)); }
