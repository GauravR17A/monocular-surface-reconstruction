// Actual import -> API -> browser regression check. Leaves the local servers running.
// node scripts/verify_terrain_presentation.mjs <agent-browser.exe> <original.tif> <another-hilly.tif>
import {execFileSync} from 'node:child_process';
import {mkdirSync,writeFileSync,readFileSync,readdirSync} from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const [binary,source,other]=process.argv.slice(2);
assert.ok(binary&&source&&other,'Pass browser executable, original TIFF and another hilly TIFF');
const output=path.resolve('outputs/runtime/video-match-review');mkdirSync(output,{recursive:true});
const resume=process.argv.includes('--resume');
const finalOnly=process.argv.includes('--final-only')||resume;
const report={source,checks:finalOnly?JSON.parse(readFileSync(path.join(output,'final-verification.json'),'utf8')).checks:[]};
function call(...args){
  const input=args[0]==='eval'?args[1]:undefined;
  let raw;
  try{raw=execFileSync(binary,['--session','terrain-presentation-review','--json',...(input?['eval','--stdin']:args)],{input,encoding:'utf8',windowsHide:true,timeout:45000,maxBuffer:12*1024*1024});}
  catch(error){if(error.stdout?.trim().startsWith('{'))raw=error.stdout;else throw error;}
  const r=JSON.parse(raw);if(!r.success)throw new Error(`${args[0]}: ${r.error}`);return r.data;
}
const evaluate=code=>call('eval',code).result;
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function waitFor(code,seconds=60){const end=Date.now()+seconds*1000;while(Date.now()<end){if(evaluate(code))return;await sleep(1500);}throw new Error('Timed out: '+code);}
const mesh=()=>evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.terrainMesh)");
const check=(name,detail=true)=>{report.checks=report.checks.filter(check=>check.name!==name);report.checks.push({name,detail});console.log('PASS',name);};
const capture=name=>call('screenshot',path.join(output,name+'.png'));
async function importImage(file){call('upload','input[aria-label="Import images"]',path.resolve(file));await waitFor(`document.querySelector('.source-card')?.innerText.includes(${JSON.stringify(path.basename(file))}) && document.querySelector('.scene-label')?.innerText.includes('INTERACTIVE MESH READY')`,240);}
const clickPoint=(x,y)=>{call('mouse','move',String(x),String(y));call('mouse','down');call('mouse','up');};
try{
  if(!resume){call('open','http://localhost:3000');call('set','viewport','1600','1000');}
  else if(evaluate("!!document.querySelector('.terrain-profile-heading')"))call('click','.terrain-profile-heading button:last-child');
  await waitFor("document.querySelector('.run-strip strong')?.textContent==='ENGINE READY'");
  for(const demo of finalOnly?[]:['Urban','Sparse','Forest']){
    call('click',`button[aria-label="Load ${demo} demonstration"]`);
    await waitFor("document.querySelector('.scene-label')?.innerText.includes('INTERACTIVE MESH READY')");
    assert.equal(mesh().terrainPresentation,false);
    assert.equal(evaluate("!!document.querySelector('.terrain-preview-switch')"),false);
    if(demo==='Urban'){
      const scale=evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.terrainScale)");
      assert.equal(scale.verticalScale,.36);assert.equal(evaluate("+document.querySelector('.terrain-canvas').dataset.displayExaggeration"),1.7);
      call('click','.fullscreen-button');capture('urban-preserved');call('click','.fullscreen-button');
    }
    check(`${demo} object scene bypasses terrain smoothing`,mesh());
  }
  if(!finalOnly){
  const regionFiles=process.argv.includes('--all-regions')
    ? readdirSync(path.dirname(other)).filter(name=>/\.tiff?$/i.test(name)).map(name=>path.join(path.dirname(other),name)) : [other];
  for(const regionalFile of regionFiles){
  await importImage(regionalFile);
  assert.equal(mesh().terrainPresentation,true);assert.ok(mesh().maximumDisplayAdjustmentM>0);
  const regionalScale=evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.terrainScale)");
  const displayExaggeration=evaluate("+document.querySelector('.terrain-canvas').dataset.displayExaggeration");
  assert.ok(displayExaggeration>=1&&displayExaggeration<=1.5,'Imported terrain must not be automatically stretched into spikes');
  assert.ok(Number.isFinite(mesh().minimumY)&&Number.isFinite(mesh().maximumY));
  check(`Automatic terrain reconstruction: ${path.basename(regionalFile)}`,{...mesh(),scale:regionalScale,displayExaggeration,
    products:evaluate("[...document.querySelectorAll('.export-card a')].map(a=>({name:a.innerText,url:a.href}))")});
  capture(path.basename(regionalFile).replace(/\.tiff?$/i,'')+'-smoothed');
  }
  const dsmUrl=evaluate("document.querySelector('.export-card a[href$=\"/dsm_absolute_m.tif\"]').href");
  const terrainJob=dsmUrl.split('/results/')[1].split('/')[0];
  call('upload','.source-card > input[accept=".tif,.tiff,image/tiff"]',path.resolve('outputs/web_jobs',terrainJob,'dsm_absolute_m.tif'));
  await waitFor("document.querySelector('.source-card strong')?.textContent==='dsm_absolute_m.tif' && document.querySelector('.scene-label')?.innerText.includes('INTERACTIVE MESH READY')");
  assert.equal(mesh().terrainPresentation,true);check('Existing metric elevation GeoTIFF also receives terrain processing',mesh());capture('height-raster-smoothed');
  }
  // Final end-to-end test is deliberately the user's original downloaded TIFF.
  if(!resume)await importImage(source);
  assert.ok(evaluate(`document.querySelector('.source-card')?.innerText.includes(${JSON.stringify(path.basename(source))})`));
  if(!evaluate("document.querySelector('.fullscreen-button').getAttribute('aria-pressed')==='true'"))call('click','.fullscreen-button');await sleep(400);
  call('click','.terrain-camera-tools button:last-child');await sleep(600);
  report.mesh=mesh();assert.equal(report.mesh.vertices,1048576);assert.ok(report.mesh.maximumDisplayAdjustmentM>3);
  report.scale=evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.terrainScale)");
  report.products=evaluate("[...document.querySelectorAll('.export-card a')].map(a=>({name:a.innerText,url:a.href}))");
  const camera=evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.navigation)");
  call('click','.terrain-preview-switch button:nth-child(2)');await sleep(400);
  assert.equal(mesh().maximumDisplayAdjustmentM,0);
  assert.deepEqual(evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.navigation)"),camera);
  capture('final-source');
  call('click','.terrain-preview-switch button:first-child');await sleep(400);
  assert.equal(evaluate("document.querySelector('.terrain-canvas').dataset.terrainPresentation"),'smooth');
  assert.deepEqual(evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.navigation)"),camera);
  check('Smooth/Source switches geometry without changing the camera or source range',mesh());
  for(const level of [1,2,3]){
    call('click','input[aria-label="Surface smoothness"]');call('press','Home');
    for(let i=1;i<level;i++)call('press','ArrowRight');await sleep(150);
    assert.equal(mesh().smoothingLevel,level);
    assert.deepEqual(evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.navigation)"),camera);
    assert.deepEqual(evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.terrainScale)"),report.scale);
  }
  call('click','input[aria-label="Surface smoothness"]');call('press','End');
  check('All three smoothing strengths preserve the camera and source elevation range');
  call('click','.segmented > button:nth-of-type(2)');clickPoint(800,540);
  await waitFor("!!document.querySelector('.inspect-popover')");
  report.inspection=evaluate("({text:document.querySelector('.inspect-popover').innerText,u:+document.querySelector('.inspect-popover').dataset.sampleU,v:+document.querySelector('.inspect-popover').dataset.sampleV})");
  assert.match(report.inspection.text,/Terrain reference/);capture('final-inspection');
  call('click','.segmented > button:last-child');clickPoint(600,490);clickPoint(1050,670);
  await waitFor("!!document.querySelector('.terrain-profile-heading a')");
  const csvUrl=evaluate("document.querySelector('.terrain-profile-heading a').href");
  writeFileSync(path.join(output,'final-profile.csv'),decodeURIComponent(csvUrl.slice(csvUrl.indexOf(',')+1)));
  call('click','input[aria-label="Vertical exaggeration"]');call('press','End');await sleep(300);
  assert.equal(evaluate("document.querySelector('.terrain-profile-heading a').href"),csvUrl);
  check('Source profile CSV stays identical when display relief changes');
  call('click','.terrain-camera-tools button:last-child');call('click','.terrain-profile-heading button:last-child');
  call('click','.fullscreen-button');
  for(const index of [2,3,4,5,1]){
    call('check',`.layer-stack label:nth-of-type(${index}) input`);await sleep(200);
    assert.deepEqual(evaluate("JSON.parse(document.querySelector('.terrain-canvas').dataset.terrainScale)"),report.scale);
  }
  check('All terrain visual layers retain their source scale');
  call('click','.fullscreen-button');
  call('click','.terrain-camera-tools button:first-child');await sleep(700);
  report.animation=evaluate("new Promise(resolve=>{const frames=[];let last=performance.now();const end=last+2000;function tick(now){frames.push(now-last);last=now;if(now<end)requestAnimationFrame(tick);else{frames.sort((a,b)=>a-b);resolve({frames:frames.length,medianMs:frames[Math.floor(frames.length/2)],p95Ms:frames[Math.floor(frames.length*.95)]});}}requestAnimationFrame(tick);})");
  report.mesh=mesh();capture('final-joshimath');report.errors=call('errors');assert.equal(report.errors.errors.length,0);
  check('Original downloaded TIFF completes final browser test without errors',report.animation);
}finally{writeFileSync(path.join(output,'final-verification.json'),JSON.stringify(report,null,2));if(!process.argv.includes('--keep-open'))call('close');}
