// Real local browser + inference checks. Output contains screenshots and request/scale evidence.
// node scripts/verify_image_imports.mjs <agent-browser-executable> [downloaded-geotiff]
// First run scripts/data/create_import_test_fixtures.py with the project Python environment.
import {execFileSync} from 'node:child_process';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import path from 'node:path';
const binary=process.argv[2] ?? 'agent-browser';
const out=path.resolve('outputs/runtime/import-review');mkdirSync(out,{recursive:true});
function call(...args) {
 const input=args[0]==='eval'?args[1]:undefined;
 const raw=execFileSync(binary,['--session','import-review','--json',...(input?['eval','--stdin']:args)],{input,encoding:'utf8',windowsHide:true,timeout:45000,maxBuffer:12*1024*1024});
 const result=JSON.parse(raw);if(!result.success)throw new Error(result.error);return result.data;
}
const evaluate=code=>call('eval',code).result;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function waitFor(code,seconds=30){const until=Date.now()+seconds*1000;while(Date.now()<until){if(evaluate(code))return;await sleep(1500);}throw new Error('Timed out: '+code);}
const screenshot=name=>call('screenshot',path.join(out,name+'.png'));
const save=(name,data)=>writeFileSync(path.join(out,name+'.json'),JSON.stringify(data,null,2));
const report={checks:[],scenes:[]};
const check=(name,detail=true)=>{report.checks.push({name,detail});console.log('PASS',name);save('verification',report);};
const fixture=name=>path.join(out,name);
const original=process.argv[3] ?? path.join(process.env.USERPROFILE ?? process.env.HOME,'Downloads','joshimath_s2cloudless_2024.tif');
let ws,sequence=0,pageSession;
const pending=new Map();
async function protocol(method,params={},sessionId){
 const id=++sequence;
 return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{pending.delete(id);reject(new Error('Protocol timed out: '+method));},15000);pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});
}
async function connect(){
 ws=new WebSocket(call('get','cdp-url').cdpUrl);
 await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
 ws.onmessage=event=>{const message=JSON.parse(event.data);const request=pending.get(message.id);if(request){pending.delete(message.id);clearTimeout(request.timer);message.error?request.reject(new Error(message.error.message)):request.resolve(message.result);}};
 const targets=await protocol('Target.getTargets');
 const target=targets.targetInfos.find(t=>t.type==='page'&&t.url.startsWith('http://localhost:3000'));
 pageSession=(await protocol('Target.attachToTarget',{targetId:target.targetId,flatten:true})).sessionId;
 await protocol('Browser.grantPermissions',{origin:'http://localhost:3000',permissions:['clipboardReadWrite','clipboardSanitizedWrite']});
}
async function drop(files,preview=false){
 const rect=evaluate("document.querySelector('.terrain-canvas').getBoundingClientRect().toJSON()");
 const params={x:rect.x+rect.width*.5,y:rect.y+rect.height*.45,data:{items:[],files:files.map(f=>path.resolve(f)),dragOperationsMask:1}};
 await protocol('Input.dispatchDragEvent',{type:'dragEnter',...params},pageSession);
 await protocol('Input.dispatchDragEvent',{type:'dragOver',...params},pageSession);
 if(preview){await waitFor("!!document.querySelector('.image-drop-overlay')");screenshot('02-drop-preview');}
 await protocol('Input.dispatchDragEvent',{type:'drop',...params},pageSession);
}
async function ready(label){
 await waitFor(`document.querySelector('.source-card')?.innerText.includes(${JSON.stringify(label)}) && document.querySelector('.scene-label')?.textContent.includes('INTERACTIVE MESH READY') && !document.querySelector('.scene-reveal-card')`,240);
 const scene=evaluate("({source:document.querySelector('.source-card strong').innerText,scale:JSON.parse(document.querySelector('.terrain-canvas').dataset.terrainScale),mesh:JSON.parse(document.querySelector('.terrain-canvas').dataset.terrainMesh),exaggeration:Number(document.querySelector('.terrain-canvas').dataset.displayExaggeration),ruler:{...document.querySelector('.scene-scale-ruler').dataset},readout:document.querySelector('.scene-scale').innerText})");
 report.scenes.push(scene);save('verification',report);return scene;
}
async function copyPng(){
 const bytes=readFileSync(fixture('urban-import.png')).toString('base64');
 evaluate(`(async()=>{const bytes=Uint8Array.from(atob(${JSON.stringify(bytes)}),c=>c.charCodeAt(0));await navigator.clipboard.write([new ClipboardItem({'image/png':new Blob([bytes],{type:'image/png'})})]);return true;})()`);
}
try{
 call('open','http://localhost:3000');call('set','viewport','1600','1000');
 await waitFor("document.querySelector('.run-strip strong')?.textContent==='ENGINE READY'");
 await connect();
 evaluate("window.__imports=[];const originalFetch=window.fetch;window.fetch=async function(url,options){const response=await originalFetch.apply(this,arguments);if(String(url).includes('/api/predict')){const payload=await response.clone().json();window.__imports.push({filename:options.body.get('image').name,auto:options.body.get('auto_dem'),conditional:options.body.get('auto_dem_if_georeferenced'),status:response.status,payload});}return response;};true");
 await drop([fixture('unsupported.txt')],true);
 assert.match(evaluate("document.querySelector('.image-import-feedback').innerText"),/unsupported/);
 assert.equal(evaluate('window.__imports.length'),0);
 check('Real file drop rejects unsupported data before calling inference');
 call('click','button[aria-label="Dismiss import message"]');
 await drop([fixture('empty.png')]);assert.match(evaluate("document.querySelector('.image-import-feedback').innerText"),/empty/);
 check('Empty files show an actionable error');call('press','Escape');
 await drop([fixture('urban-import.png'),fixture('urban-import.jpg'),fixture('unsupported.txt')]);
 assert.equal(evaluate("document.querySelectorAll('.image-import-choices button').length"),2);
 check('Multiple dropped files offer an explicit scene choice, preserving valid files');
 screenshot('03-multiple-files');call('click','.image-import-choices button:first-child');
 await drop([fixture('urban-import.jpg')]);
 assert.match(evaluate("document.querySelector('.image-import-feedback').innerText"),/still processing/);
 call('press','Escape');
 const png=await ready('urban-import.png');
 assert.equal(png.scale.metric,false);assert.ok(png.scale.verticalScale<=.36);assert.equal(png.exaggeration,1.7);assert.equal(png.ruler.unit,'px');assert.equal(png.mesh.boundaryTriangles,0);
 check('Dropped PNG runs real inference with restored urban scaling and a numbered pixel ruler',png);
 assert.equal(evaluate('window.__imports.length'),1);check('Dropping during inference does not start a second request');
 screenshot('04-png-import');
 await copyPng();call('click','.segmented button:first-child');call('press','Control+V');
 await ready('image.png');assert.equal(evaluate('window.__imports.length'),2);
 check('Real Ctrl+V clipboard image reaches inference and the 3D viewer');
 await copyPng();call('click','.paste-image-button');await waitFor('window.__imports.length===3',240);await ready('clipboard-');
 check('Paste image button reads a real clipboard PNG and creates a scene');
 evaluate("navigator.clipboard.writeText('ordinary text that is not an image')");call('click','.paste-image-button');
 await waitFor("document.querySelector('.image-import-feedback')?.innerText.includes('No image')");
 check('Text-only clipboard gives clear feedback');call('press','Escape');
 evaluate("const field=document.createElement('textarea');field.id='paste-text-check';document.body.append(field);field.focus();true");
 call('press','Control+V');assert.equal(evaluate("document.querySelector('#paste-text-check').value"),'ordinary text that is not an image');
 evaluate("document.querySelector('#paste-text-check').remove();true");check('Normal text paste remains available in text fields');
 for(const suffix of ['jpg','webp','bmp','jp2','tif']){
  call('upload','input[aria-label="Import images"]',fixture('urban-import.'+suffix));
  const scene=await ready('urban-import.'+suffix);assert.equal(scene.scale.metric,false);assert.ok(scene.scale.verticalScale<=.36);assert.equal(scene.mesh.boundaryTriangles,0);
  check(`File picker ${suffix.toUpperCase()} -> real API -> mesh`);
 }
 call('click','button[aria-label="Load Urban demonstration"]');const urban=await ready('Copenhagen');
 assert.equal(urban.scale.verticalScale,.36);assert.equal(urban.exaggeration,1.7);check('Bundled Urban restores original 0.36 base scale and 1.7 display multiplier');screenshot('05-urban-final');
 const widths=[320,360,390,600,720,820,1040,1280,1600,1920];
 for(const width of widths){call('set','viewport',String(width),'1000');await sleep(200);const layout=evaluate("({overflow:document.documentElement.scrollWidth>innerWidth,elements:[...document.querySelectorAll('.image-import-actions,.scene-scale,.viewport-toolbar')].map(el=>{const r=el.getBoundingClientRect();return {name:el.className,left:r.left,right:r.right,bottom:r.bottom};})})");assert.equal(layout.overflow,false);for(const rect of layout.elements){assert.ok(rect.left>=0&&rect.right<=width+1&&rect.bottom<=1000,JSON.stringify({width,rect}));}}
 check('Import actions, ruler and toolbar fit 10 viewport widths from 320 to 1920 px');call('set','viewport','390','844');screenshot('06-mobile');call('set','viewport','1600','1000');
 call('click','.fullscreen-button');await sleep(500);await drop([fixture('unsupported.txt')]);
 assert.equal(evaluate("document.fullscreenElement?.contains(document.querySelector('.image-import-feedback')) ?? document.querySelector('.viewport').dataset.fullscreenKind==='window'"),true);
 check('Drop feedback remains visible in fullscreen');call('click','button[aria-label="Dismiss import message"]');call('click','.fullscreen-button');
 // Final source is the user's original downloaded GeoTIFF, with intact coordinates.
 await drop([original],true);const mountain=await ready('joshimath_s2cloudless_2024.tif');
 assert.equal(mountain.scale.metric,true);assert.equal(mountain.ruler.unit,'m');assert.ok(mountain.mesh.maximumY-mountain.mesh.minimumY>15);assert.ok(mountain.mesh.boundaryTriangles>0);
 const request=evaluate('window.__imports.at(-1)');assert.equal(request.auto,'true');assert.ok(request.payload.absolute_dsm_url);
 check('Original downloaded Joshimath GeoTIFF dropped unchanged -> public terrain -> physical 3D elevations and metre ruler',mountain);
 report.finalProducts=request.payload;screenshot('07-joshimath-final');
 const start=evaluate("Number(document.querySelector('.scene-scale-ruler').dataset.length)");
 call('click','.terrain-camera-tools button:nth-child(2)');await sleep(600);
 assert.ok(evaluate("Number(document.querySelector('.scene-scale-ruler').dataset.length)>0"));
 call('click','.terrain-camera-tools button:first-child');
 check('Numbered scale recalculates after camera changes',{initial:start,after:evaluate("Number(document.querySelector('.scene-scale-ruler').dataset.length)")});
 report.browserErrors=call('errors');assert.deepEqual(report.browserErrors.errors,[]);check('No browser runtime errors');
 report.requests=evaluate('window.__imports.map(({payload,...request})=>request)');save('verification',report);
 console.log('Verification complete:',report.checks.length,'checks');
}catch(error){report.error=error.stack;save('verification',report);screenshot('failure');throw error;}
finally{ws?.close();call('close');}
