import {test} from 'node:test';
import assert from 'node:assert/strict';
import {terrainMeshResolution,terrainPresentationSurface} from '../app/terrain-presentation.ts';
const raster=(values,width,height,stats={})=>({values:new Float32Array(values),stats:{width,height,units:'m',gsdX:30,gsdY:30,...stats}});

test('terrain uses source samples, while existing urban geometry stays at its previous resolution',()=>{
  assert.deepEqual(terrainMeshResolution(1024,1024,true),{segmentsX:1023,segmentsY:1023});
  assert.deepEqual(terrainMeshResolution(1024,1024,false),{segmentsX:512,segmentsY:512});
  for(const [w,h] of [[4096,4096],[4000,1000],[16000,32],[800,600]]){
    const {segmentsX:x,segmentsY:y}=terrainMeshResolution(w,h,true);
    assert.ok((x+1)*(y+1)<=1024*1024);assert.ok(Math.max(x,y)<2048);
    assert.ok(x<=w-1&&y<=h-1);
  }
});
test('flat terrain and tilted planes remain flat/planar, without modifying the source',()=>{
  for(const a of [raster(Array(49).fill(1500),7,7),raster(Array.from({length:49},(_,i)=>1000+(i%7)*20+Math.floor(i/7)*10),7,7)]){
    const original=a.values.slice();const result=terrainPresentationSurface(a);
    assert.deepEqual(a.values,original);assert.deepEqual(result.values,original);
  }
});
test('jagged variations are substantially reduced without modifying the source',()=>{
  const a=raster(Array.from({length:441},(_,i)=>1000+((i%21+Math.floor(i/21))%2 ? 20:-20)),21,21);
  const original=a.values.slice(),result=terrainPresentationSurface(a);
  assert.ok(Math.abs(result.values[220]-1000)<Math.abs(a.values[220]-1000)*.25);
  assert.ok(result.maximumAdjustment>0&&result.maximumAdjustment<=40);
  assert.ok(result.values.every(v=>v>=980&&v<=1020));
  assert.deepEqual(a.values,original);
});

test('stronger settings round small bumps while retaining the broad hill shape',()=>{
  const size=129;
  const hill=Array.from({length:size*size},(_,i)=>1000+800*Math.exp(-((i%size-64)**2+(Math.floor(i/size)-64)**2)/(2*32**2)));
  const noisy=raster(hill.map((v,i)=>v+25*Math.cos(i%size*Math.PI/8)*Math.cos(Math.floor(i/size)*Math.PI/8)),size,size);
  let previousError=Infinity;
  for(const strength of [1,2,3]){
    const result=terrainPresentationSurface(noisy,null,strength);
    const smoothHill=terrainPresentationSurface(raster(hill,size,size),null,strength);
    let error=0,count=0;
    for(let row=20;row<size-20;row++)for(let col=20;col<size-20;col++){
      const i=row*size+col;error+=(result.values[i]-smoothHill.values[i])**2;count++;
    }
    const rms=Math.sqrt(error/count);
    // Broad hill height is retained; smoothing removes the superimposed ripples.
    assert.ok(rms<previousError,`${strength}: ${rms} >= ${previousError}`);previousError=rms;
    assert.ok(Math.abs(result.values[64*size+64]-hill[64*size+64])<800*.1);
  }
});

test('continuous reconstruction has no overshoot and safely handles out-of-range strength',()=>{
  const a=raster(Array.from({length:81*81},(_,i)=>1000+80*Math.cos(i%81/2)+80*Math.cos(Math.floor(i/81)/2)),81,81);
  const minimum=Math.min(...a.values),maximum=Math.max(...a.values);
  for(const level of [1,2,3]){
    const result=terrainPresentationSurface(a,null,level);
    assert.ok(result.values.every(Number.isFinite));
    assert.ok(result.values.every(v=>v>=minimum-.001&&v<=maximum+.001));
  }
  for(const [value,expected] of [[NaN,3],[Infinity,3],[-100,1],[100,3]]){
    assert.deepEqual(terrainPresentationSurface(a,null,value).values,terrainPresentationSurface(a,null,expected).values);
  }
});

test('smooth ground does not retain pointed source residuals as strength increases',()=>{
  const size=101;
  const ground=raster(Array.from({length:size*size},(_,i)=>1000+300*Math.exp(-((i%size-50)**2+(Math.floor(i/size)-50)**2)/18)),size,size);
  let previousCurvature=Infinity;
  for(const strength of [1,2,3]){
    const {values}=terrainPresentationSurface(ground,ground,strength);
    let curvature=0;
    for(let row=30;row<70;row++)for(let col=30;col<70;col++){
      const i=row*size+col;curvature+=(4*values[i]-values[i-1]-values[i+1]-values[i-size]-values[i+size])**2;
    }
    assert.ok(curvature<previousCurvature);previousCurvature=curvature;
    assert.ok(values[50*size+50]>1000&&values[50*size+50]<1300);
  }
});
test('smooth ground retains object-height residuals and source arrays exactly',()=>{
  const ground=raster(Array.from({length:441},(_,i)=>1000+((i%21+Math.floor(i/21))%2 ? 15:-15)),21,21);
  const surface=raster(ground.values.map((v,i)=>v+(i%21>8&&i%21<13?25:0)),21,21);
  const originalGround=ground.values.slice(),originalSurface=surface.values.slice();
  const smoothGround=terrainPresentationSurface(ground);
  const smoothSurface=terrainPresentationSurface(surface,ground);
  for(let i=0;i<surface.values.length;i++) assert.ok(Math.abs((smoothSurface.values[i]-smoothGround.values[i])-(surface.values[i]-ground.values[i]))<.001);
  assert.deepEqual(ground.values,originalGround);assert.deepEqual(surface.values,originalSurface);
});
test('fine-resolution DSMs without a ground reference keep small structures intact',()=>{
  const a=raster([0,0,0,0,7,0,0,0,0],3,3,{gsdX:1,gsdY:1});
  assert.equal(terrainPresentationSurface(a).values,a.values);
});
test('sharp breaks, nodata and borders stay intact',()=>{
  const a=raster(Array.from({length:49},(_,i)=>i%7<3 ? 1000:1100),7,7,{gsdX:10,gsdY:10});
  a.values[24]=NaN;
  const result=terrainPresentationSurface(a);
  assert.deepEqual(result.values,a.values);
});
test('unknown scale and dimensionless imagery never receive terrain processing',()=>{
  for(const stats of [{units:'relative'},{gsdX:null},{gsdY:0},{gsdX:Infinity}]){
    const a=raster([0,2,0,2,0,2,0,2,0],3,3,stats);
    assert.equal(terrainPresentationSurface(a).values,a.values);
  }
});
