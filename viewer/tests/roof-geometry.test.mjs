import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { buildRoofDetail, supportedRoofHeight, insideRoof } from '../app/roof-geometry.ts';

const outline = [[.1,.1],[.9,.1],[.9,.9],[.1,.9]];
function source(fn=()=>10) {
  const values = new Float32Array(41*41);
  for(let y=0;y<41;y++)for(let x=0;x<41;x++)values[y*41+x]=fn(x,y);
  return {height:{values,stats:{width:41,height:41,units:'m'}},semantic:null,probability:null};
}
const build=(data,polygon=outline)=>buildRoofDetail(polygon,data,100,100,.4,2,10,1000);

test('constant roofs and isolated spikes keep the existing solid',()=>{
  assert.equal(build(source()),null);
  assert.equal(build(source((x,y)=>x===20&&y===20?100:10)),null);
});
test('a coherent rooftop block survives, with upwards faces and unchanged source values',()=>{
  const data=source((x,y)=>x>=17&&x<=25&&y>=17&&y<=25?14:10);
  const before=data.height.values.slice();
  const result=build(data); assert.ok(result);
  assert.equal(result.minimum,10); assert.equal(result.maximum,14);
  assert.deepEqual(data.height.values,before);
  const mesh=new THREE.Mesh(result.geometry,[new THREE.MeshBasicMaterial(),new THREE.MeshBasicMaterial()]);
  mesh.updateMatrixWorld();
  const ray=new THREE.Raycaster(new THREE.Vector3(2.5,100,2.5),new THREE.Vector3(0,-1,0));
  const hit=ray.intersectObject(mesh)[0];
  assert.ok(hit); assert.ok(hit.face.normal.y>0);
  assert.ok(Math.abs(hit.point.y-6.4)<.1,'raised block should be above the 4.8 scene-unit summary roof');
  assert.ok(Math.abs(hit.uv.x-.525)<1e-5 && Math.abs(hit.uv.y-.475)<1e-5);
  result.geometry.dispose(); mesh.material.forEach(m=>m.dispose());
});
test('unknown, non-building and missing labels do not create rooftop structures',()=>{
  const data=source((x)=>x>20?14:10);
  data.semantic={values:new Float32Array(41*41).fill(255),stats:data.height.stats};
  assert.equal(build(data),null);
  data.semantic.values.fill(2); assert.equal(build(data),null);
  data.semantic.values.fill(1); data.height.values.fill(NaN); assert.equal(build(data),null);
});
test('relative inputs and invalid footprints never become metre-scale roof detail',()=>{
  const data=source((x)=>x>20?14:10); data.height.stats.units='relative';
  assert.equal(build(data),null);
  data.height.stats.units='m'; assert.equal(build(data,[[-1,0],[1,0],[1,1]]),null);
});
test('low edge estimates cannot collapse the building base, while raised patches remain',()=>{
  const data=source((x,y)=>x<8||x>32||y<8||y>32?1:x>17&&x<26&&y>17&&y<26?14:10);
  const result=build(data); assert.ok(result); assert.equal(result.minimum,10);
  result.geometry.dispose();
});
test('concave footprints are not filled across the missing corner',()=>{
  const concave=[[.1,.1],[.9,.1],[.9,.4],[.4,.4],[.4,.9],[.1,.9]];
  const result=build(source((x)=>x>10&&x<15?14:10),concave); assert.ok(result);
  const {geometry}=result,position=geometry.attributes.position,index=geometry.index;
  for(let i=0;i<geometry.groups[0].count;i+=3){
    let u=0,v=0;
    for(let j=0;j<3;j++){ const k=index.getX(i+j); u+=position.getX(k)/100+.5; v+=position.getZ(k)/100+.5; }
    assert.ok(insideRoof(u/3,v/3,concave));
  }
  geometry.dispose();
});
test('height support rejects values from neighbouring vegetation',()=>{
  const data=source(()=>40);
  data.semantic={values:new Float32Array(41*41).fill(2),stats:data.height.stats};
  assert.equal(supportedRoofHeight(data,outline,.5,.5),null);
});
