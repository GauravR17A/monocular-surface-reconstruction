import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { createSelectionRegions, sampleLocator } from '../app/selection-region.ts';
import { buildingSelectionEdges, createSelectionGlow, drapeSelectionEdges } from '../app/selection-glow.ts';

function regions(values, width, height, max = 512) {
  return createSelectionRegions(width, height, (u, v) => values[Math.min(height - 1, Math.floor(v * height)) * width + Math.min(width - 1, Math.floor(u * width))], max);
}

test('only the connected selected patch is outlined, not every pixel of its class', () => {
  const source = new Uint8Array([1, 1, 0, 1, 1, 1, 0, 1]);
  const before = source.slice();
  const index = regions(source, 4, 2), first = index.at(.1, .1);
  assert.equal(index.at(.4, .9), first);
  assert.notEqual(index.at(.9, .1), first);
  const boundary = index.boundary(first);
  for (let i = 0; i < boundary.length; i += 2) assert.ok(boundary[i] <= .5);
  assert.equal(boundary.length / 4, 8);
  assert.strictEqual(index.boundary(first), boundary, 'same selection reuses cached contour');
  assert.deepEqual(source, before);
});

test('region contours keep holes, ignore unknowns and do not bridge diagonal neighbours', () => {
  const index = regions([1,1,1,1,255,1,1,1,1], 3, 3);
  assert.equal(index.at(.5,.5), null);
  assert.equal(index.boundary(index.at(0,0)).length / 4, 16);
  const diagonals = regions([5,255,255,5],2,2);
  assert.notEqual(diagonals.at(0,0), diagonals.at(1,1));
  for (const uv of [[NaN,0],[-.1,0],[1.1,0],[0,Infinity]]) assert.equal(index.at(...uv),null);
  assert.equal(index.boundary(-1).length,0);
});

test('bounded display grid does not outline a neighbouring class for a lost tiny feature', () => {
  const index = createSelectionRegions(3072,1536,()=>0);
  assert.equal(index.width,512); assert.equal(index.height,256);
  assert.equal(index.at(.5,.5,5),null);
  assert.ok(index.at(.5,.5,0));
  assert.throws(()=>createSelectionRegions(0,2,()=>0));
});

test('unknown point locator remains local and inside the image', () => {
  const points = sampleLocator(0,1,.01,.01);
  assert.equal(points.length,160);
  for (let i=0;i<points.length;i+=2) {
    assert.ok(points[i]>=0 && points[i]<=.011);
    assert.ok(points[i+1]>=.989 && points[i+1]<=1);
  }
});

test('draped outlines follow surface elevations without modifying the raster or inventing nodata', () => {
  const uv = new Float32Array([0,0,1,0,0,1,1,1]);
  const before=uv.slice();
  const positions=drapeSelectionEdges(uv,100,200,(u,v)=>v===1?null:2+u*3,.1);
  assert.equal(positions.length,6);
  assert.deepEqual(Array.from(positions).filter((_,i)=>i%3!==1),[-50,-100,50,-100]);
  assert.ok(Math.abs(positions[1]-2.1)<1e-6); assert.ok(Math.abs(positions[4]-5.1)<1e-6);
  assert.deepEqual(uv,before);
});

test('building edges use exact transformed geometry, preserving height and geometry buffers', () => {
  const mesh=new THREE.Mesh(new THREE.BoxGeometry(2,4,3),new THREE.MeshBasicMaterial());
  mesh.position.set(10,7,20); mesh.rotation.y=Math.PI/4;
  const parent=new THREE.Group(); parent.scale.y=3; parent.add(mesh);
  const before=mesh.geometry.attributes.position.array.slice(), points=buildingSelectionEdges(mesh);
  assert.equal(points.length,12*6);
  const ys=Array.from(points).filter((_,i)=>i%3===1);
  assert.equal(Math.min(...ys),5); assert.equal(Math.max(...ys),9, 'exaggeration is not baked into measured geometry');
  assert.deepEqual(mesh.geometry.attributes.position.array,before);
  mesh.geometry.dispose(); mesh.material.dispose();
});

test('glow has one selection, soft pixel-width edges, depth occlusion and no picking interference', () => {
  const glow=createSelectionGlow();
  assert.equal(glow.group.visible,false);
  const points=new Float32Array([0,0,0,1,2,1]); glow.set(points);
  assert.equal(glow.group.children.length,3);
  const oldGeometry=glow.group.children[0].geometry;
  let disposed=0; oldGeometry.addEventListener('dispose',()=>disposed++);
  glow.update(.2,1.7,true);
  assert.equal(glow.group.scale.y,1.7);
  for (const line of glow.group.children) {
    assert.equal(line.material.depthTest,true); assert.equal(line.material.depthWrite,false);
    assert.equal(line.material.worldUnits,false); assert.equal(line.material.toneMapped,false);
    const intersections=[]; line.raycast({},intersections); assert.equal(intersections.length,0);
  }
  glow.set(new Float32Array([3,3,3,4,4,4])); assert.equal(disposed,1);
  assert.equal(glow.group.children.length,3,'no accumulating old outlines');
  glow.clear(); assert.equal(glow.group.visible,false);
  glow.dispose(); assert.equal(glow.group.children.length,0);
});
