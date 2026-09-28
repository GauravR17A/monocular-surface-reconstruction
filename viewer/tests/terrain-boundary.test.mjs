import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {createTerrainBoundary} from '../app/terrain-boundary.ts';

test('terrain edges retain exact surface vertices and descend only to the scene minimum',()=>{
  const source = new THREE.Float32BufferAttribute([-1,2,-1,1,4,-1,-1,1,1,1,3,1],3);
  const original=source.array.slice();
  const group=createTerrainBoundary(source,new Uint8Array([1,1,1,1]),1,1);
  const walls=group.children[0].geometry.attributes.position;
  assert.equal(walls.count,24);
  assert.deepEqual(source.array,original);
  for(let i=0;i<walls.count;i++) {
    const x=walls.getX(i),y=walls.getY(i),z=walls.getZ(i);
    assert.ok(y===0 || Array.from({length:source.count},(_,j)=>j).some(j=>source.getX(j)===x && source.getY(j)===y && source.getZ(j)===z));
  }
  assert.equal(createTerrainBoundary(source,new Uint8Array([0,1,1,1]),1,1).children[0].geometry.attributes.position.count,12);
});
