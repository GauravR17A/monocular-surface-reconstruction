import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { applyFacadeUVs, facadeLayout } from '../app/building-facades.ts';

test('facade tiling stays bounded for small, tall and wide buildings', () => {
  assert.equal(facadeLayout(0, .15, 1, 1).rows, 1);
  assert.equal(facadeLayout(7, 2.52, 12, 8).rows, 2);
  assert.equal(facadeLayout(1000, 100, 20, 20).rows, 48);
  assert.ok(facadeLayout(7, 2.52, 100, 10).tileWidth >= 100 / 14);
});

test('wall detailing preserves all geometry and optical roof UVs', () => {
  const geometry = new THREE.BoxGeometry(8, 6, 4);
  const originalPositions = Array.from(geometry.attributes.position.array);
  const originalUV = Array.from(geometry.attributes.uv.array);
  applyFacadeUVs(geometry, 1, 1, -3, 6, 2, 4);
  assert.deepEqual(Array.from(geometry.attributes.position.array), originalPositions);
  const { normal, position, uv, color } = geometry.attributes;
  for (let i = 0; i < position.count; i++) {
    if (Math.abs(normal.getY(i)) > .5) {
      assert.equal(uv.getX(i), originalUV[i * 2]);
      assert.equal(uv.getY(i), originalUV[i * 2 + 1]);
    } else {
      assert.equal(uv.getY(i), position.getY(i) < 0 ? 0 : 2);
      assert.ok(color.getX(i) >= .75 && color.getX(i) <= 1);
    }
  }
  geometry.dispose();
});

test('diagonal extruded walls keep noncollapsed UVs under rectangular scene scaling', () => {
  const shape = new THREE.Shape();
  shape.moveTo(0, 0); shape.lineTo(.4, .25); shape.lineTo(.2, .7); shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, {depth: 4, bevelEnabled: false});
  geometry.rotateX(-Math.PI / 2);
  applyFacadeUVs(geometry, 112, 68, 0, 4, 2, 3);
  const { normal, uv } = geometry.attributes;
  let sideTriangles = 0;
  for (let i = 0; i < normal.count; i += 3) {
    if (Math.abs(normal.getY(i)) > .5) continue;
    const area = (uv.getX(i+1)-uv.getX(i)) * (uv.getY(i+2)-uv.getY(i))
      - (uv.getX(i+2)-uv.getX(i)) * (uv.getY(i+1)-uv.getY(i));
    assert.ok(Math.abs(area) > .01, 'wall UV triangle should have usable area');
    sideTriangles++;
  }
  assert.equal(sideTriangles, 6);
  geometry.dispose();
});
