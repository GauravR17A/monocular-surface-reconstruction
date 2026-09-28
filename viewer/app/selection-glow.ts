import * as THREE from 'three';
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';

/** Three screen-space strokes give a soft halo without a full-screen bloom pass. */
export function createSelectionGlow() {
  const group = new THREE.Group();
  group.name = 'active-inspection-boundary'; group.visible = false;
  const strokes = [[7, .035], [3, .10], [1.15, .75]] as const;
  const materials = strokes.map(([linewidth, opacity]) => {
    const material = new LineMaterial({ color: '#65ffe1', linewidth, opacity, transparent: true,
      depthTest: true, depthWrite: false, toneMapped: false, worldUnits: false });
    material.onBeforeCompile = shader => {
      // Small depth bias keeps coplanar edges stable without seeing through buildings/hills.
      shader.vertexShader = shader.vertexShader.replace('#include <fog_vertex>',
        '#include <fog_vertex>\n gl_Position.z -= 0.00001 * gl_Position.w;');
    };
    return material;
  });
  let geometry: LineSegmentsGeometry | null = null;
  let fade = 0;
  return {
    group,
    set(positions: Float32Array) {
      group.clear(); geometry?.dispose(); geometry = null; fade = 0;
      if (!positions.length) { group.visible = false; return; }
      geometry = new LineSegmentsGeometry().setPositions(positions);
      materials.forEach((material, i) => {
        const line = new LineSegments2(geometry!, material);
        line.renderOrder = 20 + i; line.raycast = () => {};
        group.add(line);
      });
      group.visible = true;
    },
    clear() { group.visible = false; },
    update(delta: number, scaleY: number, reducedMotion: boolean) {
      if (!group.visible) return;
      group.scale.y = scaleY;
      fade = reducedMotion ? 1 : Math.min(1, fade + Math.max(0, delta) / .16);
      materials.forEach((material, i) => { material.opacity = strokes[i][1] * fade; });
    },
    dispose() { group.removeFromParent(); group.clear(); geometry?.dispose(); materials.forEach(m => m.dispose()); },
  };
}

export function buildingSelectionEdges(mesh: THREE.Mesh): Float32Array {
  const edges = new THREE.EdgesGeometry(mesh.geometry, 32);
  mesh.updateMatrix();
  // Parent Y exaggeration is applied to the glow group, not baked into these positions.
  edges.applyMatrix4(mesh.matrix);
  const result = new Float32Array(edges.getAttribute('position').array);
  edges.dispose();
  return result;
}

export function drapeSelectionEdges(edges: Float32Array, width: number, depth: number,
  surface: (u: number, v: number) => number | null, lift: number): Float32Array {
  const points: number[] = [];
  for (let i = 0; i < edges.length; i += 4) {
    const [u, v, u2, v2] = edges.subarray(i, i + 4);
    const a = surface(u, v), b = surface(u2, v2);
    if (a === null || b === null || !Number.isFinite(a) || !Number.isFinite(b)) continue;
    points.push((u - .5) * width, a + lift, (v - .5) * depth, (u2 - .5) * width, b + lift, (v2 - .5) * depth);
  }
  return new Float32Array(points);
}
