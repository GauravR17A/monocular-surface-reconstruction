import * as THREE from 'three';

/** Close valid outer edges down to the scene minimum, without inventing interior heights. */
export function createTerrainBoundary(positions: THREE.BufferAttribute, valid: Uint8Array, segmentsX: number, segmentsY: number) {
  const wallPositions: number[] = [], wallColors: number[] = [], lines: number[] = [];
  const topColor = new THREE.Color('#4f8070'), baseColor = new THREE.Color('#10271f');
  const append = (a: number, b: number) => {
    if (!valid[a] || !valid[b]) return;
    const p = [positions.getX(a), positions.getY(a), positions.getZ(a)];
    const q = [positions.getX(b), positions.getY(b), positions.getZ(b)];
    const floorP = [p[0], 0, p[2]], floorQ = [q[0], 0, q[2]];
    for (const [point, color] of [[p, topColor], [floorP, baseColor], [q, topColor], [q, topColor], [floorP, baseColor], [floorQ, baseColor]] as const) {
      wallPositions.push(...point); wallColors.push(color.r, color.g, color.b);
    }
    lines.push(...p, ...q, ...floorP, ...floorQ);
  };
  const row = segmentsX + 1;
  for (let x = 0; x < segmentsX; x++) { append(x, x + 1); append(segmentsY * row + x, segmentsY * row + x + 1); }
  for (let y = 0; y < segmentsY; y++) { append(y * row, (y + 1) * row); append(y * row + segmentsX, (y + 1) * row + segmentsX); }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(wallPositions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(wallColors, 3));
  geometry.computeVertexNormals();
  const group = new THREE.Group(); group.name = 'terrain-elevation-boundary';
  group.add(new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, side: THREE.DoubleSide })));
  const outline = new THREE.BufferGeometry();
  outline.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
  group.add(new THREE.LineSegments(outline, new THREE.LineBasicMaterial({ color: '#8db6a5', transparent: true, opacity: .5 })));
  return group;
}
