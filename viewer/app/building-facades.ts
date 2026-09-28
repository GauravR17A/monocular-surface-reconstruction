import * as THREE from 'three';

/** Decorative wall materials only. These are not inferred windows or floor counts. */
export type FacadeMaps = { map: THREE.CanvasTexture; bumpMap: THREE.CanvasTexture; roughnessMap: THREE.CanvasTexture };

export function createFacadeMaps(variant: number): FacadeMaps {
  const makeCanvas = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 256;
    return canvas;
  };
  const color = makeCanvas(), relief = makeCanvas(), roughness = makeCanvas();
  const ctx = color.getContext('2d')!, bump = relief.getContext('2d')!, matte = roughness.getContext('2d')!;
  const palettes = [
    { plaster: [211, 207, 197], glass: '#687880', lower: '#46565e', trim: '#bcbcb5' },
    { plaster: [198, 205, 204], glass: '#64757e', lower: '#45545f', trim: '#b2bcbd' },
    { plaster: [211, 198, 181], glass: '#747b79', lower: '#515e61', trim: '#bfb8ac' },
  ];
  const palette = palettes[variant % palettes.length];
  const grain = ctx.createImageData(512, 256);
  let seed = 619 + variant * 137;
  for (let i = 0; i < 512 * 256; i++) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const noise = (seed % 9) - 4;
    grain.data.set([...palette.plaster.map(value => value + noise), 255], i * 4);
  }
  ctx.putImageData(grain, 0, 0);
  bump.fillStyle = '#b8b8b8'; bump.fillRect(0, 0, 512, 256);
  matte.fillStyle = '#eeeeee'; matte.fillRect(0, 0, 512, 256);

  // Two bays / one illustrative storey. Low-contrast joints survive close views
  // without turning distant facades into high-frequency stripes.
  for (const x of [0, 256]) {
    ctx.fillStyle = 'rgba(45,50,50,.08)'; ctx.fillRect(x, 0, 2, 256);
    ctx.fillStyle = 'rgba(255,255,245,.16)'; ctx.fillRect(x + 2, 0, 1, 256);
    if (variant === 2) {
      for (let row = 0; row < 4; row++) {
        ctx.fillStyle = 'rgba(76,64,52,.045)'; ctx.fillRect(x, 30 + row * 54, 256, 1);
      }
    }
    const wx = x + (variant === 1 ? 40 : 54), wy = 52;
    const ww = variant === 1 ? 176 : 148, wh = 133;
    ctx.fillStyle = 'rgba(43,48,48,.18)'; ctx.fillRect(wx - 6, wy - 5, ww + 12, wh + 14);
    ctx.fillStyle = palette.trim; ctx.fillRect(wx - 3, wy - 2, ww + 6, wh + 6);
    const glass = ctx.createLinearGradient(0, wy, 0, wy + wh);
    glass.addColorStop(0, palette.lower);
    glass.addColorStop(.5, palette.glass);
    glass.addColorStop(1, palette.lower);
    ctx.fillStyle = glass; ctx.fillRect(wx, wy, ww, wh);
    // Muted sky reflection, inset frame and sill; no glow or fabricated imagery.
    ctx.fillStyle = 'rgba(216,232,231,.10)'; ctx.fillRect(wx + 4, wy + 4, ww - 8, 31);
    ctx.fillStyle = 'rgba(25,35,37,.27)'; ctx.fillRect(wx, wy, ww, 4);
    ctx.fillStyle = palette.trim; ctx.fillRect(wx + ww / 2 - 2, wy, 4, wh);
    ctx.fillStyle = 'rgba(235,232,219,.48)'; ctx.fillRect(wx - 4, wy + wh + 2, ww + 8, 3);
    ctx.fillStyle = 'rgba(46,45,43,.16)'; ctx.fillRect(wx - 4, wy + wh + 5, ww + 8, 5);
    bump.fillStyle = '#777777'; bump.fillRect(wx, wy, ww, wh);
    bump.fillStyle = '#c4c4c4'; bump.fillRect(wx + ww / 2 - 2, wy, 4, wh);
    bump.fillRect(wx - 4, wy + wh + 2, ww + 8, 3);
    matte.fillStyle = '#8e8e8e'; matte.fillRect(wx, wy, ww, wh);
  }
  // A restrained floor band with an integrated contact-shadow strip.
  ctx.fillStyle = 'rgba(38,45,46,.12)'; ctx.fillRect(0, 238, 512, 6);
  ctx.fillStyle = 'rgba(247,242,226,.22)'; ctx.fillRect(0, 244, 512, 3);
  bump.fillStyle = '#c7c7c7'; bump.fillRect(0, 242, 512, 5);
  const texture = (canvas: HTMLCanvasElement, isColor = false) => {
    const result = new THREE.CanvasTexture(canvas);
    result.wrapS = result.wrapT = THREE.RepeatWrapping;
    result.anisotropy = 8;
    if (isColor) result.colorSpace = THREE.SRGBColorSpace;
    return result;
  };
  return { map: texture(color, true), bumpMap: texture(relief), roughnessMap: texture(roughness) };
}

export function facadeLayout(heightM: number, sceneHeight: number, width: number, depth: number) {
  // Visual tiling only; never shown as a measured storey count.
  const rows = Math.min(48, Math.max(1, Math.round(Math.max(0, heightM) / 3.4)));
  return { rows, tileWidth: Math.max(1.25, sceneHeight / rows * 1.9, Math.max(width, depth) / 14) };
}

/** Keep roof UVs intact and map walls along their true horizontal tangent. */
export function applyFacadeUVs(
  geometry: THREE.BufferGeometry,
  scaleX: number, scaleZ: number, bottom: number, height: number,
  rows: number, tileWidth: number,
) {
  const positions = geometry.getAttribute('position'), normals = geometry.getAttribute('normal');
  const uvs = geometry.getAttribute('uv');
  const colors = new Float32Array(positions.count * 3).fill(1);
  for (let i = 0; i < positions.count; i++) {
    if (Math.abs(normals.getY(i)) > .5) continue;
    // Inverse-scale the normal before constructing a world-space wall tangent.
    const nx = normals.getX(i) / scaleX, nz = normals.getZ(i) / scaleZ;
    const length = Math.hypot(nx, nz) || 1;
    const along = (-nz * positions.getX(i) * scaleX + nx * positions.getZ(i) * scaleZ) / length;
    const fraction = Math.min(1, Math.max(0, (positions.getY(i) - bottom) / height));
    uvs.setXY(i, along / tileWidth, fraction * rows);
    // Gentle ground contact shading, without modifying the geometry itself.
    const shade = .76 + .24 * fraction;
    colors.set([shade, shade, shade], i * 3);
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  uvs.needsUpdate = true;
}
