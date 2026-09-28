import * as THREE from 'three';

type GridStats = {
  width: number;
  height: number;
  minimum: number;
  maximum: number;
  gsdX: number | null;
  gsdY: number | null;
  units: 'm' | 'relative';
};

export type MetricGridSpec = {
  horizontalUnit: 'm' | 'px';
  heightUnit: 'm' | 'relative';
  extentX: number;
  extentY: number;
  spacing: number;
  heightInterval: number;
  minimum: number;
  maximum: number;
  worldWidth: number;
  worldDepth: number;
  worldStepX: number;
  worldStepZ: number;
};

const positive = (value: number | null | undefined): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

/** Never turn an unreferenced TIFF's pixel transform or degrees into metres. */
export function metricPixelScale(
  explicit: number, native: number | null, modelType?: number,
  unitCode?: number, unitSize?: number,
): number | null {
  if (positive(explicit)) return explicit;
  if (modelType !== 1 || !positive(native)) return null;
  const factor = unitCode === 9001 ? 1 : unitCode === 9002 ? .3048
    : unitCode === 9003 ? 1200 / 3937 : positive(unitSize) ? unitSize : null;
  return factor === null ? null : native * factor;
}

export function niceGridStep(value: number) {
  if (!positive(value)) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  const scaled = value / power;
  return (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10) * power;
}

/** Extents are between the first and last sample centres, matching the mesh. */
export function metricGridSpec(stats: GridStats, worldWidth: number, worldDepth: number): MetricGridSpec {
  const metric = positive(stats.gsdX) && positive(stats.gsdY);
  const extentX = Math.max(1, stats.width - 1) * (metric ? stats.gsdX! : 1);
  const extentY = Math.max(1, stats.height - 1) * (metric ? stats.gsdY! : 1);
  const spacing = niceGridStep(Math.max(extentX, extentY) / 16);
  return {
    horizontalUnit: metric ? 'm' : 'px', heightUnit: stats.units,
    extentX, extentY, spacing,
    heightInterval: niceGridStep(Math.max(stats.maximum - stats.minimum, stats.units === 'm' ? 1 : .01) / 10),
    minimum: stats.minimum, maximum: stats.maximum,
    worldWidth, worldDepth,
    worldStepX: spacing / extentX * worldWidth,
    worldStepZ: spacing / extentY * worldDepth,
  };
}

export function formatGridNumber(value: number) {
  return Number(value.toPrecision(4)).toLocaleString('en-US', { maximumFractionDigits: 3 });
}

/** Reversible 650 ms transition; no camera/geometry changes or frame-rate dependence. */
export function advanceMetricBlend(current: number, target: number, deltaSeconds: number, reducedMotion = false) {
  if (reducedMotion) return target;
  const step = Math.max(0, deltaSeconds) / .65;
  return current < target ? Math.min(target, current + step) : Math.max(target, current - step);
}

/** A display-only material; geometry, raw predictions and object metadata stay intact. */
export function createMetricGridMaterial(spec: MetricGridSpec, verticalScale: number, heightOrigin = spec.minimum, opticalMaterial?: THREE.MeshStandardMaterial, objectClass = -1) {
  const material = opticalMaterial ?? new THREE.MeshStandardMaterial({
    color: '#ffffff', roughness: .92, metalness: 0, side: THREE.DoubleSide,
  });
  const originalBumpScale = material.bumpScale;
  const uniforms = {
    dwMetricBlend: { value: opticalMaterial ? 0 : 1 },
    dwGridStep: { value: new THREE.Vector2(spec.worldStepX, spec.worldStepZ) },
    dwGridOrigin: { value: new THREE.Vector2(-spec.worldWidth / 2, -spec.worldDepth / 2) },
    dwHeightScale: { value: verticalScale },
    dwHeightMinimum: { value: spec.minimum },
    dwHeightOrigin: { value: heightOrigin },
    dwHeightRange: { value: Math.max(.000001, spec.maximum - spec.minimum) },
    dwHeightInterval: { value: spec.heightInterval },
    dwClassMap: { value: null as THREE.DataTexture | null },
    dwMapSize: { value: new THREE.Vector2(spec.worldWidth, spec.worldDepth) },
    dwFocusAmount: { value: 0 },
    dwSelectedClass: { value: -1 },
    dwObjectClass: { value: objectClass },
    dwSixMap: { value: null as THREE.DataTexture | null },
    dwSixBlend: { value: 0 },
    dwContours: { value: 0 },
    dwAnalysisMode: { value: 0 },
  };
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = 'attribute float surfaceSlope;\nvarying float dwSlope;\nvarying vec3 dwWorldPosition;\n' + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\ndwSlope = surfaceSlope;\ndwWorldPosition = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = `
      varying vec3 dwWorldPosition;
      varying float dwSlope;
      uniform float dwContours;
      uniform float dwAnalysisMode;
      uniform float dwMetricBlend;
      uniform vec2 dwGridStep;
      uniform vec2 dwGridOrigin;
      uniform float dwHeightScale;
      uniform float dwHeightMinimum;
      uniform float dwHeightOrigin;
      uniform float dwHeightRange;
      uniform float dwHeightInterval;
      uniform sampler2D dwClassMap;
      uniform vec2 dwMapSize;
      uniform float dwFocusAmount;
      uniform float dwSelectedClass;
      uniform float dwObjectClass;
      uniform sampler2D dwSixMap;
      uniform float dwSixBlend;
      float dwLine(float coordinate) {
        float derivative = fwidth(coordinate);
        float distanceToLine = abs(fract(coordinate + .5) - .5);
        // Constant coordinates (flat roofs / axis-aligned walls) must not fill
        // the entire face with a line. Fade subpixel dense lines at distance.
        return (1.0 - smoothstep(.35, 1.25, distanceToLine / max(derivative, .00001)))
          * step(.00001, derivative) * (1.0 - smoothstep(.3, 1.0, derivative));
      }
      vec3 dwRamp(float t) {
        vec3 low = vec3(.018, .13, .18);
        vec3 mint = vec3(.06, .62, .44);
        vec3 lime = vec3(.48, .73, .10);
        vec3 amber = vec3(.9, .48, .06);
        vec3 peak = vec3(.9, .16, .13);
        if (t < .35) return mix(low, mint, t / .35);
        if (t < .64) return mix(mint, lime, (t - .35) / .29);
        if (t < .82) return mix(lime, amber, (t - .64) / .18);
        return mix(amber, peak, (t - .82) / .18);
      }
    ` + shader.fragmentShader;
    shader.fragmentShader = shader.fragmentShader.replace('#include <color_fragment>', `
      #include <color_fragment>
      float dwHeight = dwWorldPosition.y / max(dwHeightScale, .000001) + dwHeightOrigin;
      vec2 dwCell = (dwWorldPosition.xz - dwGridOrigin) / dwGridStep;
      float dwGridLine = max(dwLine(dwCell.x), dwLine(dwCell.y));
      float dwContour = dwLine(dwHeight / dwHeightInterval);
      vec3 dwBase = dwRamp(clamp((dwHeight - dwHeightMinimum) / dwHeightRange, 0.0, 1.0));
      vec3 dwGridColour = mix(dwBase, vec3(.52, .87, .85), dwGridLine * .43);
      dwGridColour = mix(dwGridColour, vec3(.025, .065, .08), dwContour * .55);
      diffuseColor.rgb = mix(diffuseColor.rgb, dwGridColour, dwMetricBlend);
      if (dwAnalysisMode > 1.5) diffuseColor.rgb = dwSlope < 0.0 ? vec3(.2) : dwRamp(clamp(dwSlope / 60.0, 0.0, 1.0));
      else if (dwAnalysisMode > .5) diffuseColor.rgb = vec3(.62, .68, .66);
      diffuseColor.rgb = mix(diffuseColor.rgb, vec3(.04, .12, .11), dwContour * dwContours * .8 * (1.0 - dwMetricBlend));
      float dwChosen = 0.0;
      vec3 dwAccent = dwSelectedClass < .5 ? vec3(.08,.72,.95) : dwSelectedClass < 1.5 ? vec3(1.0,.55,.12) : vec3(.25,.95,.42);
      if (dwFocusAmount > .001) {
        vec2 dwClassUV = (dwWorldPosition.xz - dwGridOrigin) / dwMapSize;
        float dwClass = dwObjectClass >= 0.0 ? dwObjectClass : floor(texture2D(dwClassMap, clamp(dwClassUV, 0.0, 1.0)).r * 255.0 + .5);
        dwChosen = 1.0 - step(.5, abs(dwClass - dwSelectedClass));
        vec3 dwFocusedColour = mix(vec3(.016,.035,.043) + dwGridLine * .04, mix(dwGridColour, dwAccent, .68), dwChosen);
        diffuseColor.rgb = mix(diffuseColor.rgb, dwFocusedColour, dwFocusAmount);
      }
    `);
    shader.fragmentShader = shader.fragmentShader.replace('#include <emissivemap_fragment>', `
      #include <emissivemap_fragment>
      float dwRim = pow(1.0 - abs(dot(normalize(normal), normalize(vViewPosition))), 2.0);
      totalEmissiveRadiance += dwAccent * dwChosen * dwFocusAmount * (.35 + .45 * dwRim + .25 * dwGridLine);
      if (dwSixBlend > .001) {
        vec2 dwSixUV = clamp((dwWorldPosition.xz - dwGridOrigin) / dwMapSize, 0.0, 1.0);
        vec4 dwSix = texture2D(dwSixMap, dwSixUV);
        diffuseColor.rgb = mix(diffuseColor.rgb, dwSix.rgb, dwSixBlend * dwSix.a);
        totalEmissiveRadiance += dwSix.rgb * dwSixBlend * dwSix.a * .12;
      }
    `);
    shader.fragmentShader = shader.fragmentShader.replace('#include <roughnessmap_fragment>',
      '#include <roughnessmap_fragment>\nroughnessFactor = mix(roughnessFactor, .92, dwMetricBlend);');
  };
  material.customProgramCacheKey = () => 'msr-metric-grid-six-preview-v4';
  material.needsUpdate = true;
  return {
    material,
    setTerrainAnalysis(contours: boolean, mode: number) { uniforms.dwContours.value = contours ? 1 : 0; uniforms.dwAnalysisMode.value = mode; },
    setSixClassOverlay(map: THREE.DataTexture | null, amount: number) {
      uniforms.dwSixMap.value = map;
      uniforms.dwSixBlend.value = map ? THREE.MathUtils.clamp(amount, 0, 1) : 0;
    },
    setExaggeration(value: number) { uniforms.dwHeightScale.value = verticalScale * value; },
    setBlend(value: number) {
      uniforms.dwMetricBlend.value = THREE.MathUtils.clamp(value, 0, 1);
      material.bumpScale = originalBumpScale * (1 - uniforms.dwMetricBlend.value);
    },
    setClassFocus(map: THREE.DataTexture | null, selected: number, amount: number) {
      uniforms.dwClassMap.value = map;
      uniforms.dwSelectedClass.value = selected;
      uniforms.dwFocusAmount.value = map && selected >= 0 ? THREE.MathUtils.clamp(amount, 0, 1) : 0;
    },
  };
}

export function createMetricGridFloor(spec: MetricGridSpec) {
  const points: number[] = [];
  const left = -spec.worldWidth / 2, top = -spec.worldDepth / 2;
  for (let x = 0; x <= spec.extentX / spec.spacing; x++) {
    const worldX = left + x * spec.worldStepX;
    points.push(worldX, -.13, top, worldX, -.13, -top);
  }
  for (let z = 0; z <= spec.extentY / spec.spacing; z++) {
    const worldZ = top + z * spec.worldStepZ;
    points.push(left, -.13, worldZ, -left, -.13, worldZ);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  return new THREE.LineSegments(geometry,
    new THREE.LineBasicMaterial({ color: '#548d98', transparent: true, opacity: .45 }));
}
