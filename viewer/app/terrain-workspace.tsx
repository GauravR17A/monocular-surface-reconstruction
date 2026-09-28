'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Image from 'next/image';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { fromArrayBuffer, fromBlob } from 'geotiff';
import { LandscapeEvaluation, type EvaluationEvidenceId } from './landscape-evaluation';
import { ValidationComparison, type ReferenceValidationMetrics } from './validation-comparison';
import { gridSurfaceHeight, moveFlight, touchesFootprint, type FlightObstacle } from './flight-navigation';
import { createWalkMotion, findWalkSpawn, stepWalk, walkHeadMotion, walkIntent, WALK_EYE_METRES, WALK_SPEED_MPS, type WalkPoint, type WalkWorld, type WalkMotion } from './walk-navigation';
import { querySurface, rasterValue, surfaceClassCode, type InspectionData, type SurfaceQuery } from './surface-query';
import { ALL_CLASS_HEIGHTS, classFocusFrame, focusedVertexHeights, type FocusClass, type ClassWeights } from './class-focus';
import { fullscreenEscapeAction, fullscreenKeyboard, type EscapeKeyboard } from './fullscreen-navigation';
import { applyFacadeUVs, createFacadeMaps, facadeLayout, type FacadeMaps } from './building-facades';
import { advanceMetricBlend, createMetricGridFloor, createMetricGridMaterial, formatGridNumber, metricGridSpec, metricPixelScale, type MetricGridSpec } from './metric-grid';
import { REVEAL_LABELS, sceneRevealAt, type RevealPhase } from './scene-reveal';
import { SixClassPreview } from './six-class-preview';
import { classLayerPixels, SIX_CLASS_NAMES, sixClassOverlayVisible, withSixClass, type SixClassLayer } from './six-class-layer';
import { DEFAULT_CLASS_VIEW, stateForScene, type SixClassState, type SixClassView } from './six-class-request';
import { inspectionReadout } from './inspection-readout';
import { terrainDimensions, automaticTerrainExaggeration, interpolateSurface, sampleTerrainProfile, surfaceSlope, type TerrainProfileData, type TerrainPoint } from './terrain-analysis';
import { createTerrainBoundary } from './terrain-boundary';
import { TerrainProfile } from './terrain-profile';
import { hasTerrainGeoreference, terrainCalibrationSource } from './terrain-import';
import { ImageImportControls } from './image-import-controls';
import { sceneScaleLength } from './scene-scale';
import { terrainMeshResolution, terrainPresentationSurface } from './terrain-presentation';
import { createSelectionRegions, sampleLocator, type SelectionRegionIndex } from './selection-region';
import { buildingSelectionEdges, createSelectionGlow, drapeSelectionEdges } from './selection-glow';
import { clampImageView, FIT_IMAGE_VIEW, zoomImageView, type ImageView } from './image-reference-view';
import { ExploreMap } from './explore-map';
import { mapFrame, type MapFrame } from './explore-map-math';

type ViewMode = 'orbit' | 'explore' | 'inspect' | 'sightline' | 'profile';
type ExploreMode = 'fly' | 'walk';
type SurfaceStyle = 'optical' | 'elevation' | 'metric' | 'hillshade' | 'slope';
type SurfaceProductKind = 'ndsm' | 'dsm' | 'relative';
type DemoSceneId = 'urban' | 'sparse' | 'hilly' | 'forest';

// Keep the inspector consistent with prepare_presentation_mesh in the API.
// RGB assistance is presentation-only; learned probabilities remain downloadable.
const VEGETATION_DISPLAY_THRESHOLD = 0.35;

type SceneInput = {
  heightBuffer: ArrayBuffer | null;
  analysisBuffer: ArrayBuffer | null;
  terrainBuffer?: ArrayBuffer | null;
  inferHeightUnits?: boolean;
  missionBuffer: ArrayBuffer | null;
  missionSource: MissionAnalysisData['source'] | null;
  buildingProbabilityBuffer: ArrayBuffer | null;
  vegetationProbabilityBuffer: ArrayBuffer | null;
  semanticClassBuffer: ArrayBuffer | null;
  confidenceBuffer: ArrayBuffer | null;
  textureUrl: string;
  label: string;
  productKind: SurfaceProductKind;
  proof: string;
  structures: BuildingSolid[];
  animateImport?: boolean;
};

type BuildingSolid = {
  id: number;
  pixels: number;
  center_x: number;
  center_y: number;
  width: number;
  depth: number;
  angle_rad: number;
  height_m: number;
  base_m: number;
  confidence: number;
  outline?: [number, number][];
};

type GeneratedProducts = {
  rawHeightUrl: string;
  rdsmUrl: string;
  relativeDepthUrl: string;
  buildingProbabilityUrl: string | null;
  vegetationProbabilityUrl: string | null;
  semanticClassUrl: string | null;
  predictionConfidenceUrl: string | null;
  absoluteDsmUrl: string | null;
  gcpSurfaceUrl: string | null;
  alignedDemUrl: string | null;
  publicTerrainUrl: string | null;
  validationMetricsUrl: string | null;
  validationReferenceUrl: string | null;
  validationErrorUrl: string | null;
  metadataUrl: string;
};

type ValidationMetrics = ReferenceValidationMetrics;

type DemoValidationEvidence = {
  metrics: ReferenceValidationMetrics;
  predictionUrl: string;
  referenceUrl: string;
  errorUrl: string;
  title: string;
  provenance: string;
};

type DemoSceneConfig = {
  shortLabel: string;
  label: string;
  proof: string;
  metricSummary: string;
  productKind: SurfaceProductKind;
  heightUrl: string;
  analysisUrl: string;
  terrainUrl?: string;
  textureUrl: string;
  structuresUrl: string;
  buildingProbabilityUrl: string | null;
  vegetationProbabilityUrl: string | null;
  semanticClassUrl: string | null;
  confidenceUrl: string | null;
  validation: DemoValidationEvidence | null;
};

type SceneStats = {
  width: number;
  height: number;
  vertices: number;
  minimum: number;
  maximum: number;
  mean: number;
  meanSlope: number | null;
  gsd: number | null;
  gsdX: number | null;
  gsdY: number | null;
  units: 'm' | 'relative';
};

type Layers = {
  grid: boolean;
  wireframe: boolean;
  canopy: boolean;
  contours: boolean;
};

type MissionBuilding = {
  id: number;
  groundElevationM: number;
  roofElevationM: number;
};

type MissionAnalysisData = {
  source: 'dem_anchored_dsm' | 'gcp_calibrated_surface' | 'bundled_absolute_dsm';
  sortedElevationsM: Float32Array;
  minimumM: number;
  maximumM: number;
  scenarioMaximumM: number;
  defaultScenarioRiseM: number;
  pixelAreaM2: number | null;
  mappedBuildings: MissionBuilding[];
  steepAreaFraction: number | null;
};

type SightlinePoint = {
  u: number;
  v: number;
};

type SightlineProfileSample = {
  terrain: number;
  ray: number;
};

type SightlineResult = {
  visible: boolean;
  metricHeights: boolean;
  distanceM: number | null;
  minimumClearance: number;
  blockedAtFraction: number | null;
  sampleCount: number;
  profile: SightlineProfileSample[];
};

const DEMO_SCENES: Record<DemoSceneId, DemoSceneConfig> = {
  urban: {
    shortLabel: 'Urban',
    label: 'Copenhagen · semantic solid reconstruction · held-out',
    proof: 'Included urban sample · protected building footprints with mixed-scene vegetation routing.',
    metricSummary: 'Urban reconstruction demonstration',
    productKind: 'ndsm',
    heightUrl: '/demo/copenhagen_presentation_mesh_m.tif',
    analysisUrl: '/demo/copenhagen_height_m.tif',
    textureUrl: '/demo/copenhagen_rgb.jpg',
    structuresUrl: '/demo/copenhagen_structures.json',
    buildingProbabilityUrl: '/demo/copenhagen_building_probability.tif',
    vegetationProbabilityUrl: '/demo/copenhagen_vegetation_probability.tif',
    semanticClassUrl: null,
    confidenceUrl: '/demo/copenhagen_prediction_confidence.tif',
    validation: null,
  },
  sparse: {
    shortLabel: 'Sparse',
    label: 'Amsterdam rural fringe · held-out sparse test scene',
    proof: 'Held-out HighBuild test chip selected by sparse reference coverage; prediction was then evaluated against the reference nDSM.',
    metricSummary: 'RMSE 1.48 m · MAE 0.71 m · R² 0.752',
    productKind: 'ndsm',
    heightUrl: '/demo/landscapes/sparse/presentation_height_m.tif',
    analysisUrl: '/demo/landscapes/sparse/analysis_height_m.tif',
    textureUrl: '/demo/landscapes/sparse/texture.jpg',
    structuresUrl: '/demo/landscapes/sparse/structures.json',
    buildingProbabilityUrl: '/demo/landscapes/sparse/building_probability.tif',
    vegetationProbabilityUrl: '/demo/landscapes/sparse/vegetation_probability.tif',
    semanticClassUrl: '/demo/landscapes/sparse/semantic_surface_class.tif',
    confidenceUrl: '/demo/landscapes/sparse/prediction_confidence.tif',
    validation: {
      metrics: {
        pixel_count: 1048576,
        rmse_m: 1.4800429094496625,
        mae_m: 0.7125934032699401,
        bias_m: 0.28392161209120736,
        correlation: 0.8732458753121313,
        r2: 0.7515417700234649,
        reference_kind: 'ndsm',
        prediction_product: 'height_above_ground_ndsm',
        alignment: 'pixel_aligned_same_shape',
      },
      predictionUrl: '/demo/landscapes/sparse/analysis_height_m.tif',
      referenceUrl: '/demo/landscapes/sparse/validation_reference_m.tif',
      errorUrl: '/demo/landscapes/sparse/validation_signed_error_m.tif',
      title: 'Sparse landscape: model vs reference',
      provenance: 'HighBuild official test split · rural Amsterdam fringe · independent building-height reference · 1,048,576 aligned pixels',
    },
  },
  hilly: {
    shortLabel: 'Hilly',
    label: 'Manali, Himachal Pradesh · georeferenced hilly DSM',
    proof: 'Sentinel-2 optical image anchored to Copernicus GLO-30 terrain. Large values are elevation above sea level; the model contributes a 0–19.3 m height-above-ground estimate.',
    metricSummary: '10 m image grid · 30 m terrain source · absolute DSM',
    productKind: 'dsm',
    heightUrl: '/demo/landscapes/hilly/absolute_dsm_m.tif?v=full-coverage-v2',
    analysisUrl: '/demo/landscapes/hilly/absolute_dsm_m.tif?v=full-coverage-v2',
    terrainUrl: '/demo/landscapes/hilly/terrain_dem_m.tif',
    textureUrl: '/demo/landscapes/hilly/texture.jpg?v=full-coverage-v2',
    structuresUrl: '/demo/landscapes/hilly/structures.json?v=full-coverage-v2',
    buildingProbabilityUrl: '/demo/landscapes/hilly/building_probability.tif?v=full-coverage-v2',
    vegetationProbabilityUrl: '/demo/landscapes/hilly/vegetation_probability.tif?v=full-coverage-v2',
    semanticClassUrl: '/demo/landscapes/hilly/semantic_surface_class.tif?v=full-coverage-v2',
    confidenceUrl: '/demo/landscapes/hilly/prediction_confidence.tif?v=full-coverage-v2',
    validation: {
      metrics: {
        pixel_count: 589824,
        rmse_m: 15.02895307511556,
        mae_m: 10.244428290054202,
        bias_m: -4.951612202036712,
        correlation: 0.9995649150144258,
        r2: 0.9990240195450355,
        reference_kind: 'dsm',
        prediction_product: 'absolute_surface_dsm',
        alignment: 'reprojected_to_prediction_grid',
      },
      predictionUrl: '/demo/hilly_validation/model_prediction_m.tif?v=srtm-v1',
      referenceUrl: '/demo/hilly_validation/srtm_reference_aligned_m.tif?v=srtm-v1',
      errorUrl: '/demo/hilly_validation/signed_error_m.tif?v=srtm-v1',
      title: 'Hilly landscape: absolute DSM vs independent SRTM',
      provenance: 'Manali scene · Copernicus-calibrated prediction · separate SRTM 30 m evaluation reference · no post-hoc offset fitting · vertical datum/resolution differences disclosed',
    },
  },
  forest: {
    shortLabel: 'Forest',
    label: 'Open-Canopy · held-out forest canopy scene',
    proof: 'Official Open-Canopy test chip · SPOT optical RGB · predicted canopy-height nDSM independently checked against IGN LiDAR-derived canopy height.',
    metricSummary: 'RMSE 4.22 m · MAE 2.86 m · R² 0.409',
    productKind: 'ndsm',
    heightUrl: '/demo/forest_validation/model_prediction_m.tif?v=forest-demo-v1',
    analysisUrl: '/demo/forest_validation/model_prediction_m.tif?v=forest-demo-v1',
    textureUrl: '/demo/forest_validation/satellite_rgb.jpg?v=forest-demo-v1',
    structuresUrl: '/demo/forest_validation/structures.json?v=forest-demo-v1',
    buildingProbabilityUrl: null,
    vegetationProbabilityUrl: null,
    semanticClassUrl: null,
    confidenceUrl: null,
    validation: {
      metrics: {
        pixel_count: 147456,
        rmse_m: 4.221455128914408,
        mae_m: 2.8560520831625684,
        bias_m: -2.027306752088609,
        correlation: 0.780453766961439,
        r2: 0.40943854192797247,
        reference_kind: 'ndsm',
        prediction_product: 'height_above_ground_ndsm',
        alignment: 'exact_geospatial_grid',
      },
      predictionUrl: '/demo/forest_validation/model_prediction_m.tif?v=forest-demo-v1',
      referenceUrl: '/demo/forest_validation/lidar_reference_m.tif?v=forest-demo-v1',
      errorUrl: '/demo/forest_validation/signed_error_m.tif?v=forest-demo-v1',
      title: 'Forest canopy: model vs LiDAR reference',
      provenance: 'Open-Canopy official test split · SPOT optical RGB + IGN LiDAR-derived canopy-height model · 1.5 m GSD',
    },
  },
};
const FOREST_EVIDENCE: DemoValidationEvidence & { textureUrl: string } = {
  textureUrl: '/demo/forest_validation/satellite_rgb.jpg',
  predictionUrl: '/demo/forest_validation/model_prediction_m.tif',
  referenceUrl: '/demo/forest_validation/lidar_reference_m.tif',
  errorUrl: '/demo/forest_validation/signed_error_m.tif',
  title: 'Forest canopy: model vs LiDAR reference',
  provenance: 'Open-Canopy official test split · SPOT optical RGB + IGN LiDAR-derived canopy-height model · 1.5 m GSD',
  metrics: {
    pixel_count: 147456,
    rmse_m: 4.221455128914408,
    mae_m: 2.8560520831625684,
    bias_m: -2.027306752088609,
    correlation: 0.780453766961439,
    r2: 0.40943854192797247,
    reference_kind: 'ndsm',
    prediction_product: 'height_above_ground_ndsm',
    alignment: 'exact_geospatial_grid',
  } satisfies ReferenceValidationMetrics,
} as const;
const API_BASE = process.env.NEXT_PUBLIC_MSR_API_URL || 'http://127.0.0.1:8000';
const MAX_UPLOAD_BYTES = 512 * 1024 * 1024;

function formatValue(value: number, units: SceneStats['units']) {
  return units === 'm' ? `${value.toFixed(1)} m` : value.toFixed(3);
}

function lowerBound(values: Float32Array, target: number) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (values[middle] <= target) low = middle + 1;
    else high = middle;
  }
  return low;
}

function formatScreeningArea(areaM2: number | null) {
  if (areaM2 === null) return 'Scale needed';
  if (areaM2 >= 1_000_000) return `${(areaM2 / 1_000_000).toFixed(2)} km\u00b2`;
  if (areaM2 >= 10_000) return `${(areaM2 / 10_000).toFixed(1)} ha`;
  return `${Math.round(areaM2).toLocaleString()} m\u00b2`;
}

function missionSourceLabel(source: MissionAnalysisData['source']) {
  if (source === 'dem_anchored_dsm') return 'DEM-anchored DSM';
  if (source === 'gcp_calibrated_surface') return 'GCP-calibrated surface';
  return 'Bundled absolute DSM';
}

function sampleSurface(
  values: Float32Array,
  width: number,
  height: number,
  point: SightlinePoint,
) {
  const x = Math.min(width - 1, Math.max(0, point.u * (width - 1)));
  const y = Math.min(height - 1, Math.max(0, point.v * (height - 1)));
  const left = Math.floor(x);
  const right = Math.min(width - 1, left + 1);
  const top = Math.floor(y);
  const bottom = Math.min(height - 1, top + 1);
  const xBlend = x - left;
  const yBlend = y - top;
  const samples = [
    { value: values[top * width + left], weight: (1 - xBlend) * (1 - yBlend) },
    { value: values[top * width + right], weight: xBlend * (1 - yBlend) },
    { value: values[bottom * width + left], weight: (1 - xBlend) * yBlend },
    { value: values[bottom * width + right], weight: xBlend * yBlend },
  ].filter((sample) => Number.isFinite(sample.value) && sample.weight > 0);
  const weight = samples.reduce((sum, sample) => sum + sample.weight, 0);
  return weight
    ? samples.reduce((sum, sample) => sum + sample.value * sample.weight, 0) / weight
    : Number.NaN;
}

function analyzeSightline(
  values: Float32Array,
  stats: SceneStats,
  start: SightlinePoint,
  end: SightlinePoint,
) {
  const pixelDx = (end.u - start.u) * Math.max(1, stats.width - 1);
  const pixelDy = (end.v - start.v) * Math.max(1, stats.height - 1);
  const pixelDistance = Math.hypot(pixelDx, pixelDy);
  const sampleCount = Math.max(32, Math.min(256, Math.ceil(pixelDistance) + 1));
  const startTerrain = sampleSurface(values, stats.width, stats.height, start);
  const endTerrain = sampleSurface(values, stats.width, stats.height, end);
  if (!Number.isFinite(startTerrain) || !Number.isFinite(endTerrain)) return null;
  const valueRange = Math.max(1e-6, stats.maximum - stats.minimum);
  const eyeHeight = stats.units === 'm' ? 2 : Math.max(0.02, valueRange * 0.04);
  const profile: SightlineProfileSample[] = [];
  let minimumClearance = Number.POSITIVE_INFINITY;
  let blockedAtFraction: number | null = null;
  const tolerance = stats.units === 'm' ? 0.15 : valueRange * 0.005;

  for (let index = 0; index < sampleCount; index += 1) {
    const fraction = index / (sampleCount - 1);
    const point = {
      u: start.u + (end.u - start.u) * fraction,
      v: start.v + (end.v - start.v) * fraction,
    };
    const terrain = sampleSurface(values, stats.width, stats.height, point);
    const ray = startTerrain + eyeHeight + (endTerrain - startTerrain) * fraction;
    if (!Number.isFinite(terrain)) continue;
    const clearance = ray - terrain;
    profile.push({ terrain, ray });
    if (index > 1 && index < sampleCount - 2 && clearance < minimumClearance) {
      minimumClearance = clearance;
    }
    if (blockedAtFraction === null && index > 1 && index < sampleCount - 2 && clearance < -tolerance) {
      blockedAtFraction = fraction;
    }
  }
  if (!Number.isFinite(minimumClearance)) minimumClearance = eyeHeight;
  const distanceM = stats.gsdX && stats.gsdY
    ? Math.hypot(pixelDx * stats.gsdX, pixelDy * stats.gsdY)
    : null;
  return {
    visible: blockedAtFraction === null,
    metricHeights: stats.units === 'm',
    distanceM,
    minimumClearance,
    blockedAtFraction,
    sampleCount: profile.length,
    profile,
  } satisfies SightlineResult;
}

function smoothSurface(values: Float32Array, width: number, height: number, passes: number) {
  let source = values.slice();
  for (let pass = 0; pass < passes; pass += 1) {
    const target = source.slice();
    for (let row = 1; row < height - 1; row += 1) {
      for (let col = 1; col < width - 1; col += 1) {
        let sum = 0;
        let weight = 0;
        for (let y = -1; y <= 1; y += 1) {
          for (let x = -1; x <= 1; x += 1) {
            const value = source[(row + y) * width + col + x];
            if (Number.isFinite(value)) {
              const localWeight = x === 0 && y === 0 ? 4 : (x === 0 || y === 0 ? 2 : 1);
              sum += value * localWeight;
              weight += localWeight;
            }
          }
        }
        if (weight) target[row * width + col] = sum / weight;
      }
    }
    source = target;
  }
  return source;
}

function colorForHeight(target: THREE.Color, amount: number) {
  const stops = [
    [0.0, new THREE.Color('#123d39')],
    [0.35, new THREE.Color('#45f3c4')],
    [0.64, new THREE.Color('#bbf451')],
    [0.82, new THREE.Color('#ffc857')],
    [1.0, new THREE.Color('#ff7168')],
  ] as const;
  for (let index = 1; index < stops.length; index += 1) {
    if (amount <= stops[index][0]) {
      const [leftAt, left] = stops[index - 1];
      const [rightAt, right] = stops[index];
      return target.copy(left).lerp(right, (amount - leftAt) / (rightAt - leftAt));
    }
  }
  return target.copy(stops[stops.length - 1][1]);
}

async function decodeHeight(buffer: ArrayBuffer, forceRelative = false, forceMetric = false) {
  const tiff = await fromArrayBuffer(buffer);
  const image = await tiff.getImage();
  const width = image.getWidth();
  const height = image.getHeight();
  const samples = image.getSamplesPerPixel();
  const raster = await image.readRasters({ interleave: true }) as unknown as ArrayLike<number>;
  const noData = image.getGDALNoData();
  const values = new Float32Array(width * height);
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  let sum = 0;
  let count = 0;

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const value = Number(raster[pixel * samples]);
    const valid = Number.isFinite(value) && (noData === null || value !== noData);
    values[pixel] = valid ? value : Number.NaN;
    if (valid) {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      sum += value;
      count += 1;
    }
  }
  if (!count) throw new Error('The height raster contains no finite pixels.');

  const directory = image.getFileDirectory() as unknown as { ModelPixelScale?: number[] };
  const metadata = await image.getGDALMetadata();
  const geoKeys = (image.getGeoKeys() ?? {}) as unknown as {
    GTModelTypeGeoKey?: number;
    ProjLinearUnitsGeoKey?: number;
    ProjLinearUnitSizeGeoKey?: number;
  };
  const explicitGsdX = Number(metadata?.MSR_PIXEL_SIZE_X_M);
  const explicitGsdY = Number(metadata?.MSR_PIXEL_SIZE_Y_M);
  const nativeScaleX = directory.ModelPixelScale?.[0] ?? null;
  const nativeScaleY = directory.ModelPixelScale?.[1] ?? nativeScaleX;
  const unitCode = geoKeys.ProjLinearUnitsGeoKey;
  const gsdX = metricPixelScale(explicitGsdX, nativeScaleX, geoKeys.GTModelTypeGeoKey, unitCode, geoKeys.ProjLinearUnitSizeGeoKey);
  const gsdY = metricPixelScale(explicitGsdY, nativeScaleY, geoKeys.GTModelTypeGeoKey, unitCode, geoKeys.ProjLinearUnitSizeGeoKey);
  const gsd = gsdX && gsdY ? (gsdX + gsdY) / 2 : gsdX ?? gsdY;
  const units: SceneStats['units'] = forceRelative || (!forceMetric && minimum >= -0.01 && maximum <= 1.5)
    ? 'relative'
    : 'm';

  let slopeSum = 0;
  let slopeCount = 0;
  if (units === 'm' && gsdX && gsdY) {
    const sampleStep = Math.max(1, Math.floor(Math.max(width, height) / 512));
    for (let row = sampleStep; row < height - sampleStep; row += sampleStep) {
      for (let col = sampleStep; col < width - sampleStep; col += sampleStep) {
        const left = values[row * width + col - sampleStep];
        const right = values[row * width + col + sampleStep];
        const top = values[(row - sampleStep) * width + col];
        const bottom = values[(row + sampleStep) * width + col];
        if ([left, right, top, bottom].every(Number.isFinite)) {
          const dzdx = (right - left) / (2 * sampleStep * gsdX);
          const dzdy = (bottom - top) / (2 * sampleStep * gsdY);
          slopeSum += Math.atan(Math.hypot(dzdx, dzdy)) * 180 / Math.PI;
          slopeCount += 1;
        }
      }
    }
  }

  return {
    values,
    stats: {
      width,
      height,
      vertices: 0,
      minimum,
      maximum,
      mean: sum / count,
      meanSlope: slopeCount ? slopeSum / slopeCount : null,
      gsd,
      gsdX,
      gsdY,
      units,
    } satisfies SceneStats,
  };
}

function loadTexture(url: string) {
  if (!url) {
    // A standalone height raster has no optical image. Never borrow a demo photo.
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 2;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#b8b8b8';
    context.fillRect(0, 0, 2, 2);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return Promise.resolve(texture);
  }
  return new Promise<THREE.Texture>((resolve, reject) => {
    new THREE.TextureLoader().load(url, (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = 8;
      resolve(texture);
    }, undefined, reject);
  });
}

function buildCanopyProxy(
  texture: THREE.Texture,
  terrainValues: Float32Array,
  rasterWidth: number,
  rasterHeight: number,
  terrainWidth: number,
  terrainDepth: number,
  minimum: number,
  verticalScale: number,
) {
  const group = new THREE.Group();
  const image = texture.image as HTMLImageElement;
  if (!image?.width || !image?.height) return group;
  const canvas = document.createElement('canvas');
  canvas.width = 96;
  canvas.height = 96;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) return group;
  context.drawImage(image, 0, 0, 96, 96);
  const pixels = context.getImageData(0, 0, 96, 96).data;
  const candidates: Array<{ x: number; y: number; green: number }> = [];
  for (let row = 2; row < 94; row += 3) {
    for (let col = 2; col < 94; col += 3) {
      const offset = (row * 96 + col) * 4;
      const red = pixels[offset];
      const green = pixels[offset + 1];
      const blue = pixels[offset + 2];
      if (green > 42 && green > red * 1.08 && green > blue * 1.06) {
        candidates.push({ x: col / 95, y: row / 95, green: green / 255 });
      }
    }
  }
  const candidateStride = Math.max(1, Math.ceil(candidates.length / 620));
  const selected = candidates.filter((_, index) => index % candidateStride === 0).slice(0, 620);
  const geometry = new THREE.IcosahedronGeometry(0.72, 1);
  const material = new THREE.MeshStandardMaterial({
    color: '#3d9b61',
    roughness: 0.96,
    metalness: 0,
  });
  const trees = new THREE.InstancedMesh(geometry, material, selected.length);
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const canopyColor = new THREE.Color();
  selected.forEach((candidate, index) => {
    const col = Math.min(rasterWidth - 1, Math.round(candidate.x * (rasterWidth - 1)));
    const row = Math.min(rasterHeight - 1, Math.round(candidate.y * (rasterHeight - 1)));
    const surface = terrainValues[row * rasterWidth + col];
    const terrainY = Number.isFinite(surface) ? (surface - minimum) * verticalScale : 0;
    const crownWidth = 0.72 + candidate.green * 0.72;
    const crownHeight = 0.62 + candidate.green * 0.95;
    const angle = ((index * 137.5) % 360) * Math.PI / 180;
    rotation.setFromAxisAngle(new THREE.Vector3(0, 1, 0), angle);
    position.set(
      (candidate.x - 0.5) * terrainWidth,
      terrainY + crownHeight * 0.72,
      (candidate.y - 0.5) * terrainDepth,
    );
    scale.set(crownWidth, crownHeight, crownWidth);
    matrix.compose(position, rotation, scale);
    trees.setMatrixAt(index, matrix);
    canopyColor.setHSL(0.34 + candidate.green * 0.035, 0.42, 0.25 + candidate.green * 0.15);
    trees.setColorAt(index, canopyColor);
  });
  trees.instanceMatrix.needsUpdate = true;
  if (trees.instanceColor) trees.instanceColor.needsUpdate = true;
  trees.castShadow = true;
  trees.receiveShadow = true;
  group.add(trees);
  return group;
}

function buildBuildingSolids(
  structures: BuildingSolid[],
  texture: THREE.Texture,
  terrainWidth: number,
  terrainDepth: number,
  verticalScale: number,
) {
  const group = new THREE.Group();
  // Three shared material sets, generated lazily. No network textures or new meshes.
  const facadeMaps = new Map<number, FacadeMaps>();
  const roofCanvas = document.createElement('canvas');
  roofCanvas.width = roofCanvas.height = 128;
  const roofContext = roofCanvas.getContext('2d', { willReadFrequently: true })!;
  roofContext.drawImage(texture.image as HTMLImageElement, 0, 0, 128, 128);
  const roofPixels = roofContext.getImageData(0, 0, 128, 128).data;
  const roofMaterial = new THREE.MeshStandardMaterial({
    map: texture,
    color: '#ffffff',
    roughness: 0.9,
    metalness: 0.02,
  });
  const edgeMaterial = new THREE.LineBasicMaterial({ color: '#283b35', transparent: true, opacity: 0.12 });
  for (const structure of structures) {
    const roofTint = new THREE.Color();
    let red = 0, green = 0, blue = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      const px = Math.max(0, Math.min(127, Math.round(structure.center_x * 127) + dx));
      const py = Math.max(0, Math.min(127, Math.round(structure.center_y * 127) + dy));
      const offset = (py * 128 + px) * 4;
      red += roofPixels[offset]; green += roofPixels[offset + 1]; blue += roofPixels[offset + 2];
    }
    roofTint.setRGB(red / 6375, green / 6375, blue / 6375, THREE.SRGBColorSpace);
    roofTint.lerp(new THREE.Color('#dedbd2'), 0.84);
    const variant = Math.abs(Math.imul(structure.id, 31)) % 3;
    if (!facadeMaps.has(variant)) facadeMaps.set(variant, createFacadeMaps(variant));
    const wallMaterial = new THREE.MeshStandardMaterial({ color: roofTint, ...facadeMaps.get(variant)!, vertexColors: true, bumpScale: 0.018, roughness: 0.94, metalness: 0.02 });
    const width = Math.max(0.42, structure.width * terrainWidth);
    const depth = Math.max(0.42, structure.depth * terrainDepth);
    const height = Math.max(0.15, structure.height_m * verticalScale);
    const facade = facadeLayout(structure.height_m, height, width, depth);
    let geometry: THREE.BufferGeometry;
    let materials: THREE.Material | THREE.Material[];
    let building: THREE.Mesh;
    const outline = structure.outline ?? [];
    if (outline.length >= 3) {
      const wallSink = 0.18;
      const shape = new THREE.Shape();
      outline.forEach(([x, y], index) => {
        if (index === 0) shape.moveTo(x - 0.5, -(y - 0.5));
        else shape.lineTo(x - 0.5, -(y - 0.5));
      });
      shape.closePath();
      geometry = new THREE.ExtrudeGeometry(shape, {
        depth: height + wallSink,
        bevelEnabled: false,
      });
      geometry.rotateX(-Math.PI / 2);
      const positions = geometry.attributes.position as THREE.BufferAttribute;
      const uvs = geometry.attributes.uv as THREE.BufferAttribute;
      const normals = geometry.attributes.normal as THREE.BufferAttribute;
      for (let index = 0; index < positions.count; index += 1) {
        if (Math.abs(normals.getY(index)) > 0.5) {
          uvs.setXY(index, positions.getX(index) + 0.5, 0.5 - positions.getZ(index));
        }
      }
      applyFacadeUVs(geometry, terrainWidth, terrainDepth, 0, height + wallSink, facade.rows, facade.tileWidth);
      uvs.needsUpdate = true;
      materials = [roofMaterial, wallMaterial];
      building = new THREE.Mesh(geometry, materials);
      building.scale.set(terrainWidth, 1, terrainDepth);
      building.position.y = structure.base_m * verticalScale - wallSink;
    } else {
      geometry = new THREE.BoxGeometry(width, height, depth);
      materials = [wallMaterial, wallMaterial, roofMaterial, wallMaterial, wallMaterial, wallMaterial];
      building = new THREE.Mesh(geometry, materials);
      building.position.set(
        (structure.center_x - 0.5) * terrainWidth,
        structure.base_m * verticalScale + height / 2,
        (structure.center_y - 0.5) * terrainDepth,
      );
      building.rotation.y = -structure.angle_rad;
      building.updateMatrix();
      const positions = geometry.attributes.position as THREE.BufferAttribute;
      const normals = geometry.attributes.normal as THREE.BufferAttribute;
      const uvs = geometry.attributes.uv as THREE.BufferAttribute;
      const world = new THREE.Vector3();
      for (let i = 0; i < positions.count; i++) {
        world.fromBufferAttribute(positions, i).applyMatrix4(building.matrix);
        if (normals.getY(i) > 0.5) uvs.setXY(i, world.x / terrainWidth + 0.5, 0.5 - world.z / terrainDepth);
      }
      applyFacadeUVs(geometry, 1, 1, -height / 2, height, facade.rows, facade.tileWidth);
      uvs.needsUpdate = true;
    }
    building.castShadow = true;
    building.receiveShadow = true;
    building.userData.heightM = structure.height_m;
    building.userData.structureId = structure.id;
    building.userData.confidence = structure.confidence;
    group.add(building);
    // Flat roof at the existing object-height estimate. Roof-shape estimation
    // is not configured; experimental relief/fits are not part of this viewer.
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 32), edgeMaterial);
    edges.position.copy(building.position);
    edges.rotation.copy(building.rotation);
    edges.scale.copy(building.scale);
    group.add(edges);
  }
  return group;
}

const ROOF_HEIGHT_STATUS = 'Roof height estimation yet to be configured';

// Reuse the exact RGB texture paired with the scene, including the API's
// browser-readable TIFF preview. Never substitute a demo or classified image.
function SceneImageReference({ src, label, compact, onOpen }: { src: string; label: string; compact: boolean; onOpen: () => void }) {
  const [collapsed, setCollapsed] = useState(false);
  const [imageState, setImageState] = useState<'loading' | 'ready' | 'error'>('loading');
  return <section className={`scene-image-reference${compact ? ' compact' : ''}`} aria-label="Original input image reference">
    <button className="scene-image-toggle" type="button" aria-expanded={!collapsed} aria-controls="scene-input-reference-image"
      title={collapsed ? 'Show the original input image' : 'Collapse the original input image'} onClick={() => setCollapsed(value => !value)}>
      <span><i aria-hidden="true">▧</i> ORIGINAL IMAGE</span><span aria-hidden="true">{collapsed ? '+' : '−'}</span>
    </button>
    <div id="scene-input-reference-image" className="scene-image-content" hidden={collapsed}>
      <button type="button" className="scene-image-frame" aria-label="Enlarge original image" title="Open original image · zoom and pan" onClick={onOpen} disabled={imageState !== 'ready'}>
        {imageState !== 'error' && <Image src={src} alt={`Original RGB image for ${label}`} fill unoptimized sizes="236px" style={{ objectFit: 'contain' }}
          draggable={false} onLoad={() => setImageState('ready')} onError={() => setImageState('error')} />}
        {imageState !== 'ready' && <span className="scene-image-placeholder" role="status">{imageState === 'error' ? 'Image preview unavailable' : 'Loading original image…'}</span>}
        <span className="scene-image-tag" aria-hidden="true">⤢ ZOOM</span>
      </button>
      <p title={label}>{label}</p>
      <small>Input reference · not a height map</small>
    </div>
  </section>;
}

function OriginalImageViewer({ src, label, onClose }: { src: string; label: string; onClose: () => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ id: number; x: number; y: number } | null>(null);
  const [view, setView] = useState<ImageView>(FIT_IMAGE_VIEW);
  const [imageSize, setImageSize] = useState({ imageWidth: 0, imageHeight: 0 });
  const [failed, setFailed] = useState(false);
  const [dragging, setDragging] = useState(false);
  const bounds = useCallback(() => ({ width: frameRef.current?.clientWidth ?? 0, height: frameRef.current?.clientHeight ?? 0, ...imageSize }), [imageSize]);
  useEffect(() => {
    const dialog = dialogRef.current;
    dialog?.showModal();
    return () => dialog?.close();
  }, []);
  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault(); event.stopPropagation();
      const rect = frame.getBoundingClientRect();
      const delta = Math.max(-100, Math.min(100, event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? rect.height : 1)));
      setView(previous => zoomImageView(previous, previous.zoom * Math.exp(-delta * .002), bounds(), { x: event.clientX - rect.left - rect.width / 2, y: event.clientY - rect.top - rect.height / 2 }));
    };
    frame.addEventListener('wheel', wheel, { passive: false });
    const observer = new ResizeObserver(() => setView(previous => clampImageView(previous, bounds())));
    observer.observe(frame);
    return () => { frame.removeEventListener('wheel', wheel); observer.disconnect(); };
  }, [bounds]);
  const zoomBy = (factor: number) => setView(previous => zoomImageView(previous, previous.zoom * factor, bounds()));
  return <dialog ref={dialogRef} className="source-image-dialog" aria-labelledby="source-image-title" onCancel={event => { event.preventDefault(); onClose(); }}
    onClick={event => { if (event.target === event.currentTarget) onClose(); }}
    onKeyDown={event => {
      event.stopPropagation();
      if (event.key === '+' || event.key === '=') { event.preventDefault(); zoomBy(1.25); }
      if (event.key === '-') { event.preventDefault(); zoomBy(.8); }
      if (event.key === '0') { event.preventDefault(); setView(FIT_IMAGE_VIEW); }
    }}>
    <div className="source-image-shell">
      <header><div><span>INPUT REFERENCE / RGB</span><h2 id="source-image-title">Original image</h2><p title={label}>{label}</p></div>
        <button type="button" className="source-image-close" aria-label="Close original image" onClick={onClose} autoFocus>×</button>
      </header>
      <div className="source-image-toolbar" role="group" aria-label="Image zoom controls">
        <button type="button" aria-label="Zoom out original image" disabled={view.zoom <= 1 || failed} onClick={() => zoomBy(.8)}>−</button>
        <output aria-label="Image zoom">{view.zoom.toFixed(1)}×</output>
        <button type="button" aria-label="Zoom in original image" disabled={view.zoom >= 8 || failed} onClick={() => zoomBy(1.25)}>+</button>
        <button type="button" onClick={() => setView(FIT_IMAGE_VIEW)}>Fit image</button>
        <span>Scroll to zoom · drag to pan</span>
      </div>
      <div ref={frameRef} className={`source-image-stage${dragging ? ' dragging' : ''}`} data-zoom={view.zoom} data-pan-x={view.x} data-pan-y={view.y}
        onPointerDown={event => {
          if (event.button !== 0 || failed) return;
          event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
          dragRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY }; setDragging(true);
        }}
        onPointerMove={event => {
          const drag = dragRef.current;
          if (!drag || drag.id !== event.pointerId) return;
          const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
          drag.x = event.clientX; drag.y = event.clientY;
          setView(previous => clampImageView({ ...previous, x: previous.x + dx, y: previous.y + dy }, bounds()));
        }}
        onPointerUp={event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); dragRef.current = null; setDragging(false); }}
        onPointerCancel={() => { dragRef.current = null; setDragging(false); }}
        onLostPointerCapture={() => { dragRef.current = null; setDragging(false); }}>
        {!failed && <div className="source-image-transform" style={{ transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})` }}>
          <Image src={src} alt={`Enlarged original RGB image for ${label}`} fill unoptimized sizes="94vw" style={{ objectFit: 'contain' }} draggable={false}
            onLoad={event => setImageSize({ imageWidth: event.currentTarget.naturalWidth, imageHeight: event.currentTarget.naturalHeight })} onError={() => setFailed(true)} />
        </div>}
        {failed && <p role="status">Image preview unavailable. Close this view and reload the source image.</p>}
      </div>
      <footer><span>{imageSize.imageWidth ? `${imageSize.imageWidth.toLocaleString()} × ${imageSize.imageHeight.toLocaleString()} px · RGB preview` : 'Loading image…'}</span><span>Zoom is relative to fit · Esc to close</span></footer>
    </div>
  </dialog>;
}

export default function TerrainWorkspace() {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewportRef = useRef<HTMLElement>(null);
  const satelliteInputRef = useRef<HTMLInputElement>(null);
  const localHeightInputRef = useRef<HTMLInputElement>(null);
  const rgbInputRef = useRef<HTMLInputElement>(null);
  const demInputRef = useRef<HTMLInputElement>(null);
  const gcpInputRef = useRef<HTMLInputElement>(null);
  const referenceInputRef = useRef<HTMLInputElement>(null);
  const readoutRef = useRef<HTMLDivElement>(null);
  const [sceneInput, setSceneInput] = useState<SceneInput>({
    heightBuffer: null,
    analysisBuffer: null,
    missionBuffer: null,
    missionSource: null,
    buildingProbabilityBuffer: null,
    vegetationProbabilityBuffer: null,
    semanticClassBuffer: null,
    confidenceBuffer: null,
    textureUrl: '',
    label: 'No image loaded',
    productKind: 'relative',
    proof: 'Upload your image or choose one of the four reference scenes.',
    structures: [],
  });
  const [mode, setMode] = useState<ViewMode>('orbit');
  const [exploreMode, setExploreMode] = useState<ExploreMode>('fly');
  const [exploreMenuOpen, setExploreMenuOpen] = useState(false);
  const [headMotion, setHeadMotion] = useState(true);
  const exploreModeRef = useRef<ExploreMode>('fly');
  const headMotionRef = useRef(true);
  const enterWalkRef = useRef<(() => boolean) | null>(null);
  const exploreMenuRef = useRef<HTMLDivElement>(null);
  const [flyPointerLocked, setFlyPointerLocked] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [imageZoomSource, setImageZoomSource] = useState<string | null>(null);
  const [fullMapOpen, setFullMapOpen] = useState(false);
  const fullMapOpenRef = useRef(false);
  const mapPointerReleaseRef = useRef(false);
  const mapFeedRef = useRef<MapFrame | null>(null);
  const [sceneDetail, setSceneDetail] = useState<{ heightBuffer: ArrayBuffer | null; kind: 'height' | 'scale' } | null>(null);
  const openSceneDetail = sceneDetail?.heightBuffer === sceneInput.heightBuffer ? sceneDetail?.kind : null;
  const [windowFullscreen, setWindowFullscreen] = useState(false);
  const fullscreenRef = useRef(false);
  const windowFullscreenRef = useRef(false);
  const fullscreenSessionRef = useRef(0);
  const pointerEscapeUntilRef = useRef(0);
  const [surfaceStyle, setSurfaceStyle] = useState<SurfaceStyle>('optical');
  const [smoothTerrain, setSmoothTerrain] = useState(true);
  const smoothTerrainRef = useRef(true);
  const [terrainSmoothness, setTerrainSmoothness] = useState(3);
  const terrainSmoothnessRef = useRef(3);
  const terrainPresentationRef = useRef<((smooth: boolean, strength: number) => void) | null>(null);
  const [metricGrid, setMetricGrid] = useState<MetricGridSpec | null>(null);
  const [importPending, setImportPending] = useState(false);
  const [revealPhase, setRevealPhase] = useState<RevealPhase | null>(null);
  const importPendingRef = useRef(false);
  const revealActiveRef = useRef(false);
  const skipRevealRef = useRef(false);
  const lastRevealedBufferRef = useRef<ArrayBuffer | null>(null);
  const revealProgressRef = useRef<HTMLProgressElement>(null);
  const pendingValidationOpenRef = useRef(false);
  // Aerial relief is only a few metres across a wide tile, so start in a
  // conventional, explicitly labelled presentation exaggeration. Inspect
  // readouts and exported rasters always retain the unscaled model values.
  const [exaggeration, setExaggeration] = useState(1.7);
  const [automaticRelief, setAutomaticRelief] = useState(true);
  const [recommendedExaggeration, setRecommendedExaggeration] = useState(1);
  const automaticReliefRef = useRef(true);
  const [layers, setLayers] = useState<Layers>({ grid: true, wireframe: false, canopy: false, contours: false });
  const [profile, setProfile] = useState<TerrainProfileData | null>(null);
  const [profilePicking, setProfilePicking] = useState(false);
  const profileResetRef = useRef(0);
  const cameraPresetRef = useRef<((top: boolean) => void) | null>(null);
  const scaleBarRef = useRef<HTMLDivElement>(null);
  const [stats, setStats] = useState<SceneStats | null>(null);
  const [inspection, setInspection] = useState<InspectionData | null>(null);
  const [focusClass, setFocusClass] = useState<FocusClass | null>(null);
  const focusClassRef = useRef<FocusClass | null>(null);
  const [classificationState, setClassificationState] = useState<SixClassState | null>(null);
  const classification = stateForScene(classificationState, sceneInput.textureUrl);
  const [classViewState, setClassViewState] = useState<{ sceneKey: string; view: SixClassView } | null>(null);
  const classView = classViewState?.sceneKey === sceneInput.textureUrl ? classViewState.view : DEFAULT_CLASS_VIEW;
  const sixClassLayer = useMemo<SixClassLayer | null>(() => classification.result
    ? { labels: classification.result.labels, width: classification.result.width, height: classification.result.height, ...classView }
    : null, [classification.result, classView]);
  const classificationStatusRef = useRef(classification.status);
  useEffect(() => { classificationStatusRef.current = classification.status; }, [classification.status]);
  const sixClassRef = useRef<{ layer: SixClassLayer; texture: THREE.DataTexture } | null>(null);
  const onClassView = useCallback((view: SixClassView) => {
    setClassViewState({ sceneKey: sceneInput.textureUrl, view });
    setFocusClass(null); focusClassRef.current = null;
  }, [sceneInput.textureUrl]);
  useEffect(() => {
    if (!sixClassLayer) { sixClassRef.current = null; return; }
    const texture = new THREE.DataTexture(classLayerPixels(sixClassLayer), sixClassLayer.width, sixClassLayer.height);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.NearestFilter; texture.magFilter = THREE.NearestFilter;
    texture.unpackAlignment = 1; texture.needsUpdate = true;
    sixClassRef.current = { layer: sixClassLayer, texture };
    return () => { sixClassRef.current = null; texture.dispose(); };
  }, [sixClassLayer, sceneInput.textureUrl]);
  const resetClassGeometryRef = useRef<(() => void) | null>(null);
  const [importHandoff, setImportHandoff] = useState(false);
  const [flightInspection, setFlightInspection] = useState<InspectionData | null>(null);
  const [flightPinned, setFlightPinned] = useState(false);
  const activeInspectionRef = useRef<InspectionData | null>(null);
  useEffect(() => {
    activeInspectionRef.current = mode === 'inspect' ? inspection : mode === 'explore' ? flightInspection : null;
  }, [mode, inspection, flightInspection]);
  const [flightBlocked, setFlightBlocked] = useState(false);
  const [status, setStatus] = useState('Ready for an image');
  const [error, setError] = useState<string | null>(null);
  const [engineStatus, setEngineStatus] = useState<'checking' | 'ready' | 'offline'>('checking');
  const [demFile, setDemFile] = useState<File | null>(null);
  const [gcpFile, setGcpFile] = useState<File | null>(null);
  const [autoTerrain, setAutoTerrain] = useState(true);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [referenceKind, setReferenceKind] = useState<'ndsm' | 'dsm'>('ndsm');
  const [validationMetrics, setValidationMetrics] = useState<ValidationMetrics | null>(null);
  const [generatedProducts, setGeneratedProducts] = useState<GeneratedProducts | null>(null);
  const [processingNotice, setProcessingNotice] = useState<string | null>(null);
  const [validationEvidenceOpen, setValidationEvidenceOpen] = useState(false);
  const [landscapeEvaluationOpen, setLandscapeEvaluationOpen] = useState(false);
  const [evidenceOverride, setEvidenceOverride] = useState<DemoValidationEvidence & { textureUrl: string } | null>(null);
  const [activeDemoId, setActiveDemoId] = useState<DemoSceneId | null>(null);
  const [demoValidation, setDemoValidation] = useState<DemoValidationEvidence | null>(null);
  const [missionScreenOpen, setMissionScreenOpen] = useState(false);
  const [floodOverlayEnabled, setFloodOverlayEnabled] = useState(true);
  const [scenarioRiseM, setScenarioRiseM] = useState(5);
  const [missionAnalysis, setMissionAnalysis] = useState<MissionAnalysisData | null>(null);
  const [sightlineStartSelected, setSightlineStartSelected] = useState(false);
  const [sightlineResult, setSightlineResult] = useState<SightlineResult | null>(null);
  const demoRequestRef = useRef(0);
  const sightlineResetRef = useRef(0);

  useEffect(() => {
    let disposed = false;
    const checkEngine = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/health`, { cache: 'no-store' });
        if (!disposed) setEngineStatus(response.ok ? 'ready' : 'offline');
      } catch {
        if (!disposed) setEngineStatus('offline');
      }
    };
    void checkEngine();
    const timer = window.setInterval(() => void checkEngine(), 5000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const onFullscreenChange = () => {
      const native = document.fullscreenElement === viewportRef.current;
      fullscreenRef.current = native || windowFullscreenRef.current;
      setIsFullscreen(fullscreenRef.current);
      setSceneDetail(null);
      if (!native) {
        fullscreenKeyboard(navigator as Navigator & { keyboard?: EscapeKeyboard })?.unlock();
        if (!windowFullscreenRef.current) fullscreenSessionRef.current += 1;
      }
    };
    document.addEventListener('fullscreenchange', onFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', onFullscreenChange);
      fullscreenSessionRef.current += 1;
      fullscreenKeyboard(navigator as Navigator & { keyboard?: EscapeKeyboard })?.unlock();
    };
  }, []);
  const modeRef = useRef(mode);
  const closeExploreMap = useCallback((recapture = true) => {
    fullMapOpenRef.current = false;
    setFullMapOpen(false);
    // Re-capture only as part of the user's M/close-button gesture. If the
    // browser refuses, the existing click-to-capture control remains available.
    const canvas = containerRef.current?.querySelector('canvas');
    if (recapture && modeRef.current === 'explore' && canvas && document.pointerLockElement !== canvas) {
      try { void Promise.resolve(canvas.requestPointerLock()).catch(() => setFlyPointerLocked(false)); }
      catch { setFlyPointerLocked(false); }
    }
  }, []);
  const toggleExploreMap = useCallback(() => {
    if (modeRef.current !== 'explore' || importPendingRef.current || revealActiveRef.current) return;
    if (fullMapOpenRef.current) { closeExploreMap(); return; }
    fullMapOpenRef.current = true;
    setFullMapOpen(true);
    if (document.pointerLockElement && containerRef.current?.contains(document.pointerLockElement)) {
      mapPointerReleaseRef.current = true;
      document.exitPointerLock();
    }
  }, [closeExploreMap]);
  const layersRef = useRef(layers);
  const styleRef = useRef(surfaceStyle);
  const exaggerationRef = useRef(exaggeration);
  function applyDisplayScale(value: number, automatic = false) {
    automaticReliefRef.current = automatic;
    setAutomaticRelief(automatic);
    exaggerationRef.current = value;
    setExaggeration(value);
  }
  const missionScreenRef = useRef({ enabled: false, thresholdM: Number.NEGATIVE_INFINITY });

  useEffect(() => {
    modeRef.current = mode;
    layersRef.current = layers;
    styleRef.current = surfaceStyle;
    exaggerationRef.current = exaggeration;
    if (mode !== 'inspect' || surfaceStyle !== 'metric') focusClassRef.current = null;
  }, [mode, layers, surfaceStyle, exaggeration]);

  useEffect(() => {
    missionScreenRef.current = {
      enabled: missionScreenOpen && floodOverlayEnabled && Boolean(missionAnalysis),
      thresholdM: missionAnalysis ? missionAnalysis.minimumM + scenarioRiseM : Number.NEGATIVE_INFINITY,
    };
  }, [floodOverlayEnabled, missionAnalysis, missionScreenOpen, scenarioRiseM]);

  useEffect(() => {
    const onPointerLockChange = () => {
      const lockedElement = document.pointerLockElement;
      const lockedToTerrain = Boolean(
        lockedElement
        && containerRef.current
        && containerRef.current.contains(lockedElement),
      );
      setFlyPointerLocked(lockedToTerrain);
      const mapRelease = mapPointerReleaseRef.current;
      if (!lockedToTerrain) mapPointerReleaseRef.current = false;
      if (!lockedToTerrain && modeRef.current === 'explore' && !fullMapOpenRef.current && !mapRelease) {
        // Browsers without Keyboard Lock can deliver pointer loss before the
        // corresponding Escape key event. Do not consume that same press twice.
        if (fullscreenRef.current) pointerEscapeUntilRef.current = performance.now() + 80;
        modeRef.current = 'orbit';
        setMode('orbit');
      }
    };
    const onPointerLockError = () => {
      setFlyPointerLocked(false);
    };
    document.addEventListener('pointerlockchange', onPointerLockChange);
    document.addEventListener('pointerlockerror', onPointerLockError);
    return () => {
      document.removeEventListener('pointerlockchange', onPointerLockChange);
      document.removeEventListener('pointerlockerror', onPointerLockError);
    };
  }, []);

  const selectViewMode = useCallback((nextMode: ViewMode) => {
    if (importPendingRef.current || revealActiveRef.current) return;
    if (flyPointerLocked) return;
    setExploreMenuOpen(false);
    setSceneDetail(null);
    fullMapOpenRef.current = false; setFullMapOpen(false);
    if (nextMode !== 'inspect' || modeRef.current !== 'inspect') { focusClassRef.current = null; setFocusClass(null); resetClassGeometryRef.current?.(); }
    if (nextMode === 'sightline' && modeRef.current === 'sightline') nextMode = 'orbit';
    sightlineResetRef.current += 1;
    setSightlineStartSelected(false);
    setSightlineResult(null);
    setFlightInspection(null);
    setFlightBlocked(false);
    setInspection(null);
    if (nextMode !== 'explore') {
      if (flyPointerLocked) return;
      modeRef.current = nextMode;
      setMode(nextMode);
      return;
    }

    const canvas = containerRef.current?.querySelector('canvas');
    if (!canvas) {
      setError('The 3D scene is still loading. Enter Explore after the mesh is ready.');
      return;
    }
    if (exploreModeRef.current === 'walk' && !enterWalkRef.current?.()) {
      setError('No safe walking surface was found here. Try Fly: roofs, tall canopy, steep edges and missing surface data are not walkable.');
      return;
    }
    setError(null);
    modeRef.current = 'explore';
    setMode('explore');
    try {
      void Promise.resolve(canvas.requestPointerLock()).catch(() => {
        setFlyPointerLocked(false);
      });
    } catch {
      setFlyPointerLocked(false);
    }
  }, [flyPointerLocked]);

  const changeSurfaceStyle = (next: SurfaceStyle) => {
    if (next !== 'metric') { focusClassRef.current = null; setFocusClass(null); }
    styleRef.current = next;
    setSurfaceStyle(next);
  };

  const startExplore = (kind: ExploreMode) => {
    exploreModeRef.current = kind;
    setExploreMode(kind);
    selectViewMode('explore');
  };

  useEffect(() => {
    if (!exploreMenuOpen) return;
    const dismiss = (event: PointerEvent) => {
      if (event.target instanceof Node && !exploreMenuRef.current?.contains(event.target)) setExploreMenuOpen(false);
    };
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') setExploreMenuOpen(false); };
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('pointerdown', dismiss); document.removeEventListener('keydown', escape); };
  }, [exploreMenuOpen]);

  const exitFullscreen = useCallback(async () => {
    setSceneDetail(null);
    fullscreenSessionRef.current += 1;
    windowFullscreenRef.current = false;
    fullscreenRef.current = false;
    setWindowFullscreen(false);
    setIsFullscreen(false);
    fullscreenKeyboard(navigator as Navigator & { keyboard?: EscapeKeyboard })?.unlock();
    if (document.fullscreenElement === viewportRef.current) {
      try { await document.exitFullscreen(); }
      catch {
        fullscreenRef.current = document.fullscreenElement === viewportRef.current;
        setIsFullscreen(fullscreenRef.current);
        setError('The browser could not exit fullscreen. Press and hold Esc to use its safety exit.');
      }
    }
  }, []);

  useEffect(() => {
    const onEscape = (event: KeyboardEvent) => {
      if (event.code === 'KeyM' && modeRef.current === 'explore' && !event.ctrlKey && !event.altKey && !event.metaKey
        && !(event.target instanceof HTMLElement && event.target.closest('input, textarea, select, [contenteditable="true"]'))) {
        event.preventDefault(); event.stopImmediatePropagation();
        if (!event.repeat) toggleExploreMap();
        return;
      }
      if (event.code !== 'Escape' && event.key !== 'Escape') return;
      if (fullMapOpenRef.current) {
        event.preventDefault(); event.stopImmediatePropagation();
        // Native Escape can immediately cancel a lock acquired on this very
        // key event. Keep Explore active; M/button closes can recapture, whereas
        // Esc leaves the existing click-to-capture prompt available.
        if (!event.repeat) closeExploreMap(false);
        return;
      }
      if (viewportRef.current?.querySelector('.source-image-dialog[open]')) {
        event.preventDefault(); event.stopImmediatePropagation();
        if (!event.repeat) setImageZoomSource(null);
        return;
      }
      if (modeRef.current !== 'explore' && !fullscreenRef.current) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      if (event.repeat || performance.now() < pointerEscapeUntilRef.current) return;
      const action = fullscreenEscapeAction(modeRef.current, fullscreenRef.current, event.repeat);
      if (action === 'leave-explore') {
        modeRef.current = 'orbit';
        setMode('orbit');
        setExploreMenuOpen(false);
        setFlightInspection(null);
        setFlightPinned(false);
        setFlightBlocked(false);
        setFlyPointerLocked(false);
        sightlineResetRef.current += 1;
        if (document.pointerLockElement && containerRef.current?.contains(document.pointerLockElement)) document.exitPointerLock();
      } else if (action === 'leave-fullscreen') {
        void exitFullscreen();
      }
    };
    const onKeyUp = (event: KeyboardEvent) => { if (event.code === 'Escape') pointerEscapeUntilRef.current = 0; };
    // Capture before the scene's navigation listener, so a single key cannot
    // both leave Explore and then fall through to the Orbit/fullscreen action.
    window.addEventListener('keydown', onEscape, true);
    window.addEventListener('keyup', onKeyUp, true);
    return () => { window.removeEventListener('keydown', onEscape, true); window.removeEventListener('keyup', onKeyUp, true); };
  }, [exitFullscreen, toggleExploreMap, closeExploreMap]);

  const toggleFullscreen = useCallback(async () => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    if (fullscreenRef.current) { await exitFullscreen(); return; }
    const session = ++fullscreenSessionRef.current;
    const keyboard = fullscreenKeyboard(navigator as Navigator & { keyboard?: EscapeKeyboard });
    const enterWindowView = () => {
      windowFullscreenRef.current = true;
      fullscreenRef.current = true;
      setWindowFullscreen(true);
      setIsFullscreen(true);
    };
    try {
      setError(null);
      // Without Escape capture, native fullscreen always wins over the app.
      // A full-window view preserves the requested two-step exit in that case.
      if (!keyboard) { enterWindowView(); return; }
      await viewport.requestFullscreen();
      if (session !== fullscreenSessionRef.current) return;
      try {
        await keyboard.lock(['Escape']);
        if (session !== fullscreenSessionRef.current || document.fullscreenElement !== viewport) keyboard.unlock();
      } catch {
        if (session !== fullscreenSessionRef.current) return;
        enterWindowView();
        if (document.fullscreenElement === viewport) await document.exitFullscreen();
      }
    } catch {
      if (session === fullscreenSessionRef.current) enterWindowView();
    }
  }, [exitFullscreen]);

  const resetSightline = useCallback(() => {
    sightlineResetRef.current += 1;
    setSightlineStartSelected(false);
    setSightlineResult(null);
  }, []);

  const loadDemoScene = useCallback(async (sceneId: DemoSceneId) => {
    fullMapOpenRef.current = false; setFullMapOpen(false); mapFeedRef.current = null;
    const requestId = ++demoRequestRef.current;
    const demo = DEMO_SCENES[sceneId];
    const fetchBuffer = async (url: string | null, label: string) => {
      if (!url) return null;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${label} returned ${response.status}`);
      return response.arrayBuffer();
    };

    setError(null);
    setGeneratedProducts(null);
    setValidationMetrics(null);
    setProcessingNotice(null);
    setStats(null);
    setFocusClass(null);
    focusClassRef.current = null;
    setInspection(null);
    setMissionAnalysis(null);
    resetSightline();
    setActiveDemoId(sceneId);
    applyDisplayScale(1, true);
    setDemoValidation(demo.validation);
    setStatus(`Loading ${demo.shortLabel.toLowerCase()} demonstration…`);
    setSceneInput({
      heightBuffer: null,
      analysisBuffer: null,
      missionBuffer: null,
      missionSource: null,
      buildingProbabilityBuffer: null,
      vegetationProbabilityBuffer: null,
      semanticClassBuffer: null,
      confidenceBuffer: null,
      textureUrl: demo.textureUrl,
      label: demo.label,
      productKind: demo.productKind,
      proof: demo.proof,
      structures: [],
    });

    try {
      const [heightBuffer, analysisBuffer, structurePayload, buildingProbabilityBuffer, vegetationProbabilityBuffer, semanticClassBuffer, confidenceBuffer, terrainBuffer] = await Promise.all([
        fetchBuffer(demo.heightUrl, 'Demo height raster'),
        fetchBuffer(demo.analysisUrl, 'Demo analysis raster'),
        fetch(demo.structuresUrl).then((response) => {
          if (!response.ok) throw new Error(`Demo structures returned ${response.status}`);
          return response.json() as Promise<{ structures: BuildingSolid[] }>;
        }),
        fetchBuffer(demo.buildingProbabilityUrl, 'Demo building probability'),
        fetchBuffer(demo.vegetationProbabilityUrl, 'Demo vegetation probability'),
        fetchBuffer(demo.semanticClassUrl, 'Demo semantic class'),
        fetchBuffer(demo.confidenceUrl, 'Demo confidence'),
        fetchBuffer(demo.terrainUrl ?? null, 'Terrain reference'),
      ]);
      if (requestId !== demoRequestRef.current || !heightBuffer || !analysisBuffer) return;
      setSceneInput({
        heightBuffer,
        analysisBuffer,
        terrainBuffer,
        missionBuffer: demo.productKind === 'dsm' ? analysisBuffer.slice(0) : null,
        missionSource: demo.productKind === 'dsm' ? 'bundled_absolute_dsm' : null,
        buildingProbabilityBuffer,
        vegetationProbabilityBuffer,
        semanticClassBuffer,
        confidenceBuffer,
        textureUrl: demo.textureUrl,
        label: demo.label,
        productKind: demo.productKind,
        proof: demo.proof,
        structures: structurePayload.structures,
      });
    } catch (reason) {
      if (requestId !== demoRequestRef.current) return;
      setError(reason instanceof Error ? reason.message : String(reason));
      setStatus(`${demo.shortLabel} demonstration failed to load`);
    }
  }, [resetSightline]);

  useEffect(() => {
    if (!containerRef.current || !sceneInput.heightBuffer) return;
    const container = containerRef.current;
    let disposed = false;
    let frame = 0;
    modeRef.current = 'orbit';
    setMode('orbit');
    setExploreMenuOpen(false);
    setStats(null);
    setFocusClass(null);
    focusClassRef.current = null;
    setStatus('Building interactive terrain mesh…');
    setMetricGrid(null);
    setProfile(null);
    setProfilePicking(false);
    setRevealPhase(null);
    revealActiveRef.current = false;
    skipRevealRef.current = false;
    let revealStartedAt: number | null = null;
    let lastRevealPhase: RevealPhase | null = null;
    setError(null);
    setProcessingNotice(null);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#07100e');
    scene.fog = new THREE.FogExp2('#07100e', 0.0018);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1200);
    camera.position.set(72, 48, 92);
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    const terrainLighting = sceneInput.productKind === 'dsm' || Boolean(sceneInput.inferHeightUnits);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = terrainLighting ? 1.12 : 1.28;
    renderer.shadowMap.enabled = true;
    container.replaceChildren(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.maxPolarAngle = Math.PI * 0.48;
    controls.minDistance = 15;
    controls.maxDistance = 260;
    controls.target.set(0, 2.5, 0);

    // Terrain imagery already contains lighting and shadows. Softer fill keeps
    // small raster facets from looking like steep spikes; object scenes retain
    // their established lighting and facade contrast.
    scene.add(new THREE.HemisphereLight('#e5f2ff', '#45564c', terrainLighting ? 2.2 : 1.25));
    scene.add(new THREE.AmbientLight('#ffffff', terrainLighting ? .8 : .3));
    const sun = new THREE.DirectionalLight('#fff2cf', terrainLighting ? .8 : 2.6);
    sun.position.set(-55, 90, 34);
    sun.castShadow = true;
    scene.add(sun);
    const fill = new THREE.DirectionalLight('#42b8ff', terrainLighting ? .25 : .55);
    fill.position.set(45, 24, -60);
    scene.add(fill);

    const keys = new Set<string>();
    let walkJumpRequested = false;
    const movementKeys = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyG', 'Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ShiftLeft', 'ShiftRight']);
    const onKeyDown = (event: KeyboardEvent) => {
      if (fullMapOpenRef.current) { keys.clear(); walkJumpRequested = false; return; }
      if (event.code === 'Escape' && modeRef.current !== 'explore') {
        focusClassRef.current = null; setFocusClass(null);
        sightlineResetRef.current += 1;
        setSightlineStartSelected(false);
        setSightlineResult(null);
        setFlightInspection(null);
        modeRef.current = 'orbit';
        setMode('orbit');
        return;
      }
      if (event.target instanceof HTMLElement && event.target.closest('input, textarea, select, [contenteditable="true"]')) return;
      if (event.code === 'Escape' && modeRef.current === 'explore' && document.pointerLockElement !== renderer.domElement) {
        modeRef.current = 'orbit';
        setMode('orbit');
        return;
      }
      if (!movementKeys.has(event.code) || modeRef.current !== 'explore') return;
      event.preventDefault();
      if (event.code === 'Space' && exploreModeRef.current === 'walk' && !event.repeat && !keys.has('Space')) walkJumpRequested = true;
      keys.add(event.code);
    };
    const onKeyUp = (event: KeyboardEvent) => keys.delete(event.code);
    const clearMovementKeys = () => { keys.clear(); walkJumpRequested = false; };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', clearMovementKeys);

    const flyLook = new THREE.Euler(0, 0, 0, 'YXZ');
    let flyYaw = 0;
    let flyPitch = 0;
    let flyLookReady = false;
    let previousFlyMode = false;
    let walkWorld: WalkWorld | null = null;
    let walkFeet: WalkPoint | null = null;
    let walkMotion: WalkMotion | null = null;
    let walkDistance = 0, walkBobDistance = 0, walkStrength = 0;
    const walkVelocity = { x: 0, z: 0 };
    let savedWalkCamera: { position: THREE.Vector3; target: THREE.Vector3; fov: number; near: number } | null = null;
    const restoreWalkCamera = () => {
      if (!savedWalkCamera) return false;
      camera.position.copy(savedWalkCamera.position);
      controls.target.copy(savedWalkCamera.target);
      camera.fov = savedWalkCamera.fov;
      camera.near = savedWalkCamera.near;
      camera.updateProjectionMatrix();
      camera.lookAt(controls.target);
      savedWalkCamera = null;
      walkFeet = null;
      walkMotion = null;
      walkJumpRequested = false;
      return true;
    };
    const syncFlyLook = () => {
      flyLook.setFromQuaternion(camera.quaternion, 'YXZ');
      flyPitch = flyLook.x;
      flyYaw = flyLook.y;
      flyLookReady = true;
    };
    const onScenePointerLockChange = () => {
      const lockedToScene = document.pointerLockElement === renderer.domElement;
      if (lockedToScene) {
        syncFlyLook();
        return;
      }
      clearMovementKeys();
    };
    const onMouseMove = (event: MouseEvent) => {
      if (modeRef.current !== 'explore' || fullMapOpenRef.current) return;
      const lockedToScene = document.pointerLockElement === renderer.domElement;
      if (!lockedToScene && event.target !== renderer.domElement) return;
      if (!flyLookReady) syncFlyLook();
      const sensitivity = 0.0022;
      flyYaw -= event.movementX * sensitivity;
      flyPitch -= event.movementY * sensitivity;
      flyPitch = Math.max(-Math.PI / 2 + 0.04, Math.min(Math.PI / 2 - 0.04, flyPitch));
      camera.rotation.set(flyPitch, flyYaw, 0, 'YXZ');
    };
    const activateFlyPointerLock = () => {
      if (modeRef.current !== 'explore' || fullMapOpenRef.current || document.pointerLockElement === renderer.domElement) return;
      void Promise.resolve(renderer.domElement.requestPointerLock()).catch(() => undefined);
    };
    document.addEventListener('pointerlockchange', onScenePointerLockChange);
    document.addEventListener('mousemove', onMouseMove);

    let terrain: THREE.Mesh | null = null;
    let terrainBoundary: THREE.Group | null = null;
    let grid: THREE.GridHelper | null = null;
    let metricFloor: THREE.LineSegments | null = null;
    const metricMaterials: ReturnType<typeof createMetricGridMaterial>[] = [];
    let classMap: THREE.DataTexture | null = null;
    let focusClasses: Uint8Array | null = null;
    let sourceMeshHeights: Float32Array | null = null;
    let focusWeights: ClassWeights = [...ALL_CLASS_HEIGHTS];
    let focusFrom: ClassWeights = [...ALL_CLASS_HEIGHTS];
    let focusTarget: FocusClass | null = null;
    let focusStartedAt = 0, focusAnimating = false, focusGlow = 0;
    let lastFocusColour = -1;
    let handoffUntil = 0;
    const resetClassGeometry = () => {
      focusWeights = [...ALL_CLASS_HEIGHTS]; focusFrom = [...ALL_CLASS_HEIGHTS];
      focusTarget = null; focusAnimating = false; focusGlow = 0;
      if (terrain && sourceMeshHeights && focusClasses) {
        const position = terrain.geometry.getAttribute('position') as THREE.BufferAttribute;
        focusedVertexHeights(sourceMeshHeights, focusClasses, focusWeights, position.array as Float32Array);
        position.needsUpdate = true; terrain.geometry.computeVertexNormals();
      }
      if (buildings) { buildings.scale.y = exaggerationRef.current; buildings.visible = true; }
      for (const controller of metricMaterials) controller.setClassFocus(classMap, -1, 0);
    };
    resetClassGeometryRef.current = resetClassGeometry;
    let metricBlend = styleRef.current === 'metric' ? 1 : 0;
    const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
    let reducedMotion = motionPreference.matches;
    const onMotionPreference = () => { reducedMotion = motionPreference.matches; };
    motionPreference.addEventListener('change', onMotionPreference);
    let lastPhotoStyle: 'optical' | 'elevation' = styleRef.current === 'elevation' ? 'elevation' : 'optical';
    let wireframe: THREE.Mesh | null = null;
    let canopy: THREE.Group | null = null;
    let buildings: THREE.Group | null = null;
    let floodOverlay: THREE.Mesh | null = null;
    let highGroundMarkers: THREE.Group | null = null;
    const sightlineVisuals = new THREE.Group();
    scene.add(sightlineVisuals);
    const profileVisuals = new THREE.Group();
    scene.add(profileVisuals);
    let profileStart: TerrainPoint | null = null;
    let profileComplete = false;
    let handledProfileReset = profileResetRef.current;
    const clearProfile = () => {
      for (const object of [...profileVisuals.children]) {
        profileVisuals.remove(object);
        if (object instanceof THREE.Line || object instanceof THREE.Mesh) {
          object.geometry.dispose();
          (Array.isArray(object.material) ? object.material : [object.material]).forEach(m => m.dispose());
        }
      }
      profileStart = null; profileComplete = false;
      setProfile(null); setProfilePicking(false);
    };
    let opticalMaterial: THREE.MeshStandardMaterial | null = null;
    let elevationMaterial: THREE.MeshStandardMaterial | null = null;
    let terrainStats: SceneStats | null = null;
    let inspectionQuery: SurfaceQuery | null = null;
    const selectionGlow = createSelectionGlow();
    scene.add(selectionGlow.group);
    const selectionBuildings = new Map<number, THREE.Mesh>();
    let selectionRegions: SelectionRegionIndex | null = null;
    let selectionRegionSource: Uint8Array | null | undefined;
    let selectionRegionVersion = 0;
    let lastSelectionKey: string | null = null;
    let selectionSurface: ((u: number, v: number) => number | null) | null = null;
    const selectionRegionAt = (u: number, v: number) => {
      if (!inspectionQuery) return null;
      const query = inspectionQuery, layer = sixClassRef.current?.layer;
      const source = layer?.labels ?? null;
      const labelAt = (u: number, v: number) => {
        if (rasterValue(query.raw, u, v) === null) return -1;
        if (layer) {
          const code = layer.labels[Math.min(layer.height - 1, Math.floor(v * layer.height)) * layer.width + Math.min(layer.width - 1, Math.floor(u * layer.width))];
          return code < 6 ? code : -1;
        }
        const code = surfaceClassCode(query, u, v);
        return code < 3 ? code : -1;
      };
      if (!selectionRegions || selectionRegionSource !== source) {
        selectionRegions = createSelectionRegions(layer?.width ?? query.raw.stats.width, layer?.height ?? query.raw.stats.height, labelAt);
        selectionRegionSource = source; selectionRegionVersion++;
      }
      return selectionRegions.at(u, v, labelAt(u, v));
    };
    const clearSelectionGlow = () => {
      selectionGlow.clear(); lastSelectionKey = null;
      container.dataset.selectionGlow = 'off';
    };
    const updateSelectionGlow = (delta: number, displayScale: number) => {
      const data = activeInspectionRef.current;
      if (!data || !terrain || !selectionSurface || importPendingRef.current || revealActiveRef.current
        || (modeRef.current !== 'inspect' && modeRef.current !== 'explore')) { clearSelectionGlow(); return; }
      const buildingId = /^building:(\d+)(?:\||$)/.exec(data.targetKey ?? '')?.[1];
      const building = buildingId !== undefined ? selectionBuildings.get(Number(buildingId)) : undefined;
      const uv = data.sampleUv;
      const region = !building && uv ? selectionRegionAt(...uv) : null;
      const kind = building ? 'building' : region ? 'region' : 'point';
      const identity = building ? `building:${buildingId}` : region ? `region:${selectionRegionVersion}:${region}` : `point:${uv?.join(':')}`;
      const key = `${identity}:${(terrain.geometry.getAttribute('position') as THREE.BufferAttribute).version}`;
      if (key !== lastSelectionKey) {
        if (building) selectionGlow.set(buildingSelectionEdges(building));
        else if (uv) {
          const edges = region ? selectionRegions!.boundary(region) : sampleLocator(uv[0], uv[1], .009, .009);
          selectionGlow.set(drapeSelectionEdges(edges, sceneWidth, sceneDepth, selectionSurface, Math.max(sceneWidth, sceneDepth) * .00006));
        } else { clearSelectionGlow(); return; }
        lastSelectionKey = key;
        container.dataset.selectionGlow = JSON.stringify({ kind, identity, target: data.targetKey, rawValue: data.rawValue });
      }
      selectionGlow.update(delta, displayScale, reducedMotion);
    };
    let vertexMissionElevations: Float32Array | null = null;
    let sightlineSurface: { values: Float32Array; stats: SceneStats } | null = null;
    let sightlineStart: { normalized: SightlinePoint; localPoint: THREE.Vector3 } | null = null;
    let sceneVerticalScale = 1;
    let sceneWidth = 112, sceneDepth = 112, sceneMinimum = 0;
    let handledSightlineReset = sightlineResetRef.current;
    let sightlineComplete = false;
    let flightGround: (x: number, z: number) => number = () => 0;
    const flightObstacles: FlightObstacle[] = [];
    const enterWalk = () => {
      if (!walkWorld) return false;
      const spawn = findWalkSpawn(walkWorld, controls.target);
      if (!spawn) return false;
      // Flush orbit damping before saving/setting a ground-level pose.
      controls.update();
      savedWalkCamera = { position: camera.position.clone(), target: controls.target.clone(), fov: camera.fov, near: camera.near };
      controls.enabled = false;
      walkFeet = spawn;
      walkMotion = createWalkMotion(spawn);
      walkJumpRequested = false;
      walkDistance = 0; walkBobDistance = 0; walkStrength = 0; walkVelocity.x = 0; walkVelocity.z = 0;
      keys.clear();
      syncFlyLook();
      flyPitch = 0;
      const eye = WALK_EYE_METRES * walkWorld.scaleY * exaggerationRef.current;
      camera.position.set(spawn.x, spawn.y * exaggerationRef.current + eye, spawn.z);
      camera.rotation.set(0, flyYaw, 0, 'YXZ');
      camera.fov = 65;
      camera.near = Math.max(0.00005, eye * 0.04);
      camera.updateProjectionMatrix();
      return true;
    };
    enterWalkRef.current = enterWalk;
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let pinnedSelection = false;
    let hoverStartedAt = 0;
    let hoverPresented = false;
    let focusedTarget: string | null = null;
    let lastAimSample = 0;
    let lastFlightInspection: InspectionData | null = null;
    const publishFlightInspection = (data: InspectionData | null, pinned = false) => {
      pinnedSelection = pinned && data !== null;
      setFlightPinned(pinnedSelection);
      if (JSON.stringify(data) !== JSON.stringify(lastFlightInspection)) setFlightInspection(data);
      lastFlightInspection = data;
    };
    const clearHover = () => {
      hoverStartedAt = 0;
      hoverPresented = false;
      focusedTarget = null;
      publishFlightInspection(null);
    };

    const clearSightlineVisuals = () => {
      for (const object of [...sightlineVisuals.children]) {
        sightlineVisuals.remove(object);
        if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => material.dispose());
        }
      }
    };

    const addSightlineMarker = (point: THREE.Vector3, color: string) => {
      const material = new THREE.MeshBasicMaterial({ color, depthTest: false });
      const marker = new THREE.Mesh(new THREE.SphereGeometry(0.38, 16, 10), material);
      marker.position.copy(point);
      marker.renderOrder = 8;
      sightlineVisuals.add(marker);
      const ringGeometry = new THREE.RingGeometry(0.62, 0.84, 24);
      ringGeometry.rotateX(-Math.PI / 2);
      const ring = new THREE.Mesh(ringGeometry, material.clone());
      ring.position.copy(point).add(new THREE.Vector3(0, -0.18, 0));
      ring.renderOrder = 8;
      sightlineVisuals.add(ring);
    };

    let pointerDown: { x: number; y: number } | null = null;
    const rememberPointer = (event: PointerEvent) => { pointerDown = { x: event.clientX, y: event.clientY }; };
    const inspect = (event: PointerEvent) => {
      if (fullMapOpenRef.current) return;
      if (importPendingRef.current || revealActiveRef.current) return;
      if (modeRef.current !== 'explore' && (!pointerDown || Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y) > 5)) return;
      pointerDown = null;
      if (event.button !== 0 || !terrain || !inspectionQuery || modeRef.current === 'orbit') return;
      const bounds = renderer.domElement.getBoundingClientRect();
      if (modeRef.current === 'explore') pointer.set(0, 0);
      else {
        pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
        pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      }
      camera.updateMatrixWorld();
      scene.updateMatrixWorld(true);
      raycaster.setFromCamera(pointer, camera);
      if (modeRef.current === 'profile') {
        if (profileComplete) return;
        const hit = raycaster.intersectObject(terrain, false)[0];
        if (!hit?.uv) return;
        const point = { u: hit.uv.x, v: 1 - hit.uv.y };
        if (interpolateSurface(inspectionQuery.raw, point.u, point.v) === null) return;
        const marker = new THREE.Mesh(new THREE.SphereGeometry(.55, 16, 10), new THREE.MeshBasicMaterial({ color: profileStart ? '#ffc857' : '#45f3c4', depthTest: false }));
        marker.position.copy(terrain.worldToLocal(hit.point.clone())); marker.renderOrder = 9;
        profileVisuals.add(marker);
        if (!profileStart) { profileStart = point; setProfilePicking(true); return; }
        const measured = sampleTerrainProfile(inspectionQuery.raw, inspectionQuery.ground ?? null, profileStart, point);
        const points: THREE.Vector3[] = [];
        for (let i = 1; i < measured.samples.length; i++) {
          const a = measured.samples[i - 1], b = measured.samples[i];
          if (a.surface === null || b.surface === null) continue;
          for (const sample of [a, b]) points.push(new THREE.Vector3((sample.u - .5) * sceneWidth,
            (sample.surface! - sceneMinimum) * sceneVerticalScale + .08, (sample.v - .5) * sceneDepth));
        }
        const line = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: '#45f3c4', depthTest: false }));
        line.renderOrder = 8; profileVisuals.add(line);
        setProfile(measured); setProfilePicking(false); profileComplete = true;
        return;
      }
      if (modeRef.current === 'explore') {
        const data = readInspection();
        publishFlightInspection(data, true);
        focusedTarget = data?.targetKey ?? null;
        hoverStartedAt = performance.now();
        hoverPresented = data !== null;
        return;
      }
      if (modeRef.current === 'sightline') {
        if (sightlineComplete) return; // A completed measurement is disarmed until Reset.
        const hit = raycaster.intersectObject(terrain, false)[0];
        if (!hit?.uv || !sightlineSurface) return;
        const normalized = {
          u: Math.min(1, Math.max(0, hit.uv.x)),
          v: Math.min(1, Math.max(0, 1 - hit.uv.y)),
        } satisfies SightlinePoint;
        const localPoint = terrain.worldToLocal(hit.point.clone());
        const valueRange = Math.max(1e-6, sightlineSurface.stats.maximum - sightlineSurface.stats.minimum);
        const eyeHeight = sightlineSurface.stats.units === 'm' ? 2 : Math.max(0.02, valueRange * 0.04);
        const visualEyeOffset = eyeHeight * sceneVerticalScale;

        if (!sightlineStart) {
          clearSightlineVisuals();
          sightlineStart = { normalized, localPoint };
          addSightlineMarker(localPoint.clone().add(new THREE.Vector3(0, visualEyeOffset, 0)), '#42b8ff');
          setSightlineStartSelected(true);
          setSightlineResult(null);
          return;
        }

        const result = analyzeSightline(
          sightlineSurface.values,
          sightlineSurface.stats,
          sightlineStart.normalized,
          normalized,
        );
        if (!result) {
          setError('The selected sightline crosses invalid DSM pixels. Choose two points inside the valid surface.');
          return;
        }
        setError(null);
        const observerPoint = sightlineStart.localPoint.clone().add(new THREE.Vector3(0, visualEyeOffset, 0));
        const targetPoint = localPoint.clone().add(new THREE.Vector3(0, visualEyeOffset, 0));
        addSightlineMarker(targetPoint, result.visible ? '#bbf451' : '#ff7168');
        const lineGeometry = new THREE.BufferGeometry().setFromPoints([observerPoint, targetPoint]);
        const lineMaterial = new THREE.LineBasicMaterial({
          color: result.visible ? '#bbf451' : '#ff7168',
          transparent: true,
          opacity: 0.96,
          depthTest: false,
        });
        const line = new THREE.Line(lineGeometry, lineMaterial);
        line.renderOrder = 8;
        sightlineVisuals.add(line);
        if (result.blockedAtFraction !== null) {
          const blockedPoint = observerPoint.clone().lerp(targetPoint, result.blockedAtFraction);
          addSightlineMarker(blockedPoint, '#ffc857');
        }
        setSightlineStartSelected(true);
        setSightlineResult(result);
        sightlineStart = null;
        sightlineComplete = true;
        return;
      }
      if (focusAnimating) return;
      const data = readInspection();
      const selected = focusClassRef.current;
      setInspection(selected !== null && data && data.kind !== ['surface', 'building', 'vegetation'][selected] ? null : data);
    };
    // Both inspection modes use the same full-resolution data query.
    const readInspection = (): InspectionData | null => {
      if (!terrain || !inspectionQuery) return null;
      const structureHit = buildings?.visible
        ? raycaster.intersectObject(buildings, true).find((hit) => hit.object.visible && Number.isFinite(hit.object.userData.heightM))
        : undefined;
      const hit = raycaster.intersectObject(terrain, false)[0];
      if (structureHit && (!hit || structureHit.distance <= hit.distance + 0.02)) {
        const heightM = Number(structureHit.object.userData.heightM);
        return withSixClass({
          kind: 'building',
          targetKey: `building:${structureHit.object.userData.structureId}`,
          objectHeight: true,
          rawValue: heightM,
          displayValue: heightM,
          buildingProbability: Number(structureHit.object.userData.confidence),
          vegetationProbability: null,
          confidence: Number(structureHit.object.userData.confidence),
          rgbVegetation: false,
          slopeDegrees: null,
        }, sixClassRef.current?.layer ?? null, structureHit.point.x / sceneWidth + .5,
          structureHit.point.z / sceneDepth + .5, classificationStatusRef.current);
      }
      if (!hit?.uv) return null;
      const u = Math.min(1, Math.max(0, hit.uv.x)), v = Math.min(1, Math.max(0, 1 - hit.uv.y));
      if (modeRef.current === 'inspect' && focusClassRef.current !== null && classMap) {
        const map = classMap.image;
        const code = map.data?.[Math.round(v * (map.height - 1)) * map.width + Math.round(u * (map.width - 1))];
        if (code !== focusClassRef.current) return null;
      }
      const result = querySurface(inspectionQuery, u, v);
      if (result) {
        const region = selectionRegionAt(u, v);
        // Disconnected patches of the same class are different scan targets.
        result.targetKey += `|region:${selectionRegionVersion}:${region ?? `point:${Math.floor(u * 32)}:${Math.floor(v * 32)}`}`;
      }
      return result ? withSixClass(result, sixClassRef.current?.layer ?? null, u, v, classificationStatusRef.current) : null;
    };
    const updateFlyHover = (now: number) => {
      if (fullMapOpenRef.current) return;
      if (modeRef.current !== 'explore') { if (hoverStartedAt || pinnedSelection) clearHover(); return; }
      // Track identity/class at a bounded rate. Mouse jitter must not reset a
      // dwell on the same building or continuous vegetation/terrain class.
      if (now - lastAimSample < 180) return;
      lastAimSample = now;
      pointer.set(0, 0);
      camera.updateMatrixWorld();
      scene.updateMatrixWorld(true);
      raycaster.setFromCamera(pointer, camera);
      const data = readInspection();
      const key = data?.targetKey ?? null;
      if (key !== focusedTarget) {
        clearHover();
        focusedTarget = key;
        hoverStartedAt = now;
      }
      if (!key) return;
      if (hoverPresented || now - hoverStartedAt >= 2000) {
        hoverPresented = true;
        publishFlightInspection(data, pinnedSelection);
      }
    };
    window.addEventListener('blur', clearHover);
    renderer.domElement.addEventListener('pointerdown', rememberPointer);
    renderer.domElement.addEventListener('pointerup', inspect);
    renderer.domElement.addEventListener('pointerdown', activateFlyPointerLock);

    const resize = () => {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', resize);
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();

    Promise.all([
      decodeHeight(sceneInput.heightBuffer, sceneInput.productKind === 'relative' || sceneInput.label.toLowerCase().includes('relative'), !sceneInput.inferHeightUnits),
      decodeHeight(sceneInput.analysisBuffer ?? sceneInput.heightBuffer, sceneInput.productKind === 'relative' || sceneInput.label.toLowerCase().includes('relative'), !sceneInput.inferHeightUnits),
      loadTexture(sceneInput.textureUrl),
      sceneInput.buildingProbabilityBuffer ? decodeHeight(sceneInput.buildingProbabilityBuffer, true) : null,
      sceneInput.vegetationProbabilityBuffer ? decodeHeight(sceneInput.vegetationProbabilityBuffer, true) : null,
      sceneInput.semanticClassBuffer ? decodeHeight(sceneInput.semanticClassBuffer, true) : null,
      sceneInput.confidenceBuffer ? decodeHeight(sceneInput.confidenceBuffer, true) : null,
      sceneInput.missionBuffer ? decodeHeight(sceneInput.missionBuffer) : null,
      sceneInput.terrainBuffer ? decodeHeight(sceneInput.terrainBuffer, false, true) : null,
    ]).then(([decoded, analysisDecoded, texture, buildingDecoded, vegetationDecoded, semanticDecoded, confidenceDecoded, missionDecoded, groundDecoded]) => {
      if (disposed) { texture.dispose(); return; }
      const rawValues = decoded.values;
      const dimensions = terrainDimensions(decoded.stats, 112, analysisDecoded.stats);
      const terrainPresentation = dimensions.metric && (sceneInput.productKind === 'dsm' || Boolean(sceneInput.inferHeightUnits));
      const presentation = terrainPresentation ? terrainPresentationSurface(decoded, groundDecoded, terrainSmoothnessRef.current) : null;
      const presentationCache = new Map<number, NonNullable<typeof presentation>>();
      if (presentation) presentationCache.set(terrainSmoothnessRef.current, presentation);
      const values = decoded.stats.units === 'relative'
        ? smoothSurface(rawValues, decoded.stats.width, decoded.stats.height, 3)
        : smoothTerrainRef.current ? presentation?.values ?? rawValues : rawValues;
      const range = Math.max(decoded.stats.maximum - decoded.stats.minimum, 1e-6);
      // Terrain relief and object heights need different presentation defaults.
      // Preserve the established unreferenced urban/vegetation presentation.
      const recommended = dimensions.metric ? (sceneInput.productKind === 'dsm' ? automaticTerrainExaggeration(decoded) : 1) : 1.7;
      setRecommendedExaggeration(recommended);
      if (automaticReliefRef.current) {
        exaggerationRef.current = recommended;
        setExaggeration(recommended);
      }
      const terrainWidth = dimensions.width;
      const terrainDepth = dimensions.depth;
      sceneWidth = terrainWidth; sceneDepth = terrainDepth; sceneMinimum = decoded.stats.minimum;
      const { segmentsX, segmentsY } = terrainMeshResolution(decoded.stats.width, decoded.stats.height, terrainPresentation);
      const nativeMesh = segmentsX === decoded.stats.width - 1 && segmentsY === decoded.stats.height - 1;
      const geometry = new THREE.PlaneGeometry(terrainWidth, terrainDepth, segmentsX, segmentsY);
      const positions = geometry.attributes.position as THREE.BufferAttribute;
      const colors = new Float32Array(positions.count * 3);
      const slopes = new Float32Array(positions.count);
      const validVertices = new Uint8Array(positions.count);
      vertexMissionElevations = missionDecoded?.stats.units === 'm'
        ? new Float32Array(positions.count)
        : null;
      vertexMissionElevations?.fill(Number.NaN);
      const textureImage = texture.image as HTMLImageElement;
      const textureCanvas = document.createElement('canvas');
      textureCanvas.width = decoded.stats.width;
      textureCanvas.height = decoded.stats.height;
      const textureContext = textureCanvas.getContext('2d', { willReadFrequently: true });
      textureContext?.drawImage(textureImage, 0, 0, textureCanvas.width, textureCanvas.height);
      const texturePixels = textureContext?.getImageData(0, 0, textureCanvas.width, textureCanvas.height).data ?? null;
      inspectionQuery = { raw: analysisDecoded, display: presentation ? { ...decoded, values } : decoded, building: buildingDecoded, vegetation: vegetationDecoded,
        ground: sceneInput.productKind === 'dsm' ? groundDecoded : null,
        semantic: semanticDecoded, confidence: confidenceDecoded, rgba: texturePixels,
        textureWidth: textureCanvas.width, textureHeight: textureCanvas.height };
      // A dimensionless relative prior is a geometry hint, not a metric DSM.
      // Keep it subtle; aggressive exaggeration turns domain-gap noise into cliffs.
      const verticalScale = dimensions.verticalScale;
      container.dataset.terrainScale = JSON.stringify({ ...dimensions, minimum: decoded.stats.minimum, maximum: decoded.stats.maximum });
      // Absolute DSM already contains the surface. nDSM solids have ground-relative bases.
      const displayStructures = sceneInput.productKind === 'dsm' ? [] : sceneInput.structures;
      const buildingTops = displayStructures.map(structure => structure.base_m + structure.height_m).filter(Number.isFinite);
      const buildingBases = displayStructures.map(structure => structure.base_m).filter(Number.isFinite);
      const gridSpec = metricGridSpec({ ...decoded.stats,
        minimum: Math.min(decoded.stats.minimum, ...buildingBases),
        maximum: Math.max(decoded.stats.maximum, ...buildingTops),
      }, terrainWidth, terrainDepth);
      metricFloor = createMetricGridFloor(gridSpec);
      metricFloor.visible = false;
      scene.add(metricFloor);
      setMetricGrid(gridSpec);
      sceneVerticalScale = verticalScale;
      sightlineSurface = { values: analysisDecoded.values, stats: analysisDecoded.stats };
      const color = new THREE.Color();
      for (let row = 0; row <= segmentsY; row += 1) {
        for (let col = 0; col <= segmentsX; col += 1) {
          const vertex = row * (segmentsX + 1) + col;
          const u = col / segmentsX, v = row / segmentsY;
          const sampled = nativeMesh ? Number.isFinite(values[vertex]) ? values[vertex] : null
            : interpolateSurface({ values, stats: decoded.stats }, u, v);
          validVertices[vertex] = sampled === null ? 0 : 1;
          const value = sampled ?? decoded.stats.minimum;
          slopes[vertex] = surfaceSlope(analysisDecoded, u, v) ?? -1;
          const original = nativeMesh ? Number.isFinite(rawValues[vertex]) ? rawValues[vertex] : null
            : interpolateSurface(decoded, u, v);
          const normalized = Math.min(1, Math.max(0, ((original ?? decoded.stats.minimum) - decoded.stats.minimum) / range));
          positions.setZ(vertex, (value - decoded.stats.minimum) * verticalScale);
          if (vertexMissionElevations && missionDecoded) {
            const missionRow = Math.round(row / segmentsY * (missionDecoded.stats.height - 1));
            const missionCol = Math.round(col / segmentsX * (missionDecoded.stats.width - 1));
            vertexMissionElevations[vertex] = missionDecoded.values[missionRow * missionDecoded.stats.width + missionCol];
          }
          colorForHeight(color, normalized);
          colors[vertex * 3] = color.r;
          colors[vertex * 3 + 1] = color.g;
          colors[vertex * 3 + 2] = color.b;
        }
      }
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geometry.setAttribute('surfaceSlope', new THREE.BufferAttribute(slopes, 1));
      const indices = geometry.index!;
      const validIndices: number[] = [];
      for (let i = 0; i < indices.count; i += 3) {
        const a = indices.getX(i), b = indices.getX(i + 1), c = indices.getX(i + 2);
        if (validVertices[a] && validVertices[b] && validVertices[c]) validIndices.push(a, b, c);
      }
      geometry.setIndex(validIndices);
      geometry.computeVertexNormals();
      geometry.rotateX(-Math.PI / 2);
      const collisionHeights = new Float32Array(positions.count);
      for (let i = 0; i < positions.count; i++) collisionHeights[i] = positions.getY(i);
      sourceMeshHeights = collisionHeights;
      selectionSurface = (u, v) => {
        if (rasterValue(analysisDecoded, u, v) === null) return null;
        const x = u * segmentsX, y = v * segmentsY;
        const col = Math.min(segmentsX - 1, Math.floor(x)), row = Math.min(segmentsY - 1, Math.floor(y));
        const a = row * (segmentsX + 1) + col, b = a + segmentsX + 1, d = a + 1, e = b + 1;
        const fx = x - col, fy = y - row;
        if (fx + fy <= 1) return validVertices[a] && validVertices[b] && validVertices[d]
          ? positions.getY(a) + fx * (positions.getY(d) - positions.getY(a)) + fy * (positions.getY(b) - positions.getY(a)) : null;
        return validVertices[b] && validVertices[d] && validVertices[e]
          ? positions.getY(e) + (1 - fx) * (positions.getY(b) - positions.getY(e)) + (1 - fy) * (positions.getY(d) - positions.getY(e)) : null;
      };
      flightGround = (x, z) => gridSurfaceHeight(collisionHeights, segmentsX + 1, segmentsY + 1, terrainWidth, terrainDepth, x, z) * exaggerationRef.current;
      opticalMaterial = new THREE.MeshStandardMaterial({ map: texture, roughness: 0.88, metalness: 0.03, side: THREE.DoubleSide });
      elevationMaterial = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.78, metalness: 0.04, side: THREE.DoubleSide });
      metricMaterials.push(createMetricGridMaterial(gridSpec, verticalScale, decoded.stats.minimum, opticalMaterial));
      metricMaterials.push(createMetricGridMaterial(gridSpec, verticalScale, decoded.stats.minimum, elevationMaterial));
      terrain = new THREE.Mesh(geometry, opticalMaterial);
      terrain.receiveShadow = true;
      scene.add(terrain);
      if (sceneInput.productKind === 'dsm') {
        terrainBoundary = createTerrainBoundary(positions, validVertices, segmentsX, segmentsY);
        scene.add(terrainBoundary);
      }
      geometry.computeBoundingBox();
      container.dataset.terrainMesh = JSON.stringify({
        minimumY: geometry.boundingBox!.min.y, maximumY: geometry.boundingBox!.max.y,
        vertices: positions.count, triangles: validIndices.length / 3,
        boundaryTriangles: terrainBoundary ? (terrainBoundary.children[0] as THREE.Mesh).geometry.attributes.position.count / 3 : 0,
        terrainPresentation, maximumDisplayAdjustmentM: smoothTerrainRef.current ? presentation?.maximumAdjustment ?? 0 : 0,
        rmsDisplayAdjustmentM: smoothTerrainRef.current ? presentation?.rmsAdjustment ?? 0 : 0,
        smoothingLevel: terrainPresentation && smoothTerrainRef.current ? terrainSmoothnessRef.current : 0,
      });
      container.dataset.terrainPresentation = terrainPresentation && smoothTerrainRef.current ? 'smooth' : 'source';
      // Switch the presentation in place: preserve the camera, source queries,
      // active profile and exports, and make walking follow the displayed mesh.
      terrainPresentationRef.current = presentation ? (smooth, strength) => {
        if (smooth && !presentationCache.has(strength)) presentationCache.set(strength, terrainPresentationSurface(decoded, groundDecoded, strength));
        const selectedPresentation = presentationCache.get(strength) ?? presentation;
        const display = smooth ? { ...decoded, values: selectedPresentation.values } : decoded;
        resetClassGeometry();
        for (let row = 0; row <= segmentsY; row++) for (let col = 0; col <= segmentsX; col++) {
          const index = row * (segmentsX + 1) + col;
          const sample = nativeMesh ? display.values[index] : interpolateSurface(display, col / segmentsX, row / segmentsY);
          const value = sample !== null && Number.isFinite(sample) ? sample : decoded.stats.minimum;
          const y = (value - decoded.stats.minimum) * verticalScale;
          positions.setY(index, y); collisionHeights[index] = y;
        }
        positions.needsUpdate = true;
        geometry.computeVertexNormals(); geometry.computeBoundingBox(); geometry.computeBoundingSphere();
        if (inspectionQuery) inspectionQuery.display = display;
        setInspection(null);
        // Decorative tree proxies must follow the selected terrain too.
        if (canopy) {
          scene.remove(canopy);
          canopy.traverse(object => {
            if (object instanceof THREE.Mesh) {
              if (object instanceof THREE.InstancedMesh) object.dispose();
              object.geometry.dispose();
              (Array.isArray(object.material) ? object.material : [object.material]).forEach(material => material.dispose());
            }
          });
          canopy = buildCanopyProxy(texture, display.values, decoded.stats.width, decoded.stats.height, terrainWidth, terrainDepth, decoded.stats.minimum, verticalScale);
          scene.add(canopy);
        }
        if (terrainBoundary) {
          scene.remove(terrainBoundary);
          terrainBoundary.traverse(object => {
            if (object instanceof THREE.Mesh || object instanceof THREE.LineSegments) {
              object.geometry.dispose();
              (Array.isArray(object.material) ? object.material : [object.material]).forEach(material => material.dispose());
            }
          });
          terrainBoundary = createTerrainBoundary(positions, validVertices, segmentsX, segmentsY);
          scene.add(terrainBoundary);
        }
        container.dataset.terrainPresentation = smooth ? 'smooth' : 'source';
        container.dataset.terrainMesh = JSON.stringify({ ...JSON.parse(container.dataset.terrainMesh!),
          minimumY: geometry.boundingBox!.min.y, maximumY: geometry.boundingBox!.max.y,
          maximumDisplayAdjustmentM: smooth ? selectedPresentation.maximumAdjustment : 0,
          rmsDisplayAdjustmentM: smooth ? selectedPresentation.rmsAdjustment : 0,
          smoothingLevel: smooth ? strength : 0,
        });
      } : null;

      const wireMaterial = new THREE.MeshBasicMaterial({ color: '#71ffd6', wireframe: true, transparent: true, opacity: 0.13 });
      wireframe = new THREE.Mesh(geometry, wireMaterial);
      wireframe.position.y = 0.035;
      scene.add(wireframe);
      grid = new THREE.GridHelper(terrainWidth * 1.25, 22, '#2d806b', '#173b34');
      grid.position.y = -0.12;
      scene.add(grid);
      canopy = buildCanopyProxy(texture, values, decoded.stats.width, decoded.stats.height, terrainWidth, terrainDepth, decoded.stats.minimum, verticalScale);
      scene.add(canopy);
      buildings = buildBuildingSolids(displayStructures, texture, terrainWidth, terrainDepth, verticalScale);
      const decoratedMaterials = new Set<THREE.MeshStandardMaterial>();
      buildings.children.forEach(object => {
        if (!(object instanceof THREE.Mesh)) return;
        selectionBuildings.set(Number(object.userData.structureId), object);
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        for (const material of materials) if (material instanceof THREE.MeshStandardMaterial && !decoratedMaterials.has(material)) {
          decoratedMaterials.add(material);
          metricMaterials.push(createMetricGridMaterial(gridSpec, verticalScale, 0, material, 1));
        }
      });
      scene.add(buildings);
      buildings.updateMatrixWorld(true);
      const buildingObstacles = new Map<number, FlightObstacle>();
      buildings.children.forEach(object => {
        if (!(object instanceof THREE.Mesh)) return;
        const structure = sceneInput.structures.find(item => item.id === object.userData.structureId);
        if (!structure) return;
        const bounds = new THREE.Box3().setFromObject(object);
        const footprint: [number, number][] = structure.outline && structure.outline.length >= 3
          ? structure.outline.map(([x, y]) => [(x - 0.5) * terrainWidth, (y - 0.5) * terrainDepth])
          : [[-1,-1],[1,-1],[1,1],[-1,1]].map(([x, z]) => {
            const point = new THREE.Vector3(x * Math.max(.42, structure.width * terrainWidth) / 2, 0, z * Math.max(.42, structure.depth * terrainDepth) / 2).applyMatrix4(object.matrixWorld);
            return [point.x, point.z];
          });
        buildingObstacles.set(structure.id, { footprint, top: bounds.max.y });
      });
      flightObstacles.push(...buildingObstacles.values());

      // Match the inspector and the actual building solids, not a new classifier.
      const maskWidth = decoded.stats.width, maskHeight = decoded.stats.height;
      const footprintCanvas = document.createElement('canvas');
      footprintCanvas.width = maskWidth; footprintCanvas.height = maskHeight;
      const footprintContext = footprintCanvas.getContext('2d', { willReadFrequently: true });
      if (footprintContext) {
        footprintContext.fillStyle = '#fff';
        for (const obstacle of flightObstacles) {
          footprintContext.beginPath();
          obstacle.footprint.forEach(([x, z], index) => {
            const px = (x / terrainWidth + .5) * (maskWidth - 1), py = (z / terrainDepth + .5) * (maskHeight - 1);
            if (index === 0) footprintContext.moveTo(px, py); else footprintContext.lineTo(px, py);
          });
          footprintContext.closePath(); footprintContext.fill();
        }
      }
      const footprintPixels = footprintContext?.getImageData(0, 0, maskWidth, maskHeight).data;
      const classPixels = new Uint8Array(maskWidth * maskHeight), validPixels = new Uint8Array(classPixels.length);
      for (let row = 0; row < maskHeight; row++) for (let col = 0; col < maskWidth; col++) {
        const i = row * maskWidth + col, u = col / (maskWidth - 1), v = row / (maskHeight - 1);
        validPixels[i] = rasterValue(analysisDecoded, u, v) !== null ? 1 : 0;
        classPixels[i] = validPixels[i] && (footprintPixels?.[i * 4 + 3] ?? 0) >= 128 ? 1 : surfaceClassCode(inspectionQuery!, u, v);
      }
      classMap = new THREE.DataTexture(classPixels, maskWidth, maskHeight, THREE.RedFormat, THREE.UnsignedByteType);
      classMap.minFilter = THREE.NearestFilter; classMap.magFilter = THREE.NearestFilter;
      classMap.unpackAlignment = 1; classMap.needsUpdate = true;
      focusClasses = new Uint8Array(positions.count);
      for (let row = 0; row <= segmentsY; row++) for (let col = 0; col <= segmentsX; col++) {
        focusClasses[row * (segmentsX + 1) + col] = classPixels[Math.round(row / segmentsY * (maskHeight - 1)) * maskWidth + Math.round(col / segmentsX * (maskWidth - 1))];
      }

      const { gsdX, gsdY } = analysisDecoded.stats;
      const metricScaleKnown = Boolean(gsdX && gsdY && gsdX > 0 && gsdY > 0);
      const humanScale = analysisDecoded.stats.units === 'm' ? verticalScale : 0.36;
      walkWorld = {
        width: terrainWidth, depth: terrainDepth,
        scaleX: metricScaleKnown ? terrainWidth / ((analysisDecoded.stats.width - 1) * gsdX!) : humanScale,
        scaleZ: metricScaleKnown ? terrainDepth / ((analysisDecoded.stats.height - 1) * gsdY!) : humanScale,
        scaleY: humanScale,
        blocked: (x, z, radius) => flightObstacles.some(obstacle => touchesFootprint(x, z, obstacle.footprint, radius)),
        ground: (x, z) => {
          const u = x / terrainWidth + 0.5, v = z / terrainDepth + 0.5;
          const raw = rasterValue(analysisDecoded, u, v);
          if (raw === null || rasterValue(decoded, u, v) === null) return null;
          const semantic = rasterValue(semanticDecoded, u, v);
          const building = rasterValue(buildingDecoded, u, v);
          const vegetation = rasterValue(vegetationDecoded, u, v);
          if (semantic === 1 || (building ?? 0) >= 0.5) return null;
          // A DSM is a top surface, not ground underneath trees. Do not invent forest paths.
          if ((semantic === 2 || (vegetation ?? 0) >= 0.5)
            && (sceneInput.productKind !== 'ndsm' || raw > 1)) return null;
          const col = Math.min(segmentsX - 1, Math.floor(u * segmentsX));
          const row = Math.min(segmentsY - 1, Math.floor(v * segmentsY));
          for (const [dc, dr] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
            if (rasterValue(decoded, (col + dc) / segmentsX, (row + dr) / segmentsY) === null) return null;
          }
          return gridSurfaceHeight(collisionHeights, segmentsX + 1, segmentsY + 1, terrainWidth, terrainDepth, x, z);
        },
      };

      if (missionDecoded?.stats.units === 'm' && sceneInput.missionSource && vertexMissionElevations) {
        const sortedElevationsM = missionDecoded.values
          .filter((value) => Number.isFinite(value))
          .sort();
        // Robust percentiles prevent a single NoData edge/outlier from making
        // the decision-support slider useless while keeping the raw DSM intact.
        const minimumM = sortedElevationsM[Math.floor((sortedElevationsM.length - 1) * 0.02)];
        const maximumM = sortedElevationsM[Math.floor((sortedElevationsM.length - 1) * 0.98)];
        const reliefM = Math.max(0, maximumM - minimumM);
        const scenarioMaximumM = Math.max(1, Math.min(50, reliefM || 1));
        const tenthPercentileM = sortedElevationsM[Math.floor((sortedElevationsM.length - 1) * 0.1)];
        const defaultScenarioRiseM = Math.min(
          scenarioMaximumM,
          Math.max(1, tenthPercentileM - minimumM),
        );
        const pixelAreaM2 = missionDecoded.stats.gsdX && missionDecoded.stats.gsdY
          ? missionDecoded.stats.gsdX * missionDecoded.stats.gsdY
          : null;
        const mappedBuildings = sceneInput.structures.flatMap((structure) => {
          const col = Math.min(
            missionDecoded.stats.width - 1,
            Math.max(0, Math.round(structure.center_x * (missionDecoded.stats.width - 1))),
          );
          const row = Math.min(
            missionDecoded.stats.height - 1,
            Math.max(0, Math.round(structure.center_y * (missionDecoded.stats.height - 1))),
          );
          const roofElevationM = missionDecoded.values[row * missionDecoded.stats.width + col];
          if (!Number.isFinite(roofElevationM)) return [];
          return [{
            id: structure.id,
            groundElevationM: roofElevationM - Math.max(0, structure.height_m),
            roofElevationM,
          } satisfies MissionBuilding];
        });

        let steepCount = 0;
        let slopeCount = 0;
        if (missionDecoded.stats.gsdX && missionDecoded.stats.gsdY) {
          const sampleStep = Math.max(1, Math.floor(Math.max(missionDecoded.stats.width, missionDecoded.stats.height) / 512));
          for (let row = sampleStep; row < missionDecoded.stats.height - sampleStep; row += sampleStep) {
            for (let col = sampleStep; col < missionDecoded.stats.width - sampleStep; col += sampleStep) {
              const left = missionDecoded.values[row * missionDecoded.stats.width + col - sampleStep];
              const right = missionDecoded.values[row * missionDecoded.stats.width + col + sampleStep];
              const top = missionDecoded.values[(row - sampleStep) * missionDecoded.stats.width + col];
              const bottom = missionDecoded.values[(row + sampleStep) * missionDecoded.stats.width + col];
              if (![left, right, top, bottom].every(Number.isFinite)) continue;
              const dzdx = (right - left) / (2 * sampleStep * missionDecoded.stats.gsdX);
              const dzdy = (bottom - top) / (2 * sampleStep * missionDecoded.stats.gsdY);
              const slopeDegrees = Math.atan(Math.hypot(dzdx, dzdy)) * 180 / Math.PI;
              slopeCount += 1;
              if (slopeDegrees >= 30) steepCount += 1;
            }
          }
        }

        const missionData = {
          source: sceneInput.missionSource,
          sortedElevationsM,
          minimumM,
          maximumM,
          scenarioMaximumM,
          defaultScenarioRiseM,
          pixelAreaM2,
          mappedBuildings,
          steepAreaFraction: slopeCount ? steepCount / slopeCount : null,
        } satisfies MissionAnalysisData;
        setMissionAnalysis(missionData);
        setScenarioRiseM(defaultScenarioRiseM);

        const overlayGeometry = geometry.clone();
        overlayGeometry.setAttribute('floodRisk', new THREE.BufferAttribute(new Float32Array(positions.count), 1));
        const overlayMaterial = new THREE.ShaderMaterial({
          transparent: true,
          depthWrite: false,
          side: THREE.DoubleSide,
          vertexShader: `
            attribute float floodRisk;
            varying float vFloodRisk;
            void main() {
              vFloodRisk = floodRisk;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
          `,
          fragmentShader: `
            varying float vFloodRisk;
            void main() {
              if (vFloodRisk < 0.48) discard;
              float edge = smoothstep(0.48, 0.82, vFloodRisk);
              gl_FragColor = vec4(0.08, 0.72, 0.98, 0.38 + edge * 0.24);
            }
          `,
        });
        floodOverlay = new THREE.Mesh(overlayGeometry, overlayMaterial);
        floodOverlay.position.y = 0.09;
        floodOverlay.renderOrder = 4;
        floodOverlay.visible = false;
        scene.add(floodOverlay);

        highGroundMarkers = new THREE.Group();
        const highCutoff = sortedElevationsM[Math.floor((sortedElevationsM.length - 1) * 0.9)];
        const candidates: Array<{ index: number; elevation: number; x: number; z: number }> = [];
        const candidateStride = Math.max(1, Math.floor(positions.count / 5000));
        for (let index = 0; index < positions.count; index += candidateStride) {
          const elevation = vertexMissionElevations[index];
          if (Number.isFinite(elevation) && elevation >= highCutoff) {
            candidates.push({ index, elevation, x: positions.getX(index), z: positions.getZ(index) });
          }
        }
        candidates.sort((left, right) => right.elevation - left.elevation);
        const selectedCandidates: typeof candidates = [];
        for (const candidate of candidates) {
          if (selectedCandidates.every((selected) => Math.hypot(candidate.x - selected.x, candidate.z - selected.z) >= 18)) {
            selectedCandidates.push(candidate);
          }
          if (selectedCandidates.length === 3) break;
        }
        const ringMaterial = new THREE.MeshBasicMaterial({ color: '#bbf451', transparent: true, opacity: 0.94, side: THREE.DoubleSide, depthTest: false });
        const beaconMaterial = new THREE.MeshBasicMaterial({ color: '#dfff9c', transparent: true, opacity: 0.94, depthTest: false });
        selectedCandidates.forEach((candidate) => {
          const ringGeometry = new THREE.RingGeometry(1.05, 1.42, 28);
          ringGeometry.rotateX(-Math.PI / 2);
          const ring = new THREE.Mesh(ringGeometry, ringMaterial);
          ring.position.set(candidate.x, positions.getY(candidate.index) + 0.18, candidate.z);
          ring.renderOrder = 6;
          highGroundMarkers?.add(ring);
          const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 4.5, 8), beaconMaterial);
          beacon.position.set(candidate.x, positions.getY(candidate.index) + 2.45, candidate.z);
          beacon.renderOrder = 6;
          highGroundMarkers?.add(beacon);
          const beaconTip = new THREE.Mesh(new THREE.SphereGeometry(0.24, 12, 8), beaconMaterial);
          beaconTip.position.set(candidate.x, positions.getY(candidate.index) + 4.72, candidate.z);
          beaconTip.renderOrder = 6;
          highGroundMarkers?.add(beaconTip);
        });
        highGroundMarkers.visible = false;
        scene.add(highGroundMarkers);
      } else {
        setMissionAnalysis(null);
      }
      terrainStats = { ...analysisDecoded.stats, vertices: positions.count };
      cameraPresetRef.current = (top) => {
        if (!top && !dimensions.metric && sceneInput.productKind === 'ndsm') {
          // Match the original urban presentation instead of the more distant,
          // steeper camera needed to frame wide, physically scaled terrain.
          const portraitFit = Math.max(1, .9 / camera.aspect);
          controls.target.set(0, 2.5, 0);
          camera.position.set(72 * portraitFit, 2.5 + 45.5 * portraitFit, 92 * portraitFit);
          controls.maxDistance = Math.max(260, 260 * portraitFit);
          camera.far = Math.max(1200, 1200 * portraitFit);
          camera.updateProjectionMatrix(); controls.update(); return;
        }
        const fullRelief = Math.max(range, gridSpec.maximum - decoded.stats.minimum);
        const centreY = fullRelief * verticalScale * exaggerationRef.current * .42;
        const span = Math.max(terrainWidth, terrainDepth, fullRelief * verticalScale * exaggerationRef.current);
        const distance = span / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * Math.min(1, camera.aspect)) * 1.2;
        controls.target.set(0, centreY, 0);
        camera.position.set(top ? 0 : distance * .48, centreY + distance * (top ? 1 : .65), top ? .01 : distance * .72);
        controls.maxDistance = Math.max(260, distance * 3);
        camera.far = Math.max(1200, distance * 5); camera.updateProjectionMatrix(); controls.update();
      };
      cameraPresetRef.current(false);
      setStats(terrainStats);
      setInspection(null);
      importPendingRef.current = false;
      setImportPending(false);
      if (sceneInput.animateImport && !reducedMotion) { handoffUntil = performance.now() + 650; setImportHandoff(true); }
      else setImportHandoff(false);
      const shouldReveal = sceneInput.animateImport && sceneInput.heightBuffer !== lastRevealedBufferRef.current && !reducedMotion;
      if (shouldReveal) {
        lastRevealedBufferRef.current = sceneInput.heightBuffer;
        revealStartedAt = performance.now();
        revealActiveRef.current = true;
        setRevealPhase('grid');
        setStatus('Prediction ready · visual reveal');
      } else {
        setStatus('Interactive mesh ready');
        if (pendingValidationOpenRef.current) {
          pendingValidationOpenRef.current = false;
          setValidationEvidenceOpen(true);
        }
      }
    }).catch((reason: Error) => {
      setError(reason.message || 'Could not build this raster.');
      setStatus('Scene failed to load');
      revealActiveRef.current = false;
      setRevealPhase(null);
      importPendingRef.current = false;
      setImportPending(false);
      setImportHandoff(false);
      pendingValidationOpenRef.current = false;
    });

    let readoutAt = 0;
    let lastMissionThreshold = Number.NaN;
    let lastFrameTime = performance.now();
    let blockedSince = 0;
    let wasBlocked = false;
    const animate = (now = performance.now()) => {
      frame = requestAnimationFrame(animate);
      const delta = Math.min((now - lastFrameTime) / 1000, 0.05);
      lastFrameTime = now;
      // A neutral CSS grid is visible while inference runs. Pause the old scene
      // renderer so it neither masquerades as the new result nor competes for GPU.
      if (importPendingRef.current) return;
      if (handledProfileReset !== profileResetRef.current || (modeRef.current !== 'profile' && (profileStart || profileComplete))) {
        handledProfileReset = profileResetRef.current; clearProfile();
      }
      if (handoffUntil && now >= handoffUntil) { handoffUntil = 0; setImportHandoff(false); }
      let reveal: ReturnType<typeof sceneRevealAt> | null = null;
      if (revealStartedAt !== null) {
        reveal = sceneRevealAt((now - revealStartedAt) / 1000, skipRevealRef.current || reducedMotion);
        if (revealProgressRef.current) revealProgressRef.current.value = reveal.progress;
        container.dataset.revealPhase = reveal.phase;
        if (lastRevealPhase !== reveal.phase) {
          lastRevealPhase = reveal.phase;
          setRevealPhase(reveal.phase === 'complete' ? null : reveal.phase);
        }
        if (reveal.phase === 'complete') {
          revealStartedAt = null;
          revealActiveRef.current = false;
          metricBlend = styleRef.current === 'metric' ? 1 : 0;
          setStatus('Interactive mesh ready');
          if (pendingValidationOpenRef.current) {
            pendingValidationOpenRef.current = false;
            setValidationEvidenceOpen(true);
          }
        }
      }
      if (handledSightlineReset !== sightlineResetRef.current) {
        clearSightlineVisuals();
        sightlineStart = null;
        sightlineComplete = false;
        clearHover();
        handledSightlineReset = sightlineResetRef.current;
      }
      const wantedFocus = modeRef.current === 'inspect' && styleRef.current === 'metric' ? focusClassRef.current : null;
      if (wantedFocus !== focusTarget) {
        focusFrom = [...focusWeights]; focusTarget = wantedFocus; focusStartedAt = now; focusAnimating = true;
        if (wantedFocus !== null) lastFocusColour = wantedFocus;
      }
      if (modeRef.current !== 'inspect' && (focusAnimating || focusWeights.some(v => v !== 1))) resetClassGeometry();
      if (focusAnimating && terrain && sourceMeshHeights && focusClasses) {
        const focused = classFocusFrame(focusFrom, focusTarget, (now - focusStartedAt) / 1000, reducedMotion);
        focusWeights = focused.weights;
        const position = terrain.geometry.getAttribute('position') as THREE.BufferAttribute;
        focusedVertexHeights(sourceMeshHeights, focusClasses, focusWeights, position.array as Float32Array);
        position.needsUpdate = true;
        terrain.geometry.computeVertexNormals();
        focusAnimating = !focused.done;
      }
      focusGlow = advanceMetricBlend(focusGlow, wantedFocus === null ? 0 : 1, delta, reducedMotion);
      container.dataset.classFocus = JSON.stringify({ selected: wantedFocus, weights: focusWeights, animating: focusAnimating });
      const fly = modeRef.current === 'explore';
      if (fly && !previousFlyMode) syncFlyLook();
      if (!fly && previousFlyMode) {
        if (!restoreWalkCamera()) {
          const direction = new THREE.Vector3();
          camera.getWorldDirection(direction);
          controls.target.copy(camera.position).addScaledVector(direction, 20);
        }
        flyLookReady = false;
        keys.clear();
        setFlightBlocked(false);
        blockedSince = 0;
        wasBlocked = false;
      }
      previousFlyMode = fly;
      controls.enabled = !fly;
      const walking = fly && exploreModeRef.current === 'walk' && walkWorld && walkFeet;
      if (fullMapOpenRef.current) { clearMovementKeys(); walkVelocity.x = 0; walkVelocity.z = 0; }
      if (walking && walkWorld && walkFeet && !fullMapOpenRef.current) {
        const intent = walkIntent(flyYaw,
          Number(keys.has('KeyW') || keys.has('ArrowUp')) - Number(keys.has('KeyS') || keys.has('ArrowDown')),
          Number(keys.has('KeyD') || keys.has('ArrowRight')) - Number(keys.has('KeyA') || keys.has('ArrowLeft')));
        const pace = WALK_SPEED_MPS * (keys.has('KeyG') ? 3 : 1);
        const easing = 1 - Math.exp(-(intent.x || intent.z ? 24 : 32) * delta);
        walkVelocity.x += (intent.x * pace - walkVelocity.x) * easing;
        walkVelocity.z += (intent.z * pace - walkVelocity.z) * easing;
        const result = stepWalk(walkMotion ?? createWalkMotion(walkFeet), walkVelocity, delta, walkWorld, walkJumpRequested);
        walkJumpRequested = false;
        walkMotion = result;
        walkFeet = result.position;
        walkDistance += result.distance;
        walkBobDistance += result.distance / (keys.has('KeyG') ? 2.2 : 1);
        walkStrength += (Math.min(1, result.distance / Math.max(delta * pace, 0.0001)) - walkStrength) * easing;
        const head = walkHeadMotion(walkBobDistance, walkStrength, reducedMotion || !headMotionRef.current || !result.grounded);
        camera.position.set(walkFeet.x + Math.cos(flyYaw) * head.sway * walkWorld.scaleX,
          (walkFeet.y + (WALK_EYE_METRES + head.lift) * walkWorld.scaleY) * exaggerationRef.current,
          walkFeet.z - Math.sin(flyYaw) * head.sway * walkWorld.scaleZ);
        camera.rotation.set(flyPitch + head.pitch, flyYaw, head.roll, 'YXZ');
        const near = Math.max(0.00005, WALK_EYE_METRES * walkWorld.scaleY * exaggerationRef.current * 0.04);
        if (camera.near !== near) { camera.near = near; camera.updateProjectionMatrix(); }
        const moving = Math.hypot(intent.x, intent.z) > 0;
        if (moving && result.blocked) blockedSince ||= now;
        else blockedSince = 0;
        const showBlocked = blockedSince > 0 && now - blockedSince > 180;
        if (showBlocked !== wasBlocked) { setFlightBlocked(showBlocked); wasBlocked = showBlocked; }
      } else if (fly && !fullMapOpenRef.current) {
        const forward = new THREE.Vector3();
        camera.getWorldDirection(forward);
        forward.normalize();
        const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
        const speed = (keys.has('KeyG') ? 54 : 18) * delta;
        const movement = new THREE.Vector3();
        if (keys.has('KeyW') || keys.has('ArrowUp')) movement.add(forward);
        if (keys.has('KeyS') || keys.has('ArrowDown')) movement.sub(forward);
        if (keys.has('KeyA') || keys.has('ArrowLeft')) movement.sub(right);
        if (keys.has('KeyD') || keys.has('ArrowRight')) movement.add(right);
        if (keys.has('Space')) movement.y += 1;
        if (keys.has('ShiftLeft') || keys.has('ShiftRight')) movement.y -= 1;
        if (movement.lengthSq() > 1) movement.normalize();
        movement.multiplyScalar(speed);
        const safe = moveFlight(camera.position, movement, flightGround, flightObstacles, exaggerationRef.current);
        const requestedHorizontal = Math.hypot(movement.x, movement.z);
        const travelledHorizontal = Math.hypot(safe.x - camera.position.x, safe.z - camera.position.z);
        const blocked = (requestedHorizontal > 0.001 && travelledHorizontal < requestedHorizontal * 0.45)
          || (movement.y < -0.001 && safe.y >= camera.position.y - 0.001);
        if (blocked) blockedSince ||= now;
        else blockedSince = 0;
        const showBlocked = blockedSince > 0 && now - blockedSince > 180;
        if (showBlocked !== wasBlocked) { setFlightBlocked(showBlocked); wasBlocked = showBlocked; }
        camera.position.set(safe.x, safe.y, safe.z);
      } else if (!fly) {
        controls.update();
      }
      if (fly) mapFeedRef.current = mapFrame(walking && walkFeet ? walkFeet.x : camera.position.x,
        walking && walkFeet ? walkFeet.z : camera.position.z, sceneWidth, sceneDepth, flyYaw, exploreModeRef.current);
      const metricView = styleRef.current === 'metric';
      metricBlend = advanceMetricBlend(metricBlend, metricView ? 1 : 0, delta, reducedMotion);
      const easedMetricBlend = reveal?.metricMix ?? metricBlend * metricBlend * (3 - 2 * metricBlend);
      if (!metricView) lastPhotoStyle = styleRef.current === 'elevation' ? 'elevation' : 'optical';
      if (terrain && opticalMaterial && elevationMaterial) {
        terrain.material = lastPhotoStyle === 'optical' ? opticalMaterial : elevationMaterial;
      }
      for (const controller of metricMaterials) {
        controller.setTerrainAnalysis(layersRef.current.contours, styleRef.current === 'hillshade' ? 1 : styleRef.current === 'slope' ? 2 : 0);
        controller.setExaggeration(exaggerationRef.current);
        controller.setBlend(easedMetricBlend);
        controller.setClassFocus(classMap, lastFocusColour, focusGlow);
        controller.setSixClassOverlay(sixClassOverlayVisible(sixClassRef.current?.layer ?? null) ? sixClassRef.current!.texture : null, .82);
      }
      container.dataset.sixClassOverlay = sixClassOverlayVisible(sixClassRef.current?.layer ?? null) ? 'six-class; height geometry unchanged' : 'off';
      const displayScaleY = exaggerationRef.current * (reveal?.scaleY ?? 1);
      if (terrain) terrain.scale.y = displayScaleY;
      if (terrainBoundary) {
        terrainBoundary.scale.y = displayScaleY;
        terrainBoundary.visible = wantedFocus === null && !focusAnimating;
      }
      container.dataset.displayExaggeration = String(displayScaleY);
      if (wireframe) wireframe.scale.y = displayScaleY;
      if (canopy) canopy.scale.y = displayScaleY;
      if (buildings) {
        buildings.scale.y = displayScaleY * Math.max(.00001, focusWeights[1]);
        buildings.visible = (reveal?.showBuildings ?? true) && focusWeights[1] > .0001;
        for (const object of buildings.children) if (object instanceof THREE.LineSegments) {
          const material = object.material as THREE.LineBasicMaterial;
          material.opacity = wantedFocus === 1 ? .12 + .6 * focusGlow : reveal ? .12 + .43 * reveal.metricMix : .12;
          material.color.set(wantedFocus === 1 ? '#ffc36e' : reveal && reveal.phase !== 'complete' ? '#9befed' : '#283b35');
        }
      }
      if (floodOverlay) floodOverlay.scale.y = exaggerationRef.current;
      if (highGroundMarkers) highGroundMarkers.scale.y = exaggerationRef.current;
      sightlineVisuals.scale.y = exaggerationRef.current;
      profileVisuals.scale.y = exaggerationRef.current;
      if (wireframe) {
        wireframe.visible = layersRef.current.wireframe && easedMetricBlend < 1;
        (wireframe.material as THREE.MeshBasicMaterial).opacity = .13 * (1 - easedMetricBlend);
      }
      if (grid) {
        grid.visible = false; // The calibrated floor is shared by photo and analytical views.
        const materials = Array.isArray(grid.material) ? grid.material : [grid.material];
        for (const material of materials) { material.transparent = true; material.opacity = 1 - easedMetricBlend; }
      }
      if (metricFloor) {
        metricFloor.visible = layersRef.current.grid || easedMetricBlend > 0;
        (metricFloor.material as THREE.LineBasicMaterial).opacity = .3 + .15 * easedMetricBlend;
      }
      // The analytic view contains the surface and height solids, not cosmetic trees.
      if (canopy) {
        canopy.visible = layersRef.current.canopy && easedMetricBlend < 1 && wantedFocus === null && !focusAnimating;
        canopy.children.forEach(object => {
          if (!(object instanceof THREE.Mesh)) return;
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          for (const material of materials) { material.transparent = true; material.opacity = 1 - easedMetricBlend; }
        });
      }
      if (floodOverlay && vertexMissionElevations) {
        const screening = missionScreenRef.current;
        floodOverlay.visible = screening.enabled && wantedFocus === null && !focusAnimating;
        if (screening.enabled && screening.thresholdM !== lastMissionThreshold) {
          const risk = floodOverlay.geometry.getAttribute('floodRisk') as THREE.BufferAttribute;
          for (let index = 0; index < risk.count; index += 1) {
            const elevation = vertexMissionElevations[index];
            risk.setX(index, Number.isFinite(elevation) && elevation <= screening.thresholdM ? 1 : 0);
          }
          risk.needsUpdate = true;
          lastMissionThreshold = screening.thresholdM;
        }
      }
      if (highGroundMarkers) highGroundMarkers.visible = missionScreenRef.current.enabled && wantedFocus === null && !focusAnimating;
      readoutAt += delta;
      if (readoutAt > 0.35 && readoutRef.current) {
        // A perspective scale is local to one depth: measure on a horizontal
        // plane through the orbit target and explicitly label that location.
        const bar = scaleBarRef.current;
        if (bar && terrainStats) {
          const width = container.clientWidth;
          const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -controls.target.y);
          const first = new THREE.Vector3(), last = new THREE.Vector3();
          const scaleRay = new THREE.Raycaster();
          camera.updateMatrixWorld();
          scaleRay.setFromCamera(new THREE.Vector2(-120 / width, 0), camera);
          const left = scaleRay.ray.intersectPlane(plane, first);
          scaleRay.setFromCamera(new THREE.Vector2(120 / width, 0), camera);
          const right = scaleRay.ray.intersectPlane(plane, last);
          const metric = Boolean(terrainStats.gsdX && terrainStats.gsdY);
          const extentX = (terrainStats.width - 1) * (metric ? terrainStats.gsdX! : 1);
          const extentY = (terrainStats.height - 1) * (metric ? terrainStats.gsdY! : 1);
          const span = left && right ? Math.hypot((last.x - first.x) / sceneWidth * extentX, (last.z - first.z) / sceneDepth * extentY) : NaN;
          const length = sceneScaleLength(span);
          bar.hidden = length === null;
          if (length !== null) {
            bar.style.setProperty('--scale-width', `${120 * length / span}px`);
            bar.dataset.compact = String(120 * length / span < 110);
            bar.querySelector('[data-scale-half]')!.textContent = formatGridNumber(length / 2);
            bar.querySelector('[data-scale-end]')!.textContent = `${formatGridNumber(length)} ${metric ? 'm' : 'px'}`;
            bar.dataset.length = String(length); bar.dataset.unit = metric ? 'm' : 'px';
          }
        }
        const azimuth = (Math.atan2(camera.position.x, camera.position.z) * 180 / Math.PI + 360) % 360;
        readoutRef.current.textContent = walking ? !walkMotion?.grounded ? 'WALK · JUMP' : terrainStats?.units === 'm' ? 'WALK · EYE 1.80 m' : 'WALK · HUMAN-SCALE PREVIEW'
          : `AZ ${azimuth.toFixed(0)}° · ALT ${camera.position.y.toFixed(0)} scene`;
        // Read-only diagnostics for navigation regression checks; no imagery or predictions duplicated.
        container.dataset.navigation = JSON.stringify({ mode: fly ? exploreModeRef.current : modeRef.current,
          x: camera.position.x, y: camera.position.y, z: camera.position.z,
          ground: walking && walkFeet && walkWorld ? (() => { const floor = walkWorld.ground(walkFeet.x, walkFeet.z); return floor === null ? null : floor * exaggerationRef.current; })() : null,
          grounded: walking ? walkMotion?.grounded ?? true : null,
          verticalVelocity: walking ? walkMotion?.velocityY ?? 0 : null,
          eyeMetres: walking && walkWorld && walkFeet && terrainStats?.units === 'm'
            ? (camera.position.y / exaggerationRef.current - walkFeet.y) / walkWorld.scaleY : null,
          distance: walking ? walkDistance : null, yaw: flyYaw, pitch: flyPitch });
        readoutAt = 0;
      }
      updateFlyHover(now);
      updateSelectionGlow(delta, displayScaleY);
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      disposed = true;
      if (enterWalkRef.current === enterWalk) enterWalkRef.current = null;
      if (resetClassGeometryRef.current === resetClassGeometry) resetClassGeometryRef.current = null;
      delete container.dataset.navigation;
      mapFeedRef.current = null;
      delete container.dataset.classFocus;
      delete container.dataset.terrainMesh;
      delete container.dataset.displayExaggeration;
      delete container.dataset.selectionGlow;
      selectionGlow.dispose();
      selectionBuildings.clear();
      selectionRegions = null;
      classMap?.dispose();
      revealActiveRef.current = false;
      clearHover();
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', resize);
      resizeObserver.disconnect();
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', clearMovementKeys);
      document.removeEventListener('pointerlockchange', onScenePointerLockChange);
      document.removeEventListener('mousemove', onMouseMove);
      renderer.domElement.removeEventListener('pointerdown', rememberPointer);
      renderer.domElement.removeEventListener('pointerup', inspect);
      renderer.domElement.removeEventListener('pointerdown', activateFlyPointerLock);
      window.removeEventListener('blur', clearHover);
      if (document.pointerLockElement === renderer.domElement) document.exitPointerLock();
      controls.dispose();
      cameraPresetRef.current = null;
      terrainPresentationRef.current = null;
      motionPreference.removeEventListener('change', onMotionPreference);
      clearSightlineVisuals();
      const disposedTextures = new Set<THREE.Texture>();
      const materialsToDispose = new Set<THREE.Material>();
      for (const material of [opticalMaterial, elevationMaterial, ...metricMaterials.map(controller => controller.material)]) {
        if (material) materialsToDispose.add(material);
      }
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh || object instanceof THREE.LineSegments) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach(material => materialsToDispose.add(material));
        }
      });
      materialsToDispose.forEach(material => {
        if (material instanceof THREE.MeshStandardMaterial) for (const texture of [material.map, material.bumpMap, material.roughnessMap]) {
          if (texture && !disposedTextures.has(texture)) { texture.dispose(); disposedTextures.add(texture); }
        }
        material.dispose();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [sceneInput]);

  const loadHeightFile = async (file: File | undefined) => {
    if (!file) return;
    if (!/\.tiff?$/i.test(file.name)) {
      setError('Height input must be a .tif or .tiff raster.');
      return;
    }
    demoRequestRef.current += 1;
    applyDisplayScale(1, true);
    setActiveDemoId(null);
    setDemoValidation(null);
    setValidationMetrics(null);
    setGeneratedProducts(null);
    setMissionAnalysis(null);
    resetSightline();
    setStatus(`Reading ${file.name}…`);
    if (!sceneInput.textureUrl) setSurfaceStyle('metric');
    setSceneInput((current) => ({
      ...current,
      heightBuffer: null,
      analysisBuffer: null,
      terrainBuffer: null,
      inferHeightUnits: true,
      missionBuffer: null,
      missionSource: null,
      buildingProbabilityBuffer: null,
      vegetationProbabilityBuffer: null,
      semanticClassBuffer: null,
      confidenceBuffer: null,
      structures: [],
      label: file.name,
      productKind: 'ndsm',
      proof: 'User-supplied height raster with optional optical texture.',
    }));
    const heightBuffer = await file.arrayBuffer();
    setSceneInput((current) => ({ ...current, heightBuffer, analysisBuffer: heightBuffer.slice(0), missionBuffer: null, missionSource: null, structures: [], label: file.name }));
  };

  const loadRgbFile = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Texture input must be a PNG, JPG, or browser-readable image.');
      return;
    }
    const textureUrl = URL.createObjectURL(file);
    setSceneInput((current) => ({ ...current, textureUrl, animateImport: false }));
  };

  const runSatelliteInference = async (file: File | undefined) => {
    if (!file) return;
    if (importPendingRef.current || revealActiveRef.current) return;
    fullMapOpenRef.current = false; setFullMapOpen(false);
    if (file.size > MAX_UPLOAD_BYTES) {
      setError('This file is larger than the 512 MB local processing limit. Crop or resample it first.');
      setStatus('Upload rejected before processing');
      return;
    }
    setError(null);
    const requestId = ++demoRequestRef.current;
    importPendingRef.current = true;
    setImportPending(true);
    pendingValidationOpenRef.current = false;
    setValidationEvidenceOpen(false);
    revealActiveRef.current = false;
    setRevealPhase(null);
    setSurfaceStyle('optical');
    applyDisplayScale(1, true);
    styleRef.current = 'optical';
    modeRef.current = 'orbit';
    setMode('orbit');
    setInspection(null);
    setMissionScreenOpen(false);
    if (document.pointerLockElement) document.exitPointerLock();
    setActiveDemoId(null);
    setDemoValidation(null);
    setValidationMetrics(null);
    setGeneratedProducts(null);
    setMissionAnalysis(null);
    resetSightline();
    const sizeMb = file.size / (1024 * 1024);
    const startedAt = Date.now();
    setStatus(`${file.name} · ${sizeMb.toFixed(1)} MB · uploading + local inference…`);
    const progressTimer = window.setInterval(() => {
      const elapsed = Math.round((Date.now() - startedAt) / 1000);
      if (requestId === demoRequestRef.current) setStatus(`${file.name} · processing locally · ${elapsed}s elapsed…`);
    }, 1000);
    const form = new FormData();
    form.append('image', file);
    if (referenceFile) {
      form.append('reference', referenceFile);
      form.append('reference_kind', referenceKind);
    }
    let sceneQueued = false;
    try {
      let georeferenced = false;
      if (/\.tiff?$/i.test(file.name) && autoTerrain && !demFile && !gcpFile) {
        const tiff = await fromBlob(file);
        const raster = await tiff.getImage();
        try { georeferenced = hasTerrainGeoreference(raster.getGeoKeys() as Record<string, unknown> | null, raster.getBoundingBox()); }
        catch { /* An ordinary TIFF has no map transform: keep the existing surface-only path. */ }
      }
      if (requestId !== demoRequestRef.current) return;
      const calibration = terrainCalibrationSource({ georeferenced, automatic: autoTerrain, dem: Boolean(demFile), gcps: Boolean(gcpFile) });
      if (calibration === 'dem') form.append('dem', demFile!);
      else if (calibration === 'gcps') form.append('gcps', gcpFile!);
      else if (calibration === 'public') form.append('auto_dem', 'true');
      else if (/\.jp2$/i.test(file.name) && autoTerrain) form.append('auto_dem_if_georeferenced', 'true');
      const response = await fetch(`${API_BASE}/api/predict`, { method: 'POST', body: form });
      const payload = await response.json() as {
        detail?: string;
        texture_url?: string;
        height_url?: string;
        raw_height_url?: string;
        rdsm_url?: string;
        relative_depth_url?: string;
        building_probability_url?: string | null;
        vegetation_probability_url?: string | null;
        semantic_class_url?: string | null;
        prediction_confidence_url?: string | null;
        structures_url?: string;
        aligned_dem_url?: string | null;
        public_terrain_url?: string | null;
        absolute_dsm_url?: string | null;
        gcp_surface_url?: string | null;
        validation_metrics_url?: string | null;
        validation_reference_url?: string | null;
        validation_error_url?: string | null;
        validation?: ValidationMetrics | null;
        metadata_url?: string;
        metadata?: {
          input_grid?: {
            original_width: number;
            original_height: number;
            processing_width: number;
            processing_height: number;
            resampled_for_interactive_processing: boolean;
          };
          warnings?: string[];
        };
      };
      if (!response.ok || !payload.height_url || !payload.texture_url || !payload.raw_height_url || !payload.structures_url) {
        throw new Error(payload.detail || `Inference returned ${response.status}.`);
      }
      if (requestId !== demoRequestRef.current) return;
      window.clearInterval(progressTimer);
      setStatus('Loading the generated mesh…');
      const [heightResponse, analysisResponse, structuresResponse, buildingProbabilityResponse, vegetationProbabilityResponse, semanticClassResponse, confidenceResponse, missionResponse, terrainResponse] = await Promise.all([
        fetch(`${API_BASE}${payload.height_url}`),
        fetch(`${API_BASE}${payload.raw_height_url}`),
        fetch(`${API_BASE}${payload.structures_url}`),
        payload.building_probability_url ? fetch(`${API_BASE}${payload.building_probability_url}`) : null,
        payload.vegetation_probability_url ? fetch(`${API_BASE}${payload.vegetation_probability_url}`) : null,
        payload.semantic_class_url ? fetch(`${API_BASE}${payload.semantic_class_url}`) : null,
        payload.prediction_confidence_url ? fetch(`${API_BASE}${payload.prediction_confidence_url}`) : null,
        payload.absolute_dsm_url
          ? fetch(`${API_BASE}${payload.absolute_dsm_url}`)
          : payload.gcp_surface_url
            ? fetch(`${API_BASE}${payload.gcp_surface_url}`)
            : null,
        payload.aligned_dem_url ? fetch(`${API_BASE}${payload.aligned_dem_url}`) : null,
      ]);
      if (!heightResponse.ok || !analysisResponse.ok || !structuresResponse.ok) {
        throw new Error('One or more generated scene products could not be loaded.');
      }
      if ((payload.absolute_dsm_url || payload.gcp_surface_url) && !missionResponse?.ok) throw new Error('The calibrated DSM could not be loaded.');
      const [heightBuffer, analysisBuffer, structurePayload, buildingProbabilityBuffer, vegetationProbabilityBuffer, semanticClassBuffer, confidenceBuffer, missionBuffer, terrainBuffer] = await Promise.all([
        heightResponse.arrayBuffer(),
        analysisResponse.arrayBuffer(),
        structuresResponse.json() as Promise<{ structures: BuildingSolid[] }>,
        buildingProbabilityResponse?.ok ? buildingProbabilityResponse.arrayBuffer() : null,
        vegetationProbabilityResponse?.ok ? vegetationProbabilityResponse.arrayBuffer() : null,
        semanticClassResponse?.ok ? semanticClassResponse.arrayBuffer() : null,
        confidenceResponse?.ok ? confidenceResponse.arrayBuffer() : null,
        missionResponse?.ok ? missionResponse.arrayBuffer() : null,
        terrainResponse?.ok ? terrainResponse.arrayBuffer() : null,
      ]);
      if (requestId !== demoRequestRef.current) return;
      sceneQueued = true;
      setSceneInput({
        animateImport: true,
        heightBuffer: missionBuffer ?? heightBuffer,
        analysisBuffer: missionBuffer ?? analysisBuffer,
        terrainBuffer,
        missionBuffer,
        missionSource: payload.absolute_dsm_url
          ? 'dem_anchored_dsm'
          : payload.gcp_surface_url
            ? 'gcp_calibrated_surface'
            : null,
        buildingProbabilityBuffer,
        vegetationProbabilityBuffer,
        semanticClassBuffer,
        confidenceBuffer,
        textureUrl: `${API_BASE}${payload.texture_url}`,
        label: `${file.name} · local model result`,
        productKind: missionBuffer ? 'dsm' : 'ndsm',
        proof: missionBuffer ? 'Calibrated absolute surface rendered from the supplied terrain or GCP datum.' : 'Fresh local inference result. Metric elevation requires an attached terrain datum or GCP calibration.',
        structures: structurePayload.structures,
      });
      if (payload.rdsm_url && payload.relative_depth_url && payload.metadata_url) {
        setGeneratedProducts({
          rawHeightUrl: `${API_BASE}${payload.raw_height_url}`,
          rdsmUrl: `${API_BASE}${payload.rdsm_url}`,
          relativeDepthUrl: `${API_BASE}${payload.relative_depth_url}`,
          buildingProbabilityUrl: payload.building_probability_url ? `${API_BASE}${payload.building_probability_url}` : null,
          vegetationProbabilityUrl: payload.vegetation_probability_url ? `${API_BASE}${payload.vegetation_probability_url}` : null,
          semanticClassUrl: payload.semantic_class_url ? `${API_BASE}${payload.semantic_class_url}` : null,
          predictionConfidenceUrl: payload.prediction_confidence_url ? `${API_BASE}${payload.prediction_confidence_url}` : null,
          alignedDemUrl: payload.aligned_dem_url ? `${API_BASE}${payload.aligned_dem_url}` : null,
          publicTerrainUrl: payload.public_terrain_url ? `${API_BASE}${payload.public_terrain_url}` : null,
          absoluteDsmUrl: payload.absolute_dsm_url ? `${API_BASE}${payload.absolute_dsm_url}` : null,
          gcpSurfaceUrl: payload.gcp_surface_url ? `${API_BASE}${payload.gcp_surface_url}` : null,
          validationMetricsUrl: payload.validation_metrics_url ? `${API_BASE}${payload.validation_metrics_url}` : null,
          validationReferenceUrl: payload.validation_reference_url ? `${API_BASE}${payload.validation_reference_url}` : null,
          validationErrorUrl: payload.validation_error_url ? `${API_BASE}${payload.validation_error_url}` : null,
          metadataUrl: `${API_BASE}${payload.metadata_url}`,
        });
      }
      setValidationMetrics(payload.validation ?? null);
      if (
        payload.validation
        && payload.validation_reference_url
        && payload.validation_error_url
      ) {
        pendingValidationOpenRef.current = true;
      }
      const grid = payload.metadata?.input_grid;
      if (grid?.resampled_for_interactive_processing) {
        setProcessingNotice(
          `Large raster overview: ${grid.original_width} × ${grid.original_height} → ${grid.processing_width} × ${grid.processing_height}; CRS and full extent preserved.`,
        );
        setStatus(`${file.name} · overview inference complete`);
      } else {
        setStatus(`${file.name} · inference complete`);
      }
    } catch (reason) {
      if (requestId !== demoRequestRef.current) return;
      const message = reason instanceof Error ? reason.message : String(reason);
      setError(
        message.includes('fetch')
          ? 'The local inference service is not running. Start DepthWizard API, then retry.'
          : message,
      );
      setStatus('Upload inference unavailable');
    } finally {
      window.clearInterval(progressTimer);
      if (requestId === demoRequestRef.current && !sceneQueued) {
        importPendingRef.current = false;
        setImportPending(false);
      }
    }
  };

  const toggleLayer = (layer: keyof Layers) => setLayers((current) => ({ ...current, [layer]: !current[layer] }));
  const selected = inspection?.rawValue ?? stats?.mean ?? 0;
  const isAbsoluteScene = sceneInput.productKind === 'dsm';
  const overlayVisible = sixClassOverlayVisible(sixClassLayer);
  // A pinned inspection updates when classification arrives, even in photo view.
  const readoutFor = (data: InspectionData) => inspectionReadout(data.sampleUv
    ? withSixClass(data, sixClassLayer, ...data.sampleUv, classification.status) : data, stats?.units ?? 'relative', sceneInput.productKind);
  const flightReadout = flightInspection ? readoutFor(flightInspection) : null;
  const selectedReadout = inspection ? readoutFor(inspection) : null;
  const selectedLocalRelief = isAbsoluteScene && stats
    ? Math.max(0, selected - stats.minimum)
    : selected;
  const productLabel = isAbsoluteScene
    ? 'ABSOLUTE DSM'
    : sceneInput.productKind === 'relative' || stats?.units === 'relative'
      ? 'RELATIVE rDSM'
      : 'PREDICTED nDSM';
  const activeDemo = activeDemoId ? DEMO_SCENES[activeDemoId] : null;
  const hasAbsoluteProduct = isAbsoluteScene || Boolean(generatedProducts?.absoluteDsmUrl || generatedProducts?.gcpSurfaceUrl);
  const hasLiveValidationEvidence = Boolean(
    validationMetrics
    && generatedProducts?.validationReferenceUrl
    && generatedProducts.validationErrorUrl,
  );
  const visibleValidationMetrics = validationMetrics ?? demoValidation?.metrics ?? null;
  const comparisonMetrics = evidenceOverride?.metrics ?? (hasLiveValidationEvidence && validationMetrics
    ? validationMetrics
    : demoValidation?.metrics ?? FOREST_EVIDENCE.metrics);
  const comparisonTextureUrl = evidenceOverride?.textureUrl ?? (hasLiveValidationEvidence || demoValidation
    ? sceneInput.textureUrl
    : FOREST_EVIDENCE.textureUrl);
  const comparisonPredictionUrl = evidenceOverride?.predictionUrl ?? (hasLiveValidationEvidence && generatedProducts
    ? generatedProducts.rawHeightUrl
    : demoValidation?.predictionUrl ?? FOREST_EVIDENCE.predictionUrl);
  const comparisonReferenceUrl = evidenceOverride?.referenceUrl ?? (hasLiveValidationEvidence && generatedProducts?.validationReferenceUrl
    ? generatedProducts.validationReferenceUrl
    : demoValidation?.referenceUrl ?? FOREST_EVIDENCE.referenceUrl);
  const comparisonErrorUrl = evidenceOverride?.errorUrl ?? (hasLiveValidationEvidence && generatedProducts?.validationErrorUrl
    ? generatedProducts.validationErrorUrl
    : demoValidation?.errorUrl ?? FOREST_EVIDENCE.errorUrl);
  const comparisonTitle = evidenceOverride?.title ?? (hasLiveValidationEvidence
    ? 'Uploaded scene: model vs reference'
    : demoValidation?.title ?? FOREST_EVIDENCE.title);
  const comparisonProvenance = evidenceOverride?.provenance ?? (hasLiveValidationEvidence
    ? 'Uploaded reference aligned to the model grid for a dense, pixel-for-pixel comparison.'
    : demoValidation?.provenance ?? FOREST_EVIDENCE.provenance);
  const missionResults = useMemo(() => {
    if (!missionAnalysis) return null;
    const scenarioElevationM = missionAnalysis.minimumM + scenarioRiseM;
    const exposedPixels = lowerBound(missionAnalysis.sortedElevationsM, scenarioElevationM);
    const exposedFraction = exposedPixels / Math.max(1, missionAnalysis.sortedElevationsM.length);
    const exposedAreaM2 = missionAnalysis.pixelAreaM2 === null
      ? null
      : exposedPixels * missionAnalysis.pixelAreaM2;
    const affectedBuildings = missionAnalysis.mappedBuildings.filter(
      (building) => building.groundElevationM <= scenarioElevationM,
    );
    const roofCandidates = affectedBuildings.filter(
      (building) => building.roofElevationM >= scenarioElevationM + 3,
    );
    const highGroundMarginM = Math.max(5, Math.min(20, missionAnalysis.scenarioMaximumM * 0.2));
    const highGroundPixels = missionAnalysis.sortedElevationsM.length
      - lowerBound(missionAnalysis.sortedElevationsM, scenarioElevationM + highGroundMarginM);
    return {
      scenarioElevationM,
      exposedFraction,
      exposedAreaM2,
      affectedBuildings: affectedBuildings.length,
      mappedBuildings: missionAnalysis.mappedBuildings.length,
      roofCandidates: roofCandidates.length,
      highGroundFraction: highGroundPixels / Math.max(1, missionAnalysis.sortedElevationsM.length),
      highGroundMarginM,
    };
  }, [missionAnalysis, scenarioRiseM]);
  const sightlineProfilePaths = useMemo(() => {
    if (!sightlineResult?.profile.length) return null;
    const values = sightlineResult.profile.flatMap((sample) => [sample.terrain, sample.ray]);
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const range = Math.max(1e-6, maximum - minimum);
    const toPath = (key: keyof SightlineProfileSample) => sightlineResult.profile
      .map((sample, index) => {
        const x = sightlineResult.profile.length === 1
          ? 0
          : index / (sightlineResult.profile.length - 1) * 100;
        const y = 39 - (sample[key] - minimum) / range * 35;
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(' ');
    return { terrain: toPath('terrain'), ray: toPath('ray') };
  }, [sightlineResult]);

  const openEvaluationEvidence = (evidenceId: EvaluationEvidenceId) => {
    if (evidenceId === 'urban') return; // Restricted upstream imagery is not redistributed.
    const evidence = evidenceId === 'sparse'
        ? DEMO_SCENES.sparse.validation
        : evidenceId === 'hilly'
          ? DEMO_SCENES.hilly.validation
          : FOREST_EVIDENCE;
    if (!evidence) return;
    setEvidenceOverride({ ...evidence, textureUrl: evidenceId === 'forest'
        ? FOREST_EVIDENCE.textureUrl : DEMO_SCENES[evidenceId].textureUrl });
    setLandscapeEvaluationOpen(false);
    setValidationEvidenceOpen(true);
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-block"><span className="brand-mark">DW</span><div><p className="eyebrow">Single-view surface intelligence</p><h1>DepthWizard</h1></div></div>
        <div className="run-strip"><span className="pulse" /><span>LOCAL INFERENCE SERVICE</span><strong>{engineStatus === 'ready' ? 'ENGINE READY' : engineStatus === 'offline' ? 'ENGINE OFFLINE' : 'CHECKING…'}</strong></div>
        <button className="evidence-button" onClick={() => setLandscapeEvaluationOpen(true)} type="button"><span>EVAL</span> 4 categories</button>
        <button className={`reference-button${referenceFile ? ' attached' : ''}`} onClick={() => referenceInputRef.current?.click()} title={referenceFile ? `Attached: ${referenceFile.name}. Upload the matching RGB image to run validation.` : 'Attach a LiDAR-derived nDSM before uploading the matching RGB image.'} type="button">
          <span>{referenceFile ? 'LiDAR reference ready' : 'Attach LiDAR reference'}</span>
          <small>{referenceFile ? referenceFile.name : 'nDSM GeoTIFF - attach first'}</small>
        </button>
        <ImageImportControls inputRef={satelliteInputRef} disabled={importPending || Boolean(revealPhase)} onImport={file => void runSatelliteInference(file)} />
      </header>

      <section className="workspace">
        <aside className="left-rail panel">
          <div className="panel-title"><span>Pipeline</span><small>5 stages</small></div>
          <div className="pipeline-list">
            {['RGB ingest', 'Relative geometry', 'Surface height', 'Metric calibration', 'Terrain mesh'].map((label, index) => (
              <div className={`pipeline-step ${stats ? 'ready' : importPending ? 'active' : ''}`} key={label}><span className="step-number">0{index + 1}</span><span>{label}</span><i /></div>
            ))}
          </div>
          <div className="source-card">
            <div className="landscape-demo-picker">
              <div><p className="eyebrow">Landscape demonstrations</p><small>one-click evidence</small></div>
              <div className="landscape-demo-buttons">
                {(Object.keys(DEMO_SCENES) as DemoSceneId[]).map((sceneId) => (
                  <button aria-label={`Load ${DEMO_SCENES[sceneId].shortLabel} demonstration`} className={activeDemoId === sceneId ? 'active' : ''} disabled={importPending || Boolean(revealPhase) || (sceneInput.heightBuffer === null && activeDemoId === sceneId)} key={sceneId} onClick={() => void loadDemoScene(sceneId)} type="button">{DEMO_SCENES[sceneId].shortLabel}</button>
                ))}
              </div>
            </div>
            <p className="eyebrow">{importPending ? 'Previous scene · new image processing' : 'Active source'}</p><strong>{sceneInput.label}</strong>
            <div className="source-meta">{stats ? <><span>{`${stats.width} × ${stats.height}`}</span><span>{productLabel}</span><span>{stats.gsd ? `${stats.gsd.toFixed(2)} m GSD` : 'NO DATUM'}</span></> : <span>{importPending || activeDemoId ? 'PREPARING SCENE' : 'AWAITING IMAGE'}</span>}</div>
            {processingNotice && <p className="demo-proof">{processingNotice}</p>}
            <p className="demo-proof">{sceneInput.proof}</p>
            {activeDemo && <p className="demo-metric-summary">{activeDemo.metricSummary}</p>}
            <div className="source-actions"><button className="secondary-button" disabled={importPending || Boolean(revealPhase)} onClick={() => localHeightInputRef.current?.click()} type="button">Load existing height map</button><button className="secondary-button" disabled={importPending || Boolean(revealPhase)} onClick={() => rgbInputRef.current?.click()} type="button">Replace RGB texture</button></div>
            <input ref={localHeightInputRef} hidden accept=".tif,.tiff,image/tiff" type="file" onChange={(event) => void loadHeightFile(event.target.files?.[0])} />
            <input ref={rgbInputRef} hidden accept="image/png,image/jpeg,image/webp" type="file" onChange={(event) => loadRgbFile(event.target.files?.[0])} />
            <div className="calibration-card">
              <p className="eyebrow">Optional absolute calibration</p>
              <button className="secondary-button" onClick={() => demInputRef.current?.click()} type="button">{demFile ? `DEM: ${demFile.name}` : 'Attach terrain DEM'}</button>
              <button className="secondary-button" onClick={() => gcpInputRef.current?.click()} type="button">{gcpFile ? `GCP: ${gcpFile.name}` : 'Attach GCP CSV'}</button>
              <label className="terrain-toggle"><input checked={autoTerrain} onChange={(event) => { setAutoTerrain(event.target.checked); if (event.target.checked) { setDemFile(null); setGcpFile(null); } }} type="checkbox" /><span>Fetch public coarse terrain datum</span></label>
              <p className="layer-caveat">On by default for georeferenced TIFFs. Adds terrain elevation for hills and valleys. Ordinary images keep their surface-only result. Public terrain requires an internet connection; an attached DEM works offline.</p>
              {(demFile || gcpFile || autoTerrain) && <button className="text-button" onClick={() => { setDemFile(null); setGcpFile(null); setAutoTerrain(false); }} type="button">Clear calibration</button>}
              <input ref={demInputRef} hidden accept=".tif,.tiff,image/tiff" type="file" onChange={(event) => { setDemFile(event.target.files?.[0] ?? null); setGcpFile(null); setAutoTerrain(false); }} />
              <input ref={gcpInputRef} hidden accept=".csv,text/csv" type="file" onChange={(event) => { setGcpFile(event.target.files?.[0] ?? null); setDemFile(null); setAutoTerrain(false); }} />
            </div>
            <div className="validation-upload-card">
              <p className="eyebrow">Optional reference validation</p>
              <select aria-label="Reference raster type" value={referenceKind} onChange={(event) => setReferenceKind(event.target.value as 'ndsm' | 'dsm')}>
                <option value="ndsm">Reference nDSM / object height</option>
                <option value="dsm">Reference absolute DSM</option>
              </select>
              <button className="secondary-button" onClick={() => referenceInputRef.current?.click()} type="button">{referenceFile ? `Reference: ${referenceFile.name}` : 'Attach reference GeoTIFF'}</button>
              {referenceFile && <button className="text-button" onClick={() => { setReferenceFile(null); setValidationMetrics(null); }} type="button">Clear reference</button>}
              <input ref={referenceInputRef} hidden accept=".tif,.tiff,image/tiff" type="file" onChange={(event) => { const file = event.target.files?.[0] ?? null; setReferenceFile(file); setValidationMetrics(null); setError(null); if (file) setStatus(`${file.name} - LiDAR reference ready - now upload the matching RGB image`); }} />
              {referenceKind === 'dsm' && <p className="layer-caveat">Absolute DSM validation also requires a terrain DEM or GCP CSV.</p>}
            </div>
          </div>
          <div className="layer-stack">
            <div className="panel-title"><span>Layers</span><small>live</small></div>
            <label><input disabled={!stats || importPending || Boolean(revealPhase)} checked={surfaceStyle === 'optical'} onChange={() => changeSurfaceStyle('optical')} type="radio" /><span>Optical texture</span></label>
            <label><input disabled={!stats || importPending || Boolean(revealPhase)} checked={surfaceStyle === 'elevation'} onChange={() => changeSurfaceStyle('elevation')} type="radio" /><span>Height colours</span></label>
            <label><input disabled={!stats || importPending || Boolean(revealPhase)} checked={surfaceStyle === 'hillshade'} onChange={() => changeSurfaceStyle('hillshade')} type="radio" /><span>Shaded relief</span></label>
            <label><input disabled={!stats?.gsdX || !stats?.gsdY || stats.units !== 'm' || importPending || Boolean(revealPhase)} checked={surfaceStyle === 'slope'} onChange={() => changeSurfaceStyle('slope')} type="radio" /><span>Surface slope · degrees</span></label>
            <label><input disabled={!stats || importPending || Boolean(revealPhase)} checked={surfaceStyle === 'metric'} onChange={() => changeSurfaceStyle('metric')} type="radio" /><span>Metric grid · no photo</span></label>
            {surfaceStyle === 'metric' && <p className="layer-caveat">Photo, façades and illustrative trees are hidden. Surface heights and flat building solids remain. No prediction or export is changed.</p>}
            <label><input checked={layers.wireframe} onChange={() => toggleLayer('wireframe')} type="checkbox" /><span>Mesh wireframe</span></label>
            <label><input checked={layers.grid} onChange={() => toggleLayer('grid')} type="checkbox" /><span>Map grid</span></label>
            <label><input disabled={!stats} checked={layers.contours} onChange={() => toggleLayer('contours')} type="checkbox" /><span>Elevation contours{metricGrid ? ` · ${formatGridNumber(metricGrid.heightInterval)} ${metricGrid.heightUnit === 'm' ? 'm' : 'rel.'}` : ''}</span></label>
            <label title="Optional illustrative tree objects; not measured canopy shape or height"><input checked={layers.canopy} onChange={() => toggleLayer('canopy')} type="checkbox" /><span>Illustrative tree objects*</span></label>
            <p className="layer-caveat">*Off by default. Roofs use the optical image. Façades and window patterns are illustrative, not observed details or measured floors.</p>
          </div>
        </aside>

        <section ref={viewportRef} className={`viewport panel${windowFullscreen ? ' viewport-expanded' : ''}`} data-fullscreen-kind={isFullscreen ? windowFullscreen ? 'window' : 'native' : 'off'}>
          <div className="viewport-toolbar">
            <div className="segmented" role="group" aria-label="Terrain navigation modes">{(['orbit', 'explore', 'inspect', 'sightline', 'profile'] as ViewMode[]).map((item) => item === 'explore' ?
              <div className="explore-menu-anchor" ref={exploreMenuRef} key={item}>
                <button type="button" aria-expanded={exploreMenuOpen} aria-controls="explore-options" aria-pressed={mode === 'explore'}
                  className={mode === 'explore' || exploreMenuOpen ? 'selected' : ''}
                  disabled={!stats || importPending || Boolean(revealPhase) || mode === 'explore'}
                  title={mode === 'explore' ? 'Press Esc before changing Explore mode' : 'Choose aerial flight or a grounded walkthrough'}
                  onClick={() => setExploreMenuOpen(open => !open)}>Explore <span aria-hidden="true">⌄</span></button>
                {exploreMenuOpen && <div id="explore-options" className="explore-menu" role="group" aria-label="Explore options">
                  <p>CHOOSE YOUR PERSPECTIVE</p>
                  <button type="button" onClick={() => startExplore('fly')}><span>01 / FLY</span><strong>Explore from above</strong><small>Free movement · Space up · Shift down</small></button>
                  <button type="button" onClick={() => startExplore('walk')}><span>02 / WALK</span><strong>Step into the scene</strong><small>{stats?.units === 'm' ? '1.80 m eye height · ground-following' : 'Human-scale preview · uncalibrated'} · Space to jump</small></button>
                  <label><input type="checkbox" checked={headMotion} onChange={event => { setHeadMotion(event.target.checked); headMotionRef.current = event.target.checked; }} />Game-style step bob</label>
                  <small className="explore-caveat">WASD + mouse · G for 3× pace · Esc to exit.<br />Walk follows the reconstructed surface, not surveyed ground. {stats?.gsdX && stats?.gsdY ? 'Display exaggeration still applies.' : 'Ground scale unknown: walking pace is approximate.'}</small>
                </div>}
              </div> : <button aria-pressed={mode === item} className={mode === item ? 'selected' : ''} disabled={!stats || importPending || Boolean(revealPhase) || mode === 'explore'} onClick={() => selectViewMode(item)} key={item} title={mode === 'explore' ? 'Press Esc to exit Explore first' : undefined} type="button">{item}</button>)}</div>
            <div className="toolbar-readouts">
              <div className="terrain-relief-controls">
                <label className="exaggeration-control"><span>Z × {exaggeration.toFixed(1)}{exaggeration === 1 && stats?.gsdX && stats?.gsdY ? ' · true scale' : ' · display scale'}</span><input aria-label="Vertical exaggeration" min="0.6" max="12" step="0.1" type="range" value={exaggeration} onChange={(event) => applyDisplayScale(Number(event.target.value))} /></label>
                <button className="auto-relief-button" type="button" disabled={!stats || importPending || Boolean(revealPhase)} aria-pressed={automaticRelief} title="Fit elevation relief to the scene. Only the display is boosted; source measurements stay unchanged." onClick={() => { applyDisplayScale(recommendedExaggeration, true); cameraPresetRef.current?.(false); }}>Auto relief</button>
              </div>
              <div ref={readoutRef} className="view-readout">AZ 000° · ALT 0 scene</div>
            </div>
            <div className="toolbar-actions" role="group" aria-label="Viewport actions">
              <button aria-pressed={missionScreenOpen} className="mission-screen-button" onClick={() => setMissionScreenOpen((current) => !current)} type="button">{missionScreenOpen ? 'CLOSE SCREEN' : 'RISK SCREEN'}</button>
              <button aria-pressed={isFullscreen} className="fullscreen-button" onClick={() => void toggleFullscreen()} type="button">{isFullscreen ? 'EXIT FULLSCREEN' : 'FULLSCREEN'}</button>
            </div>
          </div>
          <div className="terrain-canvas-wrap">
            <div ref={containerRef} className={`terrain-canvas${mode === 'explore' ? ' fly-mode' : ''}`} aria-label={surfaceStyle === 'metric' ? 'Interactive height surface with analytical grid' : 'Interactive textured 3D terrain'} />
            {stats && !importPending && !revealPhase && <>
              <div className="terrain-camera-tools" aria-label="Terrain camera presets"><button type="button" disabled={mode === 'explore'} onClick={() => cameraPresetRef.current?.(false)}>Fit terrain</button><button type="button" disabled={mode === 'explore'} onClick={() => cameraPresetRef.current?.(true)}>Top down</button><button type="button" onClick={() => { applyDisplayScale(stats.gsdX && stats.gsdY ? 1 : 1.7); cameraPresetRef.current?.(false); }}>{stats.gsdX && stats.gsdY ? 'True scale' : 'Reset view scale'}</button></div>
              {mode !== 'profile' && mode !== 'explore' && <section id="scene-height-detail" className="terrain-elevation-key" aria-label="Surface elevation legend" hidden={!isFullscreen && openSceneDetail !== 'height'}>
                {!isFullscreen && <button type="button" className="scene-detail-close" aria-label="Close height info" onClick={() => setSceneDetail(null)}>×</button>}
                <span>{surfaceStyle === 'slope' ? 'SURFACE SLOPE' : isAbsoluteScene ? 'SURFACE ELEVATION · DATUM' : stats.units === 'm' ? 'HEIGHT ABOVE GROUND' : 'RELATIVE SURFACE'}</span>
                <div className="terrain-colour-ramp" />
                <div><b>{surfaceStyle === 'slope' ? '0°' : formatValue(stats.minimum, stats.units)}</b><b>{surfaceStyle === 'slope' ? '60°+' : formatValue(stats.maximum, stats.units)}</b></div>
                <p>{formatValue(stats.maximum - stats.minimum, stats.units)} relief · {stats.gsd ? `${formatGridNumber(stats.gsd)} m grid` : 'map scale unknown'}{activeDemoId === 'hilly' ? ' · 30 m terrain source' : ''}</p>
                <small>{stats.units === 'm' && stats.gsdX && stats.gsdY ? `Z × ${exaggeration.toFixed(1)} · ${exaggeration === 1 ? 'true proportions' : 'display exaggeration'}` : 'Normalized display · scale unavailable'}{layers.contours && metricGrid ? ` · contours ${formatGridNumber(metricGrid.heightInterval)} ${stats.units}` : ''}</small>
                {automaticRelief && exaggeration > 1 && stats.gsd && <small>Auto relief boosts the view only. Elevations and measurements stay unchanged.</small>}
                {stats.units === 'm' && stats.gsdX && stats.gsdY && (isAbsoluteScene || sceneInput.inferHeightUnits) && <>
                  <div className="terrain-preview-switch" role="group" aria-label="Terrain surface presentation">
                    {([true, false] as const).map(smooth => <button key={String(smooth)} type="button" aria-pressed={smoothTerrain === smooth}
                      onClick={() => { smoothTerrainRef.current = smooth; setSmoothTerrain(smooth); terrainPresentationRef.current?.(smooth, terrainSmoothnessRef.current); }}
                      title={smooth ? 'Round raster-scale bumps in the 3D preview. Source measurements stay unchanged.' : 'Display the unfiltered source elevation surface.'}>{smooth ? 'Smooth' : 'Source'}</button>)}
                  </div>
                  {smoothTerrain && <label className="terrain-smoothness-control"><span>Surface smoothness</span><input type="range" aria-label="Surface smoothness" min="1" max="3" step="1" value={terrainSmoothness}
                    onChange={event => { const strength = Number(event.target.value); terrainSmoothnessRef.current = strength; setTerrainSmoothness(strength); terrainPresentationRef.current?.(true, strength); }} /><output>{['', 'Fine', 'Balanced', 'Soft'][terrainSmoothness]}</output></label>}
                  <small>{smoothTerrain ? 'Smoothed preview' : 'Source surface'} · measurements use source heights.</small>
                </>}
                {!isAbsoluteScene && stats.gsd && <small>Terrain altitude is absent. Attach a DEM or enable public terrain before uploading to reconstruct hills.</small>}
                <div className="terrain-legend-actions"><button type="button" aria-pressed={surfaceStyle === 'elevation'} onClick={() => changeSurfaceStyle(surfaceStyle === 'elevation' ? 'optical' : 'elevation')}>{surfaceStyle === 'elevation' ? 'Photo texture' : 'Elevation colours'}</button><button type="button" onClick={() => selectViewMode('profile')}>Surface profile ↗</button></div>
              </section>}
              {mode === 'profile' && <TerrainProfile profile={profile} picking={profilePicking} onReset={() => { profileResetRef.current += 1; }} onClose={() => selectViewMode('orbit')} />}
            </>}
            <div className="scene-label"><span>{error ? 'SCENE ERROR' : status.toUpperCase()}</span><strong>{overlayVisible ? '6-CLASS IDENTIFICATION / HEIGHTS UNCHANGED' : revealPhase ? 'RECONSTRUCTION REVEAL' : surfaceStyle === 'metric' ? 'HEIGHT SURFACE / GRID + BUILDING SOLIDS' : surfaceStyle === 'optical' ? 'TEXTURE + DSM' : 'ELEVATION ANALYSIS'}</strong></div>
            <button className="metric-view-button" disabled={!stats || importPending || Boolean(revealPhase)} aria-label="Toggle metric grid view" aria-pressed={surfaceStyle === 'metric'} onClick={() => changeSurfaceStyle(surfaceStyle === 'metric' ? 'optical' : 'metric')} type="button"><span aria-hidden="true">▦</span>{surfaceStyle === 'metric' ? 'PHOTO VIEW' : 'METRIC GRID'}</button>
            {!sceneInput.heightBuffer && !importPending && <section className="import-waiting-stage empty-scene" aria-label="Empty 3D workspace">
              <div className="import-waiting-grid" aria-hidden="true" />
              <div className="import-waiting-copy"><span>YOUR IMAGE. YOUR 3D WORLD.</span>
                <strong>{activeDemoId && !error ? `Loading ${DEMO_SCENES[activeDemoId].shortLabel.toLowerCase()} reference…` : 'Drop an image. Explore it in 3D.'}</strong>
                <p>{error || 'Drag a satellite image here, paste a copied image with Ctrl+V / ⌘V, or browse your files. You can also choose a reference scene.'}</p>
                {(!activeDemoId || error) && <button className="empty-upload-button" onClick={() => satelliteInputRef.current?.click()} type="button">Upload an image <span aria-hidden="true">↗</span></button>}
                <small>GeoTIFF / TIFF · PNG · JPG · WebP · BMP · JP2<br />Drop the original GeoTIFF to keep its map coordinates.</small>
              </div>
            </section>}
            {(importPending || importHandoff) && <div className={`import-waiting-stage${importHandoff && !importPending ? ' scene-handoff' : ''}`} role="status">
              <div className="import-waiting-grid" aria-hidden="true"><i /></div>
              <div className="import-waiting-copy"><span>BUILDING YOUR SCENE</span><strong>Preparing the height surface</strong><p>{status}</p><small>Waiting for the model result · no heights shown yet</small></div>
            </div>}
            {revealPhase && <section className="scene-reveal-card" aria-label="Reconstruction reveal">
              <div><span>PREDICTION READY / VISUAL REVEAL</span><button type="button" onClick={() => { skipRevealRef.current = true; }}>Skip animation</button></div>
              <strong aria-live="polite">{REVEAL_LABELS[revealPhase]}</strong>
              <progress ref={revealProgressRef} max={1} defaultValue={0} aria-label="Visual reveal progress" />
              <div className="reveal-stages">{(['grid', 'footprints', 'heights', 'texture'] as const).map(stage => <span key={stage} className={stage === revealPhase ? 'active' : ''}>{stage === 'footprints' ? 'Shapes' : stage}</span>)}</div>
              <small>Actual result, animated for presentation · not live detection</small>
            </section>}
            {surfaceStyle === 'metric' && metricGrid && <section className="metric-grid-key" aria-label="Grid scale and height legend">
              <div><span>XY GRID</span><strong>{formatGridNumber(metricGrid.spacing)} × {formatGridNumber(metricGrid.spacing)} {metricGrid.horizontalUnit}</strong></div>
              <div><span>HEIGHT BANDS</span><strong>{formatGridNumber(metricGrid.heightInterval)} {metricGrid.heightUnit === 'm' ? 'm' : 'relative'}</strong></div>
              <div className="metric-height-ramp" aria-hidden="true" />
              <div className="metric-height-labels"><span>{formatGridNumber(metricGrid.minimum)}</span><span>{formatGridNumber(metricGrid.maximum)} {metricGrid.heightUnit === 'm' ? 'm' : 'rel.'}</span></div>
              <p>{metricGrid.heightUnit === 'relative' ? 'Relative surface · no metre-scale heights' : isAbsoluteScene ? 'Surface elevation above sea level' : 'Estimated height above ground'}</p>
              <small>{metricGrid.horizontalUnit === 'px' ? 'Horizontal scale unknown: grid uses pixels.' : 'Grid spacing follows the raster map scale.'} Display mesh may be smoothed; inspect for raw values. Not a bare-earth DEM.</small>
            </section>}
            {error && <div className="error-toast">{error}</div>}
            {mode === 'inspect' && stats && <section className="class-focus-compact" aria-label="Scene class selector">
              <span>HIGHLIGHT CLASS</span>
              {SIX_CLASS_NAMES.map((name, index) => <button type="button" key={name} aria-pressed={classView.selected === index} disabled={!classification.result?.classes[index].pixels}
                onClick={() => onClassView({ selected: index, visible: true })}>{name}<small>{classification.result ? `${classification.result.classes[index].coverage_percent.toFixed(1)}%` : '—'}</small></button>)}
              <button type="button" disabled={!classification.result} aria-pressed={classView.selected === null} onClick={() => onClassView({ ...classView, selected: null })}>All classes</button>
            </section>}
            {mode === 'explore' && <div className={`control-hint fly-control${flyPointerLocked ? ' locked' : ''}`}><strong>{exploreMode === 'walk' ? 'WALK MODE' : 'FLY MODE'}</strong><span>Mouse look{!flyPointerLocked && ' · click to capture'} · WASD move{exploreMode === 'fly' ? ' · Space up · Shift down' : ' · Space jump'} · G: 3× · {isFullscreen ? 'Esc → Orbit · Esc again → exit fullscreen' : 'Esc → Orbit'}<br />Aim for 2s or click to scan · {exploreMode === 'walk' ? stats?.units === 'm' ? 'Standing eye height 1.80 m' : 'Human-scale preview, not metric' : 'Collision-aware flight'}{exploreMode === 'walk' && <><br />{stats?.gsdX && stats?.gsdY ? 'Map-scaled pace · display exaggeration applies' : 'Ground scale unknown · approximate pace'} · jump low steps; walls stay solid</>}</span></div>}
            {mode === 'explore' && <div className="fly-reticle" aria-hidden="true"><i /><i /></div>}
            {mode === 'inspect' && <div className="control-hint">{classView.selected !== null && classView.visible ? `${SIX_CLASS_NAMES[classView.selected].toUpperCase()} highlighted · heights unchanged · click to inspect` : 'Click to inspect · mint outline marks the selection · Esc clear'}</div>}
            {mode === 'sightline' && <div className="control-hint sightline-control">{sightlineResult ? 'Selection finished · Reset to measure again · Esc or change mode to clear' : sightlineStartSelected ? 'Observer set · click the target point · Esc cancel' : 'Click an observer point, then a target point · Esc cancel'}</div>}
            <div className={`scene-reference-dock${mode === 'explore' ? ' exploring' : ''}`}>
            {stats && sceneInput.heightBuffer && sceneInput.textureUrl && !importPending && !revealPhase && (mode === 'explore' ? <ExploreMap
              key={sceneInput.textureUrl} src={sceneInput.textureUrl} label={sceneInput.label} scene={stats} feed={mapFeedRef}
              open={fullMapOpen} onToggle={toggleExploreMap} onClose={() => closeExploreMap()} /> : <SceneImageReference
              key={sceneInput.textureUrl} src={sceneInput.textureUrl} label={sceneInput.label} compact={Boolean(inspection && mode === 'inspect')}
              onOpen={() => setImageZoomSource(sceneInput.textureUrl)} />)}
            {mode === 'explore' && <section className={`flight-hud${flightInspection ? ' has-target' : ''}`} aria-label="Explore target metadata" aria-live="polite">
              <div className="flight-hud-header"><span><i />SURFACE SCANNER</span><b>{flightInspection ? flightPinned ? 'CLICK SCAN' : 'AIM SCAN' : 'STANDBY'}</b></div>
              {flightInspection && flightReadout && stats ? <>
                <div className="flight-hud-class"><span>CLASS / ESTIMATED</span><strong>{flightReadout.label}</strong></div>
                <div className="flight-hud-value"><span>{flightReadout.measurement}</span><strong>{flightReadout.value === null ? 'Unavailable' : `≈ ${formatValue(flightReadout.value, flightReadout.units)}`}</strong></div>
                {isAbsoluteScene && <><div className="flight-hud-detail"><span>TERRAIN REFERENCE</span><b>{flightInspection.terrainElevation == null ? 'Unavailable' : formatValue(flightInspection.terrainElevation, 'm')}</b></div><div className="flight-hud-detail"><span>SURFACE − TERRAIN</span><b>{flightInspection.aboveGroundHeight == null ? 'Unavailable' : formatValue(flightInspection.aboveGroundHeight, 'm')}</b></div></>}
                <div className="flight-hud-detail"><span>CLASS SOURCE</span><b>{flightReadout.classSource}</b></div>
                <div className="flight-hud-detail"><span>HEIGHT SOURCE</span><b>{flightReadout.heightSource}</b></div>
                {flightInspection.slopeDegrees !== null && <div className="flight-hud-detail"><span>SURFACE SLOPE</span><b>≈ {flightInspection.slopeDegrees.toFixed(1)}°</b></div>}
                <p className={flightReadout.disagreement ? 'scanner-warning' : undefined}>{flightReadout.notes.join(' ')}</p>
                {flightInspection.kind === 'building' && <p>{ROOF_HEIGHT_STATUS}</p>}
              </> : <div className="flight-hud-empty"><strong>Acquire a target</strong><span>Aim for 2 seconds<br />or left-click to scan instantly.</span></div>}
              <div className={`flight-hud-footer${flightBlocked ? ' blocked' : ''}`}>{flightBlocked ? exploreMode === 'walk' ? 'LOW STEP? SPACE TO JUMP · WALLS: WALK AROUND' : 'OBSTACLE / GROUND · SPACE TO CLIMB' : 'COLLISION GUARD ACTIVE · ESC TO EXIT'}</div>
            </section>}
            {mode === 'inspect' && inspection && selectedReadout && stats && <div className="inspect-popover" data-sample-u={inspection.sampleUv?.[0]} data-sample-v={inspection.sampleUv?.[1]}>
              <strong>{selectedReadout.label} inspection</strong>
              {inspection.kind === 'building' && <small>{ROOF_HEIGHT_STATUS}</small>}
              <div><span>{selectedReadout.measurement}</span><b>{selectedReadout.value === null ? 'Unavailable' : formatValue(selectedReadout.value, selectedReadout.units)}</b></div>
              {isAbsoluteScene && <><div><span>Terrain reference</span><b>{inspection.terrainElevation == null ? 'Unavailable' : formatValue(inspection.terrainElevation, 'm')}</b></div><div><span>Surface − terrain</span><b>{inspection.aboveGroundHeight == null ? 'Unavailable' : formatValue(inspection.aboveGroundHeight, 'm')}</b></div></>}
              <div><span>{isAbsoluteScene && !inspection.objectHeight ? 'Local relief above scene low' : 'Display relief'}</span><b>{formatValue(isAbsoluteScene && !inspection.objectHeight ? selectedLocalRelief : inspection.displayValue, stats.units)}</b></div>
              <div><span>Class basis</span><b>{selectedReadout.classSource}</b></div>
              <div><span>Height basis</span><b>{selectedReadout.heightSource}</b></div>
              <div><span>Local slope</span><b>{inspection.slopeDegrees === null ? 'Needs metric GSD' : `${inspection.slopeDegrees.toFixed(1)}°`}</b></div>
              {inspection.sampleUv && <div><span>Raster pixel · column, row</span><b>{Math.round(inspection.sampleUv[0] * (stats.width - 1))}, {Math.round(inspection.sampleUv[1] * (stats.height - 1))}</b></div>}
              <small className={selectedReadout.disagreement ? 'scanner-warning' : undefined}>{selectedReadout.notes.join(' ')}</small>
            </div>}
            </div>
            {missionScreenOpen && missionResults && focusClass === null && <div className="mission-viewport-badge"><span>SCENARIO SCREEN</span><strong>{missionResults.scenarioElevationM.toFixed(1)} m ASL</strong><small>Blue = low-elevation exposure · green = high-ground candidate</small></div>}
            {mode === 'sightline' && sightlineResult && <div className={`sightline-viewport-badge ${sightlineResult.visible ? 'visible' : 'blocked'}`}><span>TERRAIN-SURFACE SIGHTLINE</span><strong>{sightlineResult.visible ? 'VISIBLE' : 'BLOCKED'}</strong><small>{sightlineResult.distanceM === null ? 'Distance unavailable without map scale' : `${sightlineResult.distanceM.toFixed(0)} m sampled path`}</small></div>}
            {stats && metricGrid && !importPending && !revealPhase && mode !== 'profile' && mode !== 'explore' && <section id="scene-scale-detail" className="scene-scale" aria-label="Scene scale" hidden={!isFullscreen && openSceneDetail !== 'scale'}>
              {!isFullscreen && <button type="button" className="scene-detail-close" aria-label="Close map scale" onClick={() => setSceneDetail(null)}>×</button>}
              <div ref={scaleBarRef} className="scene-scale-ruler"><div><span>0</span><span data-scale-half /><span data-scale-end /></div><i /><small>Horizontal scale at view centre</small></div>
              <p><span>X × Y</span><b>{formatGridNumber(metricGrid.extentX)} × {formatGridNumber(metricGrid.extentY)} {metricGrid.horizontalUnit}</b></p>
              <p><span>Grid</span><b>{formatGridNumber(metricGrid.spacing)} {metricGrid.horizontalUnit}</b></p>
              <p><span>Z</span><b>{formatGridNumber(metricGrid.minimum)} – {formatGridNumber(metricGrid.maximum)} {metricGrid.heightUnit === 'm' ? 'm' : 'relative'}</b></p>
              <small>{metricGrid.horizontalUnit === 'px' ? 'Pixel distances · map scale unavailable' : `Display Z × ${exaggeration.toFixed(1)} · measurements unscaled`}</small>
            </section>}
            {stats && !importPending && !revealPhase && mode !== 'profile' && mode !== 'explore' && !isFullscreen && <div className="scene-detail-controls" role="group" aria-label="Scene information panels">
              <button type="button" aria-expanded={openSceneDetail === 'height'} aria-controls="scene-height-detail"
                onClick={() => setSceneDetail(openSceneDetail === 'height' ? null : { heightBuffer: sceneInput.heightBuffer, kind: 'height' })}>
                <span aria-hidden="true">▥</span> Height info <i aria-hidden="true">{openSceneDetail === 'height' ? '−' : '+'}</i>
              </button>
              {metricGrid && <button type="button" aria-expanded={openSceneDetail === 'scale'} aria-controls="scene-scale-detail"
                onClick={() => setSceneDetail(openSceneDetail === 'scale' ? null : { heightBuffer: sceneInput.heightBuffer, kind: 'scale' })}>
                <span aria-hidden="true">↔</span> Map scale <i aria-hidden="true">{openSceneDetail === 'scale' ? '−' : '+'}</i>
              </button>}
            </div>}
            {imageZoomSource && imageZoomSource === sceneInput.textureUrl && stats && !importPending && !revealPhase && <OriginalImageViewer
              key={imageZoomSource} src={imageZoomSource} label={sceneInput.label} onClose={() => setImageZoomSource(null)} />}
          </div>
          <div className="timeline"><span>RGB</span><i /><span>rDepth</span><i /><span>nDSM</span><i /><span>DSM</span><i /><strong>3D</strong></div>
        </section>

        <aside className="right-rail panel">
          <div className="panel-title"><span>Scene analysis</span><small>{importPending ? 'previous scene' : stats ? 'computed' : 'waiting'}</small></div>
          <SixClassPreview key={sceneInput.textureUrl} apiBase={API_BASE} textureUrl={sceneInput.textureUrl} demoId={activeDemoId}
            ready={!importPending && !revealPhase && Boolean(stats && sceneInput.heightBuffer)} online={engineStatus === 'ready'}
            state={classification} view={classView} onState={setClassificationState} onView={onClassView} />
          {mode === 'sightline' && <section className="sightline-analysis-card" aria-label="Terrain surface sightline analysis">
            <div className="sightline-card-heading">
              <div><p className="eyebrow">Two-point analysis</p><strong>Terrain-surface sightline</strong></div>
              {sightlineResult && <span className={sightlineResult.visible ? 'visible' : 'blocked'}>{sightlineResult.visible ? 'VISIBLE' : 'BLOCKED'}</span>}
            </div>
            {!sightlineResult ? <div className="sightline-instructions">
              <span className={sightlineStartSelected ? 'complete' : 'active'}>01</span><p><strong>Observer</strong>{sightlineStartSelected ? 'Selected' : 'Click the first surface point'}</p>
              <i />
              <span className={sightlineStartSelected ? 'active' : ''}>02</span><p><strong>Target</strong>{sightlineStartSelected ? 'Click the second surface point' : 'Waiting for observer'}</p>
            </div> : <>
              <div className="sightline-metrics">
                <div><span>Path distance</span><strong>{sightlineResult.distanceM === null ? 'Scale needed' : `${sightlineResult.distanceM.toFixed(0)} m`}</strong></div>
                <div><span>Minimum clearance</span><strong>{sightlineResult.metricHeights ? `${sightlineResult.minimumClearance >= 0 ? '+' : ''}${sightlineResult.minimumClearance.toFixed(1)} m` : `${sightlineResult.minimumClearance.toFixed(3)} relative`}</strong></div>
                <div><span>First obstruction</span><strong>{sightlineResult.blockedAtFraction === null ? 'None sampled' : `${(sightlineResult.blockedAtFraction * 100).toFixed(0)}% along path`}</strong></div>
                <div><span>Profile samples</span><strong>{sightlineResult.sampleCount}</strong></div>
              </div>
              {sightlineProfilePaths && <div className="sightline-profile">
                <svg aria-label="Terrain and line of sight elevation profile" preserveAspectRatio="none" viewBox="0 0 100 42">
                  <path className="profile-terrain" d={sightlineProfilePaths.terrain} />
                  <path className={sightlineResult.visible ? 'profile-ray visible' : 'profile-ray blocked'} d={sightlineProfilePaths.ray} />
                </svg>
                <div><span><i className="terrain-key" />{sightlineResult.metricHeights ? 'DSM surface' : 'Relative surface'}</span><span><i className={sightlineResult.visible ? 'ray-key visible' : 'ray-key blocked'} />Sight ray</span></div>
              </div>}
            </>}
            <button className="sightline-reset" onClick={resetSightline} type="button">Reset observer + target</button>
            <p className="sightline-disclaimer"><strong>Screening only:</strong> calculated against the loaded surface raster with a {stats?.units === 'm' ? '2 m' : 'relative'} observer/target offset. It is not an operational visibility guarantee and does not model Earth curvature, atmosphere, moving vegetation, or prediction uncertainty.</p>
          </section>}
          {missionScreenOpen && <section className="mission-analysis-card" aria-label="Flood exposure screening">
            <div className="mission-card-heading">
              <div><p className="eyebrow">Decision-support preview</p><strong>Flood exposure screen</strong></div>
              <span>DSM ONLY</span>
            </div>
            {missionAnalysis && missionResults ? <>
              <p className="mission-source">{missionSourceLabel(missionAnalysis.source)} · {missionAnalysis.pixelAreaM2 === null ? 'unknown map scale' : 'metric map scale'}</p>
              <label className="scenario-level-control">
                <span><b>Scenario water surface</b><strong>{missionResults.scenarioElevationM.toFixed(1)} m ASL</strong></span>
                <input
                  aria-label="Scenario water level above scene minimum"
                  min="0"
                  max={missionAnalysis.scenarioMaximumM}
                  step={missionAnalysis.scenarioMaximumM > 20 ? 1 : 0.25}
                  type="range"
                  value={scenarioRiseM}
                  onChange={(event) => setScenarioRiseM(Number(event.target.value))}
                />
                <small>+{scenarioRiseM.toFixed(1)} m above the robust scene low (2nd percentile)</small>
              </label>
              <label className="mission-overlay-toggle"><input checked={floodOverlayEnabled} onChange={(event) => setFloodOverlayEnabled(event.target.checked)} type="checkbox" /><span>Show low-elevation overlay and high-ground markers</span></label>
              <div className="mission-metric-grid">
                <div><span>Potentially exposed</span><strong>{(missionResults.exposedFraction * 100).toFixed(1)}%</strong><small>{formatScreeningArea(missionResults.exposedAreaM2)}</small></div>
                <div><span>Mapped buildings</span><strong>{missionResults.mappedBuildings ? `${missionResults.affectedBuildings} / ${missionResults.mappedBuildings}` : 'Not mapped'}</strong><small>{missionResults.mappedBuildings ? 'ground below scenario' : 'needs building outlines'}</small></div>
                <div><span>Roof candidates</span><strong>{missionResults.mappedBuildings ? missionResults.roofCandidates : '—'}</strong><small>geometry only · +3 m clearance</small></div>
                <div><span>High-ground reserve</span><strong>{(missionResults.highGroundFraction * 100).toFixed(1)}%</strong><small>&gt; scenario + {missionResults.highGroundMarginM.toFixed(0)} m</small></div>
                <div><span>Steep DSM surface</span><strong>{missionAnalysis.steepAreaFraction === null ? 'Scale needed' : `${(missionAnalysis.steepAreaFraction * 100).toFixed(1)}%`}</strong><small>DSM slope ≥ 30°</small></div>
                <div><span>Surface range</span><strong>{(missionAnalysis.maximumM - missionAnalysis.minimumM).toFixed(0)} m</strong><small>central 96% of DSM</small></div>
              </div>
              <p className="mission-disclaimer"><strong>Screening estimate:</strong> this threshold highlights low DSM cells only. Surface slopes can include object edges. It does not simulate rainfall, drainage, water flow, structural safety, or an evacuation route.</p>
            </> : <div className="mission-unavailable">
              <strong>Absolute terrain surface required</strong>
              <p>Attach a DEM or GCPs to a georeferenced image, or load the bundled Hilly example. A relative rDSM/nDSM alone cannot honestly predict flood extent.</p>
              <button onClick={() => void loadDemoScene('hilly')} type="button">Load hilly DSM example</button>
            </div>}
          </section>}
          <div className="hero-metric"><p>{selectedReadout ? `${selectedReadout.label} · ${selectedReadout.measurement}` : isAbsoluteScene ? 'Mean surface elevation (datum)' : 'Mean surface value'}</p><strong>{selectedReadout ? selectedReadout.value === null ? 'Unavailable' : `≈ ${formatValue(selectedReadout.value, selectedReadout.units)}` : stats ? formatValue(selected, stats.units) : '—'}</strong><span>{selectedReadout ? selectedReadout.notes.join(' ') : isAbsoluteScene ? 'Surface elevation anchored to the supplied terrain datum' : 'Model estimate; display relief is reported separately'}</span></div>
          {inspection && stats && <div className="metric-grid inspection-grid">
            <div><span>{selectedReadout?.measurement}</span><strong>{formatValue(inspection.rawValue, stats.units)}</strong></div>
            <div><span>{isAbsoluteScene && !inspection.objectHeight ? 'Local relief above scene low' : 'Displayed relief'}</span><strong>{formatValue(isAbsoluteScene && !inspection.objectHeight ? selectedLocalRelief : inspection.displayValue, stats.units)}</strong></div>
            <div><span>Class basis</span><strong>{selectedReadout?.classSource}</strong></div>
            <div><span>Display scaling</span><strong>Z × {exaggeration.toFixed(1)}</strong></div>
            <div><span>Local slope</span><strong>{inspection.slopeDegrees === null ? 'Needs GSD' : `${inspection.slopeDegrees.toFixed(1)}°`}</strong></div>
          </div>}
          {inspection && Math.abs(inspection.displayValue - inspection.rawValue) > 0.25 && <p className="layer-caveat">{inspection.kind === 'vegetation' && inspection.vegetationProbability !== null && inspection.vegetationProbability >= VEGETATION_DISPLAY_THRESHOLD ? 'The learned class is vegetation; displayed smoothing or relief may still differ from the raw export height.' : inspection.kind === 'vegetation' ? 'RGB colour identifies visible greenery for the demo mesh. This is a visual fallback—not validated vegetation-height accuracy.' : 'Displayed relief includes a visualization-only surface aid. Raw height above is the value used for evaluation and export.'}</p>}
          {inspection?.kind === 'building' && <p className="layer-caveat">{ROOF_HEIGHT_STATUS}</p>}
          <div className="metric-grid">
            <div><span>Mean slope</span><strong>{stats?.meanSlope === null || stats?.meanSlope === undefined ? 'Needs GSD' : `${stats.meanSlope.toFixed(1)}°`}</strong></div>
            <div><span>Surface range</span><strong>{stats ? formatValue(stats.maximum - stats.minimum, stats.units) : '—'}</strong></div>
            <div><span>Raster grid</span><strong>{stats ? `${stats.width} × ${stats.height}` : '—'}</strong></div>
            <div><span>Mesh vertices</span><strong>{stats ? `${Math.round(stats.vertices / 1000)}k` : '—'}</strong></div>
          </div>
          {stats && <div className="height-profile"><div className="panel-title"><span>Surface cross-section</span><small>source data</small></div><button className="secondary-button" type="button" onClick={() => selectViewMode('profile')}>Pick A → B on the 3D surface</button><p className="layer-caveat">Distance, elevation change and terrain reference, with CSV export.</p></div>}
          <div className="legend"><p className="eyebrow">Surface ramp</p>{[['#45f3c4', 'Low'], ['#bbf451', 'Moderate'], ['#ffc857', 'High'], ['#ff7168', 'Peak']].map(([color, label]) => <div key={label}><i style={{ background: color }} /><span>{label}</span></div>)}</div>
          <div className="confidence-note"><span>PRODUCT STATUS</span><strong>{!stats ? 'NO SCENE' : hasAbsoluteProduct ? 'ABSOLUTE DSM' : stats.units === 'm' ? 'PREDICTED nDSM' : 'RELATIVE'}</strong><div><i /></div><p>{!stats ? 'Upload an image or select a reference scene to begin.' : isAbsoluteScene ? activeDemoId === 'hilly' ? 'Absolute surface anchored to the supplied Copernicus terrain reference' : generatedProducts?.publicTerrainUrl ? 'Absolute surface anchored to public coarse terrain; see exported provenance' : 'Absolute surface calibrated to the supplied DEM/GCP datum' : 'No absolute elevation claim without terrain/GCP calibration'}</p></div>
          {visibleValidationMetrics && <div className="validation-results"><div className="panel-title"><span>Reference validation</span><small>{visibleValidationMetrics.reference_kind.toUpperCase()}</small></div><div className="metric-grid"><div><span>RMSE</span><strong>{visibleValidationMetrics.rmse_m.toFixed(2)}m</strong></div><div><span>MAE</span><strong>{visibleValidationMetrics.mae_m.toFixed(2)}m</strong></div><div><span>Correlation</span><strong>{visibleValidationMetrics.correlation === null ? '—' : visibleValidationMetrics.correlation.toFixed(3)}</strong></div><div><span>R²</span><strong>{visibleValidationMetrics.r2 === null ? '—' : visibleValidationMetrics.r2.toFixed(3)}</strong></div></div><p>{visibleValidationMetrics.pixel_count.toLocaleString()} aligned pixels · signed bias {visibleValidationMetrics.bias_m.toFixed(2)}m</p><button className="validation-open-button" onClick={() => { setEvidenceOverride(null); setValidationEvidenceOpen(true); }} type="button">Open visual comparison</button></div>}
          {generatedProducts && <div className="export-card"><div className="panel-title"><span>Generated products</span><small>download</small></div><a href={generatedProducts.rawHeightUrl} download>Raw predicted nDSM</a><a href={generatedProducts.rdsmUrl} download>Relative rDSM</a><a href={generatedProducts.relativeDepthUrl} download>Foundation depth prior</a>{generatedProducts.absoluteDsmUrl && <a className="absolute-product" href={generatedProducts.absoluteDsmUrl} download>Absolute DSM</a>}{generatedProducts.gcpSurfaceUrl && <a className="absolute-product" href={generatedProducts.gcpSurfaceUrl} download>GCP-calibrated surface</a>}{generatedProducts.alignedDemUrl && <a href={generatedProducts.alignedDemUrl} download>Aligned terrain DEM</a>}{generatedProducts.publicTerrainUrl && <a href={generatedProducts.publicTerrainUrl} download>Coarse public terrain source</a>}{generatedProducts.validationMetricsUrl && <a className="validation-product" href={generatedProducts.validationMetricsUrl} download>Validation metrics report</a>}{generatedProducts.validationReferenceUrl && <a className="validation-product" href={generatedProducts.validationReferenceUrl} download>Aligned reference GeoTIFF</a>}{generatedProducts.validationErrorUrl && <a className="validation-product" href={generatedProducts.validationErrorUrl} download>Signed error GeoTIFF</a>}{generatedProducts.semanticClassUrl && <a href={generatedProducts.semanticClassUrl} download>Height geometry groups (internal)</a>}{generatedProducts.buildingProbabilityUrl && <a href={generatedProducts.buildingProbabilityUrl} download>Building probability</a>}{generatedProducts.vegetationProbabilityUrl && <a href={generatedProducts.vegetationProbabilityUrl} download>Vegetation probability</a>}{generatedProducts.predictionConfidenceUrl && <a href={generatedProducts.predictionConfidenceUrl} download>Prediction confidence</a>}<a href={generatedProducts.metadataUrl} download>Pipeline metadata</a></div>}
        </aside>
      </section>
      <LandscapeEvaluation
        open={landscapeEvaluationOpen}
        onClose={() => setLandscapeEvaluationOpen(false)}
        onOpenEvidence={openEvaluationEvidence}
      />
      <ValidationComparison
        open={validationEvidenceOpen}
        onClose={() => { setValidationEvidenceOpen(false); setEvidenceOverride(null); }}
        textureUrl={comparisonTextureUrl}
        predictionUrl={comparisonPredictionUrl}
        referenceUrl={comparisonReferenceUrl}
        errorUrl={comparisonErrorUrl}
        metrics={comparisonMetrics}
        title={comparisonTitle}
        provenance={comparisonProvenance}
        heldOut={!hasLiveValidationEvidence}
      />
    </main>
  );
}
