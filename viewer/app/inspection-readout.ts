import type { InspectionData } from './surface-query';

export function inspectionReadout(data: InspectionData, units: 'm' | 'relative', product: 'ndsm' | 'dsm' | 'relative') {
  const classified = data.experimentalClass !== undefined;
  const pending = data.classificationStatus === 'waiting' || data.classificationStatus === 'loading';
  const unavailable = data.classificationStatus === 'error' || data.classificationStatus === 'unsupported' || data.classificationStatus === 'idle';
  const label = pending ? 'Identifying…' : unavailable ? 'Classification unavailable' : classified ? data.experimentalClass ?? 'Unclassified'
    : data.classBasis === 'Unclassified pixel' ? 'Unclassified surface'
    : data.kind === 'building' ? 'Building' : data.kind === 'vegetation' ? 'Vegetation' : product === 'dsm' ? 'Terrain surface' : 'Surface';
  const expectedKind = data.experimentalClass === 'Buildings' ? 'building'
    : ['Trees', 'Low vegetation'].includes(data.experimentalClass ?? '') ? 'vegetation'
    : ['Ground', 'Roads', 'Water'].includes(data.experimentalClass ?? '') ? 'surface' : null;
  const disagreement = expectedKind !== null && expectedKind !== data.kind;
  const relative = units === 'relative' || product === 'relative';
  const absolute = !relative && product === 'dsm' && !data.objectHeight;
  const value = Number.isFinite(data.rawValue) ? data.rawValue : null;
  const measurement = relative ? 'Relative surface score' : data.objectHeight ? 'Building height above ground'
    : absolute ? 'Surface elevation (datum)' : 'Predicted surface above ground';
  const heightSource = relative ? 'Relative surface model' : absolute ? 'Calibrated DSM / terrain datum'
    : data.objectHeight ? 'Protected height model · object summary' : 'Protected height model · raw pixel';
  const classSource = pending ? 'Six-class identification · pending' : unavailable ? 'Six-class identification · unavailable'
    : classified ? 'RGB six-class V3' : data.classBasis
    ?? (data.kind === 'building' ? 'Protected building model' : data.kind === 'vegetation'
      ? (data.vegetationProbability ?? 0) >= .35 ? 'Learned vegetation model' : 'RGB-assisted vegetation'
      : 'Existing three-group surface labels');
  const notes = [relative ? 'No metre-scale height is available.' : absolute
    ? data.terrainElevation != null ? 'Surface minus terrain reference is reported separately; the terrain reference may include vegetation or structures.' : 'Elevation is not object height. Above-ground height needs a separate ground reference.'
    : 'Estimated, not surveyed. Identification does not improve this height by itself.'];
  if (data.experimentalClass === 'Water') notes.push('This is a surface value, never water depth. Water height is not validated.');
  if (data.experimentalClass === 'Roads' || data.experimentalClass === 'Ground') notes.push('Not terrain altitude above sea level unless a calibrated DSM is supplied.');
  if (disagreement) notes.push(`Models disagree: height geometry uses ${data.kind}. Do not treat this as a validated ${label.toLowerCase()} height.`);
  if (pending) notes.push('Identification is pending. The height estimate is independently available.');
  else if (unavailable) notes.push('Six-class identification is unavailable; no replacement category is inferred from height.');
  else if (classified && data.experimentalClass === null) notes.push('No valid six-class label at this location.');
  if (classSource === 'RGB-assisted vegetation') notes.push('Greenery is color-assisted; displayed relief is not validated canopy height.');
  return { label, measurement, heightSource, classSource, value, units: relative ? 'relative' as const : 'm' as const, disagreement, notes };
}
