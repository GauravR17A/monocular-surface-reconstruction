"""CPU-only, train-only audit before another height experiment. Never edits data.

Reads the pinned corrected training manifest, its HighBuild COCO members and
the referenced training rasters only. Cached-prior shape/range are checks, not
proof that a prior was generated from the correct image/model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import rasterize
from affine import Affine

from msr.data.highbuild import annotation_masks, regression_support_mask, _annotation_polygons

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('D:/MSRData/data/multidomain_v2/highbuild_measured_coco_intersection_v1/manifests/train.csv')
EXPECTED = '8f7f85997df53c2dae82d571c52db4ca8f10d4cb5638784e729b38632a1067e4'
SPLITS = Path('D:/MSRData/data/highbuild_full/msr_splits')
EDGES = (0, 2, 5, 10, 20, 40, 80, 200, float('inf'))
BANDS = ('0-2', '2-5', '5-10', '10-20', '20-40', '40-80', '80-200', '>200')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def height_bands(values):
    values = np.asarray(values)
    values = values[np.isfinite(values) & (values > 0)]
    return {name: int(((values > low) & (values <= high)).sum())
            for name, low, high in zip(BANDS, EDGES[:-1], EDGES[1:])}


def prior_status(values, tags, expected_shape):
    finite = np.isfinite(values)
    return {
        'shape_matches': tuple(values.shape) == tuple(expected_shape),
        'all_finite': bool(finite.all()),
        'stored_01': bool(finite.all() and (values >= 0).all() and (values <= 1).all()),
        'declared_units': tags.get('MSR_UNITS'),
        'declared_model': tags.get('MSR_MODEL'),
        # A model directory name is not a source/model-content binding.
        'source_and_weights_hash_tags_present': bool(tags.get('MSR_RGB_SHA256') and tags.get('MSR_MODEL_SHA256')),
    }


def annotation_values(annotation, target, valid):
    polygons = list(_annotation_polygons(annotation))
    points = np.array([point for polygon in polygons for point in polygon['coordinates'][0]])
    x0, y0 = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(points.max(axis=0)).astype(int) + 1, [target.shape[1], target.shape[0]])
    if x1 <= x0 or y1 <= y0:
        return np.empty(0, dtype=np.float32)
    mask = rasterize([(p, 1) for p in polygons], out_shape=(y1-y0, x1-x0), transform=Affine.translation(x0,y0), dtype='uint8') > 0
    support = mask & valid[y0:y1, x0:x1]
    return target[y0:y1, x0:x1][support]


def audit(output):
    if sha(MANIFEST) != EXPECTED:
        raise ValueError('Corrected TRAIN manifest hash changed; refusing to choose another split')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    with MANIFEST.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    with (SPLITS/'train.csv').open(encoding='utf-8-sig', newline='') as stream:
        metadata = {row['webdataset_key']: row for row in csv.DictReader(stream)}
    rows = [row for row in rows if row['target_kind'] == 'building_height']
    if len(rows) != 450:
        raise ValueError('Expected exactly the pinned 450 HighBuild training rows')
    output.mkdir(parents=True)
    archives = {}
    cities = defaultdict(lambda: {'tiles': 0, 'pixels': Counter(), 'annotations': Counter(), 'estimated_excluded': 0})
    tile_reports, instances = [], []
    try:
        for index, row in enumerate(rows):
            sample = row['sample_id']
            meta = metadata[sample]
            if meta['msr_split'] != 'train':
                raise ValueError(f'Non-training source: {sample}')
            archive_path = (SPLITS/meta['msr_shard']).resolve()
            if archive_path.parent != (SPLITS/'shards/train').resolve():
                raise ValueError(f'Archive outside training directory: {archive_path}')
            if archive_path not in archives:
                archives[archive_path] = tarfile.open(archive_path, 'r')
            member = archives[archive_path].extractfile(meta['webdataset_json_member'])
            if member is None:
                raise ValueError(f'Missing COCO: {sample}')
            coco_bytes = member.read()
            coco = json.loads(coco_bytes)
            with rasterio.open(row['surface_path']) as f:
                target = f.read(1)
                shape, transform, crs, nodata = f.shape, f.transform, f.crs, f.nodata
            with rasterio.open(row['valid_mask_path']) as f:
                valid = f.read(1) > 0
                valid_aligned = f.shape == shape and f.transform == transform and f.crs == crs
            with rasterio.open(row['building_mask_path']) as f:
                buildings = f.read(1) > 0
                building_aligned = f.shape == shape and f.transform == transform and f.crs == crs
            with rasterio.open(row['rgb_path']) as f:
                image_valid = np.all(f.read_masks((1,2,3)) > 0, axis=0)
                rgb_aligned = f.shape == shape and f.transform == transform and f.crs == crs
            with rasterio.open(row['relative_prior_path']) as f:
                prior = f.read(1)
                prior_check = prior_status(prior, f.tags(), shape)
                prior_check['grid_aligned'] = f.shape == shape and f.transform == transform and f.crs == crs
            if not all((valid_aligned, building_aligned, rgb_aligned)):
                raise ValueError(f'Misaligned training data: {sample}')
            all_mask, measured, counts = annotation_masks(coco, height=shape[0], width=shape[1])
            expected_valid = regression_support_mask(all_footprints=all_mask, measured_footprints=measured, target=target, nodata=nodata, protocol='strict_measured') > 0
            supported = target[valid & image_valid]
            city = cities[row['region']]
            city['tiles'] += 1
            city['pixels'].update(height_bands(supported))
            city['estimated_excluded'] += counts['estimated_annotations']
            tile = {'sample_id': sample, 'region': row['region'], 'gsd_m': row['gsd_m'] or None,
                    'height_pixels': int(valid.sum()), 'image_pixels': int(image_valid.sum()),
                    'measured_support_matches': bool(np.array_equal(valid, expected_valid)),
                    'semantic_support_matches': bool(np.array_equal(buildings, all_mask > 0)),
                    'height_over_200_pixels': int((supported > 200).sum()),
                    'height_without_image_pixels': int((valid & ~image_valid).sum()),
                    'pixel_bands': height_bands(supported), 'prior': prior_check,
                    'coco_sha256': hashlib.sha256(coco_bytes).hexdigest(),
                    'rgb_sha256': sha(row['rgb_path']), 'prior_sha256': sha(row['relative_prior_path'])}
            tile_reports.append(tile)
            for annotation in coco['annotations']:
                attr = annotation.get('attributes') or {}
                height = attr.get('height')
                if attr.get('is_estimated_height') or not isinstance(height,(float,int)) or not np.isfinite(height) or height <= 0:
                    continue
                city['annotations'].update(height_bands([height]))
                values = annotation_values(annotation, target, valid)
                instances.append({'sample_id': sample, 'region': row['region'], 'annotation_id': annotation['id'],
                                  'height_m': height, 'estimated': False, 'support_pixels': int(values.size),
                                  'raster_median_m': float(np.median(values)) if values.size else None,
                                  'fraction_within_0_1m_annotation': float((np.abs(values-height) <= .1).mean()) if values.size else None})
            if (index+1) % 50 == 0:
                print(f'Train-only height audit: {index+1}/{len(rows)} tiles', flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    totals = Counter()
    for city in cities.values():
        totals.update(city['pixels'])
    report = {
        'schema': 'msr.train_only_height_support.v1', 'manifest': str(MANIFEST), 'manifest_sha256': EXPECTED,
        'split': 'train', 'heldout_images_read': False, 'gpu_used': False, 'training_started': False,
        'tiles': len(rows), 'measured_annotations': len(instances), 'pixel_bands': dict(totals),
        'cities': dict(cities),
        'mask_mismatch_tiles': [t['sample_id'] for t in tile_reports if not t['measured_support_matches'] or not t['semantic_support_matches']],
        'prior_numerical_or_grid_failures': [t['sample_id'] for t in tile_reports if not all(t['prior'][k] for k in ('stored_01','grid_aligned'))],
        'prior_without_source_weights_binding': sum(not t['prior']['source_and_weights_hash_tags_present'] for t in tile_reports),
        'missing_gsd_tiles': sum(t['gsd_m'] is None for t in tile_reports),
        'height_over_200_pixels': sum(t['height_over_200_pixels'] for t in tile_reports),
        'annotation_raster_mismatch_instances': sum(i['fraction_within_0_1m_annotation'] is not None and i['fraction_within_0_1m_annotation'] < .95 for i in instances),
        'limitations': ['Measured flag is annotation provenance, not an independent field survey verification.',
                       'Shape/range/hash audit does not establish visual registration or cached-prior source/model correctness.',
                       'Overlapping COCO footprints can cause raster/annotation differences; flagged instances need review.',
                       'No new validation, test or reserved external geography consumed. No automatic label removal.'],
        'tiles_detail': tile_reports,
    }
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'measured_instances.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(instances[0]))
        writer.writeheader(); writer.writerows(instances)
    print(json.dumps({k:report[k] for k in ['tiles','measured_annotations','pixel_bands','mask_mismatch_tiles','prior_numerical_or_grid_failures','prior_without_source_weights_binding','missing_gsd_tiles','height_over_200_pixels','annotation_raster_mismatch_instances']},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    # Missing georeferencing is explicitly counted in the report.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        audit(parser.parse_args().output.resolve())
