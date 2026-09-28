"""Explain large TRAIN annotation/raster disagreements without changing labels."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import warnings

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from rasterio.errors import NotGeoreferencedWarning
from affine import Affine
from msr.data.highbuild import _annotation_polygons

MANIFEST=Path('D:/MSRData/data/multidomain_v2/highbuild_measured_coco_intersection_v1/manifests/train.csv')
SPLITS=Path('D:/MSRData/data/highbuild_full/msr_splits')
EXPECTED='8f7f85997df53c2dae82d571c52db4ca8f10d4cb5638784e729b38632a1067e4'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest()!=EXPECTED: raise ValueError('TRAIN manifest changed')
    if args.output.exists(): raise FileExistsError(args.output)
    with MANIFEST.open(encoding='utf-8-sig',newline='') as f: rows={r['sample_id']:r for r in csv.DictReader(f)}
    with (SPLITS/'train.csv').open(encoding='utf-8-sig',newline='') as f: metadata={r['webdataset_key']:r for r in csv.DictReader(f)}
    with (args.audit/'measured_instances.csv').open(encoding='utf-8-sig',newline='') as f:
        suspects=[r for r in csv.DictReader(f) if r['raster_median_m'] and abs(float(r['height_m'])-float(r['raster_median_m']))>1]
    results=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',NotGeoreferencedWarning)
        for suspect in suspects:
            row=rows[suspect['sample_id']]; meta=metadata[suspect['sample_id']]
            if row['target_kind']!='building_height' or meta['msr_split']!='train': raise ValueError('Non-training instance')
            with tarfile.open(SPLITS/meta['msr_shard'],'r') as archive:
                coco=json.load(archive.extractfile(meta['webdataset_json_member']))
            with rasterio.open(row['surface_path']) as f: target=f.read(1)
            with rasterio.open(row['valid_mask_path']) as f: valid=f.read(1)>0
            shapes=[]; own=[]
            for ann in coco['annotations']:
                for polygon in _annotation_polygons(ann):
                    shapes.append((polygon,1))
                    if str(ann['id'])==suspect['annotation_id']: own.append((polygon,1))
            coverage=rasterize(shapes,out_shape=target.shape,transform=Affine.identity(),dtype='uint16',merge_alg=MergeAlg.add)
            support=(rasterize(own,out_shape=target.shape,transform=Affine.identity(),dtype='uint8')>0)&valid
            exclusive=support&(coverage==1)
            residual=np.abs(target[exclusive]-float(suspect['height_m']))
            results.append({**suspect,'overlapping_support_pixels':int((support&(coverage>1)).sum()),
                            'exclusive_support_pixels':int(exclusive.sum()),
                            'exclusive_fraction_within_0_1m':float((residual<=.1).mean()) if residual.size else None,
                            'exclusive_median_abs_difference_m':float(np.median(residual)) if residual.size else None})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'split':'train','manifest_sha256':EXPECTED,'instances':results,'labels_changed':False,
            'interpretation':'Exclusive support excludes polygon overlaps, not proof of real-world annotation accuracy. Thin edge differences may come from rasterization conventions.'}
    with args.output.open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
