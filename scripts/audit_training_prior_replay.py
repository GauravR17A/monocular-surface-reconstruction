"""Replay ten fixed TRAIN priors without replacing caches or training a model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
import transformers

from msr.inference.relative_depth import DepthAnythingV2Predictor
from msr.io.raster import read_rgb_raster

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=Path('D:/MSRData/data/multidomain_v2/highbuild_measured_coco_intersection_v1/manifests/train.csv')
EXPECTED='8f7f85997df53c2dae82d571c52db4ca8f10d4cb5638784e729b38632a1067e4'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def select_train_records(rows):
    groups=defaultdict(list)
    for row in rows:
        if row['target_kind']=='building_height': groups[row['region']].append(row)
    selected=[]
    for city in sorted(groups):
        # Independent of errors, predictions and validation performance.
        selected.extend(sorted(groups[city],key=lambda r:r['sample_id'])[:2])
    return selected


def compare_prior(cached,fresh,valid):
    if cached.shape != fresh.shape or valid.shape != fresh.shape: raise ValueError('Prior grid mismatch')
    usable=valid & np.isfinite(cached) & np.isfinite(fresh)
    if not usable.any(): raise ValueError('No shared valid prior pixels')
    delta=np.abs(cached[usable]-fresh[usable])
    correlation=float(np.corrcoef(cached[usable],fresh[usable])[0,1]) if np.std(cached[usable])>0 and np.std(fresh[usable])>0 else None
    # Numerical reproducibility criterion, not a height-accuracy gate.
    return {'valid_pixels':int(usable.sum()),'max_abs_difference':float(delta.max()),'mean_abs_difference':float(delta.mean()),
            'correlation':correlation,'replay_pass':bool(delta.max()<=.005 and delta.mean()<=.0005)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if sha(MANIFEST)!=EXPECTED: raise ValueError('Pinned TRAIN manifest changed')
    if args.output.exists(): raise FileExistsError(args.output)
    model_root=ROOT/'models/foundation/depth-anything-v2-small-hf'
    binding={name:sha(model_root/name) for name in ['config.json','preprocessor_config.json','model.safetensors']}
    with MANIFEST.open(encoding='utf-8-sig',newline='') as stream: rows=select_train_records(list(csv.DictReader(stream)))
    if len(rows)!=10: raise ValueError('Expected ten fixed training images')
    args.output.mkdir(parents=True)
    predictor=DepthAnythingV2Predictor(model_id=str(model_root),device=args.device)
    results=[]
    for index,row in enumerate(rows):
        source=read_rgb_raster(row['rgb_path'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',NotGeoreferencedWarning)
            with rasterio.open(row['relative_prior_path']) as f: cached=f.read(1)
        fresh=predictor.predict(source.rgb,valid_mask=source.valid_mask).relative_surface
        results.append({'sample_id':row['sample_id'],'region':row['region'],'rgb_sha256':sha(row['rgb_path']),
                        'cached_prior_sha256':sha(row['relative_prior_path']),**compare_prior(cached,fresh,source.valid_mask)})
        print(f"Prior replay {index+1}/10: {row['sample_id']} max={results[-1]['max_abs_difference']:.6f} pass={results[-1]['replay_pass']}",flush=True)
    report={'schema':'msr.training_prior_replay.v1','manifest_sha256':EXPECTED,'split':'train','selection':'first two sample IDs per training city',
            'device':args.device,'model_files_sha256':binding,'torch':torch.__version__,'transformers':transformers.__version__,
            'numeric_gate':{'max_abs_difference':.005,'mean_abs_difference':.0005},'results':results,
            'all_sampled_replays_pass':all(r['replay_pass'] for r in results),
            'limitations':['Ten training examples only; not a complete 450-tile provenance certification.',
                           'Reproducibility is not height accuracy or independent alignment verification.',
                           'No cached raster, model, app pointer or label was changed.']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__': main()
