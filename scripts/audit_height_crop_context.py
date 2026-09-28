"""Train-only crop/full context diagnostic on the original fixed 20 proof examples.

Not a validation score, new training run, or inference-policy selection sweep.
Identical physical pixels and source-bound priors; only available context changes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from rasterio.errors import NotGeoreferencedWarning
from msr.evaluation.metrics import StreamingRegressionMetrics

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('crop_proof_common',ROOT/'scripts/class_assisted_height_proof.py')
common=importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def masks_for_context(sample, margin=32):
    support=sample['regression_mask'][0].numpy().astype(bool)
    height=sample['height'][0].numpy()
    domain=sample['domain_target'].numpy()
    interior=np.zeros_like(support)
    if min(support.shape)>2*margin:
        interior[margin:-margin,margin:-margin]=True
    return {name+suffix:mask & (interior if suffix else True)
            for name,mask in {'building':support&(domain==1), 'ground':support&(domain==0),
                              'vegetation':support&(domain==2),
                              'tall_building':support&(domain==1)&(height>=20)}.items()
            for suffix in ('','/interior')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/diagnostics/height_crop_context_train_v1.json')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Diagnostic already exists; do not overwrite')
    config=json.loads((ROOT/'configs/class_assisted_height_proof_v1.json').read_text())
    code=common.authenticate(config)
    development=json.loads((ROOT/'configs/class_assisted_height_development_v1.json').read_text())
    outcome_path=common.resolve(development['proof_outcome'])
    if common.sha(outcome_path) != development['proof_outcome_sha256']:
        raise ValueError('Proof evidence changed')
    proof_run=outcome_path.parent
    preparation=json.loads((proof_run/'cache_manifest.json').read_text())
    items=preparation['samples']
    if len(items)!=20:
        raise ValueError('Expected original 20 train-only examples')
    records={r.sample_id:r for r in common.load_surface_manifest(common.resolve(config['train_manifest']))}
    torch.set_num_threads(1)
    device=torch.device('cuda')
    model,_=common.load_predictor(common.resolve(config['protected_checkpoint']),device=device)
    model.eval()
    acc={}
    cases=[]
    with torch.inference_mode():
        for index,item in enumerate(items):
            record=records[item['sample_id']]
            for name,path in item['paths'].items():
                if common.sha(path)!=item['sha256'][name]:
                    raise ValueError(f'Proof source changed: {record.sample_id}/{name}')
            sample=common.full_sample(record)
            cropped,crop=common.crop_sample(sample,item['band'],384,config['seed']+index)
            if crop!=item['crop']:
                raise ValueError('Fixed proof crop changed')
            with common.amp(device):
                whole=model(sample['image'][None].to(device),sample['relative_prior'][None].to(device))['height'][0,0].float().cpu().numpy()
                local=model(cropped['image'][None].to(device),cropped['relative_prior'][None].to(device))['height'][0,0].float().cpu().numpy()
            row,col,size=crop['row'],crop['col'],crop['size']
            native=whole[row:row+size,col:col+size]
            target=cropped['height'][0].numpy()
            row_result={'sample_id':record.sample_id,'landscape':record.landscape,'crop':crop,
                        'full_shape':list(whole.shape),'max_abs_height_change_m':float(np.abs(local-native).max()),'groups':{}}
            for name,mask in masks_for_context(cropped).items():
                n=int(mask.sum())
                if not n:
                    continue
                key=record.landscape+'/'+name
                if key not in acc:
                    acc[key]={'native_context':StreamingRegressionMetrics(),'cropped_context':StreamingRegressionMetrics(),
                              'sum_absolute_change':0.,'sum_signed_change':0.,'pixels':0}
                entry=acc[key]
                entry['native_context'].update(native,target,mask)
                entry['cropped_context'].update(local,target,mask)
                change=local[mask]-native[mask]
                entry['sum_absolute_change']+=float(np.abs(change).sum(dtype=np.float64))
                entry['sum_signed_change']+=float(change.sum(dtype=np.float64))
                entry['pixels']+=n
                row_result['groups'][name]={'pixels':n,'mean_abs_change_m':float(np.abs(change).mean()),'mean_signed_change_m':float(change.mean())}
            cases.append(row_result)
    if common.authenticate(config)!=code:
        raise ValueError('Protected sources changed during diagnostic')
    groups={key:{'native_context':value['native_context'].compute(), 'cropped_context':value['cropped_context'].compute(),
                 'mean_absolute_context_change_m':value['sum_absolute_change']/value['pixels'],
                 'mean_signed_context_change_m':value['sum_signed_change']/value['pixels']} for key,value in acc.items()}
    report={'schema':'msr.train_only_context.v1','images':len(items),'training_only':True,
            'validation_or_final_data_read':False,'model_changed':False,'optimizer_steps':0,
            'protected_checkpoint_sha256':config['protected_checkpoint_sha256'],
            'proof_cache_manifest_sha256':common.sha(proof_run/'cache_manifest.json'),'script_sha256':common.sha(Path(__file__)),
            'groups':groups,'cases':cases,
            'limitations':['Mechanically preselected feasibility examples are not population validation.',
                           'Full/crop context effects do not establish the cause of all height error.',
                           'No inference policy change or training promotion is performed.']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    common.atomic_json(args.output,report)
    print(json.dumps(groups,indent=2))


if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',NotGeoreferencedWarning)
        main()
