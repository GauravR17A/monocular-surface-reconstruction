"""Authenticated paired TRAIN-only feasibility proof; maximum 64 updates/arm.

No production integration, final-test reads, acceptance-threshold relaxation or
automatic retry. Inspect failure.json before recovery. Checkpoints are forensic
proof state, not releasable models or initialization for a development run.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import torch
from torch.utils.data import default_collate
import transformers

from msr.data.surface_dataset import load_surface_manifest, MultiDomainSurfaceDataset
from msr.inference.predict import load_predictor
from msr.inference.relative_depth import DepthAnythingV2Predictor
from msr.io.raster import read_rgb_raster
from msr.models.class_assisted_height import ClassAssistedHeightNet
from msr.models.rgb_segmenter import RgbSegformer
from msr.training.balanced_height_loss import SourceBalancedHeightLoss

ROOT=Path(__file__).resolve().parents[1]
SOURCES=(
    'scripts/class_assisted_height_proof.py',
    'src/msr/models/class_assisted_height.py',
    'src/msr/models/residual_height.py',
    'src/msr/models/rgb_segmenter.py',
    'src/msr/models/domain_surface_net.py',
    'src/msr/models/height_net.py',
    'src/msr/data/surface_dataset.py',
    'src/msr/data/highbuild.py',
    'src/msr/data/radiometry.py',
    'src/msr/data/raster_dataset.py',
    'src/msr/inference/relative_depth.py',
    'src/msr/inference/predict.py',
    'src/msr/io/raster.py',
    'src/msr/training/balanced_height_loss.py',
)
SEMANTIC_KEYS=('domain_logits','domain_probabilities','building_logits','protected_building_logits','vegetation_logits','fine_semantic_logits')


def resolve(path):
    p=Path(path)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def state_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def replace(source,dest):
    for attempt in range(20):
        try: os.replace(source,dest); return
        except PermissionError:
            if attempt==19: raise
            time.sleep(.1)


def atomic_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    replace(temporary,path)


def atomic_save(path,data):
    temporary=path.with_suffix('.pt.tmp'); torch.save(data,temporary); replace(temporary,path)


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def authenticate(config):
    for key in ['train_manifest','protected_checkpoint','protected_pointer','classifier_checkpoint']:
        if sha(resolve(config[key]))!=config[key+'_sha256']: raise ValueError(f'Changed binding: {key}')
    if resolve(resolve(config['protected_pointer']).read_text(encoding='utf-8-sig').strip())!=resolve(config['protected_checkpoint']):
        raise ValueError('Live pointer no longer matches protected model')
    for name,digest in config['dav2_files_sha256'].items():
        if sha(resolve(config['dav2_directory'])/name)!=digest: raise ValueError(f'DAV2 changed: {name}')
    if any(config[k] is not False for k in ['validation_used','test_used','automatic_promotion','proof_weights_reused']):
        raise ValueError('Proof scope changed')
    t=config['training']
    if t['optimizer_updates']!=64 or t['batch_size']!=2 or t['gradient_accumulation']!=2 or t['augment']:
        raise ValueError('Bounded fixed proof recipe changed')
    if Path(config['train_manifest']).name!='train.csv': raise ValueError('Train-only manifest required')
    return {path:sha(ROOT/path) for path in SOURCES}


def status(run,stage,**values):
    atomic_json(run/'status.json',{'stage':stage,'updated_utc':datetime.now(timezone.utc).isoformat(),**values})


def sample_support(record):
    with rasterio.open(record.surface_path) as f: target=f.read(1)
    with rasterio.open(record.valid_mask_path) as f: valid=f.read(1)>0
    valid &= np.isfinite(target)&(target>0)&(target<=200)
    return target,valid


def select_records(records):
    cities=defaultdict(list)
    forest=[]
    for record in records:
        if record.landscape=='urban': cities[record.region].append(record)
        elif record.landscape=='forest': forest.append(record)
        else: raise ValueError('Unexpected training landscape')
    if len(cities)!=5: raise ValueError('Expected five HighBuild training cities')
    urban=[]
    for city,items in sorted(cities.items()):
        chosen={}
        for record in sorted(items,key=lambda r:r.sample_id):
            target,valid=sample_support(record)
            for band,mask in [('short',valid&(target<=20)),('tall',valid&(target>20))]:
                if band not in chosen and mask.sum()>=64:
                    # Prefer different images so the proof covers multiple scenes.
                    if any(r.sample_id==record.sample_id for r,_ in chosen.values()): continue
                    chosen[band]=(record,band)
            if len(chosen)==2: break
        if len(chosen)!=2: raise ValueError(f'No supported tall/short pair in {city}')
        urban.extend(chosen[b] for b in ['short','tall'])
    selected_forest=[]; seen=set()
    for record in sorted(forest,key=lambda r:(r.region,r.sample_id)):
        if record.region in seen: continue
        sample=full_sample(record)
        support=sample['regression_mask'][0]
        if all(int((support&(sample['domain_target']==code)).sum())>=64 for code in [0,2]):
            selected_forest.append((record,'forest')); seen.add(record.region)
        if len(selected_forest)==10: break
    if len(selected_forest)!=10: raise ValueError('Ten forest training groups with ground/vegetation required')
    return [item for pair in zip(urban,selected_forest) for item in pair]


def full_sample(record):
    with rasterio.open(record.rgb_path) as f:
        if f.height!=f.width: raise ValueError('Proof expects native square chips')
        size=f.width
    dataset=MultiDomainSurfaceDataset([record],patch_size=size,random_crop=False,augment=False,
        radiometric_policy='raw',relative_prior_policy='stored_01',height_max_m=200)
    return {**dataset[0],'source':'legacy'}


def crop_sample(sample,band,patch,seed_value):
    height,width=sample['image'].shape[-2:]
    if min(height,width)<patch: raise ValueError('Native image smaller than proof crop')
    mask=sample['regression_mask'][0].clone()
    if band=='tall': mask &= sample['height'][0]>20
    elif band=='short': mask &= (sample['height'][0]>0)&(sample['height'][0]<=20)
    coordinates=torch.nonzero(mask)
    if len(coordinates)==0: raise ValueError('Selected crop lacks supervised support')
    point=coordinates[int(np.random.default_rng(seed_value).integers(len(coordinates)))]
    row=max(0,min(int(point[0])-patch//2,height-patch)); col=max(0,min(int(point[1])-patch//2,width-patch))
    cropped={k:(v[...,row:row+patch,col:col+patch].contiguous() if isinstance(v,torch.Tensor) else v) for k,v in sample.items()}
    return cropped,{'row':row,'col':col,'size':patch,'selection_band':band,'seed':seed_value}


def amp(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()


def prepare(config,run,device):
    records=load_surface_manifest(resolve(config['train_manifest']))
    if len(records)!=1050: raise ValueError('Expected pinned 1050-row training manifest')
    selected=select_records(records)
    status(run,'prior_replay',completed=0,total=len(selected))
    prior_model=DepthAnythingV2Predictor(model_id=str(resolve(config['dav2_directory'])),device=device)
    bindings=[]
    for index,(record,band) in enumerate(selected):
        source=read_rgb_raster(record.rgb_path)
        with rasterio.open(record.relative_prior_path) as f: cached=f.read(1)
        fresh=prior_model.predict(source.rgb,valid_mask=source.valid_mask).relative_surface
        if fresh.shape!=cached.shape: raise ValueError('Cached prior shape mismatch')
        valid=source.valid_mask
        if not np.isfinite(cached[valid]).all(): raise ValueError('Nonfinite prior')
        delta=np.abs(cached[valid]-fresh[valid])
        replay={'max_abs':float(delta.max()),'mean_abs':float(delta.mean())}
        paths={key:str(getattr(record,key)) for key in ['rgb_path','surface_path','relative_prior_path','building_mask_path','vegetation_mask_path','valid_mask_path','dtm_path'] if getattr(record,key) is not None}
        binding={'sample_id':record.sample_id,'region':record.region,'band':band,'paths':paths,
                 'sha256':{key:sha(path) for key,path in paths.items()},'prior_replay':replay}
        bindings.append(binding)
        atomic_json(run/'preparation.json',{'samples':bindings})
        if replay['max_abs']>config['prior_replay']['maximum_abs_difference'] or replay['mean_abs']>config['prior_replay']['mean_abs_difference']:
            raise ValueError(f'Prior replay failed for {record.sample_id}: {replay}')
        status(run,'prior_replay',completed=index+1,total=len(selected))
    del prior_model; gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    payload=torch.load(resolve(config['classifier_checkpoint']),map_location='cpu',weights_only=False)
    if payload.get('model_type')!=RgbSegformer.model_type or payload.get('input_contract')!='raw_rgb_01_imagenet_normalized_in_model':
        raise ValueError('Classifier contract mismatch')
    classifier=RgbSegformer.from_architecture(payload['architecture'])
    classifier.load_state_dict(payload['model_state_dict'],strict=True)
    classifier.to(device).eval().requires_grad_(False)
    del payload
    (run/'cache').mkdir()
    for index,((record,band),binding) in enumerate(zip(selected,bindings)):
        sample=full_sample(record)
        # Exactly undo the existing height dataset's image-only normalization.
        mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        raw=(sample['image'][None].to(device)*std+mean).clamp(0,1)
        with torch.inference_mode(),amp(device): probabilities=classifier(raw)['logits'].softmax(1)[0].cpu().float()
        sample['class_probabilities']=probabilities
        sample,crop=crop_sample(sample,band,config['training']['patch_size'],config['seed']+index)
        name=f'{index:03}.pt'; atomic_save(run/'cache'/name,sample)
        binding.update({'cache':name,'cache_sha256':sha(run/'cache'/name),'crop':crop})
        status(run,'classification_cache',completed=index+1,total=len(selected))
    del classifier; gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    atomic_json(run/'cache_manifest.json',{'samples':bindings,'model_sha256':config['classifier_checkpoint_sha256']})
    return bindings


def batches(samples,device):
    for i in range(0,len(samples),2):
        batch=default_collate(samples[i:i+2])
        yield {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def forward(model,batch,arm):
    return model(batch['image'],batch['relative_prior'],probabilities=batch['class_probabilities'],image_valid=batch['image_valid_mask'],arm=arm)


@torch.no_grad()
def assess(model,samples,criterion,device,arm,*,require_zero=False):
    model.eval(); losses=[]; components=defaultdict(list)
    for batch in batches(samples,device):
        with amp(device):
            output=forward(model,batch,arm)
            baseline=model.protected(batch['image'],batch['relative_prior'])
            loss,parts=criterion(output,batch)
        if require_zero and not torch.equal(output['height'],baseline['height']): raise ValueError('Zero-init height identity failed')
        for key in SEMANTIC_KEYS:
            if key in baseline and not torch.equal(output[key],baseline[key]): raise ValueError(f'Protected semantic changed: {key}')
        losses.append(float(loss))
        for key,value in parts.items(): components[key].append(float(value))
    return {'loss':float(np.mean(losses)),'components':{k:float(np.mean(v)) for k,v in components.items()},'semantic_identity':True}


def train_arm(config,run,samples,device,arm):
    t=config['training']; seed(config['seed'])
    protected,_=load_predictor(resolve(config['protected_checkpoint']),device='cpu')
    model=ClassAssistedHeightNet(protected,hidden_channels=t['hidden_channels'],maximum_correction_m=t['maximum_correction_m']).to(device)
    frozen_digest=state_digest(model.protected.state_dict())
    initial_digest=state_digest(model.state_dict())
    optimizer=torch.optim.AdamW([
        {'params':list(model.deepest.parameters())+list(model.decoder_stages.parameters()),'lr':t['decoder_learning_rate']},
        {'params':list(model.spatial_correction.parameters())+list(model.correction_head.parameters()),'lr':t['learning_rate']}
    ],weight_decay=t['weight_decay'])
    criterion=SourceBalancedHeightLoss(huber_delta_m=t['huber_delta_m'],mse_weight=t['mse_weight'])
    initial=assess(model,samples,criterion,device,arm,require_zero=True)
    model.train(); history=[]; sequence=[]; max_decoder_gradient=0.; start=time.monotonic()
    cpu_batches=list(batches(samples,torch.device('cpu')))
    for update in range(1,t['optimizer_updates']+1):
        optimizer.zero_grad(set_to_none=True); loss_sum=0.
        for accumulation in range(t['gradient_accumulation']):
            index=((update-1)*t['gradient_accumulation']+accumulation)%len(cpu_batches)
            batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in cpu_batches[index].items()}
            sequence.extend(batch['sample_id'])
            with amp(device): output=forward(model,batch,arm); loss,_=criterion(output,batch)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite proof loss')
            (loss/t['gradient_accumulation']).backward(); loss_sum+=float(loss.detach())/t['gradient_accumulation']
        trainable=[p for p in model.parameters() if p.requires_grad]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable): raise FloatingPointError('Nonfinite gradient')
        gradients=[float(p.grad.detach().abs().max()) for p in model.deepest.parameters() if p.grad is not None]
        max_decoder_gradient=max(max_decoder_gradient,max(gradients,default=0.))
        torch.nn.utils.clip_grad_norm_(trainable,t['max_gradient_norm']); optimizer.step()
        history.append(loss_sum)
        if update%8==0 or update==1:
            status(run,'training_proof',arm=arm,update=update,total=t['optimizer_updates'],loss=loss_sum)
            print(f'{arm}: {update}/64 loss={loss_sum:.4f}',flush=True)
            atomic_save(run/f'{arm}_proof_latest.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'update':update,'arm':arm,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                'config':config,'proof_only':True,'initial_model_sha256':initial_digest,'data_sequence':sequence,'losses':history})
    final=assess(model,samples,criterion,device,arm)
    frozen_ok=state_digest(model.protected.state_dict())==frozen_digest and all(p.grad is None for p in model.protected.parameters())
    reduction=(initial['loss']-final['loss'])/initial['loss'] if initial['loss']>0 else 0.
    result={'arm':arm,'updates':t['optimizer_updates'],'initial':initial,'final':final,'loss_reduction':reduction,
            'initial_model_sha256':initial_digest,'sequence_sha256':hashlib.sha256(json.dumps(sequence).encode()).hexdigest(),
            'frozen_protected_unchanged':frozen_ok,'maximum_decoder_gradient':max_decoder_gradient,'elapsed_seconds':time.monotonic()-start,
            'maximum_allocated_vram_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
            'passes':bool(frozen_ok and max_decoder_gradient>0 and math.isfinite(reduction) and reduction>=t['minimum_loss_reduction'])}
    atomic_json(run/f'{arm}_result.json',result)
    del model,protected,optimizer; gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/class_assisted_height_proof_v1.json')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args(); config=json.loads(args.config.read_text(encoding='utf-8'))
    source_binding=authenticate(config)
    run=resolve(config['output_root'])/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True,exist_ok=False)
    atomic_json(run/'config.json',config)
    atomic_json(run/'binding.json',{'source_sha256':source_binding,'config_sha256':sha(args.config),'torch':torch.__version__,
                                  'transformers':transformers.__version__,'python':sys.version,'process_id':os.getpid()})
    for path in SOURCES:
        dest=run/'source_snapshot'/path; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/path,dest)
    registration=ROOT/'outputs/orchestration/class_assisted_height_proof_active.json'
    atomic_json(registration,{'run_dir':str(run),'process_id':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat()})
    print(f'RUN_DIR={run}',flush=True)
    device=torch.device(args.device)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',NotGeoreferencedWarning)
            bindings=prepare(config,run,device)
        samples=[]
        for item in bindings:
            path=run/'cache'/item['cache']
            if sha(path)!=item['cache_sha256']: raise ValueError('Cache binding changed')
            samples.append(torch.load(path,map_location='cpu',weights_only=True))
        results={arm:train_arm(config,run,samples,device,arm) for arm in ['uniform','predicted']}
        paired=all(results['uniform'][key]==results['predicted'][key] for key in ['initial_model_sha256','sequence_sha256'])
        if authenticate(config)!=source_binding: raise ValueError('Source changed during proof')
        report={'schema':'msr.class_assisted_height_proof.v1','training_only':True,'validation_used':False,'test_used':False,
                'app_unchanged':True,'proof_weights_reused':False,'paired_identity':paired,'arms':results,
                'passes':paired and all(r['passes'] for r in results.values()),
                'interpretation':'Training-only optimization feasibility. Not height accuracy, generalization or release approval.'}
        atomic_json(run/'outcome.json',report); status(run,'complete',passes=report['passes'])
        print(json.dumps(report,indent=2),flush=True)
    except BaseException as error:
        atomic_json(run/'failure.json',{'type':type(error).__name__,'message':str(error),'no_automatic_retry':True})
        status(run,'failed',error=str(error)); raise


if __name__=='__main__': main()
