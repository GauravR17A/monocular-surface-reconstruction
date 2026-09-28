"""Capped, resumable, paired HighBuild/OpenCanopy DEVELOPMENT experiment.

No app promotion or reserved-geography access. Cache receipts bind original
files and class probability grids. Versioned step checkpoints plus atomic commit
are authoritative; resume refuses any source/config/software binding mismatch.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gc
import hashlib
import importlib.util
import json
import math
import msvcrt
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
from torch.nn import functional as F
from torch.utils.data import default_collate
import transformers

from msr.evaluation.metrics import StreamingRegressionMetrics
from msr.evaluation.class_assisted_gate import safety_checks,paired_benefit

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('height_proof_common',ROOT/'scripts/class_assisted_height_proof.py')
common=importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
EXTRA_SOURCES=('scripts/train_class_assisted_height.py','src/msr/evaluation/class_assisted_gate.py','src/msr/evaluation/metrics.py')


def binding(config_path,config,proof):
    code=common.authenticate(proof)
    code.update({p:common.sha(ROOT/p) for p in EXTRA_SOURCES})
    if common.sha(common.resolve(config['proof_outcome']))!=config['proof_outcome_sha256']: raise ValueError('Proof outcome changed')
    outcome=json.loads(common.resolve(config['proof_outcome']).read_text())
    if outcome.get('passes') is not True: raise ValueError('Paired proof did not pass')
    proof_binding=json.loads((common.resolve(config['proof_outcome']).parent/'binding.json').read_text())
    if proof_binding['source_sha256']!={p:code[p] for p in common.SOURCES}: raise ValueError('Proof/model source binding changed')
    if proof_binding['config_sha256']!=common.sha(common.resolve(config['proof_config'])): raise ValueError('Proof recipe changed')
    if common.sha(common.resolve(config['validation_manifest']))!=config['validation_manifest_sha256']: raise ValueError('Development manifest changed')
    if config['epochs']!=2 or config['pairs_per_epoch']!=600 or config['automatic_promotion'] or config['test_used']:
        raise ValueError('Bounded development scope changed')
    return {'code':code,'config_sha256':common.sha(config_path),'proof_config_sha256':common.sha(common.resolve(config['proof_config'])),
            'torch':torch.__version__,'transformers':transformers.__version__,'python':sys.version,'device':'cuda'}


def receipts_file(run,split,sample_id):
    key=hashlib.sha256(sample_id.encode()).hexdigest()
    return run/'cache'/split/(key+'.json')


def paths_binding(record):
    paths={k:str(getattr(record,k)) for k in ['rgb_path','surface_path','relative_prior_path','building_mask_path','vegetation_mask_path','valid_mask_path','dtm_path'] if getattr(record,k) is not None}
    return {'paths':paths,'sha256':{k:common.sha(p) for k,p in paths.items()}}


def verify_receipt(receipt_path,record,*,files=False):
    receipt=json.loads(receipt_path.read_text())
    if receipt['sample_id']!=record.sample_id: raise ValueError('Cache sample mismatch')
    if common.sha(receipt_path.with_suffix('.npz'))!=receipt['cache_sha256']: raise ValueError('Probability cache corrupted')
    if files and paths_binding(record)!={k:receipt[k] for k in ['paths','sha256']}:
        raise ValueError(f'Input binding changed: {record.sample_id}')
    return receipt


def prepare_cache(config,proof,run,sets,device):
    existing=[]; missing=[]
    for split,records in sets.items():
        (run/'cache'/split).mkdir(parents=True,exist_ok=True)
        for record in records:
            path=receipts_file(run,split,record.sample_id)
            if path.exists(): existing.append(verify_receipt(path,record,files=True))
            else: missing.append((split,record,path))
    if missing:
        dav2=common.DepthAnythingV2Predictor(model_id=str(common.resolve(proof['dav2_directory'])),device=device)
        payload=torch.load(common.resolve(proof['classifier_checkpoint']),map_location='cpu',weights_only=False)
        classifier=common.RgbSegformer.from_architecture(payload['architecture'])
        classifier.load_state_dict(payload['model_state_dict'],strict=True)
        classifier.to(device).eval().requires_grad_(False); del payload
        for index,(split,record,path) in enumerate(missing):
            before=paths_binding(record)
            source=common.read_rgb_raster(record.rgb_path)
            with rasterio.open(record.relative_prior_path) as f: cached=f.read(1)
            fresh=dav2.predict(source.rgb,valid_mask=source.valid_mask).relative_surface
            if fresh.shape!=cached.shape or not np.isfinite(cached[source.valid_mask]).all(): raise ValueError('Invalid prior grid/values')
            delta=np.abs(cached[source.valid_mask]-fresh[source.valid_mask])
            replay={'max_abs':float(delta.max()),'mean_abs':float(delta.mean())}
            if not all(math.isfinite(v) for v in replay.values()) or replay['max_abs']>proof['prior_replay']['maximum_abs_difference'] or replay['mean_abs']>proof['prior_replay']['mean_abs_difference']:
                raise ValueError(f'Prior replay mismatch: {record.sample_id}: {replay}')
            sample=common.full_sample(record)
            mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
            std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
            raw=(sample['image'][None].to(device)*std+mean).clamp(0,1)
            with torch.inference_mode(),common.amp(device): probs=classifier(raw)['logits'].softmax(1)[0].cpu().numpy().astype(np.float16)
            temporary=path.with_suffix('.npz.tmp')
            with temporary.open('wb') as f: np.savez_compressed(f,probabilities=probs)
            common.replace(temporary,path.with_suffix('.npz'))
            if paths_binding(record)!=before: raise ValueError('Input changed during cache generation')
            receipt={**before,'sample_id':record.sample_id,'split':split,'prior_replay':replay,'shape':list(probs.shape),
                     'cache_sha256':common.sha(path.with_suffix('.npz')),'classifier_sha256':proof['classifier_checkpoint_sha256']}
            common.atomic_json(path,receipt)
            if index%10==0 or index==len(missing)-1:
                done=len(existing)+index+1; total=len(existing)+len(missing)
                common.status(run,'preparing_verified_inputs',completed=done,total=total,split=split)
                print(f'Inputs verified: {done}/{total}',flush=True)
        del classifier,dav2; gc.collect(); torch.cuda.empty_cache()
    # Content-level leakage check, not just different filenames.
    hashes={split:{json.loads(receipts_file(run,split,r.sample_id).read_text())['sha256']['rgb_path'] for r in records} for split,records in sets.items()}
    if hashes['train'] & hashes['validation']: raise ValueError('Exact RGB overlap between train and development validation')
    receipt_hashes={split:{r.sample_id:common.sha(receipts_file(run,split,r.sample_id)) for r in records} for split,records in sets.items()}
    manifest={'receipts':receipt_hashes,'counts':{s:len(r) for s,r in sets.items()},'exact_rgb_overlap':0}
    target=run/'data_binding.json'
    if target.exists() and json.loads(target.read_text())!=manifest: raise ValueError('Data receipt binding changed')
    common.atomic_json(target,manifest)


def cached_sample(run,split,record):
    sample=common.full_sample(record)
    path=receipts_file(run,split,record.sample_id)
    receipt=verify_receipt(path,record)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as f: probabilities=f['probabilities'].astype(np.float32)
    # Remove float16 rounding from the probability simplex identically in both arms.
    probabilities/=np.maximum(probabilities.sum(axis=0,keepdims=True),1e-12)
    if list(probabilities.shape)!=receipt['shape'] or probabilities.shape[1:]!=tuple(sample['image'].shape[-2:]):
        raise ValueError('Cached classification grid mismatch')
    sample['class_probabilities']=torch.from_numpy(probabilities)
    return sample


def epoch_plan(records,epoch,seed_value):
    rng=np.random.default_rng(seed_value+epoch)
    urban=[r for r in records if r.landscape=='urban']; forest=[r for r in records if r.landscape=='forest']
    if len(urban)!=450 or len(forest)!=600: raise ValueError('Training source counts changed')
    u=list(rng.permutation(len(urban)))+list(rng.choice(len(urban),150,replace=False))
    rng.shuffle(u); f=list(rng.permutation(len(forest)))
    return [(urban[a],forest[b],seed_value+epoch*10000+i*2) for i,(a,b) in enumerate(zip(u,f))]


def train_batch(run,pair,patch,device):
    urban,forest,seed_value=pair
    samples=[common.crop_sample(cached_sample(run,'train',r),'urban' if r.landscape=='urban' else 'forest',patch,seed_value+i)[0] for i,r in enumerate([urban,forest])]
    batch=default_collate(samples)
    return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def add_groups(acc,prediction,target,valid,domain,suite,region,probabilities,legacy_domain):
    masks={suite+'/overall':valid,suite+'/region/'+region:valid}
    for name,code in [('ground',0),('building',1),('vegetation',2)]:
        mask=valid&(domain==code); masks[suite+'/domain/'+name]=mask
        masks[suite+'/region/'+region+'/'+name]=mask
        if name!='ground':
            masks[suite+'/tall/'+name]=mask&(target>=(20 if name=='building' else 15))
    masks[suite+'/short_vegetation']=valid&(domain==2)&(target>0)&(target<=2)
    coarse=np.array([0,1,0,0,2,2])[probabilities.argmax(0)]
    masks[suite+'/class_geometry_disagreement']=valid&(coarse!=legacy_domain)
    for code,name in [(1,'building'),(2,'vegetation')]:
        m=torch.from_numpy((valid&(domain==code)).astype(np.float32))[None,None]
        interior=(-F.max_pool2d(-m,3,1,1))[0,0].numpy()>0
        masks[suite+'/boundary/'+name]=valid&(domain==code)&~interior
    for key,mask in masks.items(): acc[key].update(prediction,target,mask)


@torch.no_grad()
def evaluate(model,run,records,device,arm,epoch,*,baseline=False):
    model.eval(); acc=defaultdict(StreamingRegressionMetrics); support=hashlib.sha256(); semantics=hashlib.sha256()
    for index,record in enumerate(records):
        sample=cached_sample(run,'validation',record)
        batch=default_collate([sample]); batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        with common.amp(device):
            out=model.protected(batch['image'],batch['relative_prior']) if baseline else common.forward(model,batch,arm)
        prediction=out['height'][0,0].float().cpu().numpy()
        if not np.isfinite(prediction).all(): raise FloatingPointError('Nonfinite development prediction')
        target=sample['height'][0].numpy(); valid=sample['regression_mask'][0].numpy(); domain=sample['domain_target'].numpy()
        support.update(record.sample_id.encode()); support.update(valid.tobytes()); support.update(target[valid].tobytes()); support.update(domain[valid].tobytes())
        for key in common.SEMANTIC_KEYS:
            if key in out:
                t=out[key].cpu().contiguous(); semantics.update(key.encode()); semantics.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        suite='highbuild' if record.landscape=='urban' else 'open_canopy'
        add_groups(acc,prediction,target,valid,domain,suite,record.region,sample['class_probabilities'].numpy(),out['domain_logits'].argmax(1)[0].cpu().numpy())
        if index%10==0 or index==len(records)-1:
            common.status(run,'baseline_validation' if baseline else 'development_validation',arm=arm,epoch=epoch,completed=index+1,total=len(records))
    return {'groups':{key:value.compute() for key,value in sorted(acc.items()) if value.count},
            'support_sha256':support.hexdigest(),'semantic_sha256':semantics.hexdigest(),'images':len(records),'development_only':True}


def save_commit(run,arm,model,optimizer,epoch,batch_done,stage,*,evaluation=None):
    directory=run/arm; directory.mkdir(exist_ok=True)
    name=f'epoch_{epoch:02}_batch_{batch_done:04}.pt'
    target=directory/name
    if target.exists() and stage=='trained':
        raise FileExistsError(f'Uncommitted checkpoint already exists; inspect before recovery: {target}')
    if not target.exists():
        common.atomic_save(target,{'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'batch_done':batch_done,
            'arm':arm,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
            'binding_sha256':common.sha(run/'binding.json'),'data_binding_sha256':common.sha(run/'data_binding.json'),'model_type':model.model_type,'release_eligible':False})
    commit={'checkpoint':name,'checkpoint_sha256':common.sha(target),'epoch':epoch,'batch_done':batch_done,'stage':stage}
    if evaluation is not None:
        path=directory/f'epoch_{epoch:02}_evaluation.json'; common.atomic_json(path,evaluation)
        commit.update({'evaluation':path.name,'evaluation_sha256':common.sha(path)})
    common.atomic_json(directory/'commit.json',commit)


def run_arm(config,proof,run,sets,device,arm,baseline):
    t=proof['training']; common.seed(proof['seed'])
    protected,_=common.load_predictor(common.resolve(proof['protected_checkpoint']),device='cpu')
    model=common.ClassAssistedHeightNet(protected,hidden_channels=t['hidden_channels'],maximum_correction_m=t['maximum_correction_m']).to(device)
    frozen=common.state_digest(protected.state_dict()); initial=common.state_digest(model.state_dict())
    optimizer=torch.optim.AdamW([
        {'params':list(model.deepest.parameters())+list(model.decoder_stages.parameters()),'lr':t['decoder_learning_rate']},
        {'params':list(model.spatial_correction.parameters())+list(model.correction_head.parameters()),'lr':t['learning_rate']}
    ],weight_decay=t['weight_decay'])
    directory=run/arm; directory.mkdir(exist_ok=True)
    initial_path=directory/'initial.json'
    if initial_path.exists() and json.loads(initial_path.read_text())['initial_model_sha256']!=initial: raise ValueError('Initialization changed')
    common.atomic_json(initial_path,{'initial_model_sha256':initial})
    start_epoch=1; start_batch=0
    if (directory/'commit.json').exists():
        commit=json.loads((directory/'commit.json').read_text()); path=directory/commit['checkpoint']
        if common.sha(path)!=commit['checkpoint_sha256']: raise ValueError('Committed checkpoint changed')
        payload=torch.load(path,map_location=device,weights_only=False)
        if payload['binding_sha256']!=common.sha(run/'binding.json') or payload['data_binding_sha256']!=common.sha(run/'data_binding.json') or payload['arm']!=arm:
            raise ValueError('Checkpoint binding changed')
        model.load_state_dict(payload['model_state_dict'],strict=True); optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['rng'].cpu()); torch.cuda.set_rng_state_all([x.cpu() for x in payload['cuda_rng']]); np.random.set_state(payload['numpy_rng']); random.setstate(payload['python_rng'])
        start_epoch=commit['epoch']; start_batch=commit['batch_done']
        if commit['stage']=='evaluated':
            if common.sha(directory/commit['evaluation'])!=commit['evaluation_sha256']: raise ValueError('Committed evaluation changed')
            start_epoch+=1; start_batch=0
        del payload
    if common.state_digest(protected.state_dict())!=frozen: raise ValueError('Frozen protected state changed on resume')
    criterion=common.SourceBalancedHeightLoss(huber_delta_m=t['huber_delta_m'],mse_weight=t['mse_weight'])
    for epoch in range(start_epoch,config['epochs']+1):
        plan=epoch_plan(sets['train'],epoch,proof['seed'])
        plan_document=[[a.sample_id,b.sample_id,s] for a,b,s in plan]
        common.atomic_json(directory/f'epoch_{epoch:02}_sampling.json',{'pairs':plan_document,'sha256':hashlib.sha256(json.dumps(plan_document).encode()).hexdigest()})
        model.train(); losses=[]
        for index in range(start_batch,len(plan),t['gradient_accumulation']):
            optimizer.zero_grad(set_to_none=True)
            for offset in range(t['gradient_accumulation']):
                batch=train_batch(run,plan[index+offset],t['patch_size'],device)
                with common.amp(device): out=common.forward(model,batch,arm); loss,_=criterion(out,batch)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                (loss/t['gradient_accumulation']).backward(); losses.append(float(loss.detach()))
            params=[p for p in model.parameters() if p.requires_grad]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params): raise FloatingPointError('Nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(params,t['max_gradient_norm']); optimizer.step()
            completed=index+t['gradient_accumulation']; update=completed//t['gradient_accumulation']
            if update%10==0 or completed==len(plan):
                common.status(run,'training',arm=arm,epoch=epoch,completed=completed,total=len(plan),loss=float(np.mean(losses[-20:])))
            if update%config['checkpoint_every_updates']==0 or completed==len(plan):
                save_commit(run,arm,model,optimizer,epoch,completed,'trained')
                print(f'{arm} epoch {epoch}/2: {completed}/{len(plan)} batches',flush=True)
        start_batch=0
        if common.state_digest(protected.state_dict())!=frozen or any(p.grad is not None for p in protected.parameters()): raise ValueError('Protected model changed')
        result=evaluate(model,run,sets['validation'],device,arm,epoch)
        evaluation={'epoch':epoch,'arm':arm,'validation':result,'guard':safety_checks(baseline,result,config['guards'])}
        save_commit(run,arm,model,optimizer,epoch,len(plan),'evaluated',evaluation=evaluation)
        print(f"{arm} epoch {epoch}: building {result['groups']['highbuild/domain/building']['rmse_m']:.4f} m, canopy {result['groups']['open_canopy/domain/vegetation']['rmse_m']:.4f} m; legacy safety {evaluation['guard']['legacy_safety_pass']}",flush=True)
    del model,protected,optimizer; gc.collect(); torch.cuda.empty_cache()


def execute(args,config,proof,current_binding):
    run=common.resolve(args.resume) if args.resume else common.resolve(config['output_root'])/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        if run.parent!=common.resolve(config['output_root']): raise ValueError('Resume outside named experiment root')
        if json.loads((run/'binding.json').read_text())!=current_binding: raise ValueError('Resume source/config/software mismatch')
    else:
        run.mkdir(parents=True,exist_ok=False); common.atomic_json(run/'config.json',config); common.atomic_json(run/'binding.json',current_binding)
        for path in current_binding['code']:
            dest=run/'source_snapshot'/path; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/path,dest)
    common.atomic_json(ROOT/'outputs/orchestration/class_assisted_height_active.json',{'run_dir':str(run),'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat()})
    print(f'RUN_DIR={run}',flush=True)
    try:
        sets={'train':common.load_surface_manifest(common.resolve(proof['train_manifest'])),'validation':common.load_surface_manifest(common.resolve(config['validation_manifest']))}
        if len(sets['train'])!=1050 or len(sets['validation'])!=280: raise ValueError('Dataset counts changed')
        if {r.sample_id for r in sets['train']}&{r.sample_id for r in sets['validation']}: raise ValueError('Train/development sample overlap')
        device=torch.device('cuda')
        prepare_cache(config,proof,run,sets,device)
        baseline_path=run/'baseline.json'
        baseline_binding=run/'baseline_binding.json'
        if baseline_path.exists():
            if not baseline_binding.exists() or json.loads(baseline_binding.read_text())['sha256']!=common.sha(baseline_path):
                raise ValueError('Baseline binding missing or changed; inspect before recovery')
            baseline=json.loads(baseline_path.read_text())
        else:
            common.seed(proof['seed']); protected,_=common.load_predictor(common.resolve(proof['protected_checkpoint']),device='cpu')
            model=common.ClassAssistedHeightNet(protected).to(device)
            baseline=evaluate(model,run,sets['validation'],device,'baseline',0,baseline=True)
            common.atomic_json(baseline_path,baseline)
            common.atomic_json(baseline_binding,{'sha256':common.sha(baseline_path)})
            del model,protected; gc.collect(); torch.cuda.empty_cache()
        for arm in ['uniform','predicted']:
            if binding(args.config,config,proof)!=current_binding: raise ValueError('Binding changed before arm')
            run_arm(config,proof,run,sets,device,arm,baseline)
        records=[]
        for epoch in [1,2]:
            a=json.loads((run/'uniform'/f'epoch_{epoch:02}_evaluation.json').read_text())
            b=json.loads((run/'predicted'/f'epoch_{epoch:02}_evaluation.json').read_text())
            pa=json.loads((run/'uniform'/f'epoch_{epoch:02}_sampling.json').read_text())
            pb=json.loads((run/'predicted'/f'epoch_{epoch:02}_sampling.json').read_text())
            if pa!=pb: raise ValueError('Paired sampling differs')
            benefit=paired_benefit(baseline,a['validation'],b['validation'],config['guards'])
            records.append({'epoch':epoch,**benefit,'candidate_guard':b['guard'],'control_guard':a['guard'],
                            'eligible_for_legacy_review':benefit['benefit_pass'] and b['guard']['legacy_safety_pass']})
        if json.loads((run/'uniform/initial.json').read_text())!=json.loads((run/'predicted/initial.json').read_text()): raise ValueError('Paired initialization differs')
        for split,items in sets.items():
            for record in items: verify_receipt(receipts_file(run,split,record.sample_id),record,files=True)
        if binding(args.config,config,proof)!=current_binding: raise ValueError('Final binding mismatch')
        outcome={'stage':'complete','paired_epochs':records,'release_eligible':False,'gamus_height_evaluated':False,'app_unchanged':True,
                 'best_eligible_epoch':next((r['epoch'] for r in sorted(records,key=lambda r:sum(x['candidate_rmse_m'] for x in r['priorities'].values())) if r['eligible_for_legacy_review']),None),
                 'interpretation':'Development ablation only, not final unseen-generalization evidence. GAMUS safety, per-building scorecard and app-tile checks remain before any release.'}
        common.atomic_json(run/'outcome.json',outcome); common.status(run,'complete',best_eligible_epoch=outcome['best_eligible_epoch'])
        print(json.dumps(outcome,indent=2),flush=True)
    except BaseException as error:
        common.atomic_json(run/'failure.json',{'type':type(error).__name__,'message':str(error),'inspect_before_resume':True})
        common.status(run,'failed',error=str(error)); raise


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,default=ROOT/'configs/class_assisted_height_development_v1.json'); parser.add_argument('--resume')
    args=parser.parse_args(); config=json.loads(args.config.read_text()); proof=json.loads(common.resolve(config['proof_config']).read_text())
    current=binding(args.config,config,proof)
    output=common.resolve(config['output_root']); output.mkdir(parents=True,exist_ok=True)
    with (output/'.trainer.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'0'); lock.flush()
        lock.seek(0)
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as error: raise RuntimeError('Another paired trainer owns this experiment lock') from error
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',NotGeoreferencedWarning); execute(args,config,proof,current)
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__': main()
