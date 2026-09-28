"""Four-epoch matched sampler experiment. Original loss; uniform classes in both.

New versioned run, unchanged protected model/labels/validation support. Only
training observations/crop anchors differ between control and balanced arms.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import msvcrt
import os
from pathlib import Path
import random
import shutil
import time
import warnings

import numpy as np
import torch
from torch.utils.data import default_collate
from rasterio.errors import NotGeoreferencedWarning
from msr.training.height_band_sampling import height_masks,count_bands

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('band_fast',ROOT/'scripts/train_class_assisted_height_fast.py')
fast=importlib.util.module_from_spec(spec); spec.loader.exec_module(fast)
dev,common=fast.dev,fast.common
EXTRA=('scripts/train_class_assisted_height_fast.py','scripts/train_height_band_sampling.py',
       'scripts/prepare_height_band_sampling.py','src/msr/training/height_band_sampling.py')
ARMS=('control','balanced')


def authenticate(path,config):
    original_path=ROOT/'configs/class_assisted_height_development_v1.json'
    original=json.loads(original_path.read_text())
    proof=json.loads(common.resolve(original['proof_config']).read_text())
    binding=dev.binding(original_path,original,proof)
    parent=common.resolve(config['cache_source_run'])
    if binding!=json.loads((parent/'binding.json').read_text()):
        raise ValueError('Original model/data/software contract changed')
    if config['epochs']!=4 or config['pairs_per_epoch']!=600 or config['arms']!=list(ARMS):
        raise ValueError('Declared four-epoch paired budget changed')
    if config['loss']!='original_source_balanced' or config['conditioning']!='uniform_both_arms':
        raise ValueError('Sampler must be the only between-arm intervention')
    if config['automatic_promotion'] or config['test_used'] or config['gamus_height_evaluated']:
        raise ValueError('Development-only experiment required')
    if config['guards']!=original['guards'] or config['validation_manifest']!=original['validation_manifest']:
        raise ValueError('Original gates/validation must remain fixed')
    for name,digest in config['cache_bindings'].items():
        if common.sha(parent/name)!=digest:
            raise ValueError(f'Cache binding changed: {name}')
    for key in ('sampling_audit','sampling_plans','v1_classifier_checkpoint'):
        if common.sha(common.resolve(config[key]))!=config[key+'_sha256']:
            raise ValueError(f'Binding changed: {key}')
    audit=json.loads(common.resolve(config['sampling_audit']).read_text())
    if not audit['training_only'] or audit['validation_read'] or audit['final_tests_used']:
        raise ValueError('Sampling must be prepared from training data only')
    if audit['plans_sha256']!=config['sampling_plans_sha256'] or audit['training_manifest_sha256']!=proof['train_manifest_sha256']:
        raise ValueError('Sampling plan identity mismatch')
    for source,digest in audit['source_sha256'].items():
        if common.sha(ROOT/source)!=digest:
            raise ValueError(f'Sampling preparation code changed: {source}')
    binding['original_config_sha256']=binding['config_sha256']
    binding['config_sha256']=common.sha(path)
    binding['code'].update({p:common.sha(ROOT/p) for p in EXTRA})
    binding['sampling_audit_sha256']=config['sampling_audit_sha256']
    binding['sampling_plans_sha256']=config['sampling_plans_sha256']
    return proof,binding


def extended_groups(original):
    def add(acc,prediction,target,valid,domain,suite,region,probabilities,legacy_domain):
        original(acc,prediction,target,valid,domain,suite,region,probabilities,legacy_domain)
        for name,mask in height_masks(target,valid,domain).items():
            if name in ('short_building','medium_vegetation'):
                acc[suite+'/height_band/'+name].update(prediction,target,mask)
                acc[suite+'/region/'+region+'/height_band/'+name].update(prediction,target,mask)
    return add


def crop_sample(sample,draw):
    crop=draw['crop']; r,c,size=crop['row'],crop['col'],crop['size']
    if size!=384 or r<0 or c<0 or r+size>sample['image'].shape[-2] or c+size>sample['image'].shape[-1]:
        raise ValueError('Invalid pinned crop geometry')
    result={key:value[...,r:r+size,c:c+size].contiguous() if isinstance(value,torch.Tensor) else value for key,value in sample.items()}
    actual=count_bands(result['height'][0].numpy(),result['regression_mask'][0].numpy(),result['domain_target'].numpy())
    if actual!=draw['exposure'] or int(result['regression_mask'].sum())!=draw['supported_pixels']:
        raise ValueError('Crop reference support changed')
    return result


def cpu_batch(cache,draws,records):
    samples=[crop_sample(dev.cached_sample(cache,'train',records[draw['sample_id']]),draw) for draw in draws]
    if [sample['landscape'] for sample in samples]!=['urban','forest']:
        raise ValueError('Source-paired batch contract changed')
    return default_collate(samples)


def restore(run,arm,model,optimizer):
    path=run/arm/'commit.json'
    if not path.exists():
        return 1,0
    commit=json.loads(path.read_text()); checkpoint=path.parent/commit['checkpoint']
    if common.sha(checkpoint)!=commit['checkpoint_sha256']:
        raise ValueError('Committed checkpoint changed')
    payload=torch.load(checkpoint,map_location='cuda',weights_only=False)
    if payload['binding_sha256']!=common.sha(run/'binding.json') or payload['data_binding_sha256']!=common.sha(run/'data_binding.json') or payload['arm']!=arm:
        raise ValueError('Resume binding mismatch')
    if (payload['epoch'],payload['batch_done']) != (commit['epoch'],commit['batch_done']) or commit['batch_done']%2:
        raise ValueError('Resume training cursor mismatch')
    model.load_state_dict(payload['model_state_dict'],strict=True); optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['rng'].cpu()); torch.cuda.set_rng_state_all([v.cpu() for v in payload['cuda_rng']])
    np.random.set_state(payload['numpy_rng']); random.setstate(payload['python_rng'])
    if commit['stage']=='evaluated':
        if common.sha(path.parent/commit['evaluation'])!=commit['evaluation_sha256']:
            raise ValueError('Committed evaluation changed')
        return commit['epoch']+1,0
    return commit['epoch'],commit['batch_done']


def run_arm(config,proof,run,sets,plans,arm,baseline):
    device=torch.device('cuda'); t=proof['training']; common.seed(proof['seed'])
    protected,_=common.load_predictor(common.resolve(proof['protected_checkpoint']),device='cpu')
    model=common.ClassAssistedHeightNet(protected,hidden_channels=t['hidden_channels'],maximum_correction_m=t['maximum_correction_m']).to(device)
    frozen=common.state_digest(protected.state_dict()); initial=common.state_digest(model.state_dict())
    optimizer=torch.optim.AdamW([
        {'params':list(model.deepest.parameters())+list(model.decoder_stages.parameters()),'lr':t['decoder_learning_rate']},
        {'params':list(model.spatial_correction.parameters())+list(model.correction_head.parameters()),'lr':t['learning_rate']}],weight_decay=t['weight_decay'])
    directory=run/arm; directory.mkdir(exist_ok=True)
    if (directory/'initial.json').exists() and json.loads((directory/'initial.json').read_text())['initial_model_sha256']!=initial:
        raise ValueError('Fresh initialization differs')
    common.atomic_json(directory/'initial.json',{'initial_model_sha256':initial})
    first,start=restore(run,arm,model,optimizer)
    if common.state_digest(protected.state_dict())!=frozen:
        raise ValueError('Resume changed frozen weights')
    criterion=common.SourceBalancedHeightLoss(huber_delta_m=t['huber_delta_m'],mse_weight=t['mse_weight'])
    records={r.sample_id:r for r in sets['train']}
    for epoch in range(first,config['epochs']+1):
        plan=plans[arm][str(epoch)]
        if len(plan)!=600:
            raise ValueError('Training draw budget changed')
        common.atomic_json(directory/f'epoch_{epoch:02}_sampling.json',{'pairs':plan,'plan_sha256':config['sampling_plans_sha256'],'arm':arm})
        loader=fast.OrderedPrefetch(plan[start:],lambda pair:cpu_batch(common.resolve(config['cache_source_run']),pair,records),4,4)
        model.train(); losses=[]
        try:
            for index in range(start,len(plan),t['gradient_accumulation']):
                optimizer.zero_grad(set_to_none=True)
                for offset in range(t['gradient_accumulation']):
                    draw,batch=loader.pop()
                    if draw!=plan[index+offset]:
                        raise ValueError('Prefetch changed sample order')
                    batch={k:v.pin_memory().to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                    with common.amp(device):
                        out=common.forward(model,batch,arm='uniform'); loss,_=criterion(out,batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    (loss/t['gradient_accumulation']).backward(); losses.append(float(loss.detach()))
                params=[p for p in model.parameters() if p.requires_grad]
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                    raise FloatingPointError('Nonfinite gradient')
                torch.nn.utils.clip_grad_norm_(params,t['max_gradient_norm']); optimizer.step()
                completed=index+t['gradient_accumulation']; update=completed//t['gradient_accumulation']
                if update%10==0 or completed==len(plan):
                    common.status(run,'training',arm=arm,epoch=epoch,completed=completed,total=len(plan),loss=float(np.mean(losses[-20:])))
                if update%config['checkpoint_every_updates']==0 or completed==len(plan):
                    dev.save_commit(run,arm,model,optimizer,epoch,completed,'trained')
                    print(f'{arm} epoch {epoch}/{config["epochs"]}: {completed}/{len(plan)} batches',flush=True)
        finally:
            loader.close()
        start=0
        if common.state_digest(protected.state_dict())!=frozen or any(p.grad is not None for p in protected.parameters()):
            raise ValueError('Protected model changed')
        result=dev.evaluate(model,run,sets['validation'],device,arm,epoch)
        report={'epoch':epoch,'arm':arm,'validation':result,'guard':dev.safety_checks(baseline,result,config['guards'])}
        dev.save_commit(run,arm,model,optimizer,epoch,600,'evaluated',evaluation=report)
        print(json.dumps({'arm':arm,'epoch':epoch,'legacy_safety_pass':report['guard']['legacy_safety_pass'],
            'building_rmse_m':result['groups']['highbuild/domain/building']['rmse_m'],
            'canopy_rmse_m':result['groups']['open_canopy/domain/vegetation']['rmse_m']},indent=None),flush=True)
    del model,protected,optimizer; gc.collect(); torch.cuda.empty_cache()


def write_outcome(config,run,baseline):
    rows=[]; reports=[]
    for epoch in range(1,config['epochs']+1):
        a=json.loads((run/'control'/f'epoch_{epoch:02}_evaluation.json').read_text())
        b=json.loads((run/'balanced'/f'epoch_{epoch:02}_evaluation.json').read_text())
        benefit=dev.paired_benefit(baseline,a['validation'],b['validation'],config['guards'])
        rows.append({'epoch':epoch,**benefit,'candidate_guard':b['guard'],'control_guard':a['guard'],
            'eligible_for_legacy_review':benefit['benefit_pass'] and b['guard']['legacy_safety_pass']})
        reports.extend([a,b])
    if json.loads((run/'control/initial.json').read_text())!=json.loads((run/'balanced/initial.json').read_text()):
        raise ValueError('Paired initial weights differ')
    best=next((r['epoch'] for r in sorted(rows,key=lambda r:sum(x['candidate_rmse_m'] for x in r['priorities'].values())) if r['eligible_for_legacy_review']),None)
    outcome={'stage':'complete','paired_epochs':rows,'best_eligible_epoch':best,'release_eligible':False,
        'app_unchanged':True,'gamus_height_evaluated':False,'comparison':'Sampler only; uniform conditioning and original loss in both arms.',
        'interpretation':'Development data, not untouched final tests. No automatic promotion or further training.'}
    common.atomic_json(run/'outcome.json',outcome)
    lines=['# Height-band sampling: completed development comparison','',
        '| Arm / epoch | Buildings RMSE | Canopy RMSE | Ground RMSE | Short veg RMSE | Safety |',
        '|---|---:|---:|---:|---:|---|']
    groups=('highbuild/domain/building','open_canopy/domain/vegetation','open_canopy/domain/ground','open_canopy/short_vegetation')
    def row(label,validation,safety):
        return '| '+label+' | '+' | '.join(f'{validation["groups"][g]["rmse_m"]:.3f} m' for g in groups)+' | '+safety+' |'
    lines.append(row('Protected',baseline,'baseline'))
    for report in reports:
        lines.append(row(f'{report["arm"]} / {report["epoch"]}',report['validation'],'PASS' if report['guard']['legacy_safety_pass'] else 'FAIL'))
    lines.extend(['',f'Best eligible candidate epoch: {best if best else "NONE"}.',
        'Full MAE, bias, correlation, R2, height bands, boundaries and regional guards are saved in each committed epoch evaluation.',
        'The new sampler cannot be credited with ordinary control improvement; compare same-epoch arms.',
        'All training observations are from 450 HighBuild and 600 OpenCanopy scenes. No fresh final geography or GAMUS height certification.',
        'No automatic release; protected model, pointer and classifiers unchanged.'])
    (run/'DEVELOPMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    common.status(run,'complete',best_eligible_epoch=best)
    print(f'Complete. Best eligible epoch: {best}',flush=True)


def execute(args,config,proof,binding):
    run=common.resolve(args.resume) if args.resume else common.resolve(config['output_root'])/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        if run.parent!=common.resolve(config['output_root']) or json.loads((run/'binding.json').read_text())!=binding:
            raise ValueError('Resume source/config/data/software binding mismatch')
    else:
        run.mkdir(parents=True,exist_ok=False)
        common.atomic_json(run/'config.json',config); common.atomic_json(run/'binding.json',binding)
        shutil.copy2(common.resolve(config['cache_source_run'])/'data_binding.json',run/'data_binding.json')
        shutil.copy2(common.resolve(config['sampling_audit']),run/'sampling_audit.json')
        shutil.copy2(common.resolve(config['sampling_plans']),run/'sampling_plans.json')
        for path in binding['code']:
            target=run/'source_snapshot'/path; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/path,target)
    common.status(run,'checking_reused_inputs',completed=0,total=1330)
    common.atomic_json(ROOT/'outputs/orchestration/class_assisted_height_active.json',{'run_dir':str(run),'pid':os.getpid(),
        'started_utc':datetime.now(timezone.utc).isoformat(),'experiment':'height_band_sampling_v1'})
    print(f'BAND_RUN_DIR={run}',flush=True)
    adapter=None; original_groups=dev.add_groups; original_forward=common.forward
    try:
        sets={'train':common.load_surface_manifest(common.resolve(proof['train_manifest'])),
              'validation':common.load_surface_manifest(common.resolve(config['validation_manifest']))}
        if common.sha(run/'sampling_plans.json')!=config['sampling_plans_sha256'] or common.sha(run/'sampling_audit.json')!=config['sampling_audit_sha256']:
            raise ValueError('Run sampling artifacts changed')
        if common.sha(run/'data_binding.json')!=config['cache_bindings']['data_binding.json']:
            raise ValueError('Run data binding changed')
        fast.check_reused_inputs(common.resolve(config['cache_source_run']),sets,run)
        adapter=fast.FastInputs(common.resolve(config['cache_source_run']),config['performance']); adapter.install()
        dev.add_groups=extended_groups(original_groups)
        # Arm names describe sampling, never a switch to informative probabilities.
        common.forward=lambda model,batch,arm:original_forward(model,batch,arm='uniform')
        if (run/'baseline.json').exists():
            if common.sha(run/'baseline.json')!=json.loads((run/'baseline_binding.json').read_text())['sha256']:
                raise ValueError('Baseline hash mismatch')
            baseline=json.loads((run/'baseline.json').read_text())
        else:
            common.seed(proof['seed'])
            protected,_=common.load_predictor(common.resolve(proof['protected_checkpoint']),device='cpu')
            model=common.ClassAssistedHeightNet(protected).to('cuda')
            baseline=dev.evaluate(model,run,sets['validation'],torch.device('cuda'),'baseline',0,baseline=True)
            old=json.loads((common.resolve(config['cache_source_run'])/'baseline.json').read_text())
            if any(old[key]!=baseline[key] for key in ('support_sha256','semantic_sha256')):
                raise ValueError('Original baseline support/semantics changed')
            if any(baseline['groups'].get(key)!=value for key,value in old['groups'].items()):
                raise ValueError('Original baseline metrics changed')
            common.atomic_json(run/'baseline.json',baseline)
            common.atomic_json(run/'baseline_binding.json',{'sha256':common.sha(run/'baseline.json')})
            del model,protected; gc.collect(); torch.cuda.empty_cache()
        plans=json.loads((run/'sampling_plans.json').read_text())
        for arm in ARMS:
            if authenticate(args.config,config)[1]!=binding:
                raise ValueError('Bindings changed before training arm')
            if common.sha(run/'sampling_plans.json')!=config['sampling_plans_sha256']:
                raise ValueError('Run sampling plan changed')
            run_arm(config,proof,run,sets,plans,arm,baseline)
        fast.check_reused_inputs(common.resolve(config['cache_source_run']),sets,run)
        if authenticate(args.config,config)[1]!=binding:
            raise ValueError('Final bindings changed')
        write_outcome(config,run,baseline)
    except BaseException as error:
        common.atomic_json(run/'failure.json',{'type':type(error).__name__,'message':str(error),'inspect_before_resume':True})
        common.status(run,'failed',error=str(error)); raise
    finally:
        dev.add_groups=original_groups; common.forward=original_forward
        if adapter is not None:
            adapter.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/height_band_sampling_v1.json')
    parser.add_argument('--resume'); parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args(); config=json.loads(args.config.read_text()); proof,binding=authenticate(args.config,config)
    if args.preflight:
        print(json.dumps({'passed':True,'bound_sources':len(binding['code']),'epochs_per_arm':4,'conditioning':'uniform in BOTH arms'})); return
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    with (common.resolve(config['cache_source_run']).parent/'.trainer.lock').open('a+b') as lock:
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',NotGeoreferencedWarning); execute(args,config,proof,binding)
        finally:
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':
    main()
