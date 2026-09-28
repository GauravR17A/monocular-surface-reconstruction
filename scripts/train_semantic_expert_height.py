"""Authenticated four-epoch paired soft semantic expert height experiment.

Actual frozen six-class probabilities in candidate B; uniform in control A.
No reference classes/heights passed to the model. Both use the sealed ORIGINAL
sampler, original loss and identical fresh initialization. No production writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import gc
import hashlib
import importlib.util
import json
import msvcrt
import os
from pathlib import Path
import shutil
import warnings

import numpy as np
import torch
from rasterio.errors import NotGeoreferencedWarning
from msr.models.semantic_expert_height import SemanticExpertHeightNet

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('expert_band',ROOT/'scripts/train_height_band_sampling.py')
band=importlib.util.module_from_spec(spec);spec.loader.exec_module(band)
fast,dev,common=band.fast,band.dev,band.common
ARMS=('uniform','predicted')
EXTRA=('scripts/train_semantic_expert_height.py','scripts/prove_semantic_expert_height.py',
       'src/msr/models/semantic_expert_height.py')


def validate_recipe(config,parent):
    if config['epochs']!=4 or config['pairs_per_epoch']!=600 or config['arms']!=list(ARMS):
        raise ValueError('Fixed four-epoch paired budget changed')
    if config['model_type']!=SemanticExpertHeightNet.model_type:
        raise ValueError('Unexpected model architecture')
    if config['sampler']!='sealed_original_control_plans_in_both_arms' or config['loss']!='original_source_balanced':
        raise ValueError('Only class routing may differ between arms')
    if config['guards']!=parent['guards'] or config['checkpoint_every_updates']!=50:
        raise ValueError('Original safety/checkpoint contract changed')
    if any(config[k] for k in ('automatic_promotion','test_used','gamus_height_evaluated')):
        raise ValueError('Development-only, no promotion scope required')


def authenticate(path,config):
    parent_path=ROOT/config['parent_sampler_config'];parent=json.loads(parent_path.read_text())
    proof,binding=band.authenticate(parent_path,parent)
    validate_recipe(config,parent)
    proof_run=common.resolve(config['proof_run'])
    if common.sha(proof_run/'outcome.json')!=config['proof_outcome_sha256']:
        raise ValueError('Routing proof outcome changed')
    outcome=json.loads((proof_run/'outcome.json').read_text())
    if not outcome['passes'] or not outcome['paired_identity'] or outcome['proof_weights_reused']:
        raise ValueError('Routing proof did not pass or weights reused')
    pb=json.loads((proof_run/'binding.json').read_text())
    if pb['config_sha256']!=common.sha(ROOT/config['proof_config']):
        raise ValueError('Routing proof config changed')
    for source,digest in pb['sources'].items():
        if common.sha(ROOT/source)!=digest:
            raise ValueError('Routing proof source changed: '+source)
    if common.sha(common.resolve(config['comparison_baseline']))!=config['comparison_baseline_sha256']:
        raise ValueError('Expanded protected baseline changed')
    binding['parent_sampler_config_sha256']=binding['config_sha256']
    binding['config_sha256']=common.sha(path)
    binding['code'].update({p:common.sha(ROOT/p) for p in EXTRA})
    binding['expert_proof_binding_sha256']=common.sha(proof_run/'binding.json')
    binding['expert_proof_outcome_sha256']=config['proof_outcome_sha256']
    return parent,proof,binding


def make_model(proof):
    t=proof['training'];common.seed(proof['seed'])
    protected,_=common.load_predictor(common.resolve(proof['protected_checkpoint']),device='cpu')
    return SemanticExpertHeightNet(protected,hidden_channels=t['hidden_channels'],
                                  maximum_correction_m=t['maximum_correction_m']).to('cuda')


def run_arm(config,proof,parent,run,sets,plans,arm,baseline):
    t=proof['training'];model=make_model(proof)
    frozen=common.state_digest(model.protected.state_dict());initial=common.state_digest(model.state_dict())
    optimizer=torch.optim.AdamW([
        {'params':list(model.deepest.parameters())+list(model.decoder_stages.parameters()),'lr':t['decoder_learning_rate']},
        {'params':list(model.spatial_correction.parameters())+list(model.correction_head.parameters()),'lr':t['learning_rate']}
    ],weight_decay=t['weight_decay'])
    directory=run/arm;directory.mkdir(exist_ok=True)
    if (directory/'initial.json').exists() and json.loads((directory/'initial.json').read_text())['initial_model_sha256']!=initial:
        raise ValueError('Paired initialization changed')
    common.atomic_json(directory/'initial.json',{'initial_model_sha256':initial})
    first,start=band.restore(run,arm,model,optimizer)
    if common.state_digest(model.protected.state_dict())!=frozen:
        raise ValueError('Resume changed protected tensors')
    criterion=common.SourceBalancedHeightLoss(huber_delta_m=t['huber_delta_m'],mse_weight=t['mse_weight'])
    records={r.sample_id:r for r in sets['train']}
    for epoch in range(first,config['epochs']+1):
        # Identical original draws/crops for both arms; balanced plans are not used.
        plan=plans['control'][str(epoch)]
        if len(plan)!=600:
            raise ValueError('Original draw budget changed')
        sample_hash=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        common.atomic_json(directory/f'epoch_{epoch:02}_sampling.json',{'pairs':plan,'sha256':sample_hash})
        loader=fast.OrderedPrefetch(plan[start:],lambda pair:band.cpu_batch(common.resolve(parent['cache_source_run']),pair,records),4,4)
        model.train();losses=[]
        try:
            for index in range(start,len(plan),t['gradient_accumulation']):
                optimizer.zero_grad(set_to_none=True)
                for offset in range(t['gradient_accumulation']):
                    draw,batch=loader.pop()
                    if draw!=plan[index+offset]:
                        raise ValueError('Prefetch changed paired sequence')
                    batch={k:v.pin_memory().to('cuda',non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                    with common.amp(torch.device('cuda')):
                        output=common.forward(model,batch,arm);loss,_=criterion(output,batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    (loss/t['gradient_accumulation']).backward();losses.append(float(loss.detach()))
                parameters=[p for p in model.parameters() if p.requires_grad]
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError('Nonfinite gradient')
                torch.nn.utils.clip_grad_norm_(parameters,t['max_gradient_norm']);optimizer.step()
                completed=index+t['gradient_accumulation'];update=completed//t['gradient_accumulation']
                if update%10==0 or completed==len(plan):
                    common.status(run,'training',arm=arm,epoch=epoch,completed=completed,total=len(plan),loss=float(np.mean(losses[-20:])))
                if update%config['checkpoint_every_updates']==0 or completed==len(plan):
                    dev.save_commit(run,arm,model,optimizer,epoch,completed,'trained')
                    print(f'{arm} epoch {epoch}/4: {completed}/600 batches',flush=True)
        finally:
            loader.close()
        start=0
        if common.state_digest(model.protected.state_dict())!=frozen or any(p.grad is not None for p in model.protected.parameters()):
            raise ValueError('Protected model changed')
        validation=dev.evaluate(model,run,sets['validation'],torch.device('cuda'),arm,epoch)
        report={'arm':arm,'epoch':epoch,'validation':validation,'guard':dev.safety_checks(baseline,validation,config['guards'])}
        dev.save_commit(run,arm,model,optimizer,epoch,600,'evaluated',evaluation=report)
        print(json.dumps({'arm':arm,'epoch':epoch,'safety_pass':report['guard']['legacy_safety_pass'],
            'building_rmse_m':validation['groups']['highbuild/domain/building']['rmse_m'],
            'canopy_rmse_m':validation['groups']['open_canopy/domain/vegetation']['rmse_m']}),flush=True)
    del model,optimizer;gc.collect();torch.cuda.empty_cache()


def finish(config,run,baseline):
    rows=[];reports=[]
    for epoch in range(1,5):
        a=json.loads((run/'uniform'/f'epoch_{epoch:02}_evaluation.json').read_text())
        b=json.loads((run/'predicted'/f'epoch_{epoch:02}_evaluation.json').read_text())
        for field in ('sha256','pairs'):
            if json.loads((run/'uniform'/f'epoch_{epoch:02}_sampling.json').read_text())[field]!=json.loads((run/'predicted'/f'epoch_{epoch:02}_sampling.json').read_text())[field]:
                raise ValueError('Paired sample sequences differ')
        benefit=dev.paired_benefit(baseline,a['validation'],b['validation'],config['guards'])
        rows.append({'epoch':epoch,**benefit,'candidate_guard':b['guard'],'control_guard':a['guard'],
                     'eligible_for_legacy_review':bool(benefit['benefit_pass'] and b['guard']['legacy_safety_pass'])})
        reports.extend([a,b])
    if json.loads((run/'uniform/initial.json').read_text())!=json.loads((run/'predicted/initial.json').read_text()):
        raise ValueError('Paired initialization differs')
    priority=('highbuild/domain/building','open_canopy/domain/vegetation')
    best=next((r['epoch'] for r in sorted(rows,key=lambda r:sum(x['candidate_rmse_m'] for x in r['priorities'].values())) if r['eligible_for_legacy_review']),None)
    unrestricted=min(reports,key=lambda r:sum(r['validation']['groups'][g]['rmse_m'] for g in priority))
    out={'stage':'complete','paired_epochs':rows,'best_eligible_epoch':best,
         'best_unrestricted':{'arm':unrestricted['arm'],'epoch':unrestricted['epoch'],'ranking':'equal building/canopy RMSE'},
         'release_eligible':False,'app_unchanged':True,'gamus_height_evaluated':False,'actual_six_class_predictions_in_candidate':True,
         'interpretation':'Development only. Six routes are not six validated height categories. No automatic promotion.'}
    common.atomic_json(run/'outcome.json',out)
    lines=['# Soft six-class routing: development results','',
           '| Arm/epoch | Building RMSE | Canopy RMSE | Ground RMSE | Short vegetation RMSE | Safety |',
           '|---|---:|---:|---:|---:|---|']
    groups=priority+('open_canopy/domain/ground','open_canopy/short_vegetation')
    for label,validation,safety in [('Protected',baseline,'baseline')]+[(f'{r["arm"]}/{r["epoch"]}',r['validation'],'PASS' if r['guard']['legacy_safety_pass'] else 'FAIL') for r in reports]:
        lines.append('| '+label+' | '+' | '.join(f'{validation["groups"][g]["rmse_m"]:.3f} m' for g in groups)+' | '+safety+' |')
    lines+=['',f'Best eligible candidate epoch: {best if best else "NONE"}.',f'Best unrestricted: {out["best_unrestricted"]}.',
            'Support, short/medium/tall objects, boundaries and region guards retained. Complete metrics are saved per epoch.',
            'Neither development results nor proof training loss establishes untouched-geography accuracy.',
            'Protected app and classifiers unchanged. No automatic release or further training.']
    (run/'DEVELOPMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    common.status(run,'complete',best_eligible_epoch=best);print(f'Complete. Best eligible: {best}',flush=True)


def execute(args,config,parent,proof,binding):
    run=common.resolve(args.resume) if args.resume else common.resolve(config['output_root'])/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        if run.parent!=common.resolve(config['output_root']) or json.loads((run/'binding.json').read_text())!=binding:
            raise ValueError('Compatible source/config/data/software binding required')
    else:
        run.mkdir(parents=True,exist_ok=False)
        common.atomic_json(run/'config.json',config);common.atomic_json(run/'binding.json',binding)
        shutil.copy2(common.resolve(parent['cache_source_run'])/'data_binding.json',run/'data_binding.json')
        shutil.copy2(common.resolve(parent['sampling_plans']),run/'sampling_plans.json')
        for path in binding['code']:
            dest=run/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dest)
    common.status(run,'checking_reused_inputs',completed=0,total=1330)
    common.atomic_json(ROOT/'outputs/orchestration/class_assisted_height_active.json',
        {'run_dir':str(run),'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),'experiment':'semantic_expert_height_v1'})
    print('SEMANTIC_EXPERT_RUN_DIR='+str(run),flush=True)
    adapter=None;original_groups=dev.add_groups
    try:
        sets={'train':common.load_surface_manifest(common.resolve(proof['train_manifest'])),
              'validation':common.load_surface_manifest(common.resolve(parent['validation_manifest']))}
        if common.sha(run/'data_binding.json')!=parent['cache_bindings']['data_binding.json']:
            raise ValueError('Run data binding changed')
        if common.sha(run/'sampling_plans.json')!=parent['sampling_plans_sha256']:
            raise ValueError('Sealed plans changed')
        fast.check_reused_inputs(common.resolve(parent['cache_source_run']),sets,run)
        adapter=fast.FastInputs(common.resolve(parent['cache_source_run']),parent['performance']);adapter.install()
        dev.add_groups=band.extended_groups(original_groups)
        if (run/'baseline.json').exists():
            if common.sha(run/'baseline.json')!=json.loads((run/'baseline_binding.json').read_text())['sha256']:
                raise ValueError('Baseline changed')
            baseline=json.loads((run/'baseline.json').read_text())
        else:
            model=make_model(proof)
            baseline=dev.evaluate(model,run,sets['validation'],torch.device('cuda'),'baseline',0,baseline=True)
            if baseline!=json.loads(common.resolve(config['comparison_baseline']).read_text()):
                raise ValueError('Expanded protected baseline does not exactly match')
            common.atomic_json(run/'baseline.json',baseline)
            common.atomic_json(run/'baseline_binding.json',{'sha256':common.sha(run/'baseline.json')})
            del model;gc.collect();torch.cuda.empty_cache()
        plans=json.loads((run/'sampling_plans.json').read_text())
        for arm in ARMS:
            if authenticate(args.config,config)[2]!=binding:
                raise ValueError('Bindings changed before arm')
            if common.sha(run/'sampling_plans.json')!=parent['sampling_plans_sha256']:
                raise ValueError('Run plans changed before arm')
            run_arm(config,proof,parent,run,sets,plans,arm,baseline)
        fast.check_reused_inputs(common.resolve(parent['cache_source_run']),sets,run)
        if authenticate(args.config,config)[2]!=binding:
            raise ValueError('Final bindings changed')
        finish(config,run,baseline)
    except BaseException as error:
        common.atomic_json(run/'failure.json',{'type':type(error).__name__,'message':str(error),'inspect_before_resume':True})
        common.status(run,'failed',error=str(error));raise
    finally:
        dev.add_groups=original_groups
        if adapter is not None: adapter.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/semantic_expert_height_v1.json')
    parser.add_argument('--resume');parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text());parent,proof,binding=authenticate(args.config,config)
    if args.preflight:
        print(json.dumps({'passed':True,'bound_sources':len(binding['code']),'epochs_per_arm':4,'actual_classes_in_candidate':True}));return
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with (common.resolve(parent['cache_source_run']).parent/'.trainer.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',NotGeoreferencedWarning);execute(args,config,parent,proof,binding)
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':main()
