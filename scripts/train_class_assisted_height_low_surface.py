"""Two-epoch matched low-surface loss experiment, fresh protected initialization.

Reuse the authenticated fast loader and numerical loop; change only the explicit
training loss in both arms. Old experiments and production stay immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import msvcrt
import os
from pathlib import Path
import shutil
import time
import warnings

import torch
from rasterio.errors import NotGeoreferencedWarning
from msr.training.low_surface_height_loss import LowSurfaceProtectedHeightLoss

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fast_height_low_surface', ROOT/'scripts/train_class_assisted_height_fast.py')
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)
dev, common = fast.dev, fast.common
SOURCES = ('scripts/train_class_assisted_height_fast.py', 'scripts/train_class_assisted_height_low_surface.py',
           'src/msr/training/low_surface_height_loss.py')


def validate_scope(config, prior_config):
    for key in ('epochs','pairs_per_epoch','guards','validation_manifest_sha256','checkpoint_every_updates'):
        if config[key] != prior_config[key]:
            raise ValueError(f'Unchanged development scope required: {key}')
    if config['automatic_promotion'] or config['test_used'] or config['gamus_height_evaluated']:
        raise ValueError('Development-only scope required')
    if config['initialization'] != 'fresh_protected_same_seed_both_arms':
        raise ValueError('Must not reuse failed correction weights')
    if config['loss'] != {'short_supervision_weight':.5,'low_surface_regret_weight':1.0}:
        raise ValueError('Predeclared loss recipe changed')


def authenticate(config_path, config, proof):
    binding = dev.binding(config_path, config, proof)
    cache_source = common.resolve(config['cache_source_run'])
    for name,digest in config['cache_bindings'].items():
        if common.sha(cache_source/name) != digest:
            raise ValueError(f'Cache source binding changed: {name}')
    parent_binding = json.loads((cache_source/'binding.json').read_text())
    if parent_binding['code'] != binding['code']:
        raise ValueError('Original numerical/model/data source changed')
    for key in ('torch','transformers','python','device','proof_config_sha256'):
        if parent_binding[key] != binding[key]:
            raise ValueError(f'Original runtime changed: {key}')
    validate_scope(config, json.loads((cache_source/'config.json').read_text()))
    for key in ('diagnostic_report','previous_outcome','loader_benchmark','v1_classifier_checkpoint'):
        if common.sha(common.resolve(config[key])) != config[key+'_sha256']:
            raise ValueError(f'Bound evidence changed: {key}')
    diagnostic = json.loads(common.resolve(config['diagnostic_report']).read_text())
    if not diagnostic['development_only'] or diagnostic['final_tests_used'] or diagnostic['images'] != 280:
        raise ValueError('Unexpected error-analysis scope')
    binding['code'].update({path:common.sha(ROOT/path) for path in SOURCES})
    binding['cache_bindings'] = config['cache_bindings']
    binding['loss'] = config['loss']
    binding['performance'] = config['performance']
    return binding


def assert_resume(run, config, binding):
    if run.parent != common.resolve(config['output_root']):
        raise ValueError('Resume outside experiment root')
    if json.loads((run/'binding.json').read_text()) != binding:
        raise ValueError('Resume source/config/software binding mismatch')


def execute(args, config, proof, binding):
    cache_source = common.resolve(config['cache_source_run'])
    run = common.resolve(args.resume) if args.resume else common.resolve(config['output_root'])/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        assert_resume(run,config,binding)
    else:
        run.mkdir(parents=True,exist_ok=False)
        common.atomic_json(run/'config.json',config)
        common.atomic_json(run/'binding.json',binding)
        for name in ('data_binding.json','baseline.json','baseline_binding.json'):
            shutil.copy2(cache_source/name,run/name)
        for path in binding['code']:
            target=run/'source_snapshot'/path
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/path,target)
        shutil.copy2(common.resolve(config['diagnostic_report']),run/'preceding_error_analysis.json')
        common.atomic_json(run/'experiment_change.json',{'only_training_change':'low-surface-protected loss in BOTH arms',
            'initialization':config['initialization'],'previous_outcome_sha256':config['previous_outcome_sha256'],
            'old_results_preserved':True,'no_checkpoint_migration':True})
    common.status(run,'checking_reused_inputs',completed=0,total=1330)
    common.atomic_json(ROOT/'outputs/orchestration/class_assisted_height_active.json',
                       {'run_dir':str(run),'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
                        'experiment':'low_surface_v3','cache_source_run':str(cache_source)})
    print(f'LOW_SURFACE_RUN_DIR={run}',flush=True)
    adapter=None
    original_criterion=common.SourceBalancedHeightLoss
    try:
        sets={'train':common.load_surface_manifest(common.resolve(proof['train_manifest'])),
              'validation':common.load_surface_manifest(common.resolve(config['validation_manifest']))}
        if len(sets['train']) != 1050 or len(sets['validation']) != 280:
            raise ValueError('Dataset counts changed')
        fast.check_reused_inputs(cache_source,sets,run)
        for name in ('data_binding.json','baseline.json'):
            if common.sha(run/name) != config['cache_bindings'][name]:
                raise ValueError(f'Run binding changed: {name}')
        baseline=json.loads((run/'baseline.json').read_text())
        adapter=fast.FastInputs(cache_source,config['performance'])
        adapter.install()
        common.SourceBalancedHeightLoss=lambda **kwargs: LowSurfaceProtectedHeightLoss(**kwargs,**config['loss'])
        timings={}
        for arm in ('uniform','predicted'):
            if authenticate(args.config,config,proof) != binding:
                raise ValueError('Bindings changed before arm')
            started=time.perf_counter()
            dev.run_arm(config,proof,run,sets,torch.device('cuda'),arm,baseline)
            timings[arm]=time.perf_counter()-started
        pairs=[]
        for epoch in range(1,config['epochs']+1):
            a=json.loads((run/'uniform'/f'epoch_{epoch:02}_evaluation.json').read_text())
            b=json.loads((run/'predicted'/f'epoch_{epoch:02}_evaluation.json').read_text())
            if json.loads((run/'uniform'/f'epoch_{epoch:02}_sampling.json').read_text()) != json.loads((run/'predicted'/f'epoch_{epoch:02}_sampling.json').read_text()):
                raise ValueError('Paired sampling differs')
            benefit=dev.paired_benefit(baseline,a['validation'],b['validation'],config['guards'])
            pairs.append({'epoch':epoch,**benefit,'candidate_guard':b['guard'],'control_guard':a['guard'],
                          'eligible_for_legacy_review':benefit['benefit_pass'] and b['guard']['legacy_safety_pass']})
        if json.loads((run/'uniform/initial.json').read_text()) != json.loads((run/'predicted/initial.json').read_text()):
            raise ValueError('Paired initialization differs')
        fast.check_reused_inputs(cache_source,sets,run)
        if authenticate(args.config,config,proof) != binding:
            raise ValueError('Final binding mismatch')
        outcome={'stage':'complete','paired_epochs':pairs,'release_eligible':False,'gamus_height_evaluated':False,
            'app_unchanged':True,'timings_seconds':timings,
            'best_eligible_epoch':next((r['epoch'] for r in sorted(pairs,key=lambda r:sum(x['candidate_rmse_m'] for x in r['priorities'].values())) if r['eligible_for_legacy_review']),None),
            'interpretation':'New loss experiment, same two-epoch matched protocol; development only, no automatic promotion.'}
        common.atomic_json(run/'outcome.json',outcome)
        common.status(run,'complete',best_eligible_epoch=outcome['best_eligible_epoch'])
        print(json.dumps({k:v for k,v in outcome.items() if k != 'paired_epochs'},indent=2),flush=True)
    except BaseException as error:
        common.atomic_json(run/'failure.json',{'type':type(error).__name__,'message':str(error),'inspect_before_resume':True})
        common.status(run,'failed',error=str(error))
        raise
    finally:
        common.SourceBalancedHeightLoss=original_criterion
        if adapter is not None:
            adapter.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/class_assisted_height_low_surface_v3.json')
    parser.add_argument('--resume')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    proof=json.loads(common.resolve(config['proof_config']).read_text())
    binding=authenticate(args.config,config,proof)
    if args.preflight:
        print(json.dumps({'passed':True,'bound_sources':len(binding['code']),'scope':'new low-surface loss; frozen original model; identical guards'}))
        return
    torch.set_num_threads(config['performance']['cpu_tensor_threads'])
    torch.set_num_interop_threads(1)
    lock_path=common.resolve(config['cache_source_run']).parent/'.trainer.lock'
    with lock_path.open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',NotGeoreferencedWarning)
                execute(args,config,proof,binding)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__ == '__main__':
    main()
