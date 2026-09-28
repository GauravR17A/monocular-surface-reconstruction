"""Bounded train-only soft semantic height routing proof; no production writes."""
from __future__ import annotations
import gc
import hashlib
import importlib.util
import json
import msvcrt
from pathlib import Path
import shutil
from datetime import datetime,timezone
import warnings
import torch
from rasterio.errors import NotGeoreferencedWarning
from msr.models.semantic_expert_height import SemanticExpertHeightNet

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('expert_proof_common',ROOT/'scripts/class_assisted_height_proof.py')
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
EXTRA=('scripts/prove_semantic_expert_height.py','src/msr/models/semantic_expert_height.py')
CONFIG=ROOT/'configs/semantic_expert_height_proof_v1.json'


def authenticate():
    cfg=json.loads(CONFIG.read_text());original_path=ROOT/cfg['parent_proof_config']
    original=json.loads(original_path.read_text());sources=common.authenticate(original)
    parent=Path(cfg['parent_proof_run'])
    if common.sha(parent/'outcome.json')!=cfg['parent_outcome_sha256']:
        raise ValueError('Parent proof changed')
    saved=json.loads((parent/'binding.json').read_text())
    if saved['source_sha256']!=sources or saved['config_sha256']!=common.sha(original_path):
        raise ValueError('Authenticated parent code/config changed')
    if not cfg['training_only'] or any(cfg[k] for k in ('validation_used','test_used','automatic_promotion','proof_weights_reused')):
        raise ValueError('Training-only proof required')
    if cfg['optimizer_updates']!=64 or cfg['minimum_loss_reduction']!=.1:
        raise ValueError('Bounded proof scope changed')
    binding={'original':saved,'config_sha256':common.sha(CONFIG),
             'cache_manifest_sha256':common.sha(parent/'cache_manifest.json'),
             'sources':{**sources,**{p:common.sha(ROOT/p) for p in EXTRA}}}
    return cfg,original,binding


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cfg,original,binding=authenticate()
    run=Path(cfg['output_root'])/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True,exist_ok=False)
    common.atomic_json(run/'config.json',cfg);common.atomic_json(run/'binding.json',binding)
    for p in binding['sources']:
        dest=run/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    common.atomic_json(ROOT/'outputs/orchestration/semantic_expert_proof_active.json',{'run_dir':str(run)})
    print('EXPERT_PROOF_DIR='+str(run),flush=True)
    common.status(run,'loading_fixed_training_examples',completed=0,total=20)
    try:
        parent=Path(cfg['parent_proof_run']);manifest=json.loads((parent/'cache_manifest.json').read_text())
        samples=[]
        for item in manifest['samples']:
            path=parent/'cache'/item['cache']
            if common.sha(path)!=item['cache_sha256']:
                raise ValueError('Proof sample cache changed')
            for key,source in item['paths'].items():
                if common.sha(source)!=item['sha256'][key]:
                    raise ValueError('Proof input changed')
            samples.append(torch.load(path,map_location='cpu',weights_only=True))
        if len(samples)!=20:
            raise ValueError('Expected fixed 20 TRAIN examples')
        original_model=common.ClassAssistedHeightNet
        common.ClassAssistedHeightNet=SemanticExpertHeightNet
        try:
            results={arm:common.train_arm(original,run,samples,torch.device('cuda'),arm) for arm in ('uniform','predicted')}
        finally:
            common.ClassAssistedHeightNet=original_model
        paired=all(results['uniform'][k]==results['predicted'][k] for k in ('initial_model_sha256','sequence_sha256'))
        protected,_=common.load_predictor(common.resolve(original['protected_checkpoint']),device='cpu')
        model=SemanticExpertHeightNet(protected).to('cuda').eval()
        payload=torch.load(run/'predicted_proof_latest.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(payload['model'],strict=True);del payload
        sensitivity=[]
        with torch.inference_mode():
            for batch in common.batches(samples,torch.device('cuda')):
                with common.amp(torch.device('cuda')):
                    a=common.forward(model,batch,'predicted')['height']
                    b=common.forward(model,batch,'uniform')['height']
                sensitivity.append(float((a-b).abs()[batch['image_valid_mask']].mean()))
        if authenticate()[2]!=binding:
            raise ValueError('Proof source/data/software binding changed')
        active_effect=sum(sensitivity)/len(sensitivity)
        report={'training_only':True,'validation_used':False,'test_used':False,'app_unchanged':True,
                'proof_weights_reused':False,'paired_identity':paired,'arms':results,
                'mean_abs_actual_vs_uniform_height_m':active_effect,
                'passes':bool(paired and all(v['passes'] for v in results.values()) and active_effect>1e-5),
                'interpretation':'Optimization/identity feasibility only. Not accuracy or generalization proof.'}
        common.atomic_json(run/'outcome.json',report);common.status(run,'complete',passes=report['passes'])
        print(json.dumps({'passes':report['passes'],'class_input_effect_m':active_effect,
                          'loss_reductions':{a:r['loss_reduction'] for a,r in results.items()}},indent=2),flush=True)
    except BaseException as error:
        common.atomic_json(run/'failure.json',{'type':type(error).__name__,'message':str(error),'no_automatic_retry':True})
        common.status(run,'failed',error=str(error));raise


if __name__=='__main__':
    lockpath=Path('D:/MSRData/experiments/class_assisted_height_development_v1/.trainer.lock')
    with lockpath.open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',NotGeoreferencedWarning);main()
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
