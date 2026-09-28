"""Close the sampler run and audit class sensitivity on 20 TRAIN examples only."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import default_collate
from rasterio.errors import NotGeoreferencedWarning

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('conditioning_band', ROOT/'scripts/train_height_band_sampling.py')
band = importlib.util.module_from_spec(spec); spec.loader.exec_module(band)
common, dev = band.common, band.dev
RUN = Path('D:/MSRData/experiments/height_band_sampling_v1/20260919T173241288352Z')
OLD = Path('D:/MSRData/experiments/class_assisted_height_fast_v2/20260919T164305030475Z')
DEST = ROOT/'outputs/diagnostics/height_conditioning_response_v1'
GROUPS = ('highbuild/domain/building', 'open_canopy/domain/vegetation',
          'open_canopy/domain/ground', 'open_canopy/short_vegetation',
          'highbuild/height_band/short_building', 'open_canopy/height_band/medium_vegetation',
          'highbuild/tall/building', 'open_canopy/tall/vegetation')


def verify_sampler(config):
    proof, binding = band.authenticate(ROOT/'configs/height_band_sampling_v1.json', config)
    if binding != json.loads((RUN/'binding.json').read_text()):
        raise ValueError('Finished sampler binding changed')
    expected = torch.load(common.resolve(proof['protected_checkpoint']), map_location='cpu', weights_only=False)['model']
    checks = {}
    for arm in config['arms']:
        commit = json.loads((RUN/arm/'commit.json').read_text())
        if commit['stage'] != 'evaluated' or commit['epoch'] != 4:
            raise ValueError('Sampler has not completed all epochs')
        for field in ('checkpoint', 'evaluation'):
            if common.sha(RUN/arm/commit[field]) != commit[field+'_sha256']:
                raise ValueError('Final committed artifact changed')
        p = torch.load(RUN/arm/commit['checkpoint'], map_location='cpu', weights_only=False)
        if p['binding_sha256'] != common.sha(RUN/'binding.json') or p['data_binding_sha256'] != common.sha(RUN/'data_binding.json'):
            raise ValueError('Checkpoint provenance changed')
        frozen = {k[len('protected.'):]:v for k,v in p['model_state_dict'].items() if k.startswith('protected.')}
        if frozen.keys() != expected.keys() or not all(torch.equal(v, expected[k]) for k,v in frozen.items()):
            raise ValueError('Frozen protected state changed')
        checks[arm] = {'checkpoint_sha256':commit['checkpoint_sha256'], 'evaluation_sha256':commit['evaluation_sha256'],
                       'protected_tensors_identical':True, 'protected_tensor_count':len(frozen)}
        del p, frozen
    rows = []
    for arm in config['arms']:
        for epoch in range(1, 5):
            path = RUN/arm/f'epoch_{epoch:02}_evaluation.json'
            e = json.loads(path.read_text())
            rows.append({'arm':arm, 'epoch':epoch, 'evaluation_sha256':common.sha(path),
                         'metrics':{g:e['validation']['groups'][g] for g in GROUPS},
                         'safety_pass':e['guard']['legacy_safety_pass'],
                         'failed_checks_count':len(e['guard']['failed_checks'])})
    best = min(rows, key=lambda r:sum(r['metrics'][g]['rmse_m'] for g in GROUPS[:2]))
    return proof, {'final_commits':checks, 'rows':rows, 'best_unrestricted_equal_building_canopy_rmse':
                  {'arm':best['arm'], 'epoch':best['epoch']}, 'best_eligible':None,
                  'outcome_sha256':common.sha(RUN/'outcome.json'), 'protected_bindings_verified':True}


def audit_train(proof, config):
    proof_config = json.loads((ROOT/'configs/class_assisted_height_development_v1.json').read_text())
    proof_run = common.resolve(proof_config['proof_outcome']).parent
    if common.sha(proof_run/'outcome.json') != proof_config['proof_outcome_sha256']:
        raise ValueError('Training proof selection changed')
    manifest = json.loads((proof_run/'cache_manifest.json').read_text())
    items = manifest['samples']
    if len(items) != 20:
        raise ValueError('Expected the original fixed 20 training examples')
    record_map = {r.sample_id:r for r in common.load_surface_manifest(common.resolve(proof['train_manifest']))}
    commit = json.loads((OLD/'predicted/commit.json').read_text())
    path = OLD/'predicted'/commit['checkpoint']
    if commit['epoch'] != 2 or common.sha(path) != commit['checkpoint_sha256']:
        raise ValueError('Previous class-conditioned checkpoint changed')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['binding_sha256'] != common.sha(OLD/'binding.json'):
        raise ValueError('Previous conditioning provenance mismatch')
    protected, _ = common.load_predictor(common.resolve(proof['protected_checkpoint']), device='cpu')
    model = common.ClassAssistedHeightNet(protected).to('cuda').eval()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    del payload
    sums = defaultdict(lambda:defaultdict(float)); cases = []
    device = torch.device('cuda')
    with torch.inference_mode():
        for index,item in enumerate(items):
            record = record_map[item['sample_id']]
            receipt = dev.receipts_file(common.resolve(config['cache_source_run']), 'train', record.sample_id)
            dev.verify_receipt(receipt, record, files=True)
            sample = dev.cached_sample(common.resolve(config['cache_source_run']), 'train', record)
            cropped, crop = common.crop_sample(sample, item['band'], 384, proof['seed']+index)
            if crop != item['crop']:
                raise ValueError('Fixed training crop changed')
            batch = default_collate([cropped])
            batch = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            with common.amp(device):
                actual = common.forward(model, batch, 'predicted')
                neutral = common.forward(model, batch, 'uniform')
            h = actual['height'][0,0].cpu().numpy()
            old = actual['uncorrected_height'][0,0].float().cpu().numpy()
            neutral_h = neutral['height'][0,0].cpu().numpy()
            truth = cropped['height'][0].numpy()
            domain = cropped['domain_target'].numpy()
            valid = cropped['regression_mask'][0].numpy()
            semantic = cropped['class_probabilities'].argmax(0).numpy()
            masks = band.height_masks(truth,valid,domain)
            for name,mask in masks.items():
                n = int(mask.sum())
                if not n: continue
                s = sums[name]
                s['pixels'] += n
                s['absolute_class_intervention_m'] += float(np.abs(h[mask]-neutral_h[mask]).sum(dtype=np.float64))
                s['absolute_correction_m'] += float(np.abs(h[mask]-old[mask]).sum(dtype=np.float64))
                s['signed_correction_m'] += float((h[mask]-old[mask]).sum(dtype=np.float64))
                s['raised_pixels'] += int((h[mask]>old[mask]).sum())
                s['baseline_squared_error'] += float(np.square(old[mask].astype(np.float64)-truth[mask]).sum())
                s['candidate_squared_error'] += float(np.square(h[mask].astype(np.float64)-truth[mask]).sum())
                if name in ('short_building','tall_building'):
                    s['predicted_building_pixels'] += int((semantic[mask]==1).sum())
                elif name.endswith('vegetation'):
                    s['predicted_vegetation_pixels'] += int(np.isin(semantic[mask],[4,5]).sum())
            cases.append({'sample_id':record.sample_id,'crop':crop,
                          'mean_abs_class_intervention_m':float(np.abs(h[valid]-neutral_h[valid]).mean())})
    result = {}
    for name,s in sums.items():
        n = s['pixels']
        result[name] = {'pixels':int(n), 'mean_abs_class_intervention_m':s['absolute_class_intervention_m']/n,
                        'mean_abs_correction_m':s['absolute_correction_m']/n,
                        'mean_signed_correction_m':s['signed_correction_m']/n,
                        'raised_fraction':s['raised_pixels']/n,
                        'protected_train_rmse_m':(s['baseline_squared_error']/n)**.5,
                        'candidate_train_rmse_m':(s['candidate_squared_error']/n)**.5,
                        'predicted_coarse_class_recall':((s['predicted_building_pixels']+s['predicted_vegetation_pixels'])/n) if name!='ground' else None}
    return {'training_only':True,'images':20,'validation_or_test_inference':False,'optimizer_steps':0,
            'same_checkpoint_actual_vs_uniform_intervention':True,'checkpoint_sha256':commit['checkpoint_sha256'],
            'groups':result,'cases':cases,'scope':'Small fixed training diagnostic, not generalisation evidence.'}


def main():
    if DEST.exists():
        raise FileExistsError('Do not overwrite the diagnostic; inspect existing evidence')
    torch.set_num_threads(1)
    config = json.loads((ROOT/'configs/height_band_sampling_v1.json').read_text())
    proof, verification = verify_sampler(config)
    response = audit_train(proof,config)
    band.authenticate(ROOT/'configs/height_band_sampling_v1.json',config)
    DEST.mkdir(parents=True)
    report = {'created_utc':datetime.now(timezone.utc).isoformat(),'sampler_run':str(RUN),
              'verification':verification,'training_diagnostic':response,'source_sha256':common.sha(Path(__file__))}
    common.atomic_json(DEST/'report.json',report)
    print(json.dumps({'verification':{k:v for k,v in verification.items() if k not in ('rows','final_commits')},
                      'training_groups':response['groups']},indent=2))


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',NotGeoreferencedWarning)
        main()
