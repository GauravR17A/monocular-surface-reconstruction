"""Fixed full development-set diagnosis; no selection, tuning sweep or final tests."""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import default_collate
from rasterio.errors import NotGeoreferencedWarning
from msr.evaluation.metrics import StreamingRegressionMetrics

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fast_height_audit', ROOT/'scripts/train_class_assisted_height_fast.py')
fast = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fast)
dev, common = fast.dev, fast.common


def strata(sample):
    valid = sample['regression_mask'][0].numpy().astype(bool)
    target = sample['height'][0].numpy()
    domain = sample['domain_target'].numpy()
    masks = {'ground': valid & (domain == 0),
             'short_vegetation': valid & (domain == 2) & (target > 0) & (target <= 2),
             'medium_vegetation': valid & (domain == 2) & (target > 2) & (target < 15),
             'tall_vegetation': valid & (domain == 2) & (target >= 15),
             'short_building': valid & (domain == 1) & (target < 20),
             'tall_building': valid & (domain == 1) & (target >= 20)}
    return target, masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('D:/MSRData/experiments/class_assisted_height_fast_v2/20260919T164305030475Z'))
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/diagnostics/class_assisted_low_surfaces_v1.json')
    args = parser.parse_args()
    config = json.loads((args.run/'config.json').read_text())
    proof = json.loads(common.resolve(config['proof_config']).read_text())
    current = fast.authenticate(ROOT/'configs/class_assisted_height_fast_v2.json', config, proof)
    if current != json.loads((args.run/'binding.json').read_text()):
        raise ValueError('Completed experiment binding mismatch')
    commit = json.loads((args.run/'predicted/commit.json').read_text())
    checkpoint = args.run/'predicted'/commit['checkpoint']
    if commit['stage'] != 'evaluated' or common.sha(checkpoint) != commit['checkpoint_sha256']:
        raise ValueError('Unverified candidate checkpoint')
    torch.set_num_threads(1)
    device = torch.device('cuda')
    protected, _ = common.load_predictor(common.resolve(proof['protected_checkpoint']), device='cpu')
    model = common.ClassAssistedHeightNet(protected).to(device).eval()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    del payload
    records = common.load_surface_manifest(common.resolve(config['validation_manifest']))
    cache = common.resolve(config['parent_run'])
    loader = fast.OrderedPrefetch(records, lambda r: dev.cached_sample(cache, 'validation', r), 2, 2)
    scores = defaultdict(lambda: {'protected': StreamingRegressionMetrics(), 'candidate': StreamingRegressionMetrics(),
                                 'correction_sum': 0., 'raised_pixels': 0, 'worsened_pixels': 0, 'pixels': 0})
    cases = []
    try:
        with torch.inference_mode():
            for index in range(len(records)):
                record, sample = loader.pop()
                batch = {k:v.to(device) if isinstance(v, torch.Tensor) else v for k,v in default_collate([sample]).items()}
                with common.amp(device):
                    out = common.forward(model, batch, 'predicted')
                candidate = out['height'][0,0].float().cpu().numpy()
                original = out['uncorrected_height'][0,0].float().cpu().numpy()
                target, masks = strata(sample)
                row = {'sample_id':record.sample_id, 'region':record.region, 'groups':{}}
                for name, mask in masks.items():
                    n = int(mask.sum())
                    if not n:
                        continue
                    correction = candidate[mask] - original[mask]
                    old_error = np.abs(original[mask] - target[mask])
                    new_error = np.abs(candidate[mask] - target[mask])
                    raised, worse = int((correction > 0).sum()), int((new_error > old_error).sum())
                    for key in (record.landscape+'/'+name, record.landscape+'/region/'+record.region+'/'+name):
                        entry = scores[key]
                        entry['protected'].update(original, target, mask)
                        entry['candidate'].update(candidate, target, mask)
                        entry['correction_sum'] += float(correction.sum(dtype=np.float64))
                        entry['raised_pixels'] += raised
                        entry['worsened_pixels'] += worse
                        entry['pixels'] += n
                    row['groups'][name] = {'pixels':n, 'mean_correction_m':float(correction.mean()),
                        'mae_change_m':float(new_error.mean()-old_error.mean()), 'raised_fraction':raised/n, 'worsened_fraction':worse/n}
                cases.append(row)
                if (index+1) % 50 == 0 or index+1 == len(records):
                    print(f'Diagnosed {index+1}/{len(records)} fixed development images', flush=True)
    finally:
        loader.close()
    groups = {key:{'protected':v['protected'].compute(), 'candidate':v['candidate'].compute(),
                  'mean_correction_m':v['correction_sum']/v['pixels'], 'raised_fraction':v['raised_pixels']/v['pixels'],
                  'worsened_fraction':v['worsened_pixels']/v['pixels']} for key,v in scores.items()}
    report = {'source_run':str(args.run), 'checkpoint_sha256':commit['checkpoint_sha256'], 'images':len(records),
              'development_only':True, 'final_tests_used':False, 'groups':groups, 'cases':cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.atomic_json(args.output, report)
    print(json.dumps({k:v for k,v in groups.items() if '/region/' not in k}, indent=2))


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        main()
