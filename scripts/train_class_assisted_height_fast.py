"""Authenticated performance-only continuation of the locked paired experiment.

Reuse verified native inputs, four-worker ordered CPU prefetch, pinned transfers.
No optimizer/batch/loss/model/sampler change. Parent remains immutable and a
checkpoint import is recorded explicitly rather than bypassing resume bindings.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('height_dev_original', ROOT/'scripts/train_class_assisted_height.py')
dev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dev)
common = dev.common


class OrderedPrefetch:
    """Bounded memory, deterministic delivery, no CUDA inside loader threads."""
    def __init__(self, items, load, workers, depth):
        if workers < 1 or depth < 1:
            raise ValueError('Positive workers and prefetch depth required')
        self.items = iter(items)
        self.load = load
        self.depth = depth
        self.queue = deque()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='height-input')
        self.fill()

    def fill(self):
        while len(self.queue) < self.depth:
            try:
                item = next(self.items)
            except StopIteration:
                break
            self.queue.append((item, self.pool.submit(self.load, item)))

    def pop(self):
        if not self.queue:
            raise StopIteration
        item, future = self.queue.popleft()
        result = future.result()
        self.fill()
        return item, result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.queue.clear()


def authenticate(config_path, config, proof):
    binding = dev.binding(config_path, config, proof)
    parent = common.resolve(config['parent_run'])
    for name, digest in config['parent_bindings'].items():
        if common.sha(parent/name) != digest:
            raise ValueError(f'Parent binding changed: {name}')
    if common.sha(parent/config['parent_checkpoint']) != config['parent_checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed')
    old_binding = json.loads((parent/'binding.json').read_text())
    if old_binding['code'] != binding['code']:
        raise ValueError('Original numerical/model/data source changed')
    for key in ('torch', 'transformers', 'python', 'device', 'proof_config_sha256'):
        if old_binding[key] != binding[key]:
            raise ValueError(f'Original runtime changed: {key}')
    old_config = json.loads((parent/'config.json').read_text())
    for key in ('epochs', 'pairs_per_epoch', 'guards', 'validation_manifest_sha256', 'checkpoint_every_updates'):
        if old_config[key] != config[key]:
            raise ValueError(f'Numerical experiment changed: {key}')
    if common.sha(common.resolve(config['loader_benchmark'])) != config['loader_benchmark_sha256']:
        raise ValueError('Throughput/equivalence proof changed')
    benchmark = json.loads(common.resolve(config['loader_benchmark']).read_text())
    if not benchmark['training_only'] or not all(x['byte_identical'] for x in benchmark['results']):
        raise ValueError('Loader equivalence was not established')
    binding['code']['scripts/train_class_assisted_height_fast.py'] = common.sha(Path(__file__))
    binding['performance'] = config['performance']
    binding['parent'] = config['parent_bindings']
    binding['parent_checkpoint_sha256'] = config['parent_checkpoint_sha256']
    return binding


def check_reused_inputs(parent, sets, run):
    recorded = json.loads((parent/'data_binding.json').read_text())
    total = sum(len(records) for records in sets.values())
    done = 0
    for split, records in sets.items():
        for record in records:
            path = dev.receipts_file(parent, split, record.sample_id)
            if common.sha(path) != recorded['receipts'][split][record.sample_id]:
                raise ValueError('Receipt binding changed')
            dev.verify_receipt(path, record, files=True)
            done += 1
            if done % 50 == 0 or done == total:
                common.status(run, 'checking_reused_inputs', completed=done, total=total)
    if recorded['counts'] != {key: len(value) for key,value in sets.items()} or recorded['exact_rgb_overlap'] != 0:
        raise ValueError('Original data split binding changed')


def import_parent(config, run):
    parent = common.resolve(config['parent_run'])
    for name in ('data_binding.json', 'baseline.json', 'baseline_binding.json'):
        shutil.copy2(parent/name, run/name)
    directory = run/'uniform'
    directory.mkdir()
    checkpoint = parent/config['parent_checkpoint']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['binding_sha256'] != config['parent_bindings']['binding.json'] or payload['data_binding_sha256'] != config['parent_bindings']['data_binding.json']:
        raise ValueError('Parent checkpoint provenance mismatch')
    if (payload['arm'], payload['epoch'], payload['batch_done']) != ('uniform', 1, 600):
        raise ValueError('Unexpected parent training cursor')
    origin = {'parent_run': str(parent), 'checkpoint_sha256': common.sha(checkpoint),
              'original_binding_sha256': payload['binding_sha256'], 'original_data_binding_sha256': payload['data_binding_sha256'],
              'authorized_change': 'CPU loading/prefetch only; math and optimizer/RNG state unchanged'}
    payload['binding_sha256'] = common.sha(run/'binding.json')
    payload['data_binding_sha256'] = common.sha(run/'data_binding.json')
    payload['performance_continuation_origin'] = origin
    target = directory/checkpoint.name
    common.atomic_save(target, payload)
    common.atomic_json(directory/'commit.json', {'checkpoint':target.name, 'checkpoint_sha256':common.sha(target),
                                               'epoch':1, 'batch_done':600, 'stage':'trained'})
    shutil.copy2(parent/'uniform/initial.json', directory/'initial.json')
    common.atomic_json(run/'continuation_origin.json', origin)
    del payload


class FastInputs:
    """Adapter around the original loop; no change to its gradient operations."""
    def __init__(self, parent, settings):
        self.parent = parent
        self.settings = settings
        self.receipts = dev.receipts_file
        self.cpu_batch = dev.train_batch
        self.cached = dev.cached_sample
        self.plan_function = dev.epoch_plan
        self.evaluator = dev.evaluate
        self.plan = None
        self.train_loader = None
        self.validation_loader = None

    def install(self):
        dev.receipts_file = lambda run, split, sample_id: self.receipts(self.parent, split, sample_id)
        dev.epoch_plan = self.epoch_plan
        dev.train_batch = self.train_batch
        dev.cached_sample = self.cached_sample
        dev.evaluate = self.evaluate

    def epoch_plan(self, records, epoch, seed):
        if self.train_loader is not None:
            self.train_loader.close()
            self.train_loader = None
        self.plan = self.plan_function(records, epoch, seed)
        return self.plan

    def train_batch(self, run, pair, patch, device):
        if self.train_loader is None:
            if self.plan is None:
                raise ValueError('Missing declared epoch plan')
            start = next(i for i,p in enumerate(self.plan) if p[2] == pair[2])
            self.train_loader = OrderedPrefetch(self.plan[start:],
                lambda p: self.cpu_batch(run, p, patch, torch.device('cpu')),
                self.settings['training_loader_workers'], self.settings['training_prefetch_batches'])
        expected, batch = self.train_loader.pop()
        if expected != pair:
            raise ValueError('Prefetch changed the declared sample order')
        return {key: value.pin_memory().to(device, non_blocking=True) if isinstance(value,torch.Tensor) else value
                for key,value in batch.items()}

    def cached_sample(self, run, split, record):
        if split == 'validation' and self.validation_loader is not None:
            expected, sample = self.validation_loader.pop()
            if expected.sample_id != record.sample_id:
                raise ValueError('Validation prefetch order changed')
            return sample
        return self.cached(run, split, record)

    def evaluate(self, model, run, records, device, arm, epoch, *, baseline=False):
        if self.validation_loader is not None:
            raise ValueError('Nested evaluation')
        self.validation_loader = OrderedPrefetch(records, lambda r:self.cached(run,'validation',r),
            self.settings['validation_loader_workers'], self.settings['validation_prefetch_images'])
        started = time.perf_counter()
        try:
            result = self.evaluator(model,run,records,device,arm,epoch,baseline=baseline)
            print(f'{arm} epoch {epoch} evaluation: {time.perf_counter()-started:.1f}s', flush=True)
            return result
        finally:
            self.validation_loader.close()
            self.validation_loader = None

    def close(self):
        if self.train_loader is not None:
            self.train_loader.close()
        if self.validation_loader is not None:
            self.validation_loader.close()


def execute(args, config, proof, binding):
    parent = common.resolve(config['parent_run'])
    run = common.resolve(args.resume) if args.resume else common.resolve(config['output_root'])/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if args.resume:
        if run.parent != common.resolve(config['output_root']) or json.loads((run/'binding.json').read_text()) != binding:
            raise ValueError('Fast-run resume binding mismatch')
    else:
        run.mkdir(parents=True, exist_ok=False)
        common.atomic_json(run/'config.json', config)
        common.atomic_json(run/'binding.json', binding)
        for path in binding['code']:
            target = run/'source_snapshot'/path
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/path, target)
        import_parent(config,run)
    common.atomic_json(ROOT/'outputs/orchestration/class_assisted_height_active.json',
                       {'run_dir':str(run), 'pid':os.getpid(), 'started_utc':datetime.now(timezone.utc).isoformat(), 'parent_run':str(parent)})
    print(f'FAST_RUN_DIR={run}', flush=True)
    adapter = None
    try:
        sets = {'train':common.load_surface_manifest(common.resolve(proof['train_manifest'])),
                'validation':common.load_surface_manifest(common.resolve(config['validation_manifest']))}
        check_reused_inputs(parent,sets,run)
        if common.sha(run/'data_binding.json') != config['parent_bindings']['data_binding.json']:
            raise ValueError('Continued data binding changed')
        if common.sha(run/'baseline.json') != config['parent_bindings']['baseline.json']:
            raise ValueError('Continued baseline changed')
        baseline = json.loads((run/'baseline.json').read_text())
        adapter = FastInputs(parent,config['performance'])
        adapter.install()
        timings = {}
        for arm in ('uniform','predicted'):
            if authenticate(args.config,config,proof) != binding:
                raise ValueError('Bindings changed before arm')
            started = time.perf_counter()
            dev.run_arm(config,proof,run,sets,torch.device('cuda'),arm,baseline)
            timings[arm] = time.perf_counter()-started
        pairs = []
        for epoch in (1,2):
            a = json.loads((run/'uniform'/f'epoch_{epoch:02}_evaluation.json').read_text())
            b = json.loads((run/'predicted'/f'epoch_{epoch:02}_evaluation.json').read_text())
            if json.loads((run/'uniform'/f'epoch_{epoch:02}_sampling.json').read_text()) != json.loads((run/'predicted'/f'epoch_{epoch:02}_sampling.json').read_text()):
                raise ValueError('Paired sampling differs')
            benefit = dev.paired_benefit(baseline,a['validation'],b['validation'],config['guards'])
            pairs.append({'epoch':epoch, **benefit, 'candidate_guard':b['guard'], 'control_guard':a['guard'],
                          'eligible_for_legacy_review':benefit['benefit_pass'] and b['guard']['legacy_safety_pass']})
        if json.loads((run/'uniform/initial.json').read_text()) != json.loads((run/'predicted/initial.json').read_text()):
            raise ValueError('Initial model states differ')
        check_reused_inputs(parent,sets,run)
        if authenticate(args.config,config,proof) != binding:
            raise ValueError('Final bindings changed')
        outcome = {'stage':'complete', 'paired_epochs':pairs, 'release_eligible':False, 'gamus_height_evaluated':False,
                   'app_unchanged':True, 'parent_run':str(parent), 'timings_seconds':timings,
                   'best_eligible_epoch':next((r['epoch'] for r in sorted(pairs,key=lambda r:sum(x['candidate_rmse_m'] for x in r['priorities'].values())) if r['eligible_for_legacy_review']),None),
                   'interpretation':'Performance-only development continuation, not final unseen-generalization evidence. No automatic promotion.'}
        common.atomic_json(run/'outcome.json',outcome)
        common.status(run,'complete',best_eligible_epoch=outcome['best_eligible_epoch'])
        print(json.dumps(outcome,indent=2),flush=True)
    except BaseException as error:
        common.atomic_json(run/'failure.json',{'type':type(error).__name__,'message':str(error),'inspect_before_resume':True})
        common.status(run,'failed',error=str(error))
        raise
    finally:
        if adapter is not None:
            adapter.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/class_assisted_height_fast_v2.json')
    parser.add_argument('--resume')
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    proof=json.loads(common.resolve(config['proof_config']).read_text())
    current=authenticate(args.config,config,proof)
    if args.preflight:
        print(json.dumps({'passed':True,'bound_sources':len(current['code']),'parent_checkpoint_verified':True}))
        return
    torch.set_num_threads(config['performance']['cpu_tensor_threads'])
    torch.set_num_interop_threads(1)
    # Share the original family lock so old and fast trainers cannot compete.
    lock_path=common.resolve(config['parent_run']).parent/'.trainer.lock'
    with lock_path.open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',NotGeoreferencedWarning)
                execute(args,config,proof,current)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


if __name__ == '__main__':
    main()
