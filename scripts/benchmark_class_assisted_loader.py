"""Training-input throughput diagnostic; no model updates or validation reads."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paired_dev', ROOT/'scripts/train_class_assisted_height.py')
dev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dev)


def cpu_batch(run, pair, patch):
    return dev.train_batch(run, pair, patch, torch.device('cpu'))


def digest(batch):
    return dev.common.state_digest({k:v for k,v in batch.items() if isinstance(v,torch.Tensor)})


def main():
    run = Path(json.loads((ROOT/'outputs/orchestration/class_assisted_height_active.json').read_text())['run_dir'])
    config = json.loads((run/'config.json').read_text())
    proof = json.loads(dev.common.resolve(config['proof_config']).read_text())
    records = dev.common.load_surface_manifest(dev.common.resolve(proof['train_manifest']))
    pairs = dev.epoch_plan(records, 2, proof['seed'])[:12]
    patch = proof['training']['patch_size']
    results = []
    reference = None
    for threads, workers in [(torch.get_num_threads(), 1), (1, 1), (1, 4)]:
        torch.set_num_threads(threads)
        started = time.perf_counter()
        if workers == 1:
            batches = [cpu_batch(run, p, patch) for p in pairs]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                batches = list(executor.map(lambda p: cpu_batch(run,p,patch), pairs))
        elapsed = time.perf_counter()-started
        signatures = [digest(b) for b in batches]
        if reference is None:
            reference = signatures
        if signatures != reference:
            raise RuntimeError('Input tensor equivalence failed')
        result = {'torch_cpu_threads':threads, 'loader_workers':workers, 'batches':len(pairs),
                  'seconds':elapsed, 'batches_per_second':len(pairs)/elapsed, 'byte_identical':True}
        results.append(result)
        print(json.dumps(result), flush=True)
        del batches
    output = ROOT/'outputs/diagnostics/class_assisted_loader_benchmark_20260919.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    dev.common.atomic_json(output, {'parent_run':str(run), 'training_only':True, 'results':results})


if __name__ == '__main__':
    main()
