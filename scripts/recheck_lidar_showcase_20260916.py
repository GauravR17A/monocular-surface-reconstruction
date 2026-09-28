"""Fresh, isolated inference on the 15 available DFC19 urban showcase candidates."""
from pathlib import Path
import json, hashlib
from msr.api.app import InferenceRuntime

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data/demo_sources/DFC19_candidate_selection'
OUT = ROOT / 'outputs/reports/lidar_recheck_20260916'
OUT.mkdir(parents=True, exist_ok=True)
checkpoint = Path((ROOT/'outputs/runtime/showcase_checkpoint.txt').read_text(encoding='utf-8-sig').strip())
runtime = InferenceRuntime(checkpoint=checkpoint, relative_model=str(ROOT/'models/foundation/depth-anything-v2-small-hf'))
rows = []
for p in sorted((SRC/'rgb').glob('*.tif')):
    dest = OUT/p.stem
    result = runtime.predict(p, dest, reference_path=SRC/'depth'/p.name, reference_kind='ndsm')
    row = {'sample': p.stem, 'metrics': json.loads((dest/'validation_metrics.json').read_text()), 'source_rgb':str(p),
           'reference':str(SRC/'depth'/p.name), 'output':str(dest)}
    rows.append(row)
    (OUT/'audit.json').write_text(json.dumps({'checkpoint':str(checkpoint),
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'disclosure':'Selected after inspecting results; not a blind test. Local DFC19 derivative; sharing rights require review.',
        'candidates':rows}, indent=2))
    print(json.dumps(row), flush=True)
