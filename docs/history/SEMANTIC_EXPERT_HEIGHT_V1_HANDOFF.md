# Active six-class guided height experiment

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Launched 19 September 2026 at 18:07:01 UTC. This is a launch record, not a
claim that height accuracy has improved or that the app model has changed.

- Exact run: `D:/MSRData/experiments/semantic_expert_height_v1/20260919T180701956748Z`
- Protocol: `docs/SEMANTIC_EXPERT_HEIGHT_V1_PROTOCOL.md`
- Configuration: `configs/semantic_expert_height_v1.json`
- Log/error log: `outputs/orchestration/semantic_expert_height_v1.log` and `.err.log`
- Live window: `scripts/class_assisted_dashboard.py`
- Registration: `outputs/orchestration/class_assisted_height_active.json`
- Launcher: `scripts/run_semantic_expert_height.ps1`

## Completed work and evidence

The preceding height-band sampling run finished with NO safety-and-benefit
eligible candidate. Its outcome and diagnostic are documented in
`docs/HEIGHT_BAND_SAMPLING_V1_OUTCOME.md` and
`outputs/diagnostics/height_conditioning_response_v1/report.json`. Both final
committed checkpoints/evaluations were hash verified; all protected tensors
were compared with production and matched. Protected pointer, V1 and V3 hashes
were verified too. Old experiments remain intact.

New model: `src/msr/models/semantic_expert_height.py`. Six correction
outputs, mixed at native pixels by actual frozen classifier probabilities in B.
A has identical capacity but neutral mixing. No class reference masks or target
heights enter inference; no class forces zero height. Both keep the ORIGINAL
sampler/loss/initialization/budget, four epochs each, 600 source pairs per epoch.

The bounded train-only proof at
`D:/MSRData/experiments/semantic_expert_height_proof_v1/20260919T180155574728Z`
passed initialization/semantic/frozen-state checks and finite learning. Training
loss decreased 54.13% in A and 53.30% in B on 20 fixed training examples. This is
NOT accuracy or generalisation. Proof weights were not reused. Actual-vs-uniform
class routing changed the learned B proof heights by mean 0.361m on that sample.

29 targeted tests passed; 24 training source files were bound at preflight.
The protected full-development baseline matched exactly, and uniform/control
epoch 1 training was observed at handoff preparation. Data validation covers all
source and cached-prior/classifier receipts. Only one actual GPU trainer is
allowed. Files/checkpoints and source snapshots are saved on D:.

## Monitoring and recovery

The existing `msr-residual-accuracy-check` heartbeat is ACTIVE hourly for
this exact run, quiet without meaningful changes. It must compare matched epochs,
preserve all original category/region/bias/correlation/R2 guards, and report best
eligible separately from unrestricted results. 160 HighBuild +120 OpenCanopy
images are DEVELOPMENT, not untouched final tests. Six correction routes are not
six independently validated height categories. No GAMUS height claim or automatic
app promotion.

On a visible failure, inspect actual processes/logs first. At most one compatible
resume after inspection, without source/hash/batch changes:

```powershell
.\scripts\run_semantic_expert_height.ps1 -Resume 'D:\MSRData\experiments\semantic_expert_height_v1\20260919T180701956748Z'
```

Use FileShare.ReadWrite plus FileShare.Delete for active Windows files; the
dashboard's `shared_bytes`/`read_json` helpers implement this. Never rerun a failed
proof or overwrite orphan checkpoints. At completion verify final hashes,
produce error analysis, pause the monitor and do not automatically start another
expensive run. Current prototype models remain unchanged.
