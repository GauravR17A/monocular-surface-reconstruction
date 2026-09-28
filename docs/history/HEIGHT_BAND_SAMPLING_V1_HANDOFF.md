# Active height-band sampling experiment

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Started 19 September 2026 at 17:32:41 UTC. This handoff is a launch record,
not an accuracy or completion claim; read the live status and committed reports.

## Exact run and saved work

- Run: `D:/MSRData/experiments/height_band_sampling_v1/20260919T173241288352Z`
- Protocol: `docs/HEIGHT_BAND_SAMPLING_V1_PROTOCOL.md`
- Configuration: `configs/height_band_sampling_v1.json`
- Sampler audit/plans: `D:/MSRData/experiments/height_band_sampling_preparation_v1/`
- Log: `outputs/orchestration/height_band_sampling_v1.log`
- Errors: `outputs/orchestration/height_band_sampling_v1.err.log`
- Dashboard registration: `outputs/orchestration/class_assisted_height_active.json`

The run directory contains copied source, bound configuration, data/cache hashes,
immutable sampling plans, the recomputed protected baseline, and versioned
checkpoints/atomic commits under `control` and `balanced`. Do not edit bound
training sources or recipes while this experiment is active.

## What this tests

Four epochs per arm with the same model, initialization, original loss and
training budget. Both use uniform class inputs. Only training sampling changes:
the candidate deliberately includes more labelled tall buildings and short
vegetation. All 1,050 training images appear within each two-epoch cycle.
The exposure audit is not proof of better accuracy.

Validation uses all 160 urban and 120 forest DEVELOPMENT scenes at native
resolution after every epoch. Compare the candidate with the matching control
epoch and protected baseline. Safety and improvement are separate requirements;
the existing app model must not be replaced automatically. Reserved final-test
geography and GAMUS height claims are outside this experiment.

## Verified at handoff preparation

- Preparation and preflight passed; original baseline metrics matched exactly.
- Control epochs 1 and 2 committed; epoch 3 training observed.
- Control epochs 1 and 2 did not pass all safety gates. Candidate results were
  not yet available at that observation.
- 24 targeted sampler, runner, loss, gate, prefetch and dashboard tests passed.
- The native live dashboard was opened and visually checked with both arm
  progress bars, epoch results and GPU telemetry visible.
- Existing `msr-residual-accuracy-check` heartbeat reactivated hourly
  for this exact run. It stays quiet without meaningful changes, reports final
  evidence and pauses itself after completion. No further experiment or app
  promotion is authorized by that monitor.

## Failure/recovery boundary

Inspect failure/status, logs and actual trainer processes first. A venv wrapper
and its Python child count as one job. Do not start a competing GPU job.
Only one compatible recovery attempt is permitted by the monitor:

```powershell
.\scripts\run_height_band_sampling.ps1 -Resume 'D:\MSRData\experiments\height_band_sampling_v1\20260919T173241288352Z'
```

Never bypass hashes, change batch/loss/sampling rules or overwrite orphan
artifacts. Active Windows files must be read with FileShare.ReadWrite and
FileShare.Delete. `class_assisted_dashboard.shared_bytes` provides this.

At completion, verify integrity and report best eligible separately from best
unrestricted checkpoint. If none passes both safety and benefit, say so plainly
and preserve the rejected evidence for error analysis.
