# Class-assisted height comparison: launch handoff

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Launched 2026-09-19 16:20 UTC. This is an experiment, not an app-model release.

## Saved runs

- Passed training-only proof: `D:/MSRData/experiments/class_assisted_height_proof_v1/20260919T161049828458Z`.
- Active development comparison: `D:/MSRData/experiments/class_assisted_height_development_v1/20260919T162009394247Z`.
- Protocol: `docs/CLASS_ASSISTED_HEIGHT_PILOT_PROTOCOL.md`.
- Registration: `outputs/orchestration/class_assisted_height_active.json`.
- Logs: `outputs/orchestration/class_assisted_height_v1.log` and `.err.log`.

## What is running

First authenticate and numerically replay DAV2 priors and cache frozen RGB-only
classification predictions for 1,050 training and 280 development images. Then
run two fresh, matched residual-height learners for two epochs each: uniform
class inputs versus predicted six-class inputs. All other settings and sample
orders match. Development evaluation stays at native resolution on fixed pixels.

The 64-update proof passed for both arms, with approximately 60% lower balanced
training loss. This is **not a 60% accuracy improvement**. Its near-identical
arm results do not establish a benefit from classification. Ground training loss
increased, so ground regression checks are especially important.

Fifteen targeted tests passed. The live production height checkpoint and pointer
were rehashed after launch and still match their protected values. Neither the
app model nor existing experiments were replaced.

## Monitoring and recovery

Hourly heartbeat `msr-residual-accuracy-check` is active for this exact
run; it reports meaningful changes and pauses after completion. No automatic
promotion or additional expensive training is authorized by the monitor.

For a compact local progress display, run:

```powershell
.\scripts\watch_class_assisted_height.ps1
```

Checkpoints save every 50 optimizer updates and at epoch boundaries. Inspect
failure evidence before attempting a compatible resume; never edit bound source
or configuration, bypass hashes, or overwrite orphan artifacts.

```powershell
.\scripts\run_class_assisted_height.ps1 -Resume D:\MSRData\experiments\class_assisted_height_development_v1\20260919T162009394247Z
```

Only resume after confirming the original trainer is no longer running. A venv
wrapper plus its Python child is one job, not duplicate training.

## Interpretation limits

Report protected-baseline, uniform-control and predicted-class results separately.
Safety and benefit are separate requirements. The 280 images are development
validation, not a fresh final test. GAMUS height safety remains untested because
units are unresolved. Per-building, app-path and final geography checks remain
prerequisites for any release, even if this bounded comparison succeeds.
