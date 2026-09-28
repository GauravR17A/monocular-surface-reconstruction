# Faster class-assisted height comparison

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Why this version exists

The user explicitly requested stopping and rerunning the slow pipeline using
the laptop more effectively. Stopped only the registered trainer, after control A
finished epoch 1 and saved `epoch_01_batch_0600.pt`; validation was interrupted.
The original experiment remains intact and must not be restarted by its monitor.

Parent: `D:/MSRData/experiments/class_assisted_height_development_v1/20260919T162009394247Z`.
Active: `D:/MSRData/experiments/class_assisted_height_fast_v2/20260919T164305030475Z`.

## Measured bottleneck and change

The old loop reloaded full native rasters, decompressed six probability grids,
constructed supervision tensors, and cropped them synchronously for each batch.
The GPU waited while the CPU did this work.

A training-only 12-batch check compared the same tensor values and order:

| Loader | Measured batches/s | Input tensor identity |
|---|---:|---|
| Original, 16 CPU tensor threads, serial loading | 4.71 | Reference |
| One CPU tensor thread, serial loading | 4.76 | Exact |
| One CPU tensor thread, four loader workers | 12.28 | Exact |

This is a ~2.6x loader benchmark gain, **not** a promised end-to-end speedup or
accuracy improvement. Measurement includes a small, cache-warmed sample;
real epoch timings remain the relevant whole-run evidence.

Initial live throughput check: successive 100-batch checkpoint intervals took
about 34 seconds in the old A epoch 1, versus 11.6 seconds in fast A epoch 2
(first five checkpoints). This is roughly 3x faster observed training throughput,
not a same-input whole-run benchmark; validation/checkpoint phases still vary.

Fast execution uses four ordered training loader workers with four prefetched
batches, two validation workers/images, and pinned nonblocking GPU transfers.
Memory is bounded; this 16 GB laptop had only about 4 GB RAM free when inspected.
No overclock, temperature-limit change or unrelated-process shutdown was done.
Original numerical model/loss/gradient loop, batch 2 + accumulation 2, optimizer,
crop coordinates, sample order, precision and evaluation rules are unchanged.

## Preserved training and integrity

The new run explicitly imports the authenticated A epoch-1 model, optimizer and
RNG state. It finishes that validation, trains A epoch 2, then B epochs 1 and 2.
The original checkpoint is never rewritten. `continuation_origin.json` records
the old bindings and checkpoint hash; the imported checkpoint is a new artifact
bound to the faster execution source/config. This is a declared continuation,
not a silent override of an old checkpoint's resume checks.

All 1,330 original cached inputs are reused read-only and reauthenticated against
original files and receipt hashes. DAV2 predictions and classification maps are
not regenerated. Original baseline metrics/support are reused with pinned hashes.
Both arms share the same initial tensors and matched sampling plans, checked at
completion. Protected production model and pointer are unchanged.

Six dashboard/prefetch tests passed. The executable preflight authenticated 18
source files and the parent checkpoint. Loader equality was checked on 12 real
training batches. These are implementation checks, not new accuracy evidence.

## Watching and recovery

- Dashboard: `.venv/Scripts/pythonw.exe scripts/class_assisted_dashboard.py`.
- Logs: `outputs/orchestration/class_assisted_height_fast_v2.log` and `.err.log`.
- Registration: `outputs/orchestration/class_assisted_height_active.json`.
- Detailed reports: `outputs/reports/class_assisted_height/<run ID>/`.
- Reports include text and full-group CSV; no scores are fabricated while waiting.
- Hourly heartbeat `msr-residual-accuracy-check` now monitors fast v2 only.

After inspecting an actual failure and confirming no trainer remains:

```powershell
.\scripts\run_class_assisted_height_fast.ps1 -Resume D:\MSRData\experiments\class_assisted_height_fast_v2\20260919T164305030475Z
```

Never alter the active bound source/config or bypass a hash mismatch. The old and
fast trainers share one lock. Checkpoints remain versioned and save every 50
optimizer updates. Do not replay the old run, proof or completed updates.

## Honest interpretation

This pilot covers corrected urban building labels and forest/canopy/ground labels
on the fixed development set. It does not certify broad hills/plains, water/road
heights, all sensors/resolutions, or genuinely new geography. GAMUS height units
remain unresolved. Require both legacy safety and matched benefit; neither is
automatic app-release permission. Same-epoch improvement on building OR canopy
must exceed both 0.5 m and 5%, beat the control, and pass all other safety checks.
No final reserved geography is consumed by this run.
