# Multi-region V3 launched — 19 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

- Active run: `experiments/20260919T142003929194Z_rgb_multiregion_v3`.
- Configuration/protocol: `configs/rgb_multiregion_v3.yaml`, `docs/RGB_MULTIREGION_V3_PROTOCOL.md`.
- Training is classification-only. Production app/height checkpoint and pointer remain unchanged.
- Download complete: 184 verified image/label pairs, 756,069,754 source-file bytes, on D:. 134 train / 50 development validation. All six classes have support in both splits.
- Data manifest SHA-256: `9c3b12816035cc014ec72df55a654c3920f77a3dd64836742c4d439fb6bdf5cf`.
- Preparation files: `D:/MSRData/training/oem_multiregion_v3/`; original pairs, receipts, manifest, attribution contract, label balance and alignment preview saved.
- 124 targeted tests passed before launch.
- Training-only gradient proof passed: loss 0.6420 -> 0.3276 (48.97% reduction on its fixed training batch). Proof weights discarded. This is not an accuracy result.
- Frozen V1 baseline on the new 50-image OEM development set: pooled six-class macro F1 **54.65%**; Melbourne **40.05%**, Vienna **54.37%**. Equal-region mean **47.21%**. These differ from the earlier all-six challenge and must not be compared as if they were the same benchmark.
- Verified epoch 1 progressing past batch 440/1,375. CUDA RTX 4070 active; sampled temperature 55 C. No error log entries at this check.
- Maximum eight epochs; early stopping and safety/benefit gates remain predeclared. Epoch checkpoint commits include optimizer/scheduler/RNG for compatible resume.
- Logs: `outputs/orchestration/rgb_multiregion_v3_20260919.log` and `.err.log`.
- Compact display: `scripts/watch_rgb_multiregion_v3.ps1`; terminal session 69915 was started and requested in the Codex bottom panel (UI returned queued).
- Existing heartbeat `msr-residual-accuracy-check` updated to this exact run, hourly. Notify only on new metrics, meaningful stage changes, failures or completion; pause after outcome reporting.

Recovery, only if no trainer is active: `scripts/run_rgb_multiregion_v3.ps1 -Resume .\experiments\20260919T142003929194Z_rgb_multiregion_v3`.

Do not change sealed run code/config or silently adjust batch size. Keep V1, earlier V2 experiments, the protected height checkpoint and app pointer intact. Saved development checkpoints require a later review and additional transfer validation before any app integration. No fresh reserved-final geography has been consumed.
