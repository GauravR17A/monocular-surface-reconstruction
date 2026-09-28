# RGB multi-region V3: completion and error analysis

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Run: `experiments/20260919T142003929194Z_rgb_multiregion_v3`.
Completed 19 September 2026 at 14:38 UTC after four epochs. The predeclared early-stop criterion fired: three epochs without better equal-region OEM macro F1 after epoch 1. No failure or recovery is indicated; error log is empty and no trainer remains active.

## Selection outcome

- **Best safety-passing checkpoint: none.** No epoch meets both benefit and safety; no default model promotion is justified.
- Best unrestricted development checkpoint: **epoch 1**. OEM pooled macro F1 **73.29%**, equal-region macro F1 **69.42%**, GAMUS macro F1 **83.20%**.
- Corresponding V1 baselines: OEM pooled **54.65%**, equal-region **47.21%**, GAMUS **83.83%**. OEM ground/road/low-vegetation F1 changed **30.77/53.63/5.37% → 58.86/64.18/62.98%**.
- Epoch 4 scored higher on pooled OEM F1 (**74.47%**) but lower on the predeclared equal-region selector (**68.56%**). Do not switch selectors after seeing results.

## What went wrong / what improved

The added regional training/augmentation package markedly improved OEM development transfer. It did not preserve all GAMUS behavior. At selected epoch 1, pooled ground F1 fell **1.99 percentage points**, low vegetation **1.76 points**, and macro F1 **0.63 points**. The worst city/class F1 regression was Philadelphia ground (**2.23 points**); Philadelphia roads also fell **1.37 points** and its road-boundary guard failed. These are genuine tradeoffs, not a universally improved classifier. Different label taxonomies and changed sampling are plausible contributors, not established causes: this run was not a controlled ablation.

Next steps: inspect fixed development examples for ground/low-vegetation and Philadelphia road edges; explicitly define a matched follow-up experiment preserving V1 behavior while learning the new regions (e.g. controlled replay/teacher regularization). Keep original acceptance thresholds and compare all classes/cities, not just pooled macro scores. Freeze the candidate before evaluating reserved geography. Do not launch a costly run just to extend an early-stopped recipe.

Melbourne/Vienna results are development scores, not final untouched tests. Rosario/Monrovia remain unconsumed. No height-accuracy improvement is established. Later user-authorized opt-in app integration and HighBuild transfer checks are documented separately in `docs/SIX_CLASS_APP_PREVIEW_2026_09_19.md`; the height/default checkpoint remains protected.

## Persistence / monitor completion check

Code snapshot (14 files), configuration, binding, baseline, metrics, all four epoch checkpoints, latest and best-development aliases are saved. Atomic commit points to epoch 4; latest matches that commit. Selected checkpoint SHA256 is `10fbb39c3d916695627e93d49f2115be18ecd163352b4daa96ba691c0090470c`; final epoch/latest SHA256 is `c732e452b880cb1c9c3e422dee2af94f75d3da7c6475d672cfe26b004278c6b2`.

Protected hashes rechecked unchanged: production height `e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144`; live pointer `a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9`; V1 RGB classifier `4940b2d4ae8817370384ccb54da9f73aee1b6bf3614435d8c077bf3e3363d021`.

Completion check GPU snapshot: 39% utilization, 1,893 MiB VRAM, 8.49 W, 48°C. No training process; the app inference server remains running. The snapshot does not attribute all GPU activity to a particular process. No thermal issue is indicated. The hourly monitor should be paused; the app need not be stopped.
