# Residual-height V1: completed outcome

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

15 September 2026. Run: `experiments/20260915T124056Z_residual_height_v1`.

**Decision: retain all five checkpoints as rejected development candidates. Do not change the app's active model.** The run early-stopped normally after epoch 5; no checkpoint passed every registered guard. Epoch 5 had the lowest selection score, not an eligible "best" checkpoint.

## What changed across the run

All values below are final predicted-height RMSE, not expert-head scores. Lower is better. GAMUS units remain assumed metres in the run's contract; this audit has not resolved the publisher-unit question.

| Completed epoch | HighBuild measured buildings | OpenCanopy vegetation | OpenCanopy ground | GAMUS domain-macro | Material object improvement | All guards |
|---|---:|---:|---:|---:|---|---|
| Protected baseline | 13.2426 | 8.3610 | 4.8944 | 5.2248 | Comparator | Comparator |
| 1 | 13.0781 | 7.9645 | 5.1504 | 5.0630 | No | Fail |
| 2 | 13.0710 | 7.9389 | 5.0605 | 5.0054 | Yes | Fail |
| 3 | 13.1037 | 7.8509 | 5.0697 | 4.9677 | Yes | Fail |
| 4 | 13.0162 | 7.8613 | 4.9116 | 4.8916 | Yes | Fail |
| 5 | 13.0121 | 7.8565 | 4.8741 | 4.8495 | Yes | Fail |

Epoch 5 reduces building RMSE by **1.74%** and vegetation RMSE by **6.03%**. These are error reductions on development validation, not accuracy percentages or unseen-geography results. OpenCanopy vegetation's lowest RMSE was epoch 3, so the last checkpoint is not best for every metric.

## The important findings

### Buildings: errors are concentrated in tall structures

- Reference heights of at least 20 m account for **17.64% of measured building pixels but 89.31% of baseline squared error**. At epoch 5 they still account for 87.73% of squared error.
- Tall-building RMSE: **29.797 -> 29.017 m**. Correlation: **-0.188807 -> -0.188944**. The correlation regression is tiny, but the performance level itself is poor.
- Overall building MAE barely changes: **5.361 -> 5.328 m**. Bias improves substantially: **-2.788 -> -1.018 m**.
- Decomposing `RMSE^2 = bias^2 + centered error variance`, centered error RMSE is **12.946 -> 12.972 m**, slightly worse. The aggregate RMSE gain is therefore accounted for by reduced bias, not reduced centered error spread. This is descriptive arithmetic, not proof of what visual features the network learned; no validation-derived correction has been applied.

These are pixel metrics on measured COCO-intersection support, **not per-building instance scores**. Before more height training, audit measured training buildings by height band, city and annotation provenance; verify that tall-reference outliers are aligned and trustworthy, then test height-balanced and per-building supervision separately.

### Canopy: genuine aggregate improvement, uneven local behavior

- Vegetation RMSE **8.361 -> 7.857 m**, MAE **6.505 -> 6.102 m**, correlation **0.411 -> 0.458**, R2 **-0.174 -> -0.036**.
- Ground RMSE **4.894 -> 4.874 m**, MAE **2.896 -> 2.750 m**, correlation **0.502 -> 0.548**, R2 **-0.088 -> -0.079**.
- **68 of 115 saved OpenCanopy region groups improve RMSE; 47 worsen.** These groups are not 115 independent geographies or individual trees; the validation set contains 120 chips and shares source mosaics with training.
- The largest region-group regression is `open_canopy_r143_1270`: **3.081 -> 4.437 m RMSE**, with positive bias increasing **1.049 -> 2.079 m**. An improving overall number does not establish consistent improvement everywhere.

### GAMUS: object gains coexist with ground damage

- Overall RMSE **7.121 -> 6.189**, correlation **0.366 -> 0.522**, R2 **0.005 -> 0.248**.
- Building RMSE **5.250 -> 4.917**; vegetation **8.897 -> 7.585**.
- Ground RMSE worsens **1.527 -> 2.046**, MAE **0.705 -> 1.105**, bias **+0.458 -> +0.881**, correlation **0.168 -> 0.124**, R2 **-1.810 -> -4.044**.
- Tall-vegetation correlation also worsens **0.057 -> -0.111**, despite lower tall-vegetation RMSE. Raising average heights has not recovered correct ordering within this subset.

## Region results expose trade-offs

| HighBuild city | Baseline RMSE | Epoch 5 RMSE | Baseline MAE | Epoch 5 MAE |
|---|---:|---:|---:|---:|
| Copenhagen | 6.182 | 5.185 | 4.283 | 3.445 |
| Lyon | 14.244 | 14.302 | 7.072 | 7.426 |
| Strasbourg | 18.563 | 18.263 | 5.587 | 5.590 |
| Munich | 9.302 | 9.118 | 4.433 | 4.694 |

Three cities improve RMSE, but only Copenhagen improves MAE. GAMUS DC improves RMSE **10.998 -> 9.417** and Philadelphia **3.116 -> 2.983**; Philadelphia MAE nevertheless worsens **1.674 -> 1.870**. These are existing development-city aggregates, not fresh external tests.

## Why every epoch was rejected

- All five: GAMUS ground RMSE, MAE, absolute bias, correlation and R2 regress beyond their registered limits; GAMUS tall-vegetation correlation regresses.
- Epochs 2-5: HighBuild tall-building correlation also fails the strict no-regression check.
- Epochs 1-3: OpenCanopy ground has additional failed checks. Epoch 4 still fails ground R2. Those OpenCanopy failures recover by epoch 5.
- Epoch 1 does not reach material priority-object improvement; epochs 2-5 do. A material gain alone does not override failed protection checks.

The exact per-epoch failures and paired RMSE/MAE/bias/correlation/R2 for every saved source, domain, tall-object group, region and landscape are preserved in the machine-readable outcome report.

## Next measured work, not another blind rerun

1. **Keep classification independent.** The residual run deliberately preserved identification; all five epochs' saved classification metrics exactly match baseline. A new six-class identifier can be tested separately without using its outputs to gate heights yet.
2. **Audit train-only tall buildings and ground/object errors.** Check measured annotation reliability, height-band exposure, source pixel scale, image/prior alignment and correction spillover on fixed training examples. Do not inspect the sealed external geography to choose examples or settings.
3. **Run a matched height-source ablation, if training evidence warrants it.** Compare fresh three-source training with omission of GAMUS's height loss. Keep the same protected start, architecture, seed, optimizer updates, schedule, HighBuild/OpenCanopy examples and numerical loss coefficients. Do not replace omitted GAMUS batches with extra legacy exposure and call the difference a dataset effect. GAMUS identification remains a separate task. This is a hypothesis test, not a conclusion that GAMUS caused the failure.
4. **Test one building intervention separately.** A measured-building objective or height-balanced sampler needs its own fixed train-only feasibility proof and controlled comparison. Do not simultaneously change sources, model, sampler and loss then attribute any gain to one choice.
5. **Evaluate before promotion.** Only an eligible development candidate advances to the residual-compatible per-building scorecard and real app-inference path. Freeze the candidate and one-shot protocol before consuming genuinely new geography. The older corrected-triplet evaluator still needs the compatibility adapter described in `docs/RESIDUAL_HEIGHT_POST_RUN.md`.

## Evidence and safety

- This analysis reads saved reports/configuration and hashes protected files only. **No GPU inference, new dataset reads, official-test access or external-holdout access.**
- Source/group keys and all saved support counts match baseline across all five epochs: GAMUS **730,964,838**, HighBuild **13,855,690**, OpenCanopy **17,155,570** valid height pixels. Matching counts alone are not an independent proof of spatial-mask identity; masks were not reread in this audit.
- Both protected SHA-256 values match the experiment's configuration: production checkpoint `e2d507fb...0d144`; showcase pointer `a3d51c68...baa9`. No pointer or checkpoint changes were made.
- All five epoch checkpoints, latest checkpoint, source snapshot, configuration, metrics and terminal summary remain saved.
- Derived report: `outputs/diagnostics/residual_height_v1_outcome/report.json`.
- Reproducible CPU-only summarizer: `scripts/analyze_residual_height_outcome.py`. It fails on changed protected hashes, changed support counts/classification metrics, unexpected epoch history or a non-rejected outcome. It is a report summarizer, not a release evaluator.
