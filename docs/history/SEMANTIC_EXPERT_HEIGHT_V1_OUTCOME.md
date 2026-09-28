# Six-class-guided height experiment: completion and error analysis

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Run: `D:/MSRData/experiments/semantic_expert_height_v1/20260919T180701956748Z`.
Completed 19 September 2026 at 18:25:34 UTC, four epochs per arm. This
closeout supersedes the active-run status in the original launch handoff;
the launch record and bound training files remain unchanged.

## Verdict

Actual six-class probabilities successfully guide the candidate's six height
correction outputs. However, this experiment does not demonstrate a clear
overall advantage over the equally sized, equally trained neutral-routing
control. The classifier was frozen: these are height results, not a new
classification F1 result.

No candidate met both the original safety and paired-benefit rules.
Best eligible candidate: **none**. Best unrestricted checkpoint under the
predeclared equal-building/canopy RMSE ranking: **uniform/control epoch 2**.
That checkpoint is not release-approved. The application remains unchanged.

## Matched final-epoch comparison

RMSE in metres; lower is better. All values below refer to the same development
evaluation: 160 HighBuild scenes and 120 OpenCanopy scenes, not an untouched
final geographical test.

| Group | Protected app | Neutral control, epoch 4 | Six-class candidate, epoch 4 |
|---|---:|---:|---:|
| Buildings | 13.2426 | 12.7649 | 12.7558 |
| Canopy | 8.3610 | 7.7913 | 7.8238 |
| Ground | 4.8944 | 4.5556 | 4.6083 |
| Short vegetation | 5.2049 | 4.6974 | 4.8969 |
| Short buildings | 4.7701 | 4.6421 | 4.5103 |
| Medium vegetation | 5.8957 | 5.6938 | 5.5906 |
| Tall buildings | 29.7967 | 28.6888 | 28.7640 |
| Tall canopy | 10.4863 | 9.6631 | 9.7688 |

## What improved, and what did not

- Both final arms improve the four headline RMSE groups versus the protected
  app. This supports useful height adaptation, not a clear added benefit from
  classification guidance.
- Class guidance reduces final building RMSE by just 0.0091 m versus control.
  There is no repeated-seed evidence that this tiny difference is reliable.
- The candidate improves short-building and medium-vegetation RMSE versus
  control, but worsens overall canopy, ground, short vegetation, tall buildings
  and tall canopy. Its class-specific benefit is mixed.
- No candidate epoch passes paired benefit. The final canopy gain versus the
  protected model exceeds 0.5 m and 5%, but the neutral control does better.
  Final building gain does not meet the required minimum improvement.
- All epochs also fail the unchanged safety rules. Aggregate RMSE alone does
  not establish acceptable bias, MAE, shape agreement, or regional behaviour.
  Conversely, an extremely strict correlation/R2 tolerance is not by itself
  a practical measure of the size or reliability of a regression.
- Best unrestricted control epoch 2 has building RMSE 12.7711 m and canopy
  RMSE 7.6911 m, but ground/short-vegetation RMSE 5.2373/5.8443 m. Do not
  combine different checkpoints' best group scores into a fictional model.

## Evidence and integrity checks

The run's `DEVELOPMENT_REPORT.md`, per-epoch evaluations, atomic commits,
`outcome.json`, configuration and source snapshots remain saved. Closeout
verification authenticated the source/config/data binding, both final epoch-4
checkpoint and evaluation hashes, and matching checkpoint bindings. All 240
protected model tensors match production in each final checkpoint. Production
checkpoint/pointer and V1/V3 classifier bindings match their protected hashes.
No trainer remained active at completion inspection.

The hourly `msr-residual-accuracy-check` monitor is now **PAUSED**.
No model promotion, new training, recipe change, or final-test consumption was
performed during this closeout.

## Justified next steps, not launched

1. Inspect the existing scene-level predictions and subgroup errors together
   with class-routing mistakes, especially short vegetation and tall objects.
   Determine whether errors arise from classification, height supervision,
   image scale, or the height model rather than assuming more classes fix them.
2. Predefine practical regression limits and uncertainty reporting for the next
   experiment. Keep these original failures recorded; do not retroactively
   change the rules to promote this candidate.
3. Choose one evidence-led modification and retain the matched neutral control.
   A longer run needs a stated reason and a bounded budget, not just more epochs.
4. Recheck classification independently if its inputs or weights change, and
   reserve genuinely new geography for a deliberately defined final evaluation.

Six routes do not certify six independent height categories. These results
provide no new road/water height validation, universal unseen-data accuracy,
or accuracy percentage.
