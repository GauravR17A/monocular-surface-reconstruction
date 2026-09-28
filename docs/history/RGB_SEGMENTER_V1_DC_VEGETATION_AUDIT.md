# RGB segmenter V1: DC low-vegetation audit

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

This is development diagnosis, not a new accuracy benchmark or a model change.
All 859 independently replayed validation records were checked against the exact
pooled city confusion matrices. The six preview label arrays were checked against
their hashed categorical PNGs. Three synthetic audit tests pass.

## What the evidence says

| Low vegetation | Washington DC | Philadelphia |
|---|---:|---:|
| Validation images | 359 | 500 |
| Share of labelled pixels | 8.91% | 34.43% |
| F1 | 22.49% | 87.22% |
| Recall | 15.84% | 88.30% |

DC reference low vegetation is predicted as ground 28.44%, roads 21.90%,
trees 22.09%, buildings 10.06%, water 1.66%, and correctly as low vegetation
15.84%. The problem is broader than a simple tree-versus-grass switch.

In the three fixed DC previews, removing a two-pixel boundary leaves only
20-31% of low-vegetation pixels. The corresponding Philadelphia examples retain
65-77%. DC interior recall is still weak (about 6-25% at this radius), so simply
sharpening boundaries cannot solve the whole problem. These six preselected
examples establish a useful hypothesis, not a city-wide morphology estimate or
proof that annotations are wrong.

## Controlled follow-up, not an automatic promotion

1. Audit training-only low-vegetation coverage by city and crop. Measure whether
   small DC vegetation features receive enough supervised training examples.
2. Test city/class-aware crop sampling against an equal-budget continuation
   baseline. Keep identical development evaluation and six-class regression guards.
3. Only separately test a finer-resolution decoder or boundary loss if sampling
   does not address the weakness. Do not change several variables and attribute
   gains to one of them.
4. Preserve the completed V1 checkpoint as the comparison. Never feed the new
   classification output into production heights without a separate validation.
5. Test the already-frozen V1 on the reserved external city without tuning it on
   that result. Future models need a different final holdout after this one is used.

Machine-readable evidence:
`outputs/diagnostics/rgb_segmenter_v1_dc_vegetation_audit/report.json`.

No training, app pointer change, or height-pipeline change was performed by this audit.
