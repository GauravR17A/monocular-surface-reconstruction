# Experiment register

This register links the research question to the intervention, result and decision.
Dates identify development records. A source release does not repeat training or
retroactively change the original acceptance rules.

| ID / period | Question and intervention | Observed outcome | Decision / source |
| --- | --- | --- | --- |
| E01 · Aug 26 | Can a geographically separated urban pipeline train and resume reliably? Small reviewer pilot, then a full ConvNeXt V2 height run. | Training/resume worked; early all-pixel scores were strongly affected by reference support. | Retain plumbing evidence; use corrected masks for current claims. `history/PROGRESS.md` |
| E02 · Aug 28 | Can one external pretrained height model transfer without adaptation? FusedSeg-HE evaluation adapter. | FP16 non-finite predictions invalidated the first run; FP32 was finite but poor on the selected tall-building sample. | Domain/scale compatibility is essential; not a universal model ranking. `history/PROGRESS.md` |
| E03 · Aug 28–29 | Add canopy estimation without erasing useful urban behavior. Protected urban expert plus guarded surface refinement. | A protected vegetation checkpoint was retained; later quality checks exposed remaining tall-height weaknesses. | Current protected height family. `history/PROGRESS.md` |
| E04 · Sep 12–13 | Expand with GAMUS specialists, replay and routing. | Router confidence and endpoint tradeoffs did not establish a safe universal replacement. | Retain bounded experiments and inspect final outputs, not only expert heads. `history/EXPANDED_GAMUS_PLAN.md` |
| E05 · Sep 13 | Joint height and six-class extension. | Height pilot and joint candidate underperformed protected forest results; water recall collapsed in the first joint classifier. | Reject promotion. `history/GAMUS_SIX_CLASS_EXPERIMENT_STATUS.md` |
| E06 · Sep 13 | Preserve all height tensors while training a linear semantic head; address class imbalance. | Water improved, but ground/low-vegetation/tree retention gates failed. | Reject; inherited tensor identity verified. Same status record. |
| E07 · Sep 13 | Adjust decisions through additive class bias on fixed outputs. | Development tradeoff improved; original class floors still missed. | Not probability calibration and not enabled as a certified improvement. `history/GAMUS_SIX_CLASS_DECISION_BIAS_CALIBRATION.md` |
| E08 · Sep 13–15 | Compare spatial head and hierarchical vegetation head on a matched city split. | Corrected replay macro F1: spatial V3 56.48%, hierarchical V4 55.11%. | Hierarchy did not improve the combined six-class result. `history/ACCURACY_RECOVERY_2026_09_15.md` |
| E09 · Sep 15 | Diagnose stalled height learning: tiny trainable control and gated correction pathway. | The 98-parameter control showed no eligible gain; train diagnostics found weak effective correction and gradient flow. | Test a richer residual decoder; diagnostics are not validation scores. Same recovery record. |
| E10 · Sep 15 | Audit full-native/app inference and small-image padding. | Strong forest padding sensitivity with opposing canopy and ground effects. | Keep protocols separate; no global padding switch. Same recovery record. |
| E11 · Sep 15 | Train a zero-initialized spatial residual-height decoder. | Five completed epochs; modest aggregate improvement, persistent tall-object errors and failed guards. | Reject every candidate. `history/RESIDUAL_HEIGHT_V1_OUTCOME.md` |
| E12 · Sep 15 | Train independent RGB-only six-class segmentation. | V1 improved development identification; reproducibility replay verified epoch 8. | Preserve model-specific limits and test external transfer. `history/RGB_SEGMENTER_V1_OUTCOME.md`, `history/RGB_SEGMENTER_V1_TEST_REPORT.md` |
| E13 · Sep 15 | Freeze V1 and test Christchurch once. | 61.63% macro F1, with strong water and weak ground/road boundaries. | Publish complete result; mark city consumed. `history/RGB_SEGMENTER_V1_CHRISTCHURCH_RESULTS.md` |
| E14 · Sep 15 | Does targeted ground/low-vegetation crop sampling beat an equally trained control? | Final targeted advantage was small; key benefit and tree-retention rules failed. Earlier epoch remained exploratory. | No proven sampling improvement. `history/RGB_SAMPLING_V2_OUTCOME.md` |
| E15 · Sep 19 | Test six categories together in diverse mixed scenes. | 54.02% macro F1 across 12 label-enriched OEM scenes; ground/low vegetation/roads weak. | Diagnostic challenge, not general accuracy. `history/OEM_ALL_SIX_CLASS_TEST_2026_09_19.md` |
| E16 · Sep 19 | Add regional RGB training/augmentation. | V3 improved OEM transfer but regressed on some GAMUS classes and Philadelphia road boundaries. | Best development epoch 1; no fully safety-passing checkpoint. `history/RGB_MULTIREGION_V3_OUTCOME.md` |
| E17 · Sep 19 | Does class-assisted height correction beat neutral inputs at matched budget? | Fast V2 reduced building/canopy errors but damaged short surfaces; neutral control explained most gain. | Reject release. `history/CLASS_ASSISTED_HEIGHT_FAST_V2_OUTCOME.md` |
| E18 · Sep 19 | Protect ground and short vegetation more strongly. | Short-surface errors improved, canopy suppressed too much. | Reject; investigate selective geometry instead of only loss reweighting. `history/CLASS_ASSISTED_HEIGHT_LOW_SURFACE_V3_OUTCOME.md` |
| E19 · Sep 19 | Balance height-band crop exposure. | Some tall-height gains; weak paired canopy advantage and worse other groups. | No accepted replacement. `history/HEIGHT_BAND_SAMPLING_V1_OUTCOME.md` |
| E20 · Sep 19 | Mix six learned correction experts with frozen class probabilities. | Class routing affects outputs, but no clear overall advantage over matched neutral routing. | Reject; do not claim class-guided height success. `history/SEMANTIC_EXPERT_HEIGHT_V1_OUTCOME.md` |
| E21 · Sep 19–20 | Expose independent six-class identification in the application. | Preview and then automatic class identification delivered; raw height artifacts preserved. | Authorized product integration with model limitations intact. `history/SIX_CLASS_DEFAULT_INTEGRATION_2026_09_20.md` |
| E22 · Sep 19 | Fit more detailed roof geometry. | The cited Strasbourg example yielded zero accepted fits out of 12 footprints under the experimental gates. | Flat roofs restored; modules retained inactive. `history/ROOF_RECONSTRUCTION_STAGE1_2026_09_19.md` |
| E23 · Sep 26–27 | Improve terrain imports, smoothing, profile analysis and exploration. | Source-preservation and interaction checks recorded across specified scenes. | Delivered presentation/tooling work; no new height-model accuracy claim. Dated terrain/viewer records. |

## How to add the next experiment

Record the hypothesis before running it. Specify the changed variable, unchanged
control, initialization, data exposure, training budget, evaluation grid and masks,
selector, safety thresholds and stop condition. Bind source/configuration/data
identities, save every completed epoch and disclose failures or restarts.

Compare final deployed outputs rather than an internal head that the deployed
fusion does not use. Include height strata and per-region/per-class results.
Describe a one-seed effect as a one-seed observation. If several recipe elements
change together, call the result a package comparison rather than an isolated
causal ablation. Preserve negative results because they explain future design.

No new training or reserved-geography evaluation was launched for this publication.
