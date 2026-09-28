# Residual-height evaluation handoff

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Active recovery run: `experiments/20260915T124056Z_residual_height_v1`.
Only development validation is permitted during candidate selection. Production
is protected; official GAMUS test and the external NZ holdout must not be used
for tuning. If no epoch passes guards, report that honestly before planning the
next bounded experiment.

## Compatibility audit, 15 September 2026

The generic `load_predictor` and `predict_height` support residual checkpoints,
including cached-prior input and authentication of frozen inherited weights.
However, not every older evaluation script accepts the new wrapper:

- `evaluate_corrected_legacy_triplet.py` seals three historical model identities,
  their comparison report, and DomainGatedSurfaceNet attributes. It rejects a
  residual checkpoint. Do not substitute a candidate into the historical plan
  or label a trainer metrics JSON as a corrected-triplet report.
- `height_scorecard.py` requires a complete authenticated corrected-triplet
  report. A new residual-aware report adapter is needed before using it for a
  formal release decision. Keep existing input/support/annotation digests and
  historical reports intact.
- `evaluate_locked_benchmark.py` uses generic tiled inference and accepts the
  residual model. Use the corrected **validation** manifest with benchmark
  status `development`, never relabel it as unseen test data.
- `evaluate_highbuild_instances.py` is model-independent, but it needs an
  exact-grid prediction CSV, height GeoTIFFs and protected-building-probability
  GeoTIFFs. A batch exporter or the residual-aware corrected evaluator must
  produce these; no exporter has been implemented in this recovery turn.
- `audit_height_output_identity.py` is for semantic-only candidates whose height
  output must be unchanged. It must not be applied to an intentional height
  correction model. Use inherited-weight and semantic-output identity checks.
- `audit_protected_inference_protocol_v1.py` is a frozen production-only audit,
  not a residual-candidate evaluator.

## Next actions after an eligible checkpoint exists

1. Freeze the selected checkpoint SHA and its training config/source manifest.
   Read all epoch outcomes, not just the best number.
2. Implement/test a separate residual-aware corrected evaluation adapter. Reuse
   exact native 1024 HighBuild / 384 OpenCanopy inputs and measured-only COCO
   masks. Authenticate the protected model embedded in the candidate, unwrap
   its frozen semantic configuration, and preserve pixel/input/COCO digests.
   Include per-building and boundary evidence. Do not modify running training
   source or weaken the old three-model plan's validation.
3. Evaluate the actual 512-tile, 128-overlap app path on the same development
   manifest, paired with production. This is different from full-native
   evaluation because spatial normalization/padding changes predictions.
4. Only after development gates and protocol checks pass, freeze a one-shot
   external evaluation plan and use the preregistered new geography. Never use
   external results to repeatedly tune this candidate.
5. Keep deployment separate from evaluation; no automatic pointer change.

## App-path development benchmark command template

Replace CHECKPOINT, SHA and NEW_OUTPUT with the selected immutable values. Run
the same protocol for production when the comparator is needed. The existing
foundation model directory is `models/foundation/depth-anything-v2-small-hf`.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_locked_benchmark.py --checkpoint CHECKPOINT --manifest D:\MSRData\data\multidomain_v2\highbuild_measured_coco_intersection_v1\manifests\validation.csv --deny-manifest D:\MSRData\data\multidomain_v2\highbuild_measured_coco_intersection_v1\manifests\train.csv --output NEW_OUTPUT --benchmark-id residual-height-v1-corrected-validation --benchmark-status development --variants raw --relative-model models\foundation\depth-anything-v2-small-hf --device cuda --tile-size 512 --overlap 128 --expected-manifest-sha256 2e21024fb7d7562e51d91e219c9434087fadfee58b107a4eacdc2c04cd8fa118 --expected-checkpoint-sha256 SHA
```

This command is documented but has not yet been run on a learned residual
checkpoint. Treat its reports as app-protocol development evidence, not as an
automatic replacement for the frozen native/per-building scorecard.
