# Class-assisted height pilot — controlled experiment

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: **paired training-only proof passed; bounded development run launched**.
Executable bindings are in `configs/class_assisted_height_proof_v1.json` and
`configs/class_assisted_height_development_v1.json`. No candidate is promoted.
Do not apply this recipe to an older experiment.

## Question and scope

Does independently predicted six-class information help a height learner beyond
the same RGB/DAV2/protected-height inputs? Better class F1 alone is not evidence.

Keep the protected production model, app pointer, V1 classifier, V3 classifier
and previous residual-height experiment immutable. The app continues using the
protected height model. Use a new experiment directory on D for large caches.
Do not train six disconnected height models or force water/roads to zero.

## Preparation gates

1. Authenticate exact training/development manifests, corrected COCO masks,
   protected model, frozen V3 epoch-1 classifier, DAV2 weights/preprocessor,
   code snapshot and software versions. Preserve the older failed residual
   run as historical comparison, **not** the matched ablation baseline.
2. Record height and semantic validity separately from image validity. Missing
   labels remain ignored. HighBuild supervises measured building support, not
   unlabeled background. OpenCanopy supervises available above-ground labels.
3. Use HighBuild and OpenCanopy for the first matched pair. GAMUS height units
   remain unverified, so do not silently treat its height values as trusted new
   supervision. Omitting GAMUS is common to both arms: results cannot isolate
   the effect of dropping GAMUS relative to the previous pilot.
4. Verify every cached prior used by the small proof against its image and
   frozen DAV2 recipe; record new evidence bindings without rewriting old caches.
   The completed 10-example replay is a useful check, not full certification.
5. Six-class inputs must be **model predictions from RGB**, never reference masks
   or true heights. Cache probabilities, invalidity and source/weights hashes.
   Keep per-pixel alignment under crop/rotation; do not color-augment imagery
   while treating unaltered cached class/depth predictions as freshly inferred.
6. Inspect representative training tall/short buildings and forest/ground
   transitions. The current 450-tile subset has missing explicit GSD: preserve
   native pixel handling, disclose the scale gap, and do not invent metre scale
   from filenames. Scale-aware augmentation requires recovered source metadata.
7. Resolve overlap policy for instance supervision: overlapping annotations and
   tiny supports cannot be silently treated as independent clean full buildings.
   Keep pixel support unchanged in this first experiment; add instance loss as
   a separate later experiment after versioning its support contract.

## Matched arms

- Both start from the same protected height model and copied residual decoder.
  The protected backbone, original heads and six-class classifier remain frozen.
- Both use the same extra six-channel input capacity. **Control A** receives
  uniform class probabilities on valid RGB; **candidate B** receives the frozen
  classifier's soft probabilities. Both receive the identical image-validity
  channel. Unknown pixels are not assigned a semantic class or height.
- Concatenate conditioning to the learned residual context; do not hard-gate
  the protected height output by class. Initialize the final correction to zero.
- Same seed, sample sequence, batches, augmentations, optimizer steps, loss,
  decoder capacity and schedule in both arms. Same native development pixels.
  Log actual sampled IDs/transforms so the comparison is auditable.
- Preserve independent semantic outputs byte-for-byte. A later intentional
  feature/backbone change would require rechecking classification separately.

## Bounded stages

1. Train-only feasibility: fixed examples spanning five HighBuild training cities
   and OpenCanopy training sources; maximum 64 optimizer updates per arm. Check
   finite gradients, zero-initialization identity, frozen tensors, valid masks,
   memory and at least 10% loss reduction. This is an optimization check only.
   Discard proof weights and never describe this training loss as accuracy.
2. Only after gates pass, freeze executable config and a maximum two-epoch paired
   development run. Same protected start in both arms; no final reserved geography.
   Save checkpoints, optimizer/RNG state, inputs/hashes and resumable atomic commits.
3. Report per-source/per-city RMSE, MAE, bias, correlation and R²; separate buildings,
   canopy and ground. Include >=20 m buildings, short vegetation, boundaries and
   six-class/legacy-geometry disagreements. Missing supported categories are N/A.
4. A useful candidate must reduce a priority-object RMSE by both >=0.5 m and >=5%
   versus protected production **and** beat the matched control on that same metric.
   Report all other changes, including if the control improves more elsewhere.
5. Preserve the previous pilot's safety checks; no retrospective loosening.
   At minimum no protected group RMSE regression >0.15 m, MAE >0.10 m or absolute
   bias >0.10 m; retain its correlation/R² and tall-object checks. Define exact
   machine-readable rules before training, not after seeing results.
6. Passing this pilot does not release a model. Next: residual-compatible native
   per-building scorecard, app-tiled paired inference and a frozen one-shot
   genuinely new-geography test. No default-model change without explicit review.

## Stop conditions

Stop on changed bindings, label/prior misalignment, unknown unit conversion,
non-finite loss, unsupported frozen-tensor changes or failed optimization proof.
If class conditioning offers no protected benefit, keep the scanner feature and
prioritize height-data/scale/tall-object coverage instead of repeating the run.
No confidence probability or height improvement is promised in advance.

## Executed feasibility and locked development details

Proof: `D:/MSRData/experiments/class_assisted_height_proof_v1/20260919T161049828458Z`.
Twenty training examples: one short and one tall-support tile from each of five
HighBuild cities, plus ten OpenCanopy training region groups with both ground
and vegetation supervision. Every proof prior was replayed against its source;
cached soft classifications were inferred from RGB only. Group names do not
imply ten independent forest geographies.

Both arms used batch 2, two-batch accumulation, 64 optimizer updates and identical
initial tensors, crop coordinates and sample order. Balanced training-only loss
fell from 17.1686 to 6.8253 (uniform) / 6.8178 (predicted), about 60.25% / 60.29%.
The two results are nearly identical: **this is feasibility, not evidence that
classification improves height accuracy**. Ground training loss increased in
both arms. That is a reason to examine ground protection carefully in development,
not to release either proof checkpoint. Frozen weights and semantic outputs
remained identical. Proof checkpoints are preserved but never reused for training.

Development uses all 450 urban and 600 forest training records. Each of two
epochs contains 600 source-paired batches: every forest tile once, every urban
tile once plus 150 urban repeats. Each arm receives identical crop seeds and
initialization. No color/spatial augmentation; full-image class predictions and
DAV2 priors are cropped with all other inputs. Probabilities are stored in float16
then renormalized at loading; original RGB, priors and targets remain unchanged.

Validation uses all 160 native HighBuild + 120 native OpenCanopy development
tiles, with the same support/datum as the corrected benchmark. Each used prior
must pass a numerical replay check before training. Evaluation includes source,
domain, region/domain, tall objects (20 m building / 15 m canopy), short vegetation
(0–2 m), boundaries and classifier/legacy-geometry disagreements. Guards apply
to every supported reported group; correlation/R² tolerate only 1e-6 numerical
decrease. Material benefit requires BOTH 0.5 m and 5%, plus beating the matched
same-epoch control. Report both epochs rather than switching selectors afterward.

Scope clarification: GAMUS heights are neither supervised nor evaluated in this
two-source pilot because their units remain unresolved. Report **legacy safety**,
not universal/GAMUS safety. GAMUS safety and the existing per-building/app-path/
external checks remain release prerequisites. Every artifact says release ineligible.

Recovery: `scripts/run_class_assisted_height.ps1 -Resume <exact run directory>`.
An OS file lock prevents duplicate paired trainers. Source/config/software,
proof evidence, original data, cached probabilities, baseline, and committed
checkpoints are authenticated. Save every 50 optimizer updates and every epoch.
Orphaned or changed artifacts fail closed for inspection, never bypass integrity
checks or silently change the batch. New snapshots are on D; active app unchanged.
