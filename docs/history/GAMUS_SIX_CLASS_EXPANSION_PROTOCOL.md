# GAMUS six-class expansion protocol

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: implementation and initial versioned experiments complete; no expanded
candidate has been promoted. V4 remains a controlled comparison until its
paired report exists and all gates pass.

## Objective and comparison

The expanded candidate adds a separate dense identification output for, in
this fixed order: `ground`, `buildings`, `water`, `roads`, `low_vegetation`,
and `trees`. The existing three-group `ground/building/vegetation` routing and
all height heads remain present and unchanged.

“Six-class” always refers to the final identification output and its exact
6 x 6 confusion matrix. “Three-group” refers only to the coarser
ground/building/vegetation **height-routing domains**; it is not a substitute
three-class identification result. Merging low vegetation and trees yields a
five-group diagnostic, not a six-class macro score.

The height-focused GAMUS pilot is the required warm start and comparison, not
an artifact to overwrite. The production checkpoint and
`outputs/runtime/showcase_checkpoint.txt` remain protected. No six-class
candidate may become active merely because its classification score improves.

## Implemented contract

- GAMUS source class 0, malformed values, and padding use ignore index 255.
- `image_valid_mask`, `classification_valid_mask`, and `height_valid_mask`
  describe independent availability. Effective semantic supervision uses
  image-valid AND classification-valid pixels.
- Height regression uses image-valid AND height-valid AND classification-valid
  pixels from GAMUS ground, building, low-vegetation, or tree classes.
- Water and road pixels teach identification but never height regression.
- Missing or excluded heights are stored as neutral tensor values only after
  their regression mask is false; they cannot become supervised zero height.
- The optional six-class head is disabled by default, so old configurations
  and old checkpoint state dictionaries are unchanged.
- Enabling it creates model type `domain_gated_surface_v4_six_class`. A
  dedicated compatibility loader allows only the new head tensors to be
  absent when warm-starting an old checkpoint; every height tensor stays
  strict.
- Tiled inference can return six aligned probability maps, but the application
  does not expose them until validation and an explicit model-selection step.

## Required evaluation

On the complete authenticated GAMUS validation split, report:

- the 6 x 6 reference-versus-prediction confusion matrix;
- precision, recall, F1, and IoU independently for all six classes;
- macro precision, recall, F1, and IoU;
- a two-pixel-tolerance road-boundary precision, recall, F1, and dilated
  boundary IoU (tolerance remains in pixels until ground sampling distance is
  authenticated);
- water false positives on dark non-water RGB pixels as an explicitly labelled
  proxy only. GAMUS has no shadow class, so this is not shadow ground truth;
- all existing GAMUS, urban, forest, building, canopy, tall-object, city, RMSE,
  MAE, correlation, and R-squared height guards under identical protocols.

The final report must also use genuinely new external geography. Validation
cities may guide development. NYC can be excluded from a newly fitted
DC+Philadelphia classifier head, but it is not system-unseen because earlier
GAMUS work and the inherited HighBuild-trained checkpoint already exposed New
York data. The official GAMUS test was consumed by the older Stage-3 final audit
and is permanently forbidden for V4 reuse; it is not a fresh holdout.

## Preparation blockers before a longer run

Do not enable colour augmentation or start a long six-class run until the
versioned preparation audit has resolved or explicitly bounded:

1. GAMUS AGL height units and per-city pixel scale;
2. the raw values above the current 200-metre cutoff;
3. cached Depth Anything V2 map identity, source-image identity, model identity,
   alignment, numeric range, and completeness;
4. independent image/class/height availability counts after the new mask
   contract;
5. train/validation/test and new-geography non-overlap.

## HighBuild and OpenCanopy additions

HighBuild per-building instance metrics and boundary checks use the corrected
supervision files and report measured versus estimated-height annotations
separately. Building counting is a separate validated task, not inferred from
pixel RMSE.

OpenCanopy acquisition dates, locations, and source metadata must remain with
each sample for stronger geographic and temporal holdouts. Recovering richer
vegetation classes is an evaluation/relabeling study, not something inferred
from the present binary canopy mask. Infrared inputs and before/after change
detection are later extensions requiring separately prepared inputs.

## Application outputs after validation

After the candidate passes both identification and height non-regression
checks, a review build may expose six-class overlays, building/canopy height
summaries, and per-class coverage. Area is reported in square metres only when
the input has authenticated geographic scale. Confidence remains a score until
it is calibrated on held-out data; it must not be labelled as a probability of
correctness beforehand.

Water identification can support a future flood module, but flood simulation
also needs ground terrain and hydraulic inputs. Snow cover needs suitable snow
labels, and snow depth needs separate metric-depth supervision.

## Promotion rule

Publish expanded-candidate results first. Compare them directly with the
height-focused pilot and protected production checkpoint on GAMUS plus the
corrected urban/forest suites. Only a candidate that passes every predeclared
height guard, six-class review, genuinely new external-geography evaluation, visual sanity
check, and explicit human approval may change the app's active pointer.

The canonical corrected full-scene protected baselines are:

- HighBuild strict-measured building-only RMSE: **13.243 m**;
- OpenCanopy vegetation-only RMSE: **8.361 m**;
- OpenCanopy all-valid forest RMSE: **6.707 m**.

They use different datasets, masks, and pixel supports and are not
interchangeable. Historical crop, zero-filled, inclusive-annotation, pooled, or
forest-all-valid values must keep their own protocol label.

## Versioned run and reporting

- Height comparison: `configs/multidomain_surface_gamus_direct_height_pilot.yaml`
- Expanded ablation: `configs/multidomain_surface_gamus_six_class_candidate_v1.yaml`
- Frozen gates: `configs/gamus_six_class_comparison_v1.yaml`
- Launcher/watcher: `scripts/run_gamus_six_class_candidate_v1.ps1` and
  `scripts/watch_gamus_six_class_candidate_v1.ps1`
- Report tool: `scripts/compare_gamus_six_class_candidate.py`

The launcher refuses to start until the newest height-pilot session is
terminal and successful. It then seals the exact height metrics file, selected
epoch, reproducibly matching checkpoint (if available), terminal checkpoint,
and their hashes into the candidate's saved launch config. If a raw-best epoch
has no matching saved checkpoint, the report marks that checkpoint unavailable
instead of attaching the final epoch's weights to earlier metrics.

The historical run order was height pilot, expanded candidate, identical GAMUS
comparison, and corrected HighBuild/OpenCanopy comparison. The controlled V4
order adds paired DC+Philadelphia V3/V4 training, an optional one-shot
classifier-excluded NYC check, and finally a separately reserved genuinely new
external geography. The already-consumed official GAMUS test is never reused.
Expected RTX 4070 time is roughly 75–105 minutes for each six-epoch GAMUS run,
plus 20–45 minutes for corrected legacy evaluation; early stopping can shorten
this. The initial expanded results are recorded, but no model change is claimed
and V4 accuracy remains unproven until its paired evaluation completes.
