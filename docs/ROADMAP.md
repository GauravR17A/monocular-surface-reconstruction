# Ongoing work and research roadmap

Status as of 28 September 2026. The items below are research directions and
acceptance requirements, not promises that more training will automatically solve
the measured weaknesses.

## Priority 1: reliable object height

Audit tall-building reference support, per-instance consistency, acquisition
alignment and physical pixel scale. Test context-matched training and per-instance
supervision as separate interventions. Retain fixed native-resolution development
support and report short/medium/tall groups so one aggregate cannot hide a
regression. The large concentration of squared error in tall structures justifies
this priority.

An eligible development candidate must improve the declared object-height metric
while preserving bias, MAE, correlation, shape/instance behavior and other groups
under the applicable scorecard. The canonical gate uses bounded RMSE/MAE/bias
regressions, no decrease in correlation/R-squared and at least one material gain.
Passing development is still not final external acceptance.

## Priority 2: geographic classification transfer

Improve ground/low-vegetation separation and road boundaries while retaining
behavior in the original cities. Use matched replay/teacher-regularization or
another explicitly defined intervention with the same budget and selection rule.
Multiple independent seeds would help distinguish small effects from run variation.
Keep class ontology mapping and image radiometry fixed or report their change as
part of a package experiment.

Freeze the candidate before evaluating reserved geography. Do not reinterpret
Christchurch or the inspected all-class scenes as untouched. V3's development
improvement and authorized app integration do not close the external evidence gap.

## Priority 3: final metric-height transfer

After an eligible candidate and inference protocol are frozen, seal a one-shot
evaluation plan for the prepared Bay of Plenty package. Report all-valid,
object-only and tall-object results regardless of outcome, with dates, source
identities, mask counts and datum/acquisition limitations. A failed or interrupted
attempt still consumes the test and must be disclosed.

## Priority 4: usability and operational quality

Measure unfamiliar-user task completion on the actual released workflow: choose a
scene, identify units, distinguish class from geometry, inspect a profile, compare
a reference and export evidence. Obtain domain review of the interpretation, not
only visual approval. Add unrelated physical-device checks before making broad
browser/GPU claims.

If public hosting is desired, design it as a separate service project with
authentication/admission, workload limits, result isolation and retention. The
present local package is not a validated multi-user deployment.

## Deferred geometry extensions

Detailed roof shapes and tree-object geometry remain research modules rather than
certified features. Reintroduce them only with useful coverage, reference evidence,
clear source/display separation and interaction verification. Flood simulation,
snow depth, species recognition and before/after change detection would require
their own labels, methods and validation; a current class map does not provide them.

## Documentation maintenance

Update the status date when a new release is actually checked. Keep the current
overview short enough to assess, preserve dated experimental records, and add
new evidence instead of overwriting inconvenient outcomes. Every future release
should state its exact model/source identity and which conclusions changed.
