# GAMUS six-class decision-bias calibration

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: **implemented and run once on development data; rejected for app use**.

The sealed default protocol was applied to the completed head-only v2 recipe.
It was permitted only after training finished and
`audit_gamus_head_only_checkpoint.py` proved that every inherited height/shared
tensor is bit-for-bit equal to the protected model.

## Why this exists

The head-only GAMUS recipe deliberately oversamples water tiles and gives water
a larger training-loss weight. That is useful for teaching the head not to
ignore a rare class, but it can also change the class prior learned by its raw
logits. A small additive offset per class can correct the final class decision
without retraining or touching any height prediction.

The calibrated decision is:

`chosen class = argmax(raw six-class logits + six fixed biases)`

The six biases are constrained to have mean zero because adding the same value
to every logit would not change the decision.

## What it is not

This is **class-decision calibration**, not probability/confidence calibration.
It may improve which category wins the argmax, but it does not prove that a
displayed value such as `80%` means an 80% chance of being correct. Temperature
scaling, reliability diagrams, ECE/Brier checks, and an independent calibration
set would be required before making that confidence claim.

The utility also does not change:

- any model or height tensor;
- the protected production checkpoint;
- the active application pointer;
- the three-group ground/building/vegetation height-routing machinery (this is
  not a three-class replacement for the six-class identifier);
- the app's active model.

## Data boundary

Fitting and the one exact follow-up evaluation use only the 859 official GAMUS
validation tiles from Washington, DC and Philadelphia. The approved index and
candidate training configuration are hash-sealed.

- NYC is forbidden during fitting and development evaluation.
- This calibration process never discovers or constructs the official GAMUS
  test. Project-wide, that test was already consumed by the older Stage-3 final
  audit and is forbidden for reuse.
- The calibration process never opens the future locked NYC files. The active
  v2 model was trained on the full approved training set (including NYC), so it
  cannot be presented as unseen-NYC evidence. Even a later DC+PHL-trained head
  may describe NYC only as classifier-head-excluded because the inherited
  HighBuild-trained system also has New York exposure. A separate external
  geography is required for a system-level generalisation claim.
- Because DC+PHL is development data already used for model selection, its
  calibrated result is development evidence, not an independent generalization
  result.

## Two-stage procedure

### 1. Bounded fit

All 859 DC+PHL validation tiles are visited once. At most 64 pixels per class
per tile can enter a deterministic priority reservoir, and at most 30,000
pixels per class are retained. Therefore no more than 180,000 six-logit rows
can be cached. Each retained class pixel is weighted back toward its observed
full-validation class support; the later exact pass remains authoritative.

A deterministic coordinate search tries bounded mean-zero biases. It ranks
candidates against gates declared before fitting:

- macro F1 and macro IoU floors;
- minimum water precision, recall, and F1;
- minimum road F1;
- no meaningful regression in ground, buildings, roads, low vegetation, or
  trees versus the same checkpoint with zero bias;
- a minimum macro-F1 gain.

If any sampled gate fails, the fit is recorded as rejected and the exact pass
is refused.

### 2. One exact DC+PHL pass

An accepted fit can be evaluated once over every valid labelled pixel of all
859 tiles. The evaluator streams confusion, road-boundary, and water/dark-pixel
proxy metrics and never stores full logits. It reports both uncalibrated and
decision-calibrated results from the same forward pass. Road-boundary quality
has its own predeclared floor and retention guard.

A consumption marker is written before the exact dataset is constructed. A
crash therefore cannot silently become repeated full-validation tuning.

## Reproducible commands

Protocol-only preflight; this does not open imagery or a checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_gamus_six_class_decisions.py --validate-protocol
```

After head-only v2 selects `checkpoint_best_landscape.pt`, first run the
independent tensor audit. Only after that report passes, fit the external
decisions:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_gamus_six_class_decisions.py --fit --candidate-checkpoint <path-to-checkpoint_best_landscape.pt>
```

The fit command independently repeats the state comparison and also requires
`outputs/evaluation/gamus_six_class_head_only_v2/checkpoint_best_audit.json` to
authenticate the exact same checkpoint hash. It therefore fails before opening
the calibration imagery if training/auditing has not finished.

Only if `fit.json` passes every sampled gate, run the one exact development
evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_gamus_six_class_decisions.py --evaluate-full --candidate-checkpoint <path-to-checkpoint_best_landscape.pt> --confirmation RUN_FULL_DC_PHL_DECISION_BIAS_EVALUATION_ONCE
```

Artifacts are written beneath
`outputs/evaluation/gamus_six_class_decision_bias_dc_phl_v1/`. Passing these
development gates does not promote the checkpoint, authorize app integration,
or consume the locked NYC holdout. Those remain separate review decisions.

The Python utility is candidate-path driven. Another compliant six-class
head-only candidate can use the same code, but it requires a new versioned YAML
protocol with that candidate's exact training-config hash and a separate output
folder. An arbitrary checkpoint is intentionally rejected rather than guessed.

## Reproducible implementation

- Protocol: `configs/gamus_six_class_decision_bias_v1.yaml`
- Tool: `scripts/calibrate_gamus_six_class_decisions.py`
- Pure search/sampler: `src/msr/evaluation/decision_bias_calibration.py`
- Tests: `tests/test_gamus_decision_bias_calibration.py`
