# GAMUS height-safe v2 pilot

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: predeclared before training on 2026-09-15.

## Why this run exists

The first direct-height pilot showed that GAMUS contains useful height signal:
on GAMUS validation its overall RMSE fell from 7.121 m to 4.471 m, building
RMSE from 5.250 m to 4.542 m, and vegetation RMSE from 8.897 m to 5.037 m.
It was still rejected because ground RMSE worsened from 1.527 m to 2.497 m,
building semantic F1 fell from 0.815 to 0.631, and corrected legacy validation
regressed severely. The active application model was never changed.

This v2 pilot tests a much narrower hypothesis: can the existing protected
model learn useful GAMUS height calibration through only its 33-parameter
metric output head and 65-parameter canopy output head while all shared
features, routing, building detection, fusion gates, and cached Depth Anything
V2 priors remain fixed?

## Locked data and supervision contract

- Training uses all 5,001 quality-approved GAMUS train tiles.
- Selection uses all 859 quality-approved GAMUS validation tiles.
- The official GAMUS test split is neither resolved nor evaluated.
- RGB validity, classification validity, and height validity remain separate.
- Height regression requires valid RGB, valid AGL, a valid class, and one of
  source classes 1, 2, 3, or 6.
- Background/unknown (0), water (4), and road (5) never supervise height.
- Invalid/missing heights may be stored as zero only behind a false regression
  mask; they never act as zero-height targets.
- Heights above 200 raw units remain excluded as rare, artifact-prone values.
- The local files do not declare units. Raw x1.0 is recorded as
  `metre_assumed`, so this pilot is development evidence, not a publishable
  metric claim by itself.
- Cached DAV2 priors are read-only inputs. No DAV2 model is run or updated.

## Model safety

Only `base_model.height_head.*` and `canopy_height_head.*` are trainable.
The protected checkpoint hash is
`e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144`.
The runner hashes the live checkpoint before and after training and fails if
the live pointer or checkpoint changes.

## Decision

The predeclared GAMUS gates remain identical to the first pilot: pooled RMSE
and MAE must improve, correlation/R2 must be preserved, DC and Philadelphia
must remain safe, ground/building/vegetation and tall-object guards must pass,
and three-class identification must not materially regress.

Even a GAMUS gate-passing checkpoint is not promotion eligible. It must first
pass the already-locked corrected HighBuild/OpenCanopy development protocols,
a genuinely new geographic holdout, visual/API regression checks, and explicit
human review. No official test result is permitted for this decision.
