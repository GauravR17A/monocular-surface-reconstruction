# Independent saved-checkpoint test

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Completed 15 September 2026. The best guarded checkpoint from
`experiments/20260915T140116654177Z_gamus_rgb_segmenter_v1` was freshly loaded
and tested without updating weights or changing the production app.

## Outcome

**PASS: all 859 development-validation images reproduce epoch 8 exactly.**

- Six-class macro F1: **83.8252460040%**; macro IoU: **72.4664564976%**.
- Overall, DC and Philadelphia confusion matrices match exactly, with zero
  changed integer counts. Every per-class precision, recall, F1 and IoU matches.
- Road-boundary and dark-water proxy counts/scores also match exactly.
- The fixed 27 acceptance checks still pass. Reference-grid hashes, class
  supports, IDs/order, and validation RGB/CLS source contents were authenticated.
- The replay uses an independent NumPy confusion-count/scoring implementation.
  It reuses the sealed model, input loader and auxiliary boundary/proxy helpers;
  it is **not** an independently collected dataset or external scientific audit.
- Source snapshots, config, model pointer and protected height checkpoint
  remain unchanged. No training, height-label, DAV2, official test, NYC or
  external-geography rasters were read as part of this replay.
- Replay elapsed time: **148.55 seconds**. New/related targeted CPU tests:
  **42 passed**, including 18 new evaluator tests. No GPU workload remains.

Artifacts: `outputs/evaluation/gamus_rgb_segmenter_v1_independent_replay/`.
`report.json` contains the complete scores, identities and exact comparisons;
`per_image_metrics.jsonl` covers every image, including low-vegetation confusion.

## Visual test

Local review: **http://127.0.0.1:8766/**. This serves only the diagnostic
output directory, not the production app or a new upload/inference endpoint.

Six native 1024x1024 examples were fixed before scoring: first/middle/last
lexicographic validation ID in each city. They were not chosen for good scores.

- DC: `DC_02_26`, `DC_37_26`, `DC_76_32`.
- Philadelphia: `PHL_6150`, `PHL_6486`, `PHL_6925`.

For every example, the original RGB, prediction overlay, reference overlay
and per-class metrics load correctly. Browser checks confirmed all images are
1024x1024, all six class rows render, opacity 0/65/100% works, and previous/next
navigation wraps correctly. No page runtime errors were reported. All 36
PNG/metadata files named in the preview manifest passed their SHA-256 checks.
The automated browser session was closed after verification; the lightweight
loopback review server was left running for manual testing (launcher PID 4628).

## What this does and does not prove

The saved checkpoint truly reproduces the reported improvement; it was not a
stale score or an accidental selection of another model. Building and road
regions are visibly recognizable in the inspected examples, but boundaries
are smoother than the references and low vegetation is still frequently missed.

**This does not establish accuracy on a new city, sensor, resolution or season.**
These validation images were not used for gradient updates, but their results
were used for checkpoint selection. A separately preregistered external test
is still needed. There is no new height-accuracy claim or automatic app release.

The important unresolved case is DC low vegetation: F1 **22.49%**, recall
**15.84%**. Of its reference pixels, 28.44% were predicted ground, 21.90% roads,
22.09% trees, 10.06% buildings and 1.66% water. In the inspected DC example,
some low-vegetation labels form narrow strips around other classes; the
Philadelphia example has broad low-vegetation areas. This is a reason to
investigate appearance, scale and annotation consistency, **not proof that
the reference labels are wrong**. Do not retune on an external holdout.

Next steps: audit that DC failure pattern on development data, establish
the external identification protocol, and test an explicit overlay adapter
before allowing this classifier into the app. Keep heights protected.
