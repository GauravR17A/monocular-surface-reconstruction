# RGB sampling V2: completed experiment and decision

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Completed 15 September 2026, 16:09:41 UTC. Pair:
`experiments/20260915T154725409300Z_rgb_sampling_v2_pair`.

## Decision

The primary experiment did **not** meet its predefined acceptance rules.
Preserve targeted epoch 2 as an exploratory, best-safe candidate; do not promote
it to the app or describe this experiment as a proven sampling improvement.
No height model, production pointer, or app-active model was changed.

## What was compared

Two four-epoch continuations started from the identical V1 classification
checkpoint. Control used ordinary crops; targeted used 50% ordinary and 50%
ground/low-vegetation-rich crops. Both used the same 3,837 training-tile draws
per epoch, model, optimizer recipe and budget. Each epoch was evaluated on the
same 859 full native-resolution DC/Philadelphia development tiles.

These validation tiles were used repeatedly for model selection. They are not
an untouched geographic test. No external holdout was evaluated in this run,
and no height accuracy was measured or improved by this classifier experiment.

## Results

All values below are F1 percentages; differences are percentage points (pp).

| Measure | Original V1 | Control epoch 4 | Targeted epoch 4 | Exploratory targeted epoch 2 |
|---|---:|---:|---:|---:|
| Six-class macro F1 | 83.83 | 83.94 | 84.01 | 83.98 |
| DC ground F1 | 68.17 | 68.39 | 68.46 | 68.49 |
| DC low-vegetation F1 | 22.49 | 24.38 | 24.98 | 28.51 |
| Philadelphia tree F1 | 85.86 | 84.62 | 84.40 | 85.39 |
| All predefined safety checks versus V1 | Baseline | Fail | Fail | Pass |

The primary comparison was fixed in advance as epoch 4 versus epoch 4:

- Targeted macro F1 exceeds control by only **0.067 pp**.
- Targeted DC low-vegetation F1 exceeds control by **0.599 pp**, below the
  required 2 pp. Its gain over V1 is 2.493 pp, below the required 5 pp.
- The mean of DC ground/low-vegetation improves by 1.389 pp over V1 and
  0.330 pp over control, below the respective >2.5 pp and >=1 pp rules.
- Philadelphia tree F1 drops 1.465 pp versus V1, exceeding the permitted
  1 pp regression. Thus the final candidate fails safety as well as benefit.

Targeted epoch 2 passes the V1 safety and benefit checks, but does not establish
that targeted sampling is better: control epoch 2 achieves DC low-vegetation
F1 of 29.14%, versus targeted 28.51%. Earlier checkpoints are exploratory,
not replacements for the failed primary experiment. Passing safety allows small
predeclared regressions; it does not mean every category improved.

## Error-analysis conclusions and next experiment

Both arms peak on DC low vegetation at epoch 2 and deteriorate afterward.
Additional continuation alone therefore does not reliably resolve this weakness.
The training-only sampling audit increased expected low-vegetation pixel
exposure by about 12%, but the controlled run does not establish a useful
advantage from that change. These metrics identify a pattern, not its cause.

Before more training, audit **training-only** DC/Philadelphia examples for
class-definition consistency, small vegetation patches, image scale and boundary
detail. Record class-conditioned confusion and size-related failure patterns
without using external holdouts to tune the recipe. If fine detail is the
confirmed bottleneck, define a matched higher-resolution/detail-preserving
comparison with the same baseline and explicit category regression guards.
Do not automatically lengthen this failed recipe or relax its acceptance rules.
Any eventual candidate still needs independent replay, replication and a fresh
geographic evaluation; no universal unseen-image accuracy is established here.

## Saved artifacts and completion audit

Each arm retains all four committed epoch checkpoints, metrics, hash receipts,
optimizer/scheduler/RNG state and atomic commit records. Pair configuration,
source snapshot, proof, binding, status and `comparison_report.json` are saved.
The earlier failed preflight pair is preserved separately and must not be resumed.

Authenticated SHA-256 identities:

| Artifact | SHA-256 |
|---|---|
| Control epoch 4 / latest | `97507b7aeca2238f8da07bf73971137593404ec825d2189c68df543a3bd7ebd9` |
| Targeted epoch 4 / latest | `b16822dd10a916357559e2a7314a511d9a89d978833b3708ed64b1c48d769484` |
| Targeted epoch 2 / best-safe | `879ce4cf07b46ec817ec40fcc604ce035dad44d57b757481bb5402e854bfecbd` |
| Protected production height checkpoint | `e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144` |
| Protected production pointer | `a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9` |

The read-only CPU completion audit authenticated checkpoints before loading.
Both latest histories are epochs 1-4 with optimizer/scheduler step 3840;
best-safe history is epochs 1-2 with step 1920. All 209 floating model tensors
and 615 floating optimizer tensors per inspected checkpoint are finite.
Bindings and complete saved RNG keys match. Saved configuration, 12/12 snapshot
sources, 12/12 current sources, 27/27 protected artifact identities and 3/3 bound
inputs pass verification. No GPU work or dataset decoding was needed for this
completion audit. The training process has exited normally.

The hourly completion monitor should now be paused. No new training or app
promotion is part of this completion handoff.
