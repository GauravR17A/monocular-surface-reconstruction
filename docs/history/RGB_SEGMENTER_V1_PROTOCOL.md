# Independent RGB six-class identification experiment

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: completed eight epochs on 15 September 2026 at 14:25:30 UTC.
Best development and guarded checkpoint: epoch 8, six-class macro F1 0.8382524600;
all 27 predeclared checks passed. See `docs/RGB_SEGMENTER_V1_OUTCOME.md` for
the remaining city/class weakness and release limitations. Not app-active.
Exact run: `experiments/20260915T140116654177Z_gamus_rgb_segmenter_v1`.

Subsequent verification: a fresh independent-scoring checkpoint replay passed
on all 859 development images with exact counts and scores. Diagnostic gallery
and limitations are in `docs/RGB_SEGMENTER_V1_TEST_REPORT.md`. External
generalization and app integration are still pending.

## Verified launch handoff

- Final CPU suite: **641 passed**, 12 warnings, 44.31 seconds. Includes an
  interrupted tiny training run followed by epoch-boundary resume, separate
  validity masks, evaluation contracts, and BOM-aware protected pointer reading.
- Real-data preflight passed for **3,837 training / 859 validation images**.
  Validation RGB/classification file contents match the fixed independent V3
  replay. Training source binding covers approved IDs and path/size/mtime
  metadata, not a fresh full byte hash of every training file.
- Train-only feasibility: 48 updates on one fixed training batch, loss
  **0.749761 to 0.114596** (84.7156% reduction), with nonzero encoder gradients.
  This is optimization evidence only, **not accuracy or generalization**.
  Proof weights were discarded before starting the full experiment.
- Native 1024-pixel CUDA smoke passed on one development image per city,
  peak allocated model memory **802,078,208 bytes**. The fresh decoder's smoke
  scores are not trained-candidate results.
- Epoch 1 was observed advancing through **760/960 training batches**.
  Initial NVIDIA sample: 47% utilization, 2,418 MiB total GPU memory used,
  58 C, 49.69 W. Utilization fluctuates; these are samples, not averages.
- Config, source snapshots, data/software binding and both proof reports are
  saved in the exact run. Checkpoints are created after each completed
  validation epoch; before epoch 1 completes there is no resumable checkpoint.
- Existing hourly automation now follows only this exact run, remains quiet
  for unchanged status, and never promotes it or resumes the rejected residual
  run. A failed run may resume only the same compatible latest checkpoint.
- All six-class and city gates remain unchanged. First full validation is
  pending at this handoff; **no classification improvement is claimed yet**.

## Purpose

Train an independent image-understanding model for ground, buildings, water,
roads, low vegetation and trees. The fair V3 six-class development baseline is
macro F1 0.5648374706 on 859 DC/Philadelphia tiles. The older V3/V4 classifiers
trained approximately 1,094 head parameters over a frozen adapter. The new
SegFormer-B0 encoder and decoder have 3,715,686 trainable parameters and receive
RGB only. Higher capacity is a testable hypothesis, not an accuracy claim.

The entire protected height model, its preprocessing/fusion and application
pointer remain unchanged. No predicted class is yet used to gate, clip, scale
or replace heights. Future overlay integration and full application verification
are separate release steps.

## Fixed data and comparison rules

- Reuse the approved DC/Philadelphia training index (3,837 images) and the same
  859 native 1024-pixel development images as the independently replayed V3.
- Preserve class mapping, unknown/background ignore index 255, independent
  image/class validity, raw image radiometry, water-aware/city-balanced sampling
  and the existing focal-loss class weights. Known invalid heights must never
  remove otherwise valid classification labels.
- The new dataset reads RGB and classification files only. Neither reference
  height nor a LiDAR image is an inference input; no model predictions are
  pasted into the supervision. The model's ImageNet normalization is fixed.
- Authenticate the approved index, sampling index, baseline, pretrained files,
  source code and evaluation content. Check exact validation IDs and class
  support counts against the old replay before comparing accuracy.
- This is a **new complete model recipe**, not an architecture-only causal
  ablation: encoder initialization, optimizer groups and normalization differ
  from V3. The unchanged evaluation makes outcome comparisons useful, but
  attributing gains to one architectural detail would require another control.
- NYC is excluded from this new classifier's fitting, not unseen by the whole
  historical system. Official GAMUS test was previously consumed and is not
  used again. External classification/height holdouts are not opened here.

## Preflight and training

The isolated entry point is `scripts/train_rgb_segmenter.py`, configured by
`configs/gamus_rgb_segmenter_v1.yaml`. Before the full run:

1. CPU contract and unit tests, pretrained encoder loading with no missing
   encoder tensors, protected artifact hashes and dataset checks.
2. A small fixed **training-only** learning proof: at least 48 steps, at least
   15% loss reduction, finite updates and nonzero encoder gradients.
3. Native 1024x1024 CUDA validation smoke test, class/support/evaluator checks
   and VRAM measurement. No resized/easier validation substituted for full data.
4. Seal the exact source/config/proofs. Start a fresh model from the pretrained
   encoder and fresh decoder, not from the memorized feasibility checkpoint.

The pilot is limited to eight epochs with early stopping, batch four and one
GPU workload at a time. A different batch/recipe requires explicit evidence
and new validated settings rather than an unrecorded resume mutation.
Epoch-boundary checkpoints, random states, config/source snapshots, a compact
status file and terminal completion/failure summary support safe recovery.
No code or configuration changes are allowed during a sealed active run.

## Acceptance and interpretation

Report final **six-class** macro F1/IoU and per-class precision, recall, F1 and
IoU, separately for DC and Philadelphia. Keep road-boundary tolerance and the
dark-pixel water-confusion proxy fixed; dark pixels are not true shadow labels.
Carry forward the predeclared V4 class/city acceptance thresholds from its
fixed V3 comparison. A better mean score does not excuse a collapsed class.
Keep the best macro-score checkpoint distinct from any best fully eligible
development checkpoint. Passing development gates does not authorize release
or establish generalization to other cities, sensors or landscapes.

Post-run: inspect every epoch and independent per-class evidence. If promising,
perform a separately authenticated replay before final external evaluation and
future overlay exposure. The protected application cannot load these standalone
checkpoints without a separately implemented and verified adapter.

## Launch and progress

Launch after all checks pass:

```powershell
.\scripts\run_rgb_segmenter_v1.ps1
```

Safe compatible resume only:

```powershell
.\scripts\run_rgb_segmenter_v1.ps1 -ResumeCheckpoint <exact-run>\checkpoint_latest.pt
```

Compact live viewer: `scripts/watch_rgb_segmenter_v1.ps1`. It reads the exact
active run recorded in `outputs/orchestration/rgb_segmenter_v1_active.json`,
with Windows delete-sharing enabled so it cannot block checkpoint/status
replacement. Closing the viewer does not stop training.

The existing hourly automation `msr-residual-accuracy-check` now tracks
the exact run above. It checks milestones, all class/city gates and saved
checkpoints, then pauses after reporting completion and saving an outcome
summary. No repeated short-interval polling is needed.

## Backbone source and licence

- ImageNet-only encoder: <https://huggingface.co/nvidia/mit-b0>
- Pinned revision: `80983a413c30d36a39c20203974ae7807835e2b4`.
- Files and checksums: `D:/MSRData/models/segformer-mit-b0-imagenet/backbone_manifest.json`.
- Download: approximately 14.4 MB of weights; no new large dataset download.
- Upstream NVIDIA SegFormer licence restricts use to non-commercial research
  or evaluation. Preserve notices and `LICENSE.upstream.txt`. A commercial
  product needs an appropriate licence or a separately licensed alternative;
  do not advertise this experiment as unrestricted commercial software.
- Primary model documentation: <https://huggingface.co/docs/transformers/model_doc/segformer>.

## Prior height experiment

Residual-height V1 stopped after five epochs without an eligible checkpoint.
It is preserved and must not be restarted by this monitor. Its targeted error
analysis is separate from this classifier. Improving identification alone is
not evidence that any building or canopy height became more accurate.
