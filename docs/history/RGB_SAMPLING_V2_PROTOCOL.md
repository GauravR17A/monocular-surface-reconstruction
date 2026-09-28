# RGB V2: controlled crop-sampling continuation

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: training launched on 2026-09-15; no accuracy improvement claimed yet.

## Verified launch handoff

Active pair: `experiments/20260915T154725409300Z_rgb_sampling_v2_pair`.
The earlier `20260915T154605013891Z_rgb_sampling_v2_pair` stopped during the
training-only preflight, before its first update: the legacy precision adapter
requires torch.device rather than a string. That failed attempt and source
snapshot were retained. A new adapter test passes, and the corrected code was
sealed into this fresh pair; the failed pair must not be resumed.

- Full pre-launch suite: 734 passed,12warnings. After the adapter correction,
  52 targeted trainer/dataset/gate tests passed, including the new backward-pass
  regression test. The entire suite was not rerun during active GPU training.
- Train-only proofs,24updates per arm: control loss0.09201 to0.04569 (50.34%
  reduction); targeted0.08028 to0.03020 (62.38%). Both had nonzero encoder
  gradients and their proof weights were discarded. These are optimization
  checks, not accuracy improvements or a valid comparison between arms.
- Full-nativeCUDA validation smoke passed before training; approved input
  checks cover3,837train/859validation images and matched validation content.
- Observed control epoch1 batch280/960, RTX4070 utilization72%,1385MiB total
  VRAM,64C,59.82W. These are an instantaneous sample, not sustained averages.
- Epoch-boundary checkpoints, atomic commits, source/config/binding snapshots
  and metrics are saved separately for each arm. No resumable epoch exists
  until the first full validation/checkpoint commit completes.
- Existing hourly monitor `msr-residual-accuracy-check` was updated
  to this exact pair. It reports meaningful changes, supports one compatible
  recovery, and pauses after the final comparison/outcome report.

Final training-only index:
`outputs/data_audits/rgb_targeted_v2_training/top_quartile/targeted_crop_index_v2.json`
SHA256 `fc439c45e852b65135c791652cb38e460a5a2e1a8295223247c557997374cafa`.
The436.6second scan covered every approved training tile; quantile refinement
reused cached counts without decoding more data. Under unchanged tile weights,
expected crop ground pixels increase14.22% and lowvegetation12.15% overall.
Water pixel exposure falls21.86%, which is a real tradeoff guarded in validation;
exposure changes are not predicted score improvements. Report is the sibling
`training_crop_exposure_report_v2.json`. No validation/test/external/height
pixels were decoded during this training-index audit.

## Question and scope

Does selecting richer ground/low-vegetation TRAINING windows improve the known
DC identification weakness without materially reducing other class/city scores?
This is not a global generalisation claim, a height experiment, or an app release.
Christchurch has already been consumed and is excluded from every part of this run.
Official GAMUS test, NYC and the separate external height holdout remain excluded.

## Fixed paired comparison

Both arms initialize from the exact V1 guarded epoch-eight checkpoint:
`4940b2d4ae8817370384ccb54da9f73aee1b6bf3614435d8c077bf3e3363d021`.
The control uses ordinary random crops. The candidate mixes 50% ordinary crops
with 50% target-aware crops from the selected tile. Targets are ground and low
vegetation, chosen uniformly among targets with eligible windows. Eligible
windows have at least 1% target pixels and are in the target's upper quartile
within that tile. The final training-only index/report records exact rules and
expected exposure changes before either arm trains.

Unchanged: 3,837 approved DC/PHL training tiles; V1 city-mass-preserving water-hit
tile weights; shared epoch-by-epoch tile draws; raw RGB; independent validity
masks; 384px training crop size; flips/rotations; architecture; focal-loss weights.
Private target RNG preserves ordinary proposals and geometric augmentation RNG.
Two data workers are required for the tested epoch-boundary resume contract.

Both get four complete epochs, 960 batches per epoch, batch four, CUDA bf16.
Optimizer/scheduler reset identically: encoder LR 2e-5, decoder LR 2e-4,
weight decay .01, 50 warmup updates, cosine floor .1. No score-driven early stop
that gives one arm more compute. A nonfinite update or runtime failure stops
execution and requires compatible checkpoint recovery, not hidden settings changes.

Validation is always the same 859 full native1024 DC/Philadelphia development
tiles; file/grid/support identities must match the independently replayed V1.
No resizing or favourable-crop evaluation. Sequential arms share one GPU.

## Decision rules frozen before training

Against V1 AND the same-epoch control, safety permits at most:

- 1 percentage point lower class F1, for every class overall and in each city;
- 0.5 point lower macro F1 per scope;
- 1 point lower two-pixel road-boundary F1;
- 0.1 point higher dark-non-water false-water rate.

Required benefit versus V1: DC low-vegetation F1 at least +5 points; mean DC
ground/low-vegetation F1 strictly greater than +2.5 points.
Required benefit versus control: DC low-vegetation F1 at least +2 points and
the same joint mean at least +1 point. These are pilot acceptance thresholds,
not promises or sufficient production quality.

The PRIMARY experiment decision uses epoch4 against epoch4. Earlier same-epoch
comparisons and best-safe checkpoints are exploratory. Safety, useful improvement,
and experiment victory are reported separately. A higher mean alone cannot win.
This single-seed pilot does not establish statistical reliability; a successful
result needs replication, independent replay, and later a fresh geographic test.

## Proofs, preservation and recovery

Before full training: CPU tests, source/data/config hashes, 24-step fixed training
batch optimization per arm (at least5% loss reduction and nonzero encoder gradient),
plus nativeCUDA validation smoke. Proof weights are discarded.

New paired experiments live under `experiments/*_rgb_sampling_v2_pair`, with
separate control/targeted subfolders. Immutable committed epoch files, hash
receipts, optimizer/scheduler/RNG states, source/config snapshots and metrics
allow safe epoch-boundary recovery. Atomic commit records are authoritative;
latest/best aliases can be rebuilt after interruption. Uncommitted retries are
retained. Proof reports are bound to recipe identity.

No sealed V1 source/config or protected height checkpoint/pointer is edited.
No six-class output affects heights in this experiment. No automatic app promotion.

Launch: `scripts/run_rgb_sampling_v2.ps1`.
Resume: same script with `-ResumePair <exact saved paired experiment>`.
Status: `outputs/orchestration/rgb_sampling_v2_active.json` and the paired run's
`status.json`. Final results: `comparison_report.json`.
