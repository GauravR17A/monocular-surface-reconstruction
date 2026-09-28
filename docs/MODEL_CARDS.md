# Model cards and artifact identity

## Protected above-ground height model

**Purpose:** estimate surface height above local ground from supported overhead
RGB imagery and a target-independent relative-depth prior. Intended for research,
demonstration and evaluation, not certified survey or safety-critical decisions.

**Architecture:** a ConvNeXt V2 Tiny feature encoder with a multiscale height
decoder and auxiliary building head, retained inside a guarded ground/building/
vegetation surface model. The active fusion is the protected-vegetation variant.

**Development lineage:** urban training on a geographically separated,
source-filtered HighBuild preparation, followed by multidomain work using
OpenCanopy. The protected August 29 checkpoint remains the application model.
Later residual and class-assisted experiments did not earn a replacement.

**Canonical evidence:** corrected complete-scene development evaluation gives
13.2426 m RMSE on measured building pixels and 8.3610 m on valid vegetation pixels.
These results do not establish a universal error bound or unseen-geography
acceptance. Refer to the results chapter for support counts and other metrics.

**Known limitations:** tall-height compression, domain shift, sparse/estimated
urban reference labels, source-date mismatch, pixel-scale uncertainty and
inference-context sensitivity. Shape agreement can remain weak after a reduction
in aggregate RMSE. Roof detail, individual-tree measurement and hidden-ground
recovery are not established capabilities.

**Restrictions:** the underlying ConvNeXt V2 pretrained weights are distributed
under CC BY-NC 4.0. The packaged height derivative remains research/non-commercial;
it is not described as an unrestricted commercial model. Training-source terms
and attribution remain separate. See the upstream model card and retained notice.

## RGB six-class classifier

**Purpose:** identify ground, buildings, water, roads, low vegetation and trees
from 8-bit RGB. It supports overlays and inspection independently of height.

**Architecture:** SegFormer B0 segmentation model initialized from the
ImageNet-pretrained `nvidia/mit-b0` encoder. The active artifact is V3 epoch 1,
selected by equal-region OEM development macro F1. Labels use IDs 0–5 and nodata 255.

**Input contract:** raw 8-bit RGB, scaled to [0,1], with ImageNet normalization
inside the model; image validity is independent of class/height labels. Native
tiles, up to 1024 pixels with 128-pixel overlap, are combined in probability space.

**Evidence:** 69.42% equal-region OEM macro F1, 73.29% pooled OEM macro F1 and
83.20% GAMUS macro F1 for the selected V3 development checkpoint. Some class/city
and road-boundary retention gates failed. Product integration did not certify it.
V1's Christchurch 61.63% macro F1 belongs to that older frozen artifact only.

**Limitations:** ground/low-vegetation confusion, region-dependent road quality,
merged boundaries, sensor/radiometry shift and unknown upstream pretraining
coverage. Class probability is not a calibrated probability of correctness.
Water labels do not establish water depth; tree labels do not identify species.

**Restrictions:** the NVIDIA SegFormer licence limits use to non-commercial
research/evaluation and requires its notices with redistribution. The complete
upstream licence is retained in `licenses/SEGFORMER.txt`.

## Relative-depth foundation

Depth Anything V2 Small supplies relative geometry. The upstream Small model is
Apache-2.0; this does not make the separate height/classification artifacts or all
imagery unrestricted. The downloader pins the upstream revision and checks local
file hashes. The prior's output is scale-ambiguous and is not independently a
metric elevation estimate.

## Public packaging

`model-artifacts.json` is the authoritative inventory of public inference files,
their expected sizes, download locations and SHA-256 values. Optimizer, RNG and
machine-specific training payloads are removed. The height artifact carries its
base architecture so it does not depend on a nested training-checkpoint path.

The packaging audit verifies exact equality of all 240 height tensors (31,428,489
values) and all 210 classifier tensors (3,716,205 values). A separate numerical
check compares model outputs. File hashes change when packaging changes; the new
hash is not presented as the original training-checkpoint hash. No retraining,
quantization or claimed accuracy improvement occurs in this step.

Historical source artifact identifiers remain in the dated evidence. Their
existence does not make private archive paths downloadable from this repository.
Only artifacts explicitly included in the publication manifest are part of the
public release.
