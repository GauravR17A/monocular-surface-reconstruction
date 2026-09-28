# Model and geospatial methods

## Relative geometry

Depth Anything V2 Small provides a pretrained relative-depth prior. The wrapper
handles image processing, inference and interpolation to the processing grid.
The robust normalization uses eligible finite pixels and the 2nd/98th percentiles
to form a clipped [0,1] score. A near/far convention is explicit, degenerate spans
produce a zero relative score, and invalid pixels remain missing.

This prior is target-independent. Height-label availability must not influence
normalization or the cached image geometry. Earlier cache audits locked current
RGB/prior/model identities and reproduced five deterministic examples, while
acknowledging that creation-time hashes were absent from the original cache.

## Base height network

`HeightNet` uses timm's ConvNeXt V2 Tiny multiscale encoder and a U-Net-like
decoder. At each level, the decoded field is bilinearly upsampled and concatenated
with the corresponding encoder feature map. Convolution blocks use GroupNorm and
GELU. A final softplus head produces nonnegative height; an auxiliary building
logit head provides a separate spatial signal.

Training uses geographic manifests, synchronized image/label transformations,
masked losses, mixed precision, gradient accumulation and resumable artifacts.
The initial full urban run was followed by error analysis rather than interpreting
one all-pixel score as sufficient evidence of useful building reconstruction.
Tall structures and background/support definitions were important failure sources.

## Protected surface refinement

`DomainGatedSurfaceNet` retains the base network and learns a compact adapter from
image, relative prior, base height and building evidence. Its three domain
probabilities sum to one: ground, building and vegetation. It also predicts a
bounded building residual, canopy height, refinement strength and a variance-like
quantity. The three groups are internal height-routing domains.

The active protected-vegetation mode preserves base building height within its
mixture while allowing a vegetation correction. In simplified notation, the
intermediate height is max(0, h_base + alpha_effective * (h_gate - h_base)). The
effective strength is reduced by the building-protection term. A separately
configured vegetation-expert contribution is applied only at its declared gate.
The exact computation is centralized in `fuse_surface_endpoint`.

This distinction explains an important diagnostic: improving a separate building
residual head need not improve the final height if that head is not used in the
active fusion formula. Likewise, a weak effective gate can transmit only a small
part of a supervision gradient. Head-level metrics are not substitutes for final
predicted-height metrics.

## Later height experiments

A residual decoder initialized at zero tested a richer correction pathway while
preserving the original prediction at initialization. Class-assisted experiments
tested whether frozen semantic probabilities helped an equally trained height
model beyond a neutral control. Low-surface losses and height-band sampling tested
specific tradeoffs. A semantic-expert design mixed six learned corrections using
frozen class probabilities.

These are preserved research alternatives. None established a candidate satisfying
all the required benefit and safety gates. The release keeps the protected height
model; the results chapter does not combine the best strata from different runs.

## Independent RGB classification

The visible classifier is a SegFormer B0-based six-class model initialized from the
ImageNet-pretrained MiT-B0 encoder. Raw RGB is scaled by 255 and normalized inside
the model. Class outputs are ground, buildings, water, roads, low vegetation and
trees. The active inference path uses native tiles up to 1024 pixels with 128-pixel
overlap, averages softmax probabilities and then takes argmax. Invalid image pixels
use class ID 255. No reference target is used at inference.

The V3 training package adds regional data/augmentation to improve transfer. It is
not an isolated causal test of any one augmentation. Its selected development
checkpoint improves OEM transfer but regresses on some GAMUS classes and road
boundaries. Automatic app integration does not change those results.

## Tiled height inference

Overlapping tiles are blended with spatial weights to reduce seams. Input
normalization, tile size, overlap, precision and padding policy are part of the
evaluation contract. They can change GroupNorm context and therefore output.
The full-native research benchmark and the app's 512/128 path must be labelled
separately. The optional minimal-padding audit improved canopy error but worsened
forest-ground error, so it was not silently enabled globally.

## Calibration and validation

DEM composition reprojects the supplied ground field to the prediction grid and
adds nonnegative object height where both supports are valid. Control-point
calibration estimates a scene-level affine scale/offset with robust trimming;
its fit error describes the supplied controls. Calibration does not solve hidden
ground, vertical-datum mismatch or acquisition-time change.

The reference evaluator aligns supported raster types, reports valid support and
computes signed errors and summary statistics. Instance evaluation matches
available building annotations and distinguishes measured from estimated heights.
Boundary and counting interpretations depend on the completeness of reference
annotations. The benchmark and scorecard histories give the exact acceptance rules.

## Presentation reconstruction

The viewer constructs terrain and building geometry for interaction. Terrain
smoothing uses positive normalized Gaussian weights, supports rectangular pixel
spacing, preserves missing data, and keeps source values separate. Fine-resolution
standalone DSMs without a ground reference have additional protection to avoid
flattening building structure. Map, collision and highlight behavior follow the
displayed geometry; exported source products retain their scientific meaning.
