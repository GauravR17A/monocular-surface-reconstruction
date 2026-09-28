# External six-class holdout: Christchurch OpenEarthMap v1

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: **preregistered design only; not downloaded, decoded, or evaluated**.

This holdout is reserved for one post-freeze comparison between the paired
DC+Philadelphia spatial-V3 head and an eligible hierarchical-V4 head. It is
not part of model selection. Christchurch is absent from the known
Monocular Surface Reconstruction GAMUS, HighBuild, OpenCanopy, and demo inventories, but this does
not prove absence from generic upstream pretraining corpora.

## Pinned source

- Dataset: OpenEarthMap v1
- Region: every released `christchurch` RGB tile with a same-stem label tile
- DOI: https://doi.org/10.5281/zenodo.7223446
- Published archive MD5: `64155d1dc9d3b68536063f79878e1a67`
- Source imagery attribution: AIRS / Land Information New Zealand, CC BY 4.0
- OpenEarthMap attribution: https://open-earth-map.org/attribution.html
- LINZ attribution guidance: https://www.linz.govt.nz/products-services/data/licensing-and-using-data/attributing-elevation-or-aerial-imagery-data

The repository's MIT software licence is not the imagery licence. Exact
source-layer attribution must be retained before redistribution.

## Frozen class crosswalk

| OEM id | OEM class | Monocular Surface Reconstruction class |
|---:|---|---|
| 0 | unknown / no-data | ignore (`255`) |
| 1 | bareland | ground (`0`) |
| 2 | rangeland / grass | low vegetation (`4`) |
| 3 | developed space / pavement | ground (`0`) |
| 4 | road | roads (`3`) |
| 5 | tree | trees (`5`) |
| 6 | water | water (`2`) |
| 7 | agriculture | ignore (`255`) in the primary result |
| 8 | building | buildings (`1`) |

Agriculture is ignored in the primary result because the source class can
include woody orchards and vineyards. A separately labelled sensitivity
result may map id 7 to low vegetation, but it cannot replace the primary
score. The report must also include the native 8-by-6 source-label versus
prediction table so merging cannot hide failures.

## One-shot procedure

1. First require V4 to pass every paired DC+Philadelphia class gate, state
   isolation check, independent validation replay, and bit-exact height-output
   audit.
2. Before acquisition or pixel decoding, write and externally timestamp a
   contract containing the DOI/version/archive checksum, region inclusion
   rule, crosswalk, frozen checkpoint/config/code hashes, radiometry, tiling,
   inference, metrics, gates, and no-tuning/no-reselection rule.
3. Verify the archive MD5, extract only Christchurch, pair files by exact stem,
   hash every selected RGB/label file, and seal a sorted manifest. Do not
   selectively omit difficult tiles.
4. Write a `CONSUMED.json` marker before the first pixel decode. Evaluate the
   frozen V3 and V4 checkpoints together on identical inputs in one invocation.
   A crash still counts as consumption and any exceptional rerun must be
   disclosed.
5. Publish and hash the complete report regardless of outcome. Only then may
   examples be viewed; select examples by a preregistered hash order rather
   than visual appeal.

## Required report

- Exact 6-by-6 confusion matrix (rows reference, columns prediction).
- Per-class tile/pixel support, precision, recall, F1, and IoU.
- Six-class macro precision, recall, F1, IoU, and overall accuracy.
- V4-minus-V3 deltas with a fixed-seed paired bootstrap over whole tiles.
- Native OEM 8-by-6 table, ignored share, and agriculture sensitivity result.
- Two-pixel road-boundary metrics and the explicitly labelled dark-pixel water
  false-positive proxy.
- Low-vegetation/tree confusion and combined-vegetation F1 as diagnostics only.
- No height accuracy claim: OpenEarthMap has no height truth. Report only
  V4-versus-protected height-output identity.

Before calling this a comprehensive six-class holdout, every mapped class must
occur in at least five tiles and 10,000 reference pixels. Inadequate coverage
is reported honestly and does not authorize swapping locations after results
are seen.

## Claim limit

Permitted wording: **"One-shot transfer to a preregistered Christchurch
benchmark, new to known Monocular Surface Reconstruction project data, under a fixed ontology
harmonization."**

This does not prove global generalisation, Christchurch population accuracy,
height accuracy, or absence from unknown upstream pretraining data.
