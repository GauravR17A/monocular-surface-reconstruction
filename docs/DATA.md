# Data, provenance and geographic coverage

## Dataset roles

| Source | Role in this project | Principal boundary |
| --- | --- | --- |
| HighBuild-1M | Urban height training, building annotations and measured-building evaluation. | Mixed source licences; sparse/estimated annotations and missing per-tile geospatial metadata require explicit filtering. |
| Open-Canopy | Forest RGB and LiDAR-derived canopy supervision; retained official split information. | Mosaic/acquisition overlap, LiDAR-class masks and image/reference date differences matter. |
| GAMUS | Height/semantic experiments and six-category classification training/development. | Local HDF5 lacks spatial metadata; height units remain assumed in the older height experiments. |
| OpenEarthMap | External V1 classification testing and later multi-region V3 development. | Regional licence and ontology mapping; development regions are not final tests. |
| LINZ/BOPLASS Bay of Plenty | Reserved external metric-height package from orthophoto, DSM and DEM. | Prepared and sealed, but not yet evaluated. |
| Sentinel-2 / Copernicus / public terrain | Georeferenced hilly demonstration and terrain datum. | Coarse terrain, datum and surface-versus-bare-earth differences limit interpretation. |
| US3D / DFC2019 demonstration | One selected urban RGB/LiDAR comparison case. | A disclosed selected demonstration, not a population-level validation set. |
| Roof-reconstruction sources | Prepared research inputs for roof geometry experiments. | Experimental modules are retained; detailed roof reconstruction is not enabled. |

Source links, attribution and model terms are collected in
[References](REFERENCES.md) and [Third-party notices](../THIRD_PARTY_NOTICES.md).
The source package does not redistribute the full training archives.

## HighBuild preparation

The recorded initial full-data preparation used a city-disjoint, conservative
source-licence filter: 25,446 training tiles from five cities, 1,921 validation
tiles from four held-out cities and 2,310 test tiles from five held-out cities.
Those are counts for the pinned local preparation, not a claim about the latest
upstream dataset size. The acquisition script pins revision
`2f5d76f8c5b5b4b7e925871d46db7aae64fd3daa`.

The later corrected height benchmark is a different support: 160 complete urban
scenes with strict measured-height building support, plus 120 forest scenes.
Annotations marked estimated are excluded from the primary measured-building
score. COCO footprints and height validity are independently represented. Training
crops are selected to contain useful support, while evaluation uses the complete
fixed native grid.

The first urban error analysis showed a background-dominated aggregate and strong
tall-building underestimation. That motivated support correction and stratified
reporting. Earlier zero-filled/all-pixel scores remain historical results and
cannot substitute for the corrected measured-building baseline.

## Open-Canopy preparation

The project range-streamed selected image/LiDAR chips rather than downloading the
entire release. Stored decimetre heights are converted to metres. Chips retain
region, source image, acquisition dates, GSD and source URLs. The accepted expanded
subset contains 600 training, 120 validation and 120 test chips.

The provenance audit reports accepted samples rather than every rejected cached
candidate. It preserves official spatial-split information and distinguishes
coarse-region overlap from shared source mosaics. No single overlap statistic is
treated as proof of complete geographic independence. LiDAR unclassified/building
support must not be mistaken for valid forest height supervision.

## GAMUS preparation and exposure

The locked quality index contains 5,001 approved training tiles, 859 validation
tiles and 2,860 official-test tiles. All 8,720 RGB/prior pairs were inventoried and
hash-locked. Five sampled cache recomputations reproduced 5,242,880 float16 values.
That verifies the sampled computation and current identities, not missing
creation-time provenance for the entire cache.

For a later paired head experiment, DC/Philadelphia contributed 3,837 training
tiles and 859 development-validation tiles; 1,164 NYC tiles were excluded from that
head's training. New York was already known to the inherited project, so the
experiment does not establish system-unseen geography. The DC grid audit found all
359 validation tiles adjacent to training under an eight-neighbour rule. This
split is useful for development but is not a geographically isolated final test.

The published nominal resolution is 0.33 m, but local HDF5 files contain no CRS,
transform or per-file pixel-size proof. The raw AGL unit was not independently
confirmed in the reviewed project record. Older GAMUS height scores must therefore
retain their metre-assumed label. Extreme values above 200 were inventoried before
changing any cutoff; they include implausible class/height combinations.

## Holdout ledger

| Dataset / region | Recorded state | Allowed interpretation |
| --- | --- | --- |
| GAMUS official test | Consumed by earlier Stage-3 work. | Historical audit only; not an untouched test for later tuning. |
| NYC head-exclusion partition | Excluded from specified fresh heads, but prior system exposure exists. | Head-transfer scope, not complete-system novelty. |
| Christchurch OEM V1 | All 49 publicly labelled pairs evaluated once by the frozen V1 classifier. | A completed V1 external classification transfer test; now consumed. |
| OEM Duesseldorf, Dhaka, Chiang Mai challenge | Twelve label-enriched scenes inspected and evaluated. | Diagnostic mixed-class challenge; not random global sampling. |
| OEM Melbourne/Vienna V3 | Used for development selection. | Development transfer scores. |
| OEM Rosario/Monrovia | Recorded as reserved in the V3 outcome. | Not a published final result. Recheck consumption before any future use. |
| Bay of Plenty metric-height package | Sealed manifest exists; no evaluation plan or consumption marker at release preparation. | Prepared external evidence path, no measured model result. |

## Bay of Plenty protocol

The fixed orthophoto scene is `BC36_1000_2314` from 2018–2019 urban imagery. Paired
1 m LiDAR DSM/DEM products use item `BC36_10000_0302`. Their difference defines the
reference above-ground surface. The source grid is EPSG:2193. Capture ranges overlap,
but pixel-level simultaneous acquisition is not established.

Three supports were fixed before inference: all-valid, object reference above
2 m, and tall-object reference above 5 m. Negative reference artifacts are excluded;
there is no upper-height cutoff. The sealed preparation records 326,925 all-valid
pixels, 133,567 object pixels and 124,217 tall-object pixels.

This package is not used for routine development, calibration or scene selection.
A future evaluation must freeze candidate identities first and publish all
predeclared supports regardless of whether the result is favourable.

## Reproduction and redistribution

Public scripts describe acquisition and preparation, but downloading a dataset
does not grant uniform rights to all its imagery. Keep provider and regional
attribution with derivative figures. The two older Chicago smoke assets are omitted
from the public package because the source inventory excludes their imagery from
redistribution. Demonstration assets and restricted research model weights retain
separate notices. No universal commercial-data licence is claimed.

US3D/DFC2019 raw imagery and reference rasters are excluded from this public
package because the source terms prohibit redistribution. The historical
selected-scene numerical result remains documented; its visual comparison is
unavailable in the public viewer. The Copenhagen urban reconstruction demo is
separate and remains available.
