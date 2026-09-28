# Glossary and product dictionary

| Term | Meaning in this project |
| --- | --- |
| DSM | Surface elevation in an absolute datum, including objects above terrain. |
| DEM / DTM | Terrain elevation used as a datum; provider details determine what it represents. |
| nDSM / AGL | Height above local ground; zero is local ground, not sea level. |
| Relative depth | A scene-relative depth ordering/shape product without automatic metric scale. |
| GSD | Ground sampling distance per pixel; distinguish horizontal axes and projected/geographic units. |
| CRS | Coordinate reference system for horizontal coordinates; does not by itself establish a vertical datum. |
| Valid mask | Pixels supported by input/reference observations and the declared evaluation rules. |
| Nodata | Explicit absence of a valid value; not equivalent to zero metres or semantic ground. |
| Semantic class | A categorical predicted label; not a measurement of height. |
| Confidence | Model-derived score/probability; not a calibrated probability that height is correct. |
| Promotion | Acceptance as the new scientific/default model under a predefined protocol. |
| Integration | Activation of a model in the application; separate from benchmark promotion. |
| Protected model | Retained urban/guarded-vegetation baseline whose measured behavior is not silently replaced. |
| Presentation surface | Derived mesh suitable for display; retain the raw scientific export separately. |
| Development | Data used to choose models, thresholds, sampling or implementation behavior. |
| External test | Predeclared data with appropriate novelty, untouched until a frozen evaluation; then consumed. |
| Sealed artifact | Files bound to recorded hashes/configuration; renaming source means it is no longer the same source seal. |
| Macro F1 | Equal-class mean F1 under the specified absent-class policy. |
| Equal-region F1 | Average across named regions, preventing sample-rich regions from dominating selection. |

## Output interpretation

Raw height TIFF contains the learned height-above-ground product on the working
grid. Relative-depth TIFF preserves a separate foundation prediction. A DEM-aligned
absolute DSM combines compatible terrain and above-ground estimates. A GCP surface
depends on supplied point coverage and residual fit. Probability TIFFs and the
six-category preview have their own class/validity conventions. A displayed solid
building or smoothed hill is an interpretation of these products, not a new
ground-truth observation.

See [API](API.md) for the response fields and [Scientific contracts](SCIENTIFIC_CONTRACTS.md)
for requirements governing units, alignment and export behavior.
