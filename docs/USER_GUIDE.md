# Using the research workspace

Saved demonstrations are frozen historical products with their own provenance;
they are not recomputed when opened. For example, Copenhagen's saved metadata
identifies an earlier landscape checkpoint. New uploads use the released protected
height model. Compare model identities and protocols before comparing their values.

## First investigation

1. Start the viewer and select a bundled reference scene. The neutral start screen
   does not itself initiate image inference.
2. Rotate the surface and compare its texture with the original image. Use the
   source-image reference when an oblique perspective makes a feature ambiguous.
3. Inspect a point. Read its units and product type before interpreting the number.
   Above-ground height and absolute elevation answer different questions.
4. Enable six-class identification when the local API and classifier are available.
   Select Ground, Buildings, Water, Roads, Low vegetation or Trees to inspect their
   spatial coverage. A class highlight is not a corrected height measurement.
5. Explore with fly or walk navigation. The map shows camera position and direction;
   it is a navigation aid, not a separately surveyed map.
6. For metric terrain, draw a profile and inspect elevation, distance and slope.
   Download the profile CSV or the source raster product as appropriate.
7. If an eligible reference is present, inspect the signed error and scoped metrics.
   Preserve the scene identifier and reference description with any exported figure.

## Choosing inputs

Ordinary JPG/PNG imagery supports a relative relief interpretation. A TIFF file can
contain optical imagery, an elevation raster, or another numeric product; its file
extension alone is insufficient to establish meaning. The import path uses raster
metadata, dimensions and band properties. An image-only TIFF still needs inferred
heights or an external terrain source.

The API accepts JPG, JPEG, PNG, TIFF, JP2, WebP and BMP imagery subject to decoding
support. Automatic classification currently requires 8-bit RGB. Higher-bit-depth
height processing and six-class support are separate capabilities. Unsupported
classification must produce an explicit limitation rather than a fabricated label.

Large image inputs are read onto a bounded processing grid. A smaller processing
grid preserves geographic bounds but loses spatial detail. The metadata records
both original and processing sizes; inspect it before making a fine-resolution
claim. The API's 512 MiB file limit does not mean every image of that byte size is
fast or accurate.

## Calibration and reference comparison

Choose one calibration route: a supplied terrain DEM, supplied GCP CSV, or optional
automatic public terrain. A DEM must describe the intended ground datum and align
through a supported CRS transformation. A GCP table needs `elevation_m` and either
map coordinates `x,y` or image coordinates `row,col`.

Automatic public terrain is convenient for a demonstration, but its coarse grid
does not become high-resolution ground truth when resampled. The provider product
can represent a surface rather than bare earth. The user is responsible for the
datum and acquisition compatibility of a manually supplied reference.

A validation reference is distinct from a calibration input. Select nDSM when the
reference measures object height above ground, or DSM when it measures absolute
surface elevation. The application aligns it to the prediction grid and masks
invalid support. Never evaluate against the same values used to fit a calibration
and describe the result as independent validation.

## Display controls and measurements

Terrain Source/Smooth modes and vertical exaggeration change the displayed mesh.
Smooth/Soft is the current default for compatible terrain, with a gentle 1.5x
display multiplier for non-flat metric terrain. The source elevations remain the
basis of numerical inspection, profiles and exports. The source-mode view makes
that difference directly inspectable.

Building roofs remain flat optical-textured representations with model-derived
building-height estimates. Detailed roof-height estimation is explicitly
unconfigured. Facade textures and reconstructed solids should not be interpreted
as recovered architectural detail. Current scenes do not include the reverted
illustrative tree-trunk experiment.

Class labels may disagree with the geometry. The RGB classifier does not control
the protected height-routing masks or collision geometry. Use the model-details
panel to read accuracy boundaries; pending, failed and unsupported identification
are meaningful states, not alternate class predictions.

## Export checklist

| Question | What to retain |
| --- | --- |
| What quantity is this? | Product label, units, relative/above-ground/absolute interpretation |
| Where is it located? | CRS, transform, source bounds and processing-grid information |
| Which model produced it? | Model identity and artifact hash from metadata |
| Was it calibrated? | Terrain or GCP source, method and fit information |
| Was it evaluated? | Reference identity, mask, alignment and metric scope |
| Was the view exaggerated? | Keep display settings with screenshots; use raw values for numerical claims |

Saved API jobs are stored in the local `outputs/web_jobs/` directory. There is no
cloud account or promised durable backup. Copy the artifacts you need before
cleaning local outputs. See the security chapter for retention and access limits.
