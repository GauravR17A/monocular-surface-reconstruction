# Project overview and current scope

Technical documentation — status as of 28 September 2026. Version 0.1.0 is the
initial public research package; the development history predates this release.

## The problem

Overhead imagery contains visible clues about roofs, vegetation, shadows and
terrain, but a single view does not uniquely determine physical height. Reliable
surface measurements normally require additional observations or reference data.
This project explores a lower-input alternative: estimate an above-ground surface
from one optical image, preserve its scientific limitations, and give the user a
connected environment for inspection, calibration, comparison and export.

The goal is an inspectable research pipeline. A visually convincing mesh does not
establish metric accuracy, and a georeferenced image alone does not supply the
ground elevation needed for an absolute surface model. The system exposes those
distinctions in its outputs and documentation.

## Intended users and workflow

The supported audience is researchers, students and technical reviewers working
with overhead imagery and geospatial surface estimates. An analyst can load a
saved demonstration, reconstruct a supported RGB image, add a terrain datum or
ground-control points where available, inspect local values and compare against
a supplied reference. Planning, forestry and infrastructure analysis are potential
downstream uses; no operational benefit in those applications has been established.

The complete workflow is image acquisition, input validation, relative-depth
extraction, learned height estimation, optional calibration, interactive
exploration, and export with provenance. Semantic identification is a separate
RGB-only branch. It does not silently rewrite the height field to fit class labels.

## What this project contributes

The project integrates an overhead-image height decoder and protected surface
refinement with a pretrained relative-depth prior. It adds independent validity
masks, geospatial alignment and calibration, scoped evaluation, strict candidate
comparison, and a browser workspace that distinguishes source measurements from
display geometry. The dated experiments document unsuccessful designs as well as
working components, including comparisons against matched controls.

Depth Anything V2, ConvNeXt V2, SegFormer, PyTorch, rasterio and Three.js are upstream
work and are credited as such. This project does not claim to have invented
monocular depth estimation, semantic segmentation, or 3D rendering. Its research
contribution is the implemented integration, training/evaluation work, failure
analysis and traceable handling of scientific outputs.

## Current delivery

| Area | Status in this snapshot |
| --- | --- |
| Height reconstruction | Implemented; protected model retained because later candidates did not satisfy all acceptance gates. |
| Six-class identification | Automatically available for supported inputs when the classifier is installed; regional weaknesses remain disclosed. |
| Geospatial products | Grid preservation, DEM composition, control-point calibration and scoped reference comparison implemented. |
| 3D exploration | Orbit, fly, walk, map, class selection, source-image reference, metric grid and surface-profile tools implemented. |
| Roof geometry research | Experimental modules preserved; fitted/relief roof changes are inactive in the current viewer. |
| Illustrative tree trunks | Reverted from the current interface; not a delivered accuracy improvement. |
| External metric-height validation | A preregistered Bay of Plenty package is prepared but unconsumed. |
| Scientific acceptance | Internal numerical and engineering evidence exists; independent expert acceptance and broad geographic reliability remain open. |

## Reading the record

Current chapters take precedence when explaining what this release does. Dated
historical records preserve the state at the time of each experiment. The September
13 head-only classifier work is different from the later independent RGB V1/V3
classification pipeline. Likewise, later user-interface integration is not a
retroactive pass of an earlier model-safety gate.

Status labels mean implemented, measured in a stated scope, exploratory, rejected,
or planned. They are not interchangeable. The release report records the actual
package verification; the results chapter reports experiments without treating
old test counts as a current acceptance certificate.
