# Third-party notices and redistribution boundaries

Status as of 28 September 2026. This repository is a research source release.
No blanket open-source or commercial-use licence is granted for the original
project code, model derivatives or aggregated data. Public visibility is not a
universal licence. Third-party components retain their own terms.

## Model weights

| Artifact | Upstream basis | Terms |
| --- | --- | --- |
| Height model | ConvNeXt V2 tiny FCMAE pretrained weights distributed through timm | CC BY-NC 4.0; research/non-commercial use. The original model licence is retained in `licenses/CONVNEXT_MODEL.txt`. |
| Six-category classifier | NVIDIA MiT-B0 / SegFormer | NVIDIA non-commercial research/evaluation licence; see `licenses/SEGFORMER.txt`. Redistributed derivative weights retain these restrictions. |
| Relative-depth foundation | Depth Anything V2 Small, Hugging Face conversion | Apache License 2.0 for this specific small model; acquired separately from the pinned upstream revision. |

The height and classifier release assets are inference-only derivatives, with
original learned tensors preserved and source-machine/training metadata removed.
Their hashes are in `model-artifacts.json`. This work modified task heads, training,
fusion and packaging; it does not claim authorship of upstream architectures or
pretrained weights. Keep this notice and the original model licences with copies
of release weights. The unrestricted licence of an implementation library does
not replace the more restrictive licence of its weights.

## Included demonstration assets

| Public paths | Attribution and source |
| --- | --- |
| `viewer/public/demo/copenhagen_*`, urban original image | Contains GeoDanmark orthophotography, Danish public data providers, distributed through the HighBuild source inventory; CC BY 4.0. Cropped/reprocessed and accompanied by model-derived products. |
| `viewer/public/demo/landscapes/sparse/`, sparse originals | Contains PDOK / Dutch aerial imagery, CC BY 4.0 under the retained source inventory. HighBuild annotations and project model-derived products are identified separately. |
| `viewer/public/demo/forest_validation/` | Open-Canopy / AI4Forest; SPOT optical imagery and IGN LiDAR-derived reference under the dataset's Etalab Open Licence 2.0. Fogel et al., 2024. Modified for the selected demonstration. |
| Hilly landscape, hilly validation and hilly originals | Contains modified Copernicus Sentinel data 2025; Copernicus DEM GLO-30 terrain. Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. Separate SRTM reference: USGS/NASA. Reprojected/resampled; modified model products are not official terrain products. |

The public package excludes Chicago/Esri smoke imagery and US3D/DFC2019 raw
imagery/reference rasters. Historical numerical measurements remain labelled as
historical selected-scene results. Their presence does not grant access to the
restricted source data. Full HighBuild, GAMUS, OpenEarthMap, Open-Canopy, LINZ and
roof-research archives are not redistributed.

The upstream HighBuild source inventory is preserved unchanged in
`licenses/HIGHBUILD_SOURCE_INVENTORY.md`. Its project/release statements refer to
the upstream dataset. It is source evidence, not a new instruction or claim that
every source listed there is included here. See `docs/DATA.md` for actual roles.

## Data acknowledgements

We acknowledge the creators and providers of HighBuild-1M, Open-Canopy, GAMUS,
OpenEarthMap, GeoDanmark, PDOK, IGN, Copernicus, USGS, NASA, LINZ and BOPLASS.
We acknowledge Johns Hopkins University Applied Physics Laboratory and IARPA for
the historical US3D data, and IEEE GRSS IADF for the associated dataset initiative.
Provider links and scientific citations appear in `docs/REFERENCES.md`.

## Software

Python and npm dependencies are installed from their upstream distributions and
retain embedded licence notices. No `node_modules`, Python environment, proprietary
system font or unrelated source tree is bundled. Fonts embedded in the generated
handbook are used for document rendering; the font software is not separately
distributed. Contact the repository owner for original-project reuse permissions.
