# References and upstream resources

References identify methods, upstream software, datasets and applicable source
terms. A citation does not imply endorsement or waive the linked licence.
Access review: 28 September 2026. Original project measurements are traceable
through the dated records in [Source inventory](SOURCE_INVENTORY.md).

## Methods and models

1. Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*, CVPR 2023. [Paper](https://arxiv.org/abs/2301.00808), [upstream implementation](https://github.com/facebookresearch/ConvNeXt-V2), [specific timm weight card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k).
2. Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021. [Paper](https://arxiv.org/abs/2105.15203), [NVIDIA upstream](https://github.com/NVlabs/SegFormer), [MiT-B0](https://huggingface.co/nvidia/mit-b0).
3. Yang et al., *Depth Anything V2*, 2024. [Paper](https://arxiv.org/abs/2406.09414), [small Hugging Face model](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf). This release pins the small model, whose licence differs from some larger family members.

## Data

4. [HighBuild-1M dataset and source inventory](https://huggingface.co/datasets/feifei140729/HighBuild-1M). Mixed imagery sources; use the supplied source inventory rather than a blanket imagery licence.
5. Fogel et al., *Open-Canopy: Towards Very High Resolution Forest Monitoring*, 2024. [Paper](https://arxiv.org/abs/2407.09392), [dataset](https://huggingface.co/datasets/AI4Forest/Open-Canopy).
6. [GAMUS paper](https://arxiv.org/abs/2305.14914), [EarthNets multimodal segmentation repository](https://github.com/EarthNets/RSI-MMSegmentation). Retain local-unit and spatial-provenance limitations described in this handbook.
7. Xia et al., *OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping*. [Paper](https://arxiv.org/abs/2210.10732), [dataset release](https://zenodo.org/records/7223446), [regional attribution](https://open-earth-map.org/attribution.html).
8. [LINZ aerial imagery](https://registry.opendata.aws/nz-imagery/) and [LINZ elevation](https://registry.opendata.aws/nz-elevation/). Bay of Plenty products retain BOPLASS and LINZ attribution and acquisition metadata.
9. [Copernicus Sentinel data access](https://dataspace.copernicus.eu/explore-data/data-collections/sentinel-data) and [Copernicus DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM). Optical and terrain resolution are not interchangeable.
10. [USGS SRTM](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1). Used as the separate coarse hilly comparison reference.
11. [US3D / 2019 IEEE GRSS Data Fusion Contest](https://www.grss-ieee.org/community/technical-committees/2019-ieee-grss-data-fusion-contest/). Raw source imagery and reference rasters are omitted from this public package because redistribution is prohibited by the supplied terms.
12. [GeoDanmark spring orthophoto service](https://datafordeler.dk/dataoversigt/geodanmark-ortofoto/ortofoto-foraar-wms/) and [Danish data-use terms](https://www.klimadatastyrelsen.dk/om-klimadatastyrelsen/vilkaar-og-priser).
13. [PDOK aerial imagery service](https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/). See provider metadata and the retained HighBuild inventory for the Amsterdam source.

## Software and publication tooling

The implementation uses [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models), [Transformers](https://github.com/huggingface/transformers), [Rasterio](https://rasterio.readthedocs.io/), [FastAPI](https://fastapi.tiangolo.com/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Three.js](https://threejs.org/), [GeoTIFF.js](https://geotiffjs.github.io/), [React](https://react.dev/), [Next.js](https://nextjs.org/), [vinext](https://github.com/cloudflare/vinext), [Vite](https://vite.dev/), [Playwright](https://playwright.dev/) and [ReportLab](https://www.reportlab.com/). Package lockfiles and dependency metadata identify the checked versions. Each dependency retains its own licence.
