# Imagery Source Licenses and Attribution

This file documents the aerial/satellite imagery sources used or configured by
this project. It is intended for a NeurIPS 2026 Evaluations & Datasets data
release package.

NeurIPS 2026 E&D requires hosted datasets to be accessible to reviewers and to
include Croissant metadata with core and Responsible AI fields. For this dataset,
the Croissant metadata should reference this file from `prov:wasDerivedFrom`,
`rai:dataCollection`, and the dataset/license documentation fields.

This is an engineering license inventory, not legal advice. Before public
release, re-check the official terms on the access date used in the paper and
keep a frozen copy of the source URLs/capabilities metadata.

## Release Rule

The imagery in this dataset is multi-source. Do not publish the whole image
collection under a single permissive license such as CC BY 4.0 unless every
included image source is compatible with that license.

Recommended release policy:

- Release only sources marked `OK` or `OK-CONDITIONAL` after the listed
  conditions are satisfied.
- Exclude or regenerate sources marked `EXCLUDE` before the public NeurIPS
  dataset release.
- Keep each city/source subset separable in the hosted dataset and in Croissant
  `FileSet` metadata.
- Include all attribution strings in the dataset README, paper appendix, and
  Croissant provenance fields.

## Current Processed Data Source Map

The following mapping is inferred from `src/sources/*.py` and the current
`data/processed/images/*` directories.

| Processed directories | Source key | Imagery provider / service | Release status |
|---|---:|---|---|
| `America_NewYork_Filtered`, `Chicago_buildings_full`, `LosAngeles_Downtown_buildings_full`, `NewYork_Manhattan_buildings_full`, `SanFrancisco_FiDi_buildings_full`, `Seattle_Downtown_buildings_full` | `usa` | Esri ArcGIS Online World Imagery | `EXCLUDE` |
| `France_Lyon_buildings_full`, `France_Marseille_buildings_full`, `France_Strasbourg_buildings_full`, `France_Toulouse_buildings_full`, `paris_buildings_with_height_Filtered` | `fra` | IGN / Geoportail orthophotos | `OK` |
| `Japan_Osaka_buildings_full`, `Japan_Osaka_buildings_Filtered` | `jpn` | GSI seamless aerial photo tiles | `OK-CONDITIONAL` |
| `Amsterdam_buildings_Filtered` | `nld` | PDOK `luchtfotorgb` / `Actueel_orthoHR` | `OK` |
| `HKG_Core` | `hkg` | Hong Kong LandsD Imagery Map API | `OK-CONDITIONAL` |
| `TWN_Buildings_ready`, `TWN_Buildings_ready_Filtered` | `twn` | Taiwan NLSC PHOTO2 WMTS | `EXCLUDE` unless permission is confirmed |
| `berlin_all_buildings_height_Filtered` | `deu` | Brandenburg/Berlin DOP20c WMS | `OK` |
| `Germany_Frankfurt_buildings_full` | `deu` | Hessen DOP20 WMS | `OK` |
| `Germany_Munich_buildings_full` | `deu` | Bavaria DOP20 WMS | `OK` |
| `Oceania_Sydney_buildings_full` | `syd` | NSW Imagery MapServer | `OK-CONDITIONAL` |
| `Oceania_Melbourne_buildings_full` | `mel` | City of Melbourne 2020 true ortho imagery | `OK-CONDITIONAL` |
| `Africa_CapeTown_buildings_Filtered` | `afr` | City of Cape Town Aerial Imagery 2024 MapServer | `EXCLUDE` from unrestricted public release |
| `sao_paulo_exact_Filtered` | `bax` | GeoSampa Ortofotos 2020 RGB WMS | `OK-CONDITIONAL` |
| `Canada_Vancouver_buildings_Filtered` | `van` | City of Vancouver Orthophotos 2022 | `OK` |
| `Canada_Toronto_Filtered` | `tor` | City of Toronto 2023 orthophoto MapServer | `OK` |
| `Denmark_Aarhus_Filtered`, `Denmark_Copenhagen_Filtered`, `Denmark_Odense_Filtered` | `dam` | GeoDanmark Ortofoto / Datafordeler WMTS | `OK` |

Implemented but not present in the current processed data:

| Source key | Imagery provider / service | Release status |
|---:|---|---|
| `nzl` | LINZ Basemaps aerial imagery | `OK` |
| `chn` | Tianditu imagery tiles | `EXCLUDE` unless separately licensed |

## Source License Details

### Esri World Imagery (`usa`) - EXCLUDE

- Code: `src/sources/xyz_usa.py`
- Service URL: `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer`
- Official terms found: Esri World Imagery is licensed under the Esri Master
  License Agreement. Esri states the ordinary World Imagery layer is not
  intended for exporting tiles for offline use; the export layer is for small
  offline use in ArcGIS contexts.
- Dataset release decision: Do not redistribute image chips generated from this
  source in a public NeurIPS dataset.
- Required action: Regenerate all USA image subsets from a redistributable
  source such as USDA NAIP/USGS public imagery, or exclude the USA directories
  from the public release.

Affected current directories:

- `America_NewYork_Filtered`
- `Chicago_buildings_full`
- `LosAngeles_Downtown_buildings_full`
- `NewYork_Manhattan_buildings_full`
- `SanFrancisco_FiDi_buildings_full`
- `Seattle_Downtown_buildings_full`

### IGN / Geoportail Orthophotos (`fra`) - OK

- Code: `src/sources/xyz_fra.py`
- Service URL: `https://data.geopf.fr/wmts`
- Layer: `ORTHOIMAGERY.ORTHOPHOTOS`
- License: Licence Ouverte / Open Licence version 2.0 (Etalab 2.0), for public
  IGN open data including BD ORTHO / ortho-imagery.
- Redistribution: allowed with attribution.
- Attribution: `Contains orthophotography from IGN / Geoportail, Licence Ouverte / Etalab 2.0.`
- Croissant `prov:wasDerivedFrom`: `https://data.geopf.fr/wmts`

### GSI Seamless Photo (`jpn`) - OK-CONDITIONAL

- Code: `src/sources/xyz_jpn.py`
- Service URL: `https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg`
- Dataset: GSI Tiles, latest seamless aerial photo (`全国最新写真（シームレス）`).
- License / terms: Government of Japan Standard Terms of Use 2.0 / GSI content
  terms, with source citation. GSI notes that GSI tiles can include tiles from
  third-party organizations or legally restricted survey products.
- Redistribution: acceptable only after checking that the specific seamless
  photo tiles used for Osaka do not require an additional permission beyond
  source citation.
- Attribution: `Source: Geospatial Information Authority of Japan (GSI) / GSI Tiles. Modified for dataset generation.`
- Croissant `prov:wasDerivedFrom`: `https://maps.gsi.go.jp/development/`

### PDOK Netherlands Orthophotos (`nld`) - OK

- Code: `src/sources/xyz_nld.py`
- Service URL: `https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/Actueel_orthoHR/`
- Dataset/layer: `Actueel_orthoHR`
- License: use the dataset-specific Nationaal Georegister/PDOK metadata; PDOK
  commonly uses CC BY 4.0 for geographic works and instructs users to follow the
  NGR metadata for the specific dataset.
- Redistribution: allowed if the dataset metadata license is CC BY 4.0.
- Attribution: `Contains PDOK / Dutch aerial imagery data, CC BY 4.0; see PDOK/NGR metadata.`
- Croissant `prov:wasDerivedFrom`: `https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/`

### Hong Kong LandsD Imagery Map API (`hkg`) - OK-CONDITIONAL

- Code: `src/sources/xyz_hkg.py`
- Service URL: `https://mapapi.geodata.gov.hk/gs/api/v1.0.0/xyz/imagery/WGS84/`
- Provider: Lands Department, Government of the Hong Kong SAR.
- Official terms found: CSDI/LandsD Map API terms allow use of data through the
  API subject to terms, rate limits, IP rights notice, and attribution. The API
  documentation requires Lands Department attribution and copyright notice.
- Risk: LandsD has separate notices for downloadable aerial photographs that
  restrict third-party dissemination. Because this project stores and republishes
  image chips rather than only displaying live map tiles, confirm that the Map
  API / CSDI terms cover stored derivative image redistribution.
- Attribution: `Aerial Photograph from Lands Department, Government of the Hong Kong SAR.`
- Release condition: Include the attribution and, if required by LandsD, the
  Lands Department logo/copyright notice. Obtain written confirmation if possible.
- Croissant `prov:wasDerivedFrom`: `https://tools.csdi.gov.hk/csdi-webpage/apidoc/ImageryMapAPI`

### Taiwan NLSC PHOTO2 (`twn`) - EXCLUDE unless permission is confirmed

- Code: `src/sources/xyz_twn.py`
- Service URL: `https://wmts.nlsc.gov.tw/wmts/PHOTO2/default/GoogleMapsCompatible/`
- Provider: National Land Surveying and Mapping Center (NLSC), Taiwan.
- Official terms found: NLSC terms allow showing captured/extracted service
  contents on the Internet, videos, print advertisements, and theses for legal
  purposes, but state that bulk download is forbidden.
- Risk: this dataset generation is a bulk tile extraction and public
  redistribution workflow.
- Dataset release decision: Exclude Taiwan imagery from the public release
  unless NLSC confirms redistribution is permitted for this derived dataset.
- Attribution if permission is obtained: `Source: National Land Surveying and Mapping Center (NLSC), Taiwan.`

### Germany Berlin / Brandenburg DOP20c (`deu`) - OK

- Code: `src/sources/xyz_deu.py`
- Service URL: `https://isk.geobasis-bb.de/mapproxy/dop20c/service/wms`
- Provider: Landesvermessung und Geobasisinformation Brandenburg, with Berlin
  data attribution where applicable.
- License: Datenlizenz Deutschland - Namensnennung - Version 2.0
  (`dl-de/by-2-0`).
- Redistribution: allowed with attribution.
- Attribution: `© GeoBasis-DE/LGB, dl-de/by-2-0; © Geoportal Berlin, dl-de/by-2-0 (Daten geändert)`
- Croissant `prov:wasDerivedFrom`: `https://isk.geobasis-bb.de/mapproxy/dop20c/service/wms`

### Germany Hessen DOP20 (`deu`) - OK

- Code: `src/sources/xyz_deu.py`
- Service URL: `https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-images.ows`
- Layer: `he_dop20_rgb`
- Provider: Hessische Verwaltung fuer Bodenmanagement und Geoinformation (HVBG).
- License: Datenlizenz Deutschland - Zero - Version 2.0 (`dl-zero-de/2.0`).
- Redistribution: allowed without attribution requirement under dl-zero, but
  citation is recommended for provenance.
- Attribution / citation: `Source: Geoportal Hessen / HVBG, dl-zero-de/2.0.`
- Croissant `prov:wasDerivedFrom`: `https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-images.ows`

### Germany Bavaria DOP20 (`deu`) - OK

- Code: `src/sources/xyz_deu.py`
- Service URL: `https://geoservices.bayern.de/od/wms/dop/v1/dop20`
- Layer: `by_dop20c`
- Provider: Bayerische Vermessungsverwaltung.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Attribution required by provider: `Bayerische Vermessungsverwaltung - www.geodaten.bayern.de`
- Croissant `prov:wasDerivedFrom`: `https://geoservices.bayern.de/od/wms/dop/v1/dop20`

### NSW Imagery (`syd`) - OK-CONDITIONAL

- Code: `src/sources/xyz_syd.py`
- Service URL: `https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_Imagery/MapServer`
- Provider/copyright in service metadata: Department of Customer Service, NSW.
- License status: the MapServer metadata gives copyright but not a clear open
  redistribution license in the codebase.
- Release condition: confirm the applicable NSW open data license or obtain a
  redistribution statement before including Sydney image chips in the public
  release.
- Attribution: `© Department of Customer Service, NSW; contains external imagery sources as listed in NSW_Imagery service metadata.`

### City of Melbourne 2020 Aerial Imagery (`mel`) - OK-CONDITIONAL

- Code: `src/sources/xyz_mel.py`
- Service URL: `https://gisags.melbourne.vic.gov.au/server_wa/rest/services/AerialImageryWGS84/AerialImage2020_WGS84/MapServer`
- Dataset: City of Melbourne 2020 Aerial Imagery true ortho.
- License: source portal metadata lists CC BY / CC BY-compatible terms. Confirm
  the exact version (`CC BY 4.0` versus another CC BY variant) before final
  Croissant publication.
- Redistribution: allowed if released under the source CC BY terms.
- Attribution: `Contains City of Melbourne 2020 Aerial Imagery, CC BY.`
- Croissant `prov:wasDerivedFrom`: `https://data.melbourne.vic.gov.au/explore/dataset/2020-aerial-imagery-true-ortho/`

### City of Cape Town Aerial Imagery 2024 (`afr`) - EXCLUDE from unrestricted public release

- Code: `src/sources/xyz_afr.py`
- Service URL: `https://cityimg.capetown.gov.za/erdas-iws/esri/GeoSpatial%20Datasets/rest/services/Aerial%20Imagery_Aerial%20Imagery%202024/MapServer`
- Provider/service text: City of Cape Town aerial imagery 2024. The service
  states that the original imagery is property of CCT and that there are no
  restrictions on the digital file for non-commercial purposes.
- Risk: non-commercial-only permissions are not compatible with an unrestricted
  public ML dataset license.
- Dataset release decision: either exclude this subset from the unrestricted
  public release, release it as a clearly separate non-commercial subset, or
  obtain written permission for dataset redistribution.
- Attribution: `Aerial imagery property of City of Cape Town (CCT), 2024.`

### GeoSampa Sao Paulo Ortofotos 2020 (`bax`) - OK-CONDITIONAL

- Code: `src/sources/xyz_bax.py`
- Service URL: `https://raster.geosampa.prefeitura.sp.gov.br/geoserver/geoportal/wms`
- Layer: `ORTO_RGB_2020`
- Provider: Prefeitura Municipal de Sao Paulo / GeoSampa.
- License: Creative Commons Attribution Share-Alike 4.0 (CC BY-SA 4.0), per
  GeoSampa license notice.
- Redistribution: allowed with attribution and share-alike terms.
- Release condition: keep the Sao Paulo imagery subset under CC BY-SA 4.0 and
  do not label it as merely CC BY.
- Attribution: `Source: GeoSampa / Prefeitura Municipal de Sao Paulo, CC BY-SA 4.0.`
- Croissant `prov:wasDerivedFrom`: `https://geosampa.prefeitura.sp.gov.br/`

### City of Vancouver Orthophotos 2022 (`van`) - OK

- Code: `src/sources/xyz_can.py`
- Service URL: `https://tiles.arcgis.com/tiles/qrcTTRTwUoS8N47o/arcgis/rest/services/Orthophotos_2022/MapServer`
- Dataset: City of Vancouver Orthophoto imagery 2022.
- License: Open Government Licence - Vancouver.
- Redistribution: allowed with attribution under the open data terms.
- Attribution: `Contains information licensed under the Open Government Licence - Vancouver; source: City of Vancouver Orthophotos 2022.`
- Croissant `prov:wasDerivedFrom`: `https://opendata.vancouver.ca/explore/dataset/orthophoto-imagery-2022/`

### City of Toronto Orthophotos 2023 (`tor`) - OK

- Code: `src/sources/xyz_can.py`
- Service URL: `https://gis.toronto.ca/arcgis/rest/services/basemap/cot_ortho_2023_color_10cm/MapServer`
- Dataset: City of Toronto 2023 colour orthophoto, 10 cm.
- License: Open Government Licence - Toronto.
- Redistribution: allowed with attribution.
- Attribution: `Contains information licensed under the Open Government Licence - Toronto; source: City of Toronto orthophoto imagery.`
- Croissant `prov:wasDerivedFrom`: `https://gis.toronto.ca/arcgis/rest/services/basemap/cot_ortho_2023_color_10cm/MapServer`

### GeoDanmark Ortofoto / Datafordeler (`dam`) - OK

- Code: `src/sources/xyz_dam.py`
- Service URL: `https://wmts.datafordeler.dk/GeoDanmarkOrto/orto_foraar_webm/1.0.0/WMTS`
- Dataset: GeoDanmark Ortofoto foraar Web Mercator WMTS.
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Redistribution: allowed with attribution.
- Attribution required by GeoDanmark: `@geodanmark` with a link to the GeoDanmark data terms.
- Operational note: the WMTS requires API-key/OAuth access. Do not publish the
  `SDFI_API_KEY`; publish only the derived data and source metadata.
- Croissant `prov:wasDerivedFrom`: `https://datafordeler.dk/dataoversigt/geodanmark-ortofoto/ortofoto-foraar-web-mercator-wmts/`

### LINZ Basemaps Aerial (`nzl`) - OK

- Code: `src/sources/xyz_nzl.py`
- Current data: no processed directory in this release.
- Service URL: `https://basemaps.linz.govt.nz/v1/tiles/aerial/WebMercatorQuad/`
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), with
  LINZ Basemaps attribution requirements and contributor attribution.
- Attribution: `Sourced from LINZ. CC BY 4.0. Contains LINZ Basemaps aerial imagery and contributors.`
- Croissant `prov:wasDerivedFrom`: `https://www.linz.govt.nz/products-services/data/licensing-and-using-data/attributing-linz-basemaps-data`

### Tianditu (`chn`) - EXCLUDE unless separately licensed

- Code: `src/sources/xyz_chn.py`
- Current data: no processed directory in this release.
- Service URL: `https://t0.tianditu.gov.cn/DataServer?T=img_w...`
- License status: not audited in this project.
- Dataset release decision: do not include Tianditu-derived imagery in the
  public release without a separate redistribution license.

## Recommended Public Release Subsets

If publishing now, the lowest-risk public release should include only:

- France: `fra`
- Netherlands: `nld`
- Germany: `deu` for Berlin/Brandenburg, Hessen, Bavaria
- Vancouver: `van`
- Toronto: `tor`
- Denmark: `dam`
- Sao Paulo: `bax`, but keep CC BY-SA 4.0 terms separate
- Japan: `jpn`, after tile-specific source citation checks
- Hong Kong: `hkg`, after Map API stored-derivative redistribution check
- Melbourne: `mel`, after exact CC BY version check

Exclude or regenerate before unrestricted public release:

- USA / Esri World Imagery (`usa`)
- Taiwan NLSC PHOTO2 (`twn`) unless permission is confirmed
- Cape Town 2024 (`afr`) unless released as non-commercial-only or permission is obtained
- Sydney NSW imagery (`syd`) until an explicit redistributable license is confirmed
- Tianditu (`chn`) if ever used

## Suggested Dataset-Level License Text

Use this text in the dataset README and hosting page instead of assigning a
single false license:

> This dataset is a multi-source derived dataset. Image chips are licensed under
> the terms of their respective upstream imagery providers, as documented in
> `data/IMAGERY_LICENSES.md`. Building geometry, height labels, masks, and
> metadata have separate source provenance documented in the dataset card and
> Croissant metadata. Users must comply with the license and attribution
> requirements for each source subset.

For a NeurIPS-hosted package, prefer splitting files by source, for example:

- `images/fra/...`
- `images/deu_berlin_brandenburg/...`
- `images/deu_hessen/...`
- `images/deu_bavaria/...`
- `images/nld/...`
- `images/can_vancouver/...`
- `images/can_toronto/...`
- `images/dam/...`
- `images/bax/...`

Then assign per-subset license/provenance in Croissant `FileSet` entries.

## Sources Consulted

- NeurIPS 2026 E&D Call: `https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets`
- NeurIPS 2026 E&D Hosting Guidelines: `https://neurips.cc/Conferences/2026/EvaluationsDatasetsHosting`
- Esri World Imagery item / terms notice: `https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9`
- IGN / Geoportail orthophoto open data notices: `https://data.geopf.fr/wmts`, `https://www.data.gouv.fr/`
- GSI tiles and terms: `https://maps.gsi.go.jp/development/`, `https://www.gsi.go.jp/ENGLISH/page_e30286.html`
- PDOK copyright and NGR metadata guidance: `https://www.pdok.nl/copyright`
- Hong Kong CSDI / LandsD Map API docs and terms: `https://tools.csdi.gov.hk/csdi-webpage/apidoc/ImageryMapAPI`
- Taiwan NLSC terms: `https://maps.nlsc.gov.tw/pro/use_clause_en.jsp`
- Brandenburg/Berlin DOP20c metadata: `https://geobroker.geobasis-bb.de/`
- Hessen DOP20 metadata: `https://www.geoportal.hessen.de/`
- Bavaria DOP20 metadata: `https://geodatenonline.bayern.de/geodatenonline/seiten/wms_dop20cm`
- NSW Imagery MapServer: `https://maps.six.nsw.gov.au/arcgis/rest/services/public/NSW_Imagery/MapServer`
- City of Melbourne 2020 Aerial Imagery: `https://data.melbourne.vic.gov.au/explore/dataset/2020-aerial-imagery-true-ortho/`
- City of Cape Town Aerial Imagery 2024 MapServer: `https://cityimg.capetown.gov.za/erdas-iws/esri/GeoSpatial%20Datasets/rest/services/Aerial%20Imagery_Aerial%20Imagery%202024/MapServer`
- GeoSampa license notice: `https://prefeitura.sp.gov.br/web/licenciamento/w/licen%C3%A7a-para-uso-de-dados-do-geosampa`
- City of Vancouver Orthophoto imagery 2022 and terms: `https://opendata.vancouver.ca/explore/dataset/orthophoto-imagery-2022/`
- City of Toronto Open Government Licence: `https://open.toronto.ca/open-data-licence/`
- GeoDanmark / Datafordeler Ortofoto terms: `https://www.geodanmark.dk/home/vejledninger/vilkaar-for-data-anvendelse/`
- LINZ Basemaps attribution: `https://www.linz.govt.nz/products-services/data/licensing-and-using-data/attributing-linz-basemaps-data`
