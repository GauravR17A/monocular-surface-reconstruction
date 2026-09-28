# Release evidence

This directory holds compact, machine-readable release checks and research source
indexes. Checksums bind the named files, not the truth of every research claim.
Dates distinguish earlier research from publication checks on 28 September 2026.

- [Model packaging](model-packaging.json): tensor-preserving model repackaging.
- [Raster publication](raster-publication.json): exact raster values/masks/CRS/transform through metadata cleanup; excludes omitted restricted source assets from redistribution.
- [Browser verification](browser-verification.json): actual UI, API and artifact checks with in-process transport.
- [Verification summary](verification-summary.json): test/build counts, dependency audit and scope.
- [Source inventory](source-inventory.json): dated research records, hashes and sizes.
- [Research metrics](research-metrics.json): curated numerical tables with source-record references.
- [Numerical verification](numerical-verification.json): exact forward outputs and generated raster checks.
- [Documentation build](documentation-build.json): handbook metadata and source hashes.
- [Publication manifest](publication-manifest.json): all public-file hashes except the manifest itself.

Historical scripts sometimes point to complete experiment JSON/CSV/weights in
the original local archive. Those large archives are not implied to be included.
The readable records and reproduction protocols are included. Raw host logs,
private absolute paths, source identities and credentials are not publication
evidence and are excluded.
