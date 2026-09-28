# Security review and data handling

This is a scoped source and test review of a local research application. It is not
an independent penetration test or a certification for internet-facing deployment.
The public repository does not itself operate an inference service.

## Reviewed boundaries

| Boundary | Implemented safeguard | Remaining limit |
| --- | --- | --- |
| File submission | Allowed filename suffixes; nonempty checks; streaming byte counters. | Multipart parsing precedes some application checks; filenames are not proof of safe file content. |
| Decoded image size | Bounded raster processing dimensions and pixel budget. | Native image/GDAL parsers are not sandboxed; byte limits alone do not bound every decoding resource. |
| Calibration options | Mutually exclusive DEM/GCP/automatic terrain, explicit reference kind. | Compatibility depends on real input provenance, datum and content. |
| Classification source | Allowlisted demo IDs, fixed job-ID format, resolved directory checks. | This is path validation, not authenticated ownership. |
| Model execution | Runtime lock serializes GPU work within one process. | Requests can queue; there is no distributed quota, rate limiter or bounded public admission queue. |
| Model acquisition | Release downloads are size/hash checked before replacement. | Only documented trusted model files should be loaded; legacy training loaders can deserialize Python checkpoint payloads. |
| Browser origins | Explicit local CORS origin list, configurable by the operator. | CORS is not authentication and does not prevent all non-browser requests. |
| Outputs | Random job identifiers; static file service rooted at results directory. | Anyone with access to the service and a result URL can read it; there is no per-user authorization. |

## Data lifecycle and privacy

Prediction uploads are saved under a random local job directory together with
derived products and metadata. Successful jobs are retained until the operator
removes or archives them. Failed prediction jobs are removed in the documented
error path. No automatic retention deadline is implemented. Classification
failures can have their own output/cleanup behavior and should be inspected during
operational maintenance.

The API serves generated files from the result root. That directory can include
the uploaded image and supplied reference/control files, not only the final mesh.
Treat the entire directory as potentially sensitive. There is no cloud account,
consent database, multi-tenant access model or encrypted object-storage service.

Automatic terrain uses a configured provider URL template rather than arbitrary
client-provided URLs. It still requires network access and discloses the requested
geographic area to the provider through tile requests. Raw model input is not sent
to an external inference provider by the documented local prediction path.

## Publication checks

The source package excludes environment files, credentials, hosting-account state,
runtime logs, full training datasets and private development history. Its publication
scan checks text, file paths, PDF text/metadata and bundled data metadata against
the release's allowed public identity. Required third-party attribution remains.

The release also records numerical equality of bundled raster values after metadata
normalization, and exact equality of model tensors after inference-only packaging.
Metadata cleanup is not described as byte-identical preservation of those files.

## Before a hosted service

A separate hosted design needs authentication or deliberate public access policy,
request admission/quotas, upload deadlines at the ingress layer, parser isolation,
error-detail handling, storage retention, per-user result authorization and a GPU
work queue. HTTPS and deployment/platform limits would need their own verification.
These are identified missing controls, not features claimed by this local release.

Dependency manifests record the checked versions. They do not establish absence
of vulnerabilities. Do not infer internet-facing security from a passing build or
from source publication alone.
