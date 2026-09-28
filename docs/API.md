# API and output contracts

The supported default is a local FastAPI service on `127.0.0.1:8000`. The viewer
uses this endpoint unless configured otherwise. The service exposes OpenAPI at
`/openapi.json` and interactive API documentation at `/docs` when running.

## Endpoints

| Method / path | Input | Result |
| --- | --- | --- |
| `GET /api/health` | None | Status and configured device. A ready response does not prove model files have been loaded. |
| `POST /api/predict` | Multipart image and optional calibration/reference fields | Job ID, product URLs and metadata after processing. |
| `POST /api/classify` | Exactly one completed `job_id` or allowed `demo_id` | Six-class map, coverage, raster/metadata URLs and model information. |
| `GET /results/{job}/{file}` | Generated job path | Saved image, GeoTIFF, JSON or other output file. |

## Prediction request

| Field | Type | Contract |
| --- | --- | --- |
| `image` | Required file | JPG/JPEG/PNG/TIF/TIFF/JP2/WebP/BMP; supported decoding and nonempty content required. |
| `dem` | Optional file | Terrain GeoTIFF; mutually exclusive with GCP and automatic terrain. |
| `gcps` | Optional file | CSV with `elevation_m` and map `x,y` or pixel `row,col` coordinates. |
| `reference` | Optional file | Independent reference GeoTIFF for supported comparison. |
| `reference_kind` | `ndsm` or `dsm` | Must match the reference quantity. |
| `auto_dem` | Boolean | Request automatic terrain; incompatible with manual DEM/GCP. |
| `auto_dem_if_georeferenced` | Boolean | Attempt automatic terrain for a georeferenced input under the declared route. |

Each file is limited to 512 MiB in the application streaming loop. The image
processing grid is limited to 3072 by 3072 dimensions/pixel budget. These are
application limits, not a front-proxy upload quota or a promise about every decoder.

## Response and files

The response includes `job_id`, `uploaded_bytes`, `texture_url`, `height_url`,
`raw_height_url`, `rdsm_url`, `relative_depth_url`, `structures_url`, optional
probability/semantic/calibration/reference URLs, and both embedded metadata and
`metadata_url`. Optional products are null when unavailable; their absence must
not be silently presented as zero.

| Product | Interpretation |
| --- | --- |
| `height_above_ground_m.tif` | Learned above-ground estimate, with its model and grid contract. |
| `rdsm_relative.tif` | Dimensionless relative surface score. |
| `relative_depth` product | Target-independent foundation prior. |
| `presentation_mesh_height_m.tif` | Display-oriented surface; not used as a new accuracy claim. |
| `dsm_absolute_m.tif` | Aligned terrain plus nonnegative object-height estimate. |
| `gcp_calibrated_surface_m.tif` | Surface mapped with supplied control elevations. |
| `validation_signed_error_m.tif` | Prediction minus eligible aligned reference. |
| Probability/class products | Explicitly named internal or six-class outputs; these are different branches. |
| `metadata.json` | Input and processing grids, product type, units, calibration, validation and warnings. |

Use response URLs rather than assuming every optional filename exists. Relative
URLs are resolved against the API origin, not the frontend origin.

## Classification request

Uploaded-scene identifiers must match 32 lowercase hexadecimal characters and
refer to a completed job with an available original image. Demo identifiers are
allowlisted as `urban`, `sparse`, `hilly` and `forest`. Clients do not supply an
arbitrary filesystem path or source URL.

The classifier returns `labels_url` for row-major bytes, `raster_url` for class
GeoTIFF and `metadata_url`, plus dimensions, coverage and scope. Class IDs are
0 ground, 1 buildings, 2 water, 3 roads, 4 low vegetation, 5 trees; 255 is invalid.
Area is reported only when the grid has a supported projected CRS and unit scale.

## Errors and operational behavior

Unsupported image/reference suffixes return 415; invalid option combinations,
empty inputs or reference kinds return 400; oversized files return 413; invalid
processing/classification inputs use 422; unexpected runtime failures use 500.
The client distinguishes unavailable API, failed prediction, unsupported
identification and retryable classification failure.

The model runtime is lazy and serializes GPU execution in one process. There is
no external work queue, cancellation guarantee or multi-user authentication.
Successful jobs persist locally; failed prediction jobs are removed by the error
path. Classification failure cleanup and parser-level resource control have
separate limitations described in the security review.

Changing the public package's metadata prefix to `MSR_` is a publication naming
change. Source numerical arrays and their scientific interpretations must be
preserved and checked independently from metadata byte identity.
