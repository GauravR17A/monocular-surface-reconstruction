# Release engineering and verification

## Snapshot identity

Version 0.1.1 is a research prototype snapshot with technical documentation as of
28 September 2026. Source, documentation and inference artifacts are versioned
together. The Git tag identifies the published source; the artifact manifest
independently identifies weight files and the pinned foundation revision.

This is a fresh publication repository under the scientific name Monocular
Surface Reconstruction. Original development history is not imported. Public
historical records carry a banner explaining name/path normalisation; they are
not represented as untouched original source seals.

## Publication transformations

The package namespace is `msr`. Frontend metadata, UI branding, environment names,
Python imports, commands, notebook-independent scripts and current documentation
use the scientific identity. The development application remains separately
available in its original workspace. Generated outputs, private data, model
caches, local environments and credentials are excluded from Git.

The height bundle embeds the base architecture and all protected tensors so its
loader does not require an old workstation checkpoint path. The classifier
bundle retains its architecture, inference contract and weights. Optimizer/RNG
and unused training state are not required by normal inference. State tensor
equality is checked for 240 height tensors (31,428,489 values) and 210 classifier
tensors (3,716,205 values). This is a packaging check, not new accuracy evidence.

Public GeoTIFF metadata is normalised while raster arrays, validity masks, CRS
and affine transforms are compared exactly. Restricted source imagery is omitted
rather than relicensed. Upstream model licences and data attribution remain.

## Test scope

The original workspace completed 852 Python tests. The portable package completed
826 tests and explicitly skipped 26 cases that require exact historical source
seals or experiment archives not included here. The skip manifest names each
affected test; numerical assertions are not weakened. Skips are visible in test
output and can be attempted with `--run-historical-seals` when their prerequisites
exist. The two counts describe different verification contexts.

The viewer has 155 deterministic tests covering navigation, terrain analysis,
classification state, image import and geometry behavior. TypeScript and a
production build are checked. Dependency versions and the final advisory count
are in the machine-readable verification summary.

The browser harness exercises the actual production request handler and actual
FastAPI TestClient with local model files. It does not open an application
listener. This validates real rendering, request handling and generated products
under an in-process transport; it does not establish network deployment, TLS,
reverse-proxy, authentication or multi-user service behavior.

## Remaining operational boundaries

The Python checks use the established environment with the public package source
on its import path. This release does not claim a fresh GPU-driver installation
on every supported OS. The application is a local research tool with a serialized
inference runtime. Public hosting, concurrency hardening and storage lifecycle
controls remain roadmap work. Large browser bundle notices are retained as a
performance limitation, not suppressed to imply a clean optimisation result.

Run `scripts/verify_publication.py` for source/PDF identity, link, size and manifest
checks. Rebuild documentation from its included source before changing a dated
release. Never edit a historical result merely to make a check pass.

The first clean hosted CPU check exposed an omitted direct Requests dependency:
824 tests passed and two backbone-acquisition import checks failed. Version 0.1.1
declares Requests explicitly. The per-commit GitHub workflow checks the corrected
installation independently of the original workstation environment.
