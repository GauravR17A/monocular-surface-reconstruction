# Performance and application verification

## Checked environment

Release checks ran on Windows with Python 3.10.11, an NVIDIA
RTX 4070 Laptop GPU (about 8 GiB VRAM), PyTorch 2.7.0 with CUDA 12.8, and Node.js
22. The frontend dependency versions are pinned by `viewer/package-lock.json`.
They were refreshed for publication; model weights and scientific selectors were
not changed by that dependency update.

## Actual workflow check

The browser loaded the production SSR handler and hydrated the real React/Three.js
application. API calls were routed through the real FastAPI TestClient and local
release weights. This opened no application listening socket. Urban, hilly and
forest saved scenes rendered; a Copenhagen RGB JPEG was then uploaded and produced
new raw height, relative depth, probability, presentation and metadata products.
Automatic classification ran on the actual image. Result URLs returned data. A
missing-upload request returned 422. The 390-pixel viewport had no horizontal
overflow. Browser console/page errors: zero.

![Packaged urban workspace](assets/workspace.png)

![Georeferenced hilly workspace](assets/hilly-workspace.png)

The bridge measured the following sequential request durations during that check:

| Request | In-process elapsed | HTTP result |
| --- | ---: | ---: |
| /api/classify | 3.076 s | 200 |
| /api/classify | 0.201 s | 200 |
| /api/classify | 0.336 s | 200 |
| /api/predict | 3.896 s | 200 |
| /api/classify | 0.401 s | 200 |

These are individual local observations, not percentiles, throughput, a production
latency promise or an internet-network benchmark. The first classifier request
includes lazy loading; the later requests use different scene sizes/content and
must not be treated as a controlled speed comparison. The prediction includes
the first height/foundation load of this process. Browser render time, driver
startup, transport and other hardware can change the user-visible duration.

## Numerical integrity and exports

The release bundles preserve 240 height-model tensors and 210 classifier tensors
exactly. A seeded CPU float32 comparison on a synthetic 128 x 160 input produced
20 exactly equal model outputs between original and packaged height loaders.
Eight newly generated GeoTIFF exports were reopened and had finite values on their
valid support. These checks support packaging integrity, not geographic accuracy.

## Test and build outcome

- Python public package: 826 passed, 26 explicit historical-archive skips, zero failures.
- Original workspace: 852 passed, zero failures.
- Viewer deterministic tests: 155 passed, zero failures.
- TypeScript and production build: passed.
- Browser: 10 completed checks, zero console/page errors.

The production build emits a large-bundle notice and vinext cannot statically
classify every route. Mesh downsampling bounds interactive geometry; full raster
resolution still consumes browser memory and upload processing time. Performance
work should measure decode, inference, transfer, mesh construction and frame time
separately rather than attributing every delay to the model.

## Advisory review and scope

The reviewed npm dependency tree reports 0 critical,
0 high, 4 moderate and
1 low advisories after updates. The remaining
moderate chain is vinext's OG-rendering dependency path through Satori/fflate; the
low advisory concerns esbuild's Windows development server. This viewer does not
implement OG image generation and is scoped to local research use. The counts are
the scanner's dated observations, not a general security certification. See the
machine-readable summary for package names and advisory links.

## Reproduce the browser scope

Build the viewer, install its Playwright browser, and point `MSR_PYTHON` to the
environment containing this package and models. From `viewer/`, run
`node scripts/verify-browser.mjs`. Set `MSR_TEST_DEVICE=cpu` for CPU or supply
`MSR_BROWSER_CDP` to attach to an existing controlled Chromium instance. The script
records evidence and screenshots under `docs/`; review them before publishing an
updated report. Model weights and the pinned foundation are required. No sealed
research holdout is used. This does not replace a future network deployment test.
