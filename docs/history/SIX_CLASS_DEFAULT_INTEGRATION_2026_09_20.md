# Six-class default integration — 20 September 2026

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

## Delivered today

User-authorized integration: make six-class identification the normal app experience, replace the three-category Class Explorer, and remove repeated experimental badges. This is an interface/inference integration, not a new training run or a height-model promotion.

- Supported scenes are identified automatically after reconstruction finishes. The neutral start screen does not trigger inference.
- Class Explorer offers **Ground, Buildings, Water, Roads, Low vegetation and Trees**, with coverage statistics and class-map exports. Absent categories remain visible but cannot be selected.
- Inspect and the Fly/Walk scanner use the same six-class map. Both two-second aim and click scanning remain available.
- Category selection highlights the selected class without changing surface heights or flattening other geometry. The photo remains visible by default until the user requests a classification overlay.
- Compact/expanded inspection controls contain all six categories plus All classes.
- Main badges now say Automatic / 6 categories. Known model weaknesses remain visible under **Model details & accuracy limits**; they have not been removed from provenance or release evidence.
- Pending, failed and unsupported identification are explicit. They never silently fall back to a three-group label presented as a six-class prediction. Height inspection remains available independently.
- Retry repeats identification, not reconstruction. Repeated requests for the same source share one request, and up to four successful maps are cached in the browser session. Changing scenes immediately discards the old scene's displayed labels.

## Protected height pipeline

The existing three-group height machinery remains internal. The RGB classifier controls identity/overlay only, not the height model, building geometry, collision masks or exported height values. A class label can disagree with the protected geometry; inspection explains that disagreement rather than changing the height to fit the label.

Water labels do not establish water depth. Relative scores are not metric heights. Ground/road labels do not establish absolute terrain elevation. Areas are only reported when supported geographic scale is available. Unsupported higher-bit-depth inputs retain explicit classifier limitations; the separate height workflow is not relabelled as supported classification.

V3 epoch 1 remains a development classifier with class/city and road-boundary regression failures. Its automatic integration is authorized product behavior, **not accuracy certification**, and no new accuracy improvement is claimed here.

## Verification

| Check | Result |
| --- | --- |
| Viewer Node tests | 107 passed |
| Targeted Python API, prediction, mesh and RGB-preview tests | 37 passed |
| TypeScript check | Passed |
| ESLint on changed TypeScript components/helpers | Passed |
| Automated UI flow | Passed; no page errors |
| Real uploaded RGB: reconstruction → automatic classification | Passed; no page errors |
| Raw/export/presentation artifact preservation | All 12 original job files byte-identical before and after classification |
| Fly two-second aim and click scanner | Manually verified with browser controls |
| Walk two-second aim and pointer-event scanner | Manually verified with browser controls/event handler |
| Expanded six-category selector layout | Visually checked |

Automated UI checks include neutral startup, automatic classification, all six controls, stable inspected height across filtering, scene switching without stale labels, cached revisit, simulated classifier failure, successful retry, unsupported-input handling and the hilly scene's out-of-domain notice. The Python test additionally rejects a real synthetic uint16 GeoTIFF before classifier loading.

The navigation automation initially failed because pointer-capture automation rotated the camera away from its target. The saved `navigation-verification.json` records that failure and is not a passed test. Manual checks subsequently verified the actual scanner behavior: Fly displayed Ground with its independent approximately 0.4 m height; Walk displayed Low vegetation with an approximately 0.1 m height and the geometry-disagreement warning. These example values are UI observations, not accuracy measurements. The navigation harness was adjusted afterward but was not rerun. Native fullscreen/Escape behavior was not newly certified in this turn; the expanded selector layout was checked separately. The final compact-selector CSS adjustment was visually checked after the automated UI suite.

### Saved evidence

Folder: `outputs/runtime/six_class_day1/`

- `ui-verification.json`
- `upload-verification.json`
- `urban-inspection.png`, `classifier-failure.png`, `six-class-upload.png`
- `fly-scanner.png`, `walk-scanner.png`, `fullscreen-six-classes.png`
- Repeatable browser harness: `scripts/verify_six_class_integration.mjs`

The real upload used an existing Duesseldorf diagnostic image, **not an untouched final test**. Job: `outputs/web_jobs/5f1991e16d8641f9b68161a023badae8`. The upload report records SHA-256 values for all 12 unchanged artifacts, including raw height, presentation mesh height and building solids.

Protected hashes checked and unchanged:

| Artifact | SHA-256 |
| --- | --- |
| Protected height checkpoint | `e2d507fbe180988e000a91a873db047407fc15c0fac90446f7d5f05990c0d144` |
| `outputs/runtime/showcase_checkpoint.txt` | `a3d51c686ca2abee9cb7e622c600f5ded9eca3332638ccf56f6f3d8b2f72baa9` |
| Pinned V3 classifier checkpoint | `10fbb39c3d916695627e93d49f2115be18ecd163352b4daa96ba691c0090470c` |

## Try it

1. Open or refresh `http://localhost:3000/`.
2. Upload an RGB image or select a reference scene. The 3D reconstruction appears before automatic identification completes.
3. Use Class Explorer for six-category highlights, or Inspect / Explore for point labels and the separately predicted height.
4. Open Model details for current accuracy limitations.

No training was started, no protected checkpoint was replaced, and no final reserved geography was consumed. Remaining model accuracy work needs its own controlled experiment and evaluation; this integration does not resolve those limitations.
