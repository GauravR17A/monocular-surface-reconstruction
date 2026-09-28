# GAMUS classifier-head geographic exclusion protocol

> Historical development record. Read alongside the current [release status](../RELEASE.md). Public names and local path labels have been normalized; dated measurements retain their original scope. Referenced training archives are not bundled unless listed in the evidence index.

Status: **sealed for a controlled classifier comparison; not an untouched-system
holdout**.

This protocol reserves the entire approved New York City portion of the GAMUS
training split before a fresh classifier head is fitted on Washington, DC and
Philadelphia. It is useful evidence of transfer by that new head, but it is not
a genuinely unseen-system test: earlier GAMUS experiments trained on NYC, and
the protected shared/height checkpoint inherits HighBuild training that includes
New York examples (`data/highbuild_reviewer/manifests/train.csv`, including row
132). A separate geography that has not influenced any system component remains
required for the final generalisation claim.

The official GAMUS test cannot fill that role. It was consumed once by the older
Stage-3 final audit and is now a read-only historical record, forbidden for V4
training, tuning, threshold selection, model selection, visual case selection,
or repeat evaluation.

## Sealed split

| Role | Cities | Approved tiles | Allowed use |
|---|---|---:|---|
| Learning | Washington, DC + Philadelphia | 3,837 | Model fitting and learning-only statistics |
| Development validation | Washington, DC + Philadelphia | 859 | Epoch choice, tuning, thresholds and error analysis |
| Classifier-excluded check | New York City | 1,164 | One evaluation after the new head and all decisions are frozen |
| Buffer | Not applicable for a whole-city split | 0 | Never train |
| Official GAMUS test | Previously consumed by Stage 3 | Not applicable | Historical audit only; never reuse |

The choice of NYC for this head-only exclusion uses only the predeclared city
prefix in approved GAMUS training tile IDs. No RGB pixel, semantic class, height
label, cached prior, model output, or class distribution from this 1,164-tile
partition was inspected to choose the city for the new comparison. That clean
selection procedure does not erase the system's earlier New York exposure.

## Why the old validation cannot support the claim

The existing train and validation IDs do not overlap, but they share the same
DC and Philadelphia city domains. For the only released two-dimensional tile
index (DC), **350 of 359** validation tiles touch a training tile under a
four-neighbour check, and **all 359** touch one under an eight-neighbour check.
That validation remains useful for development, but it is not genuinely unseen
geography.

NYC is absent from the official validation split. Removing every approved NYC
tile from the fresh classifier's learning set therefore gives a useful
whole-city head-transfer check, which is stronger and easier to defend than
selecting a convenient-looking local block after seeing results. It must be
reported as **excluded from this classifier head's training**, not as unseen by
the complete Monocular Surface Reconstruction system.

## What the local files can and cannot prove

All **5,860** approved train/validation RGB HDF5 containers were inspected for
metadata without reading their image pixels. None has file-level or image-level
attributes; in particular, none embeds a CRS, affine transform, coordinates,
pixel size or acquisition identity.

- DC names contain two released indices, so grid-neighbour checks are possible.
- NYC and Philadelphia names contain scalar release indices. They are treated
  as opaque IDs, not coordinates.
- City prefix is consequently the strongest geographic grouping provable from
  the files currently available.
- This does not prove transfer to a different country, satellite, season,
  ground sampling distance or terrain distribution.

The official GAMUS paper and code describe the multi-city dataset and nominal
0.33 m imagery, but the local containers still lack per-tile geospatial proof:

- <https://arxiv.org/abs/2305.14914>
- <https://github.com/EarthNets/RSI-MMSegmentation>

## Fail-closed future training use

Point the next GAMUS experiment at:

`outputs/data_audits/gamus_geographic_holdout_city_nyc_v1/learning_approved_samples.json`

Its SHA-256 is:

`d56764bc9a3cde31760189b1da40cf207de1b9142c7531b24190ed6e2312e5d9`

The derived index is compatible with the existing loader and strict split-count
validator. It contains only the 3,837 learning IDs plus the unchanged 859-row
development validation inventory. It deliberately contains no official-test
split. A direct loader/validator smoke check passed.

Before training:

1. use the derived approved index and its exact hash;
2. regenerate any class-balancing/sampling index from the 3,837 learning IDs
   only; the existing full-train sampling index is forbidden;
3. calculate normalization, weights and any calibration inputs from learning
   data only;
4. keep NYC out of training, epoch selection, thresholds, post-processing,
   error analysis and visual case selection;
5. freeze checkpoint, thresholds, fusion and guards before the one
   classifier-excluded NYC evaluation;
6. report all six class metrics and height metrics by city/object category;
7. do not promote solely because this classifier-excluded check passes; corrected HighBuild,
   OpenCanopy and the existing regression guards must also pass.

`assert_training_ids_respect_contract(...)` provides a reusable fail-closed
check. It rejects a training inventory containing a holdout, buffer,
development-validation or unknown ID, and can require the exact complete
learning partition.

## Important historical limitation

The current height-focused pilot and the completed six-class candidate were
trained on all 5,001 approved official-training tiles. They have already seen
NYC and **must not** be presented as unseen-NYC results. Even a newly trained V4
classifier head using the sealed learning index remains attached to shared
features and a protected height system with prior New York exposure. Its NYC
result is therefore classifier-head-excluded evidence only. A clean system-level
unseen-geography claim requires a separately reserved external location. The
production app checkpoint and its pointer were not changed by this work.

## Reproducible artifacts

The versioned folder
`outputs/data_audits/gamus_geographic_holdout_city_nyc_v1/` contains:

- `report.json`: concise audit and protocol result;
- `contract.json`: complete, hash-sealed assignments and use policy;
- `assignments.csv`: human-readable role for every development tile;
- `learning_approved_samples.json`: loader-compatible filtered index;
- `learning_sample_ids.txt` and `geographic_holdout_sample_ids.txt`;
- `buffer_sample_ids.txt`: intentionally empty for whole-city isolation;
- `artifact_hashes.json`: exact byte hashes for every artifact.

The planner refuses to run when the approved-index hash changes, a tile ID is
malformed, roles overlap, the holdout city appears in development validation,
the learning set loses city coverage, or an existing sealed artifact differs.

## Prepared head-only comparison

The separate candidate config is
`configs/multidomain_surface_gamus_six_class_head_only_geographic_v1.yaml`.
Its model, optimiser, schedule, loss, augmentation, and DC/Philadelphia
validation-selection recipe are required by code to match the frozen head-only
v2 recipe exactly. The only permitted data changes are the learning-only
approved index and the learning-only water sampler.

`outputs/data_audits/gamus_head_only_geographic_candidate_v1/` now contains:

- a 3,837-row DC/Philadelphia sampling index, SHA-256
  `32a71cc1cb2d9da75ce8774db5e4db342512991ebb37e4b461f776fe7196eedf`;
- a sampling audit showing 790 water-positive crop candidates and proving that
  all 1,164 NYC rows were excluded;
- a loader-compatible 1,164-row locked NYC index, SHA-256
  `98281b990b5ccddfda4349d0b43f0216455a8bebca085bf1232abab10f689417`;
- a passing strict preflight report.

Preparation decoded learning sampling rows only. It did not construct a GAMUS
dataset or open imagery, semantics, heights, priors, or official-test files.

After training, `scripts/evaluate_gamus_locked_nyc.py --freeze-only` seals the
validation-selected checkpoint and predeclared thresholds without opening the
classifier-excluded NYC partition. The explicit evaluation mode then creates a
consumption marker before it constructs that dataset. A second pass is refused.
Every per-class result, road-boundary metric, water/dark-pixel proxy, and height
metric is reported even if the candidate fails its predeclared targets.
Evaluation never promotes or changes the application model, and an NYC pass does
not replace the required new external-geography evaluation.
