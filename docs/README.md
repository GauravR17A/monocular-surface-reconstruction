# Technical documentation

**Monocular Surface Reconstruction — Status as of 28 September 2026**

[Read the full technical documentation PDF](Monocular_Surface_Reconstruction_Technical_Documentation.pdf)

This is a dated research snapshot. The opening chapters describe the current
package; the appendices preserve earlier experiments and decisions at their
recorded dates. A historical plan does not establish a currently enabled feature.

| Chapter | Contents |
| --- | --- |
| [Overview and scope](OVERVIEW.md) | Purpose, users, contributions, completed and open work |
| [Team Seifuku](TEAM.md) | Team logo, members, confirmed responsibilities and contact |
| [User guide](USER_GUIDE.md) | Sample workflow, uploads, inspection, calibration and exports |
| [Architecture](ARCHITECTURE.md) | Components, data flow, runtime and model boundaries |
| [Scientific contracts](SCIENTIFIC_CONTRACTS.md) | Units, validity, source/presentation separation and metric scope |
| [Methods](METHODS.md) | Height network, relative prior, fusion, tiling and calibration |
| [Implementation map](CODE_MAP.md) | Code ownership, extension points and maintenance |
| [Validation protocol](VALIDATION.md) | Metrics, reference isolation and reproducibility |
| [Glossary](GLOSSARY.md) | Units, surfaces, scores and product terminology |
| [Data and provenance](DATA.md) | Datasets, splits, preparation, uncertainty and holdout ledger |
| [Results](RESULTS.md) | Comparable baselines, classifier transfer and failure analysis |
| [Experiment register](EXPERIMENTS.md) | What was tried, why, outcomes and decisions |
| [Decision register](DECISIONS.md) | Alternatives, consequences and supporting records |
| [Feature register](FEATURES.md) | Implementation locations, verification and limitations |
| [Model cards](MODEL_CARDS.md) | Model identities, intended use, restrictions and failure modes |
| [API and file contracts](API.md) | Requests, responses, output products and errors |
| [Setup and reproduction](SETUP.md) | Install, acquire artifacts, run, test and reproduce research |
| [Performance](PERFORMANCE.md) | Measured package behavior and measurement scope |
| [Security and data handling](SECURITY_REVIEW.md) | Verified safeguards, retention and deployment limits |
| [Release verification](RELEASE.md) | Actual package checks and outstanding evidence |
| [Roadmap](ROADMAP.md) | Ongoing research questions and acceptance criteria |
| [References](REFERENCES.md) | Papers, dataset providers, software and attribution |
| [Source inventory](SOURCE_INVENTORY.md) | Current sources and the complete historical appendix |

The same source chapters build the PDF. Its linked contents, bookmarks, page
numbers and source labels support both a first read and detailed review. Machine
evidence is indexed in [evidence/README.md](evidence/README.md). Documentation is
not a claim of independent certification or a guarantee of future results.
