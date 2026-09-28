# Technical decisions and boundaries

The decisions below consolidate evidence already recorded in source, protocols and
outcome reports. They explain the current package rather than inventing a design
history after the fact.

| ID | Decision | Reason and consequence | Evidence |
| --- | --- | --- | --- |
| D01 | Keep relative, above-ground and absolute products separate. | A monocular image does not supply an absolute datum. Product tags and units must express the actual computation. | Raster/calibration code; scientific contracts. |
| D02 | Preserve a protected height checkpoint. | Later experiments can improve one aggregate while degrading another population. The newest checkpoint is not automatically the best release. | Height scorecard; residual and class-assisted outcomes. |
| D03 | Evaluate final fused height. | Improvements to an internal expert do not matter if the active fusion does not use its output or heavily attenuates its gradient. | Accuracy recovery; fusion implementation. |
| D04 | Use independent image, class and height masks. | Missing or invalid supervision is not zero. Classification and regression can have different valid support. | GAMUS preparation audit; surface datasets. |
| D05 | Make corrected measured support the canonical urban result. | Sparse/estimated building references and unknown background distort all-pixel claims. Preserve older scores under their original protocol labels. | Benchmark protocol; HighBuild instance evaluation. |
| D06 | Keep geography and exposure explicit. | Different IDs or excluded-head cities can still share neighbourhoods or inherited exposure. | Geographic holdout audit and consumed-test history. |
| D07 | Do not tune on a final reserved geography. | A one-shot evaluation loses its status once inspected or used for model selection. | Christchurch protocol; external-height holdout. |
| D08 | Use matched neutral controls for class-assisted height. | Better error than the old model does not prove semantic information caused the gain. | Fast V2, low-surface, sampling and expert-height outcomes. |
| D09 | Keep RGB identification separate from protected height geometry. | The recorded controlled experiments did not justify changing height from six-class guidance. Labels can disagree with geometry and must say so. | Six-class default integration. |
| D10 | Distinguish user-authorized integration from metric acceptance. | V3 identification can be available while its safety failures remain disclosed. Interface availability is not certification. | V3 outcome and September 20 integration. |
| D11 | Preserve the original selector. | Epoch 4 wins pooled OEM F1 but epoch 1 wins the declared equal-region selector. Changing criteria afterward biases the reported result. | RGB multi-region V3 outcome. |
| D12 | Audit inference context separately from learning. | Tile size, precision and padding can change predictions, especially with GroupNorm. A padding gain is not retraining evidence. | Matched app/native and padding audits. |
| D13 | Bound interactive image resolution. | Compressed file size is a poor proxy for decoded memory. Overview reads preserve map bounds while recording lost resolution. | Raster reader; large-raster recovery record. |
| D14 | Separate source measurement from presentation geometry. | Smoothing, exaggeration and solids improve readability but can move the displayed surface substantially. Inspection/export must retain the source contract. | Terrain presentation and source-preservation checks. |
| D15 | Restore flat roofs and remove illustrative tree changes. | Experimental geometry did not meet the desired visual/coverage outcome. Inactive code remains available as research history. | Roof reconstruction status and subsequent product rollback record. |
| D16 | Use a new public package and scientific identity. | A reproducible source release needs portable paths, a coherent entry point and an explicit artifact inventory. Original experiments retain their own immutable identities. | Release manifest and model-packaging checks. |
| D17 | Preserve third-party restrictions and attribution. | Model code availability does not grant unrestricted commercial model rights; imagery also has provider-specific terms. | Upstream model notices and data inventory. |
| D18 | Date the documentation and identify open work. | A research snapshot should distinguish delivered behavior, measured scope and planned investigation. | Status date: 28 September 2026. |

## Alternatives that remain open

A stronger encoder, more epochs, new data, different loss weights, per-instance
supervision and explicit class-guided experts are possible interventions, not
guaranteed improvements. Future work should isolate which problem each change
addresses and preserve the comparator. Broad retraining before resolving label
support, context and error concentration risks repeating the same failure mode.

More detailed roof or tree geometry is also separate from learning reliable
metric heights. Reintroducing a visual layer would require its own labels,
interaction checks and interpretation boundary. It must not silently change the
meaning of exported scientific products.
