"""Dataset manifests and geospatial validation utilities."""

from .manifest import PairAudit, SampleRecord, audit_pair, load_manifest
from .mixed_replay import (
    SourceRatioSampler,
    SourceTaggedDataset,
    assert_matching_sample_contract,
)
from .gamus_dataset import (
    GAMUS_CLASSES,
    GAMUS_OFFICIAL_SPLIT_COUNTS,
    GAMUS_SIX_CLASS_IGNORE_INDEX,
    GAMUS_SIX_CLASS_NAMES,
    GAMUS_SIX_CLASS_SOURCE_IDS,
    GamusSampleRecord,
    GamusSurfaceDataset,
    load_gamus_split,
)
from .surface_dataset import (
    MultiDomainSurfaceDataset,
    SurfaceSampleRecord,
    assert_surface_regions_disjoint,
    load_surface_manifest,
)
from .open_canopy import (
    OpenCanopyFeature,
    forest_semantic_masks,
    load_open_canopy_features,
    open_canopy_urls,
    stored_canopy_to_metres,
)

__all__ = [
    "MultiDomainSurfaceDataset",
    "SourceRatioSampler",
    "SourceTaggedDataset",
    "GAMUS_CLASSES",
    "GAMUS_OFFICIAL_SPLIT_COUNTS",
    "GAMUS_SIX_CLASS_IGNORE_INDEX",
    "GAMUS_SIX_CLASS_NAMES",
    "GAMUS_SIX_CLASS_SOURCE_IDS",
    "GamusSampleRecord",
    "GamusSurfaceDataset",
    "PairAudit",
    "SampleRecord",
    "SurfaceSampleRecord",
    "OpenCanopyFeature",
    "audit_pair",
    "assert_matching_sample_contract",
    "assert_surface_regions_disjoint",
    "load_surface_manifest",
    "load_gamus_split",
    "load_manifest",
    "forest_semantic_masks",
    "load_open_canopy_features",
    "open_canopy_urls",
    "stored_canopy_to_metres",
]
