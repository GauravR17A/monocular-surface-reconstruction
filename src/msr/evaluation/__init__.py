"""Height and landscape-domain evaluation."""

from .classification_metrics import (
    StreamingClassBoundaryMetrics,
    StreamingMulticlassMetrics,
    StreamingWaterDarkPixelProxy,
    compute_multiclass_metrics,
    multiclass_metric_deltas,
)
from .domain_metrics import DomainEvaluation, evaluate_by_domain, terrain_slope_degrees
from .metrics import HeightMetrics, StreamingRegressionMetrics, compute_height_metrics
from .routed_artifact import (
    GuardedRoutedSurfaceBundle,
    RoutedArtifactDiagnostics,
    RoutedArtifactValidationError,
    load_guarded_routed_surface,
)
from .routed_validation import (
    RoutedValidationError,
    evaluate_dense_surface_model,
    evaluate_validation_guards,
)

__all__ = [
    "DomainEvaluation",
    "HeightMetrics",
    "GuardedRoutedSurfaceBundle",
    "RoutedArtifactDiagnostics",
    "RoutedArtifactValidationError",
    "RoutedValidationError",
    "StreamingMulticlassMetrics",
    "StreamingClassBoundaryMetrics",
    "StreamingWaterDarkPixelProxy",
    "StreamingRegressionMetrics",
    "compute_height_metrics",
    "compute_multiclass_metrics",
    "evaluate_by_domain",
    "evaluate_dense_surface_model",
    "evaluate_validation_guards",
    "load_guarded_routed_surface",
    "multiclass_metric_deltas",
    "terrain_slope_degrees",
]
