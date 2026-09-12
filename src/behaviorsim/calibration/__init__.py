"""Calibration module for empirical parameter fitting, validation, and clustering in BehaviorSim."""

from behaviorsim.calibration.clustering import (
    ProfileClusteringResult,
    cluster_profiles,
)
from behaviorsim.calibration.fitter import (
    CalibrationData,
    extract_state_proxy,
    fit_distribution,
    fit_profile,
    fit_transition_matrix,
    validate_calibration_data,
)
from behaviorsim.calibration.validator import (
    CategoricalFeatureComparison,
    NumericFeatureComparison,
    StateOccupancyResult,
    StructuralValidationResult,
    TransitionValidationResult,
    ValidationReport,
    validate_calibration,
)

__all__ = [
    "CalibrationData",
    "CategoricalFeatureComparison",
    "NumericFeatureComparison",
    "ProfileClusteringResult",
    "StateOccupancyResult",
    "StructuralValidationResult",
    "TransitionValidationResult",
    "ValidationReport",
    "cluster_profiles",
    "extract_state_proxy",
    "fit_distribution",
    "fit_profile",
    "fit_transition_matrix",
    "validate_calibration",
    "validate_calibration_data",
]
