"""Comprehensive tests for BehaviorSim calibration validator."""

from typing import Dict
import numpy as np
import pandas as pd
import pytest

from behaviorsim.calibration.fitter import (
    CalibrationData,
    fit_profile,
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
from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.simulator import Simulator
from behaviorsim.core.state import State


@pytest.fixture
def sample_empirical_calib() -> CalibrationData:
    """Fixture providing a standard calibrated dataset."""
    df = pd.DataFrame(
        {
            "sequence_id": [1, 1, 1, 2, 2, 2],
            "state": ["idle", "active", "idle", "active", "active", "idle"],
            "response_time": [1.2, 0.4, 1.1, 0.5, 0.3, 1.0],
            "action": ["click", "submit", "click", "submit", "submit", "click"],
        }
    )
    return CalibrationData(
        data=df,
        states=["idle", "active"],
        numeric_features=["response_time"],
        categorical_features=["action"],
    )


def test_validation_identical_datasets(sample_empirical_calib: CalibrationData) -> None:
    """Verify validation of identical empirical and synthetic data yields zero divergence."""
    report = validate_calibration(
        empirical_data=sample_empirical_calib,
        synthetic_data=sample_empirical_calib,
    )

    assert isinstance(report, ValidationReport)
    assert report.structural.is_valid
    assert report.is_valid

    # State TVD must be exactly 0.0
    assert report.state_occupancy.total_variation_distance == 0.0
    assert all(np.isclose(v, 0.0) for v in report.state_occupancy.occupancy_difference.values())

    # Transition error must be 0.0
    assert report.transition is not None
    assert report.transition.max_absolute_error == 0.0
    assert report.transition.mean_absolute_error == 0.0
    assert report.transition.frobenius_distance == 0.0

    # Numeric feature comparison
    num_comp = report.numeric_features["response_time"]
    assert np.isclose(num_comp.empirical_mean, num_comp.synthetic_mean)
    assert np.isclose(num_comp.mean_absolute_error, 0.0)
    assert np.isclose(num_comp.wasserstein_distance, 0.0)
    assert np.isclose(num_comp.ks_statistic, 0.0)
    assert np.isclose(num_comp.ks_pvalue, 1.0)

    # Categorical feature comparison
    cat_comp = report.categorical_features["action"]
    assert np.isclose(cat_comp.total_variation_distance, 0.0)
    assert np.isclose(cat_comp.jensen_shannon_distance, 0.0)


def test_validation_state_occupancy_mismatch(sample_empirical_calib: CalibrationData) -> None:
    """Verify validation detects state occupancy divergence accurately."""
    # Synthetic data has 100% active state
    syn_df = pd.DataFrame(
        {
            "sequence_id": [1, 1, 1],
            "state": ["active", "active", "active"],
            "response_time": [0.4, 0.5, 0.3],
            "action": ["submit", "submit", "submit"],
        }
    )

    report = validate_calibration(
        empirical_data=sample_empirical_calib,
        synthetic_data=syn_df,
    )

    # Empirical: 3 idle (0.5), 3 active (0.5)
    # Synthetic: 0 idle (0.0), 3 active (1.0)
    # TVD = 0.5 * (|0.0 - 0.5| + |1.0 - 0.5|) = 0.5
    assert np.isclose(report.state_occupancy.total_variation_distance, 0.5)
    assert np.isclose(report.state_occupancy.occupancy_difference["active"], 0.5)
    assert np.isclose(report.state_occupancy.occupancy_difference["idle"], -0.5)


def test_validation_transition_mismatch(sample_empirical_calib: CalibrationData) -> None:
    """Verify transition dynamics comparison detects transition differences."""
    syn_df = pd.DataFrame(
        {
            "sequence_id": [1, 1, 1],
            "state": ["idle", "idle", "idle"],  # pure self-loops
            "response_time": [1.0, 1.1, 1.2],
            "action": ["click", "click", "click"],
        }
    )

    report = validate_calibration(
        empirical_data=sample_empirical_calib,
        synthetic_data=syn_df,
    )

    assert report.transition is not None
    assert report.transition.max_absolute_error > 0.0
    assert report.transition.mean_absolute_error > 0.0


def test_validation_numeric_mean_mismatch(sample_empirical_calib: CalibrationData) -> None:
    """Verify numeric feature evaluation calculates mean and distribution discrepancy."""
    syn_df = pd.DataFrame(
        {
            "sequence_id": [1, 1],
            "state": ["idle", "active"],
            "response_time": [10.0, 20.0],  # much higher response time
            "action": ["click", "submit"],
        }
    )

    report = validate_calibration(
        empirical_data=sample_empirical_calib,
        synthetic_data=syn_df,
    )

    num_comp = report.numeric_features["response_time"]
    assert num_comp.synthetic_mean == 15.0
    assert num_comp.mean_absolute_error > 10.0
    assert num_comp.wasserstein_distance > 10.0


def test_validation_categorical_divergence(sample_empirical_calib: CalibrationData) -> None:
    """Verify categorical divergence calculates TVD and JS distance."""
    syn_df = pd.DataFrame(
        {
            "sequence_id": [1, 1],
            "state": ["idle", "active"],
            "response_time": [1.0, 0.5],
            "action": ["unknown_action", "unknown_action"],
        }
    )

    report = validate_calibration(
        empirical_data=sample_empirical_calib,
        synthetic_data=syn_df,
    )

    cat_comp = report.categorical_features["action"]
    assert cat_comp.total_variation_distance == 1.0  # completely disjoint
    assert cat_comp.jensen_shannon_distance > 0.0


def test_validation_structural_profile_checks() -> None:
    """Verify structural checks on invalid Profile instances."""
    # Profile with mismatched transition matrix dimensions (3x3 for 2 states)
    invalid_prof = Profile(
        name="bad_prof",
        state_emissions={
            "idle": {"rt": FeatureDistribution("normal", {"loc": 1.0, "scale": 0.2})},
            "active": {"rt": FeatureDistribution("normal", {"loc": 0.5, "scale": 0.1})},
        },
        transition_matrix=np.eye(3),
    )

    emp_df = pd.DataFrame({"state": ["idle", "active"], "sequence_id": [1, 1]})
    syn_df = pd.DataFrame({"state": ["idle", "active"], "sequence_id": [1, 1]})

    report = validate_calibration(
        empirical_data=emp_df,
        synthetic_data=syn_df,
        profile=invalid_prof,
        states=["idle", "active"],
    )

    assert not report.structural.is_valid
    assert not report.structural.transition_matrix_valid
    assert not report.is_valid
    assert any("Transition matrix shape" in err for err in report.structural.errors)


def test_validation_thresholds_passing_and_failing(sample_empirical_calib: CalibrationData) -> None:
    """Verify user-configured validation thresholds enforce pass/fail evaluation."""
    # Synthetic data with moderate difference
    syn_df = pd.DataFrame(
        {
            "sequence_id": [1, 1, 1, 2, 2, 2],
            "state": ["idle", "active", "active", "active", "active", "active"],  # 1/6 idle, 5/6 active
            "response_time": [1.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            "action": ["click", "submit", "submit", "submit", "submit", "submit"],
        }
    )

    # Permissive threshold passes
    report_pass = validate_calibration(
        sample_empirical_calib,
        syn_df,
        thresholds={"max_state_tvd": 0.5},
    )
    assert report_pass.threshold_results["max_state_tvd"] is True
    assert report_pass.is_valid is True

    # Strict threshold fails
    report_fail = validate_calibration(
        sample_empirical_calib,
        syn_df,
        thresholds={"max_state_tvd": 0.1},
    )
    assert report_fail.threshold_results["max_state_tvd"] is False
    assert report_fail.is_valid is False


def test_validation_empty_inputs_rejected() -> None:
    """Verify empty datasets raise ValueError."""
    emp_df = pd.DataFrame({"state": ["idle"]})
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Synthetic dataset cannot be empty"):
        validate_calibration(emp_df, empty_df)

    with pytest.raises(ValueError, match="Empirical dataset cannot be empty"):
        validate_calibration(empty_df, emp_df)


def test_validation_serialization_to_dict(sample_empirical_calib: CalibrationData) -> None:
    """Verify ValidationReport serializes cleanly to a dictionary."""
    report = validate_calibration(
        empirical_data=sample_empirical_calib,
        synthetic_data=sample_empirical_calib,
    )
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "is_valid" in d
    assert "structural" in d
    assert "state_occupancy" in d
    assert "numeric_features" in d
    assert "categorical_features" in d
    assert "summary" in d
