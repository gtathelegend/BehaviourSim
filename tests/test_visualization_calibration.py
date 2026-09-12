"""Unit tests for BehaviorSim calibration diagnostic visualizations."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from behaviorsim.calibration.fitter import CalibrationData
from behaviorsim.calibration.validator import (
    CategoricalFeatureComparison,
    NumericFeatureComparison,
    StateOccupancyResult,
    StructuralValidationResult,
    TransitionValidationResult,
    ValidationReport,
)
from behaviorsim.visualization.plot_calibration import (
    plot_calibration_summary,
    plot_categorical_comparison,
    plot_numeric_comparison,
    plot_occupancy_comparison,
    plot_transition_comparison,
)


@pytest.fixture
def sample_validation_report() -> ValidationReport:
    """Fixture providing a mock ValidationReport with complete components."""
    structural = StructuralValidationResult(
        is_valid=True,
        transition_matrix_valid=True,
        state_names_valid=True,
        emissions_valid=True,
        errors=(),
    )
    occupancy = StateOccupancyResult(
        empirical_occupancy={"A": 0.6, "B": 0.4},
        synthetic_occupancy={"A": 0.55, "B": 0.45},
        occupancy_difference={"A": -0.05, "B": 0.05},
        total_variation_distance=0.05,
    )
    emp_mat = np.array([[0.8, 0.2], [0.3, 0.7]])
    syn_mat = np.array([[0.78, 0.22], [0.32, 0.68]])
    transition = TransitionValidationResult(
        empirical_matrix=emp_mat,
        synthetic_matrix=syn_mat,
        max_absolute_error=0.02,
        mean_absolute_error=0.02,
        frobenius_distance=0.035,
    )
    numeric_features = {
        "amount": NumericFeatureComparison(
            feature_name="amount",
            empirical_mean=100.0,
            synthetic_mean=98.5,
            empirical_std=15.0,
            synthetic_std=14.8,
            mean_absolute_error=1.5,
            relative_mean_error=0.015,
            wasserstein_distance=1.6,
            ks_statistic=0.04,
            ks_pvalue=0.95,
        ),
        "duration": NumericFeatureComparison(
            feature_name="duration",
            empirical_mean=30.0,
            synthetic_mean=31.2,
            empirical_std=5.0,
            synthetic_std=5.2,
            mean_absolute_error=1.2,
            relative_mean_error=0.04,
            wasserstein_distance=1.3,
            ks_statistic=0.05,
            ks_pvalue=0.90,
        ),
    }
    categorical_features = {
        "channel": CategoricalFeatureComparison(
            feature_name="channel",
            empirical_frequencies={"web": 0.7, "app": 0.3},
            synthetic_frequencies={"web": 0.68, "app": 0.32},
            frequency_difference={"web": -0.02, "app": 0.02},
            total_variation_distance=0.02,
            jensen_shannon_distance=0.015,
        ),
    }
    return ValidationReport(
        structural=structural,
        state_occupancy=occupancy,
        transition=transition,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        threshold_results={},
        is_valid=True,
        summary={"total_comparisons": 4},
    )


class TestPlotCalibrationDiagnostics:
    """Tests for calibration diagnostic visualization functions."""

    def test_occupancy_comparison_from_report(self, sample_validation_report: ValidationReport):
        """Test state occupancy comparison plot directly from ValidationReport."""
        fig, ax = plot_occupancy_comparison(sample_validation_report)
        assert fig is not None
        assert ax is not None
        assert ax.get_ylabel() == "Occupancy Proportion"
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["A", "B"]
        plt.close(fig)

    def test_occupancy_comparison_from_dataframes(self):
        """Test state occupancy comparison from empirical and synthetic DataFrames."""
        emp = pd.DataFrame({"state": ["A", "A", "B"]})
        syn = pd.DataFrame({"state": ["A", "B", "B"]})
        fig, ax = plot_occupancy_comparison(emp, syn)
        assert fig is not None
        plt.close(fig)

    def test_transition_comparison_from_report(self, sample_validation_report: ValidationReport):
        """Test transition matrix comparison (3 heatmaps) from ValidationReport."""
        fig, axes = plot_transition_comparison(sample_validation_report)
        assert fig is not None
        assert len(axes) == 3
        assert "Empirical" in axes[0].get_title()
        assert "Synthetic" in axes[1].get_title()
        assert "Difference" in axes[2].get_title()
        plt.close(fig)

    def test_transition_comparison_from_matrices(self):
        """Test transition comparison from precomputed numpy matrix pair."""
        emp_mat = np.array([[0.9, 0.1], [0.4, 0.6]])
        syn_mat = np.array([[0.85, 0.15], [0.35, 0.65]])
        fig, axes = plot_transition_comparison((emp_mat, syn_mat), states=["Low", "High"])
        assert len(axes) == 3
        assert [t.get_text() for t in axes[0].get_xticklabels()] == ["Low", "High"]
        plt.close(fig)

    def test_numeric_comparison(self, sample_validation_report: ValidationReport):
        """Test numeric feature comparisons bar plot."""
        fig, ax = plot_numeric_comparison(sample_validation_report)
        assert fig is not None
        assert len(ax.patches) == 4  # 2 features x 2 bars (empirical, synthetic)
        plt.close(fig)

    def test_numeric_comparison_subset(self, sample_validation_report: ValidationReport):
        """Test filtering numeric comparison to specific features."""
        fig, ax = plot_numeric_comparison(sample_validation_report, features=["amount"])
        assert len(ax.patches) == 2
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["amount"]
        plt.close(fig)

    def test_categorical_comparison(self, sample_validation_report: ValidationReport):
        """Test categorical comparison plot."""
        fig, ax = plot_categorical_comparison(sample_validation_report, feature="channel")
        assert fig is not None
        assert len(ax.patches) == 4  # 2 categories x 2 bars
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "web" in labels
        assert "app" in labels
        plt.close(fig)

    def test_calibration_summary_no_thresholds(self, sample_validation_report: ValidationReport):
        """Verify summary dashboard with NO thresholds presents metrics neutrally."""
        fig, axes = plot_calibration_summary(sample_validation_report)
        assert fig is not None
        assert axes.shape == (2, 2)
        # Check summary text in axes[1, 1]
        summary_text = axes[1, 1].texts[0].get_text()
        assert "No validation thresholds configured" in summary_text
        assert "State Occupancy TVD:  0.0500" in summary_text
        plt.close(fig)

    def test_calibration_summary_with_explicit_thresholds(self, sample_validation_report: ValidationReport):
        """Verify summary dashboard displays explicit PASS/FAIL when thresholds were supplied."""
        # Add threshold results to mock report
        report_with_thresh = ValidationReport(
            structural=sample_validation_report.structural,
            state_occupancy=sample_validation_report.state_occupancy,
            transition=sample_validation_report.transition,
            numeric_features=sample_validation_report.numeric_features,
            categorical_features=sample_validation_report.categorical_features,
            threshold_results={
                "state_tvd <= 0.1": True,
                "transition_mae <= 0.01": False,
            },
            is_valid=False,
            summary=sample_validation_report.summary,
        )
        fig, axes = plot_calibration_summary(report_with_thresh)
        summary_text = axes[1, 1].texts[0].get_text()
        assert "[PASS] state_tvd <= 0.1" in summary_text
        assert "[FAIL] transition_mae <= 0.01" in summary_text
        assert "Overall Calibration Status: FAILED" in summary_text
        plt.close(fig)

    def test_error_handling(self):
        """Test error conditions on invalid inputs."""
        with pytest.raises(TypeError, match="Expected ValidationReport"):
            plot_numeric_comparison("not_a_report")  # type: ignore

        with pytest.raises(TypeError, match="Expected ValidationReport"):
            plot_categorical_comparison("not_a_report")  # type: ignore

        with pytest.raises(TypeError, match="Expected ValidationReport"):
            plot_calibration_summary("not_a_report")  # type: ignore
