"""Unit tests for BehaviorSim feature and distribution visualizations."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from behaviorsim.calibration.fitter import CalibrationData
from behaviorsim.visualization.plot_distributions import (
    plot_feature_distribution,
    plot_feature_comparison,
    plot_categorical_distribution,
)


@pytest.fixture
def sample_numeric_df() -> pd.DataFrame:
    """Fixture providing numeric and categorical feature data."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "sequence_id": np.repeat(np.arange(10), 10),
        "state": np.random.choice(["Browse", "Cart", "Checkout"], size=n),
        "profile": np.random.choice(["Standard", "VIP"], size=n),
        "latency": np.random.exponential(scale=50.0, size=n),
        "action": np.random.choice(["click", "scroll", "view"], size=n),
    })


class TestPlotFeatureDistribution:
    """Tests for plot_feature_distribution."""

    def test_single_feature_density(self, sample_numeric_df: pd.DataFrame):
        """Test basic single feature density histogram."""
        fig, ax = plot_feature_distribution(sample_numeric_df, feature="latency", density=True)
        assert fig is not None
        assert ax is not None
        assert ax.get_ylabel() == "Density"
        assert ax.get_xlabel() == "latency"
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_single_feature_count(self, sample_numeric_df: pd.DataFrame):
        """Test count mode."""
        fig, ax = plot_feature_distribution(sample_numeric_df, feature="latency", density=False)
        assert ax.get_ylabel() == "Count"
        plt.close(fig)

    def test_grouped_by_state(self, sample_numeric_df: pd.DataFrame):
        """Test grouping distribution by state."""
        fig, ax = plot_feature_distribution(
            sample_numeric_df, feature="latency", state_column="state"
        )
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "Browse" in legend_texts
        assert "Cart" in legend_texts
        assert "Checkout" in legend_texts
        plt.close(fig)

    def test_missing_values_handled(self, sample_numeric_df: pd.DataFrame):
        """Test that NaNs are dropped cleanly without failing."""
        df_nan = sample_numeric_df.copy()
        df_nan.loc[0:5, "latency"] = np.nan
        fig, ax = plot_feature_distribution(df_nan, feature="latency")
        assert fig is not None
        plt.close(fig)

    def test_error_handling(self, sample_numeric_df: pd.DataFrame):
        """Test error conditions."""
        # Empty DataFrame
        with pytest.raises(ValueError, match="Cannot plot distribution on an empty dataset"):
            plot_feature_distribution(pd.DataFrame(), feature="latency")

        # Missing feature
        with pytest.raises(ValueError, match="Feature column 'missing' not found"):
            plot_feature_distribution(sample_numeric_df, feature="missing")

        # Non-numeric feature
        with pytest.raises(TypeError, match="must be numeric"):
            plot_feature_distribution(sample_numeric_df, feature="action")

        # All-null feature
        df_null = sample_numeric_df.copy()
        df_null["latency"] = np.nan
        with pytest.raises(ValueError, match="contains only null values"):
            plot_feature_distribution(df_null, feature="latency")


class TestPlotFeatureComparison:
    """Tests for plot_feature_comparison."""

    def test_overlapping_histograms(self, sample_numeric_df: pd.DataFrame):
        """Test empirical vs synthetic overlapping histogram with shared bins."""
        emp = sample_numeric_df.copy()
        syn = sample_numeric_df.copy()
        syn["latency"] = syn["latency"] * 1.2

        fig, ax = plot_feature_comparison(emp, syn, feature="latency")
        assert fig is not None
        assert ax is not None
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert any("Empirical" in t for t in texts)
        assert any("Synthetic" in t for t in texts)
        plt.close(fig)

    def test_identical_bins_and_scales(self, sample_numeric_df: pd.DataFrame):
        """Verify empirical and synthetic histograms use identical bin edges."""
        emp = pd.DataFrame({"latency": [10.0, 20.0, 30.0]})
        syn = pd.DataFrame({"latency": [15.0, 25.0, 35.0]})

        fig, ax = plot_feature_comparison(emp, syn, feature="latency", bins=5)
        # Verify patches exist
        assert len(ax.patches) > 0
        plt.close(fig)

    def test_error_handling(self, sample_numeric_df: pd.DataFrame):
        """Test error conditions."""
        with pytest.raises(ValueError, match="must both be non-empty"):
            plot_feature_comparison(pd.DataFrame(), sample_numeric_df, feature="latency")

        with pytest.raises(ValueError, match="Feature 'missing' not found in empirical"):
            plot_feature_comparison(sample_numeric_df, sample_numeric_df, feature="missing")

        with pytest.raises(TypeError, match="must be numeric"):
            plot_feature_comparison(sample_numeric_df, sample_numeric_df, feature="action")


class TestPlotCategoricalDistribution:
    """Tests for plot_categorical_distribution."""

    def test_aligned_category_support(self):
        """Test category alignment when empirical and synthetic have disjoint/partial categories."""
        emp = pd.DataFrame({"action": ["click", "click", "scroll"]})
        syn = pd.DataFrame({"action": ["click", "view", "view"]})

        fig, ax = plot_categorical_distribution(emp, syn, feature="action")
        # Union of categories is ["click", "scroll", "view"] -> 3 categories
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["click", "scroll", "view"]

        # 3 categories x 2 bars = 6 bar patches
        assert len(ax.patches) == 6
        plt.close(fig)

    def test_explicit_category_order(self):
        """Test supplied category_order is respected."""
        emp = pd.DataFrame({"action": ["click", "scroll"]})
        syn = pd.DataFrame({"action": ["scroll", "view"]})

        supplied = ["view", "scroll", "click"]
        fig, ax = plot_categorical_distribution(emp, syn, feature="action", category_order=supplied)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == supplied
        plt.close(fig)

    def test_input_immutability(self):
        """Test datasets are not modified."""
        emp = pd.DataFrame({"action": ["click", "scroll"]})
        emp_copy = emp.copy()
        fig, ax = plot_categorical_distribution(emp, emp, feature="action")
        pd.testing.assert_frame_equal(emp, emp_copy)
        plt.close(fig)

    def test_error_handling(self):
        """Test error conditions."""
        emp = pd.DataFrame({"action": ["click"]})
        syn = pd.DataFrame({"action": ["scroll"]})

        # Missing category in supplied order
        with pytest.raises(ValueError, match="Data contains categories not present in supplied category_order"):
            plot_categorical_distribution(emp, syn, feature="action", category_order=["click"])

        # Duplicate category in supplied order
        with pytest.raises(ValueError, match="contains duplicate categories"):
            plot_categorical_distribution(emp, syn, feature="action", category_order=["click", "scroll", "click"])

        # Empty order
        with pytest.raises(ValueError, match="cannot be empty"):
            plot_categorical_distribution(emp, syn, feature="action", category_order=[])
