"""Unit tests for BehaviorSim state occupancy and transition visualizations."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from behaviorsim.calibration.fitter import CalibrationData
from behaviorsim.visualization.plot_states import (
    plot_state_occupancy,
    plot_transition_matrix,
)
from behaviorsim.visualization.plot_trajectories import (
    plot_state_trajectory,
    plot_feature_trajectory,
)


@pytest.fixture
def sample_state_df() -> pd.DataFrame:
    """Fixture providing a deterministic sample DataFrame of state transitions."""
    return pd.DataFrame({
        "sequence_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "state": ["Browse", "Cart", "Checkout", "Browse", "Browse", "Cart", "Cart", "Checkout", "Checkout"],
        "profile": ["Standard", "Standard", "Standard", "Premium", "Premium", "Premium", "Standard", "Standard", "Standard"],
    })


@pytest.fixture
def sample_calibration_data(sample_state_df: pd.DataFrame) -> CalibrationData:
    """Fixture providing CalibrationData instance."""
    return CalibrationData(
        data=sample_state_df,
        states=["Browse", "Cart", "Checkout"],
        state_column="state",
        sequence_column="sequence_id",
        profile_column="profile",
    )


class TestPlotStateOccupancy:
    """Tests for plot_state_occupancy."""

    def test_occupancy_proportions(self, sample_state_df: pd.DataFrame):
        """Test default normalized occupancy proportions."""
        fig, ax = plot_state_occupancy(sample_state_df)
        assert fig is not None
        assert ax is not None
        assert ax.get_ylabel() == "Occupancy Proportion"
        assert len(ax.patches) == 3  # Browse, Cart, Checkout
        # Check sum of proportions equals 1.0
        total_height = sum(p.get_height() for p in ax.patches)
        assert np.isclose(total_height, 1.0)
        plt.close(fig)

    def test_occupancy_raw_counts(self, sample_state_df: pd.DataFrame):
        """Test raw counts mode."""
        fig, ax = plot_state_occupancy(sample_state_df, normalize=False)
        assert ax.get_ylabel() == "Observation Count"
        total_count = sum(p.get_height() for p in ax.patches)
        assert total_count == len(sample_state_df)
        plt.close(fig)

    def test_deterministic_and_supplied_state_order(self, sample_state_df: pd.DataFrame):
        """Test explicit state_order is respected and default is sorted."""
        # Default sorted: Browse, Cart, Checkout
        fig1, ax1 = plot_state_occupancy(sample_state_df)
        labels1 = [t.get_text() for t in ax1.get_xticklabels()]
        assert labels1 == ["Browse", "Cart", "Checkout"]
        plt.close(fig1)

        # Supplied order: Checkout, Cart, Browse
        supplied = ["Checkout", "Cart", "Browse"]
        fig2, ax2 = plot_state_occupancy(sample_state_df, state_order=supplied)
        labels2 = [t.get_text() for t in ax2.get_xticklabels()]
        assert labels2 == supplied
        plt.close(fig2)

    def test_occupancy_with_profile_column(self, sample_state_df: pd.DataFrame):
        """Test multi-profile grouping with legend."""
        fig, ax = plot_state_occupancy(sample_state_df, profile_column="profile")
        # 3 states x 2 profiles = 6 bars
        assert len(ax.patches) == 6
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "Premium" in legend_texts
        assert "Standard" in legend_texts
        plt.close(fig)

    def test_occupancy_from_calibration_data(self, sample_calibration_data: CalibrationData):
        """Test plotting directly from CalibrationData."""
        fig, ax = plot_state_occupancy(sample_calibration_data)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == list(sample_calibration_data.states)
        plt.close(fig)

    def test_respects_supplied_ax(self, sample_state_df: pd.DataFrame):
        """Test that supplied Axes is drawn upon without creating a new figure."""
        custom_fig, custom_ax = plt.subplots(figsize=(4, 4))
        fig, ax = plot_state_occupancy(sample_state_df, ax=custom_ax)
        assert fig is custom_fig
        assert ax is custom_ax
        plt.close(custom_fig)

    def test_input_immutability(self, sample_state_df: pd.DataFrame):
        """Test input DataFrame is not modified by plotting."""
        df_copy = sample_state_df.copy(deep=True)
        fig, ax = plot_state_occupancy(sample_state_df)
        pd.testing.assert_frame_equal(sample_state_df, df_copy)
        plt.close(fig)

    def test_error_handling(self, sample_state_df: pd.DataFrame):
        """Test error conditions for invalid inputs."""
        # Empty DataFrame
        with pytest.raises(ValueError, match="Cannot plot state occupancy on an empty dataset"):
            plot_state_occupancy(pd.DataFrame())

        # Missing state column
        with pytest.raises(ValueError, match="State column 'nonexistent' not found"):
            plot_state_occupancy(sample_state_df, state_column="nonexistent")

        # Invalid profile column
        with pytest.raises(ValueError, match="Profile column 'bad_prof' not found"):
            plot_state_occupancy(sample_state_df, profile_column="bad_prof")

        # Empty state_order
        with pytest.raises(ValueError, match="Supplied state_order cannot be empty"):
            plot_state_occupancy(sample_state_df, state_order=[])

        # Duplicate states in state_order
        with pytest.raises(ValueError, match="contains duplicate states"):
            plot_state_occupancy(sample_state_df, state_order=["Browse", "Browse"])

        # Unknown state in data missing from state_order
        with pytest.raises(ValueError, match="Data contains states not present"):
            plot_state_occupancy(sample_state_df, state_order=["Browse", "Cart"])

        # Invalid data type
        with pytest.raises(TypeError, match="Expected pandas DataFrame or CalibrationData"):
            plot_state_occupancy([1, 2, 3])  # type: ignore


class TestPlotTransitionMatrix:
    """Tests for plot_transition_matrix."""

    def test_from_numpy_array(self):
        """Test transition matrix heatmap from numpy array."""
        mat = np.array([[0.7, 0.3], [0.2, 0.8]])
        fig, ax = plot_transition_matrix(mat, state_order=["A", "B"])
        assert fig is not None
        assert ax is not None
        assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B"]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B"]
        plt.close(fig)

    def test_from_calibration_data(self, sample_calibration_data: CalibrationData):
        """Test transition matrix computed directly from CalibrationData."""
        fig, ax = plot_transition_matrix(sample_calibration_data)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == list(sample_calibration_data.states)
        plt.close(fig)

    def test_from_dataframe(self, sample_state_df: pd.DataFrame):
        """Test transition matrix computed from raw DataFrame."""
        fig, ax = plot_transition_matrix(sample_state_df, sequence_column="sequence_id")
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["Browse", "Cart", "Checkout"]
        plt.close(fig)

    def test_annot_toggle(self):
        """Test turning off cell annotations."""
        mat = np.array([[0.5, 0.5], [0.5, 0.5]])
        fig1, ax1 = plot_transition_matrix(mat, annot=True)
        assert len(ax1.texts) == 4  # 4 cells annotated
        plt.close(fig1)

        fig2, ax2 = plot_transition_matrix(mat, annot=False)
        assert len(ax2.texts) == 0
        plt.close(fig2)

    def test_error_handling(self):
        """Test error conditions for transition matrix plotting."""
        # Non-square matrix
        with pytest.raises(ValueError, match="must be square 2D array"):
            plot_transition_matrix(np.ones((2, 3)))

        # Mismatched state_order length
        with pytest.raises(ValueError, match="does not match matrix size"):
            plot_transition_matrix(np.eye(3), state_order=["A", "B"])

        # Invalid type
        with pytest.raises(TypeError, match="Expected np.ndarray, pd.DataFrame, or CalibrationData"):
            plot_transition_matrix("invalid")  # type: ignore


class TestPlotTrajectories:
    """Tests for plot_state_trajectory and plot_feature_trajectory."""

    def test_state_trajectory_basic(self, sample_state_df: pd.DataFrame):
        """Test plotting state trajectory for sequences."""
        fig, ax = plot_state_trajectory(sample_state_df, max_sequences=2)
        assert fig is not None
        assert ax is not None
        assert ax.get_ylabel() == "Discrete State"
        # Y-ticks should match sorted states
        assert [t.get_text() for t in ax.get_yticklabels()] == ["Browse", "Cart", "Checkout"]
        # Legend should show 2 sequences
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2
        plt.close(fig)

    def test_state_trajectory_sequence_selection(self, sample_state_df: pd.DataFrame):
        """Test explicitly specifying sequence_ids."""
        fig, ax = plot_state_trajectory(sample_state_df, sequence_ids=[2])
        assert fig is not None
        plt.close(fig)

    def test_state_trajectory_profile_grouping(self, sample_state_df: pd.DataFrame):
        """Test sequence labeling with profile."""
        fig, ax = plot_state_trajectory(sample_state_df, sequence_ids=[1, 2], profile_column="profile")
        legend = ax.get_legend()
        assert legend is not None
        texts = [t.get_text() for t in legend.get_texts()]
        assert any("Standard" in t for t in texts)
        assert any("Premium" in t for t in texts)
        plt.close(fig)

    def test_state_trajectory_order_column(self, sample_state_df: pd.DataFrame):
        """Test sorting sequences by an explicit order column."""
        df_with_step = sample_state_df.copy()
        df_with_step["step"] = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        fig, ax = plot_state_trajectory(df_with_step, order_column="step")
        assert ax.get_xlabel() == "step"
        plt.close(fig)

    def test_state_trajectory_errors(self, sample_state_df: pd.DataFrame):
        """Test error handling in state trajectory."""
        with pytest.raises(ValueError, match="max_sequences must be a positive integer"):
            plot_state_trajectory(sample_state_df, max_sequences=0)

        with pytest.raises(ValueError, match="Requested sequence_ids not found"):
            plot_state_trajectory(sample_state_df, sequence_ids=[999])

        with pytest.raises(ValueError, match="Sequence column 'missing' not found"):
            plot_state_trajectory(sample_state_df, sequence_column="missing")

    def test_feature_trajectory_basic(self, sample_state_df: pd.DataFrame):
        """Test feature trajectory plotting on numeric features."""
        df = sample_state_df.copy()
        df["amount"] = [10.0, 20.0, 30.0, 15.0, 25.0, 35.0, 12.0, 22.0, 32.0]
        fig, ax = plot_feature_trajectory(df, feature="amount", max_sequences=2)
        assert fig is not None
        assert ax.get_ylabel() == "amount"
        plt.close(fig)

    def test_feature_trajectory_errors(self, sample_state_df: pd.DataFrame):
        """Test feature trajectory error handling."""
        with pytest.raises(ValueError, match="Feature column 'amount' not found"):
            plot_feature_trajectory(sample_state_df, feature="amount")

        # Non-numeric feature
        with pytest.raises(TypeError, match="must be numeric"):
            plot_feature_trajectory(sample_state_df, feature="state")
