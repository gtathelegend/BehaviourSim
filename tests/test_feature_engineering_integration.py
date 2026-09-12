"""Integration tests for BehaviorSim feature engineering subsystem.

Tests end-to-end composability, compatibility with CalibrationData,
and operation on Simulator-generated synthetic traces.
"""

import numpy as np
import pandas as pd
import pytest

from behaviorsim.calibration.fitter import CalibrationData
from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.simulator import Simulator
from behaviorsim.core.state import State
from behaviorsim.feature_engineering import (
    add_rolling_features,
    add_session_features,
    add_transition_features,
    build_behavioral_features,
)


@pytest.fixture
def synthetic_simulation_trace() -> pd.DataFrame:
    """Generate a realistic synthetic multi-sequence trace using core Simulator."""
    states = [State("idle"), State("active")]
    profile = Profile(
        name="test_agent",
        state_emissions={
            "idle": {
                "latency": FeatureDistribution("normal", {"loc": 1.5, "scale": 0.2}),
                "clicks": FeatureDistribution("poisson", {"lam": 1.0}),
            },
            "active": {
                "latency": FeatureDistribution("normal", {"loc": 0.5, "scale": 0.1}),
                "clicks": FeatureDistribution("poisson", {"lam": 5.0}),
            },
        },
        transition_matrix=np.array([[0.7, 0.3], [0.4, 0.6]]),
    )
    sim = Simulator(states=states, profile=profile)
    # Generate 5 sequences of 10 interactions each
    df = sim.generate(num_interactions=10, num_sequences=5, seed=42)
    # Add synthetic cumulative time column
    df["time"] = df["interaction_id"].astype(float) * 2.0
    return df


def test_build_behavioral_features_end_to_end(synthetic_simulation_trace: pd.DataFrame) -> None:
    """Verify build_behavioral_features on Simulator generated traces."""
    enriched = build_behavioral_features(
        synthetic_simulation_trace,
        state_column="state",
        time_column="time",
        numeric_columns=["latency", "clicks"],
        rolling_windows=[3, 5],
        rolling_stats=["mean", "var"],
        session_gap_threshold=5.0,
        sequence_column="sequence_id",
    )

    assert isinstance(enriched, pd.DataFrame)
    assert len(enriched) == len(synthetic_simulation_trace)

    # Verify rolling feature columns exist
    assert "latency_rolling_mean_w3" in enriched.columns
    assert "latency_rolling_var_w3" in enriched.columns
    assert "clicks_rolling_mean_w5" in enriched.columns

    # Verify session columns exist
    assert "session_id" in enriched.columns
    assert "session_duration" in enriched.columns

    # Verify transition columns exist
    assert "previous_state" in enriched.columns
    assert "is_transition" in enriched.columns
    assert "state_persistence" in enriched.columns


def test_compatibility_with_calibration_data(synthetic_simulation_trace: pd.DataFrame) -> None:
    """Verify feature engineering functions accept CalibrationData containers directly."""
    calib = CalibrationData(
        data=synthetic_simulation_trace,
        states=["idle", "active"],
        sequence_column="sequence_id",
        numeric_features=["latency", "clicks"],
    )

    enriched = build_behavioral_features(
        calib,
        state_column="state",
        time_column="time",
        numeric_columns=["latency"],
        rolling_windows=[3],
        rolling_stats=["mean"],
        session_gap_threshold=5.0,
    )

    assert isinstance(enriched, pd.DataFrame)
    assert "latency_rolling_mean_w3" in enriched.columns
    assert "is_transition" in enriched.columns


def test_composable_chaining(synthetic_simulation_trace: pd.DataFrame) -> None:
    """Verify composable step-by-step pipeline chaining."""
    df1 = add_rolling_features(
        synthetic_simulation_trace,
        value_columns=["latency"],
        windows=[3],
        stats=["mean"],
        sequence_column="sequence_id",
    )
    df2 = add_session_features(
        df1,
        time_column="time",
        session_gap_threshold=5.0,
        sequence_column="sequence_id",
    )
    df3 = add_transition_features(
        df2,
        state_column="state",
        time_column="time",
        sequence_column="sequence_id",
    )

    assert "latency_rolling_mean_w3" in df3.columns
    assert "session_duration" in df3.columns
    assert "state_persistence" in df3.columns
    # Ensure original trace is unmutated
    assert "latency_rolling_mean_w3" not in synthetic_simulation_trace.columns


def test_empty_and_singleton_sequences() -> None:
    """Verify robust behavior on empty DataFrames and single-row sequences."""
    empty_df = pd.DataFrame(columns=["sequence_id", "state", "val", "time"])
    empty_res = build_behavioral_features(
        empty_df,
        state_column="state",
        time_column="time",
        numeric_columns=["val"],
        sequence_column="sequence_id",
    )
    assert empty_res.empty

    singleton_df = pd.DataFrame(
        {"sequence_id": [1], "state": ["idle"], "val": [42.0], "time": [0.0]}
    )
    singleton_res = build_behavioral_features(
        singleton_df,
        state_column="state",
        time_column="time",
        numeric_columns=["val"],
        sequence_column="sequence_id",
    )
    assert len(singleton_res) == 1
    assert np.isnan(singleton_res["val_rolling_mean_w3"].iloc[0])
    assert singleton_res["state_persistence"].iloc[0] == 1
    assert singleton_res["is_transition"].iloc[0] == 0
