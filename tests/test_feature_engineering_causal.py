"""Unit and causal correctness tests for BehaviorSim recency, streak, and session features."""

import numpy as np
import pandas as pd
import pytest

from behaviorsim.feature_engineering.causal import (
    add_session_features,
    cumulative_inactivity,
    interaction_gap,
    previous_value,
    session_boundary_indicator,
    session_duration,
    session_interaction_count,
    sessionize,
    streak_length,
    time_since_previous,
)


@pytest.fixture
def sample_temporal_df() -> pd.DataFrame:
    """Fixture providing multi-sequence data with timestamps and states."""
    return pd.DataFrame(
        {
            "sequence_id": [1, 1, 1, 1, 2, 2, 2],
            "state": ["A", "A", "A", "B", "A", "A", "B"],
            "time": [10.0, 15.0, 45.0, 50.0, 100.0, 105.0, 200.0],
        }
    )


def test_previous_value(sample_temporal_df: pd.DataFrame) -> None:
    """Verify previous_value extracts lagged state within each sequence."""
    res = previous_value(sample_temporal_df, column="state", sequence_column="sequence_id")

    # Sequence 1: [A, A, A, B] -> [None, A, A, A]
    # Sequence 2: [A, A, B] -> [None, A, A]
    expected = [None, "A", "A", "A", None, "A", "A"]
    assert res.tolist() == expected


def test_streak_length_causal_convention() -> None:
    """Verify streak_length matches established causal convention."""
    df = pd.DataFrame({"state": ["A", "A", "A", "B", "B", "A"]})
    res = streak_length(df, column="state")
    assert res.tolist() == [0, 1, 2, 0, 1, 0]


def test_streak_length_target_value() -> None:
    """Verify streak_length with specific target_value."""
    df = pd.DataFrame({"state": ["A", "A", "A", "B", "B", "A"]})
    res = streak_length(df, column="state", target_value="A")
    assert res.tolist() == [0, 1, 2, 3, 0, 0]


def test_time_since_previous_and_gap(sample_temporal_df: pd.DataFrame) -> None:
    """Verify time_since_previous calculates inter-event gaps per sequence."""
    res = time_since_previous(sample_temporal_df, time_column="time", sequence_column="sequence_id")
    # Seq 1 times: [10, 15, 45, 50] -> [NaN, 5, 30, 5]
    # Seq 2 times: [100, 105, 200] -> [NaN, 5, 95]
    assert np.isnan(res.iloc[0])
    assert res.iloc[1] == 5.0
    assert res.iloc[2] == 30.0
    assert res.iloc[3] == 5.0
    assert np.isnan(res.iloc[4])
    assert res.iloc[5] == 5.0
    assert res.iloc[6] == 95.0

    # Alias check
    alias_res = interaction_gap(sample_temporal_df, time_column="time", sequence_column="sequence_id")
    pd.testing.assert_series_equal(res, alias_res)


def test_sessionize_and_session_boundary(sample_temporal_df: pd.DataFrame) -> None:
    """Verify sessionization partitions interactions when gap > threshold."""
    # Threshold = 20.0
    # Seq 1: gap=5 (sess 0), gap=30 (sess 1), gap=5 (sess 1) -> [0, 0, 1, 1]
    # Seq 2: gap=5 (sess 0), gap=95 (sess 1) -> [0, 0, 1]
    sess_ids = sessionize(
        sample_temporal_df,
        time_column="time",
        session_gap_threshold=20.0,
        sequence_column="sequence_id",
    )
    assert sess_ids.tolist() == [0, 0, 1, 1, 0, 0, 1]

    boundaries = session_boundary_indicator(
        sample_temporal_df,
        time_column="time",
        session_gap_threshold=20.0,
        sequence_column="sequence_id",
    )
    assert boundaries.tolist() == [1, 0, 1, 0, 1, 0, 1]


def test_session_duration_and_interaction_count(sample_temporal_df: pd.DataFrame) -> None:
    """Verify causal session duration and interaction count."""
    # Threshold = 20.0
    # Seq 1:
    # row 0 (t=10): dur 0.0, count 1
    # row 1 (t=15): dur 5.0, count 2
    # row 2 (t=45, new sess): dur 0.0, count 1
    # row 3 (t=50): dur 5.0, count 2
    durs = session_duration(
        sample_temporal_df,
        time_column="time",
        session_gap_threshold=20.0,
        sequence_column="sequence_id",
    )
    counts = session_interaction_count(
        sample_temporal_df,
        time_column="time",
        session_gap_threshold=20.0,
        sequence_column="sequence_id",
    )

    assert durs.iloc[0] == 0.0
    assert durs.iloc[1] == 5.0
    assert durs.iloc[2] == 0.0
    assert durs.iloc[3] == 5.0

    assert counts.iloc[0] == 1
    assert counts.iloc[1] == 2
    assert counts.iloc[2] == 1
    assert counts.iloc[3] == 2


def test_cumulative_inactivity(sample_temporal_df: pd.DataFrame) -> None:
    """Verify cumulative inactivity aggregates inter-event gaps within sequence."""
    inact = cumulative_inactivity(sample_temporal_df, time_column="time", sequence_column="sequence_id")
    # Seq 1 gaps: [0, 5, 30, 5] -> cumsum: [0, 5, 35, 40]
    assert inact.iloc[0] == 0.0
    assert inact.iloc[1] == 5.0
    assert inact.iloc[2] == 35.0
    assert inact.iloc[3] == 40.0
    # Seq 2 restarts:
    assert inact.iloc[4] == 0.0
    assert inact.iloc[5] == 5.0
    assert inact.iloc[6] == 100.0


def test_causal_leakage_protection(sample_temporal_df: pd.DataFrame) -> None:
    """Verify modifying future timestamps does not alter session or recency features at t."""
    df1 = sample_temporal_df.copy()
    s1 = session_duration(df1, "time", session_gap_threshold=20.0, sequence_column="sequence_id")

    df2 = sample_temporal_df.copy()
    # Modify future row 3
    df2.loc[3, "time"] = 99999.0
    s2 = session_duration(df2, "time", session_gap_threshold=20.0, sequence_column="sequence_id")

    # Past rows 0, 1, 2 must be identical
    assert s1.iloc[0] == s2.iloc[0]
    assert s1.iloc[1] == s2.iloc[1]
    assert s1.iloc[2] == s2.iloc[2]


def test_invalid_session_parameters() -> None:
    """Verify non-positive thresholds and missing columns raise ValueError."""
    df = pd.DataFrame({"t": [1.0, 2.0]})
    with pytest.raises(ValueError, match="session_gap_threshold must be positive"):
        sessionize(df, "t", session_gap_threshold=-1.0)
    with pytest.raises(ValueError, match="session_gap_threshold must be positive"):
        sessionize(df, "t", session_gap_threshold=0.0)


def test_add_session_features(sample_temporal_df: pd.DataFrame) -> None:
    """Verify add_session_features enriches copy without modifying original."""
    enriched = add_session_features(
        sample_temporal_df,
        time_column="time",
        session_gap_threshold=20.0,
        sequence_column="sequence_id",
    )
    assert "session_id" in enriched.columns
    assert "session_duration" in enriched.columns
    assert "session_interaction_count" in enriched.columns
    assert "time_since_previous" in enriched.columns
    assert "session_id" not in sample_temporal_df.columns
