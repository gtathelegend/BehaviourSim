"""Unit and causal correctness tests for BehaviorSim state-transition behavioral features."""

import numpy as np
import pandas as pd
import pytest

from behaviorsim.feature_engineering.transitions import (
    add_transition_features,
    cumulative_transition_count,
    historical_transition_count,
    previous_state,
    state_persistence,
    time_in_state,
    transition_indicator,
    transition_recency,
)


@pytest.fixture
def sample_transition_df() -> pd.DataFrame:
    """Fixture providing multi-sequence data with transitions and times."""
    return pd.DataFrame(
        {
            "sequence_id": [1, 1, 1, 1, 1, 1, 2, 2, 2],
            "state": ["idle", "idle", "active", "active", "idle", "active", "active", "idle", "idle"],
            "time": [10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 100.0, 110.0, 120.0],
        }
    )


def test_previous_state_and_transition_indicator(sample_transition_df: pd.DataFrame) -> None:
    """Verify previous_state and transition_indicator causal calculations."""
    prev_s = previous_state(sample_transition_df, "state", sequence_column="sequence_id")
    is_trans = transition_indicator(sample_transition_df, "state", sequence_column="sequence_id")

    # Sequence 1: idle, idle, active, active, idle, active
    # prev_state: None, idle, idle, active, active, idle
    # is_trans:   0,    0,    1,     0,      1,    1
    assert prev_s.iloc[0] is None
    assert prev_s.iloc[1] == "idle"
    assert prev_s.iloc[2] == "idle"
    assert prev_s.iloc[3] == "active"

    assert is_trans.iloc[0] == 0
    assert is_trans.iloc[1] == 0
    assert is_trans.iloc[2] == 1
    assert is_trans.iloc[3] == 0
    assert is_trans.iloc[4] == 1
    assert is_trans.iloc[5] == 1

    # Sequence 2 restart:
    assert prev_s.iloc[6] is None
    assert is_trans.iloc[6] == 0
    assert is_trans.iloc[7] == 1


def test_cumulative_transition_count(sample_transition_df: pd.DataFrame) -> None:
    """Verify cumulative_transition_count resets at sequence boundaries."""
    trans_count = cumulative_transition_count(sample_transition_df, "state", sequence_column="sequence_id")
    # Seq 1: [0, 0, 1, 1, 2, 3]
    # Seq 2: [0, 1, 1]
    assert trans_count.tolist() == [0, 0, 1, 1, 2, 3, 0, 1, 1]


def test_state_persistence(sample_transition_df: pd.DataFrame) -> None:
    """Verify state_persistence increments consecutively and resets on transitions."""
    persist = state_persistence(sample_transition_df, "state", sequence_column="sequence_id")
    # Seq 1: idle (1), idle (2), active (1), active (2), idle (1), active (1)
    # Seq 2: active (1), idle (1), idle (2)
    assert persist.tolist() == [1, 2, 1, 2, 1, 1, 1, 1, 2]


def test_time_in_state(sample_transition_df: pd.DataFrame) -> None:
    """Verify time_in_state measures elapsed time since entry into current state."""
    tis = time_in_state(sample_transition_df, "state", time_column="time", sequence_column="sequence_id")
    # Seq 1 times: [10, 12, 15, 20, 25, 30]
    # idle: 10 -> 0.0, 12 -> 2.0
    # active: 15 -> 0.0, 20 -> 5.0
    # idle: 25 -> 0.0
    # active: 30 -> 0.0
    assert tis.iloc[0] == 0.0
    assert tis.iloc[1] == 2.0
    assert tis.iloc[2] == 0.0
    assert tis.iloc[3] == 5.0
    assert tis.iloc[4] == 0.0
    assert tis.iloc[5] == 0.0

    # Seq 2 times: [100, 110, 120]
    assert tis.iloc[6] == 0.0
    assert tis.iloc[7] == 0.0
    assert tis.iloc[8] == 10.0


def test_transition_recency(sample_transition_df: pd.DataFrame) -> None:
    """Verify transition_recency tracks interactions or time since the most recent transition."""
    # Interaction count recency
    rec = transition_recency(sample_transition_df, "state", sequence_column="sequence_id")
    # Seq 1:
    # 0 (idle): start -> 0
    # 1 (idle): no trans -> 1
    # 2 (active): trans -> 0
    # 3 (active): no trans -> 1
    # 4 (idle): trans -> 0
    # 5 (active): trans -> 0
    assert rec.iloc[0] == 0
    assert rec.iloc[1] == 1
    assert rec.iloc[2] == 0
    assert rec.iloc[3] == 1
    assert rec.iloc[4] == 0
    assert rec.iloc[5] == 0


def test_historical_transition_count(sample_transition_df: pd.DataFrame) -> None:
    """Verify historical_transition_count counts specific transition pairs."""
    # Count idle -> active transitions in Seq 1
    # transitions occur at row 2 and row 5
    hist_cnt = historical_transition_count(
        sample_transition_df,
        state_column="state",
        from_state="idle",
        to_state="active",
        sequence_column="sequence_id",
    )
    # Seq 1: [0, 0, 1, 1, 1, 2]
    # Seq 2: [0, 0, 0] (no idle -> active transition in Seq 2)
    assert hist_cnt.tolist() == [0, 0, 1, 1, 1, 2, 0, 0, 0]


def test_no_future_leakage_transitions(sample_transition_df: pd.DataFrame) -> None:
    """Hard requirement: altering future states does NOT alter transition features at t."""
    df1 = sample_transition_df.copy()
    c1 = cumulative_transition_count(df1, "state", sequence_column="sequence_id")

    df2 = sample_transition_df.copy()
    # Change future rows 4 and 5
    df2.loc[4, "state"] = "active"
    df2.loc[5, "state"] = "idle"
    c2 = cumulative_transition_count(df2, "state", sequence_column="sequence_id")

    # Past rows 0, 1, 2, 3 must remain identical
    assert c1.iloc[0] == c2.iloc[0]
    assert c1.iloc[1] == c2.iloc[1]
    assert c1.iloc[2] == c2.iloc[2]
    assert c1.iloc[3] == c2.iloc[3]


def test_add_transition_features(sample_transition_df: pd.DataFrame) -> None:
    """Verify add_transition_features enriches copy without mutating original."""
    enriched = add_transition_features(
        sample_transition_df,
        state_column="state",
        time_column="time",
        sequence_column="sequence_id",
    )
    assert "previous_state" in enriched.columns
    assert "is_transition" in enriched.columns
    assert "transition_count" in enriched.columns
    assert "state_persistence" in enriched.columns
    assert "transition_recency" in enriched.columns
    assert "time_in_state" in enriched.columns
    assert "previous_state" not in sample_transition_df.columns
