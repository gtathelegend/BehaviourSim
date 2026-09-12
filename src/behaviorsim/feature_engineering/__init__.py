"""Feature engineering module for BehaviorSim.

Provides causal, sequence-aware rolling windows, streaks, recency, sessionization,
and state-transition behavioral dynamics.
All temporal feature functions strictly adhere to the causal convention:
    feature[t] = f(x[0:t])
preventing historical data from being influenced by current or future observations.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union
import pandas as pd

from behaviorsim.feature_engineering.causal import (
    add_session_features,
    cumulative_inactivity,
    interaction_gap,
    previous_value,
    session_boundary_indicator,
    session_duration,
    session_elapsed,
    session_interaction_count,
    sessionize,
    streak_length,
    time_since_previous,
)
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
from behaviorsim.feature_engineering.windows import (
    add_rolling_features,
    expanding_max,
    expanding_mean,
    expanding_min,
    expanding_sum,
    expanding_variance,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_sum,
    rolling_variance,
)


def build_behavioral_features(
    data: Any,
    *,
    state_column: Optional[str] = "state",
    time_column: Optional[str] = None,
    numeric_columns: Optional[Sequence[str]] = None,
    rolling_windows: Sequence[int] = (3, 5),
    rolling_stats: Sequence[str] = ("mean", "var"),
    session_gap_threshold: Optional[float] = None,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.DataFrame:
    """Non-destructively generate a comprehensive suite of causal behavioral features.

    Composes rolling statistics, session tracking, and state-transition dynamics
    on a DataFrame or CalibrationData container.

    Args:
        data: pd.DataFrame or CalibrationData instance.
        state_column: Column name with state labels (default 'state').
        time_column: Optional column name with interaction timestamps or ordering.
        numeric_columns: Optional sequence of numeric columns for rolling statistics.
        rolling_windows: Sequence of integer window sizes (default (3, 5)).
        rolling_stats: Rolling statistics to compute (subset of 'mean', 'var', 'sum', 'min', 'max').
        session_gap_threshold: Inactivity threshold for session identification.
        sequence_column: Column partitioning sequences (e.g. 'sequence_id').
        order_column: Column specifying interaction order.

    Returns:
        Enriched copy of the DataFrame containing new causal features.
    """
    if hasattr(data, "data") and isinstance(getattr(data, "data"), pd.DataFrame):
        df = getattr(data, "data").copy()
        if sequence_column is None:
            sequence_column = getattr(data, "sequence_column", None)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"data must be pd.DataFrame or CalibrationData, got {type(data).__name__}.")

    # 1. Rolling features
    if numeric_columns:
        df = add_rolling_features(
            df,
            value_columns=numeric_columns,
            windows=rolling_windows,
            stats=rolling_stats,
            sequence_column=sequence_column,
            order_column=order_column,
        )

    # 2. Session features
    if time_column is not None and session_gap_threshold is not None and time_column in df.columns:
        df = add_session_features(
            df,
            time_column=time_column,
            session_gap_threshold=session_gap_threshold,
            sequence_column=sequence_column,
            order_column=order_column,
        )

    # 3. Transition features
    if state_column is not None and state_column in df.columns:
        df = add_transition_features(
            df,
            state_column=state_column,
            time_column=time_column if time_column in df.columns else None,
            sequence_column=sequence_column,
            order_column=order_column,
        )

    return df


__all__ = [
    "add_rolling_features",
    "add_session_features",
    "add_transition_features",
    "build_behavioral_features",
    "cumulative_inactivity",
    "cumulative_transition_count",
    "expanding_max",
    "expanding_mean",
    "expanding_min",
    "expanding_sum",
    "expanding_variance",
    "historical_transition_count",
    "interaction_gap",
    "previous_state",
    "previous_value",
    "rolling_max",
    "rolling_mean",
    "rolling_min",
    "rolling_sum",
    "rolling_variance",
    "session_boundary_indicator",
    "session_duration",
    "session_elapsed",
    "session_interaction_count",
    "sessionize",
    "state_persistence",
    "streak_length",
    "time_in_state",
    "time_since_previous",
    "transition_indicator",
    "transition_recency",
]
