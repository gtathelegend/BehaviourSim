"""Causal recency, streak, and session feature engineering for BehaviorSim.

Provides sequence-aware, causal behavioral features including streak lengths,
previous values, inter-event time gaps, session identification, session duration,
and cumulative inactivity.
All features operate strictly under temporal causality: interaction t never
depends on events after t.
"""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from behaviorsim.feature_engineering.windows import _validate_and_extract_df


def _get_time_diff_seconds(s: pd.Series) -> pd.Series:
    """Calculate forward differences in numeric or datetime series as float seconds."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.diff().dt.total_seconds()
    return pd.to_numeric(s, errors="coerce").diff()


def previous_value(
    data: Any,
    column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    default: Any = None,
) -> pd.Series:
    """Extract causal previous value within each sequence.

    For sequence [A, B, B, C], returns [default, A, B, B].
    At sequence boundaries, strictly returns `default`.

    Args:
        data: pd.DataFrame or CalibrationData.
        column: Name of column to lag.
        sequence_column: Optional sequence partition column.
        order_column: Optional temporal ordering column.
        default: Fallback value for first interaction of a sequence (default None).

    Returns:
        pd.Series containing previous value, aligned with input data index.
    """
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, order_column, column)
    if df.empty:
        return pd.Series(dtype=object, index=orig_index, name=column)

    if order_column is not None:
        sort_cols = [seq_col, order_column] if seq_col is not None else [order_column]
        sorted_df = df.sort_values(sort_cols, kind="stable")
    else:
        sorted_df = df

    if seq_col is not None and seq_col in sorted_df.columns:
        results: List[pd.Series] = []
        for _, group in sorted_df.groupby(seq_col, sort=False):
            res = group[column].shift(1)
            results.append(res)
        out = pd.concat(results) if results else pd.Series(dtype=object, index=sorted_df.index)
    else:
        out = sorted_df[column].shift(1)

    if default is not None:
        out = out.fillna(default)
    else:
        out = pd.Series([None if pd.isna(x) else x for x in out], index=out.index, dtype=object)

    final = out.reindex(orig_index)
    final.name = f"{column}_prev"
    return final


def streak_length(
    data: Any,
    column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    target_value: Optional[Any] = None,
) -> pd.Series:
    """Calculate causal consecutive streak length prior to interaction t.

    Causal convention:
        Represents the number of consecutive prior occurrences of the value observed
        at interaction t immediately before row t.
        Example for column [A, A, A, B, B, A]:
            returns [0, 1, 2, 0, 1, 0]
        If `target_value` is specified (e.g. target_value='A'):
            returns the number of consecutive prior occurrences of target_value:
            [0, 1, 2, 3, 0, 0]
        At sequence boundaries, the streak resets to 0.

    Args:
        data: pd.DataFrame or CalibrationData.
        column: Categorical/state column to compute streak for.
        sequence_column: Optional sequence partition column.
        order_column: Optional temporal order column.
        target_value: Optional specific value to track streak for.

    Returns:
        pd.Series of integer streak lengths aligned with input index.
    """
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, order_column, column)
    if df.empty:
        return pd.Series(dtype=int, index=orig_index, name=f"{column}_streak")

    if order_column is not None:
        sort_cols = [seq_col, order_column] if seq_col is not None else [order_column]
        sorted_df = df.sort_values(sort_cols, kind="stable")
    else:
        sorted_df = df

    def _calc_streak(s: pd.Series) -> pd.Series:
        n = len(s)
        out = np.zeros(n, dtype=int)
        if n == 0:
            return pd.Series(out, index=s.index)

        if target_value is not None:
            count = 0
            for i in range(n):
                out[i] = count
                if s.iloc[i] == target_value:
                    count += 1
                else:
                    count = 0
        else:
            current_val = None
            count = 0
            for i in range(n):
                val = s.iloc[i]
                if i == 0:
                    out[i] = 0
                    current_val = val
                    count = 1
                else:
                    if val == current_val:
                        out[i] = count
                        count += 1
                    else:
                        out[i] = 0
                        current_val = val
                        count = 1
        return pd.Series(out, index=s.index)

    if seq_col is not None and seq_col in sorted_df.columns:
        results: List[pd.Series] = []
        for _, group in sorted_df.groupby(seq_col, sort=False):
            results.append(_calc_streak(group[column]))
        out_series = pd.concat(results) if results else pd.Series(dtype=int, index=sorted_df.index)
    else:
        out_series = _calc_streak(sorted_df[column])

    final_series = out_series.reindex(orig_index)
    final_series.name = f"{column}_streak"
    return final_series


def time_since_previous(
    data: Any,
    time_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Calculate elapsed time / gap since the previous interaction in the sequence.

    Supports numeric and datetime columns. At sequence boundaries, returns `default`.

    Args:
        data: pd.DataFrame or CalibrationData.
        time_column: Name of numeric or datetime timestamp column.
        sequence_column: Optional sequence partition column.
        order_column: Optional order column (defaults to time_column if None).
        default: Fallback value for the first interaction of a sequence (default np.nan).

    Returns:
        pd.Series of float elapsed time deltas aligned with input index.
    """
    effective_order = order_column if order_column is not None else time_column
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, effective_order, time_column)
    if df.empty:
        return pd.Series(dtype=float, index=orig_index, name=f"{time_column}_gap")

    sort_cols = [seq_col, effective_order] if seq_col is not None else [effective_order]
    sorted_df = df.sort_values(sort_cols, kind="stable")

    if seq_col is not None and seq_col in sorted_df.columns:
        results: List[pd.Series] = []
        for _, group in sorted_df.groupby(seq_col, sort=False):
            res = _get_time_diff_seconds(group[time_column])
            results.append(res)
        out_series = pd.concat(results) if results else pd.Series(dtype=float, index=sorted_df.index)
    else:
        out_series = _get_time_diff_seconds(sorted_df[time_column])

    if default is not None:
        out_series = out_series.fillna(default)

    final = out_series.reindex(orig_index)
    final.name = f"{time_column}_gap"
    return final


def interaction_gap(
    data: Any,
    time_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Alias for time_since_previous."""
    return time_since_previous(
        data=data,
        time_column=time_column,
        sequence_column=sequence_column,
        order_column=order_column,
        default=default,
    )


def sessionize(
    data: Any,
    time_column: str,
    session_gap_threshold: float,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    global_unique: bool = False,
) -> pd.Series:
    """Causally partition interactions into sessions based on an inactivity gap threshold.

    A new session is initiated when the elapsed time since the previous interaction
    exceeds `session_gap_threshold` or at the start of a sequence.

    Args:
        data: pd.DataFrame or CalibrationData.
        time_column: Timestamp or sequential time column.
        session_gap_threshold: Positive threshold above which a new session begins.
        sequence_column: Optional sequence partition column.
        order_column: Optional temporal order column.
        global_unique: If True, returns unique string IDs (f'{seq}_{sess}').
            If False, returns 0-indexed integer session IDs within sequence.

    Returns:
        pd.Series containing session IDs aligned with input index.

    Raises:
        ValueError: If session_gap_threshold is non-positive or columns are missing.
    """
    if isinstance(session_gap_threshold, bool) or not isinstance(session_gap_threshold, (int, float)):
        raise TypeError(f"session_gap_threshold must be numeric, got {type(session_gap_threshold).__name__}.")
    if session_gap_threshold <= 0:
        raise ValueError(f"session_gap_threshold must be positive, got {session_gap_threshold}.")

    effective_order = order_column if order_column is not None else time_column
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, effective_order, time_column)
    if df.empty:
        dtype = object if global_unique else int
        return pd.Series(dtype=dtype, index=orig_index, name="session_id")

    sort_cols = [seq_col, effective_order] if seq_col is not None else [effective_order]
    sorted_df = df.sort_values(sort_cols, kind="stable")

    def _calc_session(sub_df: pd.DataFrame) -> pd.Series:
        diffs = _get_time_diff_seconds(sub_df[time_column])
        is_new = diffs.isna() | (diffs > session_gap_threshold)
        sess_ids = is_new.cumsum() - 1
        if global_unique and seq_col is not None and seq_col in sub_df.columns:
            seq_val = str(sub_df[seq_col].iloc[0])
            return seq_val + "_" + sess_ids.astype(str)
        return sess_ids

    if seq_col is not None and seq_col in sorted_df.columns:
        results: List[pd.Series] = []
        for _, group in sorted_df.groupby(seq_col, sort=False):
            results.append(_calc_session(group))
        out_series = pd.concat(results) if results else pd.Series(index=sorted_df.index)
    else:
        out_series = _calc_session(sorted_df)

    final = out_series.reindex(orig_index)
    final.name = "session_id"
    return final


def session_boundary_indicator(
    data: Any,
    time_column: str,
    session_gap_threshold: float,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.Series:
    """Return 1 if the interaction initiates a new session, else 0."""
    gaps = time_since_previous(
        data, time_column, sequence_column=sequence_column, order_column=order_column, default=np.nan
    )
    is_boundary = gaps.isna() | (gaps > session_gap_threshold)
    res = is_boundary.astype(int)
    res.name = "is_session_start"
    return res


def session_duration(
    data: Any,
    time_column: str,
    *,
    session_gap_threshold: Optional[float] = None,
    session_column: Optional[str] = None,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.Series:
    """Calculate causal elapsed time from current session start to interaction t.

    The duration for interaction t depends only on historical events up to t.
    At the first interaction of a session, duration is 0.0.

    Args:
        data: pd.DataFrame or CalibrationData.
        time_column: Timestamp column name.
        session_gap_threshold: Inactivity threshold if sessions are derived from gaps.
        session_column: Pre-existing session column name (if sessions already identified).
        sequence_column: Optional sequence partition column.
        order_column: Optional order column.

    Returns:
        pd.Series of float elapsed seconds from session start.
    """
    effective_order = order_column if order_column is not None else time_column
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, effective_order, time_column)
    if df.empty:
        return pd.Series(dtype=float, index=orig_index, name="session_duration")

    # Resolve session assignments
    if session_column is not None:
        if session_column not in df.columns:
            raise ValueError(f"session_column '{session_column}' not found in DataFrame.")
        sess_series = df[session_column]
    elif session_gap_threshold is not None:
        sess_series = sessionize(
            df,
            time_column=time_column,
            session_gap_threshold=session_gap_threshold,
            sequence_column=seq_col,
            order_column=effective_order,
        )
    else:
        raise ValueError("Either session_column or session_gap_threshold must be provided.")

    working_df = df.copy()
    working_df["_temp_sess"] = sess_series

    sort_cols = [seq_col, effective_order] if seq_col is not None else [effective_order]
    sorted_df = working_df.sort_values(sort_cols, kind="stable")

    def _calc_dur(group_df: pd.DataFrame) -> pd.Series:
        times = group_df[time_column]
        if pd.api.types.is_datetime64_any_dtype(times):
            start_time = times.iloc[0]
            return (times - start_time).dt.total_seconds()
        start_time = float(times.iloc[0])
        return (pd.to_numeric(times, errors="coerce") - start_time).astype(float)

    group_keys = [seq_col, "_temp_sess"] if seq_col is not None else ["_temp_sess"]
    results: List[pd.Series] = []
    for _, grp in sorted_df.groupby(group_keys, sort=False):
        results.append(_calc_dur(grp))

    out = pd.concat(results) if results else pd.Series(dtype=float, index=sorted_df.index)
    final = out.reindex(orig_index)
    final.name = "session_duration"
    return final


def session_elapsed(
    data: Any,
    time_column: str,
    *,
    session_gap_threshold: Optional[float] = None,
    session_column: Optional[str] = None,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.Series:
    """Alias for session_duration."""
    return session_duration(
        data=data,
        time_column=time_column,
        session_gap_threshold=session_gap_threshold,
        session_column=session_column,
        sequence_column=sequence_column,
        order_column=order_column,
    )


def session_interaction_count(
    data: Any,
    *,
    time_column: Optional[str] = None,
    session_gap_threshold: Optional[float] = None,
    session_column: Optional[str] = None,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.Series:
    """Calculate number of interactions in the current session up to row t (1-indexed)."""
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, order_column)
    if df.empty:
        return pd.Series(dtype=int, index=orig_index, name="session_interaction_count")

    if session_column is not None:
        if session_column not in df.columns:
            raise ValueError(f"session_column '{session_column}' not found in DataFrame.")
        sess_series = df[session_column]
    elif session_gap_threshold is not None and time_column is not None:
        sess_series = sessionize(
            df,
            time_column=time_column,
            session_gap_threshold=session_gap_threshold,
            sequence_column=seq_col,
            order_column=order_column,
        )
    else:
        raise ValueError("Must provide either session_column or (time_column and session_gap_threshold).")

    working_df = df.copy()
    working_df["_temp_sess"] = sess_series

    effective_order = order_column if order_column is not None else time_column
    if effective_order is not None:
        sort_cols = [seq_col, effective_order] if seq_col is not None else [effective_order]
        sorted_df = working_df.sort_values(sort_cols, kind="stable")
    else:
        sorted_df = working_df

    group_keys = [seq_col, "_temp_sess"] if seq_col is not None else ["_temp_sess"]
    results: List[pd.Series] = []
    for _, grp in sorted_df.groupby(group_keys, sort=False):
        counts = pd.Series(np.arange(1, len(grp) + 1, dtype=int), index=grp.index)
        results.append(counts)

    out = pd.concat(results) if results else pd.Series(dtype=int, index=sorted_df.index)
    final = out.reindex(orig_index)
    final.name = "session_interaction_count"
    return final


def cumulative_inactivity(
    data: Any,
    time_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.Series:
    """Calculate cumulative inactivity time (sum of inter-event gaps) within the sequence."""
    gaps = time_since_previous(
        data, time_column, sequence_column=sequence_column, order_column=order_column, default=0.0
    )
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, order_column, time_column)
    if df.empty:
        return pd.Series(dtype=float, index=orig_index, name="cumulative_inactivity")

    working_df = df.copy()
    working_df["_gap"] = gaps

    if order_column is not None:
        sort_cols = [seq_col, order_column] if seq_col is not None else [order_column]
        sorted_df = working_df.sort_values(sort_cols, kind="stable")
    else:
        sorted_df = working_df

    if seq_col is not None and seq_col in sorted_df.columns:
        results: List[pd.Series] = []
        for _, grp in sorted_df.groupby(seq_col, sort=False):
            results.append(grp["_gap"].cumsum())
        out = pd.concat(results) if results else pd.Series(dtype=float, index=sorted_df.index)
    else:
        out = sorted_df["_gap"].cumsum()

    final = out.reindex(orig_index)
    final.name = "cumulative_inactivity"
    return final


def add_session_features(
    data: pd.DataFrame,
    time_column: str,
    session_gap_threshold: float,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.DataFrame:
    """Non-destructively enrich a DataFrame with session ID, duration, count, and gap columns."""
    df = data.copy()
    df["session_id"] = sessionize(
        df,
        time_column=time_column,
        session_gap_threshold=session_gap_threshold,
        sequence_column=sequence_column,
        order_column=order_column,
    )
    df["time_since_previous"] = time_since_previous(
        df,
        time_column=time_column,
        sequence_column=sequence_column,
        order_column=order_column,
    )
    df["session_duration"] = session_duration(
        df,
        time_column=time_column,
        session_column="session_id",
        sequence_column=sequence_column,
        order_column=order_column,
    )
    df["session_interaction_count"] = session_interaction_count(
        df,
        session_column="session_id",
        sequence_column=sequence_column,
        order_column=order_column,
    )
    return df
