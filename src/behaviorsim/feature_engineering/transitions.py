"""State-transition behavioral feature engineering for BehaviorSim.

Provides sequence-aware, causal state-transition features including previous states,
transition indicators, cumulative transition counts, state persistence, time in state,
transition recency, and historical transition pair frequencies.
All features operate strictly under temporal causality: interaction t never
depends on transitions occurring after interaction t.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from behaviorsim.feature_engineering.windows import _validate_and_extract_df


def previous_state(
    data: Any,
    state_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    default: Any = None,
) -> pd.Series:
    """Extract causal previous state within each sequence.

    For sequence [A, B, B, C], returns [default, A, B, B].
    At sequence boundaries, strictly returns `default`.

    Args:
        data: pd.DataFrame or CalibrationData.
        state_column: Name of column containing state labels.
        sequence_column: Optional sequence partition column.
        order_column: Optional temporal order column.
        default: Value assigned to the first interaction of a sequence (default None).

    Returns:
        pd.Series containing previous state labels aligned with input index.
    """
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, order_column, state_column)
    if df.empty:
        return pd.Series(dtype=object, index=orig_index, name="previous_state")

    if order_column is not None:
        sort_cols = [seq_col, order_column] if seq_col is not None else [order_column]
        sorted_df = df.sort_values(sort_cols, kind="stable")
    else:
        sorted_df = df

    if seq_col is not None and seq_col in sorted_df.columns:
        results: List[pd.Series] = []
        for _, group in sorted_df.groupby(seq_col, sort=False):
            res = group[state_column].shift(1)
            results.append(res)
        out = pd.concat(results) if results else pd.Series(dtype=object, index=sorted_df.index)
    else:
        out = sorted_df[state_column].shift(1)

    if default is not None:
        out = out.fillna(default)
    else:
        out = pd.Series([None if pd.isna(x) else x for x in out], index=out.index, dtype=object)

    final = out.reindex(orig_index)
    final.name = "previous_state"
    return final


def transition_indicator(
    data: Any,
    state_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.Series:
    """Return 1 if current interaction represents a state transition, else 0.

    At sequence start (t=0), returns 0 because there is no prior within-sequence transition.

    Args:
        data: pd.DataFrame or CalibrationData.
        state_column: Column containing state labels.
        sequence_column: Optional sequence partition column.
        order_column: Optional temporal order column.

    Returns:
        pd.Series of integer 0 or 1 indicator flags.
    """
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, order_column, state_column)
    if df.empty:
        return pd.Series(dtype=int, index=orig_index, name="is_transition")

    prev_s = previous_state(df, state_column, sequence_column=seq_col, order_column=order_column, default=None)
    # Where previous state is None/NaN (sequence start), transition indicator is 0
    is_trans = (prev_s.notna()) & (df[state_column] != prev_s)
    res = is_trans.astype(int)
    res.name = "is_transition"
    return res


def cumulative_transition_count(
    data: Any,
    state_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.Series:
    """Calculate cumulative number of state transitions observed up to interaction t.

    Resets to 0 at the start of each sequence.
    """
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, order_column, state_column)
    if df.empty:
        return pd.Series(dtype=int, index=orig_index, name="transition_count")

    trans = transition_indicator(df, state_column, sequence_column=seq_col, order_column=order_column)
    working_df = df.copy()
    working_df["_trans"] = trans

    if order_column is not None:
        sort_cols = [seq_col, order_column] if seq_col is not None else [order_column]
        sorted_df = working_df.sort_values(sort_cols, kind="stable")
    else:
        sorted_df = working_df

    if seq_col is not None and seq_col in sorted_df.columns:
        results: List[pd.Series] = []
        for _, grp in sorted_df.groupby(seq_col, sort=False):
            results.append(grp["_trans"].cumsum())
        out = pd.concat(results) if results else pd.Series(dtype=int, index=sorted_df.index)
    else:
        out = sorted_df["_trans"].cumsum()

    final = out.reindex(orig_index)
    final.name = "transition_count"
    return final


def state_persistence(
    data: Any,
    state_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.Series:
    """Calculate consecutive interactions spent in the current state (1-indexed).

    Example:
        state: [A, A, A, B, B, A] -> persistence: [1, 2, 3, 1, 2, 1]
    Resets to 1 upon state transition or sequence boundary.

    Args:
        data: pd.DataFrame or CalibrationData.
        state_column: Name of state column.
        sequence_column: Optional sequence partition column.
        order_column: Optional temporal order column.

    Returns:
        pd.Series of integer persistence counts aligned with input index.
    """
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, order_column, state_column)
    if df.empty:
        return pd.Series(dtype=int, index=orig_index, name="state_persistence")

    if order_column is not None:
        sort_cols = [seq_col, order_column] if seq_col is not None else [order_column]
        sorted_df = df.sort_values(sort_cols, kind="stable")
    else:
        sorted_df = df

    def _calc_persistence(s: pd.Series) -> pd.Series:
        n = len(s)
        out = np.zeros(n, dtype=int)
        if n == 0:
            return pd.Series(out, index=s.index)

        count = 1
        out[0] = 1
        for i in range(1, n):
            if s.iloc[i] == s.iloc[i - 1]:
                count += 1
            else:
                count = 1
            out[i] = count
        return pd.Series(out, index=s.index)

    if seq_col is not None and seq_col in sorted_df.columns:
        results: List[pd.Series] = []
        for _, grp in sorted_df.groupby(seq_col, sort=False):
            results.append(_calc_persistence(grp[state_column]))
        out_series = pd.concat(results) if results else pd.Series(dtype=int, index=sorted_df.index)
    else:
        out_series = _calc_persistence(sorted_df[state_column])

    final = out_series.reindex(orig_index)
    final.name = "state_persistence"
    return final


def time_in_state(
    data: Any,
    state_column: str,
    time_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.Series:
    """Calculate causal elapsed time since entry into current state.

    At state entry (t=0 or immediate transition), returns 0.0.
    """
    effective_order = order_column if order_column is not None else time_column
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, effective_order, state_column)
    if df.empty:
        return pd.Series(dtype=float, index=orig_index, name="time_in_state")

    sort_cols = [seq_col, effective_order] if seq_col is not None else [effective_order]
    sorted_df = df.sort_values(sort_cols, kind="stable")

    def _calc_time_in_state(grp: pd.DataFrame) -> pd.Series:
        n = len(grp)
        out = np.zeros(n, dtype=float)
        if n == 0:
            return pd.Series(out, index=grp.index)

        times = grp[time_column]
        is_dt = pd.api.types.is_datetime64_any_dtype(times)
        states = grp[state_column].values

        entry_time = times.iloc[0]
        out[0] = 0.0

        for i in range(1, n):
            if states[i] != states[i - 1]:
                entry_time = times.iloc[i]

            curr_time = times.iloc[i]
            if is_dt:
                out[i] = float((curr_time - entry_time).total_seconds())
            else:
                out[i] = float(curr_time - entry_time)

        return pd.Series(out, index=grp.index)

    if seq_col is not None and seq_col in sorted_df.columns:
        results: List[pd.Series] = []
        for _, grp in sorted_df.groupby(seq_col, sort=False):
            results.append(_calc_time_in_state(grp))
        out_series = pd.concat(results) if results else pd.Series(dtype=float, index=sorted_df.index)
    else:
        out_series = _calc_time_in_state(sorted_df)

    final = out_series.reindex(orig_index)
    final.name = "time_in_state"
    return final


def transition_recency(
    data: Any,
    state_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    time_column: Optional[str] = None,
) -> pd.Series:
    """Calculate interactions (or elapsed time) since the most recent state transition.

    At sequence start (t=0), recency is 0.
    At an interaction where a transition occurs, recency is 0.
    If `time_column` is provided, computes elapsed time in seconds since the last transition.
    If `time_column` is None, computes integer interaction count since the last transition.
    """
    effective_order = order_column if order_column is not None else time_column
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, effective_order, state_column)
    if df.empty:
        dtype = float if time_column is not None else int
        return pd.Series(dtype=dtype, index=orig_index, name="transition_recency")

    if effective_order is not None:
        sort_cols = [seq_col, effective_order] if seq_col is not None else [effective_order]
        sorted_df = df.sort_values(sort_cols, kind="stable")
    else:
        sorted_df = df

    def _calc_recency(grp: pd.DataFrame) -> pd.Series:
        n = len(grp)
        states = grp[state_column].values

        if time_column is not None:
            times = grp[time_column]
            is_dt = pd.api.types.is_datetime64_any_dtype(times)
            out = np.zeros(n, dtype=float)
            last_trans_time = times.iloc[0]
            out[0] = 0.0

            for i in range(1, n):
                if states[i] != states[i - 1]:
                    last_trans_time = times.iloc[i]
                curr_time = times.iloc[i]
                if is_dt:
                    out[i] = float((curr_time - last_trans_time).total_seconds())
                else:
                    out[i] = float(curr_time - last_trans_time)
        else:
            out = np.zeros(n, dtype=int)
            count_since_trans = 0
            out[0] = 0
            for i in range(1, n):
                if states[i] != states[i - 1]:
                    count_since_trans = 0
                else:
                    count_since_trans += 1
                out[i] = count_since_trans

        return pd.Series(out, index=grp.index)

    if seq_col is not None and seq_col in sorted_df.columns:
        results: List[pd.Series] = []
        for _, grp in sorted_df.groupby(seq_col, sort=False):
            results.append(_calc_recency(grp))
        out_series = pd.concat(results) if results else pd.Series(index=sorted_df.index)
    else:
        out_series = _calc_recency(sorted_df)

    final = out_series.reindex(orig_index)
    final.name = "transition_recency"
    return final


def historical_transition_count(
    data: Any,
    state_column: str,
    from_state: str,
    to_state: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.Series:
    """Calculate causal cumulative count of specific transition (from_state -> to_state).

    At interaction t, reflects the number of times (from_state -> to_state) occurred
    in interactions up to and including t within the current sequence.
    """
    df, seq_col, orig_index = _validate_and_extract_df(data, sequence_column, order_column, state_column)
    feat_name = f"count_trans_{from_state}_to_{to_state}"
    if df.empty:
        return pd.Series(dtype=int, index=orig_index, name=feat_name)

    prev_s = previous_state(df, state_column, sequence_column=seq_col, order_column=order_column)
    is_target_trans = (prev_s == from_state) & (df[state_column] == to_state)
    working_df = df.copy()
    working_df["_match"] = is_target_trans.astype(int)

    if order_column is not None:
        sort_cols = [seq_col, order_column] if seq_col is not None else [order_column]
        sorted_df = working_df.sort_values(sort_cols, kind="stable")
    else:
        sorted_df = working_df

    if seq_col is not None and seq_col in sorted_df.columns:
        results: List[pd.Series] = []
        for _, grp in sorted_df.groupby(seq_col, sort=False):
            results.append(grp["_match"].cumsum())
        out = pd.concat(results) if results else pd.Series(dtype=int, index=sorted_df.index)
    else:
        out = sorted_df["_match"].cumsum()

    final = out.reindex(orig_index)
    final.name = feat_name
    return final


def add_transition_features(
    data: pd.DataFrame,
    state_column: str,
    *,
    time_column: Optional[str] = None,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.DataFrame:
    """Non-destructively enrich a DataFrame with previous state, transition flag, count, and persistence."""
    df = data.copy()
    df["previous_state"] = previous_state(
        df, state_column, sequence_column=sequence_column, order_column=order_column
    )
    df["is_transition"] = transition_indicator(
        df, state_column, sequence_column=sequence_column, order_column=order_column
    )
    df["transition_count"] = cumulative_transition_count(
        df, state_column, sequence_column=sequence_column, order_column=order_column
    )
    df["state_persistence"] = state_persistence(
        df, state_column, sequence_column=sequence_column, order_column=order_column
    )
    df["transition_recency"] = transition_recency(
        df, state_column, sequence_column=sequence_column, order_column=order_column, time_column=time_column
    )
    if time_column is not None:
        df["time_in_state"] = time_in_state(
            df, state_column, time_column, sequence_column=sequence_column, order_column=order_column
        )
    return df
