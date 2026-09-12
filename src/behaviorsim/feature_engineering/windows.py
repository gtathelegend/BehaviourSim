"""Causal rolling and expanding feature engineering for BehaviorSim.

This module provides sequence-aware, causal rolling and cumulative window features.
All temporal statistics strictly follow the causal convention:
    feature[t] = f(x[0:t])
meaning the observation at interaction t is NEVER allowed to influence its own
historical feature value (closed='left'). Observations are strictly isolated
between sequences to prevent cross-sequence leakage.
"""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd


def _validate_and_extract_df(
    data: Any,
    sequence_column: Optional[str],
    order_column: Optional[str],
    value_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str], pd.Index]:
    """Validate input data and extract underlying DataFrame and ordering index."""
    # Support CalibrationData if imported
    if hasattr(data, "data") and isinstance(getattr(data, "data"), pd.DataFrame):
        df = getattr(data, "data")
        if sequence_column is None:
            sequence_column = getattr(data, "sequence_column", None)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(
            f"data must be a pandas DataFrame or CalibrationData, got {type(data).__name__}."
        )

    if value_column is not None:
        if not isinstance(value_column, str) or not value_column.strip():
            raise ValueError("value_column must be a non-empty string.")
        if value_column not in df.columns:
            raise ValueError(
                f"value_column '{value_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}."
            )

    if sequence_column is not None:
        if not isinstance(sequence_column, str) or not sequence_column.strip():
            raise ValueError("sequence_column must be a non-empty string.")
        if sequence_column not in df.columns:
            raise ValueError(
                f"sequence_column '{sequence_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}."
            )

    if order_column is not None:
        if not isinstance(order_column, str) or not order_column.strip():
            raise ValueError("order_column must be a non-empty string.")
        if order_column not in df.columns:
            raise ValueError(
                f"order_column '{order_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}."
            )

    return df, sequence_column, df.index


def _apply_sequence_transform(
    df: pd.DataFrame,
    sequence_column: Optional[str],
    order_column: Optional[str],
    value_column: str,
    calc_fn: Callable[[pd.Series], pd.Series],
    fill_default: Optional[float] = None,
) -> pd.Series:
    """Apply a causal window calculation function per sequence with strict ordering."""
    if df.empty:
        return pd.Series(dtype=float, index=df.index, name=value_column)

    orig_index = df.index

    # Sort if order_column is provided
    if order_column is not None:
        sort_cols = [sequence_column, order_column] if sequence_column is not None else [order_column]
        sorted_df = df.sort_values(sort_cols, kind="stable")
    else:
        sorted_df = df

    if sequence_column is not None and sequence_column in sorted_df.columns:
        results: List[pd.Series] = []
        for _, group in sorted_df.groupby(sequence_column, sort=False):
            s = group[value_column]
            res_group = calc_fn(s)
            results.append(res_group)
        out_series = pd.concat(results) if results else pd.Series(dtype=float, index=sorted_df.index)
    else:
        out_series = calc_fn(sorted_df[value_column])

    if fill_default is not None:
        out_series = out_series.fillna(fill_default)

    # Reindex back to original DataFrame index to ensure exact row alignment
    final_series = out_series.reindex(orig_index)
    final_series.name = value_column
    return final_series


def rolling_mean(
    data: Any,
    value_column: str,
    window: int,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    min_periods: int = 1,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Calculate causal rolling mean strictly excluding the current interaction.

    Convention:
        feature[t] = mean(x[max(0, t - window) : t])
    At interaction t=0, the feature value has no prior observations and returns
    `default` (np.nan). Observations are strictly isolated across sequences.

    Args:
        data: pd.DataFrame or CalibrationData containing interactions.
        value_column: Name of the numeric column to aggregate.
        window: Positive integer window size.
        sequence_column: Optional column partitioning sequences (e.g. 'sequence_id').
        order_column: Optional column specifying temporal ordering.
        min_periods: Minimum number of historical observations required (default 1).
        default: Fallback value when historical observations are fewer than min_periods.

    Returns:
        pd.Series of rolling means aligned with input data index.

    Raises:
        TypeError: If input types are invalid.
        ValueError: If window <= 0 or required columns are missing.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}.")

    df, seq_col, _ = _validate_and_extract_df(data, sequence_column, order_column, value_column)

    def _calc(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.rolling(window=window, min_periods=min_periods).mean()

    return _apply_sequence_transform(df, seq_col, order_column, value_column, _calc, fill_default=default)


def rolling_variance(
    data: Any,
    value_column: str,
    window: int,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    min_periods: int = 1,
    ddof: int = 0,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Calculate causal rolling variance strictly excluding the current interaction.

    Convention:
        feature[t] = var(x[max(0, t - window) : t])
    For a single previous observation with ddof=0, variance is 0.0.
    At interaction t=0, returns `default` (np.nan).

    Args:
        data: pd.DataFrame or CalibrationData.
        value_column: Name of numeric column.
        window: Positive integer window size.
        sequence_column: Optional sequence partition column.
        order_column: Optional temporal order column.
        min_periods: Minimum required observations (default 1).
        ddof: Delta Degrees of Freedom (default 0 for population variance).
        default: Fallback value when history is insufficient.

    Returns:
        pd.Series of rolling variance aligned with input index.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}.")

    df, seq_col, _ = _validate_and_extract_df(data, sequence_column, order_column, value_column)

    def _calc(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        res = shifted.rolling(window=window, min_periods=min_periods).var(ddof=ddof)
        # For single observation with ddof=0, fill variance as 0.0
        if ddof == 0 and min_periods == 1:
            valid_single = shifted.rolling(window=window, min_periods=1).count() == 1
            res = res.mask(valid_single & res.isna(), 0.0)
        return res

    return _apply_sequence_transform(df, seq_col, order_column, value_column, _calc, fill_default=default)


def rolling_sum(
    data: Any,
    value_column: str,
    window: int,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    min_periods: int = 1,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Calculate causal rolling sum strictly excluding the current interaction.

    Convention:
        feature[t] = sum(x[max(0, t - window) : t])
    """
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}.")

    df, seq_col, _ = _validate_and_extract_df(data, sequence_column, order_column, value_column)

    def _calc(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.rolling(window=window, min_periods=min_periods).sum()

    return _apply_sequence_transform(df, seq_col, order_column, value_column, _calc, fill_default=default)


def rolling_min(
    data: Any,
    value_column: str,
    window: int,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    min_periods: int = 1,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Calculate causal rolling minimum strictly excluding the current interaction."""
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}.")

    df, seq_col, _ = _validate_and_extract_df(data, sequence_column, order_column, value_column)

    def _calc(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.rolling(window=window, min_periods=min_periods).min()

    return _apply_sequence_transform(df, seq_col, order_column, value_column, _calc, fill_default=default)


def rolling_max(
    data: Any,
    value_column: str,
    window: int,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    min_periods: int = 1,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Calculate causal rolling maximum strictly excluding the current interaction."""
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be a positive integer, got {window}.")

    df, seq_col, _ = _validate_and_extract_df(data, sequence_column, order_column, value_column)

    def _calc(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.rolling(window=window, min_periods=min_periods).max()

    return _apply_sequence_transform(df, seq_col, order_column, value_column, _calc, fill_default=default)


def expanding_mean(
    data: Any,
    value_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    min_periods: int = 1,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Calculate causal cumulative/expanding mean of all prior observations."""
    df, seq_col, _ = _validate_and_extract_df(data, sequence_column, order_column, value_column)

    def _calc(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.expanding(min_periods=min_periods).mean()

    return _apply_sequence_transform(df, seq_col, order_column, value_column, _calc, fill_default=default)


def expanding_variance(
    data: Any,
    value_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    min_periods: int = 1,
    ddof: int = 0,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Calculate causal cumulative/expanding variance of all prior observations."""
    df, seq_col, _ = _validate_and_extract_df(data, sequence_column, order_column, value_column)

    def _calc(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        res = shifted.expanding(min_periods=min_periods).var(ddof=ddof)
        if ddof == 0 and min_periods == 1:
            valid_single = shifted.expanding(min_periods=1).count() == 1
            res = res.mask(valid_single & res.isna(), 0.0)
        return res

    return _apply_sequence_transform(df, seq_col, order_column, value_column, _calc, fill_default=default)


def expanding_sum(
    data: Any,
    value_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    min_periods: int = 1,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Calculate causal cumulative/expanding sum of all prior observations."""
    df, seq_col, _ = _validate_and_extract_df(data, sequence_column, order_column, value_column)

    def _calc(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.expanding(min_periods=min_periods).sum()

    return _apply_sequence_transform(df, seq_col, order_column, value_column, _calc, fill_default=default)


def expanding_min(
    data: Any,
    value_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    min_periods: int = 1,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Calculate causal cumulative/expanding minimum of all prior observations."""
    df, seq_col, _ = _validate_and_extract_df(data, sequence_column, order_column, value_column)

    def _calc(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.expanding(min_periods=min_periods).min()

    return _apply_sequence_transform(df, seq_col, order_column, value_column, _calc, fill_default=default)


def expanding_max(
    data: Any,
    value_column: str,
    *,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    min_periods: int = 1,
    default: Optional[float] = np.nan,
) -> pd.Series:
    """Calculate causal cumulative/expanding maximum of all prior observations."""
    df, seq_col, _ = _validate_and_extract_df(data, sequence_column, order_column, value_column)

    def _calc(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.expanding(min_periods=min_periods).max()

    return _apply_sequence_transform(df, seq_col, order_column, value_column, _calc, fill_default=default)


def add_rolling_features(
    data: pd.DataFrame,
    value_columns: Sequence[str],
    windows: Sequence[int],
    *,
    stats: Sequence[str] = ("mean", "var"),
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
    min_periods: int = 1,
    default: Optional[float] = np.nan,
) -> pd.DataFrame:
    """Non-destructively enrich a DataFrame with causal rolling feature columns.

    Args:
        data: Input pandas DataFrame.
        value_columns: Sequence of column names to compute rolling features for.
        windows: Window lengths (e.g. [3, 5, 10]).
        stats: Statistics to compute: subset of ('mean', 'var', 'sum', 'min', 'max').
        sequence_column: Optional column partitioning sequences.
        order_column: Optional temporal order column.
        min_periods: Minimum required observations.
        default: Fallback value for insufficient history.

    Returns:
        New pd.DataFrame containing original columns plus generated rolling features.
    """
    df = data.copy()
    stat_funcs = {
        "mean": rolling_mean,
        "var": rolling_variance,
        "sum": rolling_sum,
        "min": rolling_min,
        "max": rolling_max,
    }

    for col in value_columns:
        for w in windows:
            for st in stats:
                if st not in stat_funcs:
                    raise ValueError(f"Unsupported rolling statistic '{st}'. Supported: {list(stat_funcs.keys())}.")
                fn = stat_funcs[st]
                feat_name = f"{col}_rolling_{st}_w{w}"
                df[feat_name] = fn(
                    df,
                    value_column=col,
                    window=w,
                    sequence_column=sequence_column,
                    order_column=order_column,
                    min_periods=min_periods,
                    default=default,
                )
    return df
