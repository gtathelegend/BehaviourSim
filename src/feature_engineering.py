"""Feature engineering module for CLSI-Adapt Simulator.

Constructs supervised features and target labels from simulated learner interaction logs.

Temporal Alignment Contract
----------------------------
For each interaction at position t (0-indexed within a learner's sequence):

  X[t]  = feature vector built from interactions 0..t (inclusive). No future data.
  y[t]  = overload label derived from interactions t-4..t (window of prior 4) and t+1..t+3
           (next 3 future interactions). ONLY the target is allowed to use future information.

Learner-Boundary Contract
--------------------------
Sliding windows NEVER cross profile boundaries. When a combined multi-profile DataFrame is
supplied, per-profile processing is applied independently and concatenated.

Overload Target Alignment (exact)
-----------------------------------
An interaction at position t is labeled overload=1 when ALL of:
  1. interactions [t-3 .. t] (the 4 items ending at t, inclusive) have mean accuracy >= 0.75
  2. interactions [t+1 .. t+3] (the next 3 items after t) have mean accuracy <= 0.50
  3. Those 3 future items exist (i.e. t+3 < len(sequence)); otherwise label is NaN.

Precedence note: Condition 1 requires recent GOOD performance; condition 2 requires UPCOMING
poor performance. These conditions probe opposite tails of accuracy, so when condition 1 fires
it uses [t-3..t], not a longer history, to remain locally sensitive.

Underload Target Alignment
----------------------------
Underload is detected as a separate binary series over the same sequence:
  1. 4 consecutive items [t-3..t] all have accuracy=1.
  2. Mean NRT of those 4 items < 0.3.
  Indices 0..2 (insufficient history) receive 0.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Ordered list of raw feature columns drawn directly from the simulator output.
# These columns form the feature matrix X in this order.
FEATURE_COLUMNS: List[str] = [
    "nrt",
    "accuracy",
    "window_error_rate",
    "retries",
    "help_requested",
    "confidence",
    "streak_correct",
    "streak_incorrect",
    "nrt_variance",
    "session_time",
]


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing from the input DataFrame."""
    missing = [c for c in FEATURE_COLUMNS + ["profile", "accuracy", "nrt"] if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")


def build_features(
    df: pd.DataFrame,
    window_size: int = 10,
) -> Tuple[np.ndarray, pd.Index]:
    """Build the feature matrix X for a single learner sequence.

    Features for interaction t use only data from interactions 0..t (no future leakage).
    All features in FEATURE_COLUMNS are already computed chronologically by the simulator
    without forward-looking information, so passing them directly is safe.

    The function additionally computes a rolling-window mean NRT column that is only
    allowed to reference the current and prior rows within the same sequence.

    Args:
        df: Single-profile interaction DataFrame (must not mix profiles).
        window_size: Rolling window width for mean NRT calculation (default 10).
                     At the start of the sequence, the available history is used.

    Returns:
        X: np.ndarray of shape (n_interactions, n_features)
        index: pd.Index aligned with the rows of df (for downstream alignment).
    """
    _validate_columns(df)

    # Work on a positional copy to avoid mutating caller data
    data = df.reset_index(drop=True)

    # Rolling mean NRT (causal: min_periods=1 uses whatever history exists)
    rolling_mean_nrt = (
        data["nrt"]
        .rolling(window=window_size, min_periods=1)
        .mean()
        .to_numpy()
    )

    # Assemble feature matrix in FEATURE_COLUMNS order
    base = data[FEATURE_COLUMNS].to_numpy(dtype=float)

    # Append rolling_mean_nrt as an additional feature column
    X = np.hstack([base, rolling_mean_nrt.reshape(-1, 1)])

    return X, data.index


def build_overload_target(
    df: pd.DataFrame,
    prior_window: int = 4,
    future_window: int = 3,
    prior_accuracy_threshold: float = 0.75,
    future_accuracy_threshold: float = 0.50,
) -> pd.Series:
    """Build binary overload target labels for a single-profile interaction sequence.

    Alignment (exact):
      - For interaction at row-position t (0-indexed):
        - prior_acc  = mean(accuracy[t - prior_window + 1 .. t])   (inclusive, positions t-3..t for default 4)
        - future_acc = mean(accuracy[t + 1 .. t + future_window])  (positions t+1..t+3 for default 3)
        - overload = 1 iff prior_acc >= prior_accuracy_threshold AND future_acc <= future_accuracy_threshold
      - Rows where t < prior_window - 1 have insufficient prior history → labeled 0.
      - Rows where t + future_window >= len(df) have insufficient future observations → labeled NaN.
        Callers should drop NaN rows before model training.

    Args:
        df: Single-profile interaction DataFrame.
        prior_window: Number of preceding items (including t) checked for high accuracy.
        future_window: Number of future items checked for accuracy collapse.
        prior_accuracy_threshold: Minimum mean accuracy in prior window to qualify.
        future_accuracy_threshold: Maximum mean accuracy in future window to qualify.

    Returns:
        pd.Series of float (0.0, 1.0, or NaN) indexed the same as df.
    """
    _validate_columns(df)
    n = len(df)
    acc = df["accuracy"].to_numpy(dtype=float)
    labels = np.full(n, np.nan)

    for t in range(n):
        # Check future availability first
        if t + future_window >= n:
            # Insufficient future observations → NaN (already set)
            continue

        # Check prior availability
        prior_start = t - prior_window + 1
        if prior_start < 0:
            # Insufficient prior history → label 0
            labels[t] = 0.0
            continue

        prior_acc = float(np.mean(acc[prior_start : t + 1]))
        future_acc = float(np.mean(acc[t + 1 : t + 1 + future_window]))

        if prior_acc >= prior_accuracy_threshold and future_acc <= future_accuracy_threshold:
            labels[t] = 1.0
        else:
            labels[t] = 0.0

    return pd.Series(labels, index=df.index, name="overload")


def build_underload_target(
    df: pd.DataFrame,
    window: int = 4,
    nrt_threshold: float = 0.3,
) -> pd.Series:
    """Build binary underload detection labels for a single-profile interaction sequence.

    An interaction at position t is labeled underload=1 when:
      1. All `window` items [t - window + 1 .. t] have accuracy = 1 (perfect accuracy).
      2. Mean NRT of those `window` items < nrt_threshold.
    Interactions where t < window - 1 (insufficient history) are labeled 0.

    Args:
        df: Single-profile interaction DataFrame.
        window: Number of consecutive items to check (default 4).
        nrt_threshold: NRT mean threshold below which underload is flagged (default 0.3).

    Returns:
        pd.Series of int (0 or 1) indexed the same as df.
    """
    _validate_columns(df)
    n = len(df)
    acc = df["accuracy"].to_numpy(dtype=float)
    nrt = df["nrt"].to_numpy(dtype=float)
    labels = np.zeros(n, dtype=int)

    for t in range(window - 1, n):
        window_acc = acc[t - window + 1 : t + 1]
        window_nrt = nrt[t - window + 1 : t + 1]
        if np.all(window_acc == 1.0) and float(np.mean(window_nrt)) < nrt_threshold:
            labels[t] = 1

    return pd.Series(labels, index=df.index, name="underload")


def prepare_dataset(
    df: pd.DataFrame,
    window_size: int = 10,
    drop_nan_targets: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Prepare the full supervised dataset from a (possibly multi-profile) interaction log.

    Handles multiple profiles by processing each learner independently, preventing
    window history from crossing profile boundaries.

    Returns X (features), y (overload labels), and a metadata DataFrame aligned to X/y
    rows. The metadata DataFrame carries columns: profile, interaction_id, overload,
    underload for downstream evaluation and debugging.

    Args:
        df: Interaction DataFrame, either single-profile or combined multi-profile.
        window_size: Rolling window size for feature computation.
        drop_nan_targets: If True, rows with NaN overload targets (end-of-sequence)
                          are dropped from X and y before returning.

    Returns:
        X: np.ndarray of shape (n_valid_rows, n_features)
        y: np.ndarray of shape (n_valid_rows,) with binary overload labels
        meta: pd.DataFrame with aligned metadata columns
    """
    _validate_columns(df)

    all_X: List[np.ndarray] = []
    all_y_overload: List[np.ndarray] = []
    all_y_underload: List[np.ndarray] = []
    all_meta: List[pd.DataFrame] = []

    profiles = df["profile"].unique() if "profile" in df.columns else ["__single__"]

    for profile in profiles:
        if "profile" in df.columns:
            sub = df[df["profile"] == profile].copy().reset_index(drop=True)
        else:
            sub = df.copy().reset_index(drop=True)

        X_sub, _ = build_features(sub, window_size=window_size)
        y_overload_sub = build_overload_target(sub)
        y_underload_sub = build_underload_target(sub)

        meta_sub = pd.DataFrame(
            {
                "profile": sub["profile"] if "profile" in sub.columns else profile,
                "interaction_id": sub["interaction_id"] if "interaction_id" in sub.columns else np.arange(len(sub)),
                "overload": y_overload_sub.to_numpy(),
                "underload": y_underload_sub.to_numpy(),
            }
        )

        all_X.append(X_sub)
        all_y_overload.append(y_overload_sub.to_numpy())
        all_y_underload.append(y_underload_sub.to_numpy())
        all_meta.append(meta_sub)

    X = np.vstack(all_X)
    y_overload = np.concatenate(all_y_overload)
    y_underload = np.concatenate(all_y_underload)
    meta = pd.concat(all_meta, ignore_index=True)
    meta["underload"] = y_underload

    if drop_nan_targets:
        valid_mask = ~np.isnan(y_overload)
        X = X[valid_mask]
        y_overload = y_overload[valid_mask]
        meta = meta[valid_mask].reset_index(drop=True)

    y = y_overload.astype(float)
    return X, y, meta


def extract_features(
    data: pd.DataFrame,
    window_size: int = 10,
    drop_nan_targets: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Alias kept for pipeline compatibility with run_all.py.

    Delegates to prepare_dataset. See prepare_dataset for full documentation.

    Args:
        data: Interaction DataFrame (single or multi-profile dict or DataFrame).
              If a dict of {profile: DataFrame} is passed, frames are concatenated first.
        window_size: Rolling window size.
        drop_nan_targets: Drop NaN-target rows before returning.

    Returns:
        X, y, meta — see prepare_dataset.
    """
    if isinstance(data, dict):
        data = pd.concat(list(data.values()), ignore_index=True)
    return prepare_dataset(data, window_size=window_size, drop_nan_targets=drop_nan_targets)
