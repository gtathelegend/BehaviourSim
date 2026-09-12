"""State and feature trajectory visualization utilities for BehaviorSim.

Provides publication-ready plotting for:
- Discrete state trajectories across sequence interactions
- Continuous/causal feature trajectories across sequences
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from behaviorsim.calibration.fitter import CalibrationData
from behaviorsim.visualization.plot_states import _extract_dataframe_and_states


def _select_sequences(
    df: pd.DataFrame,
    sequence_column: str,
    sequence_ids: Optional[Sequence[Any]] = None,
    max_sequences: int = 10,
) -> List[Any]:
    """Deterministically select a subset of sequence identifiers."""
    if max_sequences <= 0:
        raise ValueError(f"max_sequences must be a positive integer, got {max_sequences}.")

    if sequence_column not in df.columns:
        raise ValueError(
            f"Sequence column '{sequence_column}' not found in data columns: {list(df.columns)}."
        )

    all_sequences = df[sequence_column].dropna().unique().tolist()
    if not all_sequences:
        raise ValueError(f"No valid sequences found in column '{sequence_column}'.")

    if sequence_ids is not None:
        requested = list(sequence_ids)
        if len(requested) == 0:
            raise ValueError("Supplied sequence_ids cannot be empty.")
        missing = [s for s in requested if s not in all_sequences]
        if missing:
            raise ValueError(f"Requested sequence_ids not found in data: {missing}.")
        return requested[:max_sequences]

    # Deterministic selection: order by first occurrence in data, capped at max_sequences
    seen = set()
    ordered = []
    for s in df[sequence_column]:
        if pd.notna(s) and s not in seen:
            seen.add(s)
            ordered.append(s)
            if len(ordered) >= max_sequences:
                break
    return ordered


def plot_state_trajectory(
    data: Union[pd.DataFrame, CalibrationData],
    state_column: str = "state",
    sequence_column: str = "sequence_id",
    order_column: Optional[str] = None,
    max_sequences: int = 10,
    sequence_ids: Optional[Sequence[Any]] = None,
    state_order: Optional[Sequence[str]] = None,
    profile_column: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot discrete state trajectories across interactions for one or more sequences.

    Visualizes states as categorical levels without implying continuous numeric measurements.

    Parameters
    ----------
    data : pd.DataFrame or CalibrationData
        Behavioral traces containing sequences and state transitions.
    state_column : str, default "state"
        Column containing state labels.
    sequence_column : str, default "sequence_id"
        Column identifying individual sequences/agents.
    order_column : Optional[str], default None
        Optional column indicating interaction order/time step. If None, uses row order.
    max_sequences : int, default 10
        Maximum number of sequences to display to avoid clutter.
    sequence_ids : Optional[Sequence[Any]], default None
        Specific sequence identifiers to plot.
    state_order : Optional[Sequence[str]], default None
        Explicit vertical ordering of discrete states.
    profile_column : Optional[str], default None
        Optional column containing profile information for legend labeling.
    ax : Optional[Axes], default None
        Matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title : Optional[str], default None
        Plot title. If None, a descriptive title is generated.

    Returns
    -------
    Tuple[Figure, Axes]
        The Figure and Axes containing the trajectory plot.
    """
    df, s_col, states = _extract_dataframe_and_states(
        data, state_column=state_column, state_order=state_order
    )

    selected_seqs = _select_sequences(
        df,
        sequence_column=sequence_column,
        sequence_ids=sequence_ids,
        max_sequences=max_sequences,
    )

    state_to_idx = {s: i for i, s in enumerate(states)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(4, len(states) * 0.7)))
    else:
        fig = ax.figure

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(selected_seqs))))

    for i, seq_id in enumerate(selected_seqs):
        seq_df = df[df[sequence_column] == seq_id]
        if order_column is not None:
            if order_column not in seq_df.columns:
                raise ValueError(f"Order column '{order_column}' not found in data.")
            seq_df = seq_df.sort_values(by=order_column)
            x_vals = seq_df[order_column].to_numpy()
        else:
            x_vals = np.arange(len(seq_df))

        y_states = seq_df[s_col].tolist()
        y_indices = np.array([state_to_idx[s] for s in y_states], dtype=float)

        # Label
        if profile_column is not None and profile_column in seq_df.columns:
            prof = seq_df[profile_column].iloc[0]
            label = f"Seq {seq_id} ({prof})"
        else:
            label = f"Seq {seq_id}"

        # Slight vertical jitter if multiple sequences to prevent perfect overlap
        offset = (i - (len(selected_seqs) - 1) / 2) * 0.05 if len(selected_seqs) > 1 else 0.0
        ax.step(
            x_vals,
            y_indices + offset,
            where="mid",
            marker="o",
            markersize=4,
            alpha=0.8,
            color=colors[i % len(colors)],
            label=label,
        )

    ax.set_yticks(np.arange(len(states)))
    ax.set_yticklabels(states)
    ax.set_ylim(-0.5, len(states) - 0.5)
    ax.set_xlabel("Time Step (t)" if order_column is None else str(order_column))
    ax.set_ylabel("Discrete State")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"State Trajectories ({len(selected_seqs)} sequence{'s' if len(selected_seqs) > 1 else ''})")

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    if len(selected_seqs) > 1 or profile_column is not None:
        ax.legend(loc="upper right", frameon=True, fontsize=8)

    return fig, ax


def plot_feature_trajectory(
    data: Union[pd.DataFrame, CalibrationData],
    feature: str,
    sequence_column: str = "sequence_id",
    order_column: Optional[str] = None,
    max_sequences: int = 5,
    sequence_ids: Optional[Sequence[Any]] = None,
    profile_column: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot continuous or causal feature trajectories across sequence interactions.

    Parameters
    ----------
    data : pd.DataFrame or CalibrationData
        Behavioral traces containing feature observations.
    feature : str
        Name of the numeric feature column to plot.
    sequence_column : str, default "sequence_id"
        Column identifying individual sequences/agents.
    order_column : Optional[str], default None
        Optional column indicating interaction order/time step. If None, uses row order.
    max_sequences : int, default 5
        Maximum number of sequences to display.
    sequence_ids : Optional[Sequence[Any]], default None
        Specific sequence identifiers to plot.
    profile_column : Optional[str], default None
        Optional column containing profile information for legend labeling.
    ax : Optional[Axes], default None
        Matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title : Optional[str], default None
        Plot title. If None, a descriptive title is generated.

    Returns
    -------
    Tuple[Figure, Axes]
        The Figure and Axes containing the feature trajectory plot.
    """
    if isinstance(data, CalibrationData):
        df = data.data
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(
            f"Expected pandas DataFrame or CalibrationData, got {type(data).__name__}."
        )

    if df.empty:
        raise ValueError("Cannot plot feature trajectory on an empty dataset.")

    if feature not in df.columns:
        raise ValueError(f"Feature column '{feature}' not found in data columns: {list(df.columns)}.")

    if not pd.api.types.is_numeric_dtype(df[feature]):
        raise TypeError(
            f"Feature '{feature}' must be numeric, got dtype {df[feature].dtype}."
        )

    selected_seqs = _select_sequences(
        df,
        sequence_column=sequence_column,
        sequence_ids=sequence_ids,
        max_sequences=max_sequences,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(selected_seqs))))

    for i, seq_id in enumerate(selected_seqs):
        seq_df = df[df[sequence_column] == seq_id]
        if order_column is not None:
            if order_column not in seq_df.columns:
                raise ValueError(f"Order column '{order_column}' not found in data.")
            seq_df = seq_df.sort_values(by=order_column)
            x_vals = seq_df[order_column].to_numpy()
        else:
            x_vals = np.arange(len(seq_df))

        y_vals = seq_df[feature].to_numpy()

        if profile_column is not None and profile_column in seq_df.columns:
            prof = seq_df[profile_column].iloc[0]
            label = f"Seq {seq_id} ({prof})"
        else:
            label = f"Seq {seq_id}"

        ax.plot(
            x_vals,
            y_vals,
            marker="o",
            markersize=3,
            alpha=0.85,
            color=colors[i % len(colors)],
            label=label,
        )

    ax.set_xlabel("Time Step (t)" if order_column is None else str(order_column))
    ax.set_ylabel(feature)

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Feature Trajectory: {feature} ({len(selected_seqs)} sequences)")

    ax.grid(True, linestyle="--", alpha=0.4)
    if len(selected_seqs) > 1 or profile_column is not None:
        ax.legend(loc="best", frameon=True, fontsize=8)

    return fig, ax
