"""State sequence and transition visualization utilities for BehaviorSim.

Provides publication-ready plotting for:
- State occupancy proportions and raw counts
- Transition matrices (empirical and calibrated)
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from behaviorsim.calibration.fitter import CalibrationData, fit_transition_matrix


def _extract_dataframe_and_states(
    data: Union[pd.DataFrame, CalibrationData],
    state_column: str = "state",
    state_order: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, str, List[str]]:
    """Extract DataFrame, state column name, and deterministic state ordering.

    Does not modify input data.
    """
    if isinstance(data, CalibrationData):
        df = data.data
        col = state_column if state_column != "state" or "state" in df.columns else data.state_column
        declared_states = list(data.states)
    elif isinstance(data, pd.DataFrame):
        df = data
        col = state_column
        declared_states = []
    else:
        raise TypeError(
            f"Expected pandas DataFrame or CalibrationData, got {type(data).__name__}."
        )

    if df.empty:
        raise ValueError("Cannot plot state occupancy on an empty dataset.")

    if col not in df.columns:
        raise ValueError(
            f"State column '{col}' not found in data columns: {list(df.columns)}."
        )

    # Determine state ordering
    if state_order is not None:
        ordered_states = list(state_order)
        if len(ordered_states) == 0:
            raise ValueError("Supplied state_order cannot be empty.")
        if len(set(ordered_states)) != len(ordered_states):
            raise ValueError("Supplied state_order contains duplicate states.")
        # Check that observed states are within state_order
        observed = set(df[col].dropna().unique())
        unknown = observed - set(ordered_states)
        if unknown:
            raise ValueError(
                f"Data contains states not present in supplied state_order: {sorted(list(unknown))}."
            )
    elif declared_states:
        ordered_states = declared_states
    else:
        # Deterministic sorting
        observed_states = df[col].dropna().unique().tolist()
        ordered_states = sorted([str(s) for s in observed_states])

    return df, col, ordered_states


def plot_state_occupancy(
    data: Union[pd.DataFrame, CalibrationData],
    state_column: str = "state",
    profile_column: Optional[str] = None,
    state_order: Optional[Sequence[str]] = None,
    normalize: bool = True,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot state occupancy as normalized proportions or raw counts.

    Supports single-population data as well as multi-profile grouping.

    Parameters
    ----------
    data : pd.DataFrame or CalibrationData
        Input traces containing state observations.
    state_column : str, default "state"
        Name of the column containing state labels.
    profile_column : Optional[str], default None
        Optional column name indicating agent/behavioral profile for grouping.
    state_order : Optional[Sequence[str]], default None
        Explicit state ordering. If None, uses CalibrationData declared states or sorted labels.
    normalize : bool, default True
        If True, displays occupancy proportions (summing to 1.0 per profile/dataset).
        If False, displays raw observation counts.
    ax : Optional[Axes], default None
        Matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title : Optional[str], default None
        Plot title. If None, a descriptive title is generated.

    Returns
    -------
    Tuple[Figure, Axes]
        The Figure and Axes containing the occupancy plot.
    """
    df, col, states = _extract_dataframe_and_states(
        data, state_column=state_column, state_order=state_order
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    n_states = len(states)
    x = np.arange(n_states)

    if profile_column is not None:
        if profile_column not in df.columns:
            raise ValueError(
                f"Profile column '{profile_column}' not found in data columns: {list(df.columns)}."
            )
        # Deterministic profile ordering
        profiles = sorted(df[profile_column].dropna().unique().tolist(), key=lambda p: str(p))
        if not profiles:
            raise ValueError(f"No profiles found in profile column '{profile_column}'.")

        n_profiles = len(profiles)
        width = 0.8 / max(n_profiles, 1)

        for i, prof in enumerate(profiles):
            sub_df = df[df[profile_column] == prof]
            counts = sub_df[col].value_counts()
            values = [counts.get(s, 0) for s in states]
            if normalize:
                total = sum(values)
                values = [v / total if total > 0 else 0.0 for v in values]

            offset = (i - (n_profiles - 1) / 2) * width
            ax.bar(x + offset, values, width=width, label=str(prof), alpha=0.85)

        ax.legend(title=profile_column, frameon=True)
    else:
        counts = df[col].value_counts()
        values = [counts.get(s, 0) for s in states]
        if normalize:
            total = sum(values)
            values = [v / total if total > 0 else 0.0 for v in values]

        ax.bar(x, values, width=0.6, color="steelblue", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=0 if n_states <= 5 else 30, ha="right" if n_states > 5 else "center")
    ax.set_xlabel("State")
    ax.set_ylabel("Occupancy Proportion" if normalize else "Observation Count")

    if title is not None:
        ax.set_title(title)
    else:
        metric = "Proportion" if normalize else "Count"
        ax.set_title(f"State Occupancy ({metric})")

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig, ax


def plot_transition_matrix(
    matrix_or_data: Union[np.ndarray, pd.DataFrame, CalibrationData],
    state_order: Optional[Sequence[str]] = None,
    state_column: str = "state",
    sequence_column: str = "sequence_id",
    annot: bool = True,
    cmap: str = "Blues",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot a state transition matrix as an annotated heatmap.

    Parameters
    ----------
    matrix_or_data : np.ndarray, pd.DataFrame, or CalibrationData
        Precomputed transition matrix or raw interaction traces.
    state_order : Optional[Sequence[str]], default None
        Explicit state labels corresponding to matrix rows/columns.
    state_column : str, default "state"
        Name of state column (used if data is pd.DataFrame).
    sequence_column : str, default "sequence_id"
        Name of sequence column (used if data is pd.DataFrame).
    annot : bool, default True
        Whether to display cell values as text annotations.
    cmap : str, default "Blues"
        Colormap for heatmap.
    ax : Optional[Axes], default None
        Matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title : Optional[str], default None
        Plot title. If None, defaults to "State Transition Matrix".

    Returns
    -------
    Tuple[Figure, Axes]
        The Figure and Axes containing the transition heatmap.
    """
    if isinstance(matrix_or_data, np.ndarray):
        matrix = np.asarray(matrix_or_data, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Transition matrix must be square 2D array, got shape {matrix.shape}."
            )
        n = matrix.shape[0]
        if state_order is not None:
            states = list(state_order)
            if len(states) != n:
                raise ValueError(
                    f"Length of state_order ({len(states)}) does not match matrix size ({n})."
                )
        else:
            states = [f"S{i}" for i in range(n)]
    elif isinstance(matrix_or_data, CalibrationData):
        states = list(state_order) if state_order is not None else list(matrix_or_data.states)
        if set(states) != set(matrix_or_data.states):
            raise ValueError(
                f"Supplied state_order {states} does not match CalibrationData states {matrix_or_data.states}."
            )
        matrix = fit_transition_matrix(matrix_or_data, smoothing=0.0)
        # If user supplied a different order than CalibrationData.states, reorder
        if states != list(matrix_or_data.states):
            orig = list(matrix_or_data.states)
            indices = [orig.index(s) for s in states]
            matrix = matrix[np.ix_(indices, indices)]
    elif isinstance(matrix_or_data, pd.DataFrame):
        df = matrix_or_data
        if df.empty:
            raise ValueError("Cannot compute transition matrix on empty DataFrame.")
        if state_column not in df.columns:
            raise ValueError(f"State column '{state_column}' not found in DataFrame.")
        if sequence_column not in df.columns:
            raise ValueError(f"Sequence column '{sequence_column}' not found in DataFrame.")

        if state_order is not None:
            states = list(state_order)
        else:
            states = sorted([str(s) for s in df[state_column].dropna().unique()])

        if len(states) == 0:
            raise ValueError("No valid states found in data.")

        cal_data = CalibrationData(
            data=df,
            states=states,
            state_column=state_column,
            sequence_column=sequence_column,
        )
        matrix = fit_transition_matrix(cal_data, smoothing=0.0)
    else:
        raise TypeError(
            f"Expected np.ndarray, pd.DataFrame, or CalibrationData, got {type(matrix_or_data).__name__}."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    n = matrix.shape[0]
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=max(1.0, float(np.max(matrix)) if matrix.size > 0 else 1.0))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(states, rotation=0 if n <= 5 else 30, ha="right" if n > 5 else "center")
    ax.set_yticklabels(states)
    ax.set_xlabel("Next State (t + 1)")
    ax.set_ylabel("Current State (t)")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("State Transition Matrix")

    if annot:
        # Determine contrast threshold
        max_val = np.max(matrix) if matrix.size > 0 else 1.0
        thresh = max_val / 2.0
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    return fig, ax
