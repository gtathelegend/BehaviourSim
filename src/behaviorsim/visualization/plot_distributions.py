"""Feature and distribution visualization utilities for BehaviorSim.

Provides publication-ready plotting for:
- Continuous/numeric feature distributions (single dataset, grouped by state/profile)
- Empirical vs synthetic numeric feature comparisons (aligned bins, overlapping density)
- Categorical feature comparisons (aligned categories, side-by-side frequencies)
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from behaviorsim.calibration.fitter import CalibrationData


def _extract_df(data: Union[pd.DataFrame, CalibrationData]) -> pd.DataFrame:
    """Extract pandas DataFrame without mutating inputs."""
    if isinstance(data, CalibrationData):
        return data.data
    elif isinstance(data, pd.DataFrame):
        return data
    else:
        raise TypeError(
            f"Expected pandas DataFrame or CalibrationData, got {type(data).__name__}."
        )


def plot_feature_distribution(
    data: Union[pd.DataFrame, CalibrationData],
    feature: str,
    bins: int = 30,
    state_column: Optional[str] = None,
    profile_column: Optional[str] = None,
    density: bool = True,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot numeric feature distribution as a histogram or grouped histogram.

    Parameters
    ----------
    data : pd.DataFrame or CalibrationData
        Dataset containing feature observations.
    feature : str
        Name of the numeric feature column.
    bins : int, default 30
        Number of histogram bins.
    state_column : Optional[str], default None
        Optional column to group distributions by state.
    profile_column : Optional[str], default None
        Optional column to group distributions by profile.
    density : bool, default True
        Whether to normalize the histogram to form a probability density.
    ax : Optional[Axes], default None
        Matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title : Optional[str], default None
        Plot title. If None, a descriptive title is generated.

    Returns
    -------
    Tuple[Figure, Axes]
        The Figure and Axes containing the distribution plot.
    """
    df = _extract_df(data)

    if df.empty:
        raise ValueError("Cannot plot distribution on an empty dataset.")

    if feature not in df.columns:
        raise ValueError(f"Feature column '{feature}' not found in data columns: {list(df.columns)}.")

    series = df[feature].dropna()
    if series.empty:
        raise ValueError(f"Feature column '{feature}' contains only null values.")

    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"Feature '{feature}' must be numeric, got dtype {series.dtype}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    group_col = state_column or profile_column
    if group_col is not None:
        if group_col not in df.columns:
            raise ValueError(f"Grouping column '{group_col}' not found in data.")

        # Deterministic ordering of groups
        groups = sorted(df[group_col].dropna().unique().tolist(), key=lambda g: str(g))
        if not groups:
            raise ValueError(f"No valid groups found in column '{group_col}'.")

        # Use common bins across all groups
        all_vals = series.to_numpy()
        min_v, max_v = float(np.min(all_vals)), float(np.max(all_vals))
        if min_v == max_v:
            bin_edges = np.linspace(min_v - 0.5, max_v + 0.5, bins + 1)
        else:
            bin_edges = np.linspace(min_v, max_v, bins + 1)

        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(groups))))
        for i, grp in enumerate(groups):
            grp_vals = df[df[group_col] == grp][feature].dropna().to_numpy()
            if len(grp_vals) > 0:
                ax.hist(
                    grp_vals,
                    bins=bin_edges,
                    density=density,
                    alpha=0.45,
                    color=colors[i % len(colors)],
                    label=str(grp),
                    edgecolor="black",
                    linewidth=0.5,
                )

        ax.legend(title=group_col, frameon=True)
    else:
        vals = series.to_numpy()
        ax.hist(
            vals,
            bins=bins,
            density=density,
            alpha=0.75,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel(feature)
    ax.set_ylabel("Density" if density else "Count")

    if title is not None:
        ax.set_title(title)
    else:
        metric = "Density" if density else "Count"
        ax.set_title(f"Distribution of {feature} ({metric})")

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig, ax


def plot_feature_comparison(
    empirical: Union[pd.DataFrame, CalibrationData],
    synthetic: Union[pd.DataFrame, CalibrationData],
    feature: str,
    bins: int = 30,
    density: bool = True,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot an empirical vs synthetic comparison for a continuous/numeric feature.

    Uses strictly identical binning and identical scale to prevent misleading comparisons.

    Parameters
    ----------
    empirical : pd.DataFrame or CalibrationData
        Empirical observation traces.
    synthetic : pd.DataFrame or CalibrationData
        Synthetic/simulated traces.
    feature : str
        Name of numeric feature to compare.
    bins : int, default 30
        Number of bins for the shared histogram.
    density : bool, default True
        Whether to normalize histograms to probability densities.
    ax : Optional[Axes], default None
        Matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title : Optional[str], default None
        Plot title. If None, a descriptive title is generated.

    Returns
    -------
    Tuple[Figure, Axes]
        The Figure and Axes containing the comparison plot.
    """
    emp_df = _extract_df(empirical)
    syn_df = _extract_df(synthetic)

    if emp_df.empty or syn_df.empty:
        raise ValueError("Empirical and synthetic datasets must both be non-empty.")

    if feature not in emp_df.columns:
        raise ValueError(f"Feature '{feature}' not found in empirical columns: {list(emp_df.columns)}.")
    if feature not in syn_df.columns:
        raise ValueError(f"Feature '{feature}' not found in synthetic columns: {list(syn_df.columns)}.")

    emp_vals = emp_df[feature].dropna().to_numpy()
    syn_vals = syn_df[feature].dropna().to_numpy()

    if len(emp_vals) == 0:
        raise ValueError(f"Empirical feature '{feature}' has no valid non-null values.")
    if len(syn_vals) == 0:
        raise ValueError(f"Synthetic feature '{feature}' has no valid non-null values.")

    if not pd.api.types.is_numeric_dtype(emp_df[feature]):
        raise TypeError(f"Empirical feature '{feature}' must be numeric.")
    if not pd.api.types.is_numeric_dtype(syn_df[feature]):
        raise TypeError(f"Synthetic feature '{feature}' must be numeric.")

    # Calculate unified range and bins across both empirical and synthetic
    min_val = min(float(np.min(emp_vals)), float(np.min(syn_vals)))
    max_val = max(float(np.max(emp_vals)), float(np.max(syn_vals)))

    if min_val == max_val:
        bin_edges = np.linspace(min_val - 0.5, max_val + 0.5, bins + 1)
    else:
        bin_edges = np.linspace(min_val, max_val, bins + 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.hist(
        emp_vals,
        bins=bin_edges,
        density=density,
        alpha=0.5,
        color="#1f77b4",
        label=f"Empirical (n={len(emp_vals)})",
        edgecolor="#1f77b4",
        linewidth=0.8,
    )
    ax.hist(
        syn_vals,
        bins=bin_edges,
        density=density,
        alpha=0.5,
        color="#ff7f0e",
        label=f"Synthetic (n={len(syn_vals)})",
        edgecolor="#ff7f0e",
        linewidth=0.8,
    )

    ax.set_xlabel(feature)
    ax.set_ylabel("Density" if density else "Count")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Empirical vs Synthetic: {feature}")

    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig, ax


def plot_categorical_distribution(
    empirical: Union[pd.DataFrame, CalibrationData],
    synthetic: Union[pd.DataFrame, CalibrationData],
    feature: str,
    category_order: Optional[Sequence[Any]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot an empirical vs synthetic comparison for a discrete/categorical feature.

    Guarantees aligned category support, deterministic category ordering, and
    displays categories appearing only on one side with zero on the other.

    Parameters
    ----------
    empirical : pd.DataFrame or CalibrationData
        Empirical observation traces.
    synthetic : pd.DataFrame or CalibrationData
        Synthetic/simulated traces.
    feature : str
        Name of categorical feature column to compare.
    category_order : Optional[Sequence[Any]], default None
        Explicit category ordering. If None, sorts the union of all observed categories.
    ax : Optional[Axes], default None
        Matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title : Optional[str], default None
        Plot title. If None, a descriptive title is generated.

    Returns
    -------
    Tuple[Figure, Axes]
        The Figure and Axes containing the categorical comparison plot.
    """
    emp_df = _extract_df(empirical)
    syn_df = _extract_df(synthetic)

    if emp_df.empty or syn_df.empty:
        raise ValueError("Empirical and synthetic datasets must both be non-empty.")

    if feature not in emp_df.columns:
        raise ValueError(f"Feature '{feature}' not found in empirical columns: {list(emp_df.columns)}.")
    if feature not in syn_df.columns:
        raise ValueError(f"Feature '{feature}' not found in synthetic columns: {list(syn_df.columns)}.")

    emp_series = emp_df[feature].dropna()
    syn_series = syn_df[feature].dropna()

    if emp_series.empty or syn_series.empty:
        raise ValueError(f"Feature '{feature}' contains only null values in one or both datasets.")

    emp_counts = emp_series.value_counts()
    syn_counts = syn_series.value_counts()

    emp_total = len(emp_series)
    syn_total = len(syn_series)

    # Determine deterministic category order
    if category_order is not None:
        categories = list(category_order)
        if len(categories) == 0:
            raise ValueError("Supplied category_order cannot be empty.")
        if len(set(categories)) != len(categories):
            raise ValueError("Supplied category_order contains duplicate categories.")
        # Verify no unhandled categories
        observed = set(emp_counts.index).union(set(syn_counts.index))
        missing = observed - set(categories)
        if missing:
            raise ValueError(
                f"Data contains categories not present in supplied category_order: {sorted(list(missing), key=str)}."
            )
    else:
        # Sorted union of observed categories
        union_cats = set(emp_counts.index).union(set(syn_counts.index))
        categories = sorted(list(union_cats), key=lambda c: str(c))

    emp_props = [emp_counts.get(c, 0) / emp_total for c in categories]
    syn_props = [syn_counts.get(c, 0) / syn_total for c in categories]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width / 2, emp_props, width=width, label=f"Empirical (n={emp_total})", color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, syn_props, width=width, label=f"Synthetic (n={syn_total})", color="#ff7f0e", alpha=0.85)

    cat_labels = [str(c) for c in categories]
    n_cats = len(categories)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=0 if n_cats <= 6 else 30, ha="right" if n_cats > 6 else "center")
    ax.set_xlabel(feature)
    ax.set_ylabel("Relative Frequency")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Categorical Comparison: {feature}")

    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig, ax
