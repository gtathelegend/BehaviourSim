"""Calibration diagnostic visualization utilities for BehaviorSim.

Consumes ValidationReport instances or empirical/synthetic dataset pairs to visualize:
- State occupancy comparisons (empirical vs synthetic, aligned states, TVD)
- Transition matrix comparisons (empirical, synthetic, and difference heatmaps)
- Continuous/numeric feature validation metrics (means, Wasserstein distance, KS)
- Categorical feature validation metrics (frequencies, TVD, JS distance)
- Comprehensive validation summary dashboard
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from behaviorsim.calibration.fitter import CalibrationData
from behaviorsim.calibration.validator import (
    ValidationReport,
    validate_calibration,
)
from behaviorsim.visualization.plot_distributions import _extract_df


def plot_occupancy_comparison(
    report_or_empirical: Union[ValidationReport, pd.DataFrame, CalibrationData],
    synthetic: Optional[Union[pd.DataFrame, CalibrationData]] = None,
    states: Optional[Sequence[str]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot state occupancy comparison between empirical and synthetic data.

    Parameters
    ----------
    report_or_empirical : ValidationReport, pd.DataFrame, or CalibrationData
        ValidationReport instance, or empirical dataset if synthetic is also provided.
    synthetic : Optional[Union[pd.DataFrame, CalibrationData]], default None
        Synthetic dataset (required if report_or_empirical is not a ValidationReport).
    states : Optional[Sequence[str]], default None
        Deterministic state ordering. If None, uses report order or sorted union of states.
    ax : Optional[Axes], default None
        Matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title : Optional[str], default None
        Plot title. If None, generates title including TVD.

    Returns
    -------
    Tuple[Figure, Axes]
        The Figure and Axes containing the comparison bar plot.
    """
    if isinstance(report_or_empirical, ValidationReport):
        occ = report_or_empirical.state_occupancy
        emp_occ = occ.empirical_occupancy
        syn_occ = occ.synthetic_occupancy
        tvd = occ.total_variation_distance
    else:
        if synthetic is None:
            raise ValueError("Must provide synthetic data when report_or_empirical is not a ValidationReport.")
        emp_df = _extract_df(report_or_empirical)
        syn_df = _extract_df(synthetic)

        # Infer state column
        s_col = "state"
        if isinstance(report_or_empirical, CalibrationData):
            s_col = report_or_empirical.state_column
            declared_states = list(report_or_empirical.states)
        else:
            declared_states = []

        if s_col not in emp_df.columns or s_col not in syn_df.columns:
            raise ValueError(f"State column '{s_col}' must be present in both datasets.")

        emp_counts = emp_df[s_col].value_counts()
        syn_counts = syn_df[s_col].value_counts()
        e_total = len(emp_df[s_col].dropna())
        s_total = len(syn_df[s_col].dropna())

        if states is not None:
            state_list = list(states)
        elif declared_states:
            state_list = declared_states
        else:
            state_list = sorted(list(set(emp_counts.index).union(set(syn_counts.index))), key=str)

        emp_occ = {s: emp_counts.get(s, 0) / e_total if e_total > 0 else 0.0 for s in state_list}
        syn_occ = {s: syn_counts.get(s, 0) / s_total if s_total > 0 else 0.0 for s in state_list}
        tvd = 0.5 * sum(abs(emp_occ[s] - syn_occ[s]) for s in state_list)

    if states is not None:
        state_list = list(states)
        if len(state_list) == 0:
            raise ValueError("Supplied states cannot be empty.")
    else:
        state_list = list(emp_occ.keys())

    emp_vals = [emp_occ.get(s, 0.0) for s in state_list]
    syn_vals = [syn_occ.get(s, 0.0) for s in state_list]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    x = np.arange(len(state_list))
    width = 0.35

    ax.bar(x - width / 2, emp_vals, width=width, label="Empirical", color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, syn_vals, width=width, label="Synthetic", color="#ff7f0e", alpha=0.85)

    ax.set_xticks(x)
    n_states = len(state_list)
    ax.set_xticklabels(state_list, rotation=0 if n_states <= 5 else 30, ha="right" if n_states > 5 else "center")
    ax.set_xlabel("State")
    ax.set_ylabel("Occupancy Proportion")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"State Occupancy Comparison (TVD = {tvd:.4f})")

    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig, ax


def plot_transition_comparison(
    report_or_matrix_pair: Union[ValidationReport, Tuple[np.ndarray, np.ndarray]],
    states: Optional[Sequence[str]] = None,
    axes: Optional[Sequence[Axes]] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, np.ndarray]:
    """Plot heatmaps comparing empirical, synthetic, and difference transition matrices.

    Parameters
    ----------
    report_or_matrix_pair : ValidationReport or Tuple[np.ndarray, np.ndarray]
        ValidationReport containing transition results, or tuple of (empirical_matrix, synthetic_matrix).
    states : Optional[Sequence[str]], default None
        State names for row and column tick labels.
    axes : Optional[Sequence[Axes]], default None
        3 Axes objects to draw on. If None, a new Figure with 3 subplots is created.
    title : Optional[str], default None
        Super-title for the figure.

    Returns
    -------
    Tuple[Figure, np.ndarray]
        Figure and array of 3 Axes: [ax_emp, ax_syn, ax_diff].
    """
    if isinstance(report_or_matrix_pair, ValidationReport):
        if report_or_matrix_pair.transition is None:
            raise ValueError("ValidationReport does not contain transition validation results.")
        emp_mat = report_or_matrix_pair.transition.empirical_matrix
        syn_mat = report_or_matrix_pair.transition.synthetic_matrix
        mae = report_or_matrix_pair.transition.mean_absolute_error
        frob = report_or_matrix_pair.transition.frobenius_distance
    elif isinstance(report_or_matrix_pair, tuple) and len(report_or_matrix_pair) == 2:
        emp_mat = np.asarray(report_or_matrix_pair[0], dtype=float)
        syn_mat = np.asarray(report_or_matrix_pair[1], dtype=float)
        if emp_mat.shape != syn_mat.shape or emp_mat.ndim != 2 or emp_mat.shape[0] != emp_mat.shape[1]:
            raise ValueError("Both matrices must be square 2D arrays with matching dimensions.")
        mae = float(np.mean(np.abs(syn_mat - emp_mat)))
        frob = float(np.linalg.norm(syn_mat - emp_mat, "fro"))
    else:
        raise TypeError(
            f"Expected ValidationReport or (np.ndarray, np.ndarray), got {type(report_or_matrix_pair).__name__}."
        )

    diff_mat = syn_mat - emp_mat
    n = emp_mat.shape[0]

    if states is not None:
        state_list = list(states)
        if len(state_list) != n:
            raise ValueError(f"Length of states ({len(state_list)}) must match matrix size ({n}).")
    else:
        state_list = [f"S{i}" for i in range(n)]

    if axes is None:
        fig, ax_arr = plt.subplots(1, 3, figsize=(15, 4.5))
    else:
        ax_arr = np.asarray(axes)
        if ax_arr.size != 3:
            raise ValueError(f"Expected 3 Axes, got {ax_arr.size}.")
        fig = ax_arr.flat[0].figure

    ax1, ax2, ax3 = ax_arr.flat[0], ax_arr.flat[1], ax_arr.flat[2]

    # Panel 1: Empirical
    im1 = ax1.imshow(emp_mat, cmap="Blues", vmin=0.0, vmax=1.0)
    ax1.set_title("Empirical Transitions")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Panel 2: Synthetic
    im2 = ax2.imshow(syn_mat, cmap="Blues", vmin=0.0, vmax=1.0)
    ax2.set_title("Synthetic Transitions")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Panel 3: Difference (Synthetic - Empirical)
    max_abs = max(0.01, float(np.max(np.abs(diff_mat))))
    im3 = ax3.imshow(diff_mat, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    ax3.set_title(f"Difference (Syn - Emp)\nMAE = {mae:.4f}, Frob = {frob:.4f}")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    for ax in (ax1, ax2, ax3):
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(state_list, rotation=0 if n <= 5 else 30, ha="right" if n > 5 else "center")
        ax.set_yticklabels(state_list)
        ax.set_xlabel("Next State")
        ax.set_ylabel("Current State")

    if title is not None:
        fig.suptitle(title, y=1.02)

    fig.tight_layout()
    return fig, ax_arr


def plot_numeric_comparison(
    report: ValidationReport,
    features: Optional[Sequence[str]] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot numeric feature comparison metrics from a ValidationReport.

    Displays empirical vs synthetic means with standard deviation error bars,
    along with Wasserstein distance annotations.

    Parameters
    ----------
    report : ValidationReport
        ValidationReport instance containing numeric feature comparisons.
    features : Optional[Sequence[str]], default None
        Specific numeric features to plot. If None, plots all available in report.
    ax : Optional[Axes], default None
        Matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title : Optional[str], default None
        Plot title.

    Returns
    -------
    Tuple[Figure, Axes]
        Figure and Axes containing the numeric feature comparison plot.
    """
    if not isinstance(report, ValidationReport):
        raise TypeError(f"Expected ValidationReport, got {type(report).__name__}.")

    if not report.numeric_features:
        raise ValueError("ValidationReport does not contain any numeric feature comparisons.")

    if features is not None:
        feat_list = list(features)
        missing = [f for f in feat_list if f not in report.numeric_features]
        if missing:
            raise ValueError(f"Features not found in report: {missing}.")
    else:
        feat_list = sorted(list(report.numeric_features.keys()))

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(feat_list) * 2), 5))
    else:
        fig = ax.figure

    x = np.arange(len(feat_list))
    width = 0.35

    emp_means = [report.numeric_features[f].empirical_mean for f in feat_list]
    syn_means = [report.numeric_features[f].synthetic_mean for f in feat_list]
    emp_stds = [report.numeric_features[f].empirical_std for f in feat_list]
    syn_stds = [report.numeric_features[f].synthetic_std for f in feat_list]

    ax.bar(
        x - width / 2,
        emp_means,
        yerr=emp_stds,
        width=width,
        capsize=4,
        label="Empirical (Mean ± Std)",
        color="#1f77b4",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        syn_means,
        yerr=syn_stds,
        width=width,
        capsize=4,
        label="Synthetic (Mean ± Std)",
        color="#ff7f0e",
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(feat_list, rotation=0 if len(feat_list) <= 4 else 25, ha="right" if len(feat_list) > 4 else "center")
    ax.set_ylabel("Value")
    ax.set_xlabel("Numeric Feature")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Numeric Features: Empirical vs Synthetic (Mean ± Std)")

    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig, ax


def plot_categorical_comparison(
    report: ValidationReport,
    feature: Optional[str] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot categorical feature frequency comparison from a ValidationReport.

    Displays aligned category relative frequencies with TVD and JS distance.

    Parameters
    ----------
    report : ValidationReport
        ValidationReport instance containing categorical feature comparisons.
    feature : Optional[str], default None
        Specific categorical feature to plot. If None and only one feature exists, picks that feature.
    ax : Optional[Axes], default None
        Matplotlib Axes to draw on. If None, a new Figure and Axes are created.
    title : Optional[str], default None
        Plot title.

    Returns
    -------
    Tuple[Figure, Axes]
        Figure and Axes containing the categorical comparison plot.
    """
    if not isinstance(report, ValidationReport):
        raise TypeError(f"Expected ValidationReport, got {type(report).__name__}.")

    if not report.categorical_features:
        raise ValueError("ValidationReport does not contain any categorical feature comparisons.")

    if feature is None:
        if len(report.categorical_features) == 1:
            feature = next(iter(report.categorical_features.keys()))
        else:
            raise ValueError(
                f"Multiple categorical features exist in report {list(report.categorical_features.keys())}; specify feature."
            )

    if feature not in report.categorical_features:
        raise ValueError(
            f"Feature '{feature}' not found in report categorical features: {list(report.categorical_features.keys())}."
        )

    cat_comp = report.categorical_features[feature]
    emp_freq = cat_comp.empirical_frequencies
    syn_freq = cat_comp.synthetic_frequencies
    tvd = cat_comp.total_variation_distance
    jsd = cat_comp.jensen_shannon_distance

    # Deterministic category ordering
    categories = sorted(list(set(emp_freq.keys()).union(set(syn_freq.keys()))), key=str)

    emp_vals = [emp_freq.get(c, 0.0) for c in categories]
    syn_vals = [syn_freq.get(c, 0.0) for c in categories]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width / 2, emp_vals, width=width, label="Empirical", color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, syn_vals, width=width, label="Synthetic", color="#ff7f0e", alpha=0.85)

    ax.set_xticks(x)
    n_cats = len(categories)
    ax.set_xticklabels(categories, rotation=0 if n_cats <= 6 else 30, ha="right" if n_cats > 6 else "center")
    ax.set_xlabel(f"Category ({feature})")
    ax.set_ylabel("Relative Frequency")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Categorical Comparison: {feature} (TVD = {tvd:.4f}, JSD = {jsd:.4f})")

    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    return fig, ax


def plot_calibration_summary(
    report: ValidationReport,
    fig: Optional[Figure] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, np.ndarray]:
    """Plot a comprehensive calibration diagnostic dashboard based on a ValidationReport.

    Displays:
    1. State occupancy comparison
    2. Transition difference heatmap (or transition matrix)
    3. Numeric feature divergence summary
    4. Key validation metrics and threshold results (if explicitly provided)

    IMPORTANT: Does NOT introduce arbitrary pass/fail thresholds. If no thresholds
    were supplied, displays raw metrics neutrally without green/red coloring.

    Parameters
    ----------
    report : ValidationReport
        The validation report to visualize.
    fig : Optional[Figure], default None
        Matplotlib Figure to draw on. If None, a new 2x2 Figure is created.
    title : Optional[str], default None
        Overall dashboard title.

    Returns
    -------
    Tuple[Figure, np.ndarray]
        Figure and a 2x2 array of Axes.
    """
    if not isinstance(report, ValidationReport):
        raise TypeError(f"Expected ValidationReport, got {type(report).__name__}.")

    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        axes = np.asarray(fig.axes).reshape((2, 2)) if len(fig.axes) == 4 else plt.subplots(2, 2, num=fig.number)[1]

    ax_occ = axes[0, 0]
    ax_trans = axes[0, 1]
    ax_num = axes[1, 0]
    ax_summary = axes[1, 1]

    # Panel 1: State Occupancy
    plot_occupancy_comparison(report, ax=ax_occ, title="1. State Occupancy Comparison")

    # Panel 2: Transition Difference
    if report.transition is not None:
        diff_mat = report.transition.synthetic_matrix - report.transition.empirical_matrix
        max_abs = max(0.01, float(np.max(np.abs(diff_mat))))
        im = ax_trans.imshow(diff_mat, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
        fig.colorbar(im, ax=ax_trans, fraction=0.046, pad=0.04)
        n = diff_mat.shape[0]
        states = list(report.state_occupancy.empirical_occupancy.keys())
        if len(states) != n:
            states = [f"S{i}" for i in range(n)]
        ax_trans.set_xticks(np.arange(n))
        ax_trans.set_yticks(np.arange(n))
        ax_trans.set_xticklabels(states, rotation=0 if n <= 5 else 30, ha="right" if n > 5 else "center")
        ax_trans.set_yticklabels(states)
        ax_trans.set_xlabel("Next State")
        ax_trans.set_ylabel("Current State")
        ax_trans.set_title(
            f"2. Transition Difference (Syn - Emp)\nMAE = {report.transition.mean_absolute_error:.4f}"
        )
    else:
        ax_trans.text(0.5, 0.5, "No transition data available", ha="center", va="center", transform=ax_trans.transAxes)
        ax_trans.set_title("2. Transition Difference")

    # Panel 3: Numeric Divergence or Features
    if report.numeric_features:
        feat_names = sorted(list(report.numeric_features.keys()))
        w_dists = [report.numeric_features[f].wasserstein_distance for f in feat_names]
        x = np.arange(len(feat_names))
        ax_num.bar(x, w_dists, color="teal", alpha=0.85, width=0.5)
        ax_num.set_xticks(x)
        ax_num.set_xticklabels(feat_names, rotation=0 if len(feat_names) <= 4 else 25, ha="right" if len(feat_names) > 4 else "center")
        ax_num.set_ylabel("Wasserstein Distance")
        ax_num.set_title("3. Numeric Feature Divergence")
        ax_num.grid(axis="y", linestyle="--", alpha=0.3)
    else:
        ax_num.text(0.5, 0.5, "No numeric features evaluated", ha="center", va="center", transform=ax_num.transAxes)
        ax_num.set_title("3. Numeric Feature Divergence")

    # Panel 4: Metrics & Threshold Status (Strictly Neutral unless threshold was supplied)
    ax_summary.axis("off")
    summary_lines = [
        "Calibration Diagnostics Summary",
        "---------------------------------",
        f"Structural Integrity: {'Valid' if report.structural.is_valid else 'Invalid'}",
        f"State Occupancy TVD:  {report.state_occupancy.total_variation_distance:.4f}",
    ]
    if report.transition is not None:
        summary_lines.append(f"Transition MAE:       {report.transition.mean_absolute_error:.4f}")
        summary_lines.append(f"Frobenius Distance:   {report.transition.frobenius_distance:.4f}")

    if report.threshold_results:
        summary_lines.append("\nUser Configured Thresholds:")
        summary_lines.append("----------------------------")
        for k, passed in sorted(report.threshold_results.items()):
            status = "PASS" if passed else "FAIL"
            summary_lines.append(f"  [{status}] {k}")
        overall = "PASSED" if report.is_valid else "FAILED"
        summary_lines.append(f"\nOverall Calibration Status: {overall}")
    else:
        summary_lines.append("\n(No validation thresholds configured)")
        summary_lines.append("Metrics presented as empirical diagnostic indicators.")

    summary_text = "\n".join(summary_lines)
    ax_summary.text(
        0.05,
        0.95,
        summary_text,
        transform=ax_summary.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da"),
    )
    ax_summary.set_title("4. Summary & Verification Status")

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.99)
    else:
        fig.suptitle("BehaviorSim Calibration Validation Dashboard", fontsize=14, y=0.99)

    fig.tight_layout()
    return fig, axes
