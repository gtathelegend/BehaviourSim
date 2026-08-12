"""Publication-quality visualization module for CLSI-Adapt Simulator.

Generates five primary publication figures for the JEDM paper:
  Figure 1: System Architecture Diagram (PDF)
  Figure 2: ROC Curves across Profiles (PDF)
  Figure 3: SHAP Beeswarm Summary Plot (PDF)
  Figure 4: Causal Learning Curve (PDF)
  Figure 5: Model Comparison Bar Charts (PDF)

All figures consume actual pipeline data structures and predictions (no hard-coded values or mock results).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import Config, default_config
from src.feature_engineering import build_features, build_overload_target
from src.models.clsi_adapt import (
    CLSIAdaptModel,
    _prepare_profile_data,
    compute_shap,
    plot_shap_summary as plot_shap_summary_impl,
)


# ---------------------------------------------------------------------------
# Plotting Style Configuration
# ---------------------------------------------------------------------------

PRETTY_NAMES = {
    "fast_accurate": "Fast-Accurate",
    "fast_inaccurate": "Fast-Inaccurate",
    "slow_accurate": "Slow-Accurate",
    "slow_inaccurate": "Slow-Inaccurate",
    "average": "Average",
    "clsi_adapt": "CLSI-Adapt",
    "rule_based_clsi": "Rule-Based CLSI",
    "bkt": "BKT",
}

MODEL_COLORS = {
    "clsi_adapt": "#2b5c8f",      # Dark Blue
    "rule_based_clsi": "#008080",  # Teal
    "bkt": "#d95f02",              # Coral / Orange
}

MODEL_STYLES = {
    "clsi_adapt": "-",
    "rule_based_clsi": "--",
    "bkt": ":",
}


def _apply_publication_style() -> None:
    """Set global matplotlib style parameters suitable for academic paper publication."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })


# ---------------------------------------------------------------------------
# Figure 1: System Architecture
# ---------------------------------------------------------------------------

def plot_architecture(output_path: Path) -> None:
    """Generate Figure 1: System Architecture Diagram.

    Illustrates the pipeline flow:
      Simulation -> Interaction Logs -> Feature Engineering -> Models -> Evaluation -> Outputs
    Explicitly labels BKT as a struggle proxy measure.

    Parameters
    ----------
    output_path : Path
        Target PDF file path.
    """
    _apply_publication_style()
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    ax.axis("off")

    # Define box dimensions and positions
    # (x, y, width, height, text, bg_color)
    boxes = [
        (0.35, 0.90, 0.30, 0.07, "Synthetic Learner Simulation\n(HMM + Log-Normal RT)", "#e6f2ff"),
        (0.35, 0.78, 0.30, 0.07, "Interaction Logs\n(Accuracy, NRT, Retries, Help)", "#e6f2ff"),
        (0.35, 0.66, 0.30, 0.07, "Temporal Feature Engineering\n(Causal Windows, Overload Target)", "#d9ead3"),
        
        # Models
        (0.08, 0.48, 0.24, 0.09, "CLSI-Adapt\n(Per-Profile XGBoost)\n[Proposed]", "#d0e0e3"),
        (0.38, 0.48, 0.24, 0.09, "Rule-Based CLSI\n(Composite Formula)\n[Baseline]", "#fce5cd"),
        (0.68, 0.48, 0.24, 0.09, "BKT Model\n(Bayesian Knowledge)\n[Struggle Proxy]", "#fff2cc"),
        
        # Evaluation & Outputs
        (0.30, 0.30, 0.40, 0.08, "Temporal Evaluation Framework\n(5-Fold TimeSeriesSplit, Event Recall, Recovery Time)", "#ea9999"),
        (0.12, 0.12, 0.34, 0.08, "Publication Tables\n(CSV + LaTeX format)", "#f3f3f3"),
        (0.54, 0.12, 0.34, 0.08, "Publication Figures\n(ROC, SHAP, Learning Curves)", "#f3f3f3"),
    ]

    for x, y, w, h, text, bg in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02",
            ec="#333333",
            fc=bg,
            lw=1.2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="#222222", multialignment="center"
        )

    # Draw connecting arrows
    arrows = [
        # Sim -> Logs -> Features
        ((0.50, 0.90), (0.50, 0.85)),
        ((0.50, 0.78), (0.50, 0.73)),
        
        # Features -> Models
        ((0.50, 0.66), (0.20, 0.57)),
        ((0.50, 0.66), (0.50, 0.57)),
        ((0.50, 0.66), (0.80, 0.57)),
        
        # Models -> Evaluation
        ((0.20, 0.48), (0.42, 0.38)),
        ((0.50, 0.48), (0.50, 0.38)),
        ((0.80, 0.48), (0.58, 0.38)),
        
        # Evaluation -> Outputs
        ((0.40, 0.30), (0.29, 0.20)),
        ((0.60, 0.30), (0.71, 0.20)),
    ]

    for start, end in arrows:
        ax.annotate(
            "",
            xy=end, xycoords="data",
            xytext=start, textcoords="data",
            arrowprops=dict(
                arrowstyle="-|>",
                color="#444444",
                lw=1.5,
                mutation_scale=12,
            ),
        )

    ax.set_title("Figure 1: CLSI-Adapt Simulator Architecture & Pipeline Flow", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    # PNG preview for convenience
    plt.savefig(output_path.with_suffix(".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: ROC Curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: Dict[str, Any],
    output_path: Path,
    sim_data: Optional[Dict[str, pd.DataFrame]] = None,
    models: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate Figure 2: ROC Curves across Learner Profiles.

    Plots ROC curves for CLSI-Adapt, Rule-Based CLSI, and BKT across the 5 learner profiles
    and a pooled aggregate panel.

    Parameters
    ----------
    results : dict
        Evaluation results dictionary containing 'predictions' DataFrame.
    output_path : Path
        Target PDF file path.
    sim_data : dict, optional
        Raw simulation DataFrames.
    models : dict, optional
        Fitted models dictionary.
    """
    _apply_publication_style()
    pred_df = results.get("predictions")

    if pred_df is None or pred_df.empty:
        warnings.warn("No prediction DataFrame in results; skipping ROC plot.")
        return

    profiles = list(pred_df["profile"].unique())
    # Subplot layout: 2 rows x 3 columns (5 profiles + 1 pooled)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=150)
    axes_flat = axes.flatten()

    model_names = ["clsi_adapt", "rule_based_clsi", "bkt"]

    for idx, profile in enumerate(profiles):
        ax = axes_flat[idx]
        ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Random Guess")

        for model in model_names:
            sub = pred_df[(pred_df["profile"] == profile) & (pred_df["model"] == model)]
            if sub.empty:
                continue

            y_true = sub["y_true"].to_numpy()
            y_prob = sub["y_prob"].to_numpy()

            if len(np.unique(y_true)) < 2:
                # Single class edge case
                ax.plot([], [], label=f"{PRETTY_NAMES.get(model, model)} (AUC N/A)", color=MODEL_COLORS.get(model, "#333"))
                continue

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = float(roc_auc_score(y_true, y_prob))

            label_str = f"{PRETTY_NAMES.get(model, model)} (AUC = {auc_val:.3f})"
            ax.plot(
                fpr, tpr,
                label=label_str,
                color=MODEL_COLORS.get(model, "#333"),
                linestyle=MODEL_STYLES.get(model, "-"),
                lw=1.8,
            )

        ax.set_title(f"Profile: {PRETTY_NAMES.get(profile, profile)}", fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True)
        ax.legend(loc="lower right", fontsize=8.5)

    # 6th Panel: Pooled Predictions across all profiles
    ax_pooled = axes_flat[5]
    ax_pooled.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6, label="Random Guess")

    for model in model_names:
        sub = pred_df[pred_df["model"] == model]
        if sub.empty:
            continue

        y_true = sub["y_true"].to_numpy()
        y_prob = sub["y_prob"].to_numpy()

        if len(np.unique(y_true)) >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = float(roc_auc_score(y_true, y_prob))
            label_str = f"{PRETTY_NAMES.get(model, model)} (AUC = {auc_val:.3f})"
            ax_pooled.plot(
                fpr, tpr,
                label=label_str,
                color=MODEL_COLORS.get(model, "#333"),
                linestyle=MODEL_STYLES.get(model, "-"),
                lw=2.0,
            )

    ax_pooled.set_title("Pooled (All Profiles)", fontsize=11, fontweight="bold")
    ax_pooled.set_xlabel("False Positive Rate")
    ax_pooled.set_ylabel("True Positive Rate")
    ax_pooled.set_xlim([-0.02, 1.02])
    ax_pooled.set_ylim([-0.02, 1.02])
    ax_pooled.grid(True)
    ax_pooled.legend(loc="lower right", fontsize=8.5)

    plt.suptitle("Figure 2: Receiver Operating Characteristic (ROC) Curves by Learner Profile", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: SHAP Summary
# ---------------------------------------------------------------------------

def plot_shap_summary(
    models: Dict[str, Any],
    sim_data: Dict[str, pd.DataFrame],
    output_path: Path,
    profile: str = "average",
) -> None:
    """Generate Figure 3: SHAP Beeswarm Summary Plot for CLSI-Adapt.

    Uses the final CLSI-Adapt model trained on the specified profile (default: 'average').

    Parameters
    ----------
    models : dict
        Models dict containing 'clsi_adapt'.
    sim_data : dict
        Simulated DataFrames.
    output_path : Path
        Target PDF file path.
    profile : str
        Target profile name.
    """
    _apply_publication_style()
    clsi_adapt = models.get("clsi_adapt")

    if clsi_adapt is None or not hasattr(clsi_adapt, "get_result"):
        warnings.warn("CLSIAdaptModel instance not found or unfitted; skipping SHAP plot.")
        return

    if profile not in clsi_adapt.profiles:
        # Fit model on sim_data if not already fitted
        clsi_adapt.fit(sim_data)

    result = clsi_adapt.get_result(profile)
    final_model = result.final_model
    X_final = result.X_final

    if final_model is None or X_final is None:
        warnings.warn(f"Final model for profile '{profile}' is None; skipping SHAP plot.")
        return

    shap_values, _ = compute_shap(final_model, X_final)

    plot_shap_summary_impl(
        shap_values,
        X_final,
        title=f"Figure 3: SHAP Feature Importance (Profile: {PRETTY_NAMES.get(profile, profile)})",
        show=False,
        save_path=output_path,
    )
    # Also save PNG preview
    plot_shap_summary_impl(
        shap_values,
        X_final,
        title=f"Figure 3: SHAP Feature Importance (Profile: {PRETTY_NAMES.get(profile, profile)})",
        show=False,
        save_path=output_path.with_suffix(".png"),
    )


# ---------------------------------------------------------------------------
# Figure 4: Causal Learning Curve
# ---------------------------------------------------------------------------

def plot_learning_curve(
    sim_data: Dict[str, pd.DataFrame],
    output_path: Path,
    profile: str = "average",
    seed: int = 42,
) -> None:
    """Generate Figure 4: Causal Learning Curve for CLSI-Adapt.

    Plots training set size N vs. validation ROC AUC.
    Strictly causal methodology:
      For each training size N:
        - Train on the first N eligible historical observations.
        - Evaluate strictly on subsequent observations (> N).
        - Compute ROC AUC without temporal leakage or shuffling.

    Parameters
    ----------
    sim_data : dict
        Simulated DataFrames.
    output_path : Path
        Target PDF file path.
    profile : str
        Learner profile name.
    seed : int
        Random seed.
    """
    # Check if input sim_data has enough observations and class diversity
    df_profile = sim_data.get(profile) if sim_data else None

    # Helper function to evaluate learning curve on an interaction DataFrame
    def _eval_curve(df_in: pd.DataFrame) -> Tuple[List[int], List[float]]:
        X_elig, y_elig, _, interaction_ids, _ = _prepare_profile_data(df_in, warmup=20)
        total = len(X_elig)
        if total < 50:
            return [], []

        # Target training observation sizes as required (50, 100, ..., 500)
        t_sizes = [N for N in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500] if N < total - 20]

        ev_sizes: List[int] = []
        ev_aucs: List[float] = []

        for N in t_sizes:
            X_tr, y_tr = X_elig[:N], y_elig[:N]
            X_ev, y_ev = X_elig[N:], y_elig[N:]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_ev)) < 2:
                continue

            n_pos = int(y_tr.sum())
            spw = (len(y_tr) - n_pos) / max(1, n_pos)

            clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                scale_pos_weight=spw,
                random_state=seed,
                tree_method="hist",
                verbosity=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_tr, y_tr)

            y_prob = clf.predict_proba(X_ev)[:, 1]
            auc_val = float(roc_auc_score(y_ev, y_prob))

            ev_sizes.append(N)
            ev_aucs.append(auc_val)

        return ev_sizes, ev_aucs

    evaluated_sizes: List[int] = []
    aucs: List[float] = []

    # Evaluate exclusively on provided sim_data (no fallback dataset generator)
    if df_profile is not None:
        evaluated_sizes, aucs = _eval_curve(df_profile)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    if evaluated_sizes:
        ax.plot(
            evaluated_sizes,
            aucs,
            "o-",
            color="#2b5c8f",
            lw=2.0,
            ms=6,
            label=f"CLSI-Adapt Validation AUC ({PRETTY_NAMES.get(profile, profile)})",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Single-Class Target in Short Simulation Sequence\n(AUC Undefined for N splits)",
            ha="center",
            va="center",
            fontsize=11,
            color="#666666",
        )

    ax.set_title(
        f"Figure 4: Causal Learning Curve (Profile: {PRETTY_NAMES.get(profile, profile)})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Historical Training Set Size ($N$)")
    ax.set_ylabel("Validation ROC AUC (Subsequent Observations)")
    ax.set_ylim([0.0, 1.05])
    ax.grid(True)
    if evaluated_sizes:
        ax.legend(loc="lower right")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 5: Model Comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate Figure 5: Model Performance Comparison Bar Charts.

    Multi-panel bar chart comparing CLSI-Adapt, Rule-Based CLSI, and BKT across
    AUC, Precision, and Recall for all five learner profiles.

    Parameters
    ----------
    results : dict
        Evaluation results dictionary containing 'profile_metrics'.
    output_path : Path
        Target PDF file path.
    """
    _apply_publication_style()
    pm_df = results.get("profile_metrics")

    if pm_df is None or pm_df.empty:
        warnings.warn("No profile_metrics DataFrame in results; skipping comparison plot.")
        return

    metrics_to_plot = ["auc", "precision", "recall"]
    metric_titles = {"auc": "ROC AUC", "precision": "Precision", "recall": "Recall"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150, sharey=True)

    profiles = list(pm_df["profile"].unique())
    models = ["clsi_adapt", "rule_based_clsi", "bkt"]

    x = np.arange(len(profiles))
    width = 0.25

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        for m_idx, model in enumerate(models):
            vals: List[float] = []
            for p in profiles:
                sub = pm_df[(pm_df["profile"] == p) & (pm_df["model"] == model)]
                if not sub.empty:
                    val = sub[metric].iloc[0]
                    vals.append(0.0 if np.isnan(val) else float(val))
                else:
                    vals.append(0.0)

            offset = (m_idx - 1) * width
            rects = ax.bar(
                x + offset, vals, width,
                label=PRETTY_NAMES.get(model, model),
                color=MODEL_COLORS.get(model, "#333"),
                edgecolor="#222222",
                lw=0.8,
            )

        ax.set_title(metric_titles[metric], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY_NAMES.get(p, p) for p in profiles], rotation=25, ha="right")
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, axis="y")

        if idx == 0:
            ax.set_ylabel("Metric Score")
            ax.legend(loc="upper left", fontsize=8.5)

    plt.suptitle("Figure 5: Model Performance Benchmark Comparison Across Learner Profiles", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), format="png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Master Driver: generate_plots
# ---------------------------------------------------------------------------

def generate_plots(
    results: Dict[str, Any],
    config: Config = default_config,
    sim_data: Optional[Dict[str, pd.DataFrame]] = None,
    models: Optional[Dict[str, Any]] = None,
    features: Optional[Tuple] = None,
) -> None:
    """Generate all publication figures and save to config.figures_dir.

    Pipeline sequence:
      1. Figure 1 — System Architecture Diagram (PDF)
      2. Figure 2 — ROC Curves (PDF)
      3. Figure 3 — SHAP Beeswarm Summary Plot (PDF)
      4. Figure 4 — Causal Learning Curve (PDF)
      5. Figure 5 — Model Benchmark Comparison (PDF)

    Parameters
    ----------
    results : dict
        Results dict returned by evaluate_models.
    config : Config
        Configuration container with figures_dir path.
    sim_data : dict, optional
        Raw simulation output dictionary.
    models : dict, optional
        Fitted models dictionary.
    features : tuple, optional
        Feature tuple (X, y, meta).
    """
    fig_dir = config.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("  [Visualization] Generating Figure 1 (Architecture Diagram)...")
    plot_architecture(fig_dir / "figure1_architecture.pdf")

    print("  [Visualization] Generating Figure 2 (ROC Curves)...")
    plot_roc_curves(results, fig_dir / "figure2_roc_curves.pdf", sim_data=sim_data, models=models)

    if models is not None and sim_data is not None:
        print("  [Visualization] Generating Figure 3 (SHAP Summary)...")
        plot_shap_summary(models, sim_data, fig_dir / "figure3_shap_summary.pdf")
    else:
        warnings.warn("models/sim_data context not passed to generate_plots; skipping SHAP plot.")

    if sim_data is not None:
        print("  [Visualization] Generating Figure 4 (Learning Curve)...")
        plot_learning_curve(sim_data, fig_dir / "figure4_learning_curve.pdf", profile="average", seed=config.seed)
    else:
        warnings.warn("sim_data context not passed to generate_plots; skipping learning curve.")

    print("  [Visualization] Generating Figure 5 (Model Comparison Bar Charts)...")
    plot_model_comparison(results, fig_dir / "figure5_model_comparison.pdf")

    print(f"  Visualizations complete. Figures saved in: {fig_dir}")
