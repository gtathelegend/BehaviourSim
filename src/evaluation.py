"""Evaluation module for CLSI-Adapt Simulator.

Provides a unified framework to benchmark CLSI-Adapt, Rule-Based CLSI, and BKT.
Computes classification metrics, event detection recall, and recovery times.
Generates publication-ready CSV and LaTeX tables for the JEDM paper.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.config import Config, default_config
from src.feature_engineering import FEATURE_COLUMNS, build_overload_target


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard classification metrics: AUC, precision, recall, f1.

    Handles edge cases explicitly:
    - If y_true contains only one class, ROC AUC is undefined. It is set to NaN
      and documented in the output rather than filled with a default.
    - Zero-division for precision, recall, and f1 is set to 0.0 to prevent crash,
      as documented in the evaluation methodology.

    Parameters
    ----------
    y_true : np.ndarray
        True binary overload target labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class (overload).
    y_pred : np.ndarray
        Predicted binary labels (0 or 1).

    Returns
    -------
    metrics : dict
        Dict with keys: 'auc', 'precision', 'recall', 'f1'. Values are float.
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred_arr = np.asarray(y_pred)

    if len(np.unique(y_true_arr)) < 2:
        # ROC AUC is undefined if only one class exists in the target.
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))

    # zero_division=0.0 is used when there are no positive predictions / targets
    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0.0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0.0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0.0))

    return {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Overload-Event Detection
# ---------------------------------------------------------------------------

def extract_overload_events(
    state_sequence: Union[pd.Series, pd.DataFrame],
    profile: str,
) -> List[Dict[str, Any]]:
    """Extract contiguous overload events from ground-truth state sequence(s).

    An overload event is defined as a contiguous block of interactions where
    the simulator's ground-truth cognitive state is "Overload".
    When a DataFrame with learner_id is provided, events are extracted strictly
    within each learner sequence.

    Parameters
    ----------
    state_sequence : pd.Series or pd.DataFrame
        The ground-truth state sequence or full simulation DataFrame.
    profile : str
        Profile name for metadata.

    Returns
    -------
    events : list of dict
        Each dict represents an event:
            {
                'event_id': str,
                'profile': str,
                'learner_id': int,
                'start_interaction': int,
                'end_interaction': int,
                'start_idx': int,
                'end_idx': int,
                'duration': int
            }
    """
    events: List[Dict[str, Any]] = []

    if isinstance(state_sequence, pd.DataFrame) and "learner_id" in state_sequence.columns:
        groups = state_sequence.groupby("learner_id", sort=False)
        event_counter = 1
        for learner_id, df_l in groups:
            states = df_l["state"].to_numpy()
            base_offset = df_l.index[0]
            in_event = False
            start_idx = -1

            for local_idx, state in enumerate(states):
                global_idx = base_offset + local_idx
                if state == "Overload":
                    if not in_event:
                        in_event = True
                        start_idx = global_idx
                else:
                    if in_event:
                        end_idx = global_idx - 1
                        events.append(
                            {
                                "event_id": f"{profile}_l{learner_id}_event_{event_counter}",
                                "profile": profile,
                                "learner_id": int(learner_id),
                                "start_interaction": int(df_l.loc[start_idx, "interaction_id"]) if "interaction_id" in df_l.columns else start_idx + 1,
                                "end_interaction": int(df_l.loc[end_idx, "interaction_id"]) if "interaction_id" in df_l.columns else end_idx + 1,
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                                "duration": end_idx - start_idx + 1,
                            }
                        )
                        event_counter += 1
                        in_event = False

            if in_event:
                end_idx = base_offset + len(states) - 1
                events.append(
                    {
                        "event_id": f"{profile}_l{learner_id}_event_{event_counter}",
                        "profile": profile,
                        "learner_id": int(learner_id),
                        "start_interaction": int(df_l.loc[start_idx, "interaction_id"]) if "interaction_id" in df_l.columns else start_idx + 1,
                        "end_interaction": int(df_l.loc[end_idx, "interaction_id"]) if "interaction_id" in df_l.columns else end_idx + 1,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "duration": end_idx - start_idx + 1,
                    }
                )
                event_counter += 1
        return events

    # Single sequence fallback
    states = state_sequence["state"].to_numpy() if isinstance(state_sequence, pd.DataFrame) else state_sequence.to_numpy()
    in_event = False
    start_idx = -1
    event_counter = 1

    for idx, state in enumerate(states):
        if state == "Overload":
            if not in_event:
                in_event = True
                start_idx = idx
        else:
            if in_event:
                end_idx = idx - 1
                events.append(
                    {
                        "event_id": f"{profile}_event_{event_counter}",
                        "profile": profile,
                        "learner_id": 1,
                        "start_interaction": start_idx + 1,
                        "end_interaction": end_idx + 1,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "duration": end_idx - start_idx + 1,
                    }
                )
                event_counter += 1
                in_event = False

    if in_event:
        end_idx = len(states) - 1
        events.append(
            {
                "event_id": f"{profile}_event_{event_counter}",
                "profile": profile,
                "learner_id": 1,
                "start_interaction": start_idx + 1,
                "end_interaction": end_idx + 1,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration": end_idx - start_idx + 1,
            }
        )

    return events


def compute_event_metrics(
    events: List[Dict[str, Any]],
    y_pred_seq: np.ndarray,
) -> Dict[str, Any]:
    """Calculate overload event detection rate (event recall).

    Definition:
      An event is considered detected iff the model predicts overload (y_pred == 1)
      at least once during the ground-truth event's interval (inclusive).

    Parameters
    ----------
    events : list of dict
        List of events extracted by `extract_overload_events`.
    y_pred_seq : np.ndarray
        Model's predicted binary sequence of length N (full sequence).

    Returns
    -------
    metrics : dict
        Dict with keys: 'n_events', 'n_detected', 'event_recall'.
    """
    n_events = len(events)
    if n_events == 0:
        return {
            "n_events": 0,
            "n_detected": 0,
            "event_recall": float("nan"),
        }

    n_detected = 0
    for event in events:
        start_idx = event["start_idx"]
        end_idx = event["end_idx"]
        preds_in_event = y_pred_seq[start_idx : end_idx + 1]
        if np.any(preds_in_event == 1):
            n_detected += 1

    return {
        "n_events": n_events,
        "n_detected": n_detected,
        "event_recall": n_detected / n_events,
    }


# ---------------------------------------------------------------------------
# Recovery Time Analysis
# ---------------------------------------------------------------------------

def compute_recovery_metrics(
    events: List[Dict[str, Any]],
    y_pred_seq: np.ndarray,
    state_sequence: Union[pd.Series, pd.DataFrame],
) -> Dict[str, Any]:
    """Calculate recovery-time metrics for detected overload events.

    Recovery time definition:
      1. For each ground-truth event, identify the model's first overload prediction (y_pred == 1)
         within the event boundary (step `t`). If not detected, the event is skipped.
      2. Search forward in the ground-truth sequence from step `t` to find the first step `t_opt`
         where the learner's state returns to "Optimal". If multi-learner, search stops at learner boundary.
      3. The recovery time is `t_opt - t` (number of interactions).
      4. If the learner never returns to "Optimal" before sequence ends, event is unrecovered.

    Parameters
    ----------
    events : list of dict
        Overload events extracted by `extract_overload_events`.
    y_pred_seq : np.ndarray
        Model's predicted binary sequence of length N.
    state_sequence : pd.Series or pd.DataFrame
        The ground-truth state sequence of length N.

    Returns
    -------
    metrics : dict
    """
    if isinstance(state_sequence, pd.DataFrame):
        states = state_sequence["state"].to_numpy()
        learner_ids = state_sequence["learner_id"].to_numpy() if "learner_id" in state_sequence.columns else None
    else:
        states = state_sequence.to_numpy()
        learner_ids = None

    recovery_times: List[int] = []
    n_detected = 0
    n_recovered = 0
    n_unrecovered = 0

    for event in events:
        start_idx = event["start_idx"]
        end_idx = event["end_idx"]
        event_learner_id = event.get("learner_id", None)

        detected_idx = -1
        for idx in range(start_idx, end_idx + 1):
            if y_pred_seq[idx] == 1:
                detected_idx = idx
                break

        if detected_idx == -1:
            continue

        n_detected += 1

        recovered = False
        for idx in range(detected_idx + 1, len(states)):
            if learner_ids is not None and event_learner_id is not None:
                if learner_ids[idx] != event_learner_id:
                    # Boundary of learner reached without returning to Optimal
                    break

            if states[idx] == "Optimal":
                recovery_time = idx - detected_idx
                recovery_times.append(recovery_time)
                n_recovered += 1
                recovered = True
                break

        if not recovered:
            n_unrecovered += 1

    mean_recovery = float(np.mean(recovery_times)) if recovery_times else float("nan")
    median_recovery = float(np.median(recovery_times)) if recovery_times else float("nan")

    return {
        "n_events": len(events),
        "n_detected": n_detected,
        "n_recovered": n_recovered,
        "n_unrecovered": n_unrecovered,
        "mean_recovery_time": mean_recovery,
        "median_recovery_time": median_recovery,
    }


# ---------------------------------------------------------------------------
# Simulator-level Cognitive State Statistics
# ---------------------------------------------------------------------------

def compute_state_statistics(
    sim_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Calculate simulator-level ground-truth cognitive state fractions.

    These are property statistics of the simulated learner populations. They
    should NOT be described as causal effects of the models or policies.

    Parameters
    ----------
    sim_data : dict
        Dict mapping profile name to simulated DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: profile, optimal_fraction, overload_fraction, underload_fraction, n_interactions
    """
    rows: List[Dict[str, Any]] = []
    all_states: List[str] = []

    for profile, df in sim_data.items():
        states = df["state"].to_numpy()
        all_states.extend(states)
        n = len(states)

        opt_f = float(np.sum(states == "Optimal") / n) if n > 0 else 0.0
        ov_f = float(np.sum(states == "Overload") / n) if n > 0 else 0.0
        un_f = float(np.sum(states == "Underload") / n) if n > 0 else 0.0

        rows.append(
            {
                "profile": profile,
                "optimal_fraction": opt_f,
                "overload_fraction": ov_f,
                "underload_fraction": un_f,
                "n_interactions": n,
            }
        )

    # Aggregate (pooled)
    if all_states:
        all_arr = np.array(all_states)
        n_total = len(all_arr)
        opt_f = float(np.sum(all_arr == "Optimal") / n_total)
        ov_f = float(np.sum(all_arr == "Overload") / n_total)
        un_f = float(np.sum(all_arr == "Underload") / n_total)

        rows.append(
            {
                "profile": "Aggregate",
                "optimal_fraction": opt_f,
                "overload_fraction": ov_f,
                "underload_fraction": un_f,
                "n_interactions": n_total,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _reconstruct_df(X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct an interaction DataFrame from features and metadata."""
    df = pd.DataFrame(X[:, :10], columns=FEATURE_COLUMNS)
    df["profile"] = meta["profile"].values
    df["interaction_id"] = meta["interaction_id"].values
    return df


# ---------------------------------------------------------------------------
# Latex Table Formatter
# ---------------------------------------------------------------------------

def df_to_latex(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to a publication-ready LaTeX table."""
    pretty_names = {
        "fast_accurate": "Fast-Accurate",
        "fast_inaccurate": "Fast-Inaccurate",
        "slow_accurate": "Slow-Accurate",
        "slow_inaccurate": "Slow-Inaccurate",
        "average": "Average",
        "clsi_adapt": "CLSI-Adapt",
        "rule_based_clsi": "Rule-Based CLSI",
        "bkt": "BKT",
        "pooled": "Pooled",
        "mean_across_profiles": "Across-Profile Mean",
        "std_across_profiles": "Across-Profile Std",
        "Aggregate": "Aggregate",
        "model_native": "Model-Native",
        "common_oof": "Common OOF Subset",
    }

    col_names = {
        "profile": "Profile",
        "model": "Model",
        "evaluation_subset": "Subset",
        "auc": "AUC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "n_expected": "$N_{exp}$",
        "n_predictions": "$N_{pred}$",
        "n_missing": "$N_{miss}$",
        "n_samples": "$N$",
        "n_positive": "$N_{pos}$",
        "n_negative": "$N_{neg}$",
        "aggregation_type": "Aggregation Type",
        "metric": "Metric",
        "value": "Value",
        "n_profiles_contributing": "$N_{prof}$",
        "n_events": "Total Events",
        "n_detected": "Detected",
        "n_recovered": "Recovered",
        "n_unrecovered": "Unrecovered",
        "event_recall": "Event Recall",
        "mean_recovery_time": "Mean Recovery Time",
        "median_recovery_time": "Median Recovery Time",
        "optimal_fraction": "Optimal Fraction",
        "overload_fraction": "Overload Fraction",
        "underload_fraction": "Underload Fraction",
        "n_interactions": "Total Interactions",
    }

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")

    headers = [col_names.get(c, c.replace("_", " ").title()) for c in df.columns]
    col_align = "l" * len(df.columns)
    lines.append(f"\\begin{{tabular}}{{{col_align}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        formatted_vals = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, str):
                pretty_val = pretty_names.get(val, val).replace("_", r"\_")
                formatted_vals.append(pretty_val)
            elif val is None or (isinstance(val, float) and np.isnan(val)):
                formatted_vals.append("NaN")
            elif isinstance(val, (int, np.integer)):
                formatted_vals.append(f"{val:,}")
            elif isinstance(val, (float, np.floating)):
                formatted_vals.append(f"{val:.3f}")
            else:
                formatted_vals.append(str(val))
        lines.append(" & ".join(formatted_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 5 Audit Generator
# ---------------------------------------------------------------------------

def generate_phase5_audit(
    results: Dict[str, Any],
    sim_data: Dict[str, pd.DataFrame],
    audit_file: Path,
) -> None:
    """Generate comprehensive Phase 5 evaluation audit report.

    Saves: results/evaluation_phase5_audit.txt
    """
    pm_df = results["profile_metrics"]
    am_df = results["aggregate_metrics"]
    rm_df = results["recovery_metrics"]
    ss_df = results["state_statistics"]

    lines = []
    lines.append("=" * 80)
    lines.append("CLSI-ADAPT PHASE 5 EVALUATION AUDIT REPORT")
    lines.append("=" * 80)
    lines.append("")

    # 1. Dataset Specs
    lines.append("1. DATASET SPECIFICATION & GROUND-TRUTH TARGET")
    lines.append("-" * 80)
    lines.append("Profiles                           : 5 (fast_accurate, slow_accurate, average, fast_inaccurate, slow_inaccurate)")
    lines.append("Learners per Profile               : 10")
    lines.append("Interactions per Learner           : 1000")
    lines.append("Total Raw Interactions             : 50,000 (10,000 per profile)")
    lines.append("Random Seed                        : 42")
    lines.append("Target Definition                  : Overload (prior_acc >= 0.75 AND future_acc <= 0.50)")
    lines.append("CV Strategy                        : 5-fold forward-chaining TimeSeriesSplit on unique interaction time steps")
    lines.append("CV Invariant Check                 : max(train_interaction_id) < min(validation_interaction_id)")
    lines.append("Simultaneous Time Leakage          : 0 shared interaction time steps")
    lines.append("")

    # 2. Prediction Coverage
    lines.append("2. PREDICTION COVERAGE & EVALUATION SUBSETS")
    lines.append("-" * 80)
    lines.append("OOF Prediction Calculation Note    : Corrected arithmetic:")
    lines.append("  5 validation folds * 1620 validation observations = 8100 unique OOF predictions.")
    lines.append("  The first 1670 observations constitute the initial training history and are not validation observations.")
    lines.append("")
    lines.append(f"{'Profile':<17} {'Model':<16} {'Subset':<16} {'Expected':<10} {'Predicted':<10} {'Missing':<10} {'Positives':<10} {'Negatives':<10}")
    lines.append("-" * 99)
    for _, row in pm_df.iterrows():
        lines.append(
            f"{row['profile']:<17} {row['model']:<16} {row['evaluation_subset']:<16} "
            f"{row['n_expected']:<10} {row['n_predictions']:<10} {row['n_missing']:<10} "
            f"{row['n_positive']:<10} {row['n_negative']:<10}"
        )
    lines.append("")

    # 3. Classification Metrics
    lines.append("3. PER-PROFILE CLASSIFICATION METRICS")
    lines.append("-" * 80)
    lines.append(f"{'Profile':<17} {'Model':<16} {'Subset':<16} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    lines.append("-" * 89)
    for _, row in pm_df.iterrows():
        auc_str = f"{row['auc']:.4f}" if not np.isnan(row['auc']) else "NaN"
        prec_str = f"{row['precision']:.4f}" if not np.isnan(row['precision']) else "NaN"
        rec_str = f"{row['recall']:.4f}" if not np.isnan(row['recall']) else "NaN"
        f1_str = f"{row['f1']:.4f}" if not np.isnan(row['f1']) else "NaN"
        lines.append(
            f"{row['profile']:<17} {row['model']:<16} {row['evaluation_subset']:<16} "
            f"{auc_str:<10} {prec_str:<10} {rec_str:<10} {f1_str:<10}"
        )
    lines.append("")

    # 4. Aggregated Metrics
    lines.append("4. AGGREGATED CLASSIFICATION METRICS")
    lines.append("-" * 80)
    lines.append(f"{'Aggregation':<22} {'Model':<16} {'Subset':<16} {'Metric':<10} {'Value':<10} {'Profiles Contrib':<16}")
    lines.append("-" * 90)
    for _, row in am_df.iterrows():
        val_str = f"{row['value']:.4f}" if not np.isnan(row['value']) else "NaN"
        lines.append(
            f"{row['aggregation_type']:<22} {row['model']:<16} {row['evaluation_subset']:<16} "
            f"{row['metric']:<10} {val_str:<10} {row['n_profiles_contributing']:<16}"
        )
    lines.append("")

    # 5. Overload Event Detection & Recovery
    lines.append("5. OVERLOAD EVENT DETECTION & RECOVERY TIME ANALYSIS")
    lines.append("-" * 80)
    lines.append(f"{'Profile':<17} {'Model':<16} {'Events':<8} {'Detected':<10} {'Event Recall':<14} {'Recovered':<10} {'Unrecovered':<12} {'Mean Rec Time':<15} {'Med Rec Time':<12}")
    lines.append("-" * 114)
    for _, row in rm_df.iterrows():
        ev_rec_str = f"{row['event_recall']:.4f}" if not np.isnan(row['event_recall']) else "NaN"
        mean_rec_str = f"{row['mean_recovery_time']:.3f}" if not np.isnan(row['mean_recovery_time']) else "NaN"
        med_rec_str = f"{row['median_recovery_time']:.1f}" if not np.isnan(row['median_recovery_time']) else "NaN"
        lines.append(
            f"{row['profile']:<17} {row['model']:<16} {row['n_events']:<8} {row['n_detected']:<10} "
            f"{ev_rec_str:<14} {row['n_recovered']:<10} {row['n_unrecovered']:<12} "
            f"{mean_rec_str:<15} {med_rec_str:<12}"
        )
    lines.append("")

    # 6. Simulator Cognitive State Statistics
    lines.append("6. SIMULATOR COGNITIVE STATE STATISTICS (POPULATION PROPERTIES)")
    lines.append("-" * 80)
    lines.append("Note: These state fractions are population properties of simulated learner profiles.")
    lines.append("They are NOT causal effects of the classification models.")
    lines.append("")
    lines.append(f"{'Profile':<17} {'Optimal Fraction':<18} {'Overload Fraction':<19} {'Underload Fraction':<20} {'Total Interactions':<18}")
    lines.append("-" * 92)
    for _, row in ss_df.iterrows():
        lines.append(
            f"{row['profile']:<17} {row['optimal_fraction']:<18.5f} {row['overload_fraction']:<19.5f} "
            f"{row['underload_fraction']:<20.5f} {row['n_interactions']:<18}"
        )
    lines.append("")

    # 7. Methodological Audit & Model Interpretation
    lines.append("7. METHODOLOGICAL AUDIT & MODEL INTERPRETATION")
    lines.append("-" * 80)
    lines.append("CLSI-Adapt Model:")
    lines.append("  - Supervised personalization model using 11 engineered features and XGBoost classifier.")
    lines.append("  - Evaluated out-of-fold (OOF) across 5 temporal CV validation folds.")
    lines.append("")
    lines.append("Rule-Based CLSI Model:")
    lines.append("  - Parameter-free composite index: CLSI = (0.50*acc + 0.25*(1-NRT) + 0.15*(1-wer) + 0.10*(1-retries_norm)) / 1.0.")
    lines.append("  - Applies help penalty of 0.10 when help is requested.")
    lines.append("  - Predicts overload when CLSI < 0.40 (un-tuned fixed threshold).")
    lines.append("  - Causal formulation; does not use future information.")
    lines.append("")
    lines.append("BKT Model:")
    lines.append("  - Bayesian Knowledge Tracing baseline / struggle proxy.")
    lines.append("  - Parameters: initial_mastery=0.3, learn=0.1, guess=0.2, slip=0.1.")
    lines.append("  - Resets BKT state at every learner boundary.")
    lines.append("  - Predicts struggle when mastery probability < 0.30.")
    lines.append("  - IMPORTANT: BKT is explicitly a domain mastery tracker / struggle proxy, NOT a direct cognitive-load detector.")
    lines.append("")
    lines.append("Handling of Undefined Metrics (NaN):")
    lines.append("  - Fast-inaccurate and Slow-inaccurate profiles contain 0 or near-0 positive overload targets.")
    lines.append("  - Single-class validation folds return ROC AUC = NaN.")
    lines.append("  - Aggregate across-profile mean and standard deviation explicitly exclude NaN values and report n_profiles_contributing.")
    lines.append("  - NaN values are never silently converted to zero.")
    lines.append("=" * 80)

    audit_file.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_file, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Public API: Evaluation Driver & File Exporter
# ---------------------------------------------------------------------------

def create_summary_tables(
    results: Dict[str, Any],
    output_dir: Path,
    sim_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    """Write evaluation metrics dataframes to CSV, LaTeX, and text audit files.

    Saves:
      results/tables/profile_metrics.csv & .tex
      results/tables/aggregate_metrics.csv & .tex
      results/tables/recovery_metrics.csv & .tex
      results/tables/state_statistics.csv & .tex
      results/evaluation_phase5_audit.txt
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Profile classification metrics
    pm_df = results["profile_metrics"]
    pm_df.to_csv(output_dir / "profile_metrics.csv", index=False)
    with open(output_dir / "profile_metrics.tex", "w") as f:
        f.write(df_to_latex(pm_df))

    # 2. Aggregated metrics
    am_df = results["aggregate_metrics"]
    am_df.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    with open(output_dir / "aggregate_metrics.tex", "w") as f:
        f.write(df_to_latex(am_df))

    # 3. Recovery metrics
    rm_df = results["recovery_metrics"]
    rm_df.to_csv(output_dir / "recovery_metrics.csv", index=False)
    with open(output_dir / "recovery_metrics.tex", "w") as f:
        f.write(df_to_latex(rm_df))

    # 4. State statistics
    ss_df = results["state_statistics"]
    ss_df.to_csv(output_dir / "state_statistics.csv", index=False)
    with open(output_dir / "state_statistics.tex", "w") as f:
        f.write(df_to_latex(ss_df))

    # 5. Evaluation Phase 5 Audit Report
    if sim_data is not None:
        audit_path = output_dir.parent / "evaluation_phase5_audit.txt"
        generate_phase5_audit(results, sim_data, audit_path)


def evaluate_models(
    models: Dict[str, Any],
    features: Tuple[np.ndarray, np.ndarray, pd.DataFrame],
    config: Config = default_config,
    sim_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end evaluation benchmark.

    Evaluates predictions from all models across five learner profiles.
    Generates profile-level metrics, aggregates, recovery/event metrics, and audit report.

    Parameters
    ----------
    models : dict
        Model instances: 'clsi_adapt', 'rule_based_clsi', 'bkt'.
    features : tuple
        Engineering output: (X, y, meta).
    config : Config
        Configuration parameters.
    sim_data : dict, optional
        Raw simulation output. Reconstructed from X/meta if not provided.

    Returns
    -------
    results : dict
        Dict with keys:
          'profile_metrics': pd.DataFrame
          'aggregate_metrics': pd.DataFrame
          'recovery_metrics': pd.DataFrame
          'state_statistics': pd.DataFrame
    """
    X, y, meta = features

    # Reconstruct sim_data from features if not passed directly
    if sim_data is None:
        warnings.warn("sim_data not supplied to evaluate_models; reconstructing from features.")
        df_reconstructed = _reconstruct_df(X, meta)
        sim_data = {}
        for p in df_reconstructed["profile"].unique():
            sim_data[p] = df_reconstructed[df_reconstructed["profile"] == p].copy().reset_index(drop=True)

    # ── 1. Fit CLSI-Adapt ────────────────────────────────────────────────
    clsi_adapt = models["clsi_adapt"]
    clsi_adapt.fit(sim_data)

    rule_based = models["rule_based_clsi"]
    bkt = models["bkt"]

    # ── 2. Collect predictions and align ──────────────────────────────────
    pred_rows_native: List[Dict[str, Any]] = []
    pred_rows_common: List[Dict[str, Any]] = []

    profile_event_data: Dict[str, List[Dict[str, Any]]] = {}
    profile_pred_seqs: Dict[str, Dict[str, np.ndarray]] = {}

    for profile, df_profile in sim_data.items():
        N = len(df_profile)
        y_true_full = build_overload_target(df_profile).to_numpy()
        valid_mask = ~np.isnan(y_true_full)

        # Extract overload events (based on ground-truth state sequence)
        events = extract_overload_events(df_profile, profile)
        profile_event_data[profile] = events
        profile_pred_seqs[profile] = {}

        # Rule-Based predictions across sequence
        rb_prob = rule_based.predict_proba(df_profile)[:, 1]
        rb_pred = rule_based.predict(df_profile)
        profile_pred_seqs[profile]["rule_based_clsi"] = rb_pred

        # BKT predictions across sequence (BKT resets per learner)
        bkt_prob = bkt.predict_proba(df_profile)[:, 1]
        bkt_pred = bkt.predict(df_profile)
        profile_pred_seqs[profile]["bkt"] = bkt_pred

        # CLSI-Adapt OOF predictions
        res = clsi_adapt.get_result(profile)
        oof_df = res.oof_predictions

        # OOF positions for CLSI-Adapt
        oof_positions_set = set(oof_df["row_position"].astype(int)) if not oof_df.empty else set()

        adapt_pred_seq = np.zeros(N, dtype=int)
        for _, row in oof_df.iterrows():
            pos = int(row["row_position"])
            adapt_pred_seq[pos] = int(row["y_pred"])
        profile_pred_seqs[profile]["clsi_adapt"] = adapt_pred_seq

        # Record Native and Common predictions
        for idx in range(N):
            if valid_mask[idx]:
                l_id = int(df_profile["learner_id"].iloc[idx]) if "learner_id" in df_profile.columns else 1
                i_id = int(df_profile["interaction_id"].iloc[idx])

                # Rule-Based Native
                pred_rows_native.append({
                    "profile": profile, "learner_id": l_id, "interaction_id": i_id,
                    "model": "rule_based_clsi", "evaluation_subset": "model_native",
                    "y_true": int(y_true_full[idx]), "y_pred": int(rb_pred[idx]), "y_prob": float(rb_prob[idx]),
                })
                # BKT Native
                pred_rows_native.append({
                    "profile": profile, "learner_id": l_id, "interaction_id": i_id,
                    "model": "bkt", "evaluation_subset": "model_native",
                    "y_true": int(y_true_full[idx]), "y_pred": int(bkt_pred[idx]), "y_prob": float(bkt_prob[idx]),
                })

                if idx in oof_positions_set:
                    # Common OOF subset records
                    pred_rows_common.append({
                        "profile": profile, "learner_id": l_id, "interaction_id": i_id,
                        "model": "rule_based_clsi", "evaluation_subset": "common_oof",
                        "y_true": int(y_true_full[idx]), "y_pred": int(rb_pred[idx]), "y_prob": float(rb_prob[idx]),
                    })
                    pred_rows_common.append({
                        "profile": profile, "learner_id": l_id, "interaction_id": i_id,
                        "model": "bkt", "evaluation_subset": "common_oof",
                        "y_true": int(y_true_full[idx]), "y_pred": int(bkt_pred[idx]), "y_prob": float(bkt_prob[idx]),
                    })

        # CLSI-Adapt OOF rows (both native and common_oof are identical for CLSI-Adapt)
        for _, row in oof_df.iterrows():
            pos = int(row["row_position"])
            l_id = int(row["learner_id"]) if "learner_id" in row else 1
            i_id = int(row["interaction_id"])
            item = {
                "profile": profile, "learner_id": l_id, "interaction_id": i_id,
                "model": "clsi_adapt", "evaluation_subset": "common_oof",
                "y_true": int(row["y_true"]), "y_pred": int(row["y_pred"]), "y_prob": float(row["y_prob"]),
            }
            pred_rows_common.append(item)
            item_native = dict(item)
            item_native["evaluation_subset"] = "model_native"
            pred_rows_native.append(item_native)

    pred_df_native = pd.DataFrame(pred_rows_native)
    pred_df_common = pd.DataFrame(pred_rows_common)
    all_preds_df = pd.concat([pred_df_native, pred_df_common], ignore_index=True)

    # ── 3. Profile-level classification metrics ──────────────────────────
    model_names = ["clsi_adapt", "rule_based_clsi", "bkt"]
    profile_names = list(sim_data.keys())

    profile_metrics_rows: List[Dict[str, Any]] = []

    for profile in profile_names:
        df_prof = sim_data[profile]
        n_expected_native = int((~build_overload_target(df_prof).isna()).sum())

        for subset_name, sub_df in [("model_native", pred_df_native), ("common_oof", pred_df_common)]:
            # For common_oof, n_expected is the number of OOF rows for that profile
            oof_cnt = len(clsi_adapt.get_result(profile).oof_predictions)
            n_expected = n_expected_native if subset_name == "model_native" else oof_cnt

            for model in model_names:
                sub = sub_df[(sub_df["profile"] == profile) & (sub_df["model"] == model)]
                if sub.empty:
                    # Record empty row with NaNs if no predictions present
                    profile_metrics_rows.append({
                        "profile": profile,
                        "model": model,
                        "evaluation_subset": subset_name,
                        "auc": float("nan"),
                        "precision": float("nan"),
                        "recall": float("nan"),
                        "f1": float("nan"),
                        "n_expected": n_expected,
                        "n_predictions": 0,
                        "n_missing": n_expected,
                        "n_positive": 0,
                        "n_negative": 0,
                    })
                    continue

                y_true_sub = sub["y_true"].to_numpy()
                y_prob_sub = sub["y_prob"].to_numpy()
                y_pred_sub = sub["y_pred"].to_numpy()

                metrics = compute_classification_metrics(y_true_sub, y_prob_sub, y_pred_sub)
                n_preds = len(y_true_sub)
                n_pos = int(np.sum(y_true_sub == 1))
                n_neg = int(np.sum(y_true_sub == 0))

                profile_metrics_rows.append({
                    "profile": profile,
                    "model": model,
                    "evaluation_subset": subset_name,
                    "auc": metrics["auc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "n_expected": n_expected,
                    "n_predictions": n_preds,
                    "n_missing": max(0, n_expected - n_preds),
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                })

    profile_metrics_df = pd.DataFrame(profile_metrics_rows)

    # ── 4. Aggregate metrics (Pooled & Across-profile summary) ────────────
    agg_rows: List[Dict[str, Any]] = []

    for subset_name, sub_df in [("model_native", pred_df_native), ("common_oof", pred_df_common)]:
        for model in model_names:
            model_sub = sub_df[sub_df["model"] == model]

            # A. Pooled metrics
            if not model_sub.empty:
                metrics_pooled = compute_classification_metrics(
                    model_sub["y_true"].to_numpy(),
                    model_sub["y_prob"].to_numpy(),
                    model_sub["y_pred"].to_numpy(),
                )
                # Count distinct profiles contributing to pooled metric
                n_prof_contrib = model_sub["profile"].nunique()
                for key, val in metrics_pooled.items():
                    agg_rows.append({
                        "aggregation_type": "pooled",
                        "model": model,
                        "evaluation_subset": subset_name,
                        "metric": key,
                        "value": val,
                        "n_profiles_contributing": n_prof_contrib,
                    })

            # B. Across-profile Mean & Standard Deviation
            model_pm = profile_metrics_df[
                (profile_metrics_df["model"] == model) & (profile_metrics_df["evaluation_subset"] == subset_name)
            ]

            if not model_pm.empty:
                for metric in ["auc", "precision", "recall", "f1"]:
                    vals = model_pm[metric].to_numpy()
                    valid_vals = vals[~np.isnan(vals)]
                    n_contrib = len(valid_vals)

                    if n_contrib > 0:
                        mean_val = float(np.mean(valid_vals))
                        std_val = float(np.std(valid_vals))
                    else:
                        mean_val = float("nan")
                        std_val = float("nan")

                    agg_rows.append({
                        "aggregation_type": "mean_across_profiles",
                        "model": model,
                        "evaluation_subset": subset_name,
                        "metric": metric,
                        "value": mean_val,
                        "n_profiles_contributing": n_contrib,
                    })
                    agg_rows.append({
                        "aggregation_type": "std_across_profiles",
                        "model": model,
                        "evaluation_subset": subset_name,
                        "metric": metric,
                        "value": std_val,
                        "n_profiles_contributing": n_contrib,
                    })

    aggregate_metrics_df = pd.DataFrame(agg_rows)

    # ── 5. Recovery & Event Metrics ───────────────────────────────────────
    rec_rows: List[Dict[str, Any]] = []

    for profile in profile_names:
        events = profile_event_data[profile]
        df_profile = sim_data[profile]

        for model in model_names:
            y_pred_seq = profile_pred_seqs[profile][model]

            ev_metrics = compute_event_metrics(events, y_pred_seq)
            rec_metrics = compute_recovery_metrics(events, y_pred_seq, df_profile)

            rec_rows.append({
                "profile": profile,
                "model": model,
                "n_events": len(events),
                "n_detected": rec_metrics["n_detected"],
                "n_recovered": rec_metrics["n_recovered"],
                "n_unrecovered": rec_metrics["n_unrecovered"],
                "event_recall": ev_metrics["event_recall"],
                "mean_recovery_time": rec_metrics["mean_recovery_time"],
                "median_recovery_time": rec_metrics["median_recovery_time"],
            })

    recovery_metrics_df = pd.DataFrame(rec_rows)

    # ── 6. State Statistics ───────────────────────────────────────────────
    state_statistics_df = compute_state_statistics(sim_data)

    # Compile result package
    eval_results = {
        "profile_metrics": profile_metrics_df,
        "aggregate_metrics": aggregate_metrics_df,
        "recovery_metrics": recovery_metrics_df,
        "state_statistics": state_statistics_df,
        "predictions": all_preds_df,
    }

    # Save CSV, LaTeX, and Audit Report files
    create_summary_tables(eval_results, output_dir=config.tables_dir, sim_data=sim_data)

    return eval_results

