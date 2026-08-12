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
    state_sequence: pd.Series,
    profile: str,
) -> List[Dict[str, Any]]:
    """Extract contiguous overload events from the ground-truth state sequence.

    An overload event is defined as a contiguous block of interactions where
    the simulator's ground-truth cognitive state is "Overload".

    Parameters
    ----------
    state_sequence : pd.Series of str
        The ground-truth state sequence from simulator output ('Optimal', 'Overload', 'Underload').
    profile : str
        Profile name for metadata.

    Returns
    -------
    events : list of dict
        Each dict represents an event:
            {
                'event_id': str,
                'profile': str,
                'start_interaction': int (1-indexed ID),
                'end_interaction': int (1-indexed ID),
                'start_idx': int (0-indexed position),
                'end_idx': int (0-indexed position),
                'duration': int (number of interactions)
            }
    """
    states = state_sequence.to_numpy()
    events: List[Dict[str, Any]] = []
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
    state_sequence: pd.Series,
) -> Dict[str, Any]:
    """Calculate recovery-time metrics for detected overload events.

    Recovery time definition:
      1. For each ground-truth event, identify the model's first overload prediction (y_pred == 1)
         within the event boundary (step `t`). If not detected, the event is skipped.
      2. Search forward in the ground-truth sequence from step `t` to find the first step `t_opt`
         where the learner's state returns to "Optimal".
      3. The recovery time is `t_opt - t` (number of interactions). The detection interaction itself
         is not counted (e.g. if t_opt is the step right after t, recovery time is 1).
      4. If the learner never returns to "Optimal" before the sequence ends, the event is
         unrecovered (censored) and excluded from mean/median recovery time calculations.

    Parameters
    ----------
    events : list of dict
        Overload events extracted by `extract_overload_events`.
    y_pred_seq : np.ndarray
        Model's predicted binary sequence of length N.
    state_sequence : pd.Series of str
        The ground-truth state sequence of length N.

    Returns
    -------
    metrics : dict
        Dict containing keys:
          'n_events', 'n_detected', 'n_recovered', 'n_unrecovered',
          'mean_recovery_time', 'median_recovery_time'
    """
    states = state_sequence.to_numpy()
    recovery_times: List[int] = []
    n_detected = 0
    n_recovered = 0
    n_unrecovered = 0

    for event in events:
        start_idx = event["start_idx"]
        end_idx = event["end_idx"]

        # Find first model overload prediction within this event
        detected_idx = -1
        for idx in range(start_idx, end_idx + 1):
            if y_pred_seq[idx] == 1:
                detected_idx = idx
                break

        if detected_idx == -1:
            # Undetected event: does not enter recovery analysis
            continue

        n_detected += 1

        # Search forward for first 'Optimal' state
        recovered = False
        for idx in range(detected_idx + 1, len(states)):
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
    }

    col_names = {
        "profile": "Profile",
        "model": "Model",
        "auc": "AUC",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "n_samples": "$N$",
        "n_positive": "$N_{pos}$",
        "n_negative": "$N_{neg}$",
        "aggregation_type": "Aggregation Type",
        "metric": "Metric",
        "value": "Value",
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
# Public API: Evaluation Driver & File Exporter
# ---------------------------------------------------------------------------

def create_summary_tables(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Write evaluation metrics dataframes to CSV and LaTeX files.

    Saves:
      results/tables/profile_metrics.csv & .tex
      results/tables/aggregate_metrics.csv & .tex
      results/tables/recovery_metrics.csv & .tex
      results/tables/state_statistics.csv & .tex
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


def evaluate_models(
    models: Dict[str, Any],
    features: Tuple[np.ndarray, np.ndarray, pd.DataFrame],
    config: Config = default_config,
    sim_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end evaluation benchmark.

    Evaluates predictions from all models across five learner profiles.
    Generates profile-level metrics, aggregates, and recovery/event metrics.

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
    # Standardized prediction rows list
    pred_rows: List[Dict[str, Any]] = []

    # Events and prediction sequences per (profile, model) for event/recovery evaluation
    profile_event_data: Dict[str, List[Dict[str, Any]]] = {}
    profile_pred_seqs: Dict[str, Dict[str, np.ndarray]] = {}

    for profile, df_profile in sim_data.items():
        N = len(df_profile)
        y_true_full = build_overload_target(df_profile).to_numpy()
        valid_mask = ~np.isnan(y_true_full)

        # Extract overload events (based on state sequence)
        events = extract_overload_events(df_profile["state"], profile)
        profile_event_data[profile] = events
        profile_pred_seqs[profile] = {}

        # ── Rule-Based CLSI ───────────────────────────────────────────────
        rb_prob = rule_based.predict_proba(df_profile)[:, 1]
        rb_pred = rule_based.predict(df_profile)
        profile_pred_seqs[profile]["rule_based_clsi"] = rb_pred

        for idx in range(N):
            if valid_mask[idx]:
                pred_rows.append(
                    {
                        "profile": profile,
                        "interaction_id": int(df_profile["interaction_id"].iloc[idx]),
                        "model": "rule_based_clsi",
                        "y_true": int(y_true_full[idx]),
                        "y_pred": int(rb_pred[idx]),
                        "y_prob": float(rb_prob[idx]),
                    }
                )

        # ── BKT ───────────────────────────────────────────────────────────
        bkt_prob = bkt.predict_proba(df_profile)[:, 1]
        bkt_pred = bkt.predict(df_profile)
        profile_pred_seqs[profile]["bkt"] = bkt_pred

        for idx in range(N):
            if valid_mask[idx]:
                pred_rows.append(
                    {
                        "profile": profile,
                        "interaction_id": int(df_profile["interaction_id"].iloc[idx]),
                        "model": "bkt",
                        "y_true": int(y_true_full[idx]),
                        "y_pred": int(bkt_pred[idx]),
                        "y_prob": float(bkt_prob[idx]),
                    }
                )

        # ── CLSI-Adapt (OOF) ──────────────────────────────────────────────
        res = clsi_adapt.get_result(profile)
        oof_df = res.oof_predictions

        # Map OOF predictions back to N-length sequence for event/recovery
        adapt_pred_seq = np.zeros(N, dtype=int)
        for _, row in oof_df.iterrows():
            pos = int(row["row_position"])
            adapt_pred_seq[pos] = int(row["y_pred"])
            pred_rows.append(
                {
                    "profile": profile,
                    "interaction_id": int(row["interaction_id"]),
                    "model": "clsi_adapt",
                    "y_true": int(row["y_true"]),
                    "y_pred": int(row["y_pred"]),
                    "y_prob": float(row["y_prob"]),
                }
            )
        profile_pred_seqs[profile]["clsi_adapt"] = adapt_pred_seq

    pred_df = pd.DataFrame(pred_rows)

    # ── 3. Profile-level classification metrics ──────────────────────────
    model_names = ["clsi_adapt", "rule_based_clsi", "bkt"]
    profile_names = list(sim_data.keys())

    profile_metrics_rows: List[Dict[str, Any]] = []

    for profile in profile_names:
        for model in model_names:
            sub = pred_df[(pred_df["profile"] == profile) & (pred_df["model"] == model)]
            if sub.empty:
                continue

            y_true_sub = sub["y_true"].to_numpy()
            y_prob_sub = sub["y_prob"].to_numpy()
            y_pred_sub = sub["y_pred"].to_numpy()

            metrics = compute_classification_metrics(y_true_sub, y_prob_sub, y_pred_sub)
            n_pos = int(np.sum(y_true_sub == 1))
            n_neg = int(np.sum(y_true_sub == 0))

            profile_metrics_rows.append(
                {
                    "profile": profile,
                    "model": model,
                    "auc": metrics["auc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "n_samples": len(y_true_sub),
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                }
            )

    profile_metrics_df = pd.DataFrame(profile_metrics_rows)

    # ── 4. Aggregate metrics (Pooled & Across-profile summary) ────────────
    agg_rows: List[Dict[str, Any]] = []

    for model in model_names:
        # A. Pooled (combine prediction records first)
        model_sub = pred_df[pred_df["model"] == model]
        if not model_sub.empty:
            metrics_pooled = compute_classification_metrics(
                model_sub["y_true"].to_numpy(),
                model_sub["y_prob"].to_numpy(),
                model_sub["y_pred"].to_numpy(),
            )
            for key, val in metrics_pooled.items():
                agg_rows.append(
                    {
                        "aggregation_type": "pooled",
                        "model": model,
                        "metric": key,
                        "value": val,
                    }
                )

        # B. Across-profile statistics (Mean & Std of profile-level results)
        model_pm = profile_metrics_df[profile_metrics_df["model"] == model]
        if not model_pm.empty:
            for metric in ["auc", "precision", "recall", "f1"]:
                vals = model_pm[metric].to_numpy()
                # Exclude NaNs (e.g. if AUC was undefined in one profile)
                valid_vals = vals[~np.isnan(vals)]

                if len(valid_vals) > 0:
                    mean_val = float(np.mean(valid_vals))
                    std_val = float(np.std(valid_vals))
                else:
                    mean_val = float("nan")
                    std_val = float("nan")

                agg_rows.append(
                    {
                        "aggregation_type": "mean_across_profiles",
                        "model": model,
                        "metric": metric,
                        "value": mean_val,
                    }
                )
                agg_rows.append(
                    {
                        "aggregation_type": "std_across_profiles",
                        "model": model,
                        "metric": metric,
                        "value": std_val,
                    }
                )

    aggregate_metrics_df = pd.DataFrame(agg_rows)

    # ── 5. Recovery & Event Metrics ───────────────────────────────────────
    rec_rows: List[Dict[str, Any]] = []

    for profile in profile_names:
        events = profile_event_data[profile]
        df_profile = sim_data[profile]

        for model in model_names:
            y_pred_seq = profile_pred_seqs[profile][model]

            ev_metrics = compute_event_metrics(events, y_pred_seq)
            rec_metrics = compute_recovery_metrics(events, y_pred_seq, df_profile["state"])

            rec_rows.append(
                {
                    "profile": profile,
                    "model": model,
                    "n_events": len(events),
                    "n_detected": rec_metrics["n_detected"],
                    "n_recovered": rec_metrics["n_recovered"],
                    "n_unrecovered": rec_metrics["n_unrecovered"],
                    "event_recall": ev_metrics["event_recall"],
                    "mean_recovery_time": rec_metrics["mean_recovery_time"],
                    "median_recovery_time": rec_metrics["median_recovery_time"],
                }
            )

    recovery_metrics_df = pd.DataFrame(rec_rows)

    # ── 6. State Statistics ───────────────────────────────────────────────
    state_statistics_df = compute_state_statistics(sim_data)

    # Compile result package
    eval_results = {
        "profile_metrics": profile_metrics_df,
        "aggregate_metrics": aggregate_metrics_df,
        "recovery_metrics": recovery_metrics_df,
        "state_statistics": state_statistics_df,
    }

    # Write output files programmatically
    create_summary_tables(eval_results, output_dir=config.tables_dir)

    return eval_results
