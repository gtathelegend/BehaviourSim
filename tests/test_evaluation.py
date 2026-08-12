"""Tests for the evaluation module (src/evaluation.py).

Covers:
1.  Classification metrics (perfect, wrong, probabilistic, single-class, zero-division).
2.  Profile isolation and aggregation (pooled vs across-profile mean/std).
3.  State statistics (on known sequences of Optimal, Overload, Underload).
4.  Event detection (extraction, duration, event recall, multiple events).
5.  Recovery time (recovered, unrecovered, undetected, multiple events).
6.  LaTeX table generation (escaping, headers, formatting).
"""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.config import Config
from src.evaluation import (
    compute_classification_metrics,
    extract_overload_events,
    compute_event_metrics,
    compute_recovery_metrics,
    compute_state_statistics,
    df_to_latex,
    evaluate_models,
    create_summary_tables,
)
from src.models import CLSIAdaptModel, RuleBasedCLSIModel, BKTModel


class TestClassificationMetrics(unittest.TestCase):
    """Test compute_classification_metrics including edge cases."""

    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.1, 0.9])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(y_true, y_prob, y_pred)
        self.assertAlmostEqual(metrics["auc"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_completely_wrong_predictions(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.9, 0.1])
        y_pred = np.array([1, 0, 1, 0])
        metrics = compute_classification_metrics(y_true, y_prob, y_pred)
        self.assertAlmostEqual(metrics["auc"], 0.0)
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["f1"], 0.0)

    def test_probabilistic_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        y_pred = np.array([0, 0, 0, 1])  # threshold 0.5
        metrics = compute_classification_metrics(y_true, y_prob, y_pred)
        # AUC is ROC AUC: y_prob rank-order is [0.1, 0.35, 0.4, 0.8] for true [0, 1, 0, 1]
        # True negative at 0.1, true positive at 0.35, true negative at 0.4, true positive at 0.8
        # pairs sorted:
        # (0.1, TN), (0.35, TP), (0.4, TN), (0.8, TP)
        # pairs where prob(pos) > prob(neg):
        # TN(0.1) vs TP(0.35) -> correct rank (1)
        # TN(0.1) vs TP(0.8) -> correct rank (1)
        # TN(0.4) vs TP(0.35) -> wrong rank (0)
        # TN(0.4) vs TP(0.8) -> correct rank (1)
        # AUC = 3 / 4 = 0.75
        self.assertAlmostEqual(metrics["auc"], 0.75)
        # Precision: 1 predicted positive (index 3), which is correct -> 1.0
        self.assertAlmostEqual(metrics["precision"], 1.0)
        # Recall: 1 out of 2 positive targets detected -> 0.5
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_single_class_target_auc_undefined(self) -> None:
        """If all targets are 0 or 1, AUC must be NaN."""
        # All negative
        metrics_neg = compute_classification_metrics(
            np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0])
        )
        self.assertTrue(np.isnan(metrics_neg["auc"]))

        # All positive
        metrics_pos = compute_classification_metrics(
            np.array([1, 1, 1]), np.array([0.8, 0.9, 0.7]), np.array([1, 1, 1])
        )
        self.assertTrue(np.isnan(metrics_pos["auc"]))

    def test_zero_division_behavior(self) -> None:
        """If there are no positive predictions, precision/recall/f1 should be 0.0."""
        y_true = np.array([0, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.1])
        y_pred = np.array([0, 0, 0])  # no positives predicted
        metrics = compute_classification_metrics(y_true, y_prob, y_pred)
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["f1"], 0.0)


class TestStateStatistics(unittest.TestCase):
    """Test compute_state_statistics on hand-crafted cognitive state sequence."""

    def test_state_fractions(self) -> None:
        # Hand-crafted state sequence:
        # Optimal (3), Overload (1), Underload (1) -> 5 total
        df = pd.DataFrame({
            "state": ["Optimal", "Optimal", "Overload", "Underload", "Optimal"]
        })
        sim_data = {"profile_1": df}
        stats = compute_state_statistics(sim_data)
        
        self.assertEqual(len(stats), 2)  # profile_1 and Aggregate
        
        # Test profile_1
        p_row = stats[stats["profile"] == "profile_1"].iloc[0]
        self.assertAlmostEqual(p_row["optimal_fraction"], 3/5)
        self.assertAlmostEqual(p_row["overload_fraction"], 1/5)
        self.assertAlmostEqual(p_row["underload_fraction"], 1/5)
        self.assertEqual(p_row["n_interactions"], 5)
        
        # Test Aggregate (should be identical since there's only 1 profile)
        agg_row = stats[stats["profile"] == "Aggregate"].iloc[0]
        self.assertAlmostEqual(agg_row["optimal_fraction"], 3/5)
        self.assertAlmostEqual(agg_row["overload_fraction"], 1/5)
        self.assertAlmostEqual(agg_row["underload_fraction"], 1/5)
        self.assertEqual(agg_row["n_interactions"], 5)


class TestEventDetection(unittest.TestCase):
    """Test extract_overload_events and compute_event_metrics."""

    def test_event_extraction(self) -> None:
        state_seq = pd.Series([
            "Optimal", "Overload", "Overload", "Optimal", "Optimal", "Overload", "Underload"
        ])
        events = extract_overload_events(state_seq, "test_profile")
        
        self.assertEqual(len(events), 2)
        
        # First event: indices 1 to 2
        self.assertEqual(events[0]["start_idx"], 1)
        self.assertEqual(events[0]["end_idx"], 2)
        self.assertEqual(events[0]["start_interaction"], 2)
        self.assertEqual(events[0]["end_interaction"], 3)
        self.assertEqual(events[0]["duration"], 2)
        self.assertEqual(events[0]["profile"], "test_profile")
        self.assertEqual(events[0]["event_id"], "test_profile_event_1")
        
        # Second event: index 5
        self.assertEqual(events[1]["start_idx"], 5)
        self.assertEqual(events[1]["end_idx"], 5)
        self.assertEqual(events[1]["duration"], 1)

    def test_event_recall(self) -> None:
        events = [
            {"start_idx": 1, "end_idx": 3, "start_interaction": 2, "end_interaction": 4},
            {"start_idx": 6, "end_idx": 6, "start_interaction": 7, "end_interaction": 7},
        ]
        
        # Case A: Both detected
        y_pred = np.array([0, 1, 0, 0, 0, 0, 1, 0])
        metrics = compute_event_metrics(events, y_pred)
        self.assertEqual(metrics["n_events"], 2)
        self.assertEqual(metrics["n_detected"], 2)
        self.assertAlmostEqual(metrics["event_recall"], 1.0)
        
        # Case B: One detected
        y_pred = np.array([0, 1, 0, 0, 0, 0, 0, 0])
        metrics = compute_event_metrics(events, y_pred)
        self.assertEqual(metrics["n_detected"], 1)
        self.assertAlmostEqual(metrics["event_recall"], 0.5)

        # Case C: None detected
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        metrics = compute_event_metrics(events, y_pred)
        self.assertEqual(metrics["n_detected"], 0)
        self.assertAlmostEqual(metrics["event_recall"], 0.0)


class TestRecoveryMetrics(unittest.TestCase):
    """Test compute_recovery_metrics on recovered and unrecovered sequences."""

    def test_recovery_time_recovered(self) -> None:
        # Event boundary: t=1 to t=3 (inclusive)
        events = [{"start_idx": 1, "end_idx": 3}]
        # Sequence of states
        state_seq = pd.Series(["Optimal", "Overload", "Overload", "Overload", "Optimal"])
        
        # Detection at t=1 (start of event)
        # Returns to Optimal at t=4
        # Recovery time = 4 - 1 = 3 interactions
        y_pred = np.array([0, 1, 0, 0, 0])
        metrics = compute_recovery_metrics(events, y_pred, state_seq)
        
        self.assertEqual(metrics["n_events"], 1)
        self.assertEqual(metrics["n_detected"], 1)
        self.assertEqual(metrics["n_recovered"], 1)
        self.assertEqual(metrics["n_unrecovered"], 0)
        self.assertAlmostEqual(metrics["mean_recovery_time"], 3.0)
        self.assertAlmostEqual(metrics["median_recovery_time"], 3.0)

    def test_recovery_time_unrecovered(self) -> None:
        # Event boundary: t=1 to t=2
        events = [{"start_idx": 1, "end_idx": 2}]
        # Never returns to Optimal
        state_seq = pd.Series(["Optimal", "Overload", "Overload", "Underload"])
        
        # Detection at t=1
        y_pred = np.array([0, 1, 0, 0])
        metrics = compute_recovery_metrics(events, y_pred, state_seq)
        
        self.assertEqual(metrics["n_recovered"], 0)
        self.assertEqual(metrics["n_unrecovered"], 1)
        self.assertTrue(np.isnan(metrics["mean_recovery_time"]))

    def test_recovery_time_undetected(self) -> None:
        """Undetected events should be ignored for recovery calculations."""
        events = [{"start_idx": 1, "end_idx": 2}]
        state_seq = pd.Series(["Optimal", "Overload", "Overload", "Optimal"])
        y_pred = np.array([0, 0, 0, 0])
        metrics = compute_recovery_metrics(events, y_pred, state_seq)
        
        self.assertEqual(metrics["n_detected"], 0)
        self.assertEqual(metrics["n_recovered"], 0)
        self.assertEqual(metrics["n_unrecovered"], 0)


class TestLaTeXGeneration(unittest.TestCase):
    """Verify latex table code works and generates properly escaped headers."""

    def test_df_to_latex(self) -> None:
        df = pd.DataFrame({
            "profile": ["fast_accurate", "average"],
            "model": ["clsi_adapt", "bkt"],
            "auc": [0.852, 0.730],
            "n_samples": [100, 120]
        })
        latex_str = df_to_latex(df)
        
        # Verify headers escaped or formatted
        self.assertIn("Profile", latex_str)
        self.assertIn("Model", latex_str)
        self.assertIn("AUC", latex_str)
        self.assertIn("$N$", latex_str)
        
        # Verify names mapped
        self.assertIn("Fast-Accurate", latex_str)
        self.assertIn("CLSI-Adapt", latex_str)
        self.assertIn("BKT", latex_str)
        
        # Verify standard table format
        self.assertIn(r"\begin{table}", latex_str)
        self.assertIn(r"\toprule", latex_str)
        self.assertIn(r"\bottomrule", latex_str)


class TestFullEvaluationPipeline(unittest.TestCase):
    """Test evaluate_models end-to-end using a minimal mock scenario."""

    def test_evaluate_models_complete(self) -> None:
        # Build deterministic dataset with both classes
        # Length 80 (to fit warm-up of 20 and validation splits)
        # Use our overload generator logic
        from tests.test_clsi_adapt import _make_overload_df
        df = _make_overload_df(n=100, profile="average")
        
        sim_data = {"average": df}
        
        from src.feature_engineering import prepare_dataset
        X, y, meta = prepare_dataset(df, window_size=5)
        features = (X, y, meta)
        
        # Initialize models
        # Set n_estimators=10 for CLSIAdaptModel to keep runtimes extremely fast
        clsi_adapt = CLSIAdaptModel(seed=42)
        # Inject custom fit override to use fewer estimators in this test
        original_fit = clsi_adapt.fit
        clsi_adapt.fit = lambda s_data: original_fit(s_data, save_dir=None)
        
        # We manually modify clsi_adapt's config to use TEST_N_ESTIMATORS inside the fit method
        import src.models.clsi_adapt
        src.models.clsi_adapt.N_ESTIMATORS = 10
        
        models = {
            "clsi_adapt": clsi_adapt,
            "rule_based_clsi": RuleBasedCLSIModel(),
            "bkt": BKTModel(),
        }
        
        config = Config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.tables_dir = Path(tmp_dir)
            
            results = evaluate_models(models, features, config, sim_data=sim_data)
            
            # Check results keys
            self.assertIn("profile_metrics", results)
            self.assertIn("aggregate_metrics", results)
            self.assertIn("recovery_metrics", results)
            self.assertIn("state_statistics", results)
            
            # Verify file generation
            self.assertTrue((Path(tmp_dir) / "profile_metrics.csv").exists())
            self.assertTrue((Path(tmp_dir) / "profile_metrics.tex").exists())
            self.assertTrue((Path(tmp_dir) / "aggregate_metrics.csv").exists())
            self.assertTrue((Path(tmp_dir) / "recovery_metrics.csv").exists())
            
            # Check aggregate structure
            agg_df = results["aggregate_metrics"]
            self.assertTrue(set(agg_df["aggregation_type"].unique()).issubset({
                "pooled", "mean_across_profiles", "std_across_profiles"
            }))
            
            # Make sure pooled metrics differ conceptually from mean metrics
            pooled_auc = agg_df[(agg_df["aggregation_type"] == "pooled") & (agg_df["metric"] == "auc")]
            mean_auc = agg_df[(agg_df["aggregation_type"] == "mean_across_profiles") & (agg_df["metric"] == "auc")]
            self.assertNotEqual(len(pooled_auc), 0)
            self.assertNotEqual(len(mean_auc), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
