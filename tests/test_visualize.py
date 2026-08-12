"""Tests for the visualization module (src/visualize.py).

Covers:
1.  Figure 1 (Architecture diagram generation).
2.  Figure 2 (ROC curves across profiles & pooled, handling single-class edge cases).
3.  Figure 3 (SHAP beeswarm plot on small fitted model).
4.  Figure 4 (Causal learning curve evaluation & plot).
5.  Figure 5 (Model benchmark comparison bar charts).
6.  Master driver `generate_plots` end-to-end execution.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.config import Config
from src.visualize import (
    plot_architecture,
    plot_roc_curves,
    plot_shap_summary,
    plot_learning_curve,
    plot_model_comparison,
    generate_plots,
)
from src.models import CLSIAdaptModel, RuleBasedCLSIModel, BKTModel


class TestVisualize(unittest.TestCase):
    """Test suite for publication figure generation functions."""

    def test_figure1_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_pdf = Path(tmp_dir) / "figure1_architecture.pdf"
            plot_architecture(out_pdf)
            
            self.assertTrue(out_pdf.exists(), "Architecture PDF was not created.")
            self.assertGreater(out_pdf.stat().st_size, 0, "Architecture PDF is empty.")

    def test_figure2_roc_curves(self) -> None:
        pred_rows = [
            {"profile": "average", "model": "clsi_adapt", "y_true": 0, "y_prob": 0.1, "y_pred": 0},
            {"profile": "average", "model": "clsi_adapt", "y_true": 1, "y_prob": 0.8, "y_pred": 1},
            {"profile": "average", "model": "rule_based_clsi", "y_true": 0, "y_prob": 0.3, "y_pred": 0},
            {"profile": "average", "model": "rule_based_clsi", "y_true": 1, "y_prob": 0.6, "y_pred": 1},
            {"profile": "average", "model": "bkt", "y_true": 0, "y_prob": 0.4, "y_pred": 0},
            {"profile": "average", "model": "bkt", "y_true": 1, "y_prob": 0.5, "y_pred": 0},
        ]
        pred_df = pd.DataFrame(pred_rows)
        results = {"predictions": pred_df}

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_pdf = Path(tmp_dir) / "figure2_roc_curves.pdf"
            plot_roc_curves(results, out_pdf)
            
            self.assertTrue(out_pdf.exists(), "ROC PDF was not created.")
            self.assertGreater(out_pdf.stat().st_size, 0, "ROC PDF is empty.")

    def test_figure3_shap_summary(self) -> None:
        from tests.test_clsi_adapt import _make_overload_df
        df = _make_overload_df(n=80, profile="average")
        sim_data = {"average": df}

        import src.models.clsi_adapt
        src.models.clsi_adapt.N_ESTIMATORS = 10

        clsi_adapt = CLSIAdaptModel(seed=42)
        clsi_adapt.fit(sim_data)
        models = {"clsi_adapt": clsi_adapt}

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_pdf = Path(tmp_dir) / "figure3_shap_summary.pdf"
            plot_shap_summary(models, sim_data, out_pdf, profile="average")
            
            self.assertTrue(out_pdf.exists(), "SHAP PDF was not created.")
            self.assertGreater(out_pdf.stat().st_size, 0, "SHAP PDF is empty.")

    def test_figure4_learning_curve(self) -> None:
        from tests.test_clsi_adapt import _make_overload_df
        df = _make_overload_df(n=120, profile="average")
        sim_data = {"average": df}

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_pdf = Path(tmp_dir) / "figure4_learning_curve.pdf"
            plot_learning_curve(sim_data, out_pdf, profile="average", seed=42)
            
            self.assertTrue(out_pdf.exists(), "Learning curve PDF was not created.")
            self.assertGreater(out_pdf.stat().st_size, 0, "Learning curve PDF is empty.")

    def test_figure5_model_comparison(self) -> None:
        pm_rows = [
            {"profile": "average", "model": "clsi_adapt", "auc": 0.85, "precision": 0.80, "recall": 0.75},
            {"profile": "average", "model": "rule_based_clsi", "auc": 0.65, "precision": 0.60, "recall": 0.55},
            {"profile": "average", "model": "bkt", "auc": 0.55, "precision": 0.50, "recall": 0.45},
        ]
        pm_df = pd.DataFrame(pm_rows)
        results = {"profile_metrics": pm_df}

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_pdf = Path(tmp_dir) / "figure5_model_comparison.pdf"
            plot_model_comparison(results, out_pdf)
            
            self.assertTrue(out_pdf.exists(), "Comparison PDF was not created.")
            self.assertGreater(out_pdf.stat().st_size, 0, "Comparison PDF is empty.")

    def test_generate_plots_end_to_end(self) -> None:
        from tests.test_clsi_adapt import _make_overload_df
        df = _make_overload_df(n=80, profile="average")
        sim_data = {"average": df}

        import src.models.clsi_adapt
        src.models.clsi_adapt.N_ESTIMATORS = 10

        from src.feature_engineering import prepare_dataset
        X, y, meta = prepare_dataset(df, window_size=5)
        features = (X, y, meta)

        clsi_adapt = CLSIAdaptModel(seed=42)
        models = {
            "clsi_adapt": clsi_adapt,
            "rule_based_clsi": RuleBasedCLSIModel(),
            "bkt": BKTModel(),
        }

        from src.evaluation import evaluate_models
        config = Config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            config.figures_dir = Path(tmp_dir)
            config.tables_dir = Path(tmp_dir)

            results = evaluate_models(models, features, config, sim_data=sim_data)
            generate_plots(results, config=config, sim_data=sim_data, models=models, features=features)

            # Check that all 5 PDF figures exist
            self.assertTrue((Path(tmp_dir) / "figure1_architecture.pdf").exists())
            self.assertTrue((Path(tmp_dir) / "figure2_roc_curves.pdf").exists())
            self.assertTrue((Path(tmp_dir) / "figure3_shap_summary.pdf").exists())
            self.assertTrue((Path(tmp_dir) / "figure4_learning_curve.pdf").exists())
            self.assertTrue((Path(tmp_dir) / "figure5_model_comparison.pdf").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
