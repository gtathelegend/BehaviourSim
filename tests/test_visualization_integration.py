"""End-to-end integration tests for BehaviorSim visualization and analysis layer."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from behaviorsim.calibration.clustering import cluster_profiles
from behaviorsim.calibration.fitter import CalibrationData, fit_profile
from behaviorsim.calibration.validator import validate_calibration
from behaviorsim.core.simulator import Simulator
from behaviorsim.feature_engineering import build_behavioral_features
from behaviorsim.visualization import (
    plot_calibration_summary,
    plot_categorical_comparison,
    plot_feature_comparison,
    plot_feature_distribution,
    plot_feature_trajectory,
    plot_numeric_comparison,
    plot_occupancy_comparison,
    plot_state_occupancy,
    plot_state_trajectory,
    plot_transition_comparison,
    plot_transition_matrix,
    save_figure,
)


class TestVisualizationIntegration:
    """End-to-end pipeline integration tests."""

    def test_preset_simulation_visualization_pipeline(self):
        """Pipeline 1: Simulator.from_preset -> generate -> state and trajectory visualizations."""
        sim = Simulator.from_preset("education")
        traces = sim.generate(num_interactions=10, num_sequences=5, seed=42)

        # State occupancy
        fig1, ax1 = plot_state_occupancy(traces, state_column="state")
        assert fig1 is not None
        plt.close(fig1)

        # State trajectory
        fig2, ax2 = plot_state_trajectory(traces, max_sequences=3)
        assert fig2 is not None
        plt.close(fig2)

        # Transition matrix
        fig3, ax3 = plot_transition_matrix(traces)
        assert fig3 is not None
        plt.close(fig3)

    def test_feature_engineering_visualization_pipeline(self):
        """Pipeline 2: Simulator -> build_behavioral_features -> feature trajectory and distribution."""
        sim = Simulator.from_preset("education")
        raw_traces = sim.generate(num_interactions=8, num_sequences=4, seed=101)

        enriched = build_behavioral_features(
            raw_traces,
            numeric_columns=["nrt"],
            rolling_windows=[3],
        )

        assert "nrt_rolling_mean_w3" in enriched.columns

        # Feature trajectory on rolling feature
        fig1, ax1 = plot_feature_trajectory(
            enriched,
            feature="nrt_rolling_mean_w3",
            max_sequences=3,
        )
        assert fig1 is not None
        plt.close(fig1)

        # Feature distribution
        fig2, ax2 = plot_feature_distribution(
            enriched,
            feature="nrt",
            state_column="state",
        )
        assert fig2 is not None
        plt.close(fig2)

    def test_calibration_diagnostics_pipeline(self):
        """Pipeline 3: CalibrationData -> fit_profile -> Simulator -> validate_calibration -> diagnostics."""
        from behaviorsim.core.state import State

        state_names = ["Attentive", "Distracted"]
        states = [State(s) for s in state_names]
        df_emp = pd.DataFrame({
            "sequence_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "state": ["Attentive", "Attentive", "Distracted", "Attentive", "Distracted", "Distracted", "Attentive", "Distracted"],
            "score": [85.0, 90.0, 45.0, 80.0, 50.0, 40.0, 75.0, 55.0],
        })

        cal_data = CalibrationData(
            data=df_emp,
            states=state_names,
            numeric_features=["score"],
        )

        profile = fit_profile(
            cal_data,
            name="FittedStudent",
            transition_smoothing=0.01,
        )

        sim = Simulator(states=states, profile=profile)
        syn_traces = sim.generate(num_interactions=10, num_sequences=5, seed=123)

        report = validate_calibration(
            cal_data,
            syn_traces,
            profile=profile,
            numeric_features=["score"],
            thresholds={"state_tvd <= 0.5": lambda r: r.state_occupancy.total_variation_distance <= 0.5},
        )

        # Plot calibration summary dashboard
        fig_sum, axes_sum = plot_calibration_summary(report)
        assert fig_sum is not None
        assert axes_sum.shape == (2, 2)
        plt.close(fig_sum)

        # Plot occupancy comparison
        fig_occ, ax_occ = plot_occupancy_comparison(report)
        assert fig_occ is not None
        plt.close(fig_occ)

        # Plot transition comparison
        fig_trans, axes_trans = plot_transition_comparison(report)
        assert len(axes_trans) == 3
        plt.close(fig_trans)

        # Plot numeric comparison
        fig_num, ax_num = plot_numeric_comparison(report)
        assert fig_num is not None
        plt.close(fig_num)

    def test_multi_profile_visualization_pipeline(self):
        """Pipeline 4: cluster_profiles -> multi-profile traces -> profile-grouped state occupancy."""
        df_cluster = pd.DataFrame({
            "sequence_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "state": ["A", "B", "A", "B", "B", "A", "B", "A"],
            "val": [10.0, 20.0, 11.0, 21.0, 90.0, 95.0, 91.0, 94.0],
        })
        cal_data = CalibrationData(
            data=df_cluster,
            states=["A", "B"],
            numeric_features=["val"],
        )

        clustering_result = cluster_profiles(
            cal_data,
            n_profiles=2,
            seed=42,
        )
        assert len(clustering_result.profiles) == 2

        # Build traces with assigned profiles
        df_multi = df_cluster.copy()
        df_multi["profile"] = ["Cluster_0" if s in [1, 2] else "Cluster_1" for s in df_multi["sequence_id"]]

        fig, ax = plot_state_occupancy(df_multi, profile_column="profile")
        assert fig is not None
        assert ax.get_legend() is not None
        plt.close(fig)


class TestFigureExport:
    """Tests for save_figure utility."""

    def test_save_figure_formats(self, tmp_path: Path):
        """Verify PNG, PDF, and SVG export to explicit paths."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        # PNG
        png_path = tmp_path / "exports" / "test_plot.png"
        res_png = save_figure(fig, png_path)
        assert res_png.exists()
        assert res_png.stat().st_size > 0

        # PDF
        pdf_path = tmp_path / "exports" / "test_plot.pdf"
        res_pdf = save_figure(fig, pdf_path)
        assert res_pdf.exists()
        assert res_pdf.stat().st_size > 0

        # SVG
        svg_path = tmp_path / "exports" / "test_plot.svg"
        res_svg = save_figure(fig, svg_path)
        assert res_svg.exists()
        assert res_svg.stat().st_size > 0

        # Figure remains open since close=False
        assert plt.fignum_exists(fig.number)

        # Save with close=True
        save_figure(fig, tmp_path / "test_closed.png", close=True)
        assert not plt.fignum_exists(fig.number)

    def test_save_figure_errors(self, tmp_path: Path):
        """Verify error on invalid figure type or missing extension."""
        with pytest.raises(TypeError, match="Expected matplotlib Figure"):
            save_figure("not_a_figure", tmp_path / "out.png")  # type: ignore

        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="must include a file extension"):
            save_figure(fig, tmp_path / "no_extension")
        plt.close(fig)

    def test_save_figure_tuple_support(self, tmp_path: Path):
        """Verify save_figure directly accepts (fig, ax) tuple returned by plotting functions."""
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        tuple_input = (fig, ax)

        # PNG via tuple with close=False
        png_out = tmp_path / "tuple_plot.png"
        saved_png = save_figure(tuple_input, png_out, close=False)
        assert saved_png.exists()
        assert saved_png.stat().st_size > 0
        assert plt.fignum_exists(fig.number)

        # PDF via tuple with close=False
        pdf_out = tmp_path / "tuple_plot.pdf"
        saved_pdf = save_figure(tuple_input, pdf_out, close=False)
        assert saved_pdf.exists()
        assert saved_pdf.stat().st_size > 0
        assert plt.fignum_exists(fig.number)

        # SVG via tuple with close=True
        svg_out = tmp_path / "tuple_plot.svg"
        saved_svg = save_figure(tuple_input, svg_out, close=True)
        assert saved_svg.exists()
        assert saved_svg.stat().st_size > 0
        assert not plt.fignum_exists(fig.number)
