"""Example 06: Visualization and Diagnostics

Demonstrates the publication-ready visualization layer:
1. State occupancy plots (counts and normalized proportions).
2. Discrete state trajectories across sequence interactions.
3. Transition matrix heatmap.
4. Empirical vs synthetic feature distribution comparison.
5. Calibration validation summary dashboard.
6. Clean headless figure export via save_figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and src to sys.path so examples run out of the box without prior installation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tempfile
import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import pandas as pd

from behaviorsim import Simulator
from behaviorsim.calibration import CalibrationData, fit_profile, validate_calibration
from behaviorsim.visualization import (
    plot_calibration_summary,
    plot_feature_comparison,
    plot_state_occupancy,
    plot_state_trajectory,
    plot_transition_matrix,
    save_figure,
)


def main() -> None:
    print("=== BehaviorSim: Visualization & Diagnostics Example ===")

    # 1. Generate sample empirical and synthetic data
    sim = Simulator.from_preset("education")
    emp_df: pd.DataFrame = sim.generate(num_interactions=15, num_sequences=5, seed=42)
    syn_df: pd.DataFrame = sim.generate(num_interactions=15, num_sequences=5, seed=999)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # 2. State Occupancy
        fig_occ, ax_occ = plot_state_occupancy(emp_df, state_column="state", normalize=True)
        path_occ = save_figure(fig_occ, output_dir / "state_occupancy.png", close=True)
        print(f"Saved: {path_occ.name}")

        # 3. Discrete State Trajectories
        fig_traj, ax_traj = plot_state_trajectory(emp_df, max_sequences=3)
        path_traj = save_figure(fig_traj, output_dir / "state_trajectories.png", close=True)
        print(f"Saved: {path_traj.name}")

        # 4. Transition Matrix Heatmap
        fig_trans, ax_trans = plot_transition_matrix(emp_df)
        path_trans = save_figure(fig_trans, output_dir / "transition_matrix.png", close=True)
        print(f"Saved: {path_trans.name}")

        # 5. Feature Comparison (Empirical vs Synthetic with shared bins)
        fig_comp, ax_comp = plot_feature_comparison(emp_df, syn_df, feature="nrt", bins=15)
        path_comp = save_figure(fig_comp, output_dir / "feature_comparison.png", close=True)
        print(f"Saved: {path_comp.name}")

        # 6. Calibration Diagnostics Dashboard
        states = ["Optimal", "Overload", "Underload"]
        cal_data = CalibrationData(emp_df, states=states, numeric_features=["nrt"])
        profile = fit_profile(cal_data, name="CalibratedEducation", transition_smoothing=0.01)
        report = validate_calibration(cal_data, syn_df, profile=profile, numeric_features=["nrt"])

        fig_dash, axes_dash = plot_calibration_summary(report)
        path_dash = save_figure(fig_dash, output_dir / "calibration_dashboard.png", close=True)
        print(f"Saved: {path_dash.name}")

    print("\nVisualization and diagnostics executed cleanly in headless environment.")


if __name__ == "__main__":
    main()
