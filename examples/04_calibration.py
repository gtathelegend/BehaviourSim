"""Example 04: Empirical Calibration and Validation

Demonstrates fitting a generative BehaviorSim model from observed empirical traces:
1. Wrapping empirical interaction data in CalibrationData.
2. Fitting an empirical transition matrix and parametric state emissions via fit_profile.
3. Generating synthetic traces using the calibrated Profile.
4. Validating calibration fidelity with validate_calibration.
5. Inspecting the comprehensive ValidationReport.

NOTE ON CALIBRATION:
BehaviorSim uses explicit proxy-based state identification, not unsupervised latent inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from behaviorsim import Simulator, State
from behaviorsim.calibration import (
    CalibrationData,
    fit_profile,
    validate_calibration,
)



def main() -> None:
    print("=== BehaviorSim: Calibration & Validation Example ===")

    # 1. Create mock observed empirical dataset
    np.random.seed(42)
    rows = []
    states = ["LowDemand", "HighDemand"]
    for seq in range(25):
        curr = "LowDemand"
        for step in range(15):
            # True empirical dynamics
            if curr == "LowDemand":
                next_state = "HighDemand" if np.random.rand() < 0.3 else "LowDemand"
                latency = np.random.normal(20.0, 4.0)
            else:
                next_state = "LowDemand" if np.random.rand() < 0.5 else "HighDemand"
                latency = np.random.normal(60.0, 10.0)

            rows.append({
                "sequence_id": seq,
                "interaction_index": step,
                "state": curr,
                "latency": latency,
            })
            curr = next_state

    empirical_df = pd.DataFrame(rows)

    # 2. Package in CalibrationData container
    cal_data = CalibrationData(
        data=empirical_df,
        states=states,
        state_column="state",
        sequence_column="sequence_id",
        numeric_features=["latency"],
    )

    # 3. Fit Profile from empirical data
    calibrated_profile = fit_profile(
        data=cal_data,
        name="CalibratedAgent",
        default_distribution="normal",
        transition_smoothing=0.01,
    )

    print("\nFitted Transition Matrix:")
    print(calibrated_profile.transition_matrix)

    # 4. Simulate synthetic counterpart traces
    sim = Simulator(
        states=[State(s) for s in states],
        profile=calibrated_profile,
    )
    synthetic_traces = sim.generate(
        num_interactions=15,
        num_sequences=25,
        seed=123,
    )

    # 5. Validate calibration fidelity
    report = validate_calibration(
        empirical_data=cal_data,
        synthetic_data=synthetic_traces,
        profile=calibrated_profile,
        numeric_features=["latency"],
        thresholds={
            "max_state_tvd": 0.15,
            "max_transition_mae": 0.10,
        },
    )

    print("\n--- Validation Report Summary ---")
    print(f"Structural Integrity Valid: {report.structural.is_valid}")
    print(f"State Occupancy TVD:       {report.state_occupancy.total_variation_distance:.4f}")
    print(f"Transition MAE:            {report.transition.mean_absolute_error:.4f}")
    print(f"Latency Wasserstein Dist:  {report.numeric_features['latency'].wasserstein_distance:.4f}")
    print(f"Latency KS Statistic:      {report.numeric_features['latency'].ks_statistic:.4f}")

    print("\nThreshold Verification Results:")
    for check_name, passed in report.threshold_results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check_name}")

    print(f"\nOverall Validation Status: {'PASSED' if report.is_valid else 'FAILED'}")
    print("Calibration workflow completed successfully.")


if __name__ == "__main__":
    main()
