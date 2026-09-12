"""Example 02: Domain Presets

Demonstrates out-of-the-box domain preset simulations:
1. Education (cognitive load states, accuracy, response times)
2. Mobile App (user lifecycle, engagement, churn)
3. Healthcare (synthetic adherence modeling; RESEARCH BENCHMARK ONLY)
4. Finance (synthetic transaction risk patterns; BENCHMARK ONLY)

DISCLAIMERS:
- Healthcare: Synthetic evaluation model only. NOT clinically validated and NOT for medical diagnosis or treatment decisions.
- Finance: Synthetic evaluation model only. NOT for financial, investment, or fraud prevention decisions.
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

import pandas as pd
from behaviorsim import Simulator
from behaviorsim.presets import list_presets


def main() -> None:
    print("=== BehaviorSim: Domain Presets Example ===")
    print("Available presets:", list_presets())

    # 1. Education Preset
    print("\n--- 1. Education Preset ---")
    sim_edu = Simulator.from_preset("education", profile="fast_accurate")
    edu_traces: pd.DataFrame = sim_edu.generate(num_interactions=10, num_sequences=2, seed=42)
    print("Education columns:", edu_traces.columns.tolist())
    print("Education states observed:", edu_traces["state"].unique().tolist())
    print(edu_traces[["sequence_id", "interaction_id", "state", "accuracy", "nrt"]].head(4))

    # 2. Mobile App Preset
    print("\n--- 2. Mobile App Preset ---")
    sim_mobile = Simulator.from_preset("mobile_app", profile="power_user")
    mobile_traces: pd.DataFrame = sim_mobile.generate(num_interactions=10, num_sequences=2, seed=42)
    print("Mobile App columns:", mobile_traces.columns.tolist())
    print("Mobile states observed:", mobile_traces["state"].unique().tolist())
    print(mobile_traces[["sequence_id", "interaction_id", "state", "scroll_depth", "session_time_seconds"]].head(4))

    # 3. Healthcare Preset (Synthetic Adherence Telemetry)
    print("\n--- 3. Healthcare Preset ---")
    print("> DISCLAIMER: Synthetic evaluation model only. Not for clinical diagnosis.")
    sim_health = Simulator.from_preset("healthcare", profile="chronic_risk")
    health_traces: pd.DataFrame = sim_health.generate(num_interactions=10, num_sequences=2, seed=42)
    print("Healthcare columns:", health_traces.columns.tolist())
    print("Healthcare states observed:", health_traces["state"].unique().tolist())
    print(health_traces[["sequence_id", "interaction_id", "state", "heart_rate_bpm", "systolic_bp"]].head(4))

    # 4. Finance Preset (Synthetic Transaction Dynamics)
    print("\n--- 4. Finance Preset ---")
    print("> DISCLAIMER: Synthetic evaluation model only. Not for financial/investment decisions.")
    sim_fin = Simulator.from_preset("finance", profile="active_trader")
    fin_traces: pd.DataFrame = sim_fin.generate(num_interactions=10, num_sequences=2, seed=42)
    print("Finance columns:", fin_traces.columns.tolist())
    print("Finance states observed:", fin_traces["state"].unique().tolist())
    print(fin_traces[["sequence_id", "interaction_id", "state", "portfolio_value", "trade_volume"]].head(4))

    print("\nPresets execution completed successfully.")


if __name__ == "__main__":
    main()
