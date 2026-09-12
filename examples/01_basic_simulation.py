"""Example 01: Basic Simulation

Demonstrates core BehaviorSim simulation:
1. Defining discrete States.
2. Specifying state-dependent FeatureDistributions.
3. Defining Markovian state transition dynamics.
4. Generating reproducible synthetic sequences.
5. Inspecting ground-truth hidden states and emitted features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from behaviorsim import FeatureDistribution, Profile, Simulator, State


def main() -> None:
    print("=== BehaviorSim: Basic Simulation Example ===")

    # 1. Declare discrete states
    states = [
        State("Exploration", description="Exploring options"),
        State("Engagement", description="Actively engaged"),
        State("Checkout", description="Conversion state"),
    ]

    # 2. Configure state-conditioned feature emissions
    state_emissions = {
        "Exploration": {
            "dwell_time": FeatureDistribution("exponential", {"scale": 15.0}),
            "actions_count": FeatureDistribution("poisson", {"lam": 3.0}),
        },
        "Engagement": {
            "dwell_time": FeatureDistribution("normal", {"mean": 45.0, "std": 10.0}),
            "actions_count": FeatureDistribution("poisson", {"lam": 8.0}),
        },
        "Checkout": {
            "dwell_time": FeatureDistribution("normal", {"mean": 90.0, "std": 15.0}),
            "actions_count": FeatureDistribution("uniform_discrete", {"low": 1, "high": 4}),
        },
    }

    # 3. Transition matrix: P(state_t -> state_{t+1})
    transition_matrix = np.array([
        [0.60, 0.35, 0.05],  # From Exploration
        [0.10, 0.70, 0.20],  # From Engagement
        [0.00, 0.00, 1.00],  # From Checkout (terminal absorbing)
    ])

    # 4. Build Profile and Simulator
    profile = Profile(
        name="StandardUser",
        state_emissions=state_emissions,
        transition_matrix=transition_matrix,
    )

    sim = Simulator(
        states=states,
        profile=profile,
        initial_state="Exploration",
    )

    # 5. Generate reproducible sequences
    traces: pd.DataFrame = sim.generate(
        num_interactions=10,
        num_sequences=3,
        seed=42,
    )

    print("\nGenerated traces (head):")
    print(traces[["sequence_id", "interaction_id", "state", "dwell_time", "actions_count"]].head(10))

    print("\nState distribution across all interactions:")
    print(traces["state"].value_counts(normalize=True))

    print("\nGround-truth states are explicitly tracked in the 'state' column without latent ambiguity.")
    print("Basic simulation completed successfully.")


if __name__ == "__main__":
    main()
