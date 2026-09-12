"""Example 03: Multi-Profile Simulation

Demonstrates simulating multiple behavioral profiles within a single Simulator:
1. Constructing distinct behavioral profiles (Novice vs Expert).
2. Assigning mixture probabilities across profiles.
3. Generating sequences sampled according to the mixture distribution.
4. Verifying sequence-level profile assignments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from behaviorsim import FeatureDistribution, Profile, Simulator, State


def main() -> None:
    print("=== BehaviorSim: Multi-Profile Simulation Example ===")

    states = [
        State("Reading", description="Reading content"),
        State("Quiz", description="Solving quiz problems"),
        State("Review", description="Reviewing mistakes"),
    ]

    # Profile 1: Novice (lower quiz accuracy, longer reading)
    novice_emissions = {
        "Reading": {"time_spent": FeatureDistribution("normal", {"mean": 60.0, "std": 10.0})},
        "Quiz": {"time_spent": FeatureDistribution("normal", {"mean": 45.0, "std": 8.0})},
        "Review": {"time_spent": FeatureDistribution("normal", {"mean": 50.0, "std": 12.0})},
    }
    novice_transitions = np.array([
        [0.40, 0.40, 0.20],
        [0.10, 0.50, 0.40],
        [0.20, 0.50, 0.30],
    ])
    novice_profile = Profile(
        name="Novice",
        state_emissions=novice_emissions,
        transition_matrix=novice_transitions,
    )

    # Profile 2: Expert (faster reading, less review dwell time)
    expert_emissions = {
        "Reading": {"time_spent": FeatureDistribution("normal", {"mean": 25.0, "std": 5.0})},
        "Quiz": {"time_spent": FeatureDistribution("normal", {"mean": 20.0, "std": 4.0})},
        "Review": {"time_spent": FeatureDistribution("normal", {"mean": 15.0, "std": 3.0})},
    }
    expert_transitions = np.array([
        [0.20, 0.70, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.70, 0.20],
    ])
    expert_profile = Profile(
        name="Expert",
        state_emissions=expert_emissions,
        transition_matrix=expert_transitions,
    )

    # Instantiate simulator with a 70% Novice / 30% Expert mixture
    sim = Simulator(
        states=states,
        profiles=[novice_profile, expert_profile],
        profile_distribution={"Novice": 0.70, "Expert": 0.30},
        initial_state="Reading",
    )

    # Generate multi-profile cohort
    traces: pd.DataFrame = sim.generate(
        num_interactions=12,
        num_sequences=20,
        seed=101,
    )

    print("\nSequence counts by assigned profile:")
    seq_profiles = traces.groupby("sequence_id")["profile"].first().value_counts()
    print(seq_profiles)

    print("\nMean time_spent by state and profile:")
    print(traces.groupby(["profile", "state"])["time_spent"].mean().unstack())

    print("\nMulti-profile simulation completed successfully.")


if __name__ == "__main__":
    main()
