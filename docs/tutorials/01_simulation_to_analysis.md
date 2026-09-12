# Tutorial 1: End-to-End Simulation to Analysis

This workflow demonstrates how to generate synthetic behavioral interaction traces, extract causal behavioral features, and visualize the resulting dynamics.

---

## 1. Constructing a Simulator

BehaviorSim models behavior as a sequence of interactions through discrete states, state-conditioned parametric feature emissions, and Markovian or rule-based transition dynamics.

```python
import numpy as np
from behaviorsim import Simulator, State, Profile, FeatureDistribution

# Define discrete states
states = [
    State("Browse", description="Browsing catalog"),
    State("Cart", description="Reviewing cart"),
    State("Checkout", description="Completing purchase"),
]

# Define state-dependent emission distributions
emissions = {
    "Browse": {
        "dwell_time": FeatureDistribution("exponential", {"scale": 15.0}),
        "items_viewed": FeatureDistribution("poisson", {"lam": 4.0}),
    },
    "Cart": {
        "dwell_time": FeatureDistribution("normal", {"mean": 45.0, "std": 10.0}),
        "items_viewed": FeatureDistribution("poisson", {"lam": 2.0}),
    },
    "Checkout": {
        "dwell_time": FeatureDistribution("normal", {"mean": 90.0, "std": 20.0}),
        "items_viewed": FeatureDistribution("uniform_discrete", {"low": 1, "high": 3}),
    },
}

# Transition matrix: row i (current state) -> column j (next state)
transition_matrix = np.array([
    [0.70, 0.25, 0.05],  # From Browse
    [0.20, 0.60, 0.20],  # From Cart
    [0.00, 0.00, 1.00],  # From Checkout (absorbing terminal)
])

# Create behavioral profile
profile = Profile(
    name="StandardShopper",
    state_emissions=emissions,
    transition_matrix=transition_matrix,
)

# Instantiate simulator
sim = Simulator(
    states=states,
    profile=profile,
    initial_state="Browse",
)
```

---

## 2. Generating Synthetic Sequences

Execute the simulation loop to produce independent sequences.

```python
traces = sim.generate(
    num_interactions=20,
    num_sequences=10,
    seed=42,
)

print(traces.head(6))
```

Columns generated:
- `sequence_id`: Sequence identifier ($0, 1, \dots$).
- `interaction_id`: 1-indexed interaction step counter within the sequence ($1, 2, \dots$).
- `state`: Ground-truth hidden state label.
- `profile`: Profile identifier (`StandardShopper`).
- `dwell_time`, `items_viewed`: State-conditioned feature values.

---

## 3. Causal Feature Engineering

Compute causal rolling aggregations and state transition dynamics without future data leakage:

```python
from behaviorsim.feature_engineering import build_behavioral_features

enriched = build_behavioral_features(
    traces,
    numeric_columns=["dwell_time", "items_viewed"],
    rolling_windows=[3, 5],
    rolling_stats=["mean", "var"],
    sequence_column="sequence_id",
)

print("Enriched columns:", enriched.columns.tolist())
```

---

## 4. Visualizing Results

Visualize state occupancies, discrete trajectories, and continuous features:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from behaviorsim.visualization import (
    plot_state_occupancy,
    plot_state_trajectory,
    plot_feature_trajectory,
    save_figure,
)

# 1. State occupancy distribution
fig_occ, ax_occ = plot_state_occupancy(enriched, state_column="state")
save_figure(fig_occ, "results/occupancy.png", close=True)

# 2. Discrete state trajectories across interactions
fig_traj, ax_traj = plot_state_trajectory(
    enriched,
    sequence_column="sequence_id",
    max_sequences=5,
)
save_figure(fig_traj, "results/state_trajectories.png", close=True)

# 3. Rolling feature trajectories
fig_feat, ax_feat = plot_feature_trajectory(
    enriched,
    feature="dwell_time_rolling_mean_w3",
    max_sequences=3,
)
save_figure(fig_feat, "results/dwell_time_rolling.png", close=True)
```
