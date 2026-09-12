# Tutorial 3: Multi-Profile Modeling & Behavioral Clustering

This tutorial demonstrates how to discover empirical behavioral cohorts from interaction traces using k-means profile clustering and simulate multi-profile mixtures.

---

## Conceptual Boundary: Empirical Clustering vs. Psychological Latent Traits

> **Important Boundary**: Behavioral clustering in BehaviorSim partitions observable sequence-level feature aggregates (state dwell times, feature means/variances, activity rates). It is an **empirical telemetry segmentation** method, NOT an inference of latent psychological, medical, or financial traits.

---

## 1. Preparing Multi-Agent Traces

```python
import numpy as np
import pandas as pd
from behaviorsim.calibration import CalibrationData

# Create multi-sequence interaction traces representing distinct cohorts
np.random.seed(42)
rows = []
for seq_id in range(40):
    # Cohort 1: Quick, low-value interactions
    # Cohort 2: Slow, high-value interactions
    is_cohort_b = seq_id >= 20
    for step in range(10):
        if is_cohort_b:
            state = np.random.choice(["Active", "Idle"], p=[0.8, 0.2])
            val = np.random.normal(100.0, 15.0)
        else:
            state = np.random.choice(["Active", "Idle"], p=[0.3, 0.7])
            val = np.random.normal(20.0, 5.0)
        rows.append({
            "sequence_id": seq_id,
            "interaction_id": step + 1,
            "state": state,
            "transaction_value": val,
        })

df = pd.DataFrame(rows)

cal_data = CalibrationData(
    data=df,
    states=["Active", "Idle"],
    numeric_features=["transaction_value"],
)
```

---

## 2. Clustering Behavioral Profiles

Partition entities into $k$ behavioral profiles:

```python
from behaviorsim.calibration import cluster_profiles

result = cluster_profiles(
    data=cal_data,
    n_profiles=2,
    seed=42,
    name_prefix="Cohort",
)

print(f"Extracted {len(result.profiles)} profiles:")
for p in result.profiles:
    print(f"  Profile '{p.name}' with mixture weight {result.profile_distribution[p.name]:.2f}")
```

---

## 3. Simulating Multi-Profile Mixtures

Instantiate a `Simulator` configured with the profile mixture:

```python
from behaviorsim import Simulator, State

sim = Simulator(
    states=[State(s) for s in cal_data.states],
    profiles=result.profiles,
    profile_distribution=result.profile_distribution,
)

# Generate multi-cohort synthetic traces
synthetic_cohorts = sim.generate(
    num_interactions=15,
    num_sequences=20,
    seed=101,
)

print(synthetic_cohorts["profile"].value_counts())
```

---

## 4. Visualizing Profile Differences

Compare state occupancy across the simulated cohorts:

```python
from behaviorsim.visualization import plot_state_occupancy, save_figure

fig, ax = plot_state_occupancy(
    synthetic_cohorts,
    state_column="state",
    profile_column="profile",
    title="State Occupancy by Behavioral Cohort",
)

save_figure(fig, "results/cohort_occupancies.png", close=True)
```
