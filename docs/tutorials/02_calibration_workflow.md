# Tutorial 2: Calibration and Validation Workflow

This tutorial demonstrates how to calibrate BehaviorSim models from observed empirical interaction traces and rigorously validate the calibrated synthetic generator.

---

## Important Conceptual Boundary: Proxy-Based Calibration

> **Important Boundary**: BehaviorSim uses **explicit proxy-based state identification**. It does NOT attempt unsupervised latent-state discovery (e.g. Baum-Welch / HMM unsupervised parameter estimation), which is mathematically unidentifiable without strong behavioral domain assumptions.
>
> Users explicitly define state mappings or observed proxy columns.

---

## 1. Preparing Empirical Traces

Wrap observed data in `CalibrationData`:

```python
import pandas as pd
from behaviorsim.calibration import CalibrationData

# Sample observed interaction log
observed_df = pd.DataFrame({
    "sequence_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    "interaction_id": [1, 2, 3, 1, 2, 3, 1, 2, 3],
    "state": ["Browse", "Cart", "Checkout", "Browse", "Browse", "Cart", "Cart", "Cart", "Checkout"],
    "response_time": [12.5, 45.2, 88.0, 15.1, 14.2, 50.1, 48.0, 52.3, 91.5],
    "clicks": [3, 2, 1, 4, 3, 2, 1, 2, 1],
})

# Declare CalibrationData container
cal_data = CalibrationData(
    data=observed_df,
    states=["Browse", "Cart", "Checkout"],
    state_column="state",
    sequence_column="sequence_id",
    numeric_features=["response_time", "clicks"],
)
```

---

## 2. Fitting Transition Matrices and Emissions

Fit transition matrices with optional Laplace smoothing and parameterize state emission distributions:

```python
from behaviorsim.calibration import fit_profile

profile = fit_profile(
    data=cal_data,
    name="CalibratedShopper",
    default_distribution="normal",
    transition_smoothing=0.05,
)

print("Calibrated Transition Matrix:\n", profile.transition_matrix)
```

---

## 3. Simulating Synthetic Counterparts

Instantiate a `Simulator` using the calibrated `Profile`:

```python
from behaviorsim import Simulator, State

sim = Simulator(
    states=[State(s) for s in cal_data.states],
    profile=profile,
)

synthetic_traces = sim.generate(
    num_interactions=10,
    num_sequences=50,
    seed=101,
)
```

---

## 4. Validating Calibration Fidelity

Evaluate empirical vs. synthetic fidelity across state occupancy, transition dynamics, and feature distributions:

```python
from behaviorsim.calibration import validate_calibration

report = validate_calibration(
    empirical_data=cal_data,
    synthetic_data=synthetic_traces,
    profile=profile,
    numeric_features=["response_time", "clicks"],
    thresholds={
        "max_state_tvd": 0.15,
        "max_transition_mae": 0.10,
    },
)

print(f"Structural Integrity: {report.structural.is_valid}")
print(f"State Occupancy TVD:  {report.state_occupancy.total_variation_distance:.4f}")
print(f"Transition MAE:       {report.transition.mean_absolute_error:.4f}")
print(f"Validation Passed:    {report.is_valid}")
```

---

## 5. Visualizing Diagnostics Dashboard

Render a comprehensive 2x2 calibration dashboard:

```python
from behaviorsim.visualization import plot_calibration_summary, save_figure

fig, axes = plot_calibration_summary(report)
save_figure(fig, "results/calibration_dashboard.png", close=True)
```
