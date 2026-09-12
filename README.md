# BehaviorSim: Synthetic Sequential Behavioral Data Generation

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-634%20passed-success.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Reproducibility](https://img.shields.io/badge/reproducible-deterministic%20seeds-brightgreen.svg)](#reproducibility)

**BehaviorSim** is a lightweight, dependency-minimal Python library for generating, calibrating, feature-engineering, and analyzing synthetic sequential behavioral telemetry with ground-truth discrete states.

---

## Why BehaviorSim?

Developing adaptive systems, recommendation engines, and user state identification algorithms requires high-fidelity sequential interaction traces. However, empirical telemetry often lacks ground-truth psychological, cognitive, or operational state labels, suffers from privacy and compliance restrictions, or presents severe class imbalance.

BehaviorSim provides:
- **Ground-truth state tracking**: Exact discrete behavioral states recorded at every step without latent ambiguity.
- **Configurable dynamics**: Markovian transition matrices alongside priority-ordered conditional transition rules.
- **State-conditioned parametric emissions**: Continuous, discrete, and categorical feature emissions conditioned on active states and interaction history.
- **Multi-profile simulation**: Heterogeneous user cohorts with configurable mixture distributions.
- **Empirical calibration & validation**: Parameter fitting from observed traces with statistical fidelity validation.
- **Strictly causal feature engineering**: History-dependent rolling windows, dwell times, and session metrics with zero forward-looking leakage.
- **Publication-ready visualization**: Restrained, publication-grade plotting utilities for states, trajectories, distributions, and diagnostic dashboards.

---

## Features

- **Discrete Behavioral States**: Immutable, named state abstractions (`State`).
- **Flexible Emission Families**: Gaussian, Log-Normal, Exponential, Uniform, Uniform-Discrete, Bernoulli, Poisson, and Categorical distributions (`FeatureDistribution`).
- **Archetype Profiles**: Encapsulated state transition rules and state-conditioned emission mappings (`Profile`).
- **Domain Presets**: Ready-to-use simulators for `education`, `mobile_app`, `healthcare`, and `finance`.
- **Deterministic Calibration**: Proxy-based state extraction, sequence-isolated transition fitting, and automated distribution parameter estimation (`CalibrationData`, `fit_profile`).
- **Behavioral Cohort Clustering**: Unsupervised entity clustering on sequential feature summaries into discrete profiles (`cluster_profiles`).
- **Calibration Validation**: Statistical discrepancy evaluation (TVD, Frobenius norm, Wasserstein distance, Kolmogorov-Smirnov test) with configurable verification thresholds (`validate_calibration`, `ValidationReport`).
- **Causal Feature Pipeline**: Sliding historical windows, state transition counters, and session boundary detectors (`build_behavioral_features`).
- **Publication-Ready Visualization**: Clean matplotlib diagnostics (`plot_state_occupancy`, `plot_transition_matrix`, `plot_state_trajectory`, `plot_feature_comparison`, `plot_calibration_summary`, `save_figure`).
- **Declarative YAML/JSON Configuration**: Safe, schema-validated configuration files with zero dynamic code injection (`load_config`, `build_simulator`).
- **Command-Line Interface**: Fast simulation execution and schema validation via `behaviorsim`.

---

## Installation

### Standard Installation
```bash
pip install behaviorsim
```

### With Optional Dependencies
```bash
# Fast Parquet storage
pip install "behaviorsim[parquet]"

# Visualization layer
pip install "behaviorsim[viz]"

# Calibration & validation
pip install "behaviorsim[calibration]"

# All optional features
pip install "behaviorsim[all]"
```

### Development / Repository Installation
For local development from a cloned repository:
```bash
pip install -e ".[dev]"
```

Requirements: Python >= 3.9, NumPy >= 1.24, Pandas >= 2.0, SciPy >= 1.10, PyYAML >= 6.0.

---

## Quick Start

Generate synthetic behavioral data in 10 lines of Python:

```python
import numpy as np
from behaviorsim import Simulator, State, Profile, FeatureDistribution

# 1. Define states
states = [State("Browse"), State("Cart"), State("Checkout")]

# 2. Define state emissions
emissions = {
    "Browse": {"dwell_sec": FeatureDistribution("exponential", {"scale": 15.0})},
    "Cart": {"dwell_sec": FeatureDistribution("normal", {"mean": 45.0, "std": 10.0})},
    "Checkout": {"dwell_sec": FeatureDistribution("normal", {"mean": 90.0, "std": 15.0})},
}

# 3. Define transition matrix P(s_t -> s_{t+1})
transitions = np.array([
    [0.70, 0.25, 0.05],
    [0.20, 0.60, 0.20],
    [0.00, 0.00, 1.00],
])

# 4. Build Profile and Simulator
profile = Profile("Shopper", state_emissions=emissions, transition_matrix=transitions)
sim = Simulator(states=states, profile=profile, initial_state="Browse")

# 5. Generate reproducible interaction traces
df = sim.generate(num_interactions=10, num_sequences=5, seed=42)
print(df[["sequence_id", "interaction_id", "state", "dwell_sec"]].head())
```

---

## Presets

BehaviorSim includes built-in domain presets constructible via `Simulator.from_preset(name, **kwargs)`:

| Preset | Ground-Truth States | Emitted Features | Profiles |
| :--- | :--- | :--- | :--- |
| **`education`** | `Optimal`, `Overload`, `Underload` | `accuracy`, `difficulty`, `nrt`, `retries`, `help_requested`, `confidence` | `fast_accurate`, `fast_inaccurate`, `slow_accurate`, `slow_inaccurate`, `average` |
| **`mobile_app`**| `Browsing`, `ActiveSession`, `CheckoutFlow` | `session_time_seconds`, `action_count`, `scroll_depth`, `button_clicks`, `notification_clicked`, `cart_value` | `casual_browser`, `power_user`, `bargain_hunter` |
| **`healthcare`**| `Baseline`, `Elevated`, `Discharged` | `heart_rate_bpm`, `systolic_bp`, `spo2_pct`, `temperature_c`, `alert_triggered`, `mobility_score` | `stable_recovery`, `chronic_risk`, `post_op` |
| **`finance`**   | `Stable`, `Active`, `Volatile`, `Drawdown` | `portfolio_value`, `daily_return`, `transaction_count`, `trade_volume`, `volatility`, `drawdown`, `risk_alert` | `passive_investor`, `active_trader`, `institutional_fund` |

> [!WARNING]
> **Domain Disclaimers**:
> - **Healthcare**: The healthcare preset is a synthetic benchmark model for evaluating telemetry algorithms. It is **NOT** a clinically validated medical model and must **NOT** be used for patient diagnosis, clinical triaging, or healthcare decisions.
> - **Finance**: The finance preset is a synthetic statistical simulation. It is **NOT** financial, investment, trading, or fraud-detection advice.

---

## Calibration

BehaviorSim provides an empirical calibration subsystem that fits parameters from observed interaction logs.

```python
from behaviorsim.calibration import CalibrationData, fit_profile, validate_calibration

# 1. Package observed traces
cal_data = CalibrationData(
    data=observed_df,
    states=["Browse", "Cart", "Checkout"],
    numeric_features=["dwell_sec"],
)

# 2. Fit generative Profile
calibrated_profile = fit_profile(cal_data, name="FittedShopper", transition_smoothing=0.01)

# 3. Simulate synthetic counterparts
sim = Simulator(states=[State(s) for s in cal_data.states], profile=calibrated_profile)
synthetic_df = sim.generate(num_interactions=10, num_sequences=50, seed=123)

# 4. Statistically validate fidelity
report = validate_calibration(
    empirical_data=cal_data,
    synthetic_data=synthetic_df,
    profile=calibrated_profile,
    thresholds={"max_state_tvd": 0.15, "max_transition_mae": 0.10},
)
print(f"Validation Passed: {report.is_valid}")
```

### Conceptual Boundary on Latent Identifiability
BehaviorSim uses **explicit proxy-based state extraction** (`extract_state_proxy`) or user-declared state columns. It deliberately does not perform unsupervised HMM/Baum-Welch latent-state discovery, which is mathematically non-identifiable without strong structural assumptions.

---

## Feature Engineering

The feature engineering layer computes historical sequential metrics under a **strict causal contract**:

$$
\text{feature}_t = f(x_0, x_1, \dots, x_{t-1})
$$

Features computed at interaction $t$ evaluate observations strictly prior to step $t$. Current step outcomes and future steps are never leaked.

```python
from behaviorsim.feature_engineering import build_behavioral_features

enriched_df = build_behavioral_features(
    df,
    numeric_columns=["dwell_sec"],
    rolling_windows=[3, 5],
    rolling_stats=["mean", "var"],
    sequence_column="sequence_id",
)
```

---

## Visualization

BehaviorSim includes a restrained, publication-grade visualization layer returning standard matplotlib `(fig, ax)` tuples:

```python
from behaviorsim.visualization import (
    plot_state_occupancy,
    plot_state_trajectory,
    plot_transition_matrix,
    plot_calibration_summary,
    save_figure,
)

# State occupancy proportions
fig, ax = plot_state_occupancy(df, state_column="state")
save_figure(fig, "figures/occupancy.png", close=True)

# Discrete state trajectories
fig, ax = plot_state_trajectory(df, max_sequences=5)
save_figure(fig, "figures/trajectories.png", close=True)

# Transition heatmap
fig, ax = plot_transition_matrix(df)
save_figure(fig, "figures/transitions.png", close=True)

# Calibration diagnostic dashboard
fig, axes = plot_calibration_summary(report)
save_figure(fig, "figures/calibration_dashboard.png", close=True)
```

---

## Declarative Configuration & CLI

Simulations can be completely defined in declarative YAML or JSON files:

```yaml
version: "1.0"
states: ["Exploration", "Checkout"]
initial_state: "Exploration"
profiles:
  default:
    probability: 1.0
    transitions:
      matrix: [[0.8, 0.2], [0.0, 1.0]]
    emissions:
      Exploration:
        dwell: { distribution: "normal", params: { mean: 30.0, std: 5.0 } }
      Checkout:
        dwell: { distribution: "normal", params: { mean: 60.0, std: 10.0 } }
simulation:
  n_sequences: 20
  max_steps: 15
  seed: 42
```

### CLI Commands
```bash
# Validate configuration schema
behaviorsim validate simulation.yaml

# Run simulation to CSV, JSON, or Parquet
behaviorsim run simulation.yaml -o results/traces.csv
behaviorsim run simulation.yaml -o results/traces.parquet --sequences 100 --seed 123
```

---

## Architecture

```text
src/behaviorsim/
├── core/                   # Mathematical simulation kernel
│   ├── state.py            # Discrete State representation
│   ├── feature.py          # Parametric emission FeatureDistribution
│   ├── transition.py       # Markov transitions and conditional TransitionRule
│   ├── profile.py          # Behavioral Profile container
│   ├── simulator.py        # Master Simulator engine
│   └── utils.py            # Stochastic matrix and seed utilities
├── config.py               # Declarative schema, validation, & compilation
├── cli.py                  # Command-line interface ('behaviorsim')
├── presets/                # Domain-specific simulation presets
│   ├── registry.py         # Preset registry & factory resolution
│   ├── education.py        # Cognitive load simulation preset
│   ├── mobile_app.py       # Mobile engagement & churn preset
│   ├── healthcare.py       # Adherence telemetry preset
│   └── finance.py          # Portfolio risk dynamics preset
├── calibration/            # Empirical parameter calibration & diagnostics
│   ├── fitter.py           # CalibrationData, proxy extraction, matrix/emission fitting
│   ├── clustering.py       # Sequence-level k-means profile clustering
│   └── validator.py        # ValidationReport & statistical fidelity metrics
├── feature_engineering/    # Strictly causal sequential dynamics
│   ├── causal.py           # Pipeline compiler & causal validation
│   ├── windows.py          # Non-leaking rolling window statistics
│   └── transitions.py      # Dwell times, transition counters, session tracking
└── visualization/          # Publication-ready plotting & diagnostics
    ├── plot_states.py      # State occupancy & transition matrix heatmaps
    ├── plot_trajectories.py# Discrete state & continuous feature trajectories
    ├── plot_distributions.py# Aligned histograms & categorical comparisons
    ├── plot_calibration.py # Validation diagnostics & dashboard
    └── __init__.py         # Public exports & save_figure utility
```

---

## Reproducibility

BehaviorSim guarantees **exact cross-platform reproducibility**:
- Deterministic seeding: Calling `sim.generate(..., seed=42)` produces bit-for-bit identical DataFrames across runs and operating systems.
- Hierarchical sequence seeding: Independent sequences derive isolated seeds from the master seed via deterministic hashing (`derive_learner_seed`).
- Fully reproducible in headless environments with zero dependency on GUI backends.

---

## Testing Status

The BehaviorSim test suite includes **634 passing tests** with 100% clean diff audits across all subsystems:

```bash
# Run complete test suite
python -m pytest -q
# 634 passed, 5 skipped
```

Test breakdown:
- Core simulator & dynamics: 236 tests
- Presets (education, mobile, healthcare, finance): 112 tests
- Calibration & clustering: 102 tests
- Causal feature engineering: 33 tests
- Visualization & diagnostics: 48 tests
- Legacy benchmark evaluation: 100 tests

---

## Limitations

1. **Synthetic vs. Real Behavior**: Synthetic data generates trajectories consistent with configured distributions and transition rules. It does not automatically capture unmodeled real-world confounding or non-stationary drift.
2. **Proxy Calibration**: Calibration reflects the observable proxies supplied by the user; it does not infer hidden latent intent without explicit proxy definitions.
3. **No Clinical or Financial Advice**: Healthcare and financial presets are academic benchmarks and must not be used for medical or financial decision-making.

---

## Historical Context & Citation

BehaviorSim evolved from the **CLSI-Adapt** research framework for cognitive load state identification under temporal validation.

If you use BehaviorSim in academic research, please cite:

```bibtex
@software{behaviorsim2026,
  title = {BehaviorSim: Synthetic Sequential Behavioral Data Generation Library},
  author = {Vedaang Sharma},
  year = {2026},
  url = {https://github.com/gtathelegend/BehaviourSim}
}
```

## Credits

**Author:** Vedaang Sharma

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
