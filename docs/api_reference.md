# BehaviorSim Public API Reference

This document provides a comprehensive reference for all public classes, functions, and interfaces across the `behaviorsim` package.

---

## 1. Core Simulation Engine (`behaviorsim` / `behaviorsim.core`)

### `Simulator`
Primary simulation orchestrator executing discrete-state stochastic Markov and semi-Markov behavioral processes.

```python
from behaviorsim import Simulator, State, Profile, FeatureDistribution

sim = Simulator(
    states: Sequence[State],
    profile: Optional[Profile] = None,
    transition_matrix: Optional[np.ndarray] = None,
    initial_state: Optional[Union[str, State]] = None,
    *,
    profiles: Optional[Union[Sequence[Profile], Mapping[str, Profile]]] = None,
    profile_distribution: Optional[Mapping[str, float]] = None,
)
```

#### Key Methods:
- **`generate(num_interactions: int = 100, num_sequences: int = 1, seed: Optional[int] = None) -> pd.DataFrame`**
  Generates synthetic interaction traces across independent sequences. Ergonomic alias for `simulate()`.
- **`simulate(num_interactions: int = 100, num_sequences: int = 1, seed: Optional[int] = None) -> pd.DataFrame`**
  Main execution loop. Returns a DataFrame with columns:
  - `sequence_id`: Sequence identifier (`0, 1, ..., num_sequences - 1`).
  - `interaction_id`: 1-indexed interaction step counter within the sequence.
  - `state`: Name of the discrete ground-truth state at step $t$.
  - `profile`: Profile identifier assigned to the sequence.
  - Feature columns emitted by the state distribution.
- **`from_preset(name: str, **kwargs: Any) -> Simulator`** (classmethod)
  Constructs a simulator from registered preset factories (`'education'`, `'mobile_app'`, `'healthcare'`, `'finance'`).
- **`from_config(config: Union[SimulationConfig, Mapping, str, Path], ...) -> Simulator`** (classmethod)
  Constructs a simulator from a validated YAML/JSON configuration specification or file.

---

### `State`
Discrete behavioral state representation.

```python
from behaviorsim import State

state = State(
    name: str,
    description: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
)
```
- Immutable and hashable. States are matched strictly by `name`.

---

### `FeatureDistribution`
Parametric emission generator conditioned on state and contextual interaction history.

```python
from behaviorsim import FeatureDistribution

dist = FeatureDistribution(
    distribution_type: str,
    params: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
)
```

#### Supported Distribution Families:
- `"normal"`: `{"mean": float, "std": float}` (requires `std >= 0`)
- `"lognormal"`: `{"mean": float, "std": float}` (underlying log-scale parameters)
- `"exponential"`: `{"scale": float}` (requires `scale > 0`)
- `"uniform"`: `{"low": float, "high": float}` (requires `high >= low`)
- `"uniform_discrete"`: `{"items": Sequence[Any]}` or `{"low": int, "high": int}`
- `"bernoulli"`: `{"p": float}` (requires `0.0 <= p <= 1.0`)
- `"poisson"`: `{"lam": float}` (requires `lam >= 0`)
- `"categorical"`: `{"items": Sequence[Any], "probabilities": Optional[Sequence[float]]}`

Features support dynamic contextual evaluation via context dictionaries containing historical aggregations.

---

### `Profile`
Behavioral archetype defining state transition dynamics and state-dependent emission distributions.

```python
from behaviorsim import Profile

profile = Profile(
    name: str,
    state_emissions: Optional[Mapping[str, Mapping[str, FeatureDistribution]]] = None,
    transition_matrix: Optional[np.ndarray] = None,
    transition_rules: Optional[Sequence[TransitionRule]] = None,
    probability: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
)
```
- **`state_emissions`**: Nested mapping: `{state_name: {feature_name: FeatureDistribution}}`.
- **`transition_matrix`**: 2D square row-stochastic matrix of transition probabilities $P(s_t \to s_{t+1})$.
- **`transition_rules`**: Optional priority-ordered conditional rules evaluated before falling back to the transition matrix.
- **`probability`**: Relative mixture weight when used in multi-profile simulators.

---

### `TransitionRule`
Deterministic or stochastic state transition rule conditioned on sequence history.

```python
from behaviorsim import TransitionRule

rule = TransitionRule(
    source_state: str,
    target_state: str,
    condition: Optional[Callable[[HistoryContext], bool]] = None,
    probability: float = 1.0,
    priority: int = 0,
    description: str = "",
)
```
- Evaluated when the current state equals `source_state`. If `condition(history)` evaluates to `True`, transitions to `target_state` with probability `probability`. Higher `priority` values take precedence.

---

## 2. Configuration Subsystem (`behaviorsim.config`)

```python
from behaviorsim.config import SimulationConfig, load_config, build_simulator
```

- **`load_config(path: Union[str, Path]) -> SimulationConfig`**
  Loads and validates a YAML or JSON configuration file against the declarative schema.
- **`build_simulator(config: SimulationConfig, profile_name: Optional[str] = None) -> Simulator`**
  Compiles a `SimulationConfig` object into a runnable `Simulator` instance.
- **`SimulationConfig`**: Immutable container specifying `states`, `profiles`, `initial_state`, `n_sequences`, `max_steps`, `seed`, and output parameters.

---

## 3. Presets Subsystem (`behaviorsim.presets`)

```python
from behaviorsim.presets import get_preset, list_presets, register_preset
```

- **`list_presets() -> List[str]`**: Returns names of all registered presets:
  `["education", "finance", "healthcare", "mobile_app"]`.
- **`get_preset(name: str) -> Callable[..., Simulator]`**: Retrieves simulator factory for a preset.
- **`register_preset(name: str, factory: Callable[..., Simulator], overwrite: bool = False) -> None`**:
  Registers custom domain simulator factories.

### Available Domain Presets:
1. **`education`**: Cognitive load states (`Optimal`, `Overload`, `Underload`), response times, accuracy, hint requests. Profiles: `fast_accurate`, `fast_inaccurate`, `slow_accurate`, `slow_inaccurate`, `average`.
2. **`mobile_app`**: Engagement states (`Engaged`, `Browsing`, `Idle`, `Churned`), scroll depth, latency, session lengths.
3. **`healthcare`**: Adherence telemetry (`Adherent`, `Irregular`, `NonAdherent`), medication logging, symptom reports. *(Synthetic evaluation model only — not for clinical use).*
4. **`finance`**: Transaction behavior (`Normal`, `ElevatedRisk`, `Suspicious`), transaction amounts, velocity, fraud markers. *(Synthetic evaluation model only — not for trading/financial decisions).*

---

## 4. Calibration Subsystem (`behaviorsim.calibration`)

```python
from behaviorsim.calibration import (
    CalibrationData,
    extract_state_proxy,
    fit_distribution,
    fit_profile,
    fit_transition_matrix,
    validate_calibration,
    cluster_profiles,
    ValidationReport,
)
```

### `CalibrationData`
Immutable, reference-isolated container for observed interaction traces.

```python
cal_data = CalibrationData(
    data: pd.DataFrame,
    states: Sequence[str],
    state_column: str = "state",
    feature_columns: Optional[Sequence[str]] = None,
    sequence_column: Optional[str] = "sequence_id",
    profile_column: Optional[str] = None,
    numeric_features: Optional[Sequence[str]] = None,
    categorical_features: Optional[Sequence[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
)
```

### `extract_state_proxy`
Deterministically labels observed interactions using explicit columns, mappings, or row/DataFrame callables without unsupervised latent inference.

### `fit_transition_matrix`
```python
matrix = fit_transition_matrix(
    data: CalibrationData,
    smoothing: float = 0.0,
) -> np.ndarray
```
Fits a row-stochastic empirical transition matrix. Strictly isolates sequence boundaries. Supports additive Laplace smoothing.

### `fit_profile`
```python
profile = fit_profile(
    data: CalibrationData,
    *,
    name: str = "calibrated_profile",
    distribution_types: Optional[Mapping[str, str]] = None,
    default_distribution: str = "normal",
    transition_smoothing: float = 0.0,
) -> Profile
```
Fits empirical transition matrices and state-conditioned emission distributions from `CalibrationData`.

### `validate_calibration`
```python
report = validate_calibration(
    empirical_data: Union[CalibrationData, pd.DataFrame],
    synthetic_data: Union[CalibrationData, pd.DataFrame],
    *,
    profile: Optional[Profile] = None,
    states: Optional[Sequence[str]] = None,
    state_column: str = "state",
    sequence_column: Optional[str] = "sequence_id",
    numeric_features: Optional[Sequence[str]] = None,
    categorical_features: Optional[Sequence[str]] = None,
    thresholds: Optional[Mapping[str, float]] = None,
) -> ValidationReport
```
Computes structural validity, state occupancy (TVD), transition dynamics (Frobenius, MAE), continuous feature divergence (Wasserstein, KS), and categorical divergence (TVD, Jensen-Shannon).

Supported threshold keys in `thresholds`:
- `'max_state_tvd'`: Maximum allowed state occupancy TVD.
- `'max_transition_mae'`: Maximum allowed mean absolute transition error.
- `'max_transition_max_error'`: Maximum allowed max absolute transition error.
- `'max_numeric_mean_mae'`: Maximum allowed mean absolute error across numeric features.
- `'max_categorical_tvd'`: Maximum allowed TVD across categorical features.

### `cluster_profiles`
```python
result = cluster_profiles(
    data: CalibrationData,
    n_profiles: int,
    *,
    seed: Optional[int] = None,
    name_prefix: str = "profile",
    smoothing: float = 0.0,
) -> ProfileClusteringResult
```
Aggregates traces into sequence-level behavioral summary vectors and partitions entities via deterministic k-means into discrete calibrated `Profile` instances.

---

## 5. Feature Engineering Subsystem (`behaviorsim.feature_engineering`)

```python
from behaviorsim.feature_engineering import (
    build_behavioral_features,
    compute_rolling_features,
    compute_state_transition_features,
    compute_session_features,
)
```

### `build_behavioral_features`
```python
enriched_df = build_behavioral_features(
    data: Union[pd.DataFrame, CalibrationData],
    *,
    state_column: Optional[str] = "state",
    time_column: Optional[str] = None,
    numeric_columns: Optional[Sequence[str]] = None,
    rolling_windows: Sequence[int] = (3, 5),
    rolling_stats: Sequence[str] = ("mean", "var"),
    session_gap_threshold: Optional[float] = None,
    sequence_column: Optional[str] = None,
    order_column: Optional[str] = None,
) -> pd.DataFrame
```
Generates causal features without future data leakage:
- **Rolling statistics**: `mean`, `var`, `sum`, `min`, `max` over causal history windows.
- **State transitions**: cumulative state counts, consecutive state dwell times, and transition frequencies.
- **Session tracking**: session boundaries and elapsed times based on inactivity thresholds.

---

## 6. Visualization Subsystem (`behaviorsim.visualization`)

```python
from behaviorsim.visualization import (
    plot_state_occupancy,
    plot_state_trajectory,
    plot_transition_matrix,
    plot_feature_distribution,
    plot_feature_comparison,
    plot_categorical_distribution,
    plot_feature_trajectory,
    plot_occupancy_comparison,
    plot_transition_comparison,
    plot_numeric_comparison,
    plot_categorical_comparison,
    plot_calibration_summary,
    save_figure,
)
```

All plotting functions:
- Accept an optional `ax: Optional[Axes]` parameter.
- Return `Tuple[Figure, Axes]` (or `Tuple[Figure, np.ndarray]` for multi-axis layouts).
- Do not modify global `matplotlib.rcParams`.
- Do not call `plt.show()` or close caller-owned figures.
- Run safely in headless environments (`Agg` backend).

### `save_figure`
```python
save_figure(
    fig: Figure,
    path: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = "tight",
    close: bool = False,
    **kwargs: Any,
) -> Path
```
Exports matplotlib figures to PNG, PDF, or SVG. Creates parent directories automatically.
