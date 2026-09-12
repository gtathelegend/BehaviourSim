# Tutorial 4: Declarative Configuration Guide

BehaviorSim provides a declarative YAML and JSON configuration interface for creating, sharing, and executing simulations without writing Python code.

---

## 1. Declarative Schema Overview

A simulation configuration defines:
1. `states`: List of discrete state names.
2. `initial_state`: Name of initial state.
3. `profiles`: Mapping of profile names to transition matrices, rules, and emissions.
4. `simulation`: Execution parameters (`n_sequences`, `max_steps`, `seed`).

### Sample YAML Configuration (`simulation.yaml`)

```yaml
version: "1.0"

states:
  - "Browse"
  - "Search"
  - "Purchase"

initial_state: "Browse"

profiles:
  shoppers:
    probability: 1.0
    transitions:
      matrix:
        - [0.60, 0.30, 0.10]
        - [0.20, 0.50, 0.30]
        - [0.10, 0.10, 0.80]
    emissions:
      Browse:
        latency:
          distribution: "exponential"
          params: { scale: 20.0 }
        clicks:
          distribution: "poisson"
          params: { lam: 3.0 }
      Search:
        latency:
          distribution: "normal"
          params: { mean: 35.0, std: 8.0 }
        clicks:
          distribution: "poisson"
          params: { lam: 5.0 }
      Purchase:
        latency:
          distribution: "normal"
          params: { mean: 80.0, std: 15.0 }
        clicks:
          distribution: "uniform_discrete"
          params: { low: 1, high: 3 }

simulation:
  n_sequences: 50
  max_steps: 25
  seed: 42
```

---

## 2. Declarative Transition Rules (No Code Injection)

> **Security Guarantee**: Dynamic transition conditions strictly evaluate declarative AST specifications over historical features. Unsafe Python expression execution (`eval`, `exec`, `globals()`, or code injection) is strictly forbidden and rejected during schema validation.

Example declarative condition:
```yaml
rules:
  - source: "Browse"
    target: "Search"
    condition:
      feature: "clicks"
      aggregation: "sum"
      window: 3
      operator: ">"
      value: 10
    probability: 0.85
    priority: 1
```

Supported aggregations: `mean`, `sum`, `min`, `max`, `last`.
Supported operators: `<`, `<=`, `>`, `>=`, `==`, `!=`.

---

## 3. Running via Python API

Load and execute configurations directly in Python:

```python
from behaviorsim.config import load_config, build_simulator

# Load and validate schema
config = load_config("simulation.yaml")

# Build executable simulator
sim = build_simulator(config)

# Generate synthetic interaction traces
df = sim.generate(
    num_interactions=config.max_steps,
    num_sequences=config.n_sequences,
    seed=config.seed,
)

print(f"Generated {len(df)} records across {df['sequence_id'].nunique()} sequences.")
```

---

## 4. Running via Command-Line Interface (CLI)

BehaviorSim installs a command-line tool `behaviorsim`:

### Validate Configuration
```bash
behaviorsim validate simulation.yaml
```

### Execute Simulation to CSV, JSON, or Parquet
```bash
# Export to CSV
behaviorsim run simulation.yaml -o results/traces.csv

# Export to Parquet
behaviorsim run simulation.yaml -o results/traces.parquet --seed 123 --sequences 100
```
