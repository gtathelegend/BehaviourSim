# CLSI-Adapt Simulator & Evaluation Framework

## 1. Project Purpose
`clsi-adapt-simulator` is a framework designed to simulate learner interactions, engineer feature representations, and evaluate adaptive learning path algorithms (specifically CLSI-Adapt alongside rule-based CLSI and Bayesian Knowledge Tracing baselines).

## 2. Research Objective
The objective is to evaluate the efficacy, adaptability, and predictive performance of CLSI-Adapt in personalized educational sequences, comparing its path recommendations against traditional rule-based sequencing and Bayesian Knowledge Tracing (BKT).

## 3. High-Level Architecture
The codebase is structured into modular Python packages:
- `src/config.py`: Centralized configuration management for random seeds, hyper-parameters, and file system paths.
- `src/simulator.py`: Learner interaction simulator generating synthetic or empirical sequence logs.
- `src/feature_engineering.py`: Extraction and windowing of temporal/behavioral features from interaction traces.
- `src/models/`: Implementation of adaptive and baseline learning path models (`clsi_adapt.py`, `rule_based_clsi.py`, `bkt.py`).
- `src/evaluation.py`: Performance metrics computation, cross-validation splits, and statistical benchmarking.
- `src/visualize.py`: Generation of evaluation charts, learning curves, and comparative figures.

## 4. Planned Pipeline

```text
Simulation
→ Feature Engineering
→ CLSI-Adapt
→ Rule-Based CLSI
→ BKT
→ Evaluation
→ Visualization
```

## 5. Installation Instructions

1. Ensure Python 3.10+ is installed.
2. Clone the repository and navigate to the root directory:
   ```bash
   cd clsi-adapt-simulator
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 6. Planned Execution Command

Once implementation is complete, run the end-to-end evaluation pipeline with:

```bash
python run_all.py
```

## 7. Reproducibility Statement
To ensure scientific reproducibility across simulation runs and model evaluations, all random number generators (e.g., NumPy, Scipy, Scikit-Learn) are bound to a configurable global seed defined in `src/config.py`. Output tables and figures will be saved in `results/tables/` and `results/figures/` respectively.

> **Note:** The repository structure and configuration have been initialized. Full pipeline implementation is currently under active development.
