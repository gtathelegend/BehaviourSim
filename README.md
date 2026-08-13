# CLSI-Adapt Simulator: Adaptive Cognitive Load Detection Framework

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-179%20passed-success.svg)](https://docs.pytest.org/)
[![Reproducibility](https://img.shields.io/badge/seed-42-brightgreen.svg)](#reproducibility-instructions)
[![Status](https://img.shields.io/badge/publication--status-publication%20ready-blue.svg)](#scientific-status)

An open-source synthetic learner simulation environment and benchmark framework for evaluating adaptive cognitive load state identification algorithms (**CLSI-Adapt**) against heuristic and Bayesian Knowledge Tracing (BKT) baselines under strictly causal, forward-chaining temporal validation.

---

## Executive Summary & Research Objective

Real-time cognitive load detection in digital learning platforms is essential for preventing cognitive overload and tailoring adaptive interventions. However, invasive physiological sensors (e.g., eye-tracking, EEG) are impractical for large-scale remote learning. 

The **CLSI-Adapt Simulator** evaluates whether unobtrusive behavioral telemetry—specifically interaction response times, accuracy dynamics, retry frequencies, and help requests—can predictively detect simulator-defined cognitive overload transitions before learning collapse occurs.

---

## Experimental Architecture

The framework implements a 5-stage pipeline:

```text
Synthetic Learner Simulator (5 Profiles, Seed 42, HMM + Log-Normal RT)
        ↓
Ground-Truth Cognitive State Tracking (Optimal / Overload / Underload)
        ↓
Temporal Feature Engineering (Causal Sliding Windows, Overload Target)
        ↓
 ┌──────────────────────┬────────────────────────┬────────────────────────┐
 │ CLSI-Adapt           │ Rule-Based CLSI        │ BKT Baseline           │
 │ (Per-Profile XGBoost)│ (Composite Score)      │ (Mastery / Struggle)   │
 └──────────────────────┴────────────────────────┴────────────────────────┘
        ↓
Temporal Evaluation Framework (5-Fold TimeSeriesSplit, Event Recall, Recovery Time)
        ↓
Publication Artifacts (CSV/LaTeX Tables & PDF/PNG Figures)
```

> **Baseline Interpretation Note**: Bayesian Knowledge Tracing (BKT) measures domain skill acquisition $P(L_t)$. In this benchmark, low mastery ($P(L_t) < 0.30$) serves as a **domain struggle proxy**, NOT a direct cognitive-load detector.

---

## Repository Structure

```text
CLSI-Adapt-Simulator/
├── data/                                 # Primary experimental dataset
│   ├── simulated_learners_all.csv        # 50,000 raw interactions (Seed 42)
│   └── simulated_learner_*.csv           # Per-profile raw interaction logs
├── src/                                  # Production source code
│   ├── config.py                         # Central configuration parameters
│   ├── simulator.py                      # Synthetic learner trajectory simulator
│   ├── feature_engineering.py            # Causal feature extraction & targets
│   ├── evaluation.py                     # Temporal evaluation & summary tables
│   ├── visualize.py                      # Publication figure plotting module
│   └── models/                           # Model implementations
│       ├── clsi_adapt.py                 # CLSI-Adapt XGBoost + CV model
│       ├── rule_based_clsi.py            # Rule-Based heuristic composite baseline
│       └── bkt.py                        # Bayesian Knowledge Tracing baseline
├── tests/                                # Unit test suite (pytest)
│   ├── test_simulator.py
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   ├── test_clsi_adapt.py
│   ├── test_evaluation.py
│   └── test_visualize.py
├── results/                              # Final audited publication outputs
│   ├── tables/                           # CSV and LaTeX LaTeX publication tables
│   │   ├── profile_metrics.csv / .tex
│   │   ├── aggregate_metrics.csv / .tex
│   │   ├── recovery_metrics.csv / .tex
│   │   └── state_statistics.csv / .tex
│   ├── figures/                          # Publication figures (PDF & PNG)
│   │   ├── figure1_architecture.pdf / .png
│   │   ├── figure2_roc_curves.pdf / .png
│   │   ├── figure3_shap_summary.pdf / .png
│   │   ├── figure4_learning_curve.pdf / .png
│   │   ├── figure5_model_comparison.pdf / .png
│   │   └── figure_captions.txt
│   ├── clsi_adapt_phase4_results.txt    # Phase 4 method validation log
│   ├── evaluation_phase5_audit.txt      # Phase 5 evaluation audit log
│   ├── figure4_learning_curve_audit.txt # Figure 4 temporal audit log
│   └── final_reproducibility_audit.txt  # Final reproducibility audit log
├── scratch/                              # Audit and execution scratch scripts
├── run_all.py                            # Master pipeline driver script
├── requirements.txt                      # Python dependencies
├── CITATION.cff                          # Citation metadata
└── README.md                             # Documentation
```

---

## Experimental Design & Dataset Specification

The benchmark dataset consists of **50,000 raw interaction observations** generated under fixed random seed `seed = 42`:

* **Profiles**: 5 synthetic learner profiles (`fast_accurate`, `fast_inaccurate`, `slow_accurate`, `slow_inaccurate`, `average`).
* **Learners**: 10 distinct synthetic learners per profile ($5 \times 10 = 50$ total learners).
* **Interactions**: 1,000 sequential interactions per learner.
* **Warm-up Period**: Initial 20 interactions per learner sequence excluded from prediction evaluation.
* **No Fallback Datasets**: Evaluated strictly on the primary 50,000-interaction dataset.

### Multi-Learner Design Rationale
Multiple learners per profile are simulated to:
1. Avoid relying on a single synthetic trajectory.
2. Provide adequate positive overload transition events.
3. Support profile-level pooled learning while strictly preserving independent learner boundaries.

---

## Feature Engineering & Target Definition

### Feature Matrix $X$
Contains 11 causal features computed per interaction $t$:
* **Base Features**: `nrt` (normalized response time), `accuracy`, `window_error_rate`, `retries`, `help_requested`, `confidence`, `streak_correct`, `streak_incorrect`, `nrt_variance`, `session_time`.
* **Rolling Mean NRT**: `rolling_mean_nrt` computed over historical interactions.

> **Feature Window Note**: While original PRD preliminary documents referenced an initial window of 10, the finalized production implementation uses `feature_window_size = 5` in `Config` and `run_all.py` to maintain local sensitivity to rapid cognitive transitions.

### Overload Target Definition $y$
An interaction $t$ is labeled `overload = 1` if and only if:
$$\text{mean}(\text{accuracy}[t-3 \dots t]) \ge 0.75 \quad \text{AND} \quad \text{mean}(\text{accuracy}[t+1 \dots t+3]) \le 0.50$$
* **Causal Separation**: Features use data strictly from $0 \dots t$. Future observations ($t+1 \dots t+3$) are used **exclusively** to construct the target label $y$ for supervised training.

---

## Predictive Models & Baselines

1. **CLSI-Adapt**: Profile-specific XGBoost classifiers trained with scale-position-weight balancing and 5-fold temporal forward-chaining cross-validation (`TimeSeriesSplit`). Inner CV grid search optimizes hyperparameters (`max_depth` $\in \{3,5,7\}$, `learning_rate` $\in \{0.01, 0.1\}$).
2. **Rule-Based CLSI**: Parameter-free composite heuristic index:
   $$\text{CLSI} = \frac{0.50 \cdot \text{acc} + 0.25 \cdot (1 - \text{NRT}) + 0.15 \cdot (1 - \text{wer}) + 0.10 \cdot (1 - \text{retries\_norm})}{1.0} - 0.10 \cdot \text{help\_requested}$$
   Predicts overload when $\text{CLSI} < 0.40$.
3. **BKT Baseline**: Standard Bayesian Knowledge Tracing ($P(L_0)=0.3, P(T)=0.1, P(G)=0.2, P(S)=0.1$, reset per learner). Predicts struggle when mastery $P(L_t) < 0.30$.

---

## Temporal Cross-Validation Methodology

To prevent data leakage across temporal sequences:
* `TimeSeriesSplit(n_splits=5)` is applied across unique interaction time steps.
* Strict temporal forward-chaining invariant:
  $$\max(\text{train\_pos}) < \min(\text{val\_pos}) \quad \text{and} \quad \text{train\_times} \cap \text{val\_times} = \emptyset$$
* All learners at a given interaction time step are assigned to the same temporal partition, eliminating simultaneous-time leakage.

---

## Performance Summary (Common Out-of-Fold Subset)

| Model | Evaluated Profiles | Mean ROC AUC | Mean Precision | Mean Recall | Mean F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLSI-Adapt** | 3 valid profiles | **0.9694 ± 0.0049** | **0.3490 ± 0.2025** | **0.6643 ± 0.3874** | **0.4563 ± 0.2637** |
| **Rule-Based CLSI** | 4 valid profiles | 0.1062 ± 0.0132 | 0.0055 ± 0.0059 | 0.1473 ± 0.1230 | 0.0105 ± 0.0111 |
| **BKT (struggle proxy)** | 4 valid profiles | 0.0478 ± 0.0249 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

* **Single-Class Target Handling**: Single-class profiles (`fast_inaccurate` and `slow_inaccurate`) return mathematically undefined `AUC = NaN`. These are preserved as `NaN` and excluded from aggregate means while reporting contributing profile counts.

---

## Reproducibility Instructions

### Environment Setup
```bash
# Clone repository
git clone https://github.com/gtathelegend/CLSI-Adapt-Simulator.git
cd CLSI-Adapt-Simulator

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the End-to-End Pipeline
```bash
python run_all.py
```
This executes the full pipeline under `seed = 42`:
`Simulation -> Feature Engineering -> Model Fitting -> Evaluation -> Table Export -> Figure Generation`.

### Running Unit Tests
```bash
python -m pytest
```
Expected output: **179 passed, 5 skipped** (100% clean test suite pass).

---

## Scientific Limitations

1. **Synthetic Simulation**: Trajectories are generated via Hidden Markov Models with log-normal response times. Results provide a controlled benchmark but do not replace human-subject clinical validation.
2. **Simulator Ground Truth**: Overload labels reflect simulator state transition definitions.
3. **Profile-Level Model Pooling**: Models are trained per profile; cross-profile generalization is not evaluated.
4. **Rare Overload Targets**: Highly inaccurate profiles feature near-zero overload transitions, resulting in undefined `AUC = NaN`.
5. **Baseline Proxy Alignment**: BKT measures domain mastery rather than direct cognitive load, confirming that struggle proxies alone are insufficient for load detection.
6. **Observational Recovery**: Recovery interval statistics measure elapsed time following model detection to the next optimal state; they do not represent causal intervention efficacy.

---

## Publication Artifacts

All final audited artifacts are available in `results/`:
* **Tables**: [`results/tables/profile_metrics.csv`](file:///c:/Users/vedaa/OneDrive/Documents/CLSI-ADAPT/CLSI-Adapt-Simulator/results/tables/profile_metrics.csv), [`aggregate_metrics.csv`](file:///c:/Users/vedaa/OneDrive/Documents/CLSI-ADAPT/CLSI-Adapt-Simulator/results/tables/aggregate_metrics.csv), [`recovery_metrics.csv`](file:///c:/Users/vedaa/OneDrive/Documents/CLSI-ADAPT/CLSI-Adapt-Simulator/results/tables/recovery_metrics.csv), [`state_statistics.csv`](file:///c:/Users/vedaa/OneDrive/Documents/CLSI-ADAPT/CLSI-Adapt-Simulator/results/tables/state_statistics.csv) (and `.tex` equivalents).
* **Figures**: [`results/figures/figure1_architecture.pdf`](file:///c:/Users/vedaa/OneDrive/Documents/CLSI-ADAPT/CLSI-Adapt-Simulator/results/figures/figure1_architecture.pdf), [`figure2_roc_curves.pdf`](file:///c:/Users/vedaa/OneDrive/Documents/CLSI-ADAPT/CLSI-Adapt-Simulator/results/figures/figure2_roc_curves.pdf), [`figure3_shap_summary.pdf`](file:///c:/Users/vedaa/OneDrive/Documents/CLSI-ADAPT/CLSI-Adapt-Simulator/results/figures/figure3_shap_summary.pdf), [`figure4_learning_curve.pdf`](file:///c:/Users/vedaa/OneDrive/Documents/CLSI-ADAPT/CLSI-Adapt-Simulator/results/figures/figure4_learning_curve.pdf), [`figure5_model_comparison.pdf`](file:///c:/Users/vedaa/OneDrive/Documents/CLSI-ADAPT/CLSI-Adapt-Simulator/results/figures/figure5_model_comparison.pdf).
* **Audit Logs**: `results/clsi_adapt_phase4_results.txt`, `results/evaluation_phase5_audit.txt`, `results/figure4_learning_curve_audit.txt`, `results/final_reproducibility_audit.txt`.

---

## Citation & Licensing

If you use this codebase or benchmark in your research, please cite using [`CITATION.cff`](file:///c:/Users/vedaa/OneDrive/Documents/CLSI-ADAPT/CLSI-Adapt-Simulator/CITATION.cff).

*License*: Formal open-source licensing selection is pending decision by the repository owner.
