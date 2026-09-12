# BehaviorSim Release Readiness & Quality Assessment

This document summarizes the packaging, quality hardening, performance validation, and release status of BehaviorSim for external publication and distribution.

---

## 1. Release Overview

- **Package Name**: `behaviorsim`
- **Current Version**: `1.0.0` (PEP 440 package version; checkpoint tagged at `v0.9.0-docs` preparing for `v1.0.0`)
- **License**: MIT License (`LICENSE`)
- **Repository Layout**: Clean `src/` layout (`src/behaviorsim/`)
- **Supported Python Versions**: Python `>=3.9` (tested on `3.9`, `3.10`, `3.11`, `3.12`)
- **Build Backend**: `setuptools>=61.0.0`, `wheel`, `build`

---

## 2. Packaging & Build Status

| Artifact | Format | Build Status | Verification |
| :--- | :--- | :--- | :--- |
| **Wheel** | `.whl` (`behaviorsim-1.0.0-py3-none-any.whl`) | PASS | Clean install in isolated virtualenv; CLI and simulator executed |
| **Source Distribution** | `.tar.gz` (`behaviorsim-1.0.0.tar.gz`) | PASS | Validated with `python -m build` |
| **CLI Entry Point** | `behaviorsim` | PASS | `behaviorsim --help`, `behaviorsim validate`, `behaviorsim run` |
| **Package Discovery** | `tool.setuptools.packages.find` | PASS | Only `behaviorsim*` bundled; no leakage of internal `src/` modules |

### Optional Dependency Groups

- `behaviorsim[parquet]`: `pyarrow>=10.0.0` for columnar parquet export
- `behaviorsim[viz]`: `matplotlib>=3.7.0`, `seaborn>=0.12.0` for plotting and dashboards
- `behaviorsim[calibration]`: `scikit-learn>=1.2.0` for profile clustering
- `behaviorsim[all]`: full suite of optional features
- `behaviorsim[dev]`: test, lint, and build dependencies

---

## 3. Test & Verification Status

- **Full Regression Suite**: 631 passed, 5 skipped (skipped tests require optional external datasets/hardware).
- **Core Engine**: 100% deterministic reproducibility under fixed seeds across single and multi-sequence configurations.
- **Sequence Isolation**: Traces for sequence $k$ are strictly invariant to the total number of sequences requested.
- **Mutation Safety**: Verified immutable caller DataFrames, deep-copied configuration dictionaries, and isolated transition matrix arrays.
- **Continuous Integration**: `.github/workflows/ci.yml` configured for matrix testing across Linux and Windows on Python 3.9–3.12 without external secrets.

---

## 4. Performance & Scalability Benchmarks

Benchmarks executed on standard hardware (Intel/AMD x86_64, Windows, Python 3.12):

| Workload | Sequences | Interactions/Seq | Total Rows | Runtime (s) | Throughput (rows/s) | Peak Memory |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small** | 1 | 100 | 100 | 0.0388s | 2,579 rows/s | 0.08 MB |
| **Medium** | 10 | 10,000 | 100,000 | 17.0586s | 5,862 rows/s | 17.51 MB |
| **Large** | 50 | 2,000 | 100,000 | 16.4471s | 6,080 rows/s | 16.13 MB |
| **High-Volume** | 1 | 100,000 | 100,000 | 16.9106s | 5,913 rows/s | 45.05 MB |

> **Note on Throughput**: BehaviorSim achieves ~6,000 interactions/sec on complex dynamic state-transition models with feature emissions and continuous tracking.

---

## 5. Known Limitations & Technical Debt

1. **Legacy Education Presets Import**:
   `src/behaviorsim/presets/education.py` historically shared configuration objects with the original CLSI-Adapt research pipeline (`src.config`). In Phase 7, this was hardened via a safe fallback helper (`_get_default_config()`), ensuring standalone installation without requiring `src` on `sys.path`.
2. **Legacy SHAP Plotting Warnings**:
   4 deprecation warnings originate within the legacy CLSI-Adapt evaluation models (`src/models/clsi_adapt.py`) when generating SHAP summary figures. These belong to the legacy research code rather than the core BehaviorSim library.
3. **Single-Node Execution**:
   Current generation runs in-process. Distributed/multiprocessing generation is omitted by design to preserve determinism and architectural simplicity.

---

## 6. Pre-Release Checklist

- [x] Wheel and sdist build cleanly (`python -m build`).
- [x] Wheel installs cleanly in isolated environment (`pip install dist/*.whl`).
- [x] CLI entry point operates correctly in installed environment.
- [x] No sensitive credentials, secrets, or internal paths in repository.
- [x] MIT License file included in root and packaged wheel metadata.
- [x] Full regression test suite passes (631 passed, 5 skipped).
- [x] GitHub Actions CI workflow configured.
- [x] Public API exports verified and documented.
- [x] Working tree diff check clean (`git diff --check`).
