"""Synthetic Learner Simulator module for CLSI-Adapt.

Backward-compatibility facade re-exporting from behaviorsim.presets.education.
"""

from behaviorsim.presets.education import (
    BASE_TRANSITION_MATRIX,
    LEARNER_PROFILES,
    STATES,
    LearnerProfile,
    _logistic,
    run_simulation,
    save_simulation,
    simulate_all_profiles,
    simulate_learner,
    validate_simulation_diagnostics,
)

__all__ = [
    "LearnerProfile",
    "LEARNER_PROFILES",
    "STATES",
    "BASE_TRANSITION_MATRIX",
    "_logistic",
    "simulate_learner",
    "simulate_all_profiles",
    "save_simulation",
    "validate_simulation_diagnostics",
    "run_simulation",
]
