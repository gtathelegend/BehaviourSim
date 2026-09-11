"""Healthcare patient monitoring preset for BehaviorSim.

This preset defines a synthetic behavioral telemetry model representing patient vital
sign dynamics across latent monitoring states (Baseline, Elevated, Critical, Recovery,
Discharged). It captures synthetic physiological time-series observations including heart
rate, systolic blood pressure, blood oxygen saturation (SpO2), body temperature, alert
triggers, and patient mobility.

DISCLAIMER & NON-CLINICAL BOUNDARY:
This is a synthetic behavioral telemetry model designed solely for machine learning
benchmarking, synthetic data generation, and algorithm evaluation. It is NOT clinically
or medically validated, does NOT represent real patient physiology or empirical clinical
distributions, and does NOT constitute clinical decision support or medical advice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import numpy as np

from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.simulator import Simulator
from behaviorsim.core.state import State
from behaviorsim.core.transition import HistoryContext, TransitionRule


# ---------------------------------------------------------------------------
# State Space
# ---------------------------------------------------------------------------

HEALTHCARE_STATES: List[State] = [
    State("Baseline", description="Stable, resting physiological monitoring state"),
    State("Elevated", description="Moderate physiological elevation, mild vital deviation, or stress"),
    State("Critical", description="Acute physiological instability, severe decompensation, or high risk"),
    State("Recovery", description="Stabilizing post-decompensation or post-intervention state"),
    State("Discharged", description="Terminal absorbing state representing clinical discharge or monitoring end"),
]

STATE_NAMES: List[str] = [s.name for s in HEALTHCARE_STATES]


# ---------------------------------------------------------------------------
# Base Transition Matrix
# ---------------------------------------------------------------------------

# Indices: [Baseline (0), Elevated (1), Critical (2), Recovery (3), Discharged (4)]
# Discharged is strictly absorbing: P(Discharged -> Discharged) = 1.0.
BASE_HEALTHCARE_TRANSITION_MATRIX: np.ndarray = np.array(
    [
        # Baseline -> [Baseline, Elevated, Critical, Recovery, Discharged]
        [0.70, 0.18, 0.02, 0.05, 0.05],
        # Elevated ->
        [0.25, 0.45, 0.15, 0.13, 0.02],
        # Critical ->
        [0.05, 0.20, 0.50, 0.25, 0.00],
        # Recovery ->
        [0.30, 0.15, 0.05, 0.35, 0.15],
        # Discharged -> absorbing
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Patient Cohorts / Profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HealthcareCohort:
    """Configuration parameters defining a synthetic patient cohort profile."""

    name: str
    description: str
    base_hr_loc: float          # Baseline heart rate mean (bpm)
    base_hr_scale: float        # Heart rate variance scale
    base_sbp_loc: float         # Baseline systolic BP mean (mmHg)
    base_sbp_scale: float       # Systolic BP variance scale
    base_spo2_loc: float        # Baseline SpO2 mean (pct)
    base_temp_loc: float        # Baseline body temperature mean (deg C)
    acute_risk_multiplier: float  # Multiplier for transitions into Elevated/Critical
    recovery_rate_multiplier: float  # Multiplier for transitions into Recovery/Discharged
    mobility_scale: float       # Scale multiplier for mobility score


PATIENT_COHORTS: Dict[str, HealthcareCohort] = {
    "stable_patient": HealthcareCohort(
        name="stable_patient",
        description="Low acute-risk patient with stable vitals, high recovery rate, and normal mobility",
        base_hr_loc=72.0,
        base_hr_scale=5.0,
        base_sbp_loc=118.0,
        base_sbp_scale=6.0,
        base_spo2_loc=98.5,
        base_temp_loc=36.7,
        acute_risk_multiplier=0.5,
        recovery_rate_multiplier=1.4,
        mobility_scale=1.2,
    ),
    "chronic_risk": HealthcareCohort(
        name="chronic_risk",
        description="Patient with underlying chronic conditions, frequent elevated vitals, and elevated acute risk",
        base_hr_loc=82.0,
        base_hr_scale=7.0,
        base_sbp_loc=135.0,
        base_sbp_scale=10.0,
        base_spo2_loc=95.5,
        base_temp_loc=36.8,
        acute_risk_multiplier=1.6,
        recovery_rate_multiplier=0.8,
        mobility_scale=0.8,
    ),
    "post_operative": HealthcareCohort(
        name="post_operative",
        description="Surgical recovery patient with vital volatility, moderate pain/tachycardia, and dynamic recovery",
        base_hr_loc=88.0,
        base_hr_scale=8.0,
        base_sbp_loc=128.0,
        base_sbp_scale=9.0,
        base_spo2_loc=96.5,
        base_temp_loc=37.2,
        acute_risk_multiplier=1.2,
        recovery_rate_multiplier=1.1,
        mobility_scale=0.6,
    ),
    "geriatric_frail": HealthcareCohort(
        name="geriatric_frail",
        description="Elderly patient with high vulnerability, slower physiological recovery, and reduced mobility",
        base_hr_loc=78.0,
        base_hr_scale=6.0,
        base_sbp_loc=142.0,
        base_sbp_scale=12.0,
        base_spo2_loc=94.5,
        base_temp_loc=36.5,
        acute_risk_multiplier=1.8,
        recovery_rate_multiplier=0.6,
        mobility_scale=0.4,
    ),
}


def _build_cohort_transition_matrix(cohort: HealthcareCohort) -> np.ndarray:
    """Derive a cohort-specific transition matrix respecting acute risk and recovery rates."""
    matrix = BASE_HEALTHCARE_TRANSITION_MATRIX.copy()

    # Acute risk alters transitions from Baseline & Elevated into Critical
    if cohort.acute_risk_multiplier > 1.0:
        extra_crit = 0.04 * (cohort.acute_risk_multiplier - 1.0)
        matrix[0, 2] += extra_crit
        matrix[0, 0] -= extra_crit

        extra_elev = 0.06 * (cohort.acute_risk_multiplier - 1.0)
        matrix[1, 2] += extra_elev
        matrix[1, 0] -= extra_elev
    elif cohort.acute_risk_multiplier < 1.0:
        reduction = 0.01 * (1.0 - cohort.acute_risk_multiplier)
        matrix[0, 2] = max(0.005, matrix[0, 2] - reduction)
        matrix[0, 0] += reduction

    # Recovery rate alters transitions into Recovery & Discharged
    if cohort.recovery_rate_multiplier > 1.0:
        boost = 0.05 * (cohort.recovery_rate_multiplier - 1.0)
        matrix[3, 4] += boost  # Recovery -> Discharged
        matrix[3, 3] -= boost
    elif cohort.recovery_rate_multiplier < 1.0:
        slowdown = 0.06 * (1.0 - cohort.recovery_rate_multiplier)
        matrix[3, 4] = max(0.02, matrix[3, 4] - slowdown)
        matrix[3, 3] += slowdown

    # Re-normalize rows to ensure exact stochasticity (rows sum to 1.0)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix


def _build_cohort_emissions(cohort: HealthcareCohort) -> Dict[str, Dict[str, FeatureDistribution]]:
    """Build state-dependent vital telemetry distributions for a patient cohort."""
    return {
        "Baseline": {
            "heart_rate_bpm": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_hr_loc), "scale": float(cohort.base_hr_scale)},
            ),
            "systolic_bp": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_sbp_loc), "scale": float(cohort.base_sbp_scale)},
            ),
            "spo2_pct": FeatureDistribution(
                "uniform",
                {"low": float(max(92.0, cohort.base_spo2_loc - 2.0)), "high": 100.0},
            ),
            "temperature_c": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_temp_loc), "scale": 0.25},
            ),
            "alert_triggered": FeatureDistribution(
                "bernoulli",
                {"p": 0.02},
            ),
            "mobility_score": FeatureDistribution(
                "poisson",
                {"lam": max(1.0, float(8.0 * cohort.mobility_scale))},
            ),
        },
        "Elevated": {
            "heart_rate_bpm": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_hr_loc + 22.0), "scale": float(cohort.base_hr_scale * 1.3)},
            ),
            "systolic_bp": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_sbp_loc + 18.0), "scale": float(cohort.base_sbp_scale * 1.2)},
            ),
            "spo2_pct": FeatureDistribution(
                "uniform",
                {"low": float(max(88.0, cohort.base_spo2_loc - 5.0)), "high": 96.0},
            ),
            "temperature_c": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_temp_loc + 0.8), "scale": 0.35},
            ),
            "alert_triggered": FeatureDistribution(
                "bernoulli",
                {"p": 0.25},
            ),
            "mobility_score": FeatureDistribution(
                "poisson",
                {"lam": max(0.5, float(4.0 * cohort.mobility_scale))},
            ),
        },
        "Critical": {
            "heart_rate_bpm": FeatureDistribution(
                "normal",
                {"loc": float(min(160.0, cohort.base_hr_loc + 55.0)), "scale": float(cohort.base_hr_scale * 1.8)},
            ),
            "systolic_bp": FeatureDistribution(
                "normal",
                {"loc": float(min(185.0, cohort.base_sbp_loc + 40.0)), "scale": float(cohort.base_sbp_scale * 1.5)},
            ),
            "spo2_pct": FeatureDistribution(
                "uniform",
                {"low": 80.0, "high": 89.5},
            ),
            "temperature_c": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_temp_loc + 1.8), "scale": 0.5},
            ),
            "alert_triggered": FeatureDistribution(
                "bernoulli",
                {"p": 0.88},
            ),
            "mobility_score": FeatureDistribution(
                "uniform_discrete",
                {"items": [0]},
            ),
        },
        "Recovery": {
            "heart_rate_bpm": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_hr_loc + 8.0), "scale": float(cohort.base_hr_scale * 1.1)},
            ),
            "systolic_bp": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_sbp_loc + 6.0), "scale": float(cohort.base_sbp_scale * 1.1)},
            ),
            "spo2_pct": FeatureDistribution(
                "uniform",
                {"low": float(max(92.0, cohort.base_spo2_loc - 1.5)), "high": 98.5},
            ),
            "temperature_c": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_temp_loc + 0.2), "scale": 0.25},
            ),
            "alert_triggered": FeatureDistribution(
                "bernoulli",
                {"p": 0.05},
            ),
            "mobility_score": FeatureDistribution(
                "poisson",
                {"lam": max(1.0, float(5.0 * cohort.mobility_scale))},
            ),
        },
        "Discharged": {
            "heart_rate_bpm": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_hr_loc), "scale": float(cohort.base_hr_scale * 0.8)},
            ),
            "systolic_bp": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_sbp_loc), "scale": float(cohort.base_sbp_scale * 0.8)},
            ),
            "spo2_pct": FeatureDistribution(
                "uniform",
                {"low": 97.0, "high": 100.0},
            ),
            "temperature_c": FeatureDistribution(
                "normal",
                {"loc": float(cohort.base_temp_loc), "scale": 0.15},
            ),
            "alert_triggered": FeatureDistribution(
                "uniform_discrete",
                {"items": [0]},
            ),
            "mobility_score": FeatureDistribution(
                "uniform_discrete",
                {"items": [0]},
            ),
        },
    }


# ---------------------------------------------------------------------------
# Causal Transition Rules
# ---------------------------------------------------------------------------

def _sustained_hypoxia_tachycardia_rule(history: HistoryContext) -> bool:
    """Transition rule: Sustained hypoxia (SpO2 < 90) or tachycardia (HR > 130) triggers Critical.

    Inspects only causal prior history across the previous 2 completed interactions.
    """
    recent_spo2 = history.get_recent("spo2_pct", 2)
    recent_hr = history.get_recent("heart_rate_bpm", 2)

    if len(recent_spo2) >= 2 and all(s is not None and s < 90.0 for s in recent_spo2):
        return True
    if len(recent_hr) >= 2 and all(h is not None and h > 130.0 for h in recent_hr):
        return True
    return False


def _sustained_recovery_stability_rule(history: HistoryContext) -> bool:
    """Transition rule: 3 consecutive stable observations in Recovery allows discharge.

    Inspects only causal prior history: SpO2 >= 95 and HR < 95.
    """
    recent_spo2 = history.get_recent("spo2_pct", 3)
    recent_hr = history.get_recent("heart_rate_bpm", 3)

    if len(recent_spo2) >= 3 and len(recent_hr) >= 3:
        spo2_stable = all(s is not None and s >= 95.0 for s in recent_spo2)
        hr_stable = all(h is not None and h < 95.0 for h in recent_hr)
        return spo2_stable and hr_stable
    return False


def build_healthcare_transition_rules() -> List[TransitionRule]:
    """Construct causal transition rules for synthetic healthcare monitoring."""
    return [
        TransitionRule(
            condition=_sustained_hypoxia_tachycardia_rule,
            target_state="Critical",
            probability=0.85,
        ),
        TransitionRule(
            condition=_sustained_recovery_stability_rule,
            target_state="Discharged",
            probability=0.70,
        ),
    ]


# ---------------------------------------------------------------------------
# Profile Construction
# ---------------------------------------------------------------------------

def create_healthcare_profile(cohort: HealthcareCohort) -> Profile:
    """Construct a core Profile from a HealthcareCohort."""
    matrix = _build_cohort_transition_matrix(cohort)
    emissions = _build_cohort_emissions(cohort)
    rules = build_healthcare_transition_rules()

    return Profile(
        name=cohort.name,
        state_emissions=emissions,
        transition_matrix=matrix,
        transition_rules=rules,
        metadata={"cohort": cohort, "domain": "healthcare"},
    )


HEALTHCARE_PROFILES: Dict[str, Profile] = {
    name: create_healthcare_profile(cohort)
    for name, cohort in PATIENT_COHORTS.items()
}


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

def create_healthcare_simulator(
    profile: Optional[Union[str, HealthcareCohort]] = None,
    initial_state: Optional[str] = None,
    **kwargs: Any,
) -> Simulator:
    """Factory creating a Simulator configured with the Healthcare monitoring preset.

    DISCLAIMER:
    This simulator models synthetic behavioral telemetry for evaluation, benchmarking,
    and research. It is NOT clinically validated, does NOT predict real medical outcomes,
    and must not be used for diagnosis, clinical decision support, or treatment.

    Args:
        profile: Patient cohort name (str) or HealthcareCohort instance. If None, defaults to 'stable_patient'.
        initial_state: Starting state name (must be in HEALTHCARE_STATES). Defaults to 'Baseline'.
        **kwargs: Additional parameters passed to Simulator constructor.

    Returns:
        Configured Simulator instance for healthcare monitoring.

    Raises:
        ValueError: If profile name or initial_state is unrecognized.
        TypeError: If profile is not a string, HealthcareCohort, or None.
    """
    if "profile_name" in kwargs:
        profile = kwargs.pop("profile_name")

    if profile is None:
        target_profile_name = "stable_patient"
        core_profile = HEALTHCARE_PROFILES[target_profile_name]
    elif isinstance(profile, str):
        cleaned = profile.strip()
        if cleaned not in PATIENT_COHORTS:
            raise ValueError(
                f"Unknown healthcare profile '{cleaned}'. Available profiles: {sorted(PATIENT_COHORTS.keys())}"
            )
        core_profile = HEALTHCARE_PROFILES[cleaned]
    elif isinstance(profile, HealthcareCohort):
        core_profile = create_healthcare_profile(profile)
    else:
        raise TypeError(
            f"profile must be a str, HealthcareCohort, or None, got {type(profile).__name__}."
        )

    resolved_initial_state = initial_state if initial_state is not None else "Baseline"
    if resolved_initial_state not in STATE_NAMES:
        raise ValueError(
            f"Unknown initial_state '{resolved_initial_state}'. Must be one of: {STATE_NAMES}"
        )

    return Simulator(
        states=HEALTHCARE_STATES,
        profile=core_profile,
        initial_state=resolved_initial_state,
        **kwargs,
    )
