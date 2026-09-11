"""Finance and transaction behavioral preset for BehaviorSim.

This preset defines a synthetic behavioral telemetry model representing investor and portfolio
dynamics across latent financial states (Stable, Active, Volatile, Drawdown, Recovered, Closed).
It captures synthetic behavioral time-series observations including portfolio value, daily return,
transaction count, trade volume, volatility, drawdown, and risk alerts.

DISCLAIMER & SYNTHETIC BOUNDARY:
This is a synthetic behavioral telemetry model designed solely for simulation, benchmarking,
research, and machine learning experimentation. It is not a real-market model, is not
financially validated, and does not predict real market behavior. This preset does not
constitute investment advice, trading advice, or a recommendation to buy or sell securities
or any financial instrument.
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

FINANCE_STATES: List[State] = [
    State("Stable", description="Low return variance, steady portfolio value, predictable low transaction activity, and minimal drawdown"),
    State("Active", description="Normal to elevated portfolio management, routine rebalancing, and regular transaction frequency"),
    State("Volatile", description="Heightened return variance, wide dispersion, increased transaction velocity, and elevated risk exposure"),
    State("Drawdown", description="Capital contraction, sustained negative returns, elevated drawdown magnitude, and high risk-alert propensity"),
    State("Recovered", description="Post-drawdown rebound regime, stabilizing positive returns, normalizing volatility, and gradual activity restoration"),
    State("Closed", description="Terminal absorbing state representing account closure, full liquidation, or monitoring termination"),
]

STATE_NAMES: List[str] = [s.name for s in FINANCE_STATES]


# ---------------------------------------------------------------------------
# Base Transition Matrix
# ---------------------------------------------------------------------------

# Indices: [Stable (0), Active (1), Volatile (2), Drawdown (3), Recovered (4), Closed (5)]
# Closed is strictly absorbing: P(Closed -> Closed) = 1.0.
BASE_FINANCE_TRANSITION_MATRIX: np.ndarray = np.array(
    [
        # Stable -> [Stable, Active, Volatile, Drawdown, Recovered, Closed]
        [0.70, 0.20, 0.05, 0.02, 0.02, 0.01],
        # Active ->
        [0.25, 0.50, 0.15, 0.06, 0.03, 0.01],
        # Volatile ->
        [0.10, 0.25, 0.40, 0.18, 0.05, 0.02],
        # Drawdown ->
        [0.05, 0.10, 0.20, 0.45, 0.18, 0.02],
        # Recovered ->
        [0.35, 0.30, 0.10, 0.05, 0.19, 0.01],
        # Closed -> absorbing
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Investor Personas / Cohorts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FinanceCohort:
    """Configuration parameters defining a synthetic investor profile."""

    name: str
    description: str
    base_portfolio_value: float       # Baseline portfolio valuation ($)
    activity_multiplier: float        # Multiplier for transaction count
    volume_scale: float               # Exponential scale for trade volume ($)
    volatility_multiplier: float      # Multiplier for synthetic return variance & volatility
    risk_alert_scale: float           # Multiplier for risk alert probability
    drawdown_susceptibility: float    # Susceptibility to severe drawdown states
    recovery_tendency: float          # Propensity to recover towards Stable/Recovered


INVESTOR_COHORTS: Dict[str, FinanceCohort] = {
    "conservative_investor": FinanceCohort(
        name="conservative_investor",
        description="Low transaction activity, modest trade volume, lower volatility exposure, and strong stability tendency",
        base_portfolio_value=100_000.0,
        activity_multiplier=0.4,
        volume_scale=1_500.0,
        volatility_multiplier=0.6,
        risk_alert_scale=0.4,
        drawdown_susceptibility=0.4,
        recovery_tendency=1.6,
    ),
    "balanced_investor": FinanceCohort(
        name="balanced_investor",
        description="Moderate activity, balanced volatility exposure, and standard transition dynamics (baseline profile)",
        base_portfolio_value=100_000.0,
        activity_multiplier=1.0,
        volume_scale=6_000.0,
        volatility_multiplier=1.0,
        risk_alert_scale=1.0,
        drawdown_susceptibility=1.0,
        recovery_tendency=1.0,
    ),
    "growth_investor": FinanceCohort(
        name="growth_investor",
        description="Higher transaction activity, greater volatility exposure, and elevated willingness to remain Active/Volatile",
        base_portfolio_value=100_000.0,
        activity_multiplier=1.5,
        volume_scale=15_000.0,
        volatility_multiplier=1.35,
        risk_alert_scale=1.3,
        drawdown_susceptibility=1.25,
        recovery_tendency=0.9,
    ),
    "active_trader": FinanceCohort(
        name="active_trader",
        description="Highest transaction frequency, highest trade volume, elevated volatility exposure, and frequent excursions into Volatile/Drawdown",
        base_portfolio_value=100_000.0,
        activity_multiplier=2.8,
        volume_scale=40_000.0,
        volatility_multiplier=1.8,
        risk_alert_scale=2.2,
        drawdown_susceptibility=1.8,
        recovery_tendency=0.7,
    ),
}


# ---------------------------------------------------------------------------
# Transition Matrix Construction
# ---------------------------------------------------------------------------

def _build_cohort_transition_matrix(cohort: FinanceCohort) -> np.ndarray:
    """Derive a profile-specific row-stochastic transition matrix from the base matrix."""
    matrix = BASE_FINANCE_TRANSITION_MATRIX.copy()

    # Apply cohort multipliers to non-absorbing states (indices 0..4)
    # Row 0: Stable
    matrix[0, 0] *= cohort.recovery_tendency
    matrix[0, 2] *= cohort.drawdown_susceptibility
    matrix[0, 3] *= cohort.drawdown_susceptibility

    # Row 1: Active
    matrix[1, 1] *= (1.0 + 0.1 * (cohort.activity_multiplier - 1.0))
    matrix[1, 2] *= cohort.drawdown_susceptibility
    matrix[1, 3] *= cohort.drawdown_susceptibility

    # Row 2: Volatile
    matrix[2, 2] *= (1.0 + 0.15 * (cohort.volatility_multiplier - 1.0))
    matrix[2, 3] *= cohort.drawdown_susceptibility
    matrix[2, 0] *= cohort.recovery_tendency
    matrix[2, 4] *= cohort.recovery_tendency

    # Row 3: Drawdown
    matrix[3, 3] *= cohort.drawdown_susceptibility
    matrix[3, 4] *= cohort.recovery_tendency
    matrix[3, 0] *= cohort.recovery_tendency

    # Row 4: Recovered
    matrix[4, 0] *= cohort.recovery_tendency
    matrix[4, 2] *= cohort.drawdown_susceptibility
    matrix[4, 3] *= cohort.drawdown_susceptibility

    # Ensure non-negative entries
    matrix = np.clip(matrix, 0.0, None)

    # Row 5 (Closed) is strictly absorbing
    matrix[5, :] = 0.0
    matrix[5, 5] = 1.0

    # Re-normalize rows to ensure exact stochasticity (rows sum to 1.0)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix


# ---------------------------------------------------------------------------
# Emission Model Construction
# ---------------------------------------------------------------------------

def _build_cohort_emissions(cohort: FinanceCohort) -> Dict[str, Dict[str, FeatureDistribution]]:
    """Build state-dependent financial behavioral distributions for an investor cohort."""
    base_val = cohort.base_portfolio_value
    vol_m = cohort.volatility_multiplier
    act_m = cohort.activity_multiplier
    vol_scale = cohort.volume_scale
    alert_m = cohort.risk_alert_scale
    dd_susc = cohort.drawdown_susceptibility

    return {
        "Stable": {
            "portfolio_value": FeatureDistribution(
                "lognormal",
                {"mean": float(np.log(base_val)), "sigma": float(0.015 * vol_m)},
            ),
            "daily_return": FeatureDistribution(
                "normal",
                {"loc": 0.0005, "scale": float(0.006 * vol_m)},
            ),
            "transaction_count": FeatureDistribution(
                "poisson",
                {"lam": max(0.5, float(2.0 * act_m))},
            ),
            "trade_volume": FeatureDistribution(
                "exponential",
                {"scale": max(10.0, float(vol_scale * 0.25))},
            ),
            "volatility": FeatureDistribution(
                "uniform",
                {"low": float(0.06 * vol_m), "high": float(0.14 * vol_m)},
            ),
            "drawdown": FeatureDistribution(
                "uniform",
                {"low": 0.00, "high": float(min(0.08, 0.04 * dd_susc))},
            ),
            "risk_alert": FeatureDistribution(
                "bernoulli",
                {"p": float(min(0.05, 0.01 * alert_m))},
            ),
        },
        "Active": {
            "portfolio_value": FeatureDistribution(
                "lognormal",
                {"mean": float(np.log(base_val * 1.02)), "sigma": float(0.025 * vol_m)},
            ),
            "daily_return": FeatureDistribution(
                "normal",
                {"loc": 0.0012, "scale": float(0.012 * vol_m)},
            ),
            "transaction_count": FeatureDistribution(
                "poisson",
                {"lam": max(1.0, float(7.0 * act_m))},
            ),
            "trade_volume": FeatureDistribution(
                "exponential",
                {"scale": max(50.0, float(vol_scale * 1.0))},
            ),
            "volatility": FeatureDistribution(
                "uniform",
                {"low": float(0.14 * vol_m), "high": float(0.28 * vol_m)},
            ),
            "drawdown": FeatureDistribution(
                "uniform",
                {"low": 0.01, "high": float(min(0.18, 0.08 * dd_susc))},
            ),
            "risk_alert": FeatureDistribution(
                "bernoulli",
                {"p": float(min(0.25, 0.05 * alert_m))},
            ),
        },
        "Volatile": {
            "portfolio_value": FeatureDistribution(
                "lognormal",
                {"mean": float(np.log(base_val * 0.98)), "sigma": float(0.050 * vol_m)},
            ),
            "daily_return": FeatureDistribution(
                "normal",
                {"loc": -0.0020, "scale": float(0.028 * vol_m)},
            ),
            "transaction_count": FeatureDistribution(
                "poisson",
                {"lam": max(2.0, float(12.0 * act_m))},
            ),
            "trade_volume": FeatureDistribution(
                "exponential",
                {"scale": max(100.0, float(vol_scale * 2.0))},
            ),
            "volatility": FeatureDistribution(
                "uniform",
                {"low": float(0.30 * vol_m), "high": float(0.65 * vol_m)},
            ),
            "drawdown": FeatureDistribution(
                "uniform",
                {"low": 0.08, "high": float(min(0.45, 0.22 * dd_susc))},
            ),
            "risk_alert": FeatureDistribution(
                "bernoulli",
                {"p": float(min(0.90, 0.40 * alert_m))},
            ),
        },
        "Drawdown": {
            "portfolio_value": FeatureDistribution(
                "lognormal",
                {"mean": float(np.log(base_val * 0.85)), "sigma": float(0.040 * vol_m)},
            ),
            "daily_return": FeatureDistribution(
                "normal",
                {"loc": -0.0180, "scale": float(0.022 * vol_m)},
            ),
            "transaction_count": FeatureDistribution(
                "poisson",
                {"lam": max(1.0, float(5.0 * act_m))},
            ),
            "trade_volume": FeatureDistribution(
                "exponential",
                {"scale": max(30.0, float(vol_scale * 0.8))},
            ),
            "volatility": FeatureDistribution(
                "uniform",
                {"low": float(0.25 * vol_m), "high": float(0.55 * vol_m)},
            ),
            "drawdown": FeatureDistribution(
                "uniform",
                {"low": float(min(0.40, 0.25 * min(1.5, dd_susc))), "high": float(min(0.95, 0.55 * min(1.5, dd_susc)))},
            ),
            "risk_alert": FeatureDistribution(
                "bernoulli",
                {"p": float(min(0.98, 0.75 * alert_m))},
            ),
        },
        "Recovered": {
            "portfolio_value": FeatureDistribution(
                "lognormal",
                {"mean": float(np.log(base_val * 0.96)), "sigma": float(0.020 * vol_m)},
            ),
            "daily_return": FeatureDistribution(
                "normal",
                {"loc": 0.0075, "scale": float(0.010 * vol_m)},
            ),
            "transaction_count": FeatureDistribution(
                "poisson",
                {"lam": max(1.0, float(4.0 * act_m))},
            ),
            "trade_volume": FeatureDistribution(
                "exponential",
                {"scale": max(25.0, float(vol_scale * 0.6))},
            ),
            "volatility": FeatureDistribution(
                "uniform",
                {"low": float(0.10 * vol_m), "high": float(0.22 * vol_m)},
            ),
            "drawdown": FeatureDistribution(
                "uniform",
                {"low": 0.02, "high": 0.12},
            ),
            "risk_alert": FeatureDistribution(
                "bernoulli",
                {"p": float(min(0.30, 0.08 * alert_m))},
            ),
        },
        "Closed": {
            "portfolio_value": FeatureDistribution(
                "lognormal",
                {"mean": float(np.log(base_val * 0.80)), "sigma": 0.001},
            ),
            "daily_return": FeatureDistribution(
                "normal",
                {"loc": 0.0, "scale": 0.0001},
            ),
            "transaction_count": FeatureDistribution(
                "poisson",
                {"lam": 0.0},
            ),
            "trade_volume": FeatureDistribution(
                "exponential",
                {"scale": 0.0001},
            ),
            "volatility": FeatureDistribution(
                "uniform",
                {"low": 0.0, "high": 0.0},
            ),
            "drawdown": FeatureDistribution(
                "uniform",
                {"low": 0.0, "high": 0.0},
            ),
            "risk_alert": FeatureDistribution(
                "bernoulli",
                {"p": 0.0},
            ),
        },
    }


# ---------------------------------------------------------------------------
# Causal Transition Rules
# ---------------------------------------------------------------------------

def _sustained_drawdown_rule(history: HistoryContext) -> bool:
    """Transition rule: Sustained negative returns or acute drawdown triggers Drawdown.

    Inspects only causal prior history:
    - At least 2 consecutive prior daily_return observations < -0.01 (-1.0%), OR
    - Prior drawdown >= 0.20 (20% drawdown).
    """
    recent_returns = history.get_recent("daily_return", 2)
    recent_drawdown = history.get_recent("drawdown", 1)

    if len(recent_returns) >= 2 and all(r is not None and r < -0.01 for r in recent_returns):
        return True
    if len(recent_drawdown) >= 1 and any(d is not None and d >= 0.20 for d in recent_drawdown):
        return True
    return False


def _volatility_escalation_rule(history: HistoryContext) -> bool:
    """Transition rule: Consecutive high volatility observations trigger Volatile.

    Inspects only causal prior history:
    - At least 2 consecutive prior volatility observations >= 0.35.
    """
    recent_vol = history.get_recent("volatility", 2)
    if len(recent_vol) >= 2 and all(v is not None and v >= 0.35 for v in recent_vol):
        return True
    return False


def _recovery_rule(history: HistoryContext) -> bool:
    """Transition rule: Stabilizing positive returns and receding drawdown triggers Recovered.

    Inspects only causal prior history:
    - At least 2 consecutive prior daily_return observations > 0.003 (+0.3%), AND
    - Recent drawdown <= 0.15, AND
    - History exhibits prior drawdown experience (at least one drawdown >= 0.20 in last 5 steps).
    """
    recent_returns = history.get_recent("daily_return", 2)
    recent_drawdown = history.get_recent("drawdown", 1)
    window_drawdown = history.get_recent("drawdown", 5)

    if len(recent_returns) >= 2 and len(recent_drawdown) >= 1:
        returns_positive = all(r is not None and r > 0.003 for r in recent_returns)
        drawdown_receding = all(d is not None and d <= 0.15 for d in recent_drawdown)
        experienced_drawdown = any(d is not None and d >= 0.20 for d in window_drawdown)
        return returns_positive and drawdown_receding and experienced_drawdown
    return False


def build_finance_transition_rules() -> List[TransitionRule]:
    """Construct causal transition rules for synthetic financial behavior."""
    return [
        TransitionRule(
            condition=_sustained_drawdown_rule,
            target_state="Drawdown",
            probability=0.80,
        ),
        TransitionRule(
            condition=_volatility_escalation_rule,
            target_state="Volatile",
            probability=0.75,
        ),
        TransitionRule(
            condition=_recovery_rule,
            target_state="Recovered",
            probability=0.70,
        ),
    ]


# ---------------------------------------------------------------------------
# Profile Construction
# ---------------------------------------------------------------------------

def create_finance_profile(cohort: FinanceCohort) -> Profile:
    """Construct a core Profile from a FinanceCohort."""
    matrix = _build_cohort_transition_matrix(cohort)
    emissions = _build_cohort_emissions(cohort)
    rules = build_finance_transition_rules()

    return Profile(
        name=cohort.name,
        state_emissions=emissions,
        transition_matrix=matrix,
        transition_rules=rules,
        metadata={"cohort": cohort, "domain": "finance"},
    )


FINANCE_PROFILES: Dict[str, Profile] = {
    name: create_finance_profile(cohort)
    for name, cohort in INVESTOR_COHORTS.items()
}


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

def create_finance_simulator(
    profile: Optional[Union[str, FinanceCohort]] = None,
    initial_state: Optional[str] = None,
    **kwargs: Any,
) -> Simulator:
    """Factory creating a Simulator configured with the Finance behavioral preset.

    DISCLAIMER & SYNTHETIC BOUNDARY:
    This simulator models synthetic behavioral telemetry for simulation, benchmarking,
    research, and machine learning experimentation. It is not a real-market model,
    is not financially validated, does not predict real market behavior, and does not
    constitute investment advice, trading advice, or a recommendation to buy or sell securities.

    Args:
        profile: Investor persona name (str) or FinanceCohort instance. If None, defaults to 'balanced_investor'.
        initial_state: Starting state name (must be in FINANCE_STATES). Defaults to 'Stable'.
        **kwargs: Additional parameters passed to Simulator constructor.

    Returns:
        Configured Simulator instance for financial behavior simulation.

    Raises:
        ValueError: If profile name or initial_state is unrecognized.
        TypeError: If profile is not a string, FinanceCohort, or None.
    """
    if "profile_name" in kwargs:
        profile = kwargs.pop("profile_name")

    if profile is None:
        target_profile_name = "balanced_investor"
        core_profile = FINANCE_PROFILES[target_profile_name]
    elif isinstance(profile, str):
        cleaned = profile.strip()
        if cleaned not in INVESTOR_COHORTS:
            raise ValueError(
                f"Unknown finance profile '{cleaned}'. Available profiles: {sorted(INVESTOR_COHORTS.keys())}"
            )
        core_profile = FINANCE_PROFILES[cleaned]
    elif isinstance(profile, FinanceCohort):
        core_profile = create_finance_profile(profile)
    else:
        raise TypeError(
            f"profile must be a str, FinanceCohort, or None, got {type(profile).__name__}."
        )

    resolved_initial_state = initial_state if initial_state is not None else "Stable"
    if resolved_initial_state not in STATE_NAMES:
        raise ValueError(
            f"Unknown initial_state '{resolved_initial_state}'. Must be one of: {STATE_NAMES}"
        )

    return Simulator(
        states=FINANCE_STATES,
        profile=core_profile,
        initial_state=resolved_initial_state,
        **kwargs,
    )
