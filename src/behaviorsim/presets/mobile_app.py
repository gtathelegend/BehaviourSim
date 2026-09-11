"""Mobile app user engagement preset for BehaviorSim.

This preset defines a synthetic behavioral model representing user engagement dynamics
within a digital mobile application. It captures user transitions across latent session
engagement states (Browsing, ActiveSession, CheckoutFlow, Idle, Churned) with persona-specific
telemetry emissions (session duration, action count, scroll depth, button clicks,
notification interactions, and cart value).

NOTE: This is a synthetic behavioral simulation model designed for benchmarking, testing,
and algorithm evaluation. It is not calibrated to or reflective of any specific commercial
mobile application or empirical user dataset.
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

MOBILE_APP_STATES: List[State] = [
    State("Browsing", description="Casual catalog/feed exploration with moderate interaction"),
    State("ActiveSession", description="High-engagement active session with frequent interactions"),
    State("CheckoutFlow", description="Conversion-oriented funnel exploring purchases and cart"),
    State("Idle", description="Passive/low-activity state with backgrounded or dormant app"),
    State("Churned", description="Inactive/uninstalled absorbing or quasi-absorbing state"),
]

STATE_NAMES: List[str] = [s.name for s in MOBILE_APP_STATES]


# ---------------------------------------------------------------------------
# Base Transition Matrix
# ---------------------------------------------------------------------------

# Rows sum to 1.0; indices: [Browsing (0), ActiveSession (1), CheckoutFlow (2), Idle (3), Churned (4)]
# Churned is an absorbing state (P(Churned -> Churned) = 1.0).
BASE_MOBILE_APP_TRANSITION_MATRIX: np.ndarray = np.array(
    [
        # Browsing -> [Browsing, ActiveSession, CheckoutFlow, Idle, Churned]
        [0.45, 0.25, 0.10, 0.18, 0.02],
        # ActiveSession ->
        [0.20, 0.50, 0.20, 0.08, 0.02],
        # CheckoutFlow ->
        [0.30, 0.20, 0.35, 0.14, 0.01],
        # Idle ->
        [0.25, 0.10, 0.05, 0.50, 0.10],
        # Churned -> absorbing
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Personas & Feature Emission Profiles
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MobileAppPersona:
    """Configuration parameters defining a synthetic mobile app user persona."""

    name: str
    description: str
    session_time_scale: float  # Scale factor for session duration (lognormal / exponential)
    action_intensity: float    # Intensity multiplier for actions and button clicks
    checkout_prob: float       # Propensity to explore checkout
    cart_mean: float           # Typical cart value when shopping
    notification_responsiveness: float  # Probability of clicking notifications
    churn_resistance: float    # Resistance to idling and churn (1.0 = high resistance)


PERSONAS: Dict[str, MobileAppPersona] = {
    "power_user": MobileAppPersona(
        name="power_user",
        description="Highly engaged daily user with intense interaction and frequent checkouts",
        session_time_scale=2.0,
        action_intensity=2.2,
        checkout_prob=0.35,
        cart_mean=85.0,
        notification_responsiveness=0.55,
        churn_resistance=0.95,
    ),
    "casual_browser": MobileAppPersona(
        name="casual_browser",
        description="Regular user browsing content and feeds with moderate engagement",
        session_time_scale=1.0,
        action_intensity=1.0,
        checkout_prob=0.15,
        cart_mean=35.0,
        notification_responsiveness=0.20,
        churn_resistance=0.75,
    ),
    "deal_seeker": MobileAppPersona(
        name="deal_seeker",
        description="Discount/promotion-oriented user with high cart interaction and checkout frequency",
        session_time_scale=1.2,
        action_intensity=1.3,
        checkout_prob=0.40,
        cart_mean=45.0,
        notification_responsiveness=0.45,
        churn_resistance=0.80,
    ),
    "infrequent_visitor": MobileAppPersona(
        name="infrequent_visitor",
        description="Sporadic user with low engagement, high idle time, and higher churn risk",
        session_time_scale=0.5,
        action_intensity=0.5,
        checkout_prob=0.05,
        cart_mean=20.0,
        notification_responsiveness=0.08,
        churn_resistance=0.30,
    ),
}


def _build_persona_transition_matrix(persona: MobileAppPersona) -> np.ndarray:
    """Derive a persona-specific transition matrix respecting persona churn resistance."""
    matrix = BASE_MOBILE_APP_TRANSITION_MATRIX.copy()
    if persona.churn_resistance > 0.8:
        # Power user / deal seeker: reduce idle and churn probability from active states
        matrix[0, 1] += 0.05  # More Browsing -> ActiveSession
        matrix[0, 3] -= 0.04
        matrix[0, 4] -= 0.01

        matrix[1, 2] += 0.05  # More ActiveSession -> CheckoutFlow
        matrix[1, 3] -= 0.04
        matrix[1, 4] -= 0.01

        matrix[3, 0] += 0.10  # Idle easily reactivates to Browsing
        matrix[3, 3] -= 0.05
        matrix[3, 4] -= 0.05
    elif persona.churn_resistance < 0.5:
        # Infrequent visitor: higher Idle and Churn tendency
        matrix[0, 3] += 0.08
        matrix[0, 4] += 0.04
        matrix[0, 1] -= 0.08
        matrix[0, 2] -= 0.04

        matrix[1, 3] += 0.06
        matrix[1, 4] += 0.04
        matrix[1, 1] -= 0.10

        matrix[3, 4] += 0.15  # Idle frequently becomes Churned
        matrix[3, 3] -= 0.10
        matrix[3, 0] -= 0.05

    # Re-normalize rows to ensure exact stochasticity
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix


def _build_persona_emissions(persona: MobileAppPersona) -> Dict[str, Dict[str, FeatureDistribution]]:
    """Build state-dependent feature distributions for a persona."""
    return {
        "Browsing": {
            "session_time_seconds": FeatureDistribution(
                "lognormal",
                {"mean": float(np.log(45.0 * persona.session_time_scale)), "sigma": 0.4},
            ),
            "action_count": FeatureDistribution(
                "poisson",
                {"lam": max(1.0, float(12.0 * persona.action_intensity))},
            ),
            "scroll_depth": FeatureDistribution(
                "uniform",
                {"low": 0.15, "high": 0.85},
            ),
            "button_clicks": FeatureDistribution(
                "poisson",
                {"lam": max(1.0, float(5.0 * persona.action_intensity))},
            ),
            "notification_clicked": FeatureDistribution(
                "bernoulli",
                {"p": min(1.0, float(persona.notification_responsiveness * 0.5))},
            ),
            "cart_value": FeatureDistribution(
                "exponential",
                {"scale": float(persona.cart_mean * 0.25)},
            ),
        },
        "ActiveSession": {
            "session_time_seconds": FeatureDistribution(
                "lognormal",
                {"mean": float(np.log(120.0 * persona.session_time_scale)), "sigma": 0.35},
            ),
            "action_count": FeatureDistribution(
                "poisson",
                {"lam": max(2.0, float(30.0 * persona.action_intensity))},
            ),
            "scroll_depth": FeatureDistribution(
                "uniform",
                {"low": 0.40, "high": 0.98},
            ),
            "button_clicks": FeatureDistribution(
                "poisson",
                {"lam": max(2.0, float(15.0 * persona.action_intensity))},
            ),
            "notification_clicked": FeatureDistribution(
                "bernoulli",
                {"p": min(1.0, float(persona.notification_responsiveness * 0.8))},
            ),
            "cart_value": FeatureDistribution(
                "normal",
                {"loc": float(persona.cart_mean * 0.6), "scale": 10.0},
            ),
        },
        "CheckoutFlow": {
            "session_time_seconds": FeatureDistribution(
                "lognormal",
                {"mean": float(np.log(90.0 * persona.session_time_scale)), "sigma": 0.3},
            ),
            "action_count": FeatureDistribution(
                "poisson",
                {"lam": max(2.0, float(20.0 * persona.action_intensity))},
            ),
            "scroll_depth": FeatureDistribution(
                "uniform",
                {"low": 0.60, "high": 1.0},
            ),
            "button_clicks": FeatureDistribution(
                "poisson",
                {"lam": max(3.0, float(18.0 * persona.action_intensity))},
            ),
            "notification_clicked": FeatureDistribution(
                "bernoulli",
                {"p": min(1.0, float(persona.notification_responsiveness))},
            ),
            "cart_value": FeatureDistribution(
                "normal",
                {"loc": float(persona.cart_mean), "scale": max(5.0, persona.cart_mean * 0.2)},
            ),
        },
        "Idle": {
            "session_time_seconds": FeatureDistribution(
                "lognormal",
                {"mean": float(np.log(10.0 * max(0.2, persona.session_time_scale))), "sigma": 0.5},
            ),
            "action_count": FeatureDistribution(
                "poisson",
                {"lam": max(0.2, float(1.0 * persona.action_intensity))},
            ),
            "scroll_depth": FeatureDistribution(
                "uniform",
                {"low": 0.0, "high": 0.25},
            ),
            "button_clicks": FeatureDistribution(
                "poisson",
                {"lam": max(0.1, float(0.5 * persona.action_intensity))},
            ),
            "notification_clicked": FeatureDistribution(
                "bernoulli",
                {"p": min(1.0, float(persona.notification_responsiveness * 0.15))},
            ),
            "cart_value": FeatureDistribution(
                "uniform",
                {"low": 0.0, "high": 0.0},
            ),
        },
        "Churned": {
            "session_time_seconds": FeatureDistribution(
                "uniform",
                {"low": 0.0, "high": 0.0},
            ),
            "action_count": FeatureDistribution(
                "uniform_discrete",
                {"items": [0]},
            ),
            "scroll_depth": FeatureDistribution(
                "uniform",
                {"low": 0.0, "high": 0.0},
            ),
            "button_clicks": FeatureDistribution(
                "uniform_discrete",
                {"items": [0]},
            ),
            "notification_clicked": FeatureDistribution(
                "uniform_discrete",
                {"items": [0]},
            ),
            "cart_value": FeatureDistribution(
                "uniform",
                {"low": 0.0, "high": 0.0},
            ),
        },
    }


# ---------------------------------------------------------------------------
# Transition Rules (Causal History Rules)
# ---------------------------------------------------------------------------

def _consecutive_idle_rule(history: HistoryContext) -> bool:
    """Trigger churn transition if user has been continuously idle with zero actions."""
    recent_actions = history.get_recent("action_count", 4)
    if len(recent_actions) >= 4 and all(a == 0 for a in recent_actions):
        return True
    return False


def _cart_engagement_rule(history: HistoryContext) -> bool:
    """Prompt transition to CheckoutFlow if recent cart value is substantial."""
    recent_cart = history.get_recent("cart_value", 2)
    if len(recent_cart) >= 2 and all(c is not None and c > 50.0 for c in recent_cart):
        return True
    return False


def build_mobile_app_transition_rules() -> List[TransitionRule]:
    """Construct domain transition rules for mobile app engagement."""
    return [
        TransitionRule(
            condition=_consecutive_idle_rule,
            target_state="Churned",
            probability=0.75,
        ),
        TransitionRule(
            condition=_cart_engagement_rule,
            target_state="CheckoutFlow",
            probability=0.60,
        ),
    ]


# ---------------------------------------------------------------------------
# Profile Construction
# ---------------------------------------------------------------------------

def create_mobile_app_profile(persona: MobileAppPersona) -> Profile:
    """Construct a core Profile from a MobileAppPersona."""
    matrix = _build_persona_transition_matrix(persona)
    emissions = _build_persona_emissions(persona)
    rules = build_mobile_app_transition_rules()

    return Profile(
        name=persona.name,
        state_emissions=emissions,
        transition_matrix=matrix,
        transition_rules=rules,
        metadata={"persona": persona, "domain": "mobile_app"},
    )


# Pre-built core profiles for all registered personas
MOBILE_APP_PROFILES: Dict[str, Profile] = {
    name: create_mobile_app_profile(persona)
    for name, persona in PERSONAS.items()
}


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

def create_mobile_app_simulator(
    profile: Optional[Union[str, MobileAppPersona]] = None,
    initial_state: Optional[str] = None,
    **kwargs: Any,
) -> Simulator:
    """Factory creating a Simulator configured with the Mobile App engagement preset.

    Args:
        profile: Persona name (str) or MobileAppPersona instance. If None, defaults to 'casual_browser'.
        initial_state: Optional starting state name (must be in MOBILE_APP_STATES). Defaults to 'Browsing'.
        **kwargs: Additional parameters passed to Simulator constructor.

    Returns:
        Configured Simulator instance for mobile app user engagement.

    Raises:
        ValueError: If profile name or initial_state is unrecognized.
        TypeError: If profile is not a string, MobileAppPersona, or None.
    """
    if "profile_name" in kwargs:
        profile = kwargs.pop("profile_name")

    if profile is None:
        target_profile_name = "casual_browser"
        core_profile = MOBILE_APP_PROFILES[target_profile_name]
    elif isinstance(profile, str):
        cleaned = profile.strip()
        if cleaned not in PERSONAS:
            raise ValueError(
                f"Unknown mobile app profile '{cleaned}'. Available profiles: {sorted(PERSONAS.keys())}"
            )
        core_profile = MOBILE_APP_PROFILES[cleaned]
    elif isinstance(profile, MobileAppPersona):
        core_profile = create_mobile_app_profile(profile)
    else:
        raise TypeError(
            f"profile must be a str, MobileAppPersona, or None, got {type(profile).__name__}."
        )

    resolved_initial_state = initial_state if initial_state is not None else "Browsing"
    if resolved_initial_state not in STATE_NAMES:
        raise ValueError(
            f"Unknown initial_state '{resolved_initial_state}'. Must be one of: {STATE_NAMES}"
        )

    return Simulator(
        states=MOBILE_APP_STATES,
        profile=core_profile,
        initial_state=resolved_initial_state,
        **kwargs,
    )
