"""Unit tests for BehaviorSim TransitionRule and Profile abstractions."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.state import State
from behaviorsim.core.transition import HistoryContext, TransitionRule


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

class FakeContext:
    """Mock context implementing HistoryContext protocol for test evaluation."""

    def __init__(self, data: dict):
        self._data = data

    def get_recent(self, feature_name: str, window: int):
        values = self._data.get(feature_name, [])
        return values[-window:]


# ---------------------------------------------------------------------------
# TransitionRule Tests
# ---------------------------------------------------------------------------

def test_transition_rule_valid_construction() -> None:
    """Verify valid TransitionRule construction and evaluation."""
    rule = TransitionRule(
        condition=lambda ctx: True,
        target_state="TargetState",
        probability=0.75,
    )
    assert rule.target_state == "TargetState"
    assert rule.probability == 0.75
    assert rule.condition(None) is True


def test_transition_rule_evaluation_with_fake_context() -> None:
    """Verify TransitionRule condition evaluates history using HistoryContext protocol."""
    ctx = FakeContext({"error_rate": [0.5, 0.6, 0.5]})
    rule = TransitionRule(
        condition=lambda c: float(np.mean(c.get_recent("error_rate", 3))) > 0.4,
        target_state="HighErrorState",
    )
    assert isinstance(ctx, HistoryContext)
    assert bool(rule.condition(ctx)) is True


def test_transition_rule_invalid_condition() -> None:
    """Verify non-callable condition raises TypeError."""
    with pytest.raises(TypeError, match="must be a callable"):
        TransitionRule(condition="not_callable", target_state="StateA")  # type: ignore[arg-type]


def test_transition_rule_invalid_target_state() -> None:
    """Verify invalid target state raises ValueError."""
    with pytest.raises(ValueError, match="non-empty string"):
        TransitionRule(condition=lambda c: True, target_state="")

    with pytest.raises(ValueError, match="non-empty string"):
        TransitionRule(condition=lambda c: True, target_state="   ")


def test_transition_rule_invalid_probability() -> None:
    """Verify invalid probability values raise TypeError or ValueError."""
    with pytest.raises(TypeError, match="must be numeric"):
        TransitionRule(condition=lambda c: True, target_state="StateA", probability="0.5")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite number"):
        TransitionRule(condition=lambda c: True, target_state="StateA", probability=float("nan"))

    with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
        TransitionRule(condition=lambda c: True, target_state="StateA", probability=-0.1)

    with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
        TransitionRule(condition=lambda c: True, target_state="StateA", probability=1.1)


# ---------------------------------------------------------------------------
# Profile Tests
# ---------------------------------------------------------------------------

def test_profile_valid_minimal_construction() -> None:
    """Verify valid Profile with minimal required parameters."""
    emissions = {
        "StateA": {
            "feature_1": FeatureDistribution("normal", {"loc": 0.0, "scale": 1.0})
        }
    }
    profile = Profile(name="minimal_profile", state_emissions=emissions)
    assert profile.name == "minimal_profile"
    assert profile.state_emissions == emissions
    assert profile.transition_matrix is None
    assert profile.transition_rules is None
    assert profile.metadata is None


def test_profile_valid_full_construction() -> None:
    """Verify valid Profile with transition matrix, rules, and metadata."""
    emissions = {
        "StateA": {"f1": FeatureDistribution("bernoulli", {"p": 0.5})},
        "StateB": {"f1": FeatureDistribution("bernoulli", {"p": 0.1})},
    }
    matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    rule = TransitionRule(condition=lambda c: True, target_state="StateB")

    profile = Profile(
        name="full_profile",
        state_emissions=emissions,
        transition_matrix=matrix,
        transition_rules=[rule],
        metadata={"domain": "test"},
    )
    assert profile.name == "full_profile"
    assert profile.transition_matrix is not None
    assert profile.transition_rules is not None
    assert len(profile.transition_rules) == 1
    assert profile.metadata == {"domain": "test"}


def test_profile_invalid_name() -> None:
    """Verify invalid profile name raises ValueError."""
    with pytest.raises(ValueError, match="non-empty string"):
        Profile(name="", state_emissions={})

    with pytest.raises(ValueError, match="non-empty string"):
        Profile(name="   ", state_emissions={})


def test_profile_invalid_state_emissions() -> None:
    """Verify non-mapping state_emissions raises ValueError."""
    with pytest.raises(ValueError, match="must be a mapping"):
        Profile(name="bad_emissions", state_emissions="not_a_mapping")  # type: ignore[arg-type]


def test_profile_invalid_transition_matrix() -> None:
    """Verify invalid transition matrix delegates to validate_transition_matrix and raises ValueError."""
    invalid_matrix = np.array([[0.5, 0.5], [0.5, 0.8]])  # Row 2 sums to 1.3
    with pytest.raises(ValueError, match="rows must sum to 1.0"):
        Profile(name="bad_matrix", state_emissions={}, transition_matrix=invalid_matrix)


def test_profile_invalid_transition_rules() -> None:
    """Verify non-iterable or non-TransitionRule items in rules raise TypeError."""
    with pytest.raises(TypeError, match="must be an iterable"):
        Profile(name="bad_rules", state_emissions={}, transition_rules=123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="not a TransitionRule instance"):
        Profile(name="bad_rules", state_emissions={}, transition_rules=["not_a_rule"])  # type: ignore[list-item]


def test_profile_invalid_metadata() -> None:
    """Verify non-mapping metadata raises ValueError."""
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        Profile(name="bad_meta", state_emissions={}, metadata="not_a_mapping")  # type: ignore[arg-type]


def test_profile_domain_neutrality() -> None:
    """Verify profiles can represent distinct non-education domains seamlessly."""
    # Domain 1: Mobile App User Engagement
    mobile_profile = Profile(
        name="power_user",
        state_emissions={
            "Engaged": {
                "click_count": FeatureDistribution("poisson", {"lam": 15}),
                "session_duration": FeatureDistribution("lognormal", {"mean": 4.5, "sigma": 0.5}),
            },
            "Idle": {
                "click_count": FeatureDistribution("poisson", {"lam": 1}),
                "session_duration": FeatureDistribution("exponential", {"scale": 10.0}),
            },
        },
    )
    assert "Engaged" in mobile_profile.state_emissions
    assert mobile_profile.state_emissions["Engaged"]["click_count"].distribution_type == "poisson"

    # Domain 2: Financial Transaction Fraud Risk
    finance_profile = Profile(
        name="high_volume_merchant",
        state_emissions={
            "Normal": {
                "transaction_amount": FeatureDistribution("lognormal", {"mean": 3.0, "sigma": 0.2}),
                "is_international": FeatureDistribution("bernoulli", {"p": 0.05}),
            },
            "Suspicious": {
                "transaction_amount": FeatureDistribution("lognormal", {"mean": 7.0, "sigma": 1.2}),
                "is_international": FeatureDistribution("bernoulli", {"p": 0.80}),
            },
        },
    )
    assert "Suspicious" in finance_profile.state_emissions
    assert finance_profile.state_emissions["Suspicious"]["is_international"].params["p"] == 0.80
