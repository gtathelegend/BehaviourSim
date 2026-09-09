"""Unit tests for BehaviorSim State and Feature abstractions (behaviorsim.core.state and feature)."""

import sys
from pathlib import Path
from dataclasses import FrozenInstanceError

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from behaviorsim.core.state import State
from behaviorsim.core.feature import FeatureDistribution


# ---------------------------------------------------------------------------
# State Tests
# ---------------------------------------------------------------------------

def test_state_valid_construction() -> None:
    """Verify valid State instances can be constructed."""
    s1 = State("Optimal")
    assert s1.name == "Optimal"
    assert s1.description == ""

    s2 = State("Overload", "High cognitive load state")
    assert s2.name == "Overload"
    assert s2.description == "High cognitive load state"

    s3 = State(" Optimal ")
    assert s3.name == " Optimal "  # Exact string preserved without silent trimming


def test_state_invalid_name() -> None:
    """Verify State raises ValueError on invalid name."""
    with pytest.raises(ValueError, match="non-empty string"):
        State("")

    with pytest.raises(ValueError, match="non-empty string"):
        State("   ")

    with pytest.raises(ValueError, match="must be a string"):
        State(123)  # type: ignore[arg-type]


def test_state_invalid_description() -> None:
    """Verify State raises ValueError on non-string description."""
    with pytest.raises(ValueError, match="description must be a string"):
        State("Optimal", 123)  # type: ignore[arg-type]


def test_state_immutability() -> None:
    """Verify State instance is frozen/immutable."""
    s = State("Optimal")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        s.name = "Underload"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FeatureDistribution Tests
# ---------------------------------------------------------------------------

def test_feature_distribution_valid_construction() -> None:
    """Verify valid FeatureDistribution instances can be constructed."""
    d1 = FeatureDistribution(
        distribution_type="normal",
        params={"loc": 0.0, "scale": 1.0},
    )
    assert d1.distribution_type == "normal"
    assert d1.params == {"loc": 0.0, "scale": 1.0}

    d2 = FeatureDistribution(
        distribution_type="categorical",
        params={"items": ["a", "b"], "probabilities": [0.5, 0.5]},
    )
    assert d2.distribution_type == "categorical"
    assert d2.params["items"] == ["a", "b"]


def test_feature_distribution_invalid_type() -> None:
    """Verify FeatureDistribution raises ValueError on invalid distribution_type."""
    with pytest.raises(ValueError, match="non-empty string"):
        FeatureDistribution("", {"loc": 0.0})

    with pytest.raises(ValueError, match="non-empty string"):
        FeatureDistribution("   ", {"loc": 0.0})

    with pytest.raises(ValueError, match="must be a string"):
        FeatureDistribution(123, {"loc": 0.0})  # type: ignore[arg-type]


def test_feature_distribution_invalid_params() -> None:
    """Verify FeatureDistribution raises ValueError on non-mapping params."""
    with pytest.raises(ValueError, match="must be a mapping"):
        FeatureDistribution("normal", [0.0, 1.0])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be a mapping"):
        FeatureDistribution("normal", "invalid")  # type: ignore[arg-type]


def test_feature_distribution_immutability() -> None:
    """Verify FeatureDistribution instance is frozen/immutable."""
    d = FeatureDistribution("normal", {"loc": 0.0})
    with pytest.raises((FrozenInstanceError, AttributeError)):
        d.distribution_type = "lognormal"  # type: ignore[misc]
