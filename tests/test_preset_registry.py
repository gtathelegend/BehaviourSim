"""Unit tests for BehaviorSim preset registry (behaviorsim.presets.registry)."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.simulator import Simulator
from behaviorsim.core.state import State
from behaviorsim.presets import (
    clear_presets,
    get_preset,
    list_presets,
    register_preset,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Ensure registry built-ins are cleanly maintained for tests."""
    yield
    # Re-register built-in presets after tests that might mutate
    from behaviorsim.presets.education import create_education_simulator
    from behaviorsim.presets.healthcare import create_healthcare_simulator
    from behaviorsim.presets.mobile_app import create_mobile_app_simulator

    register_preset("education", create_education_simulator, overwrite=True)
    register_preset("healthcare", create_healthcare_simulator, overwrite=True)
    register_preset("mobile_app", create_mobile_app_simulator, overwrite=True)




def test_list_presets_deterministic() -> None:
    """Verify list_presets returns a deterministic sorted list including education."""
    presets = list_presets()
    assert isinstance(presets, list)
    assert presets == sorted(presets)
    assert "education" in presets


def test_get_preset_success() -> None:
    """Verify get_preset returns the registered factory callable."""
    factory = get_preset("education")
    assert callable(factory)
    sim = factory()
    assert isinstance(sim, Simulator)


def test_registry_returns_factory_not_shared_instance() -> None:
    """Verify registry resolves factories that produce independent simulator instances."""
    factory1 = get_preset("education")
    factory2 = get_preset("education")
    assert factory1 is factory2

    sim1 = factory1()
    sim2 = factory2()
    assert isinstance(sim1, Simulator)
    assert isinstance(sim2, Simulator)
    assert sim1 is not sim2


def test_get_preset_unknown() -> None:
    """Verify requesting an unknown preset raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="Unknown preset 'nonexistent'"):
        get_preset("nonexistent")


def test_get_preset_invalid_type() -> None:
    """Verify passing non-string preset name raises TypeError."""
    with pytest.raises(TypeError, match="Preset name must be a string"):
        get_preset(123)  # type: ignore


def test_register_preset_custom() -> None:
    """Verify registering a custom domain-neutral factory callable."""

    def dummy_factory(**kwargs):
        s = [State("idle")]
        p = Profile(
            name="dummy",
            state_emissions={"idle": {"x": FeatureDistribution("normal", {"loc": 0})}},
            transition_matrix=np.array([[1.0]]),
        )
        return Simulator(states=s, profile=p)

    register_preset("custom_preset", dummy_factory, overwrite=True)
    assert "custom_preset" in list_presets()
    retrieved = get_preset("custom_preset")
    assert retrieved is dummy_factory
    sim = retrieved()
    assert isinstance(sim, Simulator)


def test_register_preset_duplicate_fails() -> None:
    """Verify duplicate registration without overwrite flag raises ValueError."""
    with pytest.raises(ValueError, match="is already registered"):
        register_preset("education", lambda: None, overwrite=False)


def test_register_preset_overwrite_allowed() -> None:
    """Verify duplicate registration succeeds when overwrite=True."""

    def dummy_factory():
        return None

    register_preset("overwrite_test", dummy_factory, overwrite=True)
    assert get_preset("overwrite_test") is dummy_factory

    def dummy_factory2():
        return "replaced"

    register_preset("overwrite_test", dummy_factory2, overwrite=True)
    assert get_preset("overwrite_test") is dummy_factory2


def test_register_preset_invalid_name() -> None:
    """Verify registering invalid preset names raises appropriate exceptions."""
    with pytest.raises(TypeError, match="Preset name must be a string"):
        register_preset(12345, lambda: None)  # type: ignore

    with pytest.raises(ValueError, match="Preset name must be a non-empty string"):
        register_preset("", lambda: None)

    with pytest.raises(ValueError, match="Preset name must be a non-empty string"):
        register_preset("   ", lambda: None)


def test_register_preset_invalid_factory() -> None:
    """Verify registering a non-callable factory raises TypeError."""
    with pytest.raises(TypeError, match="Preset factory must be callable"):
        register_preset("bad_factory", "not_a_callable")  # type: ignore
