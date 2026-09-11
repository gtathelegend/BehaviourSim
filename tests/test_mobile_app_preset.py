"""Unit and integration tests for Mobile App User Engagement Preset (behaviorsim.presets.mobile_app)."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from behaviorsim.core.simulator import Simulator
from behaviorsim.presets import get_preset, list_presets
from behaviorsim.presets.mobile_app import (
    BASE_MOBILE_APP_TRANSITION_MATRIX,
    MOBILE_APP_PROFILES,
    MOBILE_APP_STATES,
    PERSONAS,
    STATE_NAMES,
    MobileAppPersona,
    create_mobile_app_profile,
    create_mobile_app_simulator,
)


# ---------------------------------------------------------------------------
# Construction & State Space Tests
# ---------------------------------------------------------------------------

def test_mobile_app_factory_returns_simulator() -> None:
    """Verify create_mobile_app_simulator instantiates a valid Simulator instance."""
    sim = create_mobile_app_simulator()
    assert isinstance(sim, Simulator)
    assert len(sim.states) == 5
    assert [s.name for s in sim.states] == STATE_NAMES


def test_mobile_app_states_unique_and_valid() -> None:
    """Verify all mobile app states have unique non-empty names and descriptions."""
    names = [s.name for s in MOBILE_APP_STATES]
    assert len(names) == len(set(names))
    for s in MOBILE_APP_STATES:
        assert s.name.strip() != ""
        assert s.description is not None and s.description.strip() != ""


def test_base_transition_matrix_properties() -> None:
    """Verify base transition matrix dimensions, stochasticity, and absorbing churn state."""
    matrix = BASE_MOBILE_APP_TRANSITION_MATRIX
    assert matrix.shape == (5, 5)
    assert np.all(matrix >= 0.0)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    # Churned state index is 4
    churn_idx = STATE_NAMES.index("Churned")
    assert matrix[churn_idx, churn_idx] == 1.0
    assert np.sum(matrix[churn_idx, :churn_idx]) == 0.0


# ---------------------------------------------------------------------------
# Registry Integration Tests
# ---------------------------------------------------------------------------

def test_mobile_app_registered_and_discoverable() -> None:
    """Verify mobile_app preset is listed in list_presets() and resolved by get_preset()."""
    presets = list_presets()
    assert "mobile_app" in presets

    factory = get_preset("mobile_app")
    assert callable(factory)
    assert factory is create_mobile_app_simulator


def test_simulator_from_preset_mobile_app() -> None:
    """Verify Simulator.from_preset('mobile_app') instantiates an independent Simulator."""
    sim1 = Simulator.from_preset("mobile_app")
    sim2 = Simulator.from_preset("mobile_app", profile="power_user")

    assert isinstance(sim1, Simulator)
    assert isinstance(sim2, Simulator)
    assert sim1 is not sim2
    assert sim1.profile is not None and sim1.profile.name == "casual_browser"
    assert sim2.profile is not None and sim2.profile.name == "power_user"


# ---------------------------------------------------------------------------
# Profile & Persona Tests
# ---------------------------------------------------------------------------

def test_all_defined_personas_construct_profiles() -> None:
    """Verify all registered personas generate valid Profile instances."""
    assert len(PERSONAS) >= 4
    for name, persona in PERSONAS.items():
        assert isinstance(persona, MobileAppPersona)
        profile = create_mobile_app_profile(persona)
        assert profile.name == name
        assert set(profile.state_emissions.keys()) == set(STATE_NAMES)
        # Check transition matrix stochasticity
        assert profile.transition_matrix is not None
        assert profile.transition_matrix.shape == (5, 5)
        assert np.all(profile.transition_matrix >= 0.0)
        assert np.allclose(profile.transition_matrix.sum(axis=1), 1.0)


@pytest.mark.parametrize("persona_name", list(PERSONAS.keys()))
def test_create_simulator_all_profiles(persona_name: str) -> None:
    """Verify create_mobile_app_simulator accepts every registered persona by name."""
    sim = create_mobile_app_simulator(profile=persona_name)
    assert sim.profile is not None
    assert sim.profile.name == persona_name


def test_create_simulator_custom_persona_instance() -> None:
    """Verify create_mobile_app_simulator accepts a direct MobileAppPersona instance."""
    custom_persona = MobileAppPersona(
        name="custom_vip",
        description="VIP high-engagement user",
        session_time_scale=3.0,
        action_intensity=3.0,
        checkout_prob=0.5,
        cart_mean=150.0,
        notification_responsiveness=0.8,
        churn_resistance=0.99,
    )
    sim = create_mobile_app_simulator(profile=custom_persona)
    assert sim.profile is not None
    assert sim.profile.name == "custom_vip"


def test_create_simulator_invalid_profile_name() -> None:
    """Verify specifying an unknown profile name raises ValueError with available list."""
    with pytest.raises(ValueError, match="Unknown mobile app profile 'invalid_persona'"):
        create_mobile_app_simulator(profile="invalid_persona")


def test_create_simulator_invalid_profile_type() -> None:
    """Verify passing an unsupported profile type raises TypeError."""
    with pytest.raises(TypeError, match="profile must be a str, MobileAppPersona, or None"):
        create_mobile_app_simulator(profile=12345)  # type: ignore


def test_create_simulator_initial_state_selection() -> None:
    """Verify initial_state parameter works for valid states and rejects unknown states."""
    sim = create_mobile_app_simulator(initial_state="CheckoutFlow")
    assert sim.initial_state == "CheckoutFlow"

    with pytest.raises(ValueError, match="Unknown initial_state 'NonExistentState'"):
        create_mobile_app_simulator(initial_state="NonExistentState")


# ---------------------------------------------------------------------------
# Simulation Output & Telemetry Schema Tests
# ---------------------------------------------------------------------------

def test_simulation_reproducibility_seed() -> None:
    """Verify deterministic reproducibility when generating with a fixed seed."""
    sim = create_mobile_app_simulator(profile="power_user")

    df1 = sim.simulate(num_interactions=30, num_sequences=2, seed=42)
    df2 = sim.simulate(num_interactions=30, num_sequences=2, seed=42)

    pd.testing.assert_frame_equal(df1, df2)

    df3 = sim.simulate(num_interactions=30, num_sequences=2, seed=99)
    assert not df1["session_time_seconds"].equals(df3["session_time_seconds"])


def test_simulation_output_schema_and_columns() -> None:
    """Verify DataFrame structure contains ground-truth state, IDs, and mobile features."""
    sim = create_mobile_app_simulator(profile="casual_browser")
    df = sim.simulate(num_interactions=25, num_sequences=3, seed=123)

    assert len(df) == 75  # 3 * 25
    expected_cols = [
        "profile",
        "sequence_id",
        "interaction_id",
        "state",
        "session_time_seconds",
        "action_count",
        "scroll_depth",
        "button_clicks",
        "notification_clicked",
        "cart_value",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column '{col}' in output"

    assert set(df["sequence_id"].unique()) == {1, 2, 3}
    assert (df["profile"] == "casual_browser").all()
    assert set(df["state"].unique()).issubset(set(STATE_NAMES))


def test_telemetry_bounds_and_semantic_types() -> None:
    """Verify semantic bounds: non-negative counts, bounded scroll depth, binary notifications."""
    sim = create_mobile_app_simulator(profile="deal_seeker")
    df = sim.simulate(num_interactions=100, num_sequences=5, seed=777)

    # session_time_seconds must be non-negative
    assert (df["session_time_seconds"] >= 0.0).all()

    # action_count and button_clicks must be non-negative integers
    assert (df["action_count"] >= 0).all()
    assert (df["button_clicks"] >= 0).all()

    # scroll_depth must be within [0.0, 1.0]
    assert (df["scroll_depth"] >= 0.0).all()
    assert (df["scroll_depth"] <= 1.0 + 1e-6).all()

    # notification_clicked must be binary {0, 1}
    assert set(df["notification_clicked"].unique()).issubset({0, 1})

    # cart_value must be non-negative
    assert (df["cart_value"] >= 0.0).all()


def test_persona_distinguishability() -> None:
    """Verify power_user exhibits significantly higher action and session metrics than infrequent_visitor."""
    sim_power = create_mobile_app_simulator(profile="power_user")
    sim_infreq = create_mobile_app_simulator(profile="infrequent_visitor")

    df_power = sim_power.simulate(num_interactions=100, num_sequences=10, seed=101)
    df_infreq = sim_infreq.simulate(num_interactions=100, num_sequences=10, seed=101)

    mean_actions_power = df_power["action_count"].mean()
    mean_actions_infreq = df_infreq["action_count"].mean()
    assert mean_actions_power > mean_actions_infreq * 1.5

    mean_time_power = df_power["session_time_seconds"].mean()
    mean_time_infreq = df_infreq["session_time_seconds"].mean()
    assert mean_time_power > mean_time_infreq * 1.5


def test_churned_state_is_absorbing() -> None:
    """Verify once a sequence enters Churned, all subsequent interactions remain Churned."""
    sim = create_mobile_app_simulator(profile="infrequent_visitor", initial_state="Churned")
    df = sim.simulate(num_interactions=20, num_sequences=3, seed=42)

    # Since initial_state is Churned and Churned is absorbing, all rows must be Churned
    assert (df["state"] == "Churned").all()
    assert (df["action_count"] == 0).all()
    assert (df["button_clicks"] == 0).all()
    assert (df["session_time_seconds"] == 0.0).all()
    assert (df["scroll_depth"] == 0.0).all()
    assert (df["cart_value"] == 0.0).all()


def test_multi_sequence_isolation() -> None:
    """Verify sequence 1 produces identical results whether num_sequences=1 or num_sequences=5."""
    sim = create_mobile_app_simulator(profile="casual_browser")

    df_single = sim.simulate(num_interactions=25, num_sequences=1, seed=555)
    df_multi = sim.simulate(num_interactions=25, num_sequences=5, seed=555)

    df_multi_seq1 = df_multi[df_multi["sequence_id"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(df_single, df_multi_seq1)


# ---------------------------------------------------------------------------
# History Rules & Causality Tests
# ---------------------------------------------------------------------------

def test_consecutive_idle_rule_triggers_churn() -> None:
    """Verify 4 consecutive interactions with zero actions triggers churn rule."""
    from behaviorsim.core.simulator import SimulationHistory
    from behaviorsim.presets.mobile_app import _consecutive_idle_rule

    history = SimulationHistory()
    # Empty history
    assert not _consecutive_idle_rule(history)

    # 3 idle actions (insufficient)
    for _ in range(3):
        history.record("action_count", 0)
    assert not _consecutive_idle_rule(history)

    # 4th idle action -> triggers
    history.record("action_count", 0)
    assert _consecutive_idle_rule(history)

    # Active action interrupts streak
    history.record("action_count", 5)
    assert not _consecutive_idle_rule(history)


def test_cart_engagement_rule_triggers_checkout() -> None:
    """Verify 2 consecutive interactions with cart_value > 50 triggers checkout rule."""
    from behaviorsim.core.simulator import SimulationHistory
    from behaviorsim.presets.mobile_app import _cart_engagement_rule

    history = SimulationHistory()
    assert not _cart_engagement_rule(history)

    # 1 high cart value interaction (insufficient)
    history.record("cart_value", 75.0)
    assert not _cart_engagement_rule(history)

    # 2nd high cart value -> triggers
    history.record("cart_value", 80.0)
    assert _cart_engagement_rule(history)

    # Low cart value interrupts
    history.record("cart_value", 10.0)
    assert not _cart_engagement_rule(history)
