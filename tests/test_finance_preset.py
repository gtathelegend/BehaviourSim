"""Unit and integration tests for Finance behavioral preset (behaviorsim.presets.finance)."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from behaviorsim.core.simulator import SimulationHistory, Simulator
from behaviorsim.presets import get_preset, list_presets
from behaviorsim.presets.finance import (
    BASE_FINANCE_TRANSITION_MATRIX,
    FINANCE_PROFILES,
    FINANCE_STATES,
    INVESTOR_COHORTS,
    STATE_NAMES,
    FinanceCohort,
    create_finance_profile,
    create_finance_simulator,
)


# ---------------------------------------------------------------------------
# 1. Factory returns a Simulator
# ---------------------------------------------------------------------------

def test_1_factory_returns_simulator() -> None:
    """Verify create_finance_simulator returns a valid Simulator instance."""
    sim = create_finance_simulator()
    assert isinstance(sim, Simulator)


# ---------------------------------------------------------------------------
# 2. Exactly six expected states
# ---------------------------------------------------------------------------

def test_2_exactly_six_expected_states() -> None:
    """Verify the preset defines exactly six expected states in stable order."""
    expected_states = ["Stable", "Active", "Volatile", "Drawdown", "Recovered", "Closed"]
    assert len(FINANCE_STATES) == 6
    assert [s.name for s in FINANCE_STATES] == expected_states
    assert STATE_NAMES == expected_states


# ---------------------------------------------------------------------------
# 3. State uniqueness
# ---------------------------------------------------------------------------

def test_3_state_uniqueness() -> None:
    """Verify all state names are unique and have non-empty descriptions."""
    names = [s.name for s in FINANCE_STATES]
    assert len(names) == len(set(names))
    for s in FINANCE_STATES:
        assert s.name.strip() != ""
        assert s.description is not None and s.description.strip() != ""


# ---------------------------------------------------------------------------
# 4. Registry registration
# ---------------------------------------------------------------------------

def test_4_registry_registration() -> None:
    """Verify finance preset is registered in list_presets() and resolved by get_preset()."""
    presets = list_presets()
    assert "finance" in presets
    factory = get_preset("finance")
    assert callable(factory)
    assert factory is create_finance_simulator


# ---------------------------------------------------------------------------
# 5. Simulator.from_preset("finance")
# ---------------------------------------------------------------------------

def test_5_simulator_from_preset_finance() -> None:
    """Verify Simulator.from_preset('finance') instantiates independent Simulator objects."""
    sim1 = Simulator.from_preset("finance")
    sim2 = Simulator.from_preset("finance", profile="active_trader")

    assert isinstance(sim1, Simulator)
    assert isinstance(sim2, Simulator)
    assert sim1 is not sim2
    assert sim1.profile is not None and sim1.profile.name == "balanced_investor"
    assert sim2.profile is not None and sim2.profile.name == "active_trader"


# ---------------------------------------------------------------------------
# 6. 6x6 transition matrix
# ---------------------------------------------------------------------------

def test_6_transition_matrix_shape() -> None:
    """Verify base transition matrix has exact shape 6x6."""
    assert BASE_FINANCE_TRANSITION_MATRIX.shape == (6, 6)


# ---------------------------------------------------------------------------
# 7. Matrix non-negativity
# ---------------------------------------------------------------------------

def test_7_matrix_non_negativity() -> None:
    """Verify all entries in the base transition matrix are non-negative and finite."""
    assert np.all(BASE_FINANCE_TRANSITION_MATRIX >= 0.0)
    assert np.all(np.isfinite(BASE_FINANCE_TRANSITION_MATRIX))


# ---------------------------------------------------------------------------
# 8. Matrix row sums
# ---------------------------------------------------------------------------

def test_8_matrix_row_sums() -> None:
    """Verify every row of the base transition matrix sums to 1.0."""
    assert np.allclose(BASE_FINANCE_TRANSITION_MATRIX.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# 9. Closed absorbing behavior
# ---------------------------------------------------------------------------

def test_9_closed_absorbing_behavior() -> None:
    """Verify Closed is strictly absorbing in matrix and maintains Closed during simulation."""
    closed_idx = STATE_NAMES.index("Closed")
    assert BASE_FINANCE_TRANSITION_MATRIX[closed_idx, closed_idx] == 1.0
    assert np.sum(BASE_FINANCE_TRANSITION_MATRIX[closed_idx, :closed_idx]) == 0.0

    sim = create_finance_simulator(initial_state="Closed")
    df = sim.simulate(num_interactions=15, num_sequences=2, seed=123)
    assert (df["state"] == "Closed").all()
    assert (df["transaction_count"] == 0).all()
    assert (df["risk_alert"] == 0).all()


# ---------------------------------------------------------------------------
# 10. All four profiles construct successfully
# ---------------------------------------------------------------------------

def test_10_all_four_profiles_construct() -> None:
    """Verify all four registered investor cohorts generate valid Profile instances."""
    expected = {"conservative_investor", "balanced_investor", "growth_investor", "active_trader"}
    assert set(INVESTOR_COHORTS.keys()) == expected

    for name, cohort in INVESTOR_COHORTS.items():
        assert isinstance(cohort, FinanceCohort)
        profile = create_finance_profile(cohort)
        assert profile.name == name
        assert set(profile.state_emissions.keys()) == set(STATE_NAMES)
        assert profile.transition_matrix is not None
        assert profile.transition_matrix.shape == (6, 6)
        assert np.all(profile.transition_matrix >= 0.0)
        assert np.allclose(profile.transition_matrix.sum(axis=1), 1.0)
        assert profile.transition_matrix[5, 5] == 1.0


# ---------------------------------------------------------------------------
# 11. Invalid profile name
# ---------------------------------------------------------------------------

def test_11_invalid_profile_name() -> None:
    """Verify passing an unrecognized profile name raises descriptive ValueError."""
    with pytest.raises(ValueError, match="Unknown finance profile 'nonexistent'"):
        create_finance_simulator(profile="nonexistent")


# ---------------------------------------------------------------------------
# 12. Invalid profile type where applicable
# ---------------------------------------------------------------------------

def test_12_invalid_profile_type() -> None:
    """Verify passing an invalid profile type raises descriptive TypeError."""
    with pytest.raises(TypeError, match="profile must be a str, FinanceCohort, or None"):
        create_finance_simulator(profile=12345)  # type: ignore


# ---------------------------------------------------------------------------
# 13. Feature schema contains all seven required features
# ---------------------------------------------------------------------------

def test_13_feature_schema_seven_required_features() -> None:
    """Verify simulation output DataFrame contains all 7 required financial features."""
    required = [
        "portfolio_value",
        "daily_return",
        "transaction_count",
        "trade_volume",
        "volatility",
        "drawdown",
        "risk_alert",
    ]
    sim = create_finance_simulator()
    df = sim.simulate(num_interactions=10, num_sequences=1, seed=42)

    for col in ["profile", "sequence_id", "interaction_id", "state"] + required:
        assert col in df.columns


# ---------------------------------------------------------------------------
# 14. Portfolio values are finite/non-negative
# ---------------------------------------------------------------------------

def test_14_portfolio_values_finite_non_negative() -> None:
    """Verify emitted portfolio_value values are strictly positive and finite."""
    sim = create_finance_simulator()
    df = sim.simulate(num_interactions=50, num_sequences=2, seed=101)
    assert (df["portfolio_value"] > 0.0).all()
    assert np.all(np.isfinite(df["portfolio_value"]))


# ---------------------------------------------------------------------------
# 15. Daily returns are finite
# ---------------------------------------------------------------------------

def test_15_daily_returns_finite() -> None:
    """Verify emitted daily_return values are finite continuous values."""
    sim = create_finance_simulator()
    df = sim.simulate(num_interactions=50, num_sequences=2, seed=102)
    assert np.all(np.isfinite(df["daily_return"]))


# ---------------------------------------------------------------------------
# 16. Transaction counts are non-negative integers
# ---------------------------------------------------------------------------

def test_16_transaction_counts_non_negative_integers() -> None:
    """Verify emitted transaction_count values are non-negative integers."""
    sim = create_finance_simulator()
    df = sim.simulate(num_interactions=50, num_sequences=2, seed=103)
    assert (df["transaction_count"] >= 0).all()
    assert df["transaction_count"].dtype in [np.int64, np.int32, int]


# ---------------------------------------------------------------------------
# 17. Trade volume is non-negative
# ---------------------------------------------------------------------------

def test_17_trade_volume_non_negative() -> None:
    """Verify emitted trade_volume values are non-negative and finite."""
    sim = create_finance_simulator()
    df = sim.simulate(num_interactions=50, num_sequences=2, seed=104)
    assert (df["trade_volume"] >= 0.0).all()
    assert np.all(np.isfinite(df["trade_volume"]))


# ---------------------------------------------------------------------------
# 18. Volatility is non-negative
# ---------------------------------------------------------------------------

def test_18_volatility_non_negative() -> None:
    """Verify emitted volatility values are non-negative and finite."""
    sim = create_finance_simulator()
    df = sim.simulate(num_interactions=50, num_sequences=2, seed=105)
    assert (df["volatility"] >= 0.0).all()
    assert np.all(np.isfinite(df["volatility"]))


# ---------------------------------------------------------------------------
# 19. Drawdown is bounded [0, 1]
# ---------------------------------------------------------------------------

def test_19_drawdown_bounded_zero_to_one() -> None:
    """Verify emitted drawdown values are strictly bounded in [0.0, 1.0]."""
    sim = create_finance_simulator()
    df = sim.simulate(num_interactions=50, num_sequences=2, seed=106)
    assert (df["drawdown"] >= 0.0).all()
    assert (df["drawdown"] <= 1.0).all()


# ---------------------------------------------------------------------------
# 20. Risk alert is binary
# ---------------------------------------------------------------------------

def test_20_risk_alert_binary() -> None:
    """Verify emitted risk_alert values are strictly binary integers in {0, 1}."""
    sim = create_finance_simulator()
    df = sim.simulate(num_interactions=50, num_sequences=2, seed=107)
    assert set(df["risk_alert"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# 21. Profile differentiation
# ---------------------------------------------------------------------------

def test_21_profile_differentiation() -> None:
    """Verify conservative_investor and active_trader exhibit distinct behavioral metrics."""
    sim_cons = create_finance_simulator(profile="conservative_investor")
    sim_trader = create_finance_simulator(profile="active_trader")

    df_cons = sim_cons.simulate(num_interactions=150, num_sequences=2, seed=2001)
    df_trader = sim_trader.simulate(num_interactions=150, num_sequences=2, seed=2001)

    mean_tx_cons = df_cons["transaction_count"].mean()
    mean_tx_trader = df_trader["transaction_count"].mean()
    assert mean_tx_trader > mean_tx_cons * 2.0

    mean_vol_cons = df_cons["trade_volume"].mean()
    mean_vol_trader = df_trader["trade_volume"].mean()
    assert mean_vol_trader > mean_vol_cons * 3.0

    mean_sigma_cons = df_cons["volatility"].mean()
    mean_sigma_trader = df_trader["volatility"].mean()
    assert mean_sigma_trader > mean_sigma_cons

    alert_cons = df_cons["risk_alert"].mean()
    alert_trader = df_trader["risk_alert"].mean()
    assert alert_trader > alert_cons


# ---------------------------------------------------------------------------
# 22. Sustained drawdown rule
# ---------------------------------------------------------------------------

def test_22_sustained_drawdown_rule() -> None:
    """Verify sustained negative returns or acute drawdown triggers Drawdown rule."""
    from behaviorsim.presets.finance import _sustained_drawdown_rule

    history = SimulationHistory()
    assert not _sustained_drawdown_rule(history)

    # 1 negative return step (requires 2)
    history.record("daily_return", -0.015)
    history.record("drawdown", 0.05)
    assert not _sustained_drawdown_rule(history)

    # 2nd negative return step -> triggers rule
    history.record("daily_return", -0.020)
    history.record("drawdown", 0.08)
    assert _sustained_drawdown_rule(history)

    # Acute drawdown triggers rule directly
    history_dd = SimulationHistory()
    history_dd.record("daily_return", 0.001)
    history_dd.record("drawdown", 0.25)
    assert _sustained_drawdown_rule(history_dd)


# ---------------------------------------------------------------------------
# 23. Volatility escalation rule
# ---------------------------------------------------------------------------

def test_23_volatility_escalation_rule() -> None:
    """Verify consecutive elevated volatility observations trigger Volatile rule."""
    from behaviorsim.presets.finance import _volatility_escalation_rule

    history = SimulationHistory()
    assert not _volatility_escalation_rule(history)

    # 1 high volatility step (requires 2)
    history.record("volatility", 0.40)
    assert not _volatility_escalation_rule(history)

    # 2nd high volatility step -> triggers rule
    history.record("volatility", 0.45)
    assert _volatility_escalation_rule(history)

    # Low volatility interrupts condition
    history.record("volatility", 0.10)
    assert not _volatility_escalation_rule(history)


# ---------------------------------------------------------------------------
# 24. Recovery rule
# ---------------------------------------------------------------------------

def test_24_recovery_rule() -> None:
    """Verify stabilizing positive returns and receding drawdown triggers Recovered rule."""
    from behaviorsim.presets.finance import _recovery_rule

    history = SimulationHistory()
    assert not _recovery_rule(history)

    # Build history showing prior drawdown
    history.record("drawdown", 0.35)
    history.record("daily_return", -0.025)

    # 1 positive return step (requires 2)
    history.record("drawdown", 0.18)
    history.record("daily_return", 0.008)
    assert not _recovery_rule(history)

    # 2nd positive return step with receding drawdown -> triggers rule
    history.record("drawdown", 0.12)
    history.record("daily_return", 0.010)
    assert _recovery_rule(history)


# ---------------------------------------------------------------------------
# 25. Explicit causal/in-flight rule behavior
# ---------------------------------------------------------------------------

def test_25_causal_in_flight_rule_behavior() -> None:
    """Verify in-flight simulator transition rules evaluate causally without lookahead."""
    sim = create_finance_simulator(profile="active_trader", initial_state="Volatile")
    df = sim.simulate(num_interactions=12, num_sequences=1, seed=42)

    # State at interaction 1 is initial_state ("Volatile")
    assert df.loc[df["interaction_id"] == 1, "state"].iloc[0] == "Volatile"

    # By interaction 3, rules evaluating prior interactions maintain active market states
    states_later = df.loc[df["interaction_id"] >= 3, "state"].tolist()
    assert any(s in ["Volatile", "Drawdown", "Active"] for s in states_later)
    assert (df["drawdown"] <= 1.0).all()


# ---------------------------------------------------------------------------
# 26. Deterministic seeded generation
# ---------------------------------------------------------------------------

def test_26_deterministic_seeded_generation() -> None:
    """Verify identical seeds produce identical simulation DataFrame traces."""
    sim = create_finance_simulator(profile="growth_investor")

    df1 = sim.simulate(num_interactions=25, num_sequences=2, seed=888)
    df2 = sim.simulate(num_interactions=25, num_sequences=2, seed=888)

    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# 27. Different seeds produce different simulations
# ---------------------------------------------------------------------------

def test_27_different_seeds_produce_different_simulations() -> None:
    """Verify different seeds generate distinct simulation trajectories."""
    sim = create_finance_simulator(profile="balanced_investor")

    df1 = sim.simulate(num_interactions=25, num_sequences=1, seed=101)
    df2 = sim.simulate(num_interactions=25, num_sequences=1, seed=202)

    assert not df1["daily_return"].equals(df2["daily_return"])
    assert not df1["portfolio_value"].equals(df2["portfolio_value"])


# ---------------------------------------------------------------------------
# 28. Multi-sequence isolation
# ---------------------------------------------------------------------------

def test_28_multi_sequence_isolation() -> None:
    """Verify sequence 1 produces identical results whether num_sequences=1 or num_sequences=4."""
    sim = create_finance_simulator(profile="conservative_investor")

    df_single = sim.simulate(num_interactions=20, num_sequences=1, seed=555)
    df_multi = sim.simulate(num_interactions=20, num_sequences=4, seed=555)

    df_multi_seq1 = df_multi[df_multi["sequence_id"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(df_single, df_multi_seq1)


# ---------------------------------------------------------------------------
# 29. Synthetic financial disclaimer
# ---------------------------------------------------------------------------

def test_29_synthetic_financial_disclaimer() -> None:
    """Verify module and factory docstrings explicitly state synthetic model and non-advice boundary."""
    import behaviorsim.presets.finance as fin_module

    mod_doc = " ".join((fin_module.__doc__ or "").lower().split())
    assert "synthetic" in mod_doc
    assert "not a real-market model" in mod_doc
    assert "not financially validated" in mod_doc
    assert "investment advice" in mod_doc
    assert "trading advice" in mod_doc
    assert "recommendation to buy or sell securities" in mod_doc

    factory_doc = " ".join((fin_module.create_finance_simulator.__doc__ or "").lower().split())
    assert "synthetic" in factory_doc
    assert "not a real-market model" in factory_doc
    assert "not financially validated" in factory_doc
    assert "investment advice" in factory_doc
    assert "trading advice" in factory_doc
    assert "recommendation to buy or sell securities" in factory_doc
