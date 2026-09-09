"""Unit tests for generic BehaviorSim Simulator engine (behaviorsim.core.simulator)."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.simulator import Simulator
from behaviorsim.core.state import State
from behaviorsim.core.transition import TransitionRule


# ---------------------------------------------------------------------------
# Test Fixtures & Utilities
# ---------------------------------------------------------------------------

@pytest.fixture
def generic_states() -> list[State]:
    return [State("state_a"), State("state_b")]


@pytest.fixture
def generic_profile() -> Profile:
    emissions = {
        "state_a": {
            "signal": FeatureDistribution("bernoulli", {"p": 0.8}),
            "latency": FeatureDistribution("normal", {"loc": 10.0, "scale": 1.0}),
        },
        "state_b": {
            "signal": FeatureDistribution("bernoulli", {"p": 0.2}),
            "latency": FeatureDistribution("normal", {"loc": 30.0, "scale": 2.0}),
        },
    }
    matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
    return Profile(name="generic_persona", state_emissions=emissions, transition_matrix=matrix)


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

def test_basic_simulation(generic_states: list[State], generic_profile: Profile) -> None:
    """Verify basic simulation output structure and columns."""
    sim = Simulator(states=generic_states, profile=generic_profile)
    df = sim.simulate(num_interactions=50, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    expected_cols = ["profile", "sequence_id", "interaction_id", "state", "signal", "latency"]
    assert list(df.columns) == expected_cols
    assert (df["profile"] == "generic_persona").all()
    assert (df["sequence_id"] == 1).all()
    assert list(df["interaction_id"]) == list(range(1, 51))
    assert set(df["state"].unique()).issubset({"state_a", "state_b"})


def test_arbitrary_state_counts() -> None:
    """Verify Simulator operates with 2, 3, or 4 custom states."""
    for n in [2, 3, 4]:
        states = [State(f"s_{i}") for i in range(n)]
        matrix = np.eye(n)  # Identity transition matrix
        emissions = {s.name: {"val": FeatureDistribution("normal", {"loc": i})} for i, s in enumerate(states)}
        prof = Profile(name=f"prof_{n}", state_emissions=emissions, transition_matrix=matrix)

        sim = Simulator(states=states, profile=prof)
        df = sim.simulate(num_interactions=20, seed=123)
        assert len(df) == 20
        assert set(df["state"].unique()).issubset({s.name for s in states})


def test_simulation_determinism(generic_states: list[State], generic_profile: Profile) -> None:
    """Verify identical seeds produce bit-for-bit identical DataFrames."""
    sim = Simulator(states=generic_states, profile=generic_profile)
    df1 = sim.simulate(num_interactions=100, seed=999)
    df2 = sim.simulate(num_interactions=100, seed=999)

    pd.testing.assert_frame_equal(df1, df2)

    df_diff = sim.simulate(num_interactions=100, seed=888)
    assert not df1.equals(df_diff)


def test_transition_matrix_resolution_and_validation(generic_states: list[State]) -> None:
    """Verify matrix resolution from Profile vs Simulator parameter and shape validation."""
    # 1. Simulator fallback matrix
    prof_no_matrix = Profile(name="p", state_emissions={"state_a": {}, "state_b": {}})
    sim = Simulator(
        states=generic_states,
        profile=prof_no_matrix,
        transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
    )
    assert sim.transition_matrix.shape == (2, 2)

    # 2. No matrix anywhere raises ValueError
    with pytest.raises(ValueError, match="No transition matrix supplied"):
        Simulator(states=generic_states, profile=prof_no_matrix)

    # 3. Shape mismatch raises ValueError
    mismatch_matrix = np.array([[0.3, 0.3, 0.4], [0.3, 0.3, 0.4], [0.3, 0.3, 0.4]])
    with pytest.raises(ValueError, match="does not match state count"):
        Simulator(states=generic_states, profile=prof_no_matrix, transition_matrix=mismatch_matrix)


def test_initial_state_policies(generic_states: list[State], generic_profile: Profile) -> None:
    """Verify explicit, default, and invalid initial state handling."""
    # Default initial state is states[0]
    sim_default = Simulator(states=generic_states, profile=generic_profile)
    assert sim_default.initial_state == "state_a"

    # Explicit initial state
    sim_explicit = Simulator(states=generic_states, profile=generic_profile, initial_state="state_b")
    assert sim_explicit.initial_state == "state_b"
    df = sim_explicit.simulate(num_interactions=5, seed=1)
    assert df.loc[0, "state"] == "state_b"

    # Invalid initial state
    with pytest.raises(ValueError, match="Initial state 'non_existent' not found"):
        Simulator(states=generic_states, profile=generic_profile, initial_state="non_existent")


def test_supported_distributions() -> None:
    """Verify emission sampling for all 7 required distribution types."""
    states = [State("s1")]
    emissions = {
        "s1": {
            "norm": FeatureDistribution("normal", {"loc": 5.0, "scale": 0.1}),
            "lognorm": FeatureDistribution("lognormal", {"mean": 1.0, "sigma": 0.2}),
            "exp": FeatureDistribution("exponential", {"scale": 2.0}),
            "unif": FeatureDistribution("uniform", {"low": 10.0, "high": 20.0}),
            "unif_disc": FeatureDistribution("uniform_discrete", {"items": [10, 20, 30]}),
            "bern": FeatureDistribution("bernoulli", {"p": 1.0}),  # Always 1
            "pois": FeatureDistribution("poisson", {"lam": 5}),
            "cat": FeatureDistribution("categorical", {"items": ["x", "y"], "probabilities": [0.0, 1.0]}),  # Always 'y'
        }
    }
    prof = Profile(name="dist_prof", state_emissions=emissions, transition_matrix=np.array([[1.0]]))
    sim = Simulator(states=states, profile=prof)
    df = sim.simulate(num_interactions=10, seed=42)

    assert (df["bern"] == 1).all()
    assert (df["cat"] == "y").all()
    assert (df["unif"] >= 10.0).all() and (df["unif"] <= 20.0).all()
    assert set(df["unif_disc"].unique()).issubset({10, 20, 30})


def test_causal_history_context_in_transition_rules(generic_states: list[State]) -> None:
    """Verify transition rules access only observations 0..t-1 (causal ordering)."""
    recorded_windows: list[list[float]] = []

    def check_history(ctx) -> bool:
        recent = ctx.get_recent("signal", 2)
        recorded_windows.append(list(recent))
        return len(recent) >= 2 and sum(recent) == 0  # 2 consecutive 0s

    rule = TransitionRule(condition=check_history, target_state="state_b")
    emissions = {
        "state_a": {"signal": FeatureDistribution("bernoulli", {"p": 0.0})},  # Always 0
        "state_b": {"signal": FeatureDistribution("bernoulli", {"p": 1.0})},
    }
    prof = Profile(
        name="rule_prof",
        state_emissions=emissions,
        transition_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
        transition_rules=[rule],
    )

    sim = Simulator(states=generic_states, profile=prof)
    df = sim.simulate(num_interactions=5, seed=42)

    # At t=0, no transition rule check occurs (initial state = state_a).
    # At t=1, rule check sees get_recent("signal", 2) -> [0.0] (len 1, returns False).
    # At t=2, rule check sees get_recent("signal", 2) -> [0.0, 0.0] (len 2, returns True!).
    # Thus at t=2 state switches to state_b.
    assert df.loc[2, "state"] == "state_b"


def test_rule_precedence_first_match_wins(generic_states: list[State]) -> None:
    """Verify first matching rule applies when multiple rules evaluate True."""
    rule1 = TransitionRule(condition=lambda ctx: True, target_state="state_a")
    rule2 = TransitionRule(condition=lambda ctx: True, target_state="state_b")

    emissions = {"state_a": {}, "state_b": {}}
    prof = Profile(
        name="precedence_prof",
        state_emissions=emissions,
        transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
        transition_rules=[rule1, rule2],
    )

    sim = Simulator(states=generic_states, profile=prof, initial_state="state_b")
    df = sim.simulate(num_interactions=10, seed=1)

    # After initial step 0, rule1 matches first and targets state_a
    assert (df.loc[1:, "state"] == "state_a").all()


def test_multiple_sequences_history_reset(generic_states: list[State], generic_profile: Profile) -> None:
    """Verify multiple sequences reset history boundaries cleanly."""
    sim = Simulator(states=generic_states, profile=generic_profile)
    df = sim.simulate(num_interactions=20, num_sequences=3, seed=42)

    assert len(df) == 60
    assert set(df["sequence_id"].unique()) == {1, 2, 3}
    assert (df[df["sequence_id"] == 1]["interaction_id"].to_list() == list(range(1, 21)))
    assert (df[df["sequence_id"] == 2]["interaction_id"].to_list() == list(range(1, 21)))
    assert (df[df["sequence_id"] == 3]["interaction_id"].to_list() == list(range(1, 21)))


def test_dynamic_emission_context_and_causality(generic_states: list[State]) -> None:
    """Verify dynamic callbacks receive proper evaluation context and obey causal history."""
    seen_indices: list[int] = []
    seen_history_lengths: list[int] = []

    def dynamic_callback(eval_ctx) -> float:
        assert eval_ctx.profile.name == "dynamic_prof"
        assert eval_ctx.state in {"state_a", "state_b"}
        assert eval_ctx.rng is not None

        idx = eval_ctx.interaction_index
        seen_indices.append(idx)
        recent = eval_ctx.history.get_recent("sensor", 100)
        seen_history_lengths.append(len(recent))
        # History length must exactly equal the number of completed interactions before idx
        assert len(recent) == idx
        return float(idx * 10.0)

    emissions = {
        "state_a": {"sensor": dynamic_callback},
        "state_b": {"sensor": dynamic_callback},
    }
    prof = Profile(
        name="dynamic_prof",
        state_emissions=emissions,
        transition_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    sim = Simulator(states=generic_states, profile=prof)
    df = sim.simulate(num_interactions=5, seed=42)

    assert seen_indices == [0, 1, 2, 3, 4]
    assert seen_history_lengths == [0, 1, 2, 3, 4]
    assert list(df["sensor"]) == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_invalid_transition_rule_target_state_rejected(generic_states: list[State]) -> None:
    """Verify Simulator rejects transition rules with unregistered target states."""
    invalid_rule = TransitionRule(condition=lambda ctx: True, target_state="non_existent_state")
    prof = Profile(
        name="bad_target_prof",
        state_emissions={"state_a": {}, "state_b": {}},
        transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
        transition_rules=[invalid_rule],
    )

    with pytest.raises(ValueError, match="target_state 'non_existent_state' not found in registered states"):
        Simulator(states=generic_states, profile=prof)


def test_invalid_distribution_parameters_raise_error(generic_states: list[State]) -> None:
    """Verify invalid distribution parameters raise informative ValueError."""
    # 1. Negative scale for normal
    sim1 = Simulator(
        states=generic_states,
        profile=Profile(
            name="p1",
            state_emissions={"state_a": {"x": FeatureDistribution("normal", {"scale": -1.0})}, "state_b": {}},
            transition_matrix=np.eye(2),
        ),
    )
    with pytest.raises(ValueError, match="Normal distribution scale must be non-negative"):
        sim1.simulate(num_interactions=1)

    # 2. Negative sigma for lognormal
    sim2 = Simulator(
        states=generic_states,
        profile=Profile(
            name="p2",
            state_emissions={"state_a": {"x": FeatureDistribution("lognormal", {"sigma": -0.5})}, "state_b": {}},
            transition_matrix=np.eye(2),
        ),
    )
    with pytest.raises(ValueError, match="Lognormal distribution sigma must be non-negative"):
        sim2.simulate(num_interactions=1)

    # 3. Non-positive scale for exponential
    sim3 = Simulator(
        states=generic_states,
        profile=Profile(
            name="p3",
            state_emissions={"state_a": {"x": FeatureDistribution("exponential", {"scale": 0.0})}, "state_b": {}},
            transition_matrix=np.eye(2),
        ),
    )
    with pytest.raises(ValueError, match="Exponential distribution scale must be positive"):
        sim3.simulate(num_interactions=1)

    # 4. Uniform high < low
    sim4 = Simulator(
        states=generic_states,
        profile=Profile(
            name="p4",
            state_emissions={"state_a": {"x": FeatureDistribution("uniform", {"low": 10.0, "high": 5.0})}, "state_b": {}},
            transition_matrix=np.eye(2),
        ),
    )
    with pytest.raises(ValueError, match="Uniform distribution high .* must be >= low"):
        sim4.simulate(num_interactions=1)

    # 5. Bernoulli p out of [0, 1]
    sim5 = Simulator(
        states=generic_states,
        profile=Profile(
            name="p5",
            state_emissions={"state_a": {"x": FeatureDistribution("bernoulli", {"p": 1.5})}, "state_b": {}},
            transition_matrix=np.eye(2),
        ),
    )
    with pytest.raises(ValueError, match="Bernoulli probability p must be in"):
        sim5.simulate(num_interactions=1)

    # 6. Poisson negative lambda
    sim6 = Simulator(
        states=generic_states,
        profile=Profile(
            name="p6",
            state_emissions={"state_a": {"x": FeatureDistribution("poisson", {"lam": -2.0})}, "state_b": {}},
            transition_matrix=np.eye(2),
        ),
    )
    with pytest.raises(ValueError, match="Poisson parameter lam must be non-negative"):
        sim6.simulate(num_interactions=1)

    # 7. Categorical missing items
    sim7 = Simulator(
        states=generic_states,
        profile=Profile(
            name="p7",
            state_emissions={"state_a": {"x": FeatureDistribution("categorical", {})}, "state_b": {}},
            transition_matrix=np.eye(2),
        ),
    )
    with pytest.raises(ValueError, match="categorical distribution requires 'items' parameter"):
        sim7.simulate(num_interactions=1)

    # 8. Unsupported distribution family
    sim8 = Simulator(
        states=generic_states,
        profile=Profile(
            name="p8",
            state_emissions={"state_a": {"x": FeatureDistribution("weibull", {})}, "state_b": {}},
            transition_matrix=np.eye(2),
        ),
    )
    with pytest.raises(ValueError, match="Unsupported distribution_type 'weibull'"):
        sim8.simulate(num_interactions=1)


def test_transition_rule_probability_boundaries(generic_states: list[State]) -> None:
    """Verify rule with prob 0.0 never triggers (falls back to matrix) and prob 1.0 always triggers."""
    # Always match condition
    rule_never = TransitionRule(condition=lambda ctx: True, target_state="state_b", probability=0.0)
    # Matrix says stay at state_a
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    prof_never = Profile(
        name="prof_never",
        state_emissions={"state_a": {}, "state_b": {}},
        transition_matrix=matrix,
        transition_rules=[rule_never],
    )
    sim_never = Simulator(states=generic_states, profile=prof_never, initial_state="state_a")
    df_never = sim_never.simulate(num_interactions=10, seed=42)
    # Should stay state_a because probability 0.0 fails and falls back to matrix (which keeps state_a)
    assert (df_never["state"] == "state_a").all()

    rule_always = TransitionRule(condition=lambda ctx: True, target_state="state_b", probability=1.0)
    prof_always = Profile(
        name="prof_always",
        state_emissions={"state_a": {}, "state_b": {}},
        transition_matrix=matrix,
        transition_rules=[rule_always],
    )
    sim_always = Simulator(states=generic_states, profile=prof_always, initial_state="state_a")
    df_always = sim_always.simulate(num_interactions=10, seed=42)
    # Step 0 is initial_state ("state_a"), from step 1 onward it switches to "state_b"
    assert df_always.loc[0, "state"] == "state_a"
    assert (df_always.loc[1:, "state"] == "state_b").all()


def test_simulation_input_validation(generic_states: list[State], generic_profile: Profile) -> None:
    """Verify invalid simulation interaction and sequence counts are rejected."""
    sim = Simulator(states=generic_states, profile=generic_profile)

    with pytest.raises(ValueError, match="num_interactions must be >= 1"):
        sim.simulate(num_interactions=0)

    with pytest.raises(ValueError, match="num_interactions must be >= 1"):
        sim.simulate(num_interactions=-5)

    with pytest.raises(ValueError, match="num_sequences must be >= 1"):
        sim.simulate(num_sequences=0)

    with pytest.raises(ValueError, match="num_sequences must be >= 1"):
        sim.simulate(num_sequences=-2)

