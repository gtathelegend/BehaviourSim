"""Comprehensive edge-case and hardening tests for BehaviorSim core engine (Phase 1.4).

Audits and validates:
1. Zero-count semantics (num_interactions=0, num_sequences=0)
2. Negative count validation (< 0 raises ValueError)
3. Single-state models (N=1, [[1.0]])
4. Absorbing state semantics
5. Categorical distribution boundary conditions (valid and invalid)
6. Transition rule and profile probability boundaries (0.0 and 1.0)
7. Empty-history causality and missing feature handling
8. Output schema hardening and union preserving across zero-row cases
9. Determinism and seed reproducibility invariance
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from behaviorsim.config import validate_distribution_params
from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.simulator import Simulator
from behaviorsim.core.state import State
from behaviorsim.core.transition import TransitionRule
from behaviorsim.core.utils import create_rng, sample_categorical


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_state() -> State:
    return State(name="only_state", description="The sole state in an N=1 model")


@pytest.fixture
def two_states() -> List[State]:
    return [
        State(name="A", description="State A"),
        State(name="B", description="State B"),
    ]


# ---------------------------------------------------------------------------
# 1. Zero-Count Semantics (num_interactions=0, num_sequences=0)
# ---------------------------------------------------------------------------

def test_zero_interactions_single_profile(two_states: List[State]) -> None:
    """Verify num_interactions=0 produces empty DataFrame with deterministic schema."""
    profile = Profile(
        name="p1",
        state_emissions={
            "A": {"feat1": FeatureDistribution(distribution_type="bernoulli", params={"p": 0.5})},
            "B": {"feat1": FeatureDistribution(distribution_type="bernoulli", params={"p": 0.8})},
        },
        transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
    )
    sim = Simulator(states=two_states, profile=profile)

    df = sim.simulate(num_interactions=0, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    expected_cols = ["profile", "sequence_id", "interaction_id", "state", "feat1"]
    assert list(df.columns) == expected_cols


def test_zero_sequences_single_profile(two_states: List[State]) -> None:
    """Verify num_sequences=0 produces empty DataFrame with deterministic schema."""
    profile = Profile(
        name="p1",
        state_emissions={
            "A": {"feat1": FeatureDistribution(distribution_type="bernoulli", params={"p": 0.5})},
            "B": {"feat1": FeatureDistribution(distribution_type="bernoulli", params={"p": 0.8})},
        },
        transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
    )
    sim = Simulator(states=two_states, profile=profile)

    df = sim.simulate(num_sequences=0, seed=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    expected_cols = ["profile", "sequence_id", "interaction_id", "state", "feat1"]
    assert list(df.columns) == expected_cols


def test_zero_counts_multi_profile_heterogeneous(two_states: List[State]) -> None:
    """Verify zero-count simulation on heterogeneous profiles preserves union schema."""
    prof1 = Profile(
        name="prof_fast",
        probability=0.7,
        state_emissions={
            "A": {"speed": FeatureDistribution(distribution_type="normal", params={"loc": 10.0, "scale": 1.0})},
        },
        transition_matrix=np.array([[0.8, 0.2], [0.1, 0.9]]),
    )
    prof2 = Profile(
        name="prof_accurate",
        probability=0.3,
        state_emissions={
            "A": {"accuracy": FeatureDistribution(distribution_type="bernoulli", params={"p": 0.99})},
            "B": {"accuracy": FeatureDistribution(distribution_type="bernoulli", params={"p": 0.85})},
        },
        transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
    )

    sim = Simulator(states=two_states, profiles={"prof_fast": prof1, "prof_accurate": prof2})

    # Zero interactions
    df_zero_int = sim.simulate(num_interactions=0, num_sequences=5, seed=123)
    assert len(df_zero_int) == 0
    assert list(df_zero_int.columns) == ["profile", "sequence_id", "interaction_id", "state", "speed", "accuracy"]

    # Zero sequences
    df_zero_seq = sim.simulate(num_interactions=10, num_sequences=0, seed=123)
    assert len(df_zero_seq) == 0
    assert list(df_zero_seq.columns) == ["profile", "sequence_id", "interaction_id", "state", "speed", "accuracy"]

    # Both zero
    df_both_zero = sim.simulate(num_interactions=0, num_sequences=0, seed=123)
    assert len(df_both_zero) == 0
    assert list(df_both_zero.columns) == ["profile", "sequence_id", "interaction_id", "state", "speed", "accuracy"]


def test_zero_counts_generate_alias(two_states: List[State]) -> None:
    """Verify sim.generate() behaves identically on zero-counts."""
    profile = Profile(
        name="p1",
        state_emissions={
            "A": {"val": FeatureDistribution(distribution_type="bernoulli", params={"p": 0.5})},
        },
        transition_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    sim = Simulator(states=two_states, profile=profile)

    df1 = sim.generate(num_interactions=0)
    assert len(df1) == 0
    assert list(df1.columns) == ["profile", "sequence_id", "interaction_id", "state", "val"]

    df2 = sim.generate(num_sequences=0)
    assert len(df2) == 0
    assert list(df2.columns) == ["profile", "sequence_id", "interaction_id", "state", "val"]


def test_zero_count_does_not_consume_rng_or_disrupt_determinism(two_states: List[State]) -> None:
    """Verify simulating 0 interactions produces identical output as running cleanly."""
    profile = Profile(
        name="p1",
        state_emissions={
            "A": {"val": FeatureDistribution(distribution_type="normal", params={"loc": 0, "scale": 1})},
            "B": {"val": FeatureDistribution(distribution_type="normal", params={"loc": 10, "scale": 1})},
        },
        transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
    )
    sim = Simulator(states=two_states, profile=profile)

    df_normal = sim.simulate(num_interactions=10, num_sequences=2, seed=999)

    # Interleave zero-count simulations
    _ = sim.simulate(num_interactions=0, num_sequences=5, seed=999)
    _ = sim.simulate(num_interactions=10, num_sequences=0, seed=999)

    df_after = sim.simulate(num_interactions=10, num_sequences=2, seed=999)

    pd.testing.assert_frame_equal(df_normal, df_after)


# ---------------------------------------------------------------------------
# 2. Negative Count Validation (< 0 raises ValueError)
# ---------------------------------------------------------------------------

def test_negative_counts_raise_value_error(two_states: List[State]) -> None:
    """Verify negative simulation counts raise clear ValueError."""
    profile = Profile(
        name="p1",
        state_emissions={"A": {}},
        transition_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    sim = Simulator(states=two_states, profile=profile)

    with pytest.raises(ValueError, match="num_interactions must be non-negative, got -1"):
        sim.simulate(num_interactions=-1)

    with pytest.raises(ValueError, match="num_interactions must be non-negative, got -100"):
        sim.simulate(num_interactions=-100)

    with pytest.raises(ValueError, match="num_sequences must be non-negative, got -1"):
        sim.simulate(num_sequences=-1)

    with pytest.raises(ValueError, match="num_sequences must be non-negative, got -42"):
        sim.simulate(num_sequences=-42)


# ---------------------------------------------------------------------------
# 3. Single-State Models (N = 1, [[1.0]])
# ---------------------------------------------------------------------------

def test_single_state_model(single_state: State) -> None:
    """Verify generic engine cleanly supports N=1 models with [[1.0]] transition matrix."""
    profile = Profile(
        name="single_persona",
        state_emissions={
            "only_state": {
                "reading": FeatureDistribution(distribution_type="normal", params={"loc": 5.0, "scale": 0.5}),
                "status": FeatureDistribution(
                    distribution_type="categorical",
                    params={"items": ["active"], "probabilities": [1.0]},
                ),
            }
        },
        transition_matrix=np.array([[1.0]]),
        transition_rules=[
            TransitionRule(
                condition=lambda h: len(h.get_recent("reading", 5)) >= 2,
                target_state="only_state",
                probability=1.0,
            )
        ],
    )

    sim = Simulator(states=[single_state], profile=profile)

    # Multi-sequence, multi-interaction simulation
    df = sim.simulate(num_interactions=15, num_sequences=3, seed=42)

    assert len(df) == 45
    assert (df["state"] == "only_state").all()
    assert (df["profile"] == "single_persona").all()
    assert (df["status"] == "active").all()
    assert len(df["reading"].dropna()) == 45

    # Determinism check with same seed
    df2 = sim.simulate(num_interactions=15, num_sequences=3, seed=42)
    pd.testing.assert_frame_equal(df, df2)


# ---------------------------------------------------------------------------
# 4. Absorbing State Semantics
# ---------------------------------------------------------------------------

def test_absorbing_state_single_profile(two_states: List[State]) -> None:
    """Verify absorbing state A: once entered, the process never leaves state A."""
    # Matrix:
    # A -> A with prob 1.0, A -> B with 0.0
    # B -> A with prob 1.0, B -> B with 0.0
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    profile = Profile(
        name="absorbing_profile",
        state_emissions={
            "A": {"label": FeatureDistribution(distribution_type="categorical", params={"items": ["in_A"], "probabilities": [1.0]})},
            "B": {"label": FeatureDistribution(distribution_type="categorical", params={"items": ["in_B"], "probabilities": [1.0]})},
        },
        transition_matrix=matrix,
    )

    # Case 1: Start in absorbing state A
    sim_start_a = Simulator(states=two_states, profile=profile, initial_state="A")
    df_a = sim_start_a.simulate(num_interactions=20, seed=42)
    assert (df_a["state"] == "A").all()

    # Case 2: Start in transient state B -> step 0 is B, all subsequent steps must be A
    sim_start_b = Simulator(states=two_states, profile=profile, initial_state="B")
    df_b = sim_start_b.simulate(num_interactions=20, seed=42)
    assert df_b.loc[0, "state"] == "B"
    assert (df_b.loc[1:, "state"] == "A").all()


def test_absorbing_state_multi_profile(two_states: List[State]) -> None:
    """Verify absorbing state behavior under multi-profile mixture."""
    matrix_abs = np.array([[1.0, 0.0], [1.0, 0.0]])
    prof_abs = Profile(
        name="prof_abs",
        probability=1.0,
        state_emissions={"A": {}, "B": {}},
        transition_matrix=matrix_abs,
    )
    prof_other = Profile(
        name="prof_other",
        probability=0.0,
        state_emissions={"A": {}, "B": {}},
        transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
    )

    sim = Simulator(
        states=two_states,
        profiles={"prof_abs": prof_abs, "prof_other": prof_other},
        initial_state="B",
    )
    df = sim.simulate(num_interactions=10, num_sequences=3, seed=77)

    for seq_id, group in df.groupby("sequence_id"):
        assert group.iloc[0]["state"] == "B"
        assert (group.iloc[1:]["state"] == "A").all()


# ---------------------------------------------------------------------------
# 5. Categorical Distribution Boundaries
# ---------------------------------------------------------------------------

def test_categorical_distribution_valid_boundaries() -> None:
    """Verify valid categorical distribution boundaries: single choice, non-strings."""
    rng = create_rng(42)

    # 1. Single choice with probability 1.0
    val = sample_categorical(rng, ["only_item"], [1.0])
    assert val == "only_item"

    # 2. Non-string items: integers
    val_int = sample_categorical(rng, [10, 20, 30], [0.2, 0.3, 0.5])
    assert val_int in [10, 20, 30]

    # 3. Non-string items: arbitrary objects / tuples
    tuple_choices = [("low", 1), ("mid", 2), ("high", 3)]
    val_tuple = sample_categorical(rng, tuple_choices, [0.1, 0.8, 0.1])
    assert val_tuple in tuple_choices


def test_categorical_distribution_invalid_boundaries() -> None:
    """Verify invalid categorical parameters fail cleanly in sample_categorical and config validation."""
    rng = create_rng(42)

    # Empty items
    with pytest.raises(ValueError, match="Cannot sample from an empty items sequence"):
        sample_categorical(rng, [], [])

    # Length mismatch
    with pytest.raises(ValueError, match="Probabilities length .* does not match items length"):
        sample_categorical(rng, ["a", "b"], [1.0])

    # Negative probability
    with pytest.raises(ValueError, match="Probabilities must be non-negative"):
        sample_categorical(rng, ["a", "b"], [-0.1, 1.1])

    # Probability > 1
    with pytest.raises(ValueError, match=r"Probabilities must be <= 1\.0"):
        sample_categorical(rng, ["a", "b"], [1.5, 0.0])


    # Probabilities containing NaN
    with pytest.raises(ValueError, match="Probabilities contain non-finite values"):
        sample_categorical(rng, ["a", "b"], [float("nan"), 0.5])

    # Probabilities containing Inf
    with pytest.raises(ValueError, match="Probabilities contain non-finite values"):
        sample_categorical(rng, ["a", "b"], [float("inf"), 0.5])

    # Probabilities not summing to 1
    with pytest.raises(ValueError, match="Probabilities must sum to 1.0"):
        sample_categorical(rng, ["a", "b"], [0.2, 0.2])

    # Declarative config validator also rejects these boundaries
    with pytest.raises(ValueError, match="categorical parameter 'items' must be a non-empty sequence"):
        validate_distribution_params("categorical", {"items": []})

    with pytest.raises(ValueError, match="categorical 'probabilities' length"):
        validate_distribution_params("categorical", {"items": ["a", "b"], "probabilities": [1.0]})

    with pytest.raises(ValueError, match=r"must be within \[0, 1\]"):
        validate_distribution_params("categorical", {"items": ["a", "b"], "probabilities": [-0.1, 1.1]})

    with pytest.raises(ValueError, match=r"must be within \[0, 1\]"):
        validate_distribution_params("categorical", {"items": ["a", "b"], "probabilities": [1.5, -0.5]})

    with pytest.raises(ValueError, match="must be finite"):
        validate_distribution_params("categorical", {"items": ["a", "b"], "probabilities": [float("nan"), 0.5]})

    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_distribution_params("categorical", {"items": ["a", "b"], "probabilities": [0.3, 0.3]})


# ---------------------------------------------------------------------------
# 6. Probability Boundaries: Transition Rules and Profiles (0.0 and 1.0)
# ---------------------------------------------------------------------------

def test_transition_rule_probability_boundaries(two_states: List[State]) -> None:
    """Verify transition rules with probability 0.0 never fire and 1.0 always fire."""
    # Rule with probability 0.0
    rule_never = TransitionRule(
        condition=lambda h: True,
        target_state="B",
        probability=0.0,
    )
    prof_never = Profile(
        name="never_profile",
        state_emissions={"A": {}, "B": {}},
        transition_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
        transition_rules=[rule_never],
    )
    sim_never = Simulator(states=two_states, profile=prof_never, initial_state="A")
    df_never = sim_never.simulate(num_interactions=20, seed=123)
    # Since initial is A and base matrix A->A is 1.0, and rule prob=0 never fires, must stay A
    assert (df_never["state"] == "A").all()

    # Rule with probability 1.0
    rule_always = TransitionRule(
        condition=lambda h: True,
        target_state="B",
        probability=1.0,
    )
    prof_always = Profile(
        name="always_profile",
        state_emissions={"A": {}, "B": {}},
        transition_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
        transition_rules=[rule_always],
    )
    sim_always = Simulator(states=two_states, profile=prof_always, initial_state="A")
    df_always = sim_always.simulate(num_interactions=20, seed=123)
    # Step 0 is initial A; from step 1 onward rule always fires targeting B
    assert df_always.loc[0, "state"] == "A"
    assert (df_always.loc[1:, "state"] == "B").all()


def test_profile_mixture_probability_boundaries(two_states: List[State]) -> None:
    """Verify profile with probability 0.0 is never selected and 1.0 is always selected."""
    matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    prof_p0 = Profile(
        name="p_zero",
        probability=0.0,
        state_emissions={
            "A": {"special_f": FeatureDistribution(distribution_type="bernoulli", params={"p": 0.5})}
        },
        transition_matrix=matrix,
    )
    prof_p1 = Profile(
        name="p_one",
        probability=1.0,
        state_emissions={
            "A": {"standard_f": FeatureDistribution(distribution_type="bernoulli", params={"p": 0.9})}
        },
        transition_matrix=matrix,
    )

    sim = Simulator(states=two_states, profiles={"p_zero": prof_p0, "p_one": prof_p1})

    # Across 10 sequences, only p_one should ever be selected
    df = sim.simulate(num_interactions=5, num_sequences=10, seed=42)

    assert set(df["profile"].unique()) == {"p_one"}
    assert "p_zero" not in df["profile"].values
    # But heterogeneous schema union is preserved!
    assert "special_f" in df.columns
    assert "standard_f" in df.columns
    # special_f values are all NaN because p_zero was never sampled
    assert df["special_f"].isna().all()


# ---------------------------------------------------------------------------
# 7. Causality and Empty-History Behavior
# ---------------------------------------------------------------------------

def test_empty_history_causality(two_states: List[State]) -> None:
    """Verify first interaction has empty history and dynamic callbacks behave safely."""
    history_lengths_seen: List[int] = []

    def inspect_history(ctx: Any) -> float:
        h = ctx.history
        recent_scores = h.get_recent("step_hist_len", 100)
        history_lengths_seen.append(len(recent_scores))

        # Safe access to nonexistent or empty history
        assert h.get_recent("nonexistent", 5) == []
        if ctx.interaction_index == 0:
            assert len(recent_scores) == 0
        else:
            assert len(recent_scores) == ctx.interaction_index

        return float(len(recent_scores))

    profile = Profile(
        name="causal_profile",
        state_emissions={
            "A": {"step_hist_len": inspect_history},
            "B": {"step_hist_len": inspect_history},
        },
        transition_matrix=np.array([[0.5, 0.5], [0.5, 0.5]]),
    )

    sim = Simulator(states=two_states, profile=profile)
    df = sim.simulate(num_interactions=5, num_sequences=1, seed=10)

    # At interaction 0, history has 0 past emissions recorded.
    # At interaction 4, history has 4 past emissions recorded.
    assert list(df["step_hist_len"]) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert history_lengths_seen == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 8. Seed Invariance & Determinism Regression
# ---------------------------------------------------------------------------

def test_sequence_expansion_seed_invariance(two_states: List[State]) -> None:
    """Verify increasing num_sequences preserves identical traces for earlier sequences."""
    matrix = np.array([[0.6, 0.4], [0.3, 0.7]])
    profile = Profile(
        name="invariance_profile",
        state_emissions={
            "A": {"score": FeatureDistribution(distribution_type="normal", params={"loc": 10, "scale": 1})},
            "B": {"score": FeatureDistribution(distribution_type="normal", params={"loc": 20, "scale": 1})},
        },
        transition_matrix=matrix,
    )
    sim = Simulator(states=two_states, profile=profile)

    df_2seq = sim.simulate(num_interactions=10, num_sequences=2, seed=777)
    df_4seq = sim.simulate(num_interactions=10, num_sequences=4, seed=777)

    # Sequences 1 and 2 in df_4seq must be identical to df_2seq
    pd.testing.assert_frame_equal(df_2seq, df_4seq[df_4seq["sequence_id"].isin([1, 2])].reset_index(drop=True))
