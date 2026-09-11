"""Unit tests for Phase 1.3 — Multi-Profile Simulation & Mixture Sampling."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
import yaml

from behaviorsim import (
    FeatureDistribution,
    Profile,
    SimulationConfig,
    Simulator,
    State,
    TransitionRule,
)
from behaviorsim.config import (
    build_simulator,
    load_config,
    parse_config,
    run_simulation,
    validate_config,
)


# ---------------------------------------------------------------------------
# Test Fixtures & Utilities
# ---------------------------------------------------------------------------

@pytest.fixture
def states_ab() -> list[State]:
    return [State("state_a"), State("state_b")]


@pytest.fixture
def profile_alpha() -> Profile:
    emissions = {
        "state_a": {
            "score": FeatureDistribution("normal", {"loc": 100.0, "scale": 0.5}),
            "feature_x": FeatureDistribution("bernoulli", {"p": 1.0}),
        },
        "state_b": {
            "score": FeatureDistribution("normal", {"loc": 100.0, "scale": 0.5}),
            "feature_x": FeatureDistribution("bernoulli", {"p": 1.0}),
        },
    }
    matrix = np.array([[0.8, 0.2], [0.2, 0.8]])
    return Profile(name="alpha", state_emissions=emissions, transition_matrix=matrix)


@pytest.fixture
def profile_beta() -> Profile:
    emissions = {
        "state_a": {
            "score": FeatureDistribution("normal", {"loc": 500.0, "scale": 0.5}),
            "feature_y": FeatureDistribution("bernoulli", {"p": 0.0}),
        },
        "state_b": {
            "score": FeatureDistribution("normal", {"loc": 500.0, "scale": 0.5}),
            "feature_y": FeatureDistribution("bernoulli", {"p": 0.0}),
        },
    }
    matrix = np.array([[0.3, 0.7], [0.7, 0.3]])
    return Profile(name="beta", state_emissions=emissions, transition_matrix=matrix)


# ---------------------------------------------------------------------------
# A. Single-Profile Regression
# ---------------------------------------------------------------------------

def test_single_profile_regression_backward_compatibility(
    states_ab: list[State],
    profile_alpha: Profile,
) -> None:
    """Verify existing single-profile construction and simulation are bit-for-bit identical."""
    # Positional args Simulator(states, profile)
    sim_pos = Simulator(states_ab, profile_alpha)
    df_pos = sim_pos.simulate(num_interactions=20, num_sequences=2, seed=42)

    # Keyword arg Simulator(states=..., profile=...)
    sim_kw = Simulator(states=states_ab, profile=profile_alpha)
    df_kw = sim_kw.simulate(num_interactions=20, num_sequences=2, seed=42)

    pd.testing.assert_frame_equal(df_pos, df_kw)

    # Internal properties
    assert sim_pos.profile is profile_alpha
    assert list(sim_pos.profiles.keys()) == ["alpha"]
    assert sim_pos.profile_distribution == {"alpha": 1.0}
    assert (df_pos["profile"] == "alpha").all()
    assert (df_pos["feature_x"] == 1).all()


def test_single_profile_in_profiles_list_matches_single_profile_mode(
    states_ab: list[State],
    profile_alpha: Profile,
) -> None:
    """Verify passing a single profile via profiles=[p] behaves identically to profile=p."""
    sim_single = Simulator(states_ab, profile=profile_alpha)
    df_single = sim_single.simulate(num_interactions=20, num_sequences=3, seed=123)

    sim_list = Simulator(states=states_ab, profiles=[profile_alpha])
    df_list = sim_list.simulate(num_interactions=20, num_sequences=3, seed=123)

    pd.testing.assert_frame_equal(df_single, df_list)


# ---------------------------------------------------------------------------
# B. Two-Profile Mixture & Selection Distribution
# ---------------------------------------------------------------------------

def test_two_profile_mixture_sampling(
    states_ab: list[State],
    profile_alpha: Profile,
    profile_beta: Profile,
) -> None:
    """Verify both profiles appear according to specified distribution across sequences."""
    dist = {"alpha": 0.4, "beta": 0.6}
    sim = Simulator(
        states=states_ab,
        profiles=[profile_alpha, profile_beta],
        profile_distribution=dist,
    )

    n_sequences = 500
    df = sim.simulate(num_interactions=5, num_sequences=n_sequences, seed=999)

    # Extract assigned profile per sequence
    seq_profiles = df.groupby("sequence_id")["profile"].first()
    assert len(seq_profiles) == n_sequences
    counts = seq_profiles.value_counts().to_dict()

    alpha_frac = counts.get("alpha", 0) / n_sequences
    beta_frac = counts.get("beta", 0) / n_sequences

    # For n=500, p=0.4, std err = sqrt(0.4*0.6/500) = 0.022. 3 sigma is ~0.066
    assert abs(alpha_frac - 0.4) < 0.07
    assert abs(beta_frac - 0.6) < 0.07


def test_mixture_deterministic_assignments(
    states_ab: list[State],
    profile_alpha: Profile,
    profile_beta: Profile,
) -> None:
    """Verify exact sequence-to-profile assignments are completely reproducible."""
    sim = Simulator(
        states=states_ab,
        profiles={"alpha": profile_alpha, "beta": profile_beta},
        profile_distribution={"alpha": 0.5, "beta": 0.5},
    )

    df1 = sim.simulate(num_interactions=10, num_sequences=5, seed=42)
    df2 = sim.simulate(num_interactions=10, num_sequences=5, seed=42)

    pd.testing.assert_frame_equal(df1, df2)

    seq_profiles_1 = df1.groupby("sequence_id")["profile"].first().tolist()
    seq_profiles_2 = df2.groupby("sequence_id")["profile"].first().tolist()
    assert seq_profiles_1 == seq_profiles_2


# ---------------------------------------------------------------------------
# C. Probability & Argument Validation
# ---------------------------------------------------------------------------

def test_probability_validation_sum_must_be_one(
    states_ab: list[State],
    profile_alpha: Profile,
    profile_beta: Profile,
) -> None:
    """Verify probabilities summing to != 1.0 raise ValueError."""
    # Sum < 1
    with pytest.raises(ValueError, match="Profile probabilities must sum to 1.0"):
        Simulator(
            states=states_ab,
            profiles=[profile_alpha, profile_beta],
            profile_distribution={"alpha": 0.3, "beta": 0.3},
        )

    # Sum > 1
    with pytest.raises(ValueError, match="Profile probabilities must sum to 1.0"):
        Simulator(
            states=states_ab,
            profiles=[profile_alpha, profile_beta],
            profile_distribution={"alpha": 0.6, "beta": 0.6},
        )


def test_probability_validation_out_of_bounds(
    states_ab: list[State],
    profile_alpha: Profile,
    profile_beta: Profile,
) -> None:
    """Verify probabilities < 0 or > 1 raise ValueError."""
    # Negative probability
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        Simulator(
            states=states_ab,
            profiles=[profile_alpha, profile_beta],
            profile_distribution={"alpha": -0.2, "beta": 1.2},
        )

    # > 1.0 probability
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        Simulator(
            states=states_ab,
            profiles=[profile_alpha, profile_beta],
            profile_distribution={"alpha": 1.5, "beta": -0.5},
        )


def test_probability_validation_non_finite(
    states_ab: list[State],
    profile_alpha: Profile,
    profile_beta: Profile,
) -> None:
    """Verify NaN and Inf probabilities raise ValueError."""
    with pytest.raises(ValueError, match="must be finite"):
        Simulator(
            states=states_ab,
            profiles=[profile_alpha, profile_beta],
            profile_distribution={"alpha": float("nan"), "beta": 0.5},
        )

    with pytest.raises(ValueError, match="must be finite"):
        Simulator(
            states=states_ab,
            profiles=[profile_alpha, profile_beta],
            profile_distribution={"alpha": float("inf"), "beta": 0.0},
        )


def test_profile_distribution_missing_and_unknown_names(
    states_ab: list[State],
    profile_alpha: Profile,
    profile_beta: Profile,
) -> None:
    """Verify unknown and missing profile names fail clearly."""
    # Unknown profile name
    with pytest.raises(ValueError, match="Unknown profile name 'unknown_prof'"):
        Simulator(
            states=states_ab,
            profiles=[profile_alpha, profile_beta],
            profile_distribution={"alpha": 0.5, "unknown_prof": 0.5},
        )

    # Missing profile name
    with pytest.raises(ValueError, match="Missing profile 'beta' in profile_distribution"):
        Simulator(
            states=states_ab,
            profiles=[profile_alpha, profile_beta],
            profile_distribution={"alpha": 1.0},
        )


def test_profile_empty_and_mutual_exclusion(
    states_ab: list[State],
    profile_alpha: Profile,
) -> None:
    """Verify empty profiles or specifying both/neither profile and profiles fails."""
    # Empty profile mapping in profile_distribution
    with pytest.raises(ValueError, match="profile_distribution cannot be empty"):
        Simulator(
            states=states_ab,
            profiles=[profile_alpha],
            profile_distribution={},
        )

    # Empty profiles list
    with pytest.raises(ValueError, match="profiles must contain at least one Profile"):
        Simulator(states=states_ab, profiles=[])

    # Empty profiles mapping
    with pytest.raises(ValueError, match="profiles must contain at least one Profile"):
        Simulator(states=states_ab, profiles={})

    # Specifying both profile and profiles
    with pytest.raises(ValueError, match="Cannot specify both 'profile' and 'profiles'"):
        Simulator(states=states_ab, profile=profile_alpha, profiles=[profile_alpha])

    # Specifying neither profile nor profiles
    with pytest.raises(ValueError, match="Simulator requires either 'profile' or 'profiles'"):
        Simulator(states=states_ab)

    # Multiple profiles without profile_distribution
    p2 = Profile(name="p2", state_emissions={})
    with pytest.raises(ValueError, match="profile_distribution must be explicitly specified"):
        Simulator(states=states_ab, profiles=[profile_alpha, p2], transition_matrix=np.eye(2))


def test_profile_name_mismatch_in_mapping(
    states_ab: list[State],
    profile_alpha: Profile,
) -> None:
    """Verify mapping key mismatch with Profile.name raises ValueError."""
    with pytest.raises(ValueError, match="key 'wrong_key' does not match Profile.name 'alpha'"):
        Simulator(states=states_ab, profiles={"wrong_key": profile_alpha})


# ---------------------------------------------------------------------------
# D. Determinism
# ---------------------------------------------------------------------------

def test_determinism_across_runs(
    states_ab: list[State],
    profile_alpha: Profile,
    profile_beta: Profile,
) -> None:
    """Verify identical parameters and seed produce identical DataFrames."""
    sim = Simulator(
        states=states_ab,
        profiles=[profile_alpha, profile_beta],
        profile_distribution={"alpha": 0.5, "beta": 0.5},
    )

    df1 = sim.simulate(num_interactions=50, num_sequences=4, seed=777)
    df2 = sim.simulate(num_interactions=50, num_sequences=4, seed=777)

    pd.testing.assert_frame_equal(df1, df2)

    df_diff = sim.simulate(num_interactions=50, num_sequences=4, seed=778)
    assert not df1.equals(df_diff)


# ---------------------------------------------------------------------------
# E. Sequence Isolation & Independence
# ---------------------------------------------------------------------------

def test_sequence_isolation_earlier_sequences_invariant(
    states_ab: list[State],
    profile_alpha: Profile,
    profile_beta: Profile,
) -> None:
    """Verify sequence 1 is identical whether num_sequences is 1 or 5."""
    sim = Simulator(
        states=states_ab,
        profiles=[profile_alpha, profile_beta],
        profile_distribution={"alpha": 0.5, "beta": 0.5},
    )

    df_single_seq = sim.simulate(num_interactions=30, num_sequences=1, seed=42)
    df_multi_seq = sim.simulate(num_interactions=30, num_sequences=5, seed=42)

    # Sequence 1 from multi-sequence run (check_dtype=False because pandas dtype inference
    # for columns with None in subsequent sequences promotes int64 to object upon concatenation)
    df_seq1_extracted = df_multi_seq[df_multi_seq["sequence_id"] == 1].reset_index(drop=True)

    pd.testing.assert_frame_equal(df_single_seq, df_seq1_extracted, check_dtype=False)


def test_mixture_sequence_history_reset(states_ab: list[State]) -> None:
    """Verify history context resets cleanly between sequences under mixture."""
    seen_history_lengths: list[int] = []

    def dynamic_cb(ctx) -> float:
        hist = ctx.history.get_recent("val", 100)
        seen_history_lengths.append(len(hist))
        return 1.0

    p1 = Profile(
        name="p1",
        state_emissions={"state_a": {"val": dynamic_cb}, "state_b": {"val": dynamic_cb}},
        transition_matrix=np.eye(2),
    )
    p2 = Profile(
        name="p2",
        state_emissions={"state_a": {"val": dynamic_cb}, "state_b": {"val": dynamic_cb}},
        transition_matrix=np.eye(2),
    )

    sim = Simulator(
        states=states_ab,
        profiles=[p1, p2],
        profile_distribution={"p1": 0.5, "p2": 0.5},
    )

    # 3 sequences of 4 interactions
    sim.simulate(num_interactions=4, num_sequences=3, seed=123)

    # In every sequence, history length must reset to 0, 1, 2, 3
    assert seen_history_lengths == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]


# ---------------------------------------------------------------------------
# F. Profile-Specific Emissions
# ---------------------------------------------------------------------------

def test_profile_specific_emissions(
    states_ab: list[State],
    profile_alpha: Profile,
    profile_beta: Profile,
) -> None:
    """Verify observations strictly correspond to the selected profile."""
    sim = Simulator(
        states=states_ab,
        profiles=[profile_alpha, profile_beta],
        profile_distribution={"alpha": 0.5, "beta": 0.5},
    )

    df = sim.simulate(num_interactions=20, num_sequences=10, seed=42)

    for seq_id, seq_df in df.groupby("sequence_id"):
        prof_name = seq_df["profile"].iloc[0]
        if prof_name == "alpha":
            # loc=100.0, scale=0.5 -> all scores should be between 95 and 105
            assert (seq_df["score"] > 90.0).all() and (seq_df["score"] < 110.0).all()
        elif prof_name == "beta":
            # loc=500.0, scale=0.5 -> all scores should be between 490 and 510
            assert (seq_df["score"] > 480.0).all() and (seq_df["score"] < 520.0).all()
        else:
            pytest.fail(f"Unexpected profile {prof_name}")


# ---------------------------------------------------------------------------
# G. Profile-Specific Transition Matrices
# ---------------------------------------------------------------------------

def test_profile_specific_transition_matrices(states_ab: list[State]) -> None:
    """Verify each profile transitions according to its own transition matrix override."""
    # p_stay_a always stays in state_a
    p_stay_a = Profile(
        name="stay_a",
        state_emissions={"state_a": {"v": FeatureDistribution("normal", {"loc": 1.0})}, "state_b": {}},
        transition_matrix=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )
    # p_switch_b always switches to state_b and stays
    p_switch_b = Profile(
        name="switch_b",
        state_emissions={"state_a": {}, "state_b": {"v": FeatureDistribution("normal", {"loc": 2.0})}},
        transition_matrix=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )

    sim = Simulator(
        states=states_ab,
        profiles=[p_stay_a, p_switch_b],
        profile_distribution={"stay_a": 0.5, "switch_b": 0.5},
        initial_state="state_a",
    )

    df = sim.simulate(num_interactions=10, num_sequences=10, seed=123)

    for seq_id, seq_df in df.groupby("sequence_id"):
        prof_name = seq_df["profile"].iloc[0]
        if prof_name == "stay_a":
            # Must remain state_a for all interactions
            assert (seq_df["state"] == "state_a").all()
        elif prof_name == "switch_b":
            # Step 0 is initial_state ("state_a"), from step 1 onward it switches to "state_b"
            assert seq_df.iloc[0]["state"] == "state_a"
            assert (seq_df.iloc[1:]["state"] == "state_b").all()


# ---------------------------------------------------------------------------
# H. Heterogeneous Feature Schema
# ---------------------------------------------------------------------------

def test_heterogeneous_feature_schema(states_ab: list[State]) -> None:
    """Verify profiles with different feature sets produce a combined schema with missing values."""
    prof_xy = Profile(
        name="xy_user",
        state_emissions={
            "state_a": {
                "feat_x": FeatureDistribution("normal", {"loc": 10.0}),
                "feat_y": FeatureDistribution("normal", {"loc": 20.0}),
            },
            "state_b": {
                "feat_x": FeatureDistribution("normal", {"loc": 10.0}),
                "feat_y": FeatureDistribution("normal", {"loc": 20.0}),
            },
        },
        transition_matrix=np.eye(2),
    )

    prof_xz = Profile(
        name="xz_user",
        state_emissions={
            "state_a": {
                "feat_x": FeatureDistribution("normal", {"loc": 30.0}),
                "feat_z": FeatureDistribution("normal", {"loc": 40.0}),
            },
            "state_b": {
                "feat_x": FeatureDistribution("normal", {"loc": 30.0}),
                "feat_z": FeatureDistribution("normal", {"loc": 40.0}),
            },
        },
        transition_matrix=np.eye(2),
    )

    sim = Simulator(
        states=states_ab,
        profiles=[prof_xy, prof_xz],
        profile_distribution={"xy_user": 0.5, "xz_user": 0.5},
    )

    df = sim.simulate(num_interactions=5, num_sequences=6, seed=42)

    # Columns must contain the union in deterministic order
    expected_cols = ["profile", "sequence_id", "interaction_id", "state", "feat_x", "feat_y", "feat_z"]
    assert list(df.columns) == expected_cols

    xy_df = df[df["profile"] == "xy_user"]
    assert not xy_df.empty
    assert xy_df["feat_x"].notna().all()
    assert xy_df["feat_y"].notna().all()
    assert xy_df["feat_z"].isna().all()  # Missing feature is null / NaN

    xz_df = df[df["profile"] == "xz_user"]
    assert not xz_df.empty
    assert xz_df["feat_x"].notna().all()
    assert xz_df["feat_z"].notna().all()
    assert xz_df["feat_y"].isna().all()  # Missing feature is null / NaN


# ---------------------------------------------------------------------------
# I. Configuration Integration (YAML & JSON)
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_profile_raw_config() -> dict[str, Any]:
    return {
        "version": "1.0",
        "states": [
            {"name": "state_1", "description": "State 1"},
            {"name": "state_2", "description": "State 2"},
        ],
        "transition_matrix": [
            [0.7, 0.3],
            [0.3, 0.7],
        ],
        "profiles": [
            {
                "name": "profile_one",
                "state_emissions": {
                    "state_1": {"signal": {"distribution": "normal", "params": {"loc": 10.0, "scale": 1.0}}},
                    "state_2": {"signal": {"distribution": "normal", "params": {"loc": 10.0, "scale": 1.0}}},
                },
            },
            {
                "name": "profile_two",
                "state_emissions": {
                    "state_1": {"signal": {"distribution": "normal", "params": {"loc": 50.0, "scale": 1.0}}},
                    "state_2": {"signal": {"distribution": "normal", "params": {"loc": 50.0, "scale": 1.0}}},
                },
            },
        ],
        "simulation": {
            "num_interactions": 10,
            "num_sequences": 10,
            "seed": 42,
            "profile_distribution": {
                "profile_one": 0.4,
                "profile_two": 0.6,
            },
        },
    }


def test_config_parsing_and_execution_with_profile_distribution(
    multi_profile_raw_config: dict[str, Any],
) -> None:
    """Verify parsing and simulating multi-profile configuration with simulation.profile_distribution."""
    cfg = parse_config(multi_profile_raw_config)
    validate_config(cfg)

    assert cfg.profile_distribution == {"profile_one": 0.4, "profile_two": 0.6}
    assert cfg.simulation.profile_distribution == {"profile_one": 0.4, "profile_two": 0.6}

    # build_simulator without explicit profile_name runs mixture
    sim = build_simulator(cfg)
    assert len(sim.profiles) == 2
    assert sim.profile is None

    df = run_simulation(cfg)
    assert len(df) == 100
    assert set(df["profile"].unique()) == {"profile_one", "profile_two"}


def test_config_with_profile_level_probabilities() -> None:
    """Verify specifying probability on each ProfileConfig infers profile_distribution."""
    raw = {
        "version": "1.0",
        "states": [{"name": "s1"}],
        "transition_matrix": [[1.0]],
        "profiles": [
            {
                "name": "p_a",
                "probability": 0.3,
                "state_emissions": {"s1": {"val": {"distribution": "normal", "params": {"loc": 1.0}}}},
            },
            {
                "name": "p_b",
                "probability": 0.7,
                "state_emissions": {"s1": {"val": {"distribution": "normal", "params": {"loc": 2.0}}}},
            },
        ],
        "simulation": {"num_interactions": 5, "num_sequences": 5, "seed": 42},
    }
    cfg = parse_config(raw)
    validate_config(cfg)
    assert cfg.profile_distribution == {"p_a": 0.3, "p_b": 0.7}

    sim = build_simulator(cfg)
    assert sim.profile_distribution == {"p_a": 0.3, "p_b": 0.7}


def test_config_yaml_file_loading_multi_profile(
    multi_profile_raw_config: dict[str, Any],
) -> None:
    """Verify loading multi-profile YAML file via load_config and Simulator.from_config."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = Path(tmp_dir) / "multi_sim.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(multi_profile_raw_config, f)

        cfg = load_config(yaml_path)
        sim = Simulator.from_config(yaml_path)

        assert len(sim.profiles) == 2
        df = sim.generate(num_interactions=10, num_sequences=6, seed=42)
        assert len(df) == 60
        assert "profile" in df.columns


def test_config_json_file_loading_multi_profile(
    multi_profile_raw_config: dict[str, Any],
) -> None:
    """Verify loading multi-profile JSON file via load_config and Simulator.from_config."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "multi_sim.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(multi_profile_raw_config, f)

        cfg = load_config(json_path)
        sim = Simulator.from_config(json_path)

        assert len(sim.profiles) == 2
        df = sim.generate(num_interactions=10, num_sequences=6, seed=42)
        assert len(df) == 60


def test_config_profile_probabilities_invalid_sum() -> None:
    """Verify config validation rejects profile probabilities not summing to 1.0."""
    raw = {
        "version": "1.0",
        "states": [{"name": "s1"}],
        "transition_matrix": [[1.0]],
        "profiles": [
            {"name": "p_a", "probability": 0.2, "state_emissions": {"s1": {}}},
            {"name": "p_b", "probability": 0.5, "state_emissions": {"s1": {}}},
        ],
        "simulation": {"num_interactions": 5},
    }
    with pytest.raises(ValueError, match="profile probabilities must sum to 1.0"):
        parse_config(raw)


# ---------------------------------------------------------------------------
# J. Public API Compatibility
# ---------------------------------------------------------------------------

def test_public_api_generate_and_facade(
    states_ab: list[State],
    profile_alpha: Profile,
    profile_beta: Profile,
) -> None:
    """Verify Simulator.generate() produces identical results to simulate() for mixture."""
    sim = Simulator(
        states=states_ab,
        profiles=[profile_alpha, profile_beta],
        profile_distribution={"alpha": 0.5, "beta": 0.5},
    )

    df_sim = sim.simulate(num_interactions=30, num_sequences=4, seed=12345)
    df_gen = sim.generate(num_interactions=30, num_sequences=4, seed=12345)

    pd.testing.assert_frame_equal(df_sim, df_gen)
