"""Tests for Education Preset and compatibility facade (behaviorsim.presets.education)."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest

from src.simulator import (
    BASE_TRANSITION_MATRIX as LEGACY_BASE_TRANSITION_MATRIX,
    LEARNER_PROFILES as LEGACY_LEARNER_PROFILES,
    STATES as LEGACY_STATES,
    LearnerProfile as LegacyLearnerProfile,
    simulate_all_profiles as legacy_simulate_all_profiles,
    simulate_learner as legacy_simulate_learner,
)
from behaviorsim.presets.education import (
    BASE_TRANSITION_MATRIX as PRESET_BASE_TRANSITION_MATRIX,
    LEARNER_PROFILES as PRESET_LEARNER_PROFILES,
    STATES as PRESET_STATES,
    LearnerProfile as PresetLearnerProfile,
    simulate_all_profiles as preset_simulate_all_profiles,
    simulate_learner as preset_simulate_learner,
)


def test_import_compatibility() -> None:
    """Verify that both legacy and preset import paths resolve and function."""
    assert legacy_simulate_learner is preset_simulate_learner
    assert legacy_simulate_all_profiles is preset_simulate_all_profiles
    assert LegacyLearnerProfile is PresetLearnerProfile


def test_constants_equivalence() -> None:
    """Verify that state definitions, profiles, and transition matrices are identical."""
    assert LEGACY_STATES == PRESET_STATES
    assert LEGACY_BASE_TRANSITION_MATRIX == PRESET_BASE_TRANSITION_MATRIX
    assert list(LEGACY_LEARNER_PROFILES.keys()) == list(PRESET_LEARNER_PROFILES.keys())
    for key in LEGACY_LEARNER_PROFILES:
        assert LEGACY_LEARNER_PROFILES[key] == PRESET_LEARNER_PROFILES[key]


@pytest.mark.parametrize("profile_name", ["fast_accurate", "slow_inaccurate", "average"])
def test_deterministic_learner_simulation_equivalence(profile_name: str) -> None:
    """Verify bit-for-bit identical DataFrame output between legacy and preset single-learner simulation."""
    df_legacy = legacy_simulate_learner(
        profile_name=profile_name,
        num_interactions=100,
        learner_id=1,
        seed=12345,
    )
    df_preset = preset_simulate_learner(
        profile_name=profile_name,
        num_interactions=100,
        learner_id=1,
        seed=12345,
    )

    pd.testing.assert_frame_equal(df_legacy, df_preset)


def test_multi_profile_simulation_equivalence() -> None:
    """Verify bit-for-bit identical multi-profile DataFrames between legacy and preset simulation."""
    res_legacy = legacy_simulate_all_profiles(
        num_interactions=50,
        num_learners=2,
        seed=999,
    )
    res_preset = preset_simulate_all_profiles(
        num_interactions=50,
        num_learners=2,
        seed=999,
    )

    assert set(res_legacy.keys()) == set(res_preset.keys())
    for profile_name in res_legacy:
        pd.testing.assert_frame_equal(res_legacy[profile_name], res_preset[profile_name])
