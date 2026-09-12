"""Comprehensive tests for BehaviorSim profile clustering subsystem."""

from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from behaviorsim.calibration.clustering import (
    ProfileClusteringResult,
    cluster_profiles,
)
from behaviorsim.calibration.fitter import CalibrationData
from behaviorsim.calibration.validator import (
    ValidationReport,
    validate_calibration,
)
from behaviorsim.core.profile import Profile
from behaviorsim.core.simulator import Simulator
from behaviorsim.core.state import State


@pytest.fixture
def multi_sequence_dataset() -> CalibrationData:
    """Fixture providing 6 sequences partitioned into 2 distinct behavioral types.

    Group 1 (seq 1, 2, 3): Fast, mostly active state, low response time.
    Group 2 (seq 4, 5, 6): Slow, mostly idle state, high response time.
    """
    rows = []
    # Group 1: Fast / Active
    for seq_id in [1, 2, 3]:
        for _ in range(10):
            rows.append(
                {
                    "sequence_id": seq_id,
                    "state": "active",
                    "response_time": 0.2,
                    "clicks": 5,
                }
            )
        rows.append(
            {
                "sequence_id": seq_id,
                "state": "idle",
                "response_time": 0.3,
                "clicks": 1,
            }
        )

    # Group 2: Slow / Idle
    for seq_id in [4, 5, 6]:
        for _ in range(10):
            rows.append(
                {
                    "sequence_id": seq_id,
                    "state": "idle",
                    "response_time": 5.0,
                    "clicks": 0,
                }
            )
        rows.append(
            {
                "sequence_id": seq_id,
                "state": "active",
                "response_time": 4.0,
                "clicks": 1,
            }
        )

    df = pd.DataFrame(rows)
    return CalibrationData(
        data=df,
        states=["idle", "active"],
        numeric_features=["response_time", "clicks"],
    )


def test_cluster_profiles_two_separated_groups(multi_sequence_dataset: CalibrationData) -> None:
    """Verify clustering accurately separates two distinct behavioral personas."""
    result = cluster_profiles(multi_sequence_dataset, n_profiles=2, seed=42)

    assert isinstance(result, ProfileClusteringResult)
    assert len(result.profiles) == 2
    assert "profile_0" in result.profiles
    assert "profile_1" in result.profiles

    # Proportions should sum to 1.0 (equal group sizes: 3 sequences each -> 0.5, 0.5)
    assert np.isclose(sum(result.profile_distribution.values()), 1.0)
    assert np.isclose(result.profile_distribution["profile_0"], 0.5)
    assert np.isclose(result.profile_distribution["profile_1"], 0.5)

    # Check sequence partition: sequences 1,2,3 should share a cluster; 4,5,6 should share the other
    g1_cluster = result.assignments[1]
    assert result.assignments[2] == g1_cluster
    assert result.assignments[3] == g1_cluster

    g2_cluster = result.assignments[4]
    assert result.assignments[5] == g2_cluster
    assert result.assignments[6] == g2_cluster
    assert g1_cluster != g2_cluster

    # Check fitted profiles
    for p_name, prof in result.profiles.items():
        assert isinstance(prof, Profile)
        assert prof.transition_matrix is not None
        assert prof.probability == 0.5
        assert prof.name == p_name


def test_cluster_profiles_deterministic_same_seed(multi_sequence_dataset: CalibrationData) -> None:
    """Verify repeated clustering with the same seed produces identical results."""
    res1 = cluster_profiles(multi_sequence_dataset, n_profiles=2, seed=12345)
    res2 = cluster_profiles(multi_sequence_dataset, n_profiles=2, seed=12345)

    assert res1.assignments.equals(res2.assignments)
    assert res1.profile_distribution == res2.profile_distribution
    for p_name in res1.profiles:
        np.testing.assert_allclose(
            res1.profiles[p_name].transition_matrix,
            res2.profiles[p_name].transition_matrix,
        )


def test_cluster_profiles_single_profile(multi_sequence_dataset: CalibrationData) -> None:
    """Verify n_profiles=1 groups all sequences into a single profile with probability 1.0."""
    result = cluster_profiles(multi_sequence_dataset, n_profiles=1, seed=42)

    assert len(result.profiles) == 1
    assert "profile_0" in result.profiles
    assert result.profile_distribution["profile_0"] == 1.0
    assert (result.assignments == "profile_0").all()
    assert result.profiles["profile_0"].probability == 1.0


def test_cluster_profiles_n_profiles_zero_or_negative_rejected(
    multi_sequence_dataset: CalibrationData,
) -> None:
    """Verify n_profiles < 1 raises ValueError."""
    with pytest.raises(ValueError, match="n_profiles must be at least 1"):
        cluster_profiles(multi_sequence_dataset, n_profiles=0)

    with pytest.raises(ValueError, match="n_profiles must be at least 1"):
        cluster_profiles(multi_sequence_dataset, n_profiles=-2)


def test_cluster_profiles_more_clusters_than_sequences_rejected(
    multi_sequence_dataset: CalibrationData,
) -> None:
    """Verify n_profiles > number of sequences raises ValueError."""
    # multi_sequence_dataset has 6 sequences
    with pytest.raises(ValueError, match="Cannot form 10 clusters from 6 sequence entities"):
        cluster_profiles(multi_sequence_dataset, n_profiles=10)


def test_cluster_profiles_missing_sequence_column_rejected() -> None:
    """Verify multi-profile clustering on data without sequence_column raises ValueError."""
    df = pd.DataFrame(
        {
            "state": ["idle", "active", "idle"],
            "response_time": [1.0, 0.5, 1.2],
        }
    )
    calib = CalibrationData(
        data=df,
        states=["idle", "active"],
        sequence_column=None,
        numeric_features=["response_time"],
    )

    with pytest.raises(ValueError, match="Cannot cluster into 2 profiles when sequence_column is None"):
        cluster_profiles(calib, n_profiles=2)


def test_cluster_profiles_probability_normalization(multi_sequence_dataset: CalibrationData) -> None:
    """Verify profile mixture probabilities strictly sum to 1.0."""
    result = cluster_profiles(multi_sequence_dataset, n_profiles=2, seed=42)
    assert np.isclose(sum(result.profile_distribution.values()), 1.0)



def test_cluster_profiles_integration_with_simulator(
    multi_sequence_dataset: CalibrationData,
) -> None:
    """End-to-end integration: cluster profiles -> Simulator -> generate -> validate."""
    # 1. Cluster empirical data into 2 profiles
    clustering_res = cluster_profiles(
        multi_sequence_dataset,
        n_profiles=2,
        seed=42,
        distribution_types={"response_time": "normal", "clicks": "poisson"},
    )

    # 2. Feed directly into Simulator multi-profile architecture
    states = [State("idle"), State("active")]
    sim = Simulator(
        states=states,
        profiles=clustering_res.profiles,
        profile_distribution=clustering_res.profile_distribution,
    )

    # 3. Generate synthetic trace with multiple sequences
    synthetic_df = sim.generate(num_interactions=10, num_sequences=10, seed=100)

    assert len(synthetic_df) == 100
    assert "profile" in synthetic_df.columns
    assert set(synthetic_df["profile"].unique()).issubset({"profile_0", "profile_1"})
    assert set(synthetic_df["state"].unique()).issubset({"idle", "active"})

    # 4. Validate synthetic data against empirical data
    report = validate_calibration(
        empirical_data=multi_sequence_dataset,
        synthetic_data=synthetic_df,
    )
    assert isinstance(report, ValidationReport)
    assert report.structural.is_valid
    assert report.state_occupancy.total_variation_distance >= 0.0
