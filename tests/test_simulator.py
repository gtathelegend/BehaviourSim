"""Unit tests for synthetic learner simulator module."""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from src.config import Config
from src.simulator import (
    LEARNER_PROFILES,
    STATES,
    simulate_learner,
    simulate_all_profiles,
    save_simulation,
    run_simulation,
)


class TestSimulator(unittest.TestCase):
    """Test suite for CLSI-Adapt learner simulator."""

    def setUp(self) -> None:
        self.config = Config(
            seed=42,
            num_interactions_per_learner=50,
            feature_window_size=5,
        )
        self.expected_columns = [
            "profile",
            "interaction_id",
            "state",
            "difficulty",
            "accuracy",
            "nrt",
            "window_error_rate",
            "retries",
            "help_requested",
            "confidence",
            "streak_correct",
            "streak_incorrect",
            "nrt_variance",
            "session_time",
        ]

    def test_all_profiles_generated(self) -> None:
        """Verify all 5 learner profiles are correctly generated."""
        results = simulate_all_profiles(
            num_interactions=20, seed=42, config=self.config
        )
        expected_profiles = {
            "fast_accurate",
            "fast_inaccurate",
            "slow_accurate",
            "slow_inaccurate",
            "average",
        }
        self.assertEqual(set(results.keys()), expected_profiles)
        self.assertEqual(len(results), 5)

        for profile_name, df in results.items():
            self.assertEqual(len(df), 20)
            self.assertTrue((df["profile"] == profile_name).all())

    def test_valid_states(self) -> None:
        """Verify only valid cognitive states occur."""
        for profile in LEARNER_PROFILES:
            df = simulate_learner(profile, num_interactions=30, seed=42)
            unique_states = set(df["state"].unique())
            self.assertTrue(unique_states.issubset(set(STATES)))

    def test_difficulty_range(self) -> None:
        """Verify difficulty values are within range 1 to 5."""
        df = simulate_learner("average", num_interactions=100, seed=42)
        self.assertTrue((df["difficulty"] >= 1).all())
        self.assertTrue((df["difficulty"] <= 5).all())
        self.assertEqual(set(df["difficulty"].unique()).issubset({1, 2, 3, 4, 5}), True)

    def test_accuracy_binary(self) -> None:
        """Verify accuracy is strictly binary (0 or 1)."""
        df = simulate_learner("average", num_interactions=50, seed=42)
        unique_acc = set(df["accuracy"].unique())
        self.assertTrue(unique_acc.issubset({0, 1}))

    def test_nrt_non_negative(self) -> None:
        """Verify normalized response time (NRT) is non-negative."""
        for profile in LEARNER_PROFILES:
            df = simulate_learner(profile, num_interactions=50, seed=42)
            self.assertTrue((df["nrt"] >= 0.0).all())
            self.assertTrue((df["session_time"] >= 0.0).all())

    def test_retries_valid(self) -> None:
        """Verify retries are valid integers >= 0."""
        df = simulate_learner("fast_inaccurate", num_interactions=100, seed=42)
        self.assertTrue((df["retries"] >= 0).all())
        self.assertTrue(np.issubdtype(df["retries"].dtype, np.integer))

    def test_help_requested_binary(self) -> None:
        """Verify help_requested is strictly binary (0 or 1)."""
        df = simulate_learner("average", num_interactions=50, seed=42)
        unique_help = set(df["help_requested"].unique())
        self.assertTrue(unique_help.issubset({0, 1}))

    def test_confidence_range(self) -> None:
        """Verify confidence values are integers within 1 to 5."""
        df = simulate_learner("average", num_interactions=100, seed=42)
        unique_conf = set(df["confidence"].unique())
        self.assertTrue(unique_conf.issubset({1, 2, 3, 4, 5}))

    def test_required_columns(self) -> None:
        """Verify required exact column names exist."""
        df = simulate_learner("average", num_interactions=10, seed=42)
        self.assertListEqual(list(df.columns), self.expected_columns)

    def test_deterministic_output_identical_seed(self) -> None:
        """Verify identical seed produces identical output."""
        df1 = simulate_learner("fast_accurate", num_interactions=50, seed=123)
        df2 = simulate_learner("fast_accurate", num_interactions=50, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_produces_different_output(self) -> None:
        """Verify different seed produces different data."""
        df1 = simulate_learner("fast_accurate", num_interactions=50, seed=123)
        df2 = simulate_learner("fast_accurate", num_interactions=50, seed=999)
        self.assertFalse(df1.equals(df2))

    def test_no_nans(self) -> None:
        """Verify there are no NaN values in output columns."""
        for profile in LEARNER_PROFILES:
            df = simulate_learner(profile, num_interactions=50, seed=42)
            self.assertEqual(df.isna().sum().sum(), 0)

    def test_performance_dependent_transitions(self) -> None:
        """Verify performance-dependent transitions influence state generation.

        When a learner has very low accuracy (e.g. slow_inaccurate), Overload transitions
        should occur with higher frequency compared to high accuracy learners.
        """
        df_low = simulate_learner("slow_inaccurate", num_interactions=150, seed=42)
        df_high = simulate_learner("fast_accurate", num_interactions=150, seed=42)

        overload_ratio_low = (df_low["state"] == "Overload").mean()
        overload_ratio_high = (df_high["state"] == "Overload").mean()

        # The low accuracy profile should experience significantly more Overload states
        self.assertGreater(overload_ratio_low, overload_ratio_high)

    def test_save_simulation(self) -> None:
        """Verify saving simulation output CSV files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            results = simulate_all_profiles(num_interactions=10, seed=42)
            saved_paths = save_simulation(results, output_dir=tmp_path)

            expected_filenames = [
                "simulated_learner_fast_accurate.csv",
                "simulated_learner_fast_inaccurate.csv",
                "simulated_learner_slow_accurate.csv",
                "simulated_learner_slow_inaccurate.csv",
                "simulated_learner_average.csv",
                "simulated_learners_all.csv",
            ]
            saved_filenames = [p.name for p in saved_paths]
            self.assertEqual(sorted(saved_filenames), sorted(expected_filenames))

            # Verify combined CSV contains data for all 5 profiles
            combined_df = pd.read_csv(tmp_path / "simulated_learners_all.csv")
            self.assertEqual(len(combined_df), 50)
            self.assertListEqual(list(combined_df.columns), self.expected_columns)


if __name__ == "__main__":
    unittest.main()
