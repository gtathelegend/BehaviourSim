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
            "learner_id",
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
        """Verify all 5 learner profiles are correctly generated across multiple learners."""
        results = simulate_all_profiles(
            num_interactions=20, num_learners=2, seed=42, config=self.config
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
            self.assertEqual(len(df), 40)  # 2 learners x 20 interactions
            self.assertTrue((df["profile"] == profile_name).all())
            self.assertEqual(set(df["learner_id"].unique()), {1, 2})

    def test_multi_learner_state_reset(self) -> None:
        """Verify independent learners reset session time, streaks, and interaction_ids."""
        df_l1 = simulate_learner("average", num_interactions=20, learner_id=1, seed=42)
        df_l2 = simulate_learner("average", num_interactions=20, learner_id=2, seed=99)

        # Interaction IDs reset per learner to 1..20
        self.assertListEqual(list(df_l1["interaction_id"]), list(range(1, 21)))
        self.assertListEqual(list(df_l2["interaction_id"]), list(range(1, 21)))

        # Session time starts small for interaction 1 on both
        self.assertLess(df_l1["session_time"].iloc[0], 100.0)
        self.assertLess(df_l2["session_time"].iloc[0], 100.0)

        # Initial cognitive state starts at Optimal
        self.assertEqual(df_l1["state"].iloc[0], "Optimal")
        self.assertEqual(df_l2["state"].iloc[0], "Optimal")

    def test_different_learners_not_identical(self) -> None:
        """Verify different learners within the same profile have distinct trajectories."""
        results = simulate_all_profiles(num_interactions=50, num_learners=2, seed=42)
        df = results["fast_accurate"]
        df_l1 = df[df["learner_id"] == 1].reset_index(drop=True)
        df_l2 = df[df["learner_id"] == 2].reset_index(drop=True)
        self.assertFalse(df_l1["nrt"].equals(df_l2["nrt"]))

    def test_simulation_diagnostics(self) -> None:
        """Verify validate_simulation_diagnostics returns correct summary metrics."""
        from src.simulator import validate_simulation_diagnostics
        results = simulate_all_profiles(num_interactions=100, num_learners=2, seed=42)
        diag = validate_simulation_diagnostics(results)
        self.assertIn("profile_summary", diag)
        self.assertIn("learner_summary", diag)
        prof_df = diag["profile_summary"]
        self.assertEqual(len(prof_df), 5)
        self.assertTrue((prof_df["n_learners"] == 2).all())

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
        """Verify performance-dependent transitions influence state generation."""
        df_low = simulate_learner("slow_inaccurate", num_interactions=150, seed=42)
        df_high = simulate_learner("fast_accurate", num_interactions=150, seed=42)

        overload_ratio_low = (df_low["state"] == "Overload").mean()
        overload_ratio_high = (df_high["state"] == "Overload").mean()

        self.assertGreater(overload_ratio_low, overload_ratio_high)

    def test_save_simulation(self) -> None:
        """Verify saving simulation output CSV files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            results = simulate_all_profiles(num_interactions=10, num_learners=2, seed=42)
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

            combined_df = pd.read_csv(tmp_path / "simulated_learners_all.csv")
            self.assertEqual(len(combined_df), 100)  # 5 profiles x 2 learners x 10 interactions
            self.assertListEqual(list(combined_df.columns), self.expected_columns)


if __name__ == "__main__":
    unittest.main()
