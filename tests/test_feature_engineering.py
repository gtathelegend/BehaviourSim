"""Unit tests for CLSI-Adapt feature engineering module.

Tests cover:
  1.  Correct feature dimensions.
  2.  Correct column ordering.
  3.  First interaction handling.
  4.  Window boundary behaviour.
  5.  Learner boundary behaviour (no cross-profile leakage).
  6.  Correct overload target alignment (hand-crafted data).
  7.  Correct underload detection (hand-crafted data).
  8.  No future values in X (temporal leakage detection).
  9.  End-of-sequence target handling (NaN behaviour).
  10. Deterministic output across identical calls.
"""

import unittest
import numpy as np
import pandas as pd

from src.feature_engineering import (
    FEATURE_COLUMNS,
    build_features,
    build_overload_target,
    build_underload_target,
    prepare_dataset,
    extract_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n: int,
    profile: str = "test",
    accuracy: list | None = None,
    nrt: list | None = None,
) -> pd.DataFrame:
    """Create a minimal synthetic interaction DataFrame for testing."""
    rng = np.random.default_rng(0)
    acc = accuracy if accuracy is not None else rng.integers(0, 2, size=n).tolist()
    nrt_vals = nrt if nrt is not None else rng.uniform(0.5, 2.0, size=n).tolist()
    return pd.DataFrame(
        {
            "profile": profile,
            "interaction_id": np.arange(1, n + 1, dtype=int),
            "state": "Optimal",
            "difficulty": 3,
            "accuracy": acc,
            "nrt": nrt_vals,
            "window_error_rate": [1.0 - a for a in acc],
            "retries": 0,
            "help_requested": 0,
            "confidence": 3,
            "streak_correct": 0,
            "streak_incorrect": 0,
            "nrt_variance": 0.0,
            "session_time": np.cumsum(nrt_vals).tolist(),
        }
    )


class TestBuildFeatures(unittest.TestCase):
    """Tests for build_features()."""

    def test_feature_dimensions(self) -> None:
        """X must have shape (n, len(FEATURE_COLUMNS) + 1) for the extra rolling-mean-NRT column."""
        df = _make_df(20)
        X, idx = build_features(df, window_size=10)
        expected_cols = len(FEATURE_COLUMNS) + 1  # +1 for rolling_mean_nrt
        self.assertEqual(X.shape, (20, expected_cols))

    def test_feature_column_ordering(self) -> None:
        """First n columns of X must correspond to FEATURE_COLUMNS in order."""
        df = _make_df(10)
        X, _ = build_features(df, window_size=10)
        # Check each base column aligns correctly
        for col_idx, col_name in enumerate(FEATURE_COLUMNS):
            expected = df[col_name].to_numpy(dtype=float)
            np.testing.assert_array_almost_equal(X[:, col_idx], expected)

    def test_first_interaction_no_nan(self) -> None:
        """The very first row of X must not contain NaN (min_periods=1 ensures this)."""
        df = _make_df(1)
        X, _ = build_features(df, window_size=10)
        self.assertFalse(np.isnan(X).any())

    def test_no_nans_in_full_sequence(self) -> None:
        """No NaN values should appear anywhere in X for any standard sequence."""
        df = _make_df(50)
        X, _ = build_features(df, window_size=10)
        self.assertFalse(np.isnan(X).any())

    def test_window_boundary_rolling_mean(self) -> None:
        """Rolling mean NRT should equal the simple mean of available history."""
        n = 15
        nrt_vals = [float(i + 1) for i in range(n)]
        df = _make_df(n, nrt=nrt_vals)
        X, _ = build_features(df, window_size=5)
        rolling_col = X[:, -1]  # last column is rolling_mean_nrt

        # Manually compute expected rolling mean with window=5, min_periods=1
        expected = (
            pd.Series(nrt_vals, dtype=float)
            .rolling(5, min_periods=1)
            .mean()
            .to_numpy()
        )
        np.testing.assert_array_almost_equal(rolling_col, expected)

    def test_index_aligned_to_df(self) -> None:
        """Returned index must have the same length as X rows."""
        df = _make_df(12)
        X, idx = build_features(df, window_size=5)
        self.assertEqual(len(idx), X.shape[0])


class TestTemporalLeakage(unittest.TestCase):
    """Tests specifically designed to detect temporal leakage in feature construction."""

    def test_feature_at_t0_uses_only_t0(self) -> None:
        """Features at t=0 must only reflect t=0 values; no look-ahead into t=1..n."""
        # Build two DFs that differ only from t=1 onwards
        n = 10
        df_a = _make_df(n, accuracy=[1] * n, nrt=[1.0] * n)
        df_b = df_a.copy()
        # Corrupt all rows from t=1 onward with extreme values
        df_b.loc[1:, "nrt"] = 999.0
        df_b.loc[1:, "accuracy"] = 0

        X_a, _ = build_features(df_a, window_size=10)
        X_b, _ = build_features(df_b, window_size=10)

        # Row 0 must be identical in both (future modifications must not affect X[0])
        np.testing.assert_array_equal(X_a[0], X_b[0])

    def test_feature_at_t_does_not_contain_future_nrt(self) -> None:
        """Modifying NRT at t+k (k>0) must not change X[t]."""
        n = 20
        df_base = _make_df(n, nrt=[1.0] * n)
        df_modified = df_base.copy()
        df_modified.loc[10:, "nrt"] = 50.0  # change future rows

        X_base, _ = build_features(df_base, window_size=5)
        X_mod, _ = build_features(df_modified, window_size=5)

        # Rows 0..9 must be identical
        np.testing.assert_array_equal(X_base[:10], X_mod[:10])

    def test_feature_at_t_does_not_contain_future_accuracy(self) -> None:
        """Modifying accuracy at t+k must not change X[t]."""
        n = 20
        df_base = _make_df(n, accuracy=[1] * n)
        df_modified = df_base.copy()
        df_modified.loc[10:, "accuracy"] = 0
        # Also update window_error_rate to be consistent
        df_modified.loc[10:, "window_error_rate"] = 1.0

        X_base, _ = build_features(df_base, window_size=5)
        X_mod, _ = build_features(df_modified, window_size=5)

        np.testing.assert_array_equal(X_base[:10], X_mod[:10])


class TestOverloadTarget(unittest.TestCase):
    """Tests for build_overload_target()."""

    def test_overload_target_hand_crafted(self) -> None:
        """Manually verify overload label on a hand-crafted sequence.

        Sequence (20 items):
          - t=0..9:  accuracy=1  (high prior accuracy)
          - t=10..19: accuracy=0 (collapse)

        Expected overload=1 at t=9:
          prior [t-3..t] = [9,9,9,9] → all 1 → mean=1.0 >= 0.75 ✓
          future [t+1..t+3] = [0,0,0] → mean=0.0 <= 0.5 ✓

        Expected overload=0 at t=5 (prior mean=1.0 but future mean=1.0 > 0.5):
          prior [2..5] = [1,1,1,1] → 1.0 >= 0.75 ✓
          future [6..8] = [1,1,1] → 1.0 > 0.5 ✗  → 0
        """
        n = 20
        acc = [1] * 10 + [0] * 10
        df = _make_df(n, accuracy=acc)

        y = build_overload_target(df)

        # t=9 should be overload=1
        self.assertEqual(y.iloc[9], 1.0, f"Expected overload=1 at t=9, got {y.iloc[9]}")

        # t=5 should be overload=0 (future items 6,7,8 are all 1 → future_acc=1.0 > 0.5)
        self.assertEqual(y.iloc[5], 0.0, f"Expected overload=0 at t=5, got {y.iloc[5]}")

        # t=0..2 should be 0 (insufficient prior history)
        for t in range(3):
            self.assertEqual(y.iloc[t], 0.0, f"Expected 0 at t={t} (insufficient prior), got {y.iloc[t]}")

    def test_end_of_sequence_is_nan(self) -> None:
        """Last 3 interactions must have NaN label (insufficient future observations)."""
        n = 15
        df = _make_df(n)
        y = build_overload_target(df, future_window=3)
        # Rows n-3 .. n-1 (last 3) must be NaN because t+3 >= n
        for t in range(n - 3, n):
            self.assertTrue(
                np.isnan(y.iloc[t]),
                f"Expected NaN at t={t}, got {y.iloc[t]}",
            )

    def test_insufficient_prior_history_labeled_zero(self) -> None:
        """Rows 0..(prior_window-2) must receive label 0 (not NaN)."""
        df = _make_df(20)
        y = build_overload_target(df, prior_window=4)
        for t in range(3):  # t=0,1,2 have insufficient prior
            self.assertEqual(y.iloc[t], 0.0)

    def test_overload_is_binary_or_nan(self) -> None:
        """Overload labels must be 0, 1, or NaN only."""
        df = _make_df(30)
        y = build_overload_target(df)
        non_nan = y.dropna()
        self.assertTrue(
            set(non_nan.unique()).issubset({0.0, 1.0}),
            f"Unexpected values: {set(non_nan.unique())}",
        )

    def test_overload_target_correct_alignment(self) -> None:
        """Overload label at t must depend on future [t+1..t+3], not just history."""
        # Craft a sequence where only t=8 triggers overload
        # prior [5..8] all=1, future [9,10,11] all=0
        acc = [1] * 9 + [0] * 11  # length 20
        df = _make_df(20, accuracy=acc)
        y = build_overload_target(df)
        self.assertEqual(y.iloc[8], 1.0)
        # t=7: future [8,9,10] = [1,0,0] → mean=0.33 <=0.5 ✓ but prior [4..7] all=1 ✓ → could be 1
        # t=6: future [7,8,9] = [1,1,0] → mean=0.67 > 0.5 → 0
        self.assertEqual(y.iloc[6], 0.0)


class TestUnderloadTarget(unittest.TestCase):
    """Tests for build_underload_target()."""

    def test_underload_hand_crafted(self) -> None:
        """Manually verify underload detection on a hand-crafted sequence.

        Sequence: 10 items, first 4 have accuracy=1 and NRT=0.1 (low), rest vary.
        Expected: t=3 should be underload=1 (positions 0..3 all correct, mean NRT=0.1 < 0.3).
        """
        n = 10
        acc = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
        nrt = [0.1, 0.1, 0.1, 0.1, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
        df = _make_df(n, accuracy=acc, nrt=nrt)

        y = build_underload_target(df, window=4, nrt_threshold=0.3)

        self.assertEqual(y.iloc[3], 1, "Expected underload=1 at t=3")
        # t=4: accuracy[1..4] = [1,1,1,0] → not all 1 → 0
        self.assertEqual(y.iloc[4], 0, "Expected underload=0 at t=4 (accuracy=0 included)")

    def test_underload_insufficient_history_is_zero(self) -> None:
        """Rows 0..(window-2) must receive underload=0."""
        df = _make_df(10, accuracy=[1] * 10, nrt=[0.1] * 10)
        y = build_underload_target(df, window=4, nrt_threshold=0.3)
        for t in range(3):
            self.assertEqual(y.iloc[t], 0, f"Expected 0 at t={t}")

    def test_underload_requires_all_correct(self) -> None:
        """A single incorrect item in the window must prevent underload labeling."""
        acc = [1, 1, 0, 1] + [1] * 6  # t=3: window [0..3] has accuracy=0 at t=2
        nrt = [0.1] * 10
        df = _make_df(10, accuracy=acc, nrt=nrt)
        y = build_underload_target(df, window=4, nrt_threshold=0.3)
        self.assertEqual(y.iloc[3], 0)

    def test_underload_requires_low_nrt(self) -> None:
        """High NRT (above threshold) must prevent underload even with perfect accuracy."""
        df = _make_df(10, accuracy=[1] * 10, nrt=[2.0] * 10)
        y = build_underload_target(df, window=4, nrt_threshold=0.3)
        self.assertTrue((y == 0).all(), "Expected all 0 (NRT too high for underload)")

    def test_underload_is_binary(self) -> None:
        """Underload series must only contain 0 and 1."""
        df = _make_df(30)
        y = build_underload_target(df)
        self.assertTrue(set(y.unique()).issubset({0, 1}))


class TestLearnerBoundaryBehavior(unittest.TestCase):
    """Tests that window history never crosses profile boundaries."""

    def test_profile_boundary_resets_rolling_mean(self) -> None:
        """Rolling mean NRT at the first interaction of profile B must not reflect profile A's NRT."""
        df_a = _make_df(10, profile="A", nrt=[100.0] * 10)
        df_b = _make_df(5, profile="B", nrt=[1.0] * 5)
        combined = pd.concat([df_a, df_b], ignore_index=True)

        # Process individually (as prepare_dataset does)
        X_b_alone, _ = build_features(df_b, window_size=10)
        X_combined, _, _ = prepare_dataset(combined, window_size=10, drop_nan_targets=False)

        # Rows corresponding to profile B in combined (rows 10..14)
        X_b_in_combined = X_combined[10:15]

        # Rolling mean NRT at the first B row should NOT be contaminated by A's 100.0
        rolling_col_idx = -1  # last column
        self.assertAlmostEqual(
            X_b_in_combined[0, rolling_col_idx],
            1.0,
            places=5,
            msg="Profile B's first row rolling-mean-NRT must not be contaminated by profile A",
        )

    def test_prepare_dataset_multi_profile_row_count(self) -> None:
        """prepare_dataset on two profiles must return n_A + n_B rows (minus NaN drops)."""
        df_a = _make_df(20, profile="A")
        df_b = _make_df(20, profile="B")
        combined = pd.concat([df_a, df_b], ignore_index=True)

        X, y, meta = prepare_dataset(combined, window_size=5, drop_nan_targets=True)
        # Each profile drops last 3 rows (NaN targets with future_window=3)
        expected_rows = (20 - 3) * 2
        self.assertEqual(X.shape[0], expected_rows)
        self.assertEqual(len(y), expected_rows)


class TestPrepareDataset(unittest.TestCase):
    """Tests for prepare_dataset() and extract_features()."""

    def test_output_shapes_aligned(self) -> None:
        """X, y, meta must all have the same number of rows."""
        df = _make_df(30)
        X, y, meta = prepare_dataset(df, window_size=10)
        self.assertEqual(X.shape[0], len(y))
        self.assertEqual(X.shape[0], len(meta))

    def test_deterministic_output(self) -> None:
        """Calling prepare_dataset twice with identical input returns identical X and y."""
        df = _make_df(25)
        X1, y1, _ = prepare_dataset(df, window_size=5)
        X2, y2, _ = prepare_dataset(df, window_size=5)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_no_nan_in_X(self) -> None:
        """Feature matrix X must never contain NaN values."""
        df = _make_df(40)
        X, _, _ = prepare_dataset(df, window_size=10)
        self.assertFalse(np.isnan(X).any())

    def test_y_binary_after_drop(self) -> None:
        """After dropping NaN targets, y must only contain 0.0 and 1.0."""
        df = _make_df(30)
        _, y, _ = prepare_dataset(df, window_size=5, drop_nan_targets=True)
        self.assertTrue(set(y).issubset({0.0, 1.0}))

    def test_meta_contains_required_columns(self) -> None:
        """meta DataFrame must contain profile, interaction_id, overload, underload."""
        df = _make_df(20)
        _, _, meta = prepare_dataset(df)
        for col in ("profile", "interaction_id", "overload", "underload"):
            self.assertIn(col, meta.columns)

    def test_extract_features_dict_input(self) -> None:
        """extract_features must accept a dict of {profile: DataFrame}."""
        dfs = {
            "A": _make_df(15, profile="A"),
            "B": _make_df(15, profile="B"),
        }
        X, y, meta = extract_features(dfs, window_size=5)
        self.assertEqual(X.shape[0], len(y))
        self.assertFalse(np.isnan(X).any())

    def test_extract_features_dataframe_input(self) -> None:
        """extract_features must accept a combined DataFrame directly."""
        df = _make_df(20)
        X, y, meta = extract_features(df, window_size=5)
        self.assertEqual(X.shape[0], len(y))

    def test_no_cross_learner_feature_or_target_window(self) -> None:
        """Verify that last interaction of learner A and first interaction of learner B never share history."""
        # Learner 1: 5 interactions with high NRT (100.0)
        df_l1 = _make_df(5, profile="prof1", accuracy=[1, 1, 1, 1, 0], nrt=[100.0] * 5)
        df_l1["learner_id"] = 1

        # Learner 2: 5 interactions with low NRT (1.0)
        df_l2 = _make_df(5, profile="prof1", accuracy=[0, 0, 0, 1, 1], nrt=[1.0] * 5)
        df_l2["learner_id"] = 2

        combined = pd.concat([df_l1, df_l2], ignore_index=True)

        X, y, meta = prepare_dataset(combined, window_size=5, drop_nan_targets=False)

        # Row 5 (index 5) is the first interaction of Learner 2.
        # Its rolling mean NRT (last column of X) must be 1.0, not contaminated by Learner 1's 100.0.
        self.assertAlmostEqual(X[5, -1], 1.0, places=5)

        # Target for Learner 1 at t=4 (index 4) has future window checked only within Learner 1 (length 5 -> missing t+3 -> NaN target)
        self.assertTrue(np.isnan(y[4]))

        # Target for Learner 2 at t=0 (index 5) has prior window checked only within Learner 2 (length 0 prior -> 0.0 target)
        self.assertEqual(y[5], 0.0)


if __name__ == "__main__":
    unittest.main()
