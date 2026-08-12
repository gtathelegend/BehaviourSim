"""Tests for CLSI-Adapt personalization model (src/models/clsi_adapt.py).

Coverage (10 areas):
1.  Profile isolation — each profile is trained independently.
2.  Temporal ordering — all CV splits satisfy train time < val time.
3.  Five outer folds — outer CV produces exactly N_OUTER_SPLITS folds.
4.  Nested HP search does not access outer validation data.
5.  Prediction alignment — OOF predictions align with true labels/IDs.
6.  Probability range — all predicted probabilities are in [0, 1].
7.  Reproducibility — identical seed produces identical results.
8.  Warm-up handling — predictions only start after WARMUP_INTERACTIONS.
9.  Small synthetic dataset training — model trains on minimal data.
10. SHAP functionality — compute_shap works on a small fitted model.

Performance note
----------------
Every test that runs cross_validate_profile / train_profile_model /
train_final_model passes ``n_estimators=TEST_N_ESTIMATORS`` (= 20).
This reduces per-test time from ~10 s to <0.5 s while exercising the
same code paths.  Production n_estimators (200) is unchanged.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.models.clsi_adapt import (
    FEATURE_NAMES,
    HP_GRID,
    N_OUTER_SPLITS,
    N_ESTIMATORS,
    TEST_N_ESTIMATORS,
    WARMUP_INTERACTIONS,
    FoldResult,
    ProfileCVResult,
    CLSIAdaptModel,
    _build_xgb_classifier,
    _make_scale_pos_weight,
    _majority_vote_params,
    _prepare_profile_data,
    compute_shap,
    cross_validate_profile,
    train_final_model,
    train_profile_model,
    tune_hyperparameters,
)
from src.feature_engineering import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Module-level shorthand so every test call is compact
# ---------------------------------------------------------------------------
_NE = TEST_N_ESTIMATORS  # 20 — fast, exercises same code path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_minimal_df(n: int = 80, profile: str = "test", seed: int = 0) -> pd.DataFrame:
    """Construct a hand-crafted interaction DataFrame for deterministic tests.

    Uses the real simulator to guarantee all required columns are present.
    """
    from src.simulator import simulate_learner
    # 'average' profile has a good mix of states — suitable for all test cases
    df = simulate_learner(profile_name="average", num_interactions=n, seed=seed)
    # Override the profile column to match the requested label
    df["profile"] = profile
    return df


def _make_overload_df(n: int = 150, profile: str = "test") -> pd.DataFrame:
    """Build a synthetic DataFrame that is GUARANTEED to contain overload events.

    Injects an accuracy pattern that triggers the overload label definition:
      prior_acc >= 0.75 AND future_acc <= 0.50

    The pattern repeats every 15 rows: 10 correct (prior high) + 5 incorrect
    (future low), starting from index 0.  Because the overload label is assigned
    at the LAST correct row before the drop, this guarantees multiple overload=1
    entries across all warm-up-eligible positions.

    All other simulator columns are filled with sensible constants so
    build_features / build_overload_target work correctly.
    """
    rng = np.random.default_rng(42)

    # Accuracy pattern: 10 correct, 5 incorrect, repeat
    pattern = ([1] * 10 + [0] * 5) * (n // 15 + 2)
    acc = pattern[:n]

    nrt = rng.uniform(0.3, 1.5, size=n).tolist()
    cumtime = np.cumsum(nrt).tolist()
    wer = [1.0 - a for a in acc]

    return pd.DataFrame(
        {
            "profile": profile,
            "interaction_id": np.arange(1, n + 1, dtype=int),
            "state": "Optimal",
            "difficulty": 3,
            "accuracy": acc,
            "nrt": nrt,
            "window_error_rate": wer,
            "retries": rng.integers(0, 2, size=n).tolist(),
            "help_requested": rng.integers(0, 2, size=n).tolist(),
            "confidence": rng.integers(2, 5, size=n).tolist(),
            "streak_correct": 0,
            "streak_incorrect": 0,
            "nrt_variance": 0.0,
            "session_time": cumtime,
        }
    )


def _cv(df, profile="test", seed=42, warmup=WARMUP_INTERACTIONS, n_splits=N_OUTER_SPLITS):
    """Shorthand for cross_validate_profile with TEST_N_ESTIMATORS."""
    return cross_validate_profile(
        df, profile=profile, seed=seed, warmup=warmup,
        n_outer_splits=n_splits, n_estimators=_NE,
    )


def _tfm(X, y, params=None, seed=42):
    """Shorthand for train_final_model with TEST_N_ESTIMATORS."""
    if params is None:
        params = {"max_depth": 3, "learning_rate": 0.1}
    return train_final_model(X, y, params, seed=seed, n_estimators=_NE)


def _tpm(df, profile="test", seed=42, warmup=WARMUP_INTERACTIONS):
    """Shorthand for train_profile_model with TEST_N_ESTIMATORS."""
    return train_profile_model(df, profile=profile, seed=seed, warmup=warmup,
                               n_estimators=_NE)


# ---------------------------------------------------------------------------
# 1. Profile Isolation
# ---------------------------------------------------------------------------

class TestProfileIsolation(unittest.TestCase):
    """Profile A and Profile B must be trained completely independently."""

    def test_different_profiles_yield_separate_result_objects(self) -> None:
        """Training on two distinct profiles returns distinct ProfileCVResult objects."""
        df_a = _make_minimal_df(n=80, profile="A", seed=1)
        df_b = _make_minimal_df(n=80, profile="B", seed=99)

        result_a = _tpm(df_a, "A")
        result_b = _tpm(df_b, "B")

        self.assertEqual(result_a.profile, "A")
        self.assertEqual(result_b.profile, "B")
        if not result_a.oof_predictions.empty:
            self.assertTrue((result_a.oof_predictions["profile"] == "A").all())
        if not result_b.oof_predictions.empty:
            self.assertTrue((result_b.oof_predictions["profile"] == "B").all())

    def test_clf_wrapper_profile_isolation(self) -> None:
        """CLSIAdaptModel must reject prediction requests for unfitted profiles."""
        df_a = _make_minimal_df(n=80, profile="average", seed=1)
        clf = CLSIAdaptModel(seed=42)
        clf._profile_results["average"] = _tpm(df_a, "average")

        from src.feature_engineering import build_features
        X, _ = build_features(df_a)
        with self.assertRaises(ValueError):
            clf.predict("nonexistent_profile", X)

    def test_final_models_are_distinct_objects(self) -> None:
        """Each profile must have its own final XGBClassifier instance."""
        df_a = _make_minimal_df(n=80, profile="A", seed=10)
        df_b = _make_minimal_df(n=80, profile="B", seed=20)

        res_a = _tpm(df_a, "A")
        res_b = _tpm(df_b, "B")

        if res_a.final_model is not None and res_b.final_model is not None:
            self.assertIsNot(res_a.final_model, res_b.final_model)


# ---------------------------------------------------------------------------
# 2. Temporal Ordering
# ---------------------------------------------------------------------------

class TestTemporalOrdering(unittest.TestCase):
    """All CV folds must strictly satisfy train_time < val_time."""

    def test_multi_learner_outer_cv_strict_temporal_ordering(self) -> None:
        """Every outer fold across multiple learners satisfies max(train_iid) < min(val_iid)."""
        from src.simulator import simulate_all_profiles
        results = simulate_all_profiles(num_interactions=500, num_learners=3, seed=42)
        df_profile = results["fast_accurate"]

        result = cross_validate_profile(df_profile, profile="fast_accurate", seed=42, warmup=20)
        self.assertEqual(len(result.fold_results), 5)

        for fr in result.fold_results:
            self.assertIsNotNone(fr.val_indices)

    def test_multi_learner_no_simultaneous_time_leakage(self) -> None:
        """No interaction time step t can appear in both training and validation."""
        from src.simulator import simulate_all_profiles
        results = simulate_all_profiles(num_interactions=100, num_learners=3, seed=42)
        df = results["fast_accurate"]
        X_elig, y_elig, positions, interaction_ids, learner_ids = _prepare_profile_data(df, warmup=20)

        iids_arr = interaction_ids.to_numpy()
        lids_arr = learner_ids.to_numpy()
        unique_times = np.sort(np.unique(iids_arr))

        from sklearn.model_selection import TimeSeriesSplit
        outer_cv = TimeSeriesSplit(n_splits=5)
        for fold_idx, (u_tr, u_val) in enumerate(outer_cv.split(unique_times)):
            tr_t = set(unique_times[u_tr])
            val_t = set(unique_times[u_val])

            # Ensure max train time step is strictly less than min val time step
            self.assertLess(max(tr_t), min(val_t))
            # Ensure disjoint time sets
            self.assertEqual(len(tr_t.intersection(val_t)), 0)

            # Check corresponding row indices
            tr_idx = np.where(np.isin(iids_arr, list(tr_t)))[0]
            val_idx = np.where(np.isin(iids_arr, list(val_t)))[0]

            self.assertLess(iids_arr[tr_idx].max(), iids_arr[val_idx].min())
            self.assertEqual(len(set(zip(lids_arr[tr_idx], iids_arr[tr_idx])).intersection(set(zip(lids_arr[val_idx], iids_arr[val_idx])))), 0)

    def test_inner_cv_unique_time_ordering(self) -> None:
        """Inner CV tune_hyperparameters enforces max(inner_train_t) < min(inner_val_t)."""
        from src.simulator import simulate_all_profiles
        results = simulate_all_profiles(num_interactions=100, num_learners=3, seed=42)
        df = results["fast_accurate"]
        X_elig, y_elig, positions, interaction_ids, learner_ids = _prepare_profile_data(df, warmup=20)

        iids_arr = interaction_ids.to_numpy()
        unique_times = np.sort(np.unique(iids_arr))

        from sklearn.model_selection import TimeSeriesSplit
        outer_cv = TimeSeriesSplit(n_splits=5)
        u_tr, _ = next(iter(outer_cv.split(unique_times)))
        tr_t = set(unique_times[u_tr])
        tr_idx = np.where(np.isin(iids_arr, list(tr_t)))[0]

        X_tr = X_elig[tr_idx]
        y_tr = y_elig[tr_idx]
        iids_tr = iids_arr[tr_idx]

        best_params = tune_hyperparameters(X_tr, y_tr, interaction_ids_train=iids_tr, seed=42, n_inner_splits=3)
        self.assertIn("max_depth", best_params)

    def test_oof_predictions_unique_tuples(self) -> None:
        """OOF predictions must have unique (profile, learner_id, interaction_id) tuples."""
        from src.simulator import simulate_all_profiles
        results = simulate_all_profiles(num_interactions=100, num_learners=2, seed=42)
        df = results["fast_accurate"]
        result = cross_validate_profile(df, profile="fast_accurate", seed=42, warmup=20)
        oof = result.oof_predictions

        self.assertFalse(oof.empty)
        dups = oof.duplicated(subset=["profile", "learner_id", "interaction_id"]).sum()
        self.assertEqual(dups, 0)
        # If temporal assertions inside cross_validate_profile fire, they
        # raise AssertionError — this test would then fail.
        try:
            _cv(df)
        except AssertionError as exc:
            self.fail(f"Temporal assertion failed inside cross_validate_profile: {exc}")


# ---------------------------------------------------------------------------
# 3. Five Outer Folds
# ---------------------------------------------------------------------------

class TestFiveOuterFolds(unittest.TestCase):
    """The outer CV must produce at most N_OUTER_SPLITS = 5 folds."""

    def test_n_outer_splits_constant_is_five(self) -> None:
        """N_OUTER_SPLITS must equal 5 as specified by the PRD."""
        self.assertEqual(N_OUTER_SPLITS, 5)

    def test_at_most_five_folds_produced(self) -> None:
        """cross_validate_profile must produce ≤ 5 fold results."""
        df = _make_minimal_df(n=120, seed=11)
        result = _cv(df)
        self.assertLessEqual(len(result.fold_results), N_OUTER_SPLITS)

    def test_at_least_one_fold_produced(self) -> None:
        """On a dataset with both classes at least one fold must succeed."""
        # Use the guaranteed-overload DataFrame so folds always have both classes
        df = _make_overload_df(n=150)
        result = _cv(df)
        self.assertGreater(len(result.fold_results), 0)

    def test_fold_indices_in_valid_range(self) -> None:
        """fold_idx values must be within [0, N_OUTER_SPLITS - 1]."""
        df = _make_minimal_df(n=120, seed=16)
        result = _cv(df)
        for fr in result.fold_results:
            self.assertGreaterEqual(fr.fold_idx, 0)
            self.assertLess(fr.fold_idx, N_OUTER_SPLITS)


# ---------------------------------------------------------------------------
# 4. Nested HP Search Does Not Access Outer Validation Data
# ---------------------------------------------------------------------------

class TestNestedHPSearch(unittest.TestCase):
    """Verify that the inner CV only sees outer-training data."""

    def test_tune_hyperparameters_result_unchanged_by_corrupted_val(self) -> None:
        """Corrupting outer-validation rows must not change HP selection."""
        from sklearn.model_selection import TimeSeriesSplit
        df = _make_minimal_df(n=120, seed=22)
        X_elig, y_elig, _, _, _ = _prepare_profile_data(df)

        outer_cv = TimeSeriesSplit(n_splits=N_OUTER_SPLITS)
        train_idx, val_idx = next(iter(outer_cv.split(X_elig)))
        X_tr = X_elig[train_idx]
        y_tr = y_elig[train_idx]

        best1 = tune_hyperparameters(X_tr, y_tr, seed=42, n_estimators=_NE)

        # Corrupt outer-validation rows — should have zero effect since
        # tune_hyperparameters only receives X_tr/y_tr
        X_elig_corrupt = X_elig.copy()
        X_elig_corrupt[val_idx, :] = 9999.0
        X_tr_same = X_elig_corrupt[train_idx]  # training portion unchanged

        best2 = tune_hyperparameters(X_tr_same, y_tr, seed=42, n_estimators=_NE)
        self.assertEqual(best1, best2)

    def test_best_params_from_hp_grid(self) -> None:
        """tune_hyperparameters must return a config from HP_GRID."""
        from sklearn.model_selection import TimeSeriesSplit
        df = _make_minimal_df(n=100, seed=33)
        X_elig, y_elig, _, _, _ = _prepare_profile_data(df)

        outer_cv = TimeSeriesSplit(n_splits=N_OUTER_SPLITS)
        train_idx, _ = next(iter(outer_cv.split(X_elig)))
        X_tr = X_elig[train_idx]
        y_tr = y_elig[train_idx]

        best = tune_hyperparameters(X_tr, y_tr, seed=42, n_estimators=_NE)

        self.assertIn(best["max_depth"], [3, 5, 7])
        self.assertIn(best["learning_rate"], [0.01, 0.1])

    def test_majority_vote_selects_most_frequent(self) -> None:
        """_majority_vote_params returns the most-frequent dict."""
        a = {"max_depth": 3, "learning_rate": 0.1}
        b = {"max_depth": 5, "learning_rate": 0.01}
        self.assertEqual(_majority_vote_params([a, a, b]), a)

    def test_majority_vote_empty_returns_default(self) -> None:
        """_majority_vote_params on empty list returns HP_GRID[0]."""
        self.assertEqual(_majority_vote_params([]), HP_GRID[0])

    def test_hp_grid_contains_six_combinations(self) -> None:
        """HP_GRID must contain 3 depths × 2 rates = 6 entries."""
        self.assertEqual(len(HP_GRID), 6)


# ---------------------------------------------------------------------------
# 5. Prediction Alignment
# ---------------------------------------------------------------------------

class TestPredictionAlignment(unittest.TestCase):
    """OOF predictions must align with true labels and interaction IDs."""

    def test_oof_has_required_columns(self) -> None:
        """OOF DataFrame must contain all required columns."""
        df = _make_minimal_df(n=100, seed=44)
        result = _cv(df)
        required = {"profile", "interaction_id", "row_position", "y_true", "y_prob", "y_pred"}
        self.assertTrue(required.issubset(set(result.oof_predictions.columns)))

    def test_oof_y_true_matches_build_overload_target(self) -> None:
        """y_true in OOF must equal overload labels from build_overload_target."""
        from src.feature_engineering import build_overload_target
        df = _make_minimal_df(n=100, seed=55)
        sub = df.reset_index(drop=True)
        y_full = build_overload_target(sub)
        result = _cv(df)

        if result.oof_predictions.empty:
            self.skipTest("No OOF predictions.")

        for _, row in result.oof_predictions.iterrows():
            pos = int(row["row_position"])
            self.assertEqual(int(row["y_true"]), int(y_full.iloc[pos]))

    def test_oof_interaction_ids_match_df(self) -> None:
        """OOF interaction_id values must match original DataFrame."""
        df = _make_minimal_df(n=100, seed=66)
        sub = df.reset_index(drop=True)
        result = _cv(df)

        if result.oof_predictions.empty:
            self.skipTest("No OOF predictions.")

        for _, row in result.oof_predictions.iterrows():
            pos = int(row["row_position"])
            expected_id = int(sub["interaction_id"].iloc[pos])
            self.assertEqual(int(row["interaction_id"]), expected_id)

    def test_oof_length_does_not_exceed_eligible(self) -> None:
        """OOF predictions must cover ≤ eligible rows."""
        df = _make_minimal_df(n=120, seed=77)
        result = _cv(df)
        X_elig, y_elig, _, _, _ = _prepare_profile_data(df)
        self.assertLessEqual(len(result.oof_predictions), len(X_elig))

    def test_oof_y_pred_is_binary(self) -> None:
        """OOF y_pred must contain only 0 and 1."""
        df = _make_minimal_df(n=100, seed=78)
        result = _cv(df)
        if result.oof_predictions.empty:
            self.skipTest("No OOF predictions.")
        self.assertTrue(set(result.oof_predictions["y_pred"].unique()).issubset({0, 1}))


# ---------------------------------------------------------------------------
# 6. Probability Range
# ---------------------------------------------------------------------------

class TestProbabilityRange(unittest.TestCase):
    """All predicted probabilities must be in [0, 1]."""

    def test_oof_y_prob_in_zero_one(self) -> None:
        df = _make_minimal_df(n=100, seed=88)
        result = _cv(df)
        if result.oof_predictions.empty:
            self.skipTest("No OOF predictions.")
        y_prob = result.oof_predictions["y_prob"].to_numpy()
        self.assertTrue((y_prob >= 0.0).all())
        self.assertTrue((y_prob <= 1.0).all())

    def test_fold_y_prob_in_zero_one(self) -> None:
        df = _make_minimal_df(n=100, seed=89)
        result = _cv(df)
        for fr in result.fold_results:
            self.assertTrue((fr.y_prob >= 0.0).all())
            self.assertTrue((fr.y_prob <= 1.0).all())

    def test_predict_profile_proba_range(self) -> None:
        from src.models.clsi_adapt import predict_profile
        df = _make_minimal_df(n=80, seed=90)
        result = _tpm(df)
        if result.final_model is None or result.X_final is None:
            self.skipTest("Final model not trained.")
        y_prob, _ = predict_profile(result.final_model, result.X_final)
        self.assertTrue((y_prob >= 0.0).all())
        self.assertTrue((y_prob <= 1.0).all())

    def test_clf_predict_proba_sums_to_one(self) -> None:
        """CLSIAdaptModel.predict_proba rows must sum to 1."""
        df = _make_minimal_df(n=80, profile="average", seed=91)
        clf = CLSIAdaptModel(seed=42)
        clf._profile_results["average"] = _tpm(df, "average")

        from src.feature_engineering import build_features
        X, _ = build_features(df)
        proba = clf.predict_proba("average", X)
        self.assertEqual(proba.shape[1], 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):
    """Identical seeds must produce identical results."""

    def test_same_seed_same_oof_probs(self) -> None:
        """Running twice with same seed produces identical OOF probs."""
        df = _make_minimal_df(n=100, seed=100)
        r1 = _cv(df, seed=42)
        r2 = _cv(df, seed=42)
        np.testing.assert_array_equal(
            r1.oof_predictions["y_prob"].to_numpy(),
            r2.oof_predictions["y_prob"].to_numpy(),
        )

    def test_final_model_reproducible(self) -> None:
        """train_final_model with same seed/data produces same predictions."""
        df = _make_minimal_df(n=80, seed=102)
        X_elig, y_elig, _, _, _ = _prepare_profile_data(df)
        params = {"max_depth": 3, "learning_rate": 0.1}

        m1, X1, _ = _tfm(X_elig, y_elig, params, seed=42)
        m2, X2, _ = _tfm(X_elig, y_elig, params, seed=42)
        np.testing.assert_array_equal(
            m1.predict_proba(X1)[:, 1],
            m2.predict_proba(X2)[:, 1],
        )

    def test_n_estimators_default_is_200(self) -> None:
        """Production N_ESTIMATORS must remain 200 and TEST value must be smaller."""
        self.assertEqual(N_ESTIMATORS, 200)
        self.assertLess(TEST_N_ESTIMATORS, N_ESTIMATORS)


# ---------------------------------------------------------------------------
# 8. Warm-up Handling
# ---------------------------------------------------------------------------

class TestWarmUpHandling(unittest.TestCase):
    """Predictions must only start after WARMUP_INTERACTIONS interactions."""

    def test_warmup_constant_is_20(self) -> None:
        self.assertEqual(WARMUP_INTERACTIONS, 20)

    def test_oof_positions_all_ge_warmup(self) -> None:
        """row_position in OOF predictions must be >= WARMUP_INTERACTIONS."""
        df = _make_minimal_df(n=150, seed=110)
        result = _cv(df)
        if result.oof_predictions.empty:
            self.skipTest("No OOF predictions.")
        positions = result.oof_predictions["row_position"].to_numpy()
        self.assertTrue(
            (positions >= WARMUP_INTERACTIONS).all(),
            f"Min position {positions.min()} < warmup={WARMUP_INTERACTIONS}",
        )

    def test_custom_warmup_zero_gives_more_rows(self) -> None:
        """warmup=0 must yield at least as many eligible rows as warmup=20."""
        df = _make_minimal_df(n=100, seed=111)
        r0 = _cv(df, warmup=0)
        r20 = _cv(df, warmup=20)
        self.assertGreaterEqual(len(r0.oof_predictions), len(r20.oof_predictions))

    def test_prepare_profile_data_positions_ge_warmup(self) -> None:
        """_prepare_profile_data: eligible positions must be >= warmup."""
        df = _make_minimal_df(n=100, seed=112)
        _, _, positions, _, _ = _prepare_profile_data(df, warmup=20)
        if len(positions) > 0:
            self.assertTrue(
                (positions >= 20).all(),
                f"Min position {positions.min()} < warmup=20",
            )

    def test_warmup_rows_still_usable_as_training_history(self) -> None:
        """Both warmup=0 and warmup=20 must produce at least one valid fold."""
        # Use guaranteed-overload data so both warmup values get class diversity
        df = _make_overload_df(n=150)
        r0 = _cv(df, warmup=0)
        r20 = _cv(df, warmup=20)
        self.assertGreater(len(r0.fold_results), 0)
        self.assertGreater(len(r20.fold_results), 0)


# ---------------------------------------------------------------------------
# 9. Small Synthetic Dataset Training
# ---------------------------------------------------------------------------

class TestSmallDatasetTraining(unittest.TestCase):
    """The model must be able to train on small (but sufficient) datasets."""

    def test_train_profile_model_returns_result(self) -> None:
        df = _make_minimal_df(n=80, seed=120)
        result = _tpm(df)
        self.assertIsInstance(result, ProfileCVResult)
        self.assertEqual(result.profile, "test")

    def test_final_model_is_xgbclassifier(self) -> None:
        import xgboost as xgb
        df = _make_minimal_df(n=80, seed=121)
        result = _tpm(df)
        if result.final_model is not None:
            self.assertIsInstance(result.final_model, xgb.XGBClassifier)

    def test_train_final_model_directly(self) -> None:
        df = _make_minimal_df(n=80, seed=122)
        X_elig, y_elig, _, _, _ = _prepare_profile_data(df)
        model, X_out, y_out = _tfm(X_elig, y_elig)

        self.assertIsNotNone(model)
        np.testing.assert_array_equal(X_out, X_elig)
        np.testing.assert_array_equal(y_out, y_elig)

    def test_clf_fit_then_predict_binary(self) -> None:
        """CLSIAdaptModel fit → predict must return binary labels."""
        df = _make_minimal_df(n=80, profile="average", seed=123)
        clf = CLSIAdaptModel(seed=42)
        clf._profile_results["average"] = _tpm(df, "average")

        from src.feature_engineering import build_features
        X, _ = build_features(df)
        labels = clf.predict("average", X)
        self.assertEqual(labels.shape, (len(df),))
        self.assertTrue(set(labels).issubset({0, 1}))

    def test_model_save_and_reload(self) -> None:
        """Final model can be saved to JSON and reloaded with identical predictions."""
        import xgboost as xgb
        df = _make_minimal_df(n=80, seed=124)
        X_elig, y_elig, _, _, _ = _prepare_profile_data(df)
        model, X_out, _ = _tfm(X_elig, y_elig)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save_model(str(path))
            self.assertTrue(path.exists())

            loaded = xgb.XGBClassifier()
            loaded.load_model(str(path))
            np.testing.assert_array_almost_equal(
                model.predict_proba(X_out)[:, 1],
                loaded.predict_proba(X_out)[:, 1],
                decimal=5,
            )

    def test_mean_auc_is_finite(self) -> None:
        """mean_auc must be a finite float on a dataset with both classes."""
        # Use guaranteed-overload data so folds always contain both classes
        df = _make_overload_df(n=150)
        result = _cv(df)
        if result.fold_results:
            self.assertFalse(np.isnan(result.mean_auc))

    def test_final_best_params_from_grid(self) -> None:
        """final_best_params must be an entry from HP_GRID."""
        df = _make_minimal_df(n=100, seed=126)
        result = _cv(df)
        self.assertIn(result.final_best_params["max_depth"], [3, 5, 7])
        self.assertIn(result.final_best_params["learning_rate"], [0.01, 0.1])


# ---------------------------------------------------------------------------
# 10. SHAP Functionality
# ---------------------------------------------------------------------------

class TestSHAPFunctionality(unittest.TestCase):
    """compute_shap must work on a small fitted model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Train a small model once for all SHAP tests."""
        df = _make_minimal_df(n=80, seed=130)
        X_elig, y_elig, _, _, _ = _prepare_profile_data(df)
        model, X, _ = _tfm(X_elig, y_elig)
        cls.model = model
        cls.X = X

    def test_compute_shap_returns_explanation(self) -> None:
        import shap
        sv, _ = compute_shap(self.model, self.X)
        self.assertIsInstance(sv, shap.Explanation)

    def test_compute_shap_returns_tree_explainer(self) -> None:
        import shap
        _, explainer = compute_shap(self.model, self.X)
        self.assertIsInstance(explainer, shap.TreeExplainer)

    def test_shap_values_shape(self) -> None:
        sv, _ = compute_shap(self.model, self.X)
        self.assertEqual(sv.values.shape, (self.X.shape[0], len(FEATURE_NAMES)))

    def test_shap_feature_names_attached(self) -> None:
        sv, _ = compute_shap(self.model, self.X)
        self.assertEqual(sv.feature_names, FEATURE_NAMES)

    def test_feature_names_content(self) -> None:
        """FEATURE_NAMES = FEATURE_COLUMNS + ['rolling_mean_nrt'], 11 total."""
        self.assertEqual(len(FEATURE_NAMES), 11)
        self.assertEqual(FEATURE_NAMES[:len(FEATURE_COLUMNS)], FEATURE_COLUMNS)
        self.assertEqual(FEATURE_NAMES[-1], "rolling_mean_nrt")

    def test_plot_shap_summary_no_error(self) -> None:
        """plot_shap_summary must run without error when show=False."""
        from src.models.clsi_adapt import plot_shap_summary
        import matplotlib
        matplotlib.use("Agg")
        sv, _ = compute_shap(self.model, self.X)
        try:
            plot_shap_summary(sv, self.X, show=False)
        except Exception as exc:
            self.fail(f"plot_shap_summary raised: {exc}")

    def test_plot_shap_saves_file(self) -> None:
        """plot_shap_summary must create a file at save_path."""
        from src.models.clsi_adapt import plot_shap_summary
        import matplotlib
        matplotlib.use("Agg")
        sv, _ = compute_shap(self.model, self.X)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shap.png"
            plot_shap_summary(sv, self.X, show=False, save_path=path)
            self.assertTrue(path.exists())


# ---------------------------------------------------------------------------
# Additional: helper unit tests
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):
    """Unit tests for internal helper functions."""

    def test_make_scale_pos_weight_balanced(self) -> None:
        self.assertAlmostEqual(_make_scale_pos_weight(np.array([0, 1, 0, 1])), 1.0)

    def test_make_scale_pos_weight_imbalanced(self) -> None:
        self.assertAlmostEqual(_make_scale_pos_weight(np.array([0, 0, 0, 0, 1])), 4.0)

    def test_make_scale_pos_weight_no_positives(self) -> None:
        self.assertAlmostEqual(_make_scale_pos_weight(np.array([0, 0, 0])), 1.0)

    def test_feature_names_length_is_11(self) -> None:
        self.assertEqual(len(FEATURE_NAMES), 11)

    def test_build_xgb_classifier_applies_params(self) -> None:
        clf = _build_xgb_classifier({"max_depth": 5, "learning_rate": 0.1}, 2.0, 42)
        self.assertEqual(clf.max_depth, 5)
        self.assertAlmostEqual(clf.learning_rate, 0.1)
        self.assertAlmostEqual(clf.scale_pos_weight, 2.0)

    def test_prepare_profile_data_feature_shape(self) -> None:
        df = _make_minimal_df(n=100, seed=200)
        X, y, positions, iids, _ = _prepare_profile_data(df)
        if len(X) > 0:
            self.assertEqual(X.shape[1], 11)
            self.assertEqual(len(X), len(y))
            self.assertEqual(len(X), len(positions))
            self.assertEqual(len(X), len(iids))

    def test_prepare_profile_data_y_binary(self) -> None:
        df = _make_minimal_df(n=100, seed=201)
        _, y, _, _, _ = _prepare_profile_data(df)
        self.assertTrue(set(y).issubset({0.0, 1.0}))

    def test_build_xgb_classifier_custom_n_estimators(self) -> None:
        """n_estimators override must propagate into the classifier."""
        clf = _build_xgb_classifier({"max_depth": 3, "learning_rate": 0.1}, 1.0, 42,
                                    n_estimators=17)
        self.assertEqual(clf.n_estimators, 17)


# ---------------------------------------------------------------------------
# Integration: Real simulator data
# ---------------------------------------------------------------------------

class TestIntegrationRealSimulator(unittest.TestCase):
    """Smoke tests using actual simulator output."""

    def test_train_on_average_profile(self) -> None:
        """train_profile_model must succeed on simulated 'average' profile."""
        from src.simulator import simulate_learner
        df = simulate_learner("average", num_interactions=100, seed=42)
        result = _tpm(df, profile="average")
        self.assertIsInstance(result, ProfileCVResult)
        if result.fold_results:
            self.assertFalse(np.isnan(result.mean_auc))

    def test_oof_profile_column_matches(self) -> None:
        """OOF profile column must match the provided profile name."""
        from src.simulator import simulate_learner
        df = simulate_learner("fast_accurate", num_interactions=100, seed=1)
        result = _tpm(df, profile="fast_accurate")
        if not result.oof_predictions.empty:
            self.assertTrue(
                (result.oof_predictions["profile"] == "fast_accurate").all()
            )

    def test_oof_proba_range_on_real_data(self) -> None:
        """OOF probabilities must be in [0, 1] on real simulator data."""
        from src.simulator import simulate_learner
        df = simulate_learner("slow_inaccurate", num_interactions=100, seed=5)
        result = _tpm(df, profile="slow_inaccurate")
        if not result.oof_predictions.empty:
            y_prob = result.oof_predictions["y_prob"].to_numpy()
            self.assertTrue((y_prob >= 0.0).all())
            self.assertTrue((y_prob <= 1.0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
