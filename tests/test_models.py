"""Tests for Rule-Based CLSI and BKT baseline models.

Coverage:
  Rule-Based CLSI
  ---------------
  1.  Exact formula values.
  2.  Help penalty application and clipping.
  3.  Retry normalization (normal, zero, above max).
  4.  Overload threshold behaviour.
  5.  Score range [0, 1].
  6.  predict_proba shape and column semantics.
  7.  fit() is a no-op / returns self.

  BKT
  ---
  8.  Initial mastery equals DEFAULT_INITIAL_MASTERY.
  9.  Correct-response update (mastery should increase or stay >= prior).
  10. Incorrect-response update (mastery should decrease or stay <= prior).
  11. Repeated correct responses monotonically increase mastery.
  12. Repeated incorrect responses keep mastery low / decrease it.
  13. Struggle threshold binary label.
  14. predict_proba shape and column semantics.
  15. fit() is a no-op / returns self.
  16. Multi-profile DataFrames are processed independently.

  Temporal behaviour
  ------------------
  17. Rule-CLSI: prediction at t depends only on columns available at t;
      changing future rows does not affect predictions at row t.
  18. BKT: mastery at position t is based only on responses 0..t-1;
          changing future responses does not alter mastery at earlier positions.
"""

from __future__ import annotations

import unittest
from typing import List

import numpy as np
import pandas as pd

from src.models.rule_based_clsi import (
    RuleBasedCLSIModel,
    HELP_PENALTY,
    MAX_RETRIES,
    OVERLOAD_THRESHOLD,
    _W_ACCURACY,
    _W_NRT,
    _W_ERROR_RATE,
    _W_RETRIES,
    _WEIGHT_SUM,
    _compute_raw_score,
    _normalize_retries,
)
from src.models.bkt import (
    BKTModel,
    DEFAULT_INITIAL_MASTERY,
    DEFAULT_LEARN,
    DEFAULT_GUESS,
    DEFAULT_SLIP,
    STRUGGLE_THRESHOLD,
    _bkt_posterior,
    _bkt_learn_update,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_interaction_df(
    n: int = 10,
    accuracy: list | None = None,
    nrt: list | None = None,
    window_error_rate: list | None = None,
    retries: list | None = None,
    help_requested: list | None = None,
    profile: str = "test",
) -> pd.DataFrame:
    """Create a minimal interaction DataFrame for testing."""
    acc = accuracy if accuracy is not None else [1] * n
    nrt_ = nrt if nrt is not None else [0.5] * n
    wer = window_error_rate if window_error_rate is not None else [1.0 - a for a in acc]
    ret = retries if retries is not None else [0] * n
    hlp = help_requested if help_requested is not None else [0] * n

    return pd.DataFrame(
        {
            "profile": profile,
            "interaction_id": np.arange(1, n + 1, dtype=int),
            "accuracy": acc,
            "nrt": nrt_,
            "window_error_rate": wer,
            "retries": ret,
            "help_requested": hlp,
        }
    )


# ===========================================================================
# Rule-Based CLSI Tests
# ===========================================================================


class TestRuleBasedCLSIFormula(unittest.TestCase):
    """Test the exact CLSI formula produces expected values."""

    def setUp(self) -> None:
        self.model = RuleBasedCLSIModel()

    def test_exact_formula_all_optimal(self) -> None:
        """Perfect learner: accuracy=1, NRT=0, error_rate=0, retries=0, no help.

        Expected: score = (0.50*1 + 0.25*1 + 0.15*1 + 0.10*1) / 1.0 = 1.0
        """
        df = _make_interaction_df(
            n=1,
            accuracy=[1],
            nrt=[0.0],
            window_error_rate=[0.0],
            retries=[0],
            help_requested=[0],
        )
        scores = self.model.predict_score(df)
        self.assertAlmostEqual(scores[0], 1.0, places=9)

    def test_exact_formula_all_worst(self) -> None:
        """Worst learner: accuracy=0, NRT=1, error_rate=1, retries=MAX_RETRIES, no help.

        Expected: score = (0.50*0 + 0.25*0 + 0.15*0 + 0.10*0) / 1.0 = 0.0
        """
        df = _make_interaction_df(
            n=1,
            accuracy=[0],
            nrt=[1.0],
            window_error_rate=[1.0],
            retries=[MAX_RETRIES],
            help_requested=[0],
        )
        scores = self.model.predict_score(df)
        self.assertAlmostEqual(scores[0], 0.0, places=9)

    def test_exact_formula_midpoint(self) -> None:
        """Mid-point: accuracy=0.5 equivalent via single correct of 0 or 1 is binary;
        test a known configuration analytically.

        accuracy=1, NRT=0.5, window_error_rate=0.5, retries=0, no help.

        Expected raw = (0.50*1 + 0.25*(1-0.5) + 0.15*(1-0.5) + 0.10*1) / 1.0
                     = (0.50 + 0.125 + 0.075 + 0.10) = 0.80
        """
        df = _make_interaction_df(
            n=1,
            accuracy=[1],
            nrt=[0.5],
            window_error_rate=[0.5],
            retries=[0],
            help_requested=[0],
        )
        expected = (
            _W_ACCURACY * 1.0
            + _W_NRT * (1.0 - 0.5)
            + _W_ERROR_RATE * (1.0 - 0.5)
            + _W_RETRIES * 1.0
        ) / _WEIGHT_SUM
        scores = self.model.predict_score(df)
        self.assertAlmostEqual(scores[0], expected, places=9)

    def test_formula_with_retries_normalised(self) -> None:
        """Verify retry normalization is applied inside the formula.

        retries=1 → retries_norm = 1/MAX_RETRIES.
        """
        df = _make_interaction_df(
            n=1,
            accuracy=[1],
            nrt=[0.0],
            window_error_rate=[0.0],
            retries=[1],
            help_requested=[0],
        )
        ret_norm = 1.0 / MAX_RETRIES
        expected = (
            _W_ACCURACY * 1.0
            + _W_NRT * 1.0
            + _W_ERROR_RATE * 1.0
            + _W_RETRIES * (1.0 - ret_norm)
        ) / _WEIGHT_SUM
        scores = self.model.predict_score(df)
        self.assertAlmostEqual(scores[0], expected, places=9)

    def test_formula_matches_manual_calculation(self) -> None:
        """Apply formula manually for a row and compare with model output."""
        accuracy = 0
        nrt = 0.8
        wer = 0.6
        retries = 2

        ret_norm = min(retries / MAX_RETRIES, 1.0)
        nrt_clipped = min(nrt, 1.0)
        expected_raw = (
            _W_ACCURACY * accuracy
            + _W_NRT * (1.0 - nrt_clipped)
            + _W_ERROR_RATE * (1.0 - wer)
            + _W_RETRIES * (1.0 - ret_norm)
        ) / _WEIGHT_SUM

        df = _make_interaction_df(
            n=1,
            accuracy=[accuracy],
            nrt=[nrt],
            window_error_rate=[wer],
            retries=[retries],
            help_requested=[0],
        )
        scores = self.model.predict_score(df)
        self.assertAlmostEqual(scores[0], expected_raw, places=9)


class TestRuleBasedCLSIHelpPenalty(unittest.TestCase):
    """Test help penalty application."""

    def setUp(self) -> None:
        self.model = RuleBasedCLSIModel()

    def test_help_penalty_reduces_score(self) -> None:
        """Score with help_requested=1 must be HELP_PENALTY less than without."""
        df_no_help = _make_interaction_df(n=1, accuracy=[1], nrt=[0.5],
                                          window_error_rate=[0.0], retries=[0],
                                          help_requested=[0])
        df_help = _make_interaction_df(n=1, accuracy=[1], nrt=[0.5],
                                       window_error_rate=[0.0], retries=[0],
                                       help_requested=[1])
        score_no_help = self.model.predict_score(df_no_help)[0]
        score_help = self.model.predict_score(df_help)[0]
        self.assertAlmostEqual(score_no_help - score_help, HELP_PENALTY, places=9)

    def test_help_penalty_does_not_go_below_zero(self) -> None:
        """When raw score < HELP_PENALTY the final score must be clipped to 0."""
        # Worst case: accuracy=0, NRT=1, error_rate=1, retries=MAX → raw=0.0
        df = _make_interaction_df(
            n=1, accuracy=[0], nrt=[1.0], window_error_rate=[1.0],
            retries=[MAX_RETRIES], help_requested=[1]
        )
        scores = self.model.predict_score(df)
        self.assertGreaterEqual(scores[0], 0.0)

    def test_help_penalty_clipped_to_one(self) -> None:
        """Score is always <= 1.0 even without penalty."""
        df = _make_interaction_df(n=5, accuracy=[1]*5, nrt=[0.0]*5,
                                  window_error_rate=[0.0]*5, retries=[0]*5,
                                  help_requested=[0]*5)
        scores = self.model.predict_score(df)
        self.assertTrue((scores <= 1.0).all())

    def test_help_penalty_per_row(self) -> None:
        """Penalty must only apply to rows where help_requested==1."""
        df = _make_interaction_df(
            n=2,
            accuracy=[1, 1],
            nrt=[0.0, 0.0],
            window_error_rate=[0.0, 0.0],
            retries=[0, 0],
            help_requested=[0, 1],
        )
        scores = self.model.predict_score(df)
        self.assertAlmostEqual(scores[0] - scores[1], HELP_PENALTY, places=9)


class TestRuleBasedCLSIRetryNormalization(unittest.TestCase):
    """Test retry normalization edge cases."""

    def test_zero_retries_norm_is_zero(self) -> None:
        """Zero retries must produce retries_norm = 0."""
        result = _normalize_retries(np.array([0]))
        self.assertAlmostEqual(result[0], 0.0)

    def test_max_retries_norm_is_one(self) -> None:
        """MAX_RETRIES retries must produce retries_norm = 1."""
        result = _normalize_retries(np.array([MAX_RETRIES]))
        self.assertAlmostEqual(result[0], 1.0)

    def test_above_max_retries_clips_to_one(self) -> None:
        """Retry counts above MAX_RETRIES must be clipped to retries_norm = 1."""
        result = _normalize_retries(np.array([MAX_RETRIES + 10]))
        self.assertAlmostEqual(result[0], 1.0)

    def test_partial_retries(self) -> None:
        """retries=1 → retries_norm = 1/MAX_RETRIES."""
        result = _normalize_retries(np.array([1]))
        self.assertAlmostEqual(result[0], 1.0 / MAX_RETRIES)

    def test_retries_norm_range(self) -> None:
        """All normalized values must be in [0, 1]."""
        values = np.array([0, 1, 2, 3, 4, 100])
        norm = _normalize_retries(values)
        self.assertTrue((norm >= 0.0).all())
        self.assertTrue((norm <= 1.0).all())


class TestRuleBasedCLSIThreshold(unittest.TestCase):
    """Test overload threshold behavior."""

    def test_overload_when_below_threshold(self) -> None:
        """Score < OVERLOAD_THRESHOLD must yield overload=1."""
        # worst possible inputs → score = 0.0 < 0.40
        df = _make_interaction_df(
            n=1, accuracy=[0], nrt=[1.0], window_error_rate=[1.0],
            retries=[MAX_RETRIES], help_requested=[0]
        )
        labels = self.model.predict(df)
        self.assertEqual(labels[0], 1)

    def test_no_overload_when_above_threshold(self) -> None:
        """Score >= OVERLOAD_THRESHOLD must yield overload=0."""
        df = _make_interaction_df(
            n=1, accuracy=[1], nrt=[0.0], window_error_rate=[0.0],
            retries=[0], help_requested=[0]
        )
        labels = self.model.predict(df)
        self.assertEqual(labels[0], 0)

    def test_custom_threshold(self) -> None:
        """Custom threshold must be respected."""
        model_high = RuleBasedCLSIModel(overload_threshold=0.99)
        df = _make_interaction_df(
            n=1, accuracy=[1], nrt=[0.0], window_error_rate=[0.0],
            retries=[0], help_requested=[0]
        )
        # score = 1.0, threshold = 0.99 → 1.0 >= 0.99 → overload=0
        self.assertEqual(model_high.predict(df)[0], 0)

        model_low = RuleBasedCLSIModel(overload_threshold=1.01)
        # score = 1.0, threshold = 1.01 → 1.0 < 1.01 → overload=1
        self.assertEqual(model_low.predict(df)[0], 1)

    def setUp(self) -> None:
        self.model = RuleBasedCLSIModel()


class TestRuleBasedCLSIScoreRange(unittest.TestCase):
    """Test that scores are always in [0, 1]."""

    def setUp(self) -> None:
        self.model = RuleBasedCLSIModel()

    def _random_df(self, n: int = 50, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        acc = rng.integers(0, 2, size=n).tolist()
        nrt = rng.uniform(0.0, 3.0, size=n).tolist()   # NRT can exceed 1.0
        wer = rng.uniform(0.0, 1.0, size=n).tolist()
        ret = rng.integers(0, 6, size=n).tolist()       # can exceed MAX_RETRIES
        hlp = rng.integers(0, 2, size=n).tolist()
        return _make_interaction_df(
            n=n, accuracy=acc, nrt=nrt, window_error_rate=wer,
            retries=ret, help_requested=hlp
        )

    def test_scores_in_zero_one(self) -> None:
        """All scores must be in [0, 1]."""
        df = self._random_df()
        scores = self.model.predict_score(df)
        self.assertTrue((scores >= 0.0).all(), f"Min score: {scores.min()}")
        self.assertTrue((scores <= 1.0).all(), f"Max score: {scores.max()}")

    def test_high_nrt_still_bounded(self) -> None:
        """NRT values > 1.0 must not push score below 0."""
        df = _make_interaction_df(
            n=3, accuracy=[0, 0, 0], nrt=[5.0, 10.0, 100.0],
            window_error_rate=[1.0, 1.0, 1.0], retries=[10, 10, 10],
            help_requested=[1, 1, 1]
        )
        scores = self.model.predict_score(df)
        self.assertTrue((scores >= 0.0).all())


class TestRuleBasedCLSIPredictProba(unittest.TestCase):
    """Test predict_proba shape and semantics."""

    def setUp(self) -> None:
        self.model = RuleBasedCLSIModel()

    def test_proba_shape(self) -> None:
        """predict_proba must return shape (n, 2)."""
        df = _make_interaction_df(n=7)
        proba = self.model.predict_proba(df)
        self.assertEqual(proba.shape, (7, 2))

    def test_proba_columns_sum_to_one(self) -> None:
        """Each row of predict_proba must sum to 1.0."""
        df = _make_interaction_df(n=10)
        proba = self.model.predict_proba(df)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_proba_col1_is_overload_risk(self) -> None:
        """Column 1 (overload probability) == 1 - score."""
        df = _make_interaction_df(n=5, accuracy=[0]*5, nrt=[1.0]*5,
                                  window_error_rate=[1.0]*5)
        scores = self.model.predict_score(df)
        proba = self.model.predict_proba(df)
        np.testing.assert_allclose(proba[:, 1], 1.0 - scores, atol=1e-9)

    def test_proba_in_zero_one(self) -> None:
        """All probability values must be in [0, 1]."""
        df = _make_interaction_df(n=20)
        proba = self.model.predict_proba(df)
        self.assertTrue((proba >= 0.0).all())
        self.assertTrue((proba <= 1.0).all())


class TestRuleBasedCLSIFit(unittest.TestCase):
    """Test that fit() is a no-op."""

    def test_fit_returns_self(self) -> None:
        model = RuleBasedCLSIModel()
        result = model.fit()
        self.assertIs(result, model)

    def test_fit_with_args_returns_self(self) -> None:
        model = RuleBasedCLSIModel()
        result = model.fit(X=np.zeros((5, 3)), y=np.zeros(5))
        self.assertIs(result, model)


# ===========================================================================
# BKT Tests
# ===========================================================================


class TestBKTInitialMastery(unittest.TestCase):
    """Test initial mastery probability."""

    def test_initial_mastery_default(self) -> None:
        """First element of fit_sequence must equal initial_mastery."""
        model = BKTModel()
        mastery = model.fit_sequence([1])
        self.assertAlmostEqual(mastery[0], DEFAULT_INITIAL_MASTERY)

    def test_initial_mastery_custom(self) -> None:
        """Custom initial_mastery must be reflected at index 0."""
        model = BKTModel(initial_mastery=0.7)
        mastery = model.fit_sequence([0, 1, 0])
        self.assertAlmostEqual(mastery[0], 0.7)

    def test_empty_sequence_returns_empty(self) -> None:
        """Empty response sequence must return an empty list."""
        model = BKTModel()
        mastery = model.fit_sequence([])
        self.assertEqual(mastery, [])

    def test_single_response_length_one(self) -> None:
        """A single response produces exactly one mastery value."""
        model = BKTModel()
        mastery = model.fit_sequence([1])
        self.assertEqual(len(mastery), 1)


class TestBKTUpdateCorrect(unittest.TestCase):
    """Test mastery update after a correct response."""

    def test_correct_response_increases_mastery(self) -> None:
        """After a correct response mastery at t+1 must be >= mastery at t."""
        model = BKTModel()
        mastery = model.fit_sequence([1, 1])
        # mastery[1] (after seeing response 0=correct) should be >= mastery[0]
        self.assertGreaterEqual(mastery[1], mastery[0])

    def test_posterior_correct_directly(self) -> None:
        """Verify posterior formula for correct=1 manually."""
        L = 0.3
        G = DEFAULT_GUESS
        S = DEFAULT_SLIP

        # P(L | correct) = L*(1-S) / [L*(1-S) + (1-L)*G]
        expected_posterior = (L * (1.0 - S)) / (L * (1.0 - S) + (1.0 - L) * G)
        result = _bkt_posterior(L, 1, G, S)
        self.assertAlmostEqual(result, expected_posterior, places=9)

    def test_correct_response_posterior_higher_than_prior(self) -> None:
        """Correct response must increase posterior above prior."""
        L = 0.3
        posterior = _bkt_posterior(L, 1, DEFAULT_GUESS, DEFAULT_SLIP)
        self.assertGreater(posterior, L)


class TestBKTUpdateIncorrect(unittest.TestCase):
    """Test mastery update after an incorrect response."""

    def test_incorrect_response_decreases_mastery_or_stays_low(self) -> None:
        """After an incorrect response, mastery at t+1 must be <= mastery at t."""
        model = BKTModel()
        mastery = model.fit_sequence([0, 0])
        # mastery[1] (after seeing response 0=incorrect) should be <= mastery[0]
        self.assertLessEqual(mastery[1], mastery[0])

    def test_posterior_incorrect_directly(self) -> None:
        """Verify posterior formula for correct=0 manually."""
        L = 0.3
        G = DEFAULT_GUESS
        S = DEFAULT_SLIP

        # P(L | incorrect) = L*S / [L*S + (1-L)*(1-G)]
        expected_posterior = (L * S) / (L * S + (1.0 - L) * (1.0 - G))
        result = _bkt_posterior(L, 0, G, S)
        self.assertAlmostEqual(result, expected_posterior, places=9)

    def test_incorrect_response_posterior_lower_than_prior(self) -> None:
        """Incorrect response must reduce posterior below prior."""
        L = 0.3
        posterior = _bkt_posterior(L, 0, DEFAULT_GUESS, DEFAULT_SLIP)
        self.assertLess(posterior, L)


class TestBKTRepeatedResponses(unittest.TestCase):
    """Test monotonicity of mastery over repeated responses."""

    def test_repeated_correct_increases_mastery(self) -> None:
        """Repeated correct responses must produce non-decreasing mastery."""
        model = BKTModel()
        mastery = model.fit_sequence([1] * 20)
        # Mastery should be non-decreasing over time
        for i in range(len(mastery) - 1):
            self.assertGreaterEqual(
                mastery[i + 1], mastery[i] - 1e-9,
                f"Mastery decreased at step {i+1}: {mastery[i+1]} < {mastery[i]}"
            )

    def test_repeated_correct_approaches_high_mastery(self) -> None:
        """After many correct responses, mastery should be substantially higher than initial."""
        model = BKTModel()
        mastery = model.fit_sequence([1] * 50)
        # After 50 correct responses mastery should be well above initial 0.3
        self.assertGreater(mastery[-1], 0.5)

    def test_repeated_incorrect_keeps_mastery_low(self) -> None:
        """Repeated incorrect responses must not push mastery above initial value."""
        model = BKTModel()
        mastery = model.fit_sequence([0] * 30)
        # Mastery should stay low throughout
        self.assertLess(mastery[-1], DEFAULT_INITIAL_MASTERY + DEFAULT_LEARN)

    def test_correct_then_incorrect_fluctuation(self) -> None:
        """Alternating responses produce intermediate mastery values."""
        model = BKTModel()
        mastery_all_correct = model.fit_sequence([1] * 10)
        mastery_alternating = model.fit_sequence([1, 0] * 5)
        # All-correct mastery should be higher at the end
        self.assertGreater(mastery_all_correct[-1], mastery_alternating[-1])


class TestBKTStruggleThreshold(unittest.TestCase):
    """Test the struggle/overload proxy threshold."""

    def test_initial_state_is_struggling(self) -> None:
        """With default params, initial mastery (0.3) == STRUGGLE_THRESHOLD → struggling."""
        model = BKTModel()
        df = _make_interaction_df(n=1, accuracy=[1])
        labels = model.predict(df)
        # mastery[0] = 0.3 = STRUGGLE_THRESHOLD → 0.3 < 0.3 is False → label = 0
        # The boundary: strictly less than, so 0.3 is NOT struggling
        self.assertEqual(labels[0], 0)

    def test_struggling_after_incorrect_responses(self) -> None:
        """After a string of incorrect responses mastery falls below threshold → struggling."""
        model = BKTModel()
        # Provide sequence: first few correct, then many incorrect
        responses = [0] * 20
        mastery = model.fit_sequence(responses)
        # Mastery should drop below threshold after several incorrect responses
        # Given initial=0.3 and many incorrect, mastery should stay low
        any_struggling = any(m < STRUGGLE_THRESHOLD for m in mastery)
        # With initial_mastery=0.3 and slip=0.1, after one incorrect
        # posterior = 0.3*0.1/(0.3*0.1 + 0.7*0.8) = 0.03/0.59 ≈ 0.051
        # Then 0.051 < 0.30 → struggling
        self.assertTrue(any_struggling)

    def test_not_struggling_after_many_correct(self) -> None:
        """After many correct responses mastery exceeds threshold → not struggling."""
        model = BKTModel()
        df = _make_interaction_df(n=30, accuracy=[1] * 30)
        labels = model.predict(df)
        # At least the last few should have mastery >= threshold
        self.assertEqual(labels[-1], 0)

    def test_custom_threshold(self) -> None:
        """Custom struggle_threshold must be respected."""
        model = BKTModel(struggle_threshold=0.99)  # nearly always struggling
        df = _make_interaction_df(n=5, accuracy=[1] * 5)
        labels = model.predict(df)
        # Even after correct responses, mastery likely < 0.99 → all struggling
        self.assertTrue(labels[0] == 1)


class TestBKTPredictProba(unittest.TestCase):
    """Test predict_proba shape and semantics."""

    def setUp(self) -> None:
        self.model = BKTModel()

    def test_proba_shape(self) -> None:
        """predict_proba must return shape (n, 2)."""
        df = _make_interaction_df(n=8)
        proba = self.model.predict_proba(df)
        self.assertEqual(proba.shape, (8, 2))

    def test_proba_columns_sum_to_one(self) -> None:
        """Each row must sum to 1.0."""
        df = _make_interaction_df(n=10, accuracy=[1, 0, 1, 0, 1, 1, 0, 1, 0, 0])
        proba = self.model.predict_proba(df)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_proba_col1_is_struggle_probability(self) -> None:
        """Column 1 must equal 1 - mastery."""
        df = _make_interaction_df(n=5)
        mastery = self.model.predict_score(df)
        proba = self.model.predict_proba(df)
        np.testing.assert_allclose(proba[:, 1], 1.0 - mastery, atol=1e-9)

    def test_proba_in_zero_one(self) -> None:
        """All probability values must be in [0, 1]."""
        df = _make_interaction_df(n=20, accuracy=[1, 0] * 10)
        proba = self.model.predict_proba(df)
        self.assertTrue((proba >= 0.0).all())
        self.assertTrue((proba <= 1.0).all())


class TestBKTFit(unittest.TestCase):
    """Test that fit() is a no-op."""

    def test_fit_returns_self(self) -> None:
        model = BKTModel()
        result = model.fit()
        self.assertIs(result, model)

    def test_fit_with_args_returns_self(self) -> None:
        model = BKTModel()
        result = model.fit(X=np.zeros((5, 3)), y=np.zeros(5))
        self.assertIs(result, model)


class TestBKTMultiProfile(unittest.TestCase):
    """Test that multiple profiles are processed independently."""

    def test_multi_profile_independence(self) -> None:
        """BKT mastery for profile B must not be contaminated by profile A's responses."""
        # Profile A: all correct (drives mastery high)
        df_a = _make_interaction_df(n=20, accuracy=[1] * 20, profile="A")
        # Profile B: all incorrect (keeps mastery low)
        df_b = _make_interaction_df(n=20, accuracy=[0] * 20, profile="B")
        combined = pd.concat([df_a, df_b], ignore_index=True)

        model = BKTModel()

        # Mastery computed on combined df should match individual processing
        mastery_combined = model.predict_score(combined)
        mastery_b_combined = mastery_combined[20:]  # profile B rows

        mastery_b_alone = model.predict_score(df_b)

        np.testing.assert_allclose(mastery_b_combined, mastery_b_alone, atol=1e-9)

    def test_multi_profile_length(self) -> None:
        """predict_score on combined df must return n_A + n_B values."""
        df_a = _make_interaction_df(n=10, profile="A")
        df_b = _make_interaction_df(n=15, profile="B")
        combined = pd.concat([df_a, df_b], ignore_index=True)
        model = BKTModel()
        mastery = model.predict_score(combined)
        self.assertEqual(len(mastery), 25)


# ===========================================================================
# Temporal Behavior Tests
# ===========================================================================


class TestTemporalBehaviorRuleCLSI(unittest.TestCase):
    """Verify Rule-CLSI predictions at time t are causal (no future leakage)."""

    def setUp(self) -> None:
        self.model = RuleBasedCLSIModel()

    def test_changing_future_rows_does_not_affect_past_predictions(self) -> None:
        """Predictions at t=0..4 must not change when rows 5..9 are altered."""
        df_base = _make_interaction_df(
            n=10,
            accuracy=[1] * 10,
            nrt=[0.5] * 10,
            window_error_rate=[0.0] * 10,
            retries=[0] * 10,
            help_requested=[0] * 10,
        )
        df_modified = df_base.copy()
        # Corrupt future rows with extreme values
        df_modified.loc[5:, "accuracy"] = 0
        df_modified.loc[5:, "nrt"] = 99.0
        df_modified.loc[5:, "window_error_rate"] = 1.0
        df_modified.loc[5:, "retries"] = MAX_RETRIES
        df_modified.loc[5:, "help_requested"] = 1

        scores_base = self.model.predict_score(df_base)
        scores_modified = self.model.predict_score(df_modified)

        # Rows 0..4 must be identical (the model is row-wise)
        np.testing.assert_array_almost_equal(scores_base[:5], scores_modified[:5])

    def test_row_wise_independence(self) -> None:
        """Each row's score depends only on that row's column values."""
        # The CLSI formula is applied per-row with no temporal state.
        df = _make_interaction_df(n=5,
                                  accuracy=[1, 0, 1, 0, 1],
                                  nrt=[0.2, 0.8, 0.2, 0.8, 0.2],
                                  window_error_rate=[0.0, 1.0, 0.0, 1.0, 0.0],
                                  retries=[0, 2, 0, 2, 0],
                                  help_requested=[0, 0, 0, 0, 0])

        # Scores for even rows (accuracy=1) and odd rows (accuracy=0) should differ
        even_scores = self.model.predict_score(df)[[0, 2, 4]]
        odd_scores = self.model.predict_score(df)[[1, 3]]

        self.assertTrue((even_scores > odd_scores.max()).all())


class TestTemporalBehaviorBKT(unittest.TestCase):
    """Verify BKT mastery at time t depends only on responses 0..t-1."""

    def setUp(self) -> None:
        self.model = BKTModel()

    def test_changing_future_responses_does_not_affect_past_mastery(self) -> None:
        """Mastery at t=0..4 must not change when responses from t=5 onward are altered."""
        responses_a = [1, 0, 1, 1, 0, 1, 1, 1, 1, 1]
        responses_b = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0]  # t=5..9 differ

        mastery_a = self.model.fit_sequence(responses_a)
        mastery_b = self.model.fit_sequence(responses_b)

        # Positions 0..4 must be identical
        np.testing.assert_array_almost_equal(mastery_a[:5], mastery_b[:5])

    def test_mastery_at_t_does_not_use_response_t(self) -> None:
        """mastery[t] must equal the mastery BEFORE observing response t.

        Verifying this by comparing the mastery returned at position t
        with the mastery expected before that observation.
        """
        model = BKTModel(initial_mastery=0.3)

        responses = [1]
        mastery = model.fit_sequence(responses)

        # mastery[0] must be initial_mastery, not the post-update value
        self.assertAlmostEqual(mastery[0], 0.3)

    def test_length_matches_input_length(self) -> None:
        """fit_sequence must return exactly len(responses) values."""
        for n in [1, 5, 20, 100]:
            responses = [1] * n
            mastery = self.model.fit_sequence(responses)
            self.assertEqual(len(mastery), n)

    def test_predict_score_uses_prior_responses_only(self) -> None:
        """Verify through DataFrame interface that mastery is pre-observation."""
        model = BKTModel(initial_mastery=0.5)
        df = _make_interaction_df(n=3, accuracy=[1, 1, 1])

        mastery = model.predict_score(df)

        # mastery[0] == initial_mastery (before ANY response is observed)
        self.assertAlmostEqual(mastery[0], 0.5)

        # mastery[1] == mastery after 1 correct response
        expected_posterior_0 = _bkt_posterior(0.5, 1, DEFAULT_GUESS, DEFAULT_SLIP)
        expected_mastery_1 = _bkt_learn_update(expected_posterior_0, DEFAULT_LEARN)
        self.assertAlmostEqual(mastery[1], expected_mastery_1, places=9)


# ===========================================================================
# Integration: Both Models on Simulated Data
# ===========================================================================


class TestModelsOnSimulatedData(unittest.TestCase):
    """Smoke tests running both models on simulator output."""

    def _generate_simulated_df(self, n: int = 50) -> pd.DataFrame:
        """Generate a minimal simulated interaction DataFrame."""
        from src.simulator import simulate_learner
        return simulate_learner("average", num_interactions=n, seed=0)

    def test_rule_clsi_on_simulated_data_no_error(self) -> None:
        """Rule-CLSI must run without errors on simulator output."""
        df = self._generate_simulated_df()
        model = RuleBasedCLSIModel()
        scores = model.predict_score(df)
        labels = model.predict(df)
        proba = model.predict_proba(df)

        self.assertEqual(len(scores), len(df))
        self.assertEqual(len(labels), len(df))
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue((scores >= 0.0).all())
        self.assertTrue((scores <= 1.0).all())

    def test_bkt_on_simulated_data_no_error(self) -> None:
        """BKT must run without errors on simulator output."""
        df = self._generate_simulated_df()
        model = BKTModel()
        mastery = model.predict_score(df)
        labels = model.predict(df)
        proba = model.predict_proba(df)

        self.assertEqual(len(mastery), len(df))
        self.assertEqual(len(labels), len(df))
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue((mastery >= 0.0).all())
        self.assertTrue((mastery <= 1.0).all())

    def test_rule_clsi_labels_are_binary(self) -> None:
        """predict() labels must be 0 or 1."""
        df = self._generate_simulated_df()
        labels = RuleBasedCLSIModel().predict(df)
        self.assertTrue(set(labels).issubset({0, 1}))

    def test_bkt_labels_are_binary(self) -> None:
        """predict() labels must be 0 or 1."""
        df = self._generate_simulated_df()
        labels = BKTModel().predict(df)
        self.assertTrue(set(labels).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
