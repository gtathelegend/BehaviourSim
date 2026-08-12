"""Rule-Based CLSI model baseline module.

Formula
-------
The Cognitive Load Stress Index (CLSI) composite score is computed as:

    CLSI_raw = (
        0.50 * accuracy
      + 0.25 * (1 - NRT)
      + 0.15 * (1 - window_error_rate)
      + 0.10 * (1 - retries_norm)
    ) / (0.50 + 0.25 + 0.15 + 0.10)

where:
    accuracy          : fraction of correct responses in [0, 1].
    NRT               : normalized response time (clipped to [0, 1] before use).
    window_error_rate : recent error rate in [0, 1].
    retries_norm      : retries / MAX_RETRIES, clipped to [0, 1].  MAX_RETRIES=3.

Help penalty
------------
When help_requested == 1 a fixed penalty of HELP_PENALTY = 0.10 is subtracted
from CLSI_raw.  The final score is then clipped to [0, 1]:

    score = clip(CLSI_raw - help_requested * HELP_PENALTY, 0, 1)

Overload threshold
------------------
A learner is predicted to be in overload when:

    score < OVERLOAD_THRESHOLD   (default 0.40)  →  overload = 1
    score >= OVERLOAD_THRESHOLD                  →  overload = 0

Causality guarantee
-------------------
All inputs (accuracy, NRT, window_error_rate, retries, help_requested) are
already computed chronologically by the simulator / feature engineering layer
without forward-looking information. This model simply evaluates the formula
at each time step; it never aggregates across future rows.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Composite weight vector (must sum to 1.0)
_W_ACCURACY: float = 0.50
_W_NRT: float = 0.25
_W_ERROR_RATE: float = 0.15
_W_RETRIES: float = 0.10
_WEIGHT_SUM: float = _W_ACCURACY + _W_NRT + _W_ERROR_RATE + _W_RETRIES  # == 1.0

#: Maximum retries used for safe normalization.
MAX_RETRIES: int = 3

#: Penalty subtracted from CLSI_raw when help is requested.
HELP_PENALTY: float = 0.10

#: Overload decision threshold: score < OVERLOAD_THRESHOLD → overload=1.
OVERLOAD_THRESHOLD: float = 0.40


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_retries(retries: np.ndarray) -> np.ndarray:
    """Normalize retry counts to [0, 1] using MAX_RETRIES.

    Counts above MAX_RETRIES are clipped so the denominator is never zero
    and the result is bounded.

    Args:
        retries: 1-D array of non-negative integer retry counts.

    Returns:
        Float array in [0, 1].
    """
    return np.clip(retries / MAX_RETRIES, 0.0, 1.0)


def _compute_raw_score(
    accuracy: np.ndarray,
    nrt: np.ndarray,
    window_error_rate: np.ndarray,
    retries: np.ndarray,
) -> np.ndarray:
    """Compute the un-penalized CLSI composite score.

    Args:
        accuracy:          Binary correct/incorrect per interaction, shape (n,).
        nrt:               Normalized response time, shape (n,).  Clipped to [0,1].
        window_error_rate: Recent windowed error rate in [0,1], shape (n,).
        retries:           Retry counts (non-negative integers), shape (n,).

    Returns:
        Raw CLSI scores in [0, 1], shape (n,).
    """
    acc = np.asarray(accuracy, dtype=float)
    nrt_ = np.clip(np.asarray(nrt, dtype=float), 0.0, None)  # NRT >= 0
    wer = np.asarray(window_error_rate, dtype=float)
    ret_norm = _normalize_retries(np.asarray(retries, dtype=float))

    # NRT is already normalized relative to rt_mean_optimal by the simulator.
    # Values >1 mean the learner was slower than their typical optimal time.
    # We clip to [0,1] so very slow responses don't push NRT below 0 after inversion.
    nrt_clipped = np.clip(nrt_, 0.0, 1.0)

    raw = (
        _W_ACCURACY * acc
        + _W_NRT * (1.0 - nrt_clipped)
        + _W_ERROR_RATE * (1.0 - wer)
        + _W_RETRIES * (1.0 - ret_norm)
    ) / _WEIGHT_SUM

    return raw.astype(float)


def _apply_help_penalty(raw_scores: np.ndarray, help_requested: np.ndarray) -> np.ndarray:
    """Subtract HELP_PENALTY when help is requested and clip to [0, 1].

    The final score is clipped to [0, 1] after penalisation.

    Args:
        raw_scores:    Un-penalized CLSI scores, shape (n,).
        help_requested: Binary array (0 or 1), shape (n,).

    Returns:
        Penalized and clipped CLSI scores in [0, 1], shape (n,).
    """
    penalized = raw_scores - np.asarray(help_requested, dtype=float) * HELP_PENALTY
    return np.clip(penalized, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------

class RuleBasedCLSIModel:
    """Baseline learning path model using heuristic rule-based CLSI scoring.

    The model is **parameter-free**: there is no training step.  ``fit`` is
    provided only for interface compatibility and is a no-op.

    Parameters
    ----------
    overload_threshold:
        Score below which a learner is classified as overloaded.
        Defaults to OVERLOAD_THRESHOLD (0.40).

    Example usage
    -------------
    >>> model = RuleBasedCLSIModel()
    >>> scores = model.predict_score(df)
    >>> labels = model.predict(df)
    >>> proba  = model.predict_proba(df)
    """

    def __init__(self, overload_threshold: float = OVERLOAD_THRESHOLD) -> None:
        self.overload_threshold = overload_threshold

    # ------------------------------------------------------------------
    # Scikit-learn–style fit (no-op — the model is rule-based)
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, None] = None,
        y: Union[np.ndarray, None] = None,
    ) -> "RuleBasedCLSIModel":
        """No-op fit for interface compatibility.

        The rule-based model has no trainable parameters.

        Returns
        -------
        self
        """
        return self

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """Compute the CLSI composite score for each interaction row.

        Only columns available *at or before* each time step are used
        (accuracy, nrt, window_error_rate, retries, help_requested).
        No future information is accessed.

        Parameters
        ----------
        df:
            Interaction DataFrame with at least the columns:
            ``accuracy``, ``nrt``, ``window_error_rate``,
            ``retries``, ``help_requested``.

        Returns
        -------
        scores : np.ndarray of shape (n_interactions,)
            CLSI score per interaction, clipped to [0, 1].
            Higher scores indicate better cognitive load (less stress).
        """
        required = {"accuracy", "nrt", "window_error_rate", "retries", "help_requested"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        raw = _compute_raw_score(
            accuracy=df["accuracy"].to_numpy(dtype=float),
            nrt=df["nrt"].to_numpy(dtype=float),
            window_error_rate=df["window_error_rate"].to_numpy(dtype=float),
            retries=df["retries"].to_numpy(dtype=float),
        )
        scores = _apply_help_penalty(raw, df["help_requested"].to_numpy(dtype=float))
        return scores

    # ------------------------------------------------------------------
    # Predict (binary overload label)
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict binary overload label for each interaction.

        Parameters
        ----------
        df:
            Interaction DataFrame (see ``predict_score`` for required columns).

        Returns
        -------
        labels : np.ndarray of int, shape (n_interactions,)
            1 if overload predicted (score < threshold), 0 otherwise.
        """
        scores = self.predict_score(df)
        return (scores < self.overload_threshold).astype(int)

    # ------------------------------------------------------------------
    # Predict probability (for AUC computation)
    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return a probability-like score for the overload class.

        The overload probability is defined as ``1 - CLSI_score`` so that
        higher values correspond to higher overload risk.  This enables
        computing ROC-AUC without reducing to binary predictions.

        Parameters
        ----------
        df:
            Interaction DataFrame (see ``predict_score`` for required columns).

        Returns
        -------
        proba : np.ndarray of shape (n_interactions, 2)
            Column 0: P(no overload) = CLSI score.
            Column 1: P(overload)    = 1 - CLSI score.
        """
        scores = self.predict_score(df)
        overload_prob = 1.0 - scores
        no_overload_prob = scores
        return np.column_stack([no_overload_prob, overload_prob])
