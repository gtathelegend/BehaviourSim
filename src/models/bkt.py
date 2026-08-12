"""Bayesian Knowledge Tracing (BKT) baseline model module.

Standard BKT (Corbett & Anderson, 1994)
----------------------------------------
BKT tracks the probability that a learner has mastered a skill using four
parameters:

    P(L_0)  = initial_mastery = 0.3   (prior probability of knowing the skill)
    P(T)    = learn           = 0.1   (probability of transitioning to mastery
                                       after an unknown → practice opportunity)
    P(G)    = guess           = 0.2   (probability of answering correctly despite
                                       not knowing: correct | ¬L)
    P(S)    = slip            = 0.1   (probability of answering incorrectly despite
                                       knowing: incorrect | L)

The entire domain is treated as a **single skill** (standard flat BKT).

Update equations (applied after each observed response)
-------------------------------------------------------
Given the current mastery belief P(L_t) and observation o_{t+1} ∈ {0, 1}:

    # Step 1 – Posterior update (Bayes rule)
    if o == 1 (correct):
        P(L | correct) = P(L_t) * (1 - P(S))
                       / [P(L_t) * (1 - P(S)) + (1 - P(L_t)) * P(G)]
    if o == 0 (incorrect):
        P(L | incorrect) = P(L_t) * P(S)
                         / [P(L_t) * P(S) + (1 - P(L_t)) * (1 - P(G))]

    # Step 2 – Learning update (transition from ¬L to L)
    P(L_{t+1}) = P(L | obs) + (1 - P(L | obs)) * P(T)

Overload/struggle proxy
-----------------------
A learner is considered to be struggling (overload proxy) when:

    mastery_probability < STRUGGLE_THRESHOLD  (default 0.30)

This maps naturally: low mastery → the domain is likely too difficult, which
drives the kind of cognitive overload captured by the simulator.

Causality guarantee
-------------------
``fit_sequence`` processes responses in chronological order and updates
mastery belief *after* observing each response. At time step t the mastery
value exposed corresponds to the belief *before* seeing step t's response
(i.e., based only on steps 0..t-1), which strictly avoids using the current
or future response to label the current time step.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants – Standard BKT parameters
# ---------------------------------------------------------------------------

DEFAULT_INITIAL_MASTERY: float = 0.3
DEFAULT_LEARN: float = 0.1
DEFAULT_GUESS: float = 0.2
DEFAULT_SLIP: float = 0.1

#: Mastery below this threshold → struggle/overload proxy = 1.
STRUGGLE_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Internal update function
# ---------------------------------------------------------------------------

def _bkt_posterior(mastery: float, correct: int, guess: float, slip: float) -> float:
    """Compute the posterior mastery probability after one observation.

    Parameters
    ----------
    mastery : float
        Current mastery probability P(L_t).
    correct : int
        1 if the response was correct, 0 if incorrect.
    guess : float
        P(correct | ¬L) — probability of guessing correctly.
    slip : float
        P(incorrect | L) — probability of slipping.

    Returns
    -------
    float
        Updated posterior P(L | observation).
    """
    if correct == 1:
        numerator = mastery * (1.0 - slip)
        denominator = numerator + (1.0 - mastery) * guess
    else:
        numerator = mastery * slip
        denominator = numerator + (1.0 - mastery) * (1.0 - guess)

    if denominator < 1e-12:
        # Degenerate case: return current belief unchanged
        return mastery

    return numerator / denominator


def _bkt_learn_update(posterior: float, learn: float) -> float:
    """Apply the learning transition to move from unknown to known.

    P(L_{t+1}) = posterior + (1 - posterior) * P(T)

    Parameters
    ----------
    posterior : float
        Posterior mastery after observation.
    learn : float
        P(T) — probability of transitioning from ¬L to L.

    Returns
    -------
    float
        Updated mastery probability in [0, 1].
    """
    return posterior + (1.0 - posterior) * learn


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------

class BKTModel:
    """Bayesian Knowledge Tracing (BKT) baseline model.

    The entire domain is treated as one skill.  Parameters are fixed at
    construction time; ``fit`` is a no-op provided for interface compatibility.

    Parameters
    ----------
    initial_mastery : float
        P(L_0): prior probability that the learner already knows the skill.
    learn : float
        P(T): probability of transitioning to mastery per practice opportunity.
    guess : float
        P(G): probability of a correct response given no mastery.
    slip : float
        P(S): probability of an incorrect response despite mastery.
    struggle_threshold : float
        Mastery below this value is interpreted as overload/struggle.

    Example usage
    -------------
    >>> model = BKTModel()
    >>> mastery_over_time = model.fit_sequence([1, 0, 1, 1, 0])
    >>> scores = model.predict_score(df)   # from a simulator DataFrame
    >>> labels = model.predict(df)
    >>> proba  = model.predict_proba(df)
    """

    def __init__(
        self,
        initial_mastery: float = DEFAULT_INITIAL_MASTERY,
        learn: float = DEFAULT_LEARN,
        guess: float = DEFAULT_GUESS,
        slip: float = DEFAULT_SLIP,
        struggle_threshold: float = STRUGGLE_THRESHOLD,
        **kwargs,
    ) -> None:
        self.initial_mastery = initial_mastery
        self.learn = learn
        self.guess = guess
        self.slip = slip
        self.struggle_threshold = struggle_threshold

    # ------------------------------------------------------------------
    # Scikit-learn–style fit (no-op — parameters are fixed)
    # ------------------------------------------------------------------

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, None] = None,
        y: Union[np.ndarray, None] = None,
    ) -> "BKTModel":
        """No-op fit for interface compatibility.

        BKT parameters are specified at construction; this model does not
        estimate parameters from data.

        Returns
        -------
        self
        """
        return self

    # ------------------------------------------------------------------
    # Core sequential update
    # ------------------------------------------------------------------

    def fit_sequence(self, responses: Sequence[int]) -> List[float]:
        """Run the BKT update loop over a chronological response sequence.

        Returns the mastery probability *before* each response is observed so
        that the returned value at index t is based solely on observations
        0..t-1 (no future leakage).

        Parameters
        ----------
        responses : sequence of int
            Binary response sequence: 1 = correct, 0 = incorrect.

        Returns
        -------
        mastery_over_time : list of float, length == len(responses)
            mastery_over_time[t] = P(L_t) before observing response t.
        """
        mastery = self.initial_mastery
        mastery_over_time: List[float] = []

        for obs in responses:
            # Record mastery BEFORE updating with current observation
            mastery_over_time.append(mastery)

            # Bayesian posterior update
            posterior = _bkt_posterior(mastery, int(obs), self.guess, self.slip)

            # Learning transition
            mastery = _bkt_learn_update(posterior, self.learn)

        return mastery_over_time

    # ------------------------------------------------------------------
    # DataFrame-based interface
    # ------------------------------------------------------------------

    def _get_mastery_series(self, df: pd.DataFrame) -> np.ndarray:
        """Extract mastery probabilities from an interaction DataFrame.

        Processes each learner profile independently (no cross-profile leakage).

        Parameters
        ----------
        df : pd.DataFrame
            Interaction DataFrame with at least the column ``accuracy``
            and optionally ``profile``.

        Returns
        -------
        mastery : np.ndarray of float, shape (n_interactions,)
            Mastery probability at each interaction (before seeing that
            interaction's response).
        """
        if "accuracy" not in df.columns:
            raise ValueError("DataFrame must contain an 'accuracy' column.")

        # Process each profile independently
        if "profile" in df.columns:
            profiles = df["profile"].unique()
        else:
            profiles = ["__single__"]

        all_mastery: List[np.ndarray] = []

        for profile in profiles:
            if "profile" in df.columns:
                sub = df[df["profile"] == profile].reset_index(drop=True)
            else:
                sub = df.reset_index(drop=True)

            responses = sub["accuracy"].to_numpy(dtype=int).tolist()
            mastery_seq = self.fit_sequence(responses)
            all_mastery.append(np.array(mastery_seq, dtype=float))

        return np.concatenate(all_mastery)

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """Return the mastery probability for each interaction row.

        This is the primary continuous output of BKT.  Higher mastery means
        the learner is coping well; low mastery signals struggle/overload.

        The value at row t reflects belief formed from observations 0..t-1
        only (causal, no future leakage).

        Parameters
        ----------
        df : pd.DataFrame
            Interaction DataFrame with ``accuracy`` column.

        Returns
        -------
        mastery : np.ndarray of float, shape (n_interactions,)
            Mastery probability in (0, 1) at each time step.
        """
        return self._get_mastery_series(df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict binary struggle/overload label for each interaction.

        Parameters
        ----------
        df : pd.DataFrame
            Interaction DataFrame with ``accuracy`` column.

        Returns
        -------
        labels : np.ndarray of int, shape (n_interactions,)
            1 if mastery < struggle_threshold (struggling), 0 otherwise.
        """
        mastery = self.predict_score(df)
        return (mastery < self.struggle_threshold).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability-like outputs for the struggle class.

        The struggle probability is ``1 - mastery`` so that higher values
        indicate higher overload risk.  This allows ROC-AUC computation.

        Parameters
        ----------
        df : pd.DataFrame
            Interaction DataFrame with ``accuracy`` column.

        Returns
        -------
        proba : np.ndarray of shape (n_interactions, 2)
            Column 0: P(no struggle) = mastery probability.
            Column 1: P(struggle)    = 1 - mastery probability.
        """
        mastery = self.predict_score(df)
        struggle_prob = 1.0 - mastery
        return np.column_stack([mastery, struggle_prob])
