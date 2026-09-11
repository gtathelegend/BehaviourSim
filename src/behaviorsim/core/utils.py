"""Utility functions for BehaviorSim core engine."""

from __future__ import annotations

import bisect
from typing import Any, Optional, Sequence
import numpy as np



def create_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a new NumPy Random Number Generator instance.

    Args:
        seed: Optional integer seed for reproducibility. If None, unpredictable
            entropy is used.

    Returns:
        np.random.Generator instance.
    """
    return np.random.default_rng(seed)


def derive_sequence_seed(
    base_seed: int,
    profile_idx: int,
    sequence_idx: int,
) -> int:
    """Derive a deterministic integer seed for an individual sequence.

    Formula preserves historical compatibility:
        base_seed + profile_idx * 10000 + (sequence_idx - 1) * 100

    Args:
        base_seed: Global simulation base seed.
        profile_idx: 0-based index of the profile.
        sequence_idx: 1-based index of the sequence within the profile.

    Returns:
        Derived integer seed.
    """
    return base_seed + profile_idx * 10000 + (sequence_idx - 1) * 100


derive_learner_seed = derive_sequence_seed


def validate_transition_matrix(
    matrix: np.ndarray,
    *,
    atol: float = 1e-8,
) -> None:
    """Validate that a matrix is a valid 2D square stochastic transition matrix.

    Args:
        matrix: Input 2D NumPy array.
        atol: Absolute tolerance for probability bounds and row sums.

    Raises:
        ValueError: If matrix is invalid (not 2D, non-square, non-finite,
            probabilities outside [0, 1], or row sums != 1.0).
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.asarray(matrix)

    if matrix.ndim != 2:
        raise ValueError(f"Transition matrix must be 2-dimensional, got {matrix.ndim}D.")

    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError(f"Transition matrix must be square, got shape ({rows}, {cols}).")

    if not np.isfinite(matrix).all():
        raise ValueError("Transition matrix contains non-finite values (NaN or Inf).")

    if np.any(matrix < -atol) or np.any(matrix > 1.0 + atol):
        raise ValueError("Transition matrix probabilities must be within [0, 1].")

    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(f"Transition matrix rows must sum to 1.0. Row sums: {row_sums}")


def sample_categorical(
    rng: np.random.Generator,
    items: Sequence[Any],
    probabilities: Optional[Sequence[float]] = None,
) -> Any:
    """Sample a single item from a discrete categorical distribution.

    Args:
        rng: NumPy random Generator instance.
        items: Sequence of items to sample from.
        probabilities: Optional sequence of sampling probabilities for each item.
            If None, items are sampled uniformly.

    Returns:
        A single sampled item from items.

    Raises:
        ValueError: If items is empty, probabilities length mismatch, or probabilities
            are invalid (negative, non-finite, or sum != 1.0).
    """
    if not items:
        raise ValueError("Cannot sample from an empty items sequence.")

    n_items = len(items)

    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=float)
        if probs.shape != (n_items,):
            raise ValueError(
                f"Probabilities length ({len(probs)}) does not match items length ({n_items})."
            )
        if not np.isfinite(probs).all():
            raise ValueError("Probabilities contain non-finite values (NaN or Inf).")
        if np.any(probs < 0.0):
            raise ValueError("Probabilities must be non-negative.")
        if np.any(probs > 1.0 + 1e-6):
            raise ValueError("Probabilities must be <= 1.0.")
        if not np.isclose(probs.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Probabilities must sum to 1.0, got sum {probs.sum()}.")

        idx = rng.choice(n_items, p=probs)
    else:
        idx = rng.choice(n_items)

    return items[idx]


def _prevalidated_sample_categorical(
    rng: np.random.Generator,
    items: Sequence[Any],
    cdf: Sequence[float],
) -> Any:
    """Sample an item using precomputed cumulative probabilities without redundant validation.

    Invariant:
        `cdf` was precomputed from a previously validated probability vector (non-empty,
        non-negative, finite, monotonically non-decreasing, with cdf[-1] == 1.0) and
        len(cdf) == len(items). This function is strictly internal to BehaviorSim for
        performance-critical simulation loops and is never exposed as an unchecked public API.

    Args:
        rng: NumPy random Generator instance.
        items: Sequence of items to sample from.
        cdf: Precomputed cumulative distribution function table.

    Returns:
        A single sampled item from items.
    """
    idx = bisect.bisect_right(cdf, rng.random())
    return items[idx]


_sample_categorical_prevalidated = _prevalidated_sample_categorical

