"""Unit tests for BehaviorSim core utilities (behaviorsim.core.utils)."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from behaviorsim.core.utils import (
    create_rng,
    derive_learner_seed,
    derive_sequence_seed,
    sample_categorical,
    validate_transition_matrix,
)


def test_create_rng_returns_generator() -> None:
    """Verify create_rng returns a NumPy random Generator."""
    rng = create_rng(42)
    assert isinstance(rng, np.random.Generator)


def test_create_rng_deterministic_with_same_seed() -> None:
    """Verify identical seeds produce identical random sequences."""
    rng1 = create_rng(42)
    rng2 = create_rng(42)
    val1 = rng1.random(10)
    val2 = rng2.random(10)
    np.testing.assert_array_equal(val1, val2)


def test_create_rng_different_with_different_seeds() -> None:
    """Verify different seeds produce different random sequences."""
    rng1 = create_rng(42)
    rng2 = create_rng(99)
    assert not np.array_equal(rng1.random(10), rng2.random(10))


def test_derive_sequence_and_learner_seed_formula() -> None:
    """Verify sequence and learner seed derivation matches historical formula exactly."""
    assert derive_sequence_seed(42, 0, 1) == 42
    assert derive_sequence_seed(42, 1, 1) == 10042
    assert derive_sequence_seed(42, 0, 2) == 142
    assert derive_sequence_seed(100, 2, 5) == 20500

    # Test backward-compatible alias
    assert derive_learner_seed(42, 0, 1) == derive_sequence_seed(42, 0, 1)
    assert derive_learner_seed(100, 2, 5) == derive_sequence_seed(100, 2, 5)


def test_validate_transition_matrix_valid() -> None:
    """Verify a valid square stochastic matrix passes validation."""
    valid_matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.3, 0.2, 0.5],
    ])
    validate_transition_matrix(valid_matrix)  # Should not raise


def test_validate_transition_matrix_non_2d() -> None:
    """Verify non-2D input raises ValueError."""
    with pytest.raises(ValueError, match="2-dimensional"):
        validate_transition_matrix(np.array([0.5, 0.5]))


def test_validate_transition_matrix_non_square() -> None:
    """Verify non-square matrix raises ValueError."""
    with pytest.raises(ValueError, match="square"):
        validate_transition_matrix(np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]))


def test_validate_transition_matrix_negative_probability() -> None:
    """Verify negative probability raises ValueError."""
    with pytest.raises(ValueError, match="probabilities must be within"):
        validate_transition_matrix(np.array([[1.1, -0.1], [0.5, 0.5]]))


def test_validate_transition_matrix_probability_greater_than_one() -> None:
    """Verify probability greater than 1 raises ValueError."""
    with pytest.raises(ValueError, match="probabilities must be within"):
        validate_transition_matrix(np.array([[1.5, 0.0], [0.5, 0.5]]))


def test_validate_transition_matrix_invalid_row_sums() -> None:
    """Verify rows not summing to 1 raise ValueError."""
    with pytest.raises(ValueError, match="rows must sum to 1.0"):
        validate_transition_matrix(np.array([[0.5, 0.6], [0.4, 0.6]]))


def test_validate_transition_matrix_non_finite() -> None:
    """Verify NaN or Inf values raise ValueError."""
    with pytest.raises(ValueError, match="non-finite"):
        validate_transition_matrix(np.array([[np.nan, 1.0], [0.0, 1.0]]))

    with pytest.raises(ValueError, match="non-finite"):
        validate_transition_matrix(np.array([[np.inf, 0.0], [0.0, 1.0]]))


def test_sample_categorical_uniform() -> None:
    """Verify uniform categorical sampling across items."""
    rng = create_rng(42)
    items = ["a", "b", "c"]
    samples = [sample_categorical(rng, items) for _ in range(100)]
    assert set(samples).issubset(set(items))
    assert len(set(samples)) == 3  # All items sampled over 100 trials


def test_sample_categorical_weighted_deterministic() -> None:
    """Verify weighted categorical sampling is deterministic with fixed seed."""
    rng1 = create_rng(123)
    rng2 = create_rng(123)
    items = ["A", "B", "C"]
    probs = [0.7, 0.2, 0.1]
    sample1 = sample_categorical(rng1, items, probs)
    sample2 = sample_categorical(rng2, items, probs)
    assert sample1 == sample2


def test_sample_categorical_empty_items() -> None:
    """Verify empty items sequence raises ValueError."""
    rng = create_rng(42)
    with pytest.raises(ValueError, match="empty items"):
        sample_categorical(rng, [])


def test_sample_categorical_mismatched_probabilities_length() -> None:
    """Verify mismatch between items and probabilities length raises ValueError."""
    rng = create_rng(42)
    with pytest.raises(ValueError, match="does not match items length"):
        sample_categorical(rng, ["a", "b"], [0.5, 0.3, 0.2])


def test_sample_categorical_negative_probabilities() -> None:
    """Verify negative probability raises ValueError."""
    rng = create_rng(42)
    with pytest.raises(ValueError, match="must be non-negative"):
        sample_categorical(rng, ["a", "b"], [-0.1, 1.1])


def test_sample_categorical_invalid_sum() -> None:
    """Verify probabilities not summing to 1 raise ValueError."""
    rng = create_rng(42)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        sample_categorical(rng, ["a", "b"], [0.4, 0.4])


def test_prevalidated_sample_categorical_equivalence() -> None:
    """Verify _prevalidated_sample_categorical produces identical draws to sample_categorical."""
    from behaviorsim.core.utils import _prevalidated_sample_categorical

    items = ["low", "mid", "high"]
    probs = [0.2, 0.5, 0.3]
    cdf = (0.2, 0.7, 1.0)

    # Independent RNGs with same seed
    rng1 = create_rng(12345)
    rng2 = create_rng(12345)

    draws_std = [sample_categorical(rng1, items, probs) for _ in range(500)]
    draws_fast = [_prevalidated_sample_categorical(rng2, items, cdf) for _ in range(500)]

    assert draws_std == draws_fast


def test_prevalidated_sample_categorical_distribution_frequencies() -> None:
    """Verify statistical frequency convergence of _prevalidated_sample_categorical."""
    from collections import Counter
    from behaviorsim.core.utils import _prevalidated_sample_categorical

    items = ["A", "B", "C"]
    cdf = (0.6, 0.9, 1.0)
    rng = create_rng(999)

    n = 20_000
    counts = Counter(_prevalidated_sample_categorical(rng, items, cdf) for _ in range(n))

    assert np.isclose(counts["A"] / n, 0.60, atol=0.015)
    assert np.isclose(counts["B"] / n, 0.30, atol=0.015)
    assert np.isclose(counts["C"] / n, 0.10, atol=0.015)

