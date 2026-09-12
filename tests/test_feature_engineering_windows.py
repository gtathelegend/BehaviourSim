"""Unit and causal correctness tests for BehaviorSim rolling and expanding features."""

import numpy as np
import pandas as pd
import pytest

from behaviorsim.feature_engineering.windows import (
    add_rolling_features,
    expanding_max,
    expanding_mean,
    expanding_min,
    expanding_sum,
    expanding_variance,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_sum,
    rolling_variance,
)


@pytest.fixture
def sample_numeric_sequences() -> pd.DataFrame:
    """Fixture providing 2 isolated sequences with numeric values."""
    return pd.DataFrame(
        {
            "sequence_id": [1, 1, 1, 1, 2, 2, 2],
            "step": [1, 2, 3, 4, 1, 2, 3],
            "val": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0],
        }
    )


def test_rolling_mean_causal_convention(sample_numeric_sequences: pd.DataFrame) -> None:
    """Verify rolling mean strictly excludes current row (closed='left')."""
    res = rolling_mean(
        sample_numeric_sequences,
        value_column="val",
        window=2,
        sequence_column="sequence_id",
        order_column="step",
    )

    # Sequence 1: [10, 20, 30, 40]
    # t=0: history [] -> NaN
    # t=1: history [10] -> 10.0
    # t=2: history [10, 20] -> 15.0
    # t=3: history [20, 30] -> 25.0
    # Sequence 2: [100, 200, 300]
    # t=0: history [] -> NaN
    # t=1: history [100] -> 100.0
    # t=2: history [100, 200] -> 150.0
    assert np.isnan(res.iloc[0])
    assert res.iloc[1] == 10.0
    assert res.iloc[2] == 15.0
    assert res.iloc[3] == 25.0
    assert np.isnan(res.iloc[4])
    assert res.iloc[5] == 100.0
    assert res.iloc[6] == 150.0


def test_rolling_variance_population_convention(sample_numeric_sequences: pd.DataFrame) -> None:
    """Verify rolling variance handles single previous observation as 0.0 with ddof=0."""
    res = rolling_variance(
        sample_numeric_sequences,
        value_column="val",
        window=2,
        sequence_column="sequence_id",
        ddof=0,
    )

    # t=0: NaN
    # t=1: history [10] -> var is 0.0
    # t=2: history [10, 20] -> var is 25.0
    assert np.isnan(res.iloc[0])
    assert res.iloc[1] == 0.0
    assert np.isclose(res.iloc[2], 25.0)


def test_rolling_sum_min_max(sample_numeric_sequences: pd.DataFrame) -> None:
    """Verify rolling sum, min, max causal conventions."""
    r_sum = rolling_sum(sample_numeric_sequences, "val", window=2, sequence_column="sequence_id")
    r_min = rolling_min(sample_numeric_sequences, "val", window=2, sequence_column="sequence_id")
    r_max = rolling_max(sample_numeric_sequences, "val", window=2, sequence_column="sequence_id")

    assert np.isnan(r_sum.iloc[0])
    assert r_sum.iloc[2] == 30.0  # 10 + 20
    assert r_min.iloc[2] == 10.0
    assert r_max.iloc[2] == 20.0


def test_expanding_statistics(sample_numeric_sequences: pd.DataFrame) -> None:
    """Verify expanding/cumulative mean, sum, variance, min, max."""
    e_mean = expanding_mean(sample_numeric_sequences, "val", sequence_column="sequence_id")
    e_sum = expanding_sum(sample_numeric_sequences, "val", sequence_column="sequence_id")
    e_var = expanding_variance(sample_numeric_sequences, "val", sequence_column="sequence_id", ddof=0)
    e_min = expanding_min(sample_numeric_sequences, "val", sequence_column="sequence_id")
    e_max = expanding_max(sample_numeric_sequences, "val", sequence_column="sequence_id")

    # Sequence 1, t=3 (row 3): history is [10, 20, 30]
    assert e_mean.iloc[3] == 20.0
    assert e_sum.iloc[3] == 60.0
    assert np.isclose(e_var.iloc[3], np.var([10, 20, 30], ddof=0))
    assert e_min.iloc[3] == 10.0
    assert e_max.iloc[3] == 30.0


def test_sequence_isolation_strict(sample_numeric_sequences: pd.DataFrame) -> None:
    """Verify Sequence 2 never receives history from Sequence 1."""
    res = rolling_mean(sample_numeric_sequences, "val", window=10, sequence_column="sequence_id")
    # Row 4 is first row of sequence 2. Must be NaN, not affected by seq 1 (which had values up to 40)
    assert np.isnan(res.iloc[4])
    # Row 5 must only see 100.0
    assert res.iloc[5] == 100.0


def test_no_future_leakage(sample_numeric_sequences: pd.DataFrame) -> None:
    """Hard requirement: altering x[t+1], x[t+2] never changes the feature at t."""
    df1 = sample_numeric_sequences.copy()
    res1 = rolling_mean(df1, "val", window=2, sequence_column="sequence_id")

    # Modify future rows 2 and 3 in sequence 1
    df2 = sample_numeric_sequences.copy()
    df2.loc[2, "val"] = 9999.0
    df2.loc[3, "val"] = 8888.0
    res2 = rolling_mean(df2, "val", window=2, sequence_column="sequence_id")

    # Past rows 0 and 1 must remain identical
    assert np.isnan(res1.iloc[0]) and np.isnan(res2.iloc[0])
    assert res1.iloc[1] == res2.iloc[1]


def test_window_larger_than_sequence() -> None:
    """Verify window larger than sequence aggregates all available historical rows."""
    df = pd.DataFrame({"seq": [1, 1], "val": [5.0, 15.0]})
    res = rolling_mean(df, "val", window=100, sequence_column="seq")
    assert np.isnan(res.iloc[0])
    assert res.iloc[1] == 5.0


def test_constant_values() -> None:
    """Verify rolling variance on constant values produces 0.0."""
    df = pd.DataFrame({"seq": [1, 1, 1], "val": [7.0, 7.0, 7.0]})
    res = rolling_variance(df, "val", window=2, sequence_column="seq", ddof=0)
    assert res.iloc[2] == 0.0


def test_invalid_arguments_rejection() -> None:
    """Verify non-positive windows and missing columns raise ValueError."""
    df = pd.DataFrame({"val": [1.0, 2.0]})
    with pytest.raises(ValueError, match="window must be a positive integer"):
        rolling_mean(df, "val", window=0)
    with pytest.raises(ValueError, match="window must be a positive integer"):
        rolling_mean(df, "val", window=-3)
    with pytest.raises(ValueError, match="value_column 'missing' not found"):
        rolling_mean(df, "missing", window=2)
    with pytest.raises(ValueError, match="sequence_column 'missing_seq' not found"):
        rolling_mean(df, "val", window=2, sequence_column="missing_seq")


def test_non_destructive_and_deterministic(sample_numeric_sequences: pd.DataFrame) -> None:
    """Verify original DataFrame is never modified and repeated calls yield identical results."""
    df_copy = sample_numeric_sequences.copy()
    r1 = rolling_mean(sample_numeric_sequences, "val", window=2, sequence_column="sequence_id")
    r2 = rolling_mean(sample_numeric_sequences, "val", window=2, sequence_column="sequence_id")

    pd.testing.assert_frame_equal(sample_numeric_sequences, df_copy)
    pd.testing.assert_series_equal(r1, r2)


def test_add_rolling_features_enrichment(sample_numeric_sequences: pd.DataFrame) -> None:
    """Verify add_rolling_features returns enriched copy without mutating original."""
    enriched = add_rolling_features(
        sample_numeric_sequences,
        value_columns=["val"],
        windows=[2, 3],
        stats=["mean", "var"],
        sequence_column="sequence_id",
    )
    assert "val_rolling_mean_w2" in enriched.columns
    assert "val_rolling_var_w2" in enriched.columns
    assert "val_rolling_mean_w3" in enriched.columns
    assert "val_rolling_var_w3" in enriched.columns
    assert "val_rolling_mean_w2" not in sample_numeric_sequences.columns
