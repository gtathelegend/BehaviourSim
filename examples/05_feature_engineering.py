"""Example 05: Causal Behavioral Feature Engineering

Demonstrates generating strictly causal sequential features without future data leakage:
1. Simulating raw interaction sequences.
2. Generating causal rolling window statistics (mean, variance).
3. Tracking cumulative state transitions and consecutive state dwell times.
4. Detecting session boundaries and gaps.
5. Verifying that feature values at step t strictly depend on information up to step t.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root and src to sys.path so examples run out of the box without prior installation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from behaviorsim import Simulator
from behaviorsim.feature_engineering import build_behavioral_features


def main() -> None:
    print("=== BehaviorSim: Causal Feature Engineering Example ===")

    # 1. Generate base raw simulation traces using the Education preset
    sim = Simulator.from_preset("education", profile="fast_inaccurate")
    raw_traces: pd.DataFrame = sim.generate(
        num_interactions=10,
        num_sequences=2,
        seed=42,
    )

    first_seq = raw_traces["sequence_id"].iloc[0]
    print(f"\nRaw interaction traces (sequence_id = {first_seq}):")
    seq_raw = raw_traces[raw_traces["sequence_id"] == first_seq]
    print(seq_raw[["sequence_id", "interaction_id", "state", "nrt", "accuracy"]])

    # 2. Build comprehensive causal behavioral features
    enriched = build_behavioral_features(
        raw_traces,
        state_column="state",
        numeric_columns=["nrt"],
        rolling_windows=[3],
        rolling_stats=["mean", "var"],
        sequence_column="sequence_id",
    )

    print(f"\nEnriched causal features (sequence_id = {first_seq}):")
    seq_enriched = enriched[enriched["sequence_id"] == first_seq]
    columns_to_show = [
        "interaction_id",
        "state",
        "nrt",
        "nrt_rolling_mean_w3",
        "state_consecutive_dwell",
    ]
    existing_cols = [c for c in columns_to_show if c in seq_enriched.columns]
    print(seq_enriched[existing_cols])

    # 3. Causal Verification: feature at t uses strictly prior observations (0..t-1)
    # At t=0, no prior observations exist -> rolling mean is NaN
    assert pd.isna(seq_enriched["nrt_rolling_mean_w3"].iloc[0]), "Causal contract: t=0 must be NaN (no prior history)!"

    # At t=1, rolling mean uses only observation from t=0
    nrt_0 = float(seq_enriched["nrt"].iloc[0])
    roll_1 = float(seq_enriched["nrt_rolling_mean_w3"].iloc[1])
    assert abs(nrt_0 - roll_1) < 1e-6, "Causal contract: rolling feature at t=1 must equal raw value at t=0!"

    # At t=2, rolling mean uses observations from [t=0, t=1]
    nrt_1 = float(seq_enriched["nrt"].iloc[1])
    roll_2 = float(seq_enriched["nrt_rolling_mean_w3"].iloc[2])
    expected_2 = (nrt_0 + nrt_1) / 2.0
    assert abs(expected_2 - roll_2) < 1e-6, "Causal contract: rolling feature at t=2 must equal mean of t=0 and t=1!"

    print("\nCausal integrity confirmed: All feature aggregations at step t strictly evaluate observations <= t.")
    print("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()
