"""Unit tests for BehaviorSim calibration data model and input validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from behaviorsim.calibration import (
    CalibrationData,
    extract_state_proxy,
    fit_distribution,
    fit_profile,
    fit_transition_matrix,
    validate_calibration_data,
)
from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.simulator import Simulator
from behaviorsim.core.state import State


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_valid_df() -> pd.DataFrame:
    """Fixture providing a clean behavioral DataFrame."""
    return pd.DataFrame(
        {
            "sequence_id": [1, 1, 1, 2, 2, 2],
            "step": [0, 1, 2, 0, 1, 2],
            "state": ["idle", "active", "active", "idle", "idle", "active"],
            "profile": ["beginner", "beginner", "beginner", "expert", "expert", "expert"],
            "reaction_time": [1.2, 0.8, 0.9, 0.5, 0.4, 0.6],
            "clicks": [2, 5, 4, 10, 8, 12],
            "action_type": ["click", "submit", "click", "skip", "click", "submit"],
        }
    )


# ---------------------------------------------------------------------------
# Positive Tests: Valid Instantiation and Accessors
# ---------------------------------------------------------------------------

def test_calibration_data_minimal_instantiation(sample_valid_df: pd.DataFrame) -> None:
    """Verify minimal valid instantiation with defaults."""
    calib = CalibrationData(
        data=sample_valid_df,
        states=["idle", "active"],
    )
    assert calib.num_records == 6
    assert calib.num_sequences == 2
    assert calib.state_names == ["idle", "active"]
    assert calib.state_column == "state"
    assert calib.sequence_column == "sequence_id"


def test_calibration_data_full_instantiation(sample_valid_df: pd.DataFrame) -> None:
    """Verify instantiation with all optional parameters provided."""
    calib = CalibrationData(
        data=sample_valid_df,
        states=["idle", "active", "dormant"],  # dormant not observed, but declared
        state_column="state",
        feature_columns=["reaction_time", "clicks", "action_type"],
        sequence_column="sequence_id",
        profile_column="profile",
        numeric_features=["reaction_time", "clicks"],
        categorical_features=["action_type"],
        metadata={"source": "empirical_pilot_study", "version": 1},
    )
    assert calib.num_records == 6
    assert calib.num_sequences == 2
    assert calib.state_names == ["idle", "active", "dormant"]
    assert calib.metadata == {"source": "empirical_pilot_study", "version": 1}

    # State data extraction
    idle_df = calib.get_state_data("idle")
    assert len(idle_df) == 3
    assert set(idle_df["state"].unique()) == {"idle"}

    dormant_df = calib.get_state_data("dormant")
    assert len(dormant_df) == 0

    # Profile data extraction
    beg_df = calib.get_profile_data("beginner")
    assert len(beg_df) == 3
    assert (beg_df["profile"] == "beginner").all()


def test_calibration_data_no_sequence_column(sample_valid_df: pd.DataFrame) -> None:
    """Verify sequence_column can be None, in which case num_sequences is 1."""
    calib = CalibrationData(
        data=sample_valid_df,
        states=["idle", "active"],
        sequence_column=None,
    )
    assert calib.sequence_column is None
    assert calib.num_sequences == 1


# ---------------------------------------------------------------------------
# Negative Tests: Structural & Type Validation
# ---------------------------------------------------------------------------

def test_non_dataframe_raises_type_error() -> None:
    """Verify non-DataFrame data input raises TypeError."""
    with pytest.raises(TypeError, match="Calibration data must be a pandas DataFrame"):
        CalibrationData(data=[{"state": "idle"}], states=["idle"])  # type: ignore


def test_empty_dataframe_raises_value_error() -> None:
    """Verify empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame(columns=["state", "sequence_id"])
    with pytest.raises(ValueError, match="Calibration DataFrame cannot be empty"):
        CalibrationData(data=empty_df, states=["idle"])


def test_invalid_declared_states(sample_valid_df: pd.DataFrame) -> None:
    """Verify invalid states sequence raises ValueError."""
    with pytest.raises(ValueError, match="Declared states must be a non-empty sequence"):
        CalibrationData(data=sample_valid_df, states=[])

    with pytest.raises(ValueError, match="Declared states must be a non-empty sequence"):
        CalibrationData(data=sample_valid_df, states="idle")  # type: ignore

    with pytest.raises(ValueError, match="State at index 0 must be a non-empty string"):
        CalibrationData(data=sample_valid_df, states=[""])

    with pytest.raises(ValueError, match="Duplicate state name 'idle'"):
        CalibrationData(data=sample_valid_df, states=["idle", "active", "idle"])


# ---------------------------------------------------------------------------
# Negative Tests: State Column & Unknown / NaN Labels
# ---------------------------------------------------------------------------

def test_missing_state_column(sample_valid_df: pd.DataFrame) -> None:
    """Verify missing state column raises ValueError."""
    with pytest.raises(ValueError, match="State column 'nonexistent' not found in DataFrame"):
        CalibrationData(data=sample_valid_df, states=["idle"], state_column="nonexistent")


def test_unknown_state_labels(sample_valid_df: pd.DataFrame) -> None:
    """Verify unknown observed state raises ValueError listing undeclared states."""
    # sample_valid_df has states 'idle' and 'active', but only 'idle' is declared
    with pytest.raises(ValueError, match="Unknown state label.*active"):
        CalibrationData(data=sample_valid_df, states=["idle"])


def test_null_state_labels(sample_valid_df: pd.DataFrame) -> None:
    """Verify null/NaN values in state column raise ValueError."""
    df_with_nan = sample_valid_df.copy()
    df_with_nan.loc[2, "state"] = None
    with pytest.raises(ValueError, match="State column 'state' contains 1 null/NaN values"):
        CalibrationData(data=df_with_nan, states=["idle", "active"])


# ---------------------------------------------------------------------------
# Negative Tests: Sequence and Profile Columns
# ---------------------------------------------------------------------------

def test_missing_sequence_column(sample_valid_df: pd.DataFrame) -> None:
    """Verify missing sequence column raises ValueError."""
    with pytest.raises(ValueError, match="Sequence column 'missing_seq' not found"):
        CalibrationData(
            data=sample_valid_df,
            states=["idle", "active"],
            sequence_column="missing_seq",
        )


def test_null_sequence_column(sample_valid_df: pd.DataFrame) -> None:
    """Verify null/NaN sequence identifiers raise ValueError."""
    df_with_null = sample_valid_df.copy()
    df_with_null.loc[1, "sequence_id"] = None
    with pytest.raises(ValueError, match="Sequence column 'sequence_id' contains 1 null/NaN"):
        CalibrationData(data=df_with_null, states=["idle", "active"])


def test_missing_profile_column(sample_valid_df: pd.DataFrame) -> None:
    """Verify missing profile column raises ValueError."""
    with pytest.raises(ValueError, match="Profile column 'missing_prof' not found"):
        CalibrationData(
            data=sample_valid_df,
            states=["idle", "active"],
            profile_column="missing_prof",
        )


def test_null_profile_column(sample_valid_df: pd.DataFrame) -> None:
    """Verify null/NaN in profile column raises ValueError."""
    df_with_null = sample_valid_df.copy()
    df_with_null.loc[0, "profile"] = None
    with pytest.raises(ValueError, match="Profile column 'profile' contains 1 null/NaN"):
        CalibrationData(
            data=df_with_null,
            states=["idle", "active"],
            profile_column="profile",
        )


# ---------------------------------------------------------------------------
# Negative Tests: Feature Columns & Numeric / Categorical Validation
# ---------------------------------------------------------------------------

def test_missing_requested_feature_columns(sample_valid_df: pd.DataFrame) -> None:
    """Verify missing requested feature column raises ValueError."""
    with pytest.raises(ValueError, match="Requested feature column 'score' not found"):
        CalibrationData(
            data=sample_valid_df,
            states=["idle", "active"],
            feature_columns=["reaction_time", "score"],
        )


def test_non_numeric_dtype_in_numeric_features(sample_valid_df: pd.DataFrame) -> None:
    """Verify string/object column passed as numeric_feature raises TypeError."""
    with pytest.raises(TypeError, match="Feature column 'action_type' must have a numeric dtype"):
        CalibrationData(
            data=sample_valid_df,
            states=["idle", "active"],
            numeric_features=["action_type"],
        )


def test_bool_dtype_in_numeric_features(sample_valid_df: pd.DataFrame) -> None:
    """Verify boolean dtype in numeric_features raises TypeError."""
    df_bool = sample_valid_df.copy()
    df_bool["flag"] = [True, False, True, True, False, False]
    with pytest.raises(TypeError, match="Feature column 'flag' must have a numeric dtype"):
        CalibrationData(
            data=df_bool,
            states=["idle", "active"],
            numeric_features=["flag"],
        )


def test_nan_in_numeric_features(sample_valid_df: pd.DataFrame) -> None:
    """Verify NaN values in numeric features raise ValueError."""
    df_nan = sample_valid_df.copy()
    df_nan.loc[3, "reaction_time"] = np.nan
    with pytest.raises(ValueError, match="Numeric feature column 'reaction_time' contains 1 non-finite"):
        CalibrationData(
            data=df_nan,
            states=["idle", "active"],
            numeric_features=["reaction_time"],
        )


def test_inf_in_numeric_features(sample_valid_df: pd.DataFrame) -> None:
    """Verify +inf and -inf values in numeric features raise ValueError."""
    df_inf = sample_valid_df.copy()
    df_inf.loc[1, "reaction_time"] = np.inf
    with pytest.raises(ValueError, match="Numeric feature column 'reaction_time' contains 1 non-finite"):
        CalibrationData(
            data=df_inf,
            states=["idle", "active"],
            numeric_features=["reaction_time"],
        )

    df_neginf = sample_valid_df.copy()
    df_neginf.loc[4, "reaction_time"] = -np.inf
    with pytest.raises(ValueError, match="Numeric feature column 'reaction_time' contains 1 non-finite"):
        CalibrationData(
            data=df_neginf,
            states=["idle", "active"],
            numeric_features=["reaction_time"],
        )


def test_nan_in_categorical_features(sample_valid_df: pd.DataFrame) -> None:
    """Verify NaN values in categorical features raise ValueError."""
    df_cat_nan = sample_valid_df.copy()
    df_cat_nan.loc[2, "action_type"] = None
    with pytest.raises(ValueError, match="Categorical feature column 'action_type' contains 1 null/NaN"):
        CalibrationData(
            data=df_cat_nan,
            states=["idle", "active"],
            categorical_features=["action_type"],
        )


def test_missing_categorical_feature_column(sample_valid_df: pd.DataFrame) -> None:
    """Verify non-existent categorical column raises ValueError."""
    with pytest.raises(ValueError, match="Categorical feature column 'missing_cat' not found"):
        CalibrationData(
            data=sample_valid_df,
            states=["idle", "active"],
            categorical_features=["missing_cat"],
        )


def test_invalid_metadata_type(sample_valid_df: pd.DataFrame) -> None:
    """Verify non-mapping metadata raises TypeError."""
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        CalibrationData(
            data=sample_valid_df,
            states=["idle", "active"],
            metadata="not_a_dict",  # type: ignore
        )


# ---------------------------------------------------------------------------
# Negative Tests: Query and Filtering Helpers
# ---------------------------------------------------------------------------

def test_get_state_data_undeclared_state(sample_valid_df: pd.DataFrame) -> None:
    """Verify querying an undeclared state raises ValueError."""
    calib = CalibrationData(data=sample_valid_df, states=["idle", "active"])
    with pytest.raises(ValueError, match="State 'unknown' is not among declared states"):
        calib.get_state_data("unknown")


def test_get_profile_data_without_profile_column(sample_valid_df: pd.DataFrame) -> None:
    """Verify get_profile_data raises ValueError when profile_column is None."""
    calib = CalibrationData(data=sample_valid_df, states=["idle", "active"], profile_column=None)
    with pytest.raises(ValueError, match="Cannot filter by profile when profile_column is None"):
        calib.get_profile_data("beginner")


def test_get_profile_data_unknown_profile(sample_valid_df: pd.DataFrame) -> None:
    """Verify get_profile_data raises ValueError for unobserved profile."""
    calib = CalibrationData(
        data=sample_valid_df,
        states=["idle", "active"],
        profile_column="profile",
    )
    with pytest.raises(ValueError, match="No observations found for profile 'grandmaster'"):
        calib.get_profile_data("grandmaster")


# ---------------------------------------------------------------------------
# Phase 3.14 Hardening Tests: Immutability, Slice Copy, and Declaration Integrity
# ---------------------------------------------------------------------------

def test_dataframe_mutation_does_not_affect_calibration_data(sample_valid_df: pd.DataFrame) -> None:
    """Verify mutating the original DataFrame does not mutate CalibrationData."""
    df_copy = sample_valid_df.copy()
    calib = CalibrationData(data=df_copy, states=["idle", "active"])

    original_val = calib.data.loc[0, "reaction_time"]
    df_copy.loc[0, "reaction_time"] = 999.99

    assert calib.data.loc[0, "reaction_time"] == original_val
    assert calib.data.loc[0, "reaction_time"] != 999.99


def test_states_mutation_does_not_affect_calibration_data(sample_valid_df: pd.DataFrame) -> None:
    """Verify mutating the caller's states list does not mutate CalibrationData."""
    states_list = ["idle", "active"]
    calib = CalibrationData(data=sample_valid_df, states=states_list)

    states_list.append("rogue_state")
    assert calib.states == ("idle", "active")
    assert "rogue_state" not in calib.states
    assert isinstance(calib.states, tuple)


def test_feature_declarations_mutation_does_not_affect_calibration_data(sample_valid_df: pd.DataFrame) -> None:
    """Verify mutating feature declaration lists does not mutate CalibrationData."""
    feats = ["reaction_time", "clicks"]
    nums = ["reaction_time"]
    cats = ["action_type"]

    calib = CalibrationData(
        data=sample_valid_df,
        states=["idle", "active"],
        feature_columns=feats,
        numeric_features=nums,
        categorical_features=cats,
    )

    feats.append("new_feat")
    nums.append("new_num")
    cats.append("new_cat")

    assert calib.feature_columns == ("reaction_time", "clicks")
    assert calib.numeric_features == ("reaction_time",)
    assert calib.categorical_features == ("action_type",)
    assert isinstance(calib.feature_columns, tuple)
    assert isinstance(calib.numeric_features, tuple)
    assert isinstance(calib.categorical_features, tuple)


def test_metadata_mutation_does_not_affect_calibration_data(sample_valid_df: pd.DataFrame) -> None:
    """Verify mutating caller's metadata mapping does not mutate CalibrationData."""
    meta = {"experiment": "run_1", "notes": "initial"}
    calib = CalibrationData(data=sample_valid_df, states=["idle", "active"], metadata=meta)

    meta["notes"] = "tampered"
    meta["extra"] = 123

    assert calib.metadata == {"experiment": "run_1", "notes": "initial"}


def test_returned_state_and_profile_data_are_safe_copies(sample_valid_df: pd.DataFrame) -> None:
    """Verify mutating returned state or profile slices does not alter internal data."""
    calib = CalibrationData(
        data=sample_valid_df,
        states=["idle", "active"],
        profile_column="profile",
    )

    # Mutate state slice
    state_slice = calib.get_state_data("idle")
    state_slice.loc[state_slice.index[0], "reaction_time"] = -888.88
    assert calib.data.loc[0, "reaction_time"] != -888.88
    fresh_slice = calib.get_state_data("idle")
    assert fresh_slice.loc[fresh_slice.index[0], "reaction_time"] != -888.88

    # Mutate profile slice
    profile_slice = calib.get_profile_data("beginner")
    profile_slice.loc[profile_slice.index[0], "profile"] = "corrupted"
    assert calib.data.loc[0, "profile"] == "beginner"
    fresh_profile = calib.get_profile_data("beginner")
    assert fresh_profile.loc[fresh_profile.index[0], "profile"] == "beginner"


def test_duplicate_feature_columns_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verify duplicate feature names in feature_columns are rejected."""
    with pytest.raises(ValueError, match="Duplicate feature column 'reaction_time'"):
        CalibrationData(
            data=sample_valid_df,
            states=["idle", "active"],
            feature_columns=["reaction_time", "clicks", "reaction_time"],
        )


def test_duplicate_numeric_features_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verify duplicate names in numeric_features are rejected."""
    with pytest.raises(ValueError, match="Duplicate numeric feature 'clicks'"):
        CalibrationData(
            data=sample_valid_df,
            states=["idle", "active"],
            numeric_features=["clicks", "clicks"],
        )


def test_duplicate_categorical_features_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verify duplicate names in categorical_features are rejected."""
    with pytest.raises(ValueError, match="Duplicate categorical feature 'action_type'"):
        CalibrationData(
            data=sample_valid_df,
            states=["idle", "active"],
            categorical_features=["action_type", "action_type"],
        )


def test_numeric_and_categorical_overlap_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verify features declared as both numeric and categorical are rejected."""
    with pytest.raises(ValueError, match="Feature\\(s\\) cannot be declared as both numeric and categorical.*reaction_time"):
        CalibrationData(
            data=sample_valid_df,
            states=["idle", "active"],
            numeric_features=["reaction_time"],
            categorical_features=["reaction_time"],
        )


# ---------------------------------------------------------------------------
# Phase 3.15 State Proxy Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_existing_state_column(sample_valid_df: pd.DataFrame) -> None:
    """Verify extracting an existing state column matches the column values."""
    extracted = extract_state_proxy(
        data=sample_valid_df,
        proxy="state",
        states=["idle", "active"],
    )
    assert isinstance(extracted, pd.Series)
    assert extracted.tolist() == sample_valid_df["state"].tolist()
    assert extracted.index.equals(sample_valid_df.index)


def test_deterministic_callable_proxy_row_level(sample_valid_df: pd.DataFrame) -> None:
    """Verify deterministic row-level callable proxy accurately extracts states."""
    proxy_fn = lambda row: "active" if row["reaction_time"] < 0.8 else "idle"
    extracted = extract_state_proxy(
        data=sample_valid_df,
        proxy=proxy_fn,
        states=["idle", "active"],
    )
    # reaction_time: [1.2, 0.8, 0.9, 0.5, 0.4, 0.6] -> [idle, idle, idle, active, active, active]
    expected = ["idle", "idle", "idle", "active", "active", "active"]
    assert extracted.tolist() == expected


def test_deterministic_callable_proxy_dataframe_level(sample_valid_df: pd.DataFrame) -> None:
    """Verify deterministic DataFrame-level callable proxy accurately extracts states."""
    proxy_fn = lambda df: pd.Series(
        ["active" if rt < 0.8 else "idle" for rt in df["reaction_time"]],
        index=df.index,
    )
    extracted = extract_state_proxy(
        data=sample_valid_df,
        proxy=proxy_fn,
        states=["idle", "active"],
    )
    expected = ["idle", "idle", "idle", "active", "active", "active"]
    assert extracted.tolist() == expected


def test_deterministic_mapping_proxy(sample_valid_df: pd.DataFrame) -> None:
    """Verify deterministic Mapping proxy with source_column extracts states."""
    action_map = {"click": "active", "submit": "active", "skip": "idle"}
    extracted = extract_state_proxy(
        data=sample_valid_df,
        proxy=action_map,
        states=["idle", "active"],
        source_column="action_type",
    )
    # action_type: ["click", "submit", "click", "skip", "click", "submit"] -> [active, active, active, idle, active, active]
    expected = ["active", "active", "active", "idle", "active", "active"]
    assert extracted.tolist() == expected


def test_mapping_proxy_missing_source_column_raises_value_error(sample_valid_df: pd.DataFrame) -> None:
    """Verify Mapping proxy without source_column raises ValueError."""
    action_map = {"click": "active"}
    with pytest.raises(ValueError, match="source_column must be specified when proxy is a Mapping"):
        extract_state_proxy(
            data=sample_valid_df,
            proxy=action_map,
            states=["idle", "active"],
        )


def test_state_ordering_preservation_in_proxy(sample_valid_df: pd.DataFrame) -> None:
    """Verify state ordering defined in states is preserved through proxy extraction."""
    # Custom ordering: active first, idle second
    states_order = ["active", "idle"]
    extracted = extract_state_proxy(
        data=sample_valid_df,
        proxy="state",
        states=states_order,
    )
    assert set(extracted.unique()).issubset(set(states_order))


def test_unknown_proxy_state_rejection(sample_valid_df: pd.DataFrame) -> None:
    """Verify proxy producing undeclared state raises ValueError."""
    rogue_proxy = lambda row: "supercharged"
    with pytest.raises(ValueError, match="Unknown state label\\(s\\) produced by proxy: \\['supercharged'\\]"):
        extract_state_proxy(
            data=sample_valid_df,
            proxy=rogue_proxy,
            states=["idle", "active"],
        )


def test_null_proxy_output_rejection(sample_valid_df: pd.DataFrame) -> None:
    """Verify proxy producing null/NaN values raises ValueError."""
    incomplete_map = {"click": "active"}  # "submit" and "skip" will map to NaN
    with pytest.raises(ValueError, match="State proxy produced .* null/NaN values"):
        extract_state_proxy(
            data=sample_valid_df,
            proxy=incomplete_map,
            states=["idle", "active"],
            source_column="action_type",
        )


def test_mismatched_proxy_length_rejection(sample_valid_df: pd.DataFrame) -> None:
    """Verify proxy returning truncated output raises ValueError."""
    short_proxy = lambda df: ["idle"] * 2  # len 2 != len(data) which is 6
    with pytest.raises(ValueError, match="Proxy output length \\(2\\) does not match DataFrame length \\(6\\)"):
        extract_state_proxy(
            data=sample_valid_df,
            proxy=short_proxy,
            states=["idle", "active"],
        )


def test_mismatched_proxy_index_rejection(sample_valid_df: pd.DataFrame) -> None:
    """Verify proxy returning mismatched Series index raises ValueError."""
    bad_idx_proxy = lambda df: pd.Series(["idle"] * len(df), index=range(100, 100 + len(df)))
    with pytest.raises(ValueError, match="Proxy Series index does not align with DataFrame index"):
        extract_state_proxy(
            data=sample_valid_df,
            proxy=bad_idx_proxy,
            states=["idle", "active"],
        )


def test_invalid_proxy_type_rejection(sample_valid_df: pd.DataFrame) -> None:
    """Verify passing unsupported proxy type raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported proxy type: int. Expected str, Mapping, or callable"):
        extract_state_proxy(
            data=sample_valid_df,
            proxy=12345,  # type: ignore
            states=["idle", "active"],
        )


def test_empty_dataframe_proxy_rejection() -> None:
    """Verify passing empty DataFrame to extract_state_proxy raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
        extract_state_proxy(
            data=empty_df,
            proxy="state",
            states=["idle", "active"],
        )


def test_sequence_and_row_order_preservation(sample_valid_df: pd.DataFrame) -> None:
    """Verify row ordering and sequence partitions are preserved exactly."""
    proxy_fn = lambda row: "active" if row["reaction_time"] < 0.8 else "idle"
    extracted = extract_state_proxy(
        data=sample_valid_df,
        proxy=proxy_fn,
        states=["idle", "active"],
    )

    # Check 1:1 index alignment
    assert extracted.index.tolist() == sample_valid_df.index.tolist()

    # Check sequences alignment
    for seq_id in sample_valid_df["sequence_id"].unique():
        seq_mask = sample_valid_df["sequence_id"] == seq_id
        assert len(extracted[seq_mask]) == len(sample_valid_df[seq_mask])


def test_integration_with_calibration_data_from_proxy(sample_valid_df: pd.DataFrame) -> None:
    """Verify constructing CalibrationData using from_proxy classmethod."""
    proxy_fn = lambda row: "active" if row["reaction_time"] < 0.8 else "idle"
    calib = CalibrationData.from_proxy(
        data=sample_valid_df,
        states=["idle", "active"],
        proxy=proxy_fn,
        state_column="extracted_state",
        sequence_column="sequence_id",
        profile_column="profile",
        numeric_features=["reaction_time", "clicks"],
    )

    assert isinstance(calib, CalibrationData)
    assert calib.state_column == "extracted_state"
    assert "extracted_state" in calib.data.columns
    assert calib.num_records == 6
    assert calib.num_sequences == 2

    # Verify slicing works smoothly on the extracted state
    active_df = calib.get_state_data("active")
    assert len(active_df) == 3
    assert (active_df["extracted_state"] == "active").all()


def test_deterministic_repeated_extraction(sample_valid_df: pd.DataFrame) -> None:
    """Verify repeated extraction on identical data produces identical Series."""
    proxy_fn = lambda row: "active" if row["reaction_time"] < 0.8 else "idle"
    res1 = extract_state_proxy(sample_valid_df, proxy_fn, ["idle", "active"])
    res2 = extract_state_proxy(sample_valid_df, proxy_fn, ["idle", "active"])

    assert res1.equals(res2)


def test_callable_mode_explicit_row(sample_valid_df: pd.DataFrame) -> None:
    """Verify callable_mode='row' explicitly executes row-level callable."""
    proxy_fn = lambda row: "active" if row["reaction_time"] < 0.8 else "idle"
    res = extract_state_proxy(
        sample_valid_df, proxy_fn, ["idle", "active"], callable_mode="row"
    )
    assert len(res) == len(sample_valid_df)
    assert set(res.unique()).issubset({"idle", "active"})


def test_callable_mode_explicit_dataframe(sample_valid_df: pd.DataFrame) -> None:
    """Verify callable_mode='dataframe' executes DataFrame-level vectorized function."""
    proxy_fn = lambda df: pd.Series(
        np.where(df["reaction_time"] < 0.8, "active", "idle"),
        index=df.index,
    )
    res = extract_state_proxy(
        sample_valid_df, proxy_fn, ["idle", "active"], callable_mode="dataframe"
    )
    assert len(res) == len(sample_valid_df)
    assert set(res.unique()).issubset({"idle", "active"})


def test_callable_mode_invalid_mode_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verify invalid callable_mode raises ValueError."""
    proxy_fn = lambda row: "idle"
    with pytest.raises(ValueError, match="callable_mode must be 'auto', 'row', or 'dataframe'"):
        extract_state_proxy(
            sample_valid_df, proxy_fn, ["idle", "active"], callable_mode="invalid"  # type: ignore
        )


def test_callable_auto_mode_does_not_mask_dataframe_user_exception(sample_valid_df: pd.DataFrame) -> None:
    """Verify callable errors in DataFrame functions raise directly and are not silently masked."""
    def buggy_df_proxy(df: pd.DataFrame) -> pd.Series:
        return df["non_existent_column_for_bug"]  # Raises KeyError

    with pytest.raises(KeyError):
        extract_state_proxy(
            sample_valid_df, buggy_df_proxy, ["idle", "active"], callable_mode="auto"
        )



# ---------------------------------------------------------------------------
# Phase 3.16 Transition-Matrix Fitting Tests
# ---------------------------------------------------------------------------

def test_fit_transition_matrix_known_hand_calculated() -> None:
    """Verify transition matrix fitting on a known hand-calculated sequence trace."""
    # Sequence 1: S0 -> S1 -> S0
    # Sequence 2: S1 -> S1 -> S0
    df = pd.DataFrame(
        {
            "sequence_id": [1, 1, 1, 2, 2, 2],
            "state": ["S0", "S1", "S0", "S1", "S1", "S0"],
        }
    )
    calib = CalibrationData(data=df, states=["S0", "S1"])
    matrix = fit_transition_matrix(calib, smoothing=0.0)

    # Expected:
    # S0 outgoing transitions: S0->S1 (1 time from seq 1 step 0) -> [0.0, 1.0]
    # S1 outgoing transitions:
    #   S1->S0 (seq 1 step 1, seq 2 step 1) = 2
    #   S1->S1 (seq 2 step 0) = 1
    #   Total S1 outgoing = 3 -> [2/3, 1/3]
    expected = np.array([[0.0, 1.0], [2.0 / 3.0, 1.0 / 3.0]])
    np.testing.assert_allclose(matrix, expected, atol=1e-8)


def test_fit_transition_matrix_sequence_isolation() -> None:
    """Verify transitions are strictly isolated within sequences (no cross-sequence boundary transitions)."""
    # Seq 1 ends with S0, Seq 2 starts with S1.
    # If boundary leaked, we would see an extra S0 -> S1 transition.
    df = pd.DataFrame(
        {
            "sequence_id": [1, 1, 2, 2],
            "state": ["S0", "S0", "S1", "S1"],
        }
    )
    calib = CalibrationData(data=df, states=["S0", "S1"])
    matrix = fit_transition_matrix(calib, smoothing=0.0)

    # S0 only transitions to S0 within seq 1: [1.0, 0.0]
    # S1 only transitions to S1 within seq 2: [0.0, 1.0]
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(matrix, expected, atol=1e-8)


def test_fit_transition_matrix_state_ordering() -> None:
    """Verify matrix rows and columns strictly follow data.states ordering."""
    df = pd.DataFrame(
        {
            "sequence_id": [1, 1],
            "state": ["S0", "S1"],
        }
    )
    # Order 1: S0, S1
    calib1 = CalibrationData(data=df, states=["S0", "S1"])
    mat1 = fit_transition_matrix(calib1)
    # S0 -> S1 is index (0, 1)
    assert mat1[0, 1] == 1.0

    # Order 2: S1, S0
    calib2 = CalibrationData(data=df, states=["S1", "S0"])
    mat2 = fit_transition_matrix(calib2)
    # S0 -> S1 is index (1, 0)
    assert mat2[1, 0] == 1.0


def test_fit_transition_matrix_unobserved_state() -> None:
    """Verify unobserved declared states receive deterministic self-loop = 1.0."""
    df = pd.DataFrame(
        {
            "sequence_id": [1, 1],
            "state": ["active", "active"],
        }
    )
    calib = CalibrationData(data=df, states=["active", "dormant"])
    matrix = fit_transition_matrix(calib, smoothing=0.0)

    # active -> active: [1.0, 0.0]
    # dormant has zero observations -> self-loop [0.0, 1.0]
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(matrix, expected, atol=1e-8)


def test_fit_transition_matrix_zero_outgoing_transitions() -> None:
    """Verify states that only appear as absorbing/terminal states receive deterministic self-loop."""
    df = pd.DataFrame(
        {
            "sequence_id": [1, 1, 2, 2],
            "state": ["idle", "active", "idle", "active"],
        }
    )
    calib = CalibrationData(data=df, states=["idle", "active"])
    matrix = fit_transition_matrix(calib, smoothing=0.0)

    # idle transitions to active: [0.0, 1.0]
    # active is never an origin (zero outgoing): self-loop [0.0, 1.0]
    expected = np.array([[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(matrix, expected, atol=1e-8)


def test_fit_transition_matrix_positive_smoothing() -> None:
    """Verify additive Laplace smoothing produces valid stochastic probabilities."""
    df = pd.DataFrame(
        {
            "sequence_id": [1, 1],
            "state": ["S0", "S1"],
        }
    )
    calib = CalibrationData(data=df, states=["S0", "S1"])
    # With smoothing = 1.0:
    # S0 has 1 transition to S1.
    # C(S0->S0) = 0 + 1 = 1, C(S0->S1) = 1 + 1 = 2. Denom = 1 + 2*1 = 3.
    # Row S0: [1/3, 2/3]
    # S1 has 0 transitions.
    # Smoothed zero obs: alpha / (N * alpha) = 1/2 for each.
    # Row S1: [0.5, 0.5]
    matrix = fit_transition_matrix(calib, smoothing=1.0)
    expected = np.array([[1.0 / 3.0, 2.0 / 3.0], [0.5, 0.5]])
    np.testing.assert_allclose(matrix, expected, atol=1e-8)


def test_fit_transition_matrix_negative_smoothing_rejected(sample_valid_df: pd.DataFrame) -> None:
    """Verify negative smoothing parameter raises ValueError."""
    calib = CalibrationData(data=sample_valid_df, states=["idle", "active"])
    with pytest.raises(ValueError, match="smoothing must be a non-negative finite number"):
        fit_transition_matrix(calib, smoothing=-0.5)


def test_fit_transition_matrix_deterministic_repeated_fitting(sample_valid_df: pd.DataFrame) -> None:
    """Verify repeated fitting is purely deterministic."""
    calib = CalibrationData(data=sample_valid_df, states=["idle", "active"])
    m1 = fit_transition_matrix(calib, smoothing=0.1)
    m2 = fit_transition_matrix(calib, smoothing=0.1)
    assert np.array_equal(m1, m2)


def test_fit_transition_matrix_compatibility_with_profile(sample_valid_df: pd.DataFrame) -> None:
    """Verify fitted matrix is accepted by core Profile validation."""
    calib = CalibrationData(data=sample_valid_df, states=["idle", "active"])
    matrix = fit_transition_matrix(calib)

    prof = Profile(
        name="test_fitted",
        state_emissions={
            "idle": {"rt": FeatureDistribution("normal", {"loc": 1.0, "scale": 0.2})},
            "active": {"rt": FeatureDistribution("normal", {"loc": 0.5, "scale": 0.1})},
        },
        transition_matrix=matrix,
    )
    assert prof.transition_matrix is not None
    assert prof.transition_matrix.shape == (2, 2)


# ---------------------------------------------------------------------------
# Phase 3.17 Emission MLE Fitting Tests
# ---------------------------------------------------------------------------

def test_fit_distribution_normal() -> None:
    """Verify normal distribution MLE fit."""
    vals = [10.0, 20.0, 30.0]
    dist = fit_distribution(vals, "normal")
    assert dist.distribution_type == "normal"
    assert dist.params["loc"] == 20.0
    assert np.isclose(dist.params["scale"], np.std([10, 20, 30], ddof=0))


def test_fit_distribution_lognormal() -> None:
    """Verify lognormal distribution MLE fit."""
    vals = [np.exp(1.0), np.exp(2.0), np.exp(3.0)]
    dist = fit_distribution(vals, "lognormal")
    assert dist.distribution_type == "lognormal"
    assert np.isclose(dist.params["mean"], 2.0)
    assert np.isclose(dist.params["sigma"], np.std([1.0, 2.0, 3.0], ddof=0))


def test_fit_distribution_exponential() -> None:
    """Verify exponential distribution MLE fit."""
    vals = [2.0, 4.0, 6.0]
    dist = fit_distribution(vals, "exponential")
    assert dist.distribution_type == "exponential"
    assert dist.params["scale"] == 4.0


def test_fit_distribution_uniform() -> None:
    """Verify uniform distribution MLE fit."""
    vals = [5.0, 2.0, 8.0, 3.0]
    dist = fit_distribution(vals, "uniform")
    assert dist.distribution_type == "uniform"
    assert dist.params["low"] == 2.0
    assert dist.params["high"] == 8.0


def test_fit_distribution_uniform_discrete() -> None:
    """Verify uniform_discrete distribution MLE fit."""
    vals = [1, 2, 5, 3]
    dist = fit_distribution(vals, "uniform_discrete")
    assert dist.distribution_type == "uniform_discrete"
    assert dist.params["low"] == 1
    assert dist.params["high"] == 5


def test_fit_distribution_bernoulli() -> None:
    """Verify bernoulli distribution MLE fit."""
    vals = [1, 0, 1, 1]
    dist = fit_distribution(vals, "bernoulli")
    assert dist.distribution_type == "bernoulli"
    assert dist.params["p"] == 0.75


def test_fit_distribution_poisson() -> None:
    """Verify poisson distribution MLE fit."""
    vals = [2, 3, 5, 2]
    dist = fit_distribution(vals, "poisson")
    assert dist.distribution_type == "poisson"
    assert dist.params["lam"] == 3.0


def test_fit_distribution_categorical() -> None:
    """Verify categorical distribution MLE fit."""
    vals = ["a", "b", "a", "c"]
    dist = fit_distribution(vals, "categorical")
    assert dist.distribution_type == "categorical"
    assert dist.params["items"] == ["a", "b", "c"]
    assert np.allclose(dist.params["probabilities"], [0.5, 0.25, 0.25])


def test_fit_distribution_empty_input_rejected() -> None:
    """Verify empty input raises ValueError."""
    with pytest.raises(ValueError, match="Cannot fit distribution from empty"):
        fit_distribution([], "normal")


def test_fit_distribution_one_observation() -> None:
    """Verify fitting a single observation produces valid parameter values."""
    dist = fit_distribution([42.0], "normal")
    assert dist.params["loc"] == 42.0
    assert dist.params["scale"] == 0.0


def test_fit_distribution_constant_observations() -> None:
    """Verify constant observations yield zero variance without error."""
    dist = fit_distribution([5.0, 5.0, 5.0], "normal")
    assert dist.params["loc"] == 5.0
    assert dist.params["scale"] == 0.0

    uni = fit_distribution([5.0, 5.0], "uniform")
    assert uni.params["low"] == 5.0
    assert uni.params["high"] == 5.0


def test_fit_distribution_all_zero_poisson() -> None:
    """Verify all-zero Poisson data estimates lam=0.0."""
    dist = fit_distribution([0, 0, 0], "poisson")
    assert dist.params["lam"] == 0.0


def test_fit_distribution_all_zero_and_one_bernoulli() -> None:
    """Verify all-zero and all-one Bernoulli data estimates p=0.0 and p=1.0."""
    d0 = fit_distribution([0, 0], "bernoulli")
    assert d0.params["p"] == 0.0
    d1 = fit_distribution([1, 1], "bernoulli")
    assert d1.params["p"] == 1.0


def test_fit_distribution_invalid_negative_exponential() -> None:
    """Verify negative observations for exponential distribution raise ValueError."""
    with pytest.raises(ValueError, match="Exponential distribution requires non-negative"):
        fit_distribution([-1.0, 2.0], "exponential")


def test_fit_distribution_all_zero_exponential_rejected() -> None:
    """Verify all-zero observations for exponential distribution raise ValueError."""
    with pytest.raises(ValueError, match="Exponential distribution scale must be positive"):
        fit_distribution([0.0, 0.0], "exponential")


def test_fit_distribution_invalid_non_positive_lognormal() -> None:
    """Verify non-positive observations for lognormal distribution raise ValueError."""
    with pytest.raises(ValueError, match="Lognormal distribution requires strictly positive"):
        fit_distribution([0.0, 1.0, 2.0], "lognormal")


def test_fit_distribution_non_integer_poisson() -> None:
    """Verify non-integer observations for Poisson distribution raise ValueError."""
    with pytest.raises(ValueError, match="Poisson distribution requires non-negative integer"):
        fit_distribution([1.5, 2.0], "poisson")


def test_fit_distribution_non_integer_uniform_discrete() -> None:
    """Verify non-integer observations for uniform_discrete raise ValueError."""
    with pytest.raises(ValueError, match="uniform_discrete requires integer-valued"):
        fit_distribution([1.2, 3.4], "uniform_discrete")


def test_fit_distribution_invalid_bernoulli_values() -> None:
    """Verify non-binary observations for Bernoulli raise ValueError."""
    with pytest.raises(ValueError, match="Bernoulli distribution requires observations exclusively in \\{0, 1\\}"):
        fit_distribution([0, 1, 2], "bernoulli")


def test_fit_distribution_nan_infinite_values() -> None:
    """Verify NaN and infinite values raise ValueError."""
    with pytest.raises(ValueError, match="Observations contain non-finite"):
        fit_distribution([1.0, np.nan], "normal")
    with pytest.raises(ValueError, match="Observations contain non-finite"):
        fit_distribution([1.0, np.inf], "normal")


def test_fit_distribution_unsupported_distribution() -> None:
    """Verify unsupported distribution name raises ValueError."""
    with pytest.raises(ValueError, match="unsupported distribution type: 'cauchy'"):
        fit_distribution([1.0, 2.0], "cauchy")


# ---------------------------------------------------------------------------
# Phase 3.18 Profile Integration Tests
# ---------------------------------------------------------------------------

def test_fit_profile_end_to_end(sample_valid_df: pd.DataFrame) -> None:
    """Verify fitting a complete Profile from CalibrationData."""
    calib = CalibrationData(
        data=sample_valid_df,
        states=["idle", "active", "dormant"],
        numeric_features=["reaction_time", "clicks"],
        categorical_features=["action_type"],
    )

    profile = fit_profile(
        calib,
        name="calibrated_agent",
        distribution_types={
            "reaction_time": "normal",
            "clicks": "poisson",
            "action_type": "categorical",
        },
        metadata={"calibrated_by": "phase_3_18"},
    )

    assert isinstance(profile, Profile)
    assert profile.name == "calibrated_agent"
    assert profile.transition_matrix is not None
    assert profile.transition_matrix.shape == (3, 3)
    assert profile.metadata is not None
    assert profile.metadata["calibrated_by"] == "phase_3_18"
    assert profile.metadata["num_records"] == 6

    # Verify all states have feature distributions
    for s in ["idle", "active", "dormant"]:
        assert s in profile.state_emissions
        assert "reaction_time" in profile.state_emissions[s]
        assert "clicks" in profile.state_emissions[s]
        assert "action_type" in profile.state_emissions[s]
        assert isinstance(profile.state_emissions[s]["reaction_time"], FeatureDistribution)


def test_fit_profile_with_simulator_generate(sample_valid_df: pd.DataFrame) -> None:
    """Verify fitted Profile is directly usable by Simulator to generate synthetic traces."""
    calib = CalibrationData(
        data=sample_valid_df,
        states=["idle", "active"],
        numeric_features=["reaction_time", "clicks"],
        categorical_features=["action_type"],
    )

    profile = fit_profile(
        calib,
        name="sim_ready_profile",
        distribution_types={
            "reaction_time": "normal",
            "clicks": "poisson",
            "action_type": "categorical",
        },
    )

    states = [State("idle"), State("active")]
    sim = Simulator(states=states, profile=profile)
    synthetic_df = sim.generate(num_interactions=20, seed=123)

    assert len(synthetic_df) == 20
    assert "state" in synthetic_df.columns
    assert "reaction_time" in synthetic_df.columns
    assert "clicks" in synthetic_df.columns
    assert "action_type" in synthetic_df.columns
    assert set(synthetic_df["state"].unique()).issubset({"idle", "active"})
    assert set(synthetic_df["action_type"].unique()).issubset({"click", "submit", "skip"})


def test_fit_profile_unobserved_state_policy(sample_valid_df: pd.DataFrame) -> None:
    """Verify unobserved state policy deterministically falls back to global feature distributions."""
    calib = CalibrationData(
        data=sample_valid_df,
        states=["idle", "active", "unseen_state"],
        numeric_features=["reaction_time"],
    )

    profile = fit_profile(calib, name="unobserved_test")
    # unseen_state should have valid distributions matching global data
    assert "unseen_state" in profile.state_emissions
    dist = profile.state_emissions["unseen_state"]["reaction_time"]
    assert isinstance(dist, FeatureDistribution)
    # Mean should equal global mean of reaction_time
    global_mean = float(np.mean(sample_valid_df["reaction_time"]))
    assert np.isclose(dist.params["loc"], global_mean)
