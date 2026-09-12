"""Empirical parameter fitter and calibration data model for BehaviorSim."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, List, Mapping, Optional, Sequence, Set, Union
import numpy as np
import pandas as pd

from behaviorsim.config import SUPPORTED_DISTRIBUTIONS, validate_distribution_params
from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.utils import validate_transition_matrix


def validate_calibration_data(
    data: pd.DataFrame,
    states: Sequence[str],
    *,
    state_column: str = "state",
    feature_columns: Optional[Sequence[str]] = None,
    sequence_column: Optional[str] = "sequence_id",
    profile_column: Optional[str] = None,
    numeric_features: Optional[Sequence[str]] = None,
    categorical_features: Optional[Sequence[str]] = None,
) -> None:
    """Validate empirical observation dataset for BehaviorSim calibration.

    Args:
        data: Observed interactions or events as a pandas DataFrame.
        states: Declared sequence of valid state names.
        state_column: Column name in `data` representing the state label.
        feature_columns: Optional list of feature column names to validate.
        sequence_column: Optional column name partitioning independent sequences.
        profile_column: Optional column name indicating agent persona/profile cohort.
        numeric_features: Optional subset of feature columns that must contain finite numeric values.
        categorical_features: Optional subset of feature columns intended for discrete/categorical values.

    Raises:
        TypeError: If arguments are of improper types.
        ValueError: If data fails semantic or structural validation checks.
    """
    # 1. Type and structural checks on inputs
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Calibration data must be a pandas DataFrame, got {type(data).__name__}.")

    if data.empty:
        raise ValueError("Calibration DataFrame cannot be empty.")

    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)) or len(states) == 0:
        raise ValueError("Declared states must be a non-empty sequence of strings.")

    seen_states: Set[str] = set()
    for idx, s in enumerate(states):
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"State at index {idx} must be a non-empty string.")
        if s in seen_states:
            raise ValueError(f"Duplicate state name '{s}' in declared states.")
        seen_states.add(s)

    # 2. State column validation
    if not isinstance(state_column, str) or not state_column.strip():
        raise ValueError("state_column must be a non-empty string.")

    if state_column not in data.columns:
        raise ValueError(
            f"State column '{state_column}' not found in DataFrame. "
            f"Available columns: {list(data.columns)}."
        )

    if data[state_column].isna().any():
        null_count = int(data[state_column].isna().sum())
        raise ValueError(
            f"State column '{state_column}' contains {null_count} null/NaN values."
        )

    # Unknown state labels check
    observed_states = set(data[state_column].unique())
    unknown_states = observed_states - seen_states
    if unknown_states:
        raise ValueError(
            f"Unknown state label(s) observed in '{state_column}': {sorted(unknown_states)}. "
            f"Declared states: {sorted(seen_states)}."
        )

    # 3. Sequence column validation
    if sequence_column is not None:
        if not isinstance(sequence_column, str) or not sequence_column.strip():
            raise ValueError("sequence_column must be a non-empty string or None.")
        if sequence_column not in data.columns:
            raise ValueError(
                f"Sequence column '{sequence_column}' not found in DataFrame. "
                f"Available columns: {list(data.columns)}."
            )
        if data[sequence_column].isna().any():
            null_count = int(data[sequence_column].isna().sum())
            raise ValueError(
                f"Sequence column '{sequence_column}' contains {null_count} null/NaN values."
            )

    # 4. Profile column validation
    if profile_column is not None:
        if not isinstance(profile_column, str) or not profile_column.strip():
            raise ValueError("profile_column must be a non-empty string or None.")
        if profile_column not in data.columns:
            raise ValueError(
                f"Profile column '{profile_column}' not found in DataFrame. "
                f"Available columns: {list(data.columns)}."
            )
        if data[profile_column].isna().any():
            null_count = int(data[profile_column].isna().sum())
            raise ValueError(
                f"Profile column '{profile_column}' contains {null_count} null/NaN values."
            )

    # 5. Feature columns validation
    seen_features: Set[str] = set()
    resolved_features = list(feature_columns) if feature_columns is not None else []
    for f in resolved_features:
        if not isinstance(f, str) or not f.strip():
            raise ValueError("Feature column names must be non-empty strings.")
        if f in seen_features:
            raise ValueError(f"Duplicate feature column '{f}' in feature_columns.")
        seen_features.add(f)
        if f not in data.columns:
            raise ValueError(
                f"Requested feature column '{f}' not found in DataFrame. "
                f"Available columns: {list(data.columns)}."
            )

    # 6. Numeric features validation
    seen_numeric: Set[str] = set()
    num_cols = list(numeric_features) if numeric_features is not None else []
    for col in num_cols:
        if not isinstance(col, str) or not col.strip():
            raise ValueError("Numeric feature names must be non-empty strings.")
        if col in seen_numeric:
            raise ValueError(f"Duplicate numeric feature '{col}' in numeric_features.")
        seen_numeric.add(col)
        if col not in data.columns:
            raise ValueError(f"Numeric feature column '{col}' not found in DataFrame.")

        series = data[col]
        # Check dtype is numeric (not object/string/bool)
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise TypeError(
                f"Feature column '{col}' must have a numeric dtype, got '{series.dtype}'."
            )

        # Non-finite check (NaN, +inf, -inf)
        vals = series.to_numpy(dtype=float)
        if not np.isfinite(vals).all():
            non_finite_count = int((~np.isfinite(vals)).sum())
            raise ValueError(
                f"Numeric feature column '{col}' contains {non_finite_count} non-finite (NaN or Inf) values."
            )

    # 7. Categorical features validation
    seen_cat: Set[str] = set()
    cat_cols = list(categorical_features) if categorical_features is not None else []
    for col in cat_cols:
        if not isinstance(col, str) or not col.strip():
            raise ValueError("Categorical feature names must be non-empty strings.")
        if col in seen_cat:
            raise ValueError(f"Duplicate categorical feature '{col}' in categorical_features.")
        seen_cat.add(col)
        if col not in data.columns:
            raise ValueError(f"Categorical feature column '{col}' not found in DataFrame.")
        if data[col].isna().any():
            null_count = int(data[col].isna().sum())
            raise ValueError(
                f"Categorical feature column '{col}' contains {null_count} null/NaN values."
            )

    # 8. Numeric and Categorical exclusivity
    overlap = seen_numeric & seen_cat
    if overlap:
        raise ValueError(
            f"Feature(s) cannot be declared as both numeric and categorical: {sorted(overlap)}."
        )


def extract_state_proxy(
    data: pd.DataFrame,
    proxy: Union[str, Callable[..., Any], Mapping[Any, str]],
    states: Sequence[str],
    *,
    source_column: Optional[str] = None,
    callable_mode: str = "auto",
) -> pd.Series:
    """Extract and validate deterministic state labels from an observed DataFrame.

    Supports:
    1. Explicit state column name (str).
    2. Deterministic callable (row-level or dataframe-level function).
    3. Deterministic category mapping (Mapping with source_column specified).

    Args:
        data: Input DataFrame containing interaction/event rows.
        proxy: State proxy specification: string column name, callable, or mapping.
        states: Sequence of declared valid state names.
        source_column: Optional column name required when proxy is a Mapping.
        callable_mode: Dispatch mode for callable proxies: 'auto', 'row', or 'dataframe'.

    Returns:
        pd.Series containing the extracted state labels, strictly matching data.index.

    Raises:
        TypeError: If input types or proxy types are unsupported.
        ValueError: If validation fails (empty data, unknown states, null outputs,
            mismatched length/index, duplicate states, or invalid callable_mode).
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}.")

    if data.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)) or len(states) == 0:
        raise ValueError("Declared states must be a non-empty sequence of strings.")

    if callable_mode not in {"auto", "row", "dataframe"}:
        raise ValueError(
            f"callable_mode must be 'auto', 'row', or 'dataframe', got '{callable_mode}'."
        )

    seen_states: Set[str] = set()
    for idx, s in enumerate(states):
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"State at index {idx} must be a non-empty string.")
        if s in seen_states:
            raise ValueError(f"Duplicate state name '{s}' in declared states.")
        seen_states.add(s)

    if isinstance(proxy, str):
        if not proxy.strip():
            raise ValueError("Proxy column name must be a non-empty string.")
        if proxy not in data.columns:
            raise ValueError(
                f"Proxy column '{proxy}' not found in DataFrame. Available columns: {list(data.columns)}."
            )
        extracted = data[proxy]

    elif isinstance(proxy, Mapping):
        if source_column is None:
            raise ValueError("source_column must be specified when proxy is a Mapping.")
        if not isinstance(source_column, str) or not source_column.strip():
            raise ValueError("source_column must be a non-empty string.")
        if source_column not in data.columns:
            raise ValueError(
                f"Source column '{source_column}' not found in DataFrame. Available columns: {list(data.columns)}."
            )
        extracted = data[source_column].map(proxy)

    elif callable(proxy):
        if callable_mode == "row":
            extracted = data.apply(proxy, axis=1)
        elif callable_mode == "dataframe":
            raw_result = proxy(data)
            if isinstance(raw_result, pd.Series):
                if len(raw_result) != len(data):
                    raise ValueError(
                        f"Proxy output length ({len(raw_result)}) does not match DataFrame length ({len(data)})."
                    )
                if not raw_result.index.equals(data.index):
                    raise ValueError("Proxy Series index does not align with DataFrame index.")
                extracted = raw_result
            elif isinstance(raw_result, (list, tuple, np.ndarray)):
                if len(raw_result) != len(data):
                    raise ValueError(
                        f"Proxy output length ({len(raw_result)}) does not match DataFrame length ({len(data)})."
                    )
                extracted = pd.Series(raw_result, index=data.index)
            else:
                raise TypeError(
                    f"DataFrame callable proxy must return a Series, ndarray, or sequence, got {type(raw_result).__name__}."
                )
        else:
            # "auto" mode: safely determine whether proxy is row-level or DataFrame-level
            # Test proxy on first row without swallowing unexpected DataFrame-level exceptions
            sample_row = data.iloc[0]
            is_row_level = False
            try:
                sample_out = proxy(sample_row)
                if isinstance(sample_out, (str, int, float, bool, np.number)) or sample_out is None:
                    is_row_level = True
            except Exception:
                is_row_level = False

            if is_row_level:
                extracted = data.apply(proxy, axis=1)
            else:
                raw_result = proxy(data)
                if isinstance(raw_result, pd.Series):
                    if len(raw_result) != len(data):
                        raise ValueError(
                            f"Proxy output length ({len(raw_result)}) does not match DataFrame length ({len(data)})."
                        )
                    if not raw_result.index.equals(data.index):
                        raise ValueError("Proxy Series index does not align with DataFrame index.")
                    extracted = raw_result
                elif isinstance(raw_result, (list, tuple, np.ndarray)):
                    if len(raw_result) != len(data):
                        raise ValueError(
                            f"Proxy output length ({len(raw_result)}) does not match DataFrame length ({len(data)})."
                        )
                    extracted = pd.Series(raw_result, index=data.index)
                else:
                    raise TypeError(
                        f"Proxy callable must return a scalar state per row or a sequence matching DataFrame length, got {type(raw_result).__name__}."
                    )

    else:
        raise TypeError(
            f"Unsupported proxy type: {type(proxy).__name__}. Expected str, Mapping, or callable."
        )

    # Validate extracted output length and index
    if len(extracted) != len(data):
        raise ValueError(
            f"Proxy output length ({len(extracted)}) does not match DataFrame length ({len(data)})."
        )
    if not extracted.index.equals(data.index):
        raise ValueError("Proxy Series index does not align with DataFrame index.")

    # Validate nulls
    if extracted.isna().any():
        null_count = int(extracted.isna().sum())
        raise ValueError(f"State proxy produced {null_count} null/NaN values.")

    # Validate unknown states
    observed_states = set(extracted.unique())
    unknown_states = observed_states - seen_states
    if unknown_states:
        raise ValueError(
            f"Unknown state label(s) produced by proxy: {sorted(unknown_states)}. "
            f"Declared states: {sorted(seen_states)}."
        )

    result_series = extracted.copy()
    result_series.name = "state"
    return result_series


@dataclass(frozen=True)
class CalibrationData:
    """Immutable data container encapsulating empirical observations for calibration.

    Attributes:
        data: Observed event/interaction traces as a pandas DataFrame.
        states: Sequence of declared valid state names.
        state_column: Name of the column containing state labels.
        feature_columns: Optional sequence of feature column names to calibrate.
        sequence_column: Optional column partitioning sequences (e.g. 'sequence_id').
        profile_column: Optional column identifying behavioral profiles / cohorts.
        numeric_features: Optional subset of feature columns required to be numeric.
        categorical_features: Optional subset of feature columns for categorical modeling.
        metadata: Optional metadata dictionary.
    """

    data: pd.DataFrame
    states: Sequence[str]
    state_column: str = "state"
    feature_columns: Optional[Sequence[str]] = None
    sequence_column: Optional[str] = "sequence_id"
    profile_column: Optional[str] = None
    numeric_features: Optional[Sequence[str]] = None
    categorical_features: Optional[Sequence[str]] = None
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        # Validate all inputs
        validate_calibration_data(
            data=self.data,
            states=self.states,
            state_column=self.state_column,
            feature_columns=self.feature_columns,
            sequence_column=self.sequence_column,
            profile_column=self.profile_column,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
        )

        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise TypeError(
                    f"metadata must be a mapping (e.g. dict), got {type(self.metadata).__name__}."
                )
            object.__setattr__(self, "metadata", dict(self.metadata))

        # Defensive immutability: copy DataFrame and freeze sequences as tuples
        object.__setattr__(self, "data", self.data.copy())
        object.__setattr__(self, "states", tuple(self.states))
        if self.feature_columns is not None:
            object.__setattr__(self, "feature_columns", tuple(self.feature_columns))
        if self.numeric_features is not None:
            object.__setattr__(self, "numeric_features", tuple(self.numeric_features))
        if self.categorical_features is not None:
            object.__setattr__(self, "categorical_features", tuple(self.categorical_features))

    @property
    def num_records(self) -> int:
        """Total number of interaction records in dataset."""
        return len(self.data)

    @property
    def num_sequences(self) -> int:
        """Total number of unique sequences, or 1 if sequence_column is None."""
        if self.sequence_column is None:
            return 1
        return int(self.data[self.sequence_column].nunique())

    @property
    def state_names(self) -> List[str]:
        """Ordered list of declared state names."""
        return list(self.states)

    def get_state_data(self, state_name: str) -> pd.DataFrame:
        """Return subset of observations belonging to a specific state.

        Args:
            state_name: Name of the state to filter by.

        Returns:
            Filtered pd.DataFrame slice copy.

        Raises:
            ValueError: If state_name is not in declared states.
        """
        if state_name not in self.states:
            raise ValueError(
                f"State '{state_name}' is not among declared states: {list(self.states)}."
            )
        return self.data[self.data[self.state_column] == state_name].copy()

    def get_profile_data(self, profile_name: str) -> pd.DataFrame:
        """Return subset of observations belonging to a specific profile.

        Args:
            profile_name: Name of profile/cohort to filter by.

        Returns:
            Filtered pd.DataFrame slice copy.

        Raises:
            ValueError: If profile_column is None or profile_name is not found.
        """
        if self.profile_column is None:
            raise ValueError("Cannot filter by profile when profile_column is None.")
        subset = self.data[self.data[self.profile_column] == profile_name]
        if subset.empty:
            raise ValueError(f"No observations found for profile '{profile_name}'.")
        return subset.copy()

    @classmethod
    def from_proxy(
        cls,
        data: pd.DataFrame,
        states: Sequence[str],
        proxy: Union[str, Callable[..., Any], Mapping[Any, str]],
        *,
        state_column: str = "state",
        source_column: Optional[str] = None,
        feature_columns: Optional[Sequence[str]] = None,
        sequence_column: Optional[str] = "sequence_id",
        profile_column: Optional[str] = None,
        numeric_features: Optional[Sequence[str]] = None,
        categorical_features: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        callable_mode: str = "auto",
    ) -> CalibrationData:
        """Construct a validated CalibrationData instance using a state proxy.

        Args:
            data: Input DataFrame containing interaction traces.
            states: Sequence of declared valid state names.
            proxy: State proxy specification (column name, callable, or mapping).
            state_column: Name of the state column to populate with extracted states.
            source_column: Optional source column name when proxy is a Mapping.
            feature_columns: Optional sequence of feature column names to calibrate.
            sequence_column: Optional column partitioning sequences.
            profile_column: Optional column identifying behavioral profiles.
            numeric_features: Optional subset of feature columns required to be numeric.
            categorical_features: Optional subset of feature columns for categorical modeling.
            metadata: Optional metadata mapping.
            callable_mode: Dispatch mode for callable proxies ('auto', 'row', 'dataframe').

        Returns:
            Validated, immutable CalibrationData instance.
        """
        extracted_states = extract_state_proxy(
            data=data,
            proxy=proxy,
            states=states,
            source_column=source_column,
            callable_mode=callable_mode,
        )
        data_with_state = data.copy()
        data_with_state[state_column] = extracted_states
        return cls(
            data=data_with_state,
            states=states,
            state_column=state_column,
            feature_columns=feature_columns,
            sequence_column=sequence_column,
            profile_column=profile_column,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            metadata=metadata,
        )


def fit_transition_matrix(
    data: CalibrationData,
    *,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Fit empirical Markov transition probability matrix from CalibrationData.

    Counts transitions strictly within sequences to avoid transition leakage across
    sequence boundaries. Supports additive Laplace smoothing.

    Args:
        data: Validated CalibrationData instance.
        smoothing: Non-negative additive smoothing parameter alpha (default 0.0).

    Returns:
        2D square stochastic matrix of shape (n_states, n_states).

    Raises:
        TypeError: If data is not CalibrationData or smoothing is not numeric.
        ValueError: If smoothing < 0 or non-finite.
    """
    if not isinstance(data, CalibrationData):
        raise TypeError(f"data must be a CalibrationData instance, got {type(data).__name__}.")

    if isinstance(smoothing, bool) or not isinstance(smoothing, (int, float)):
        raise TypeError(f"smoothing must be numeric, got {type(smoothing).__name__}.")

    smoothing_val = float(smoothing)
    if not np.isfinite(smoothing_val) or smoothing_val < 0.0:
        raise ValueError(f"smoothing must be a non-negative finite number, got {smoothing}.")

    states = data.state_names
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    counts = np.zeros((n_states, n_states), dtype=float)

    # Accumulate transition counts partitioned strictly by sequence
    if data.sequence_column is not None and data.sequence_column in data.data.columns:
        for _, group in data.data.groupby(data.sequence_column, sort=False):
            seq_states = group[data.state_column].to_numpy()
            if len(seq_states) > 1:
                for t in range(len(seq_states) - 1):
                    from_idx = state_to_idx[seq_states[t]]
                    to_idx = state_to_idx[seq_states[t + 1]]
                    counts[from_idx, to_idx] += 1.0
    else:
        seq_states = data.data[data.state_column].to_numpy()
        if len(seq_states) > 1:
            for t in range(len(seq_states) - 1):
                from_idx = state_to_idx[seq_states[t]]
                to_idx = state_to_idx[seq_states[t + 1]]
                counts[from_idx, to_idx] += 1.0

    # Row normalize with zero-observation deterministic fallback
    matrix = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        row_counts = counts[i]
        total_obs = float(row_counts.sum())
        if total_obs == 0.0:
            if smoothing_val > 0.0:
                # With smoothing on zero observations: alpha / (N * alpha) = 1/N
                matrix[i, :] = 1.0 / n_states
            else:
                # Unsmoothed zero-observation state: deterministic self-loop = 1.0
                matrix[i, i] = 1.0
        else:
            if smoothing_val > 0.0:
                denom = total_obs + n_states * smoothing_val
                matrix[i, :] = (row_counts + smoothing_val) / denom
            else:
                matrix[i, :] = row_counts / total_obs

    validate_transition_matrix(matrix)
    return matrix


def fit_distribution(
    values: Any,
    distribution_type: str,
) -> FeatureDistribution:
    """Fit Maximum Likelihood Estimation (MLE) parameters for a supported distribution.

    Args:
        values: Sequence or Series of empirical observations.
        distribution_type: Identifier of the distribution family (e.g. 'normal', 'bernoulli').

    Returns:
        FeatureDistribution instance containing estimated parameters.

    Raises:
        TypeError: If inputs have incompatible types.
        ValueError: If distribution is unsupported, values are empty, or values violate domain constraints.
    """
    if not isinstance(distribution_type, str) or not distribution_type.strip():
        raise ValueError("distribution_type must be a non-empty string.")

    dtype = distribution_type.lower().strip()
    if dtype not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(
            f"unsupported distribution type: '{distribution_type}'. "
            f"Supported distributions: {sorted(SUPPORTED_DISTRIBUTIONS)}."
        )

    if isinstance(values, pd.Series):
        arr = values.dropna().to_numpy()
    elif isinstance(values, (list, tuple, np.ndarray)):
        arr = np.asarray(values)
    else:
        raise TypeError(f"values must be a Sequence, Series, or ndarray, got {type(values).__name__}.")

    if len(arr) == 0:
        raise ValueError("Cannot fit distribution from empty observations.")

    # Check non-finite values for numeric families
    if dtype != "categorical":
        if not pd.api.types.is_numeric_dtype(arr) or (pd.api.types.is_bool_dtype(arr) and dtype != "bernoulli"):
            if dtype != "bernoulli":
                try:
                    arr = arr.astype(float)
                except (ValueError, TypeError) as err:
                    raise TypeError(f"Cannot convert observations to numeric for '{dtype}': {err}") from err

        if not np.isfinite(arr).all():
            raise ValueError("Observations contain non-finite numbers (NaN or Inf).")

    params: Dict[str, Any] = {}

    if dtype == "normal":
        arr_f = arr.astype(float)
        loc = float(np.mean(arr_f))
        scale = float(np.std(arr_f, ddof=0))
        params = {"loc": loc, "scale": scale}

    elif dtype == "lognormal":
        arr_f = arr.astype(float)
        if np.any(arr_f <= 0.0):
            raise ValueError("Lognormal distribution requires strictly positive observations (> 0).")
        log_vals = np.log(arr_f)
        mean = float(np.mean(log_vals))
        sigma = float(np.std(log_vals, ddof=0))
        params = {"mean": mean, "sigma": sigma}

    elif dtype == "exponential":
        arr_f = arr.astype(float)
        if np.any(arr_f < 0.0):
            raise ValueError("Exponential distribution requires non-negative observations (>= 0).")
        scale = float(np.mean(arr_f))
        if scale <= 0.0:
            raise ValueError("Exponential distribution scale must be positive (> 0); cannot fit all-zero data.")
        params = {"scale": scale}

    elif dtype == "uniform":
        arr_f = arr.astype(float)
        low = float(np.min(arr_f))
        high = float(np.max(arr_f))
        params = {"low": low, "high": high}

    elif dtype == "uniform_discrete":
        arr_f = arr.astype(float)
        if not np.all(np.isclose(arr_f, np.round(arr_f))):
            raise ValueError("uniform_discrete requires integer-valued observations.")
        low = int(np.round(np.min(arr_f)))
        high = int(np.round(np.max(arr_f)))
        params = {"low": low, "high": high}

    elif dtype == "bernoulli":
        unique_vals = set(np.unique(arr))
        if not unique_vals.issubset({0, 1, 0.0, 1.0, True, False}):
            raise ValueError(
                f"Bernoulli distribution requires observations exclusively in {{0, 1}}, got {sorted(unique_vals)}."
            )
        arr_f = arr.astype(float)
        p = float(np.mean(arr_f))
        params = {"p": p}

    elif dtype == "poisson":
        arr_f = arr.astype(float)
        if np.any(arr_f < 0.0) or not np.all(np.isclose(arr_f, np.round(arr_f))):
            raise ValueError("Poisson distribution requires non-negative integer observations.")
        lam = float(np.mean(arr_f))
        params = {"lam": lam}

    elif dtype == "categorical":
        if pd.isna(arr).any():
            raise ValueError("Observations for categorical distribution contain null/NaN values.")
        unique_items, counts = np.unique(arr, return_counts=True)
        pairs = sorted(zip(unique_items, counts), key=lambda x: str(x[0]))
        items = [p[0] for p in pairs]
        total_count = float(sum(p[1] for p in pairs))
        probs = [float(p[1] / total_count) for p in pairs]
        probs[-1] = 1.0 - sum(probs[:-1])
        params = {"items": items, "probabilities": probs}

    validate_distribution_params(dtype, params)
    return FeatureDistribution(distribution_type=dtype, params=params)


def fit_profile(
    data: CalibrationData,
    *,
    name: str = "calibrated_profile",
    distribution_types: Optional[Mapping[str, str]] = None,
    default_distribution: str = "normal",
    transition_smoothing: float = 0.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Profile:
    """Fit a complete Profile with empirical transition matrix and state emissions.

    Args:
        data: Validated CalibrationData instance.
        name: Profile name identifier.
        distribution_types: Optional mapping of feature_name -> distribution_type.
        default_distribution: Fallback distribution family for numeric features (default 'normal').
        transition_smoothing: Additive smoothing parameter alpha for transitions.
        metadata: Optional additional metadata mapping.

    Returns:
        Configured Profile instance ready for simulation.

    Raises:
        TypeError: On invalid input types.
        ValueError: On semantic validation failures.
    """
    if not isinstance(data, CalibrationData):
        raise TypeError(f"data must be a CalibrationData instance, got {type(data).__name__}.")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Profile name must be a non-empty string.")

    matrix = fit_transition_matrix(data, smoothing=transition_smoothing)

    # Determine features to calibrate
    if data.feature_columns is not None:
        features = list(data.feature_columns)
    else:
        excluded = {data.state_column}
        if data.sequence_column is not None:
            excluded.add(data.sequence_column)
        if data.profile_column is not None:
            excluded.add(data.profile_column)
        features = [col for col in data.data.columns if col not in excluded]

    state_emissions: Dict[str, Dict[str, FeatureDistribution]] = {}
    dist_types = dict(distribution_types) if distribution_types is not None else {}

    for state_name in data.states:
        state_emissions[state_name] = {}
        state_df = data.get_state_data(state_name)

        for feat in features:
            if feat in dist_types:
                dtype = dist_types[feat]
            elif data.categorical_features and feat in data.categorical_features:
                dtype = "categorical"
            elif data.numeric_features and feat in data.numeric_features:
                dtype = default_distribution
            else:
                # Infer type
                if pd.api.types.is_numeric_dtype(data.data[feat]) and not pd.api.types.is_bool_dtype(data.data[feat]):
                    dtype = default_distribution
                else:
                    dtype = "categorical"

            if not state_df.empty:
                dist = fit_distribution(state_df[feat], dtype)
            else:
                # Explicit deterministic policy for unobserved state:
                # Fit from global population feature data across all states
                dist = fit_distribution(data.data[feat], dtype)

            state_emissions[state_name][feat] = dist

    # Metadata assembly
    meta: Dict[str, Any] = {
        "calibrated_from": "CalibrationData",
        "num_records": data.num_records,
        "num_sequences": data.num_sequences,
        "states": list(data.states),
    }
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}.")
        meta.update(dict(metadata))

    return Profile(
        name=name,
        state_emissions=state_emissions,
        transition_matrix=matrix,
        metadata=meta,
    )
