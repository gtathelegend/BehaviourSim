"""Calibration validation subsystem for BehaviorSim.

Compares empirical/calibrated behavioral traces against generated synthetic
behavior across structural, state occupancy, transition dynamics, and feature
emission distributions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from behaviorsim.calibration.fitter import (
    CalibrationData,
    fit_transition_matrix,
)
from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.config import validate_distribution_params
from behaviorsim.core.utils import validate_transition_matrix


@dataclass(frozen=True)
class StructuralValidationResult:
    """Result of structural integrity checks on calibration models and profiles."""

    is_valid: bool
    transition_matrix_valid: bool
    state_names_valid: bool
    emissions_valid: bool
    errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        """Convert structural validation result to a dictionary."""
        return {
            "is_valid": self.is_valid,
            "transition_matrix_valid": self.transition_matrix_valid,
            "state_names_valid": self.state_names_valid,
            "emissions_valid": self.emissions_valid,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class StateOccupancyResult:
    """Comparison of state occupancy proportions between empirical and synthetic traces."""

    empirical_occupancy: Dict[str, float]
    synthetic_occupancy: Dict[str, float]
    occupancy_difference: Dict[str, float]
    total_variation_distance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert state occupancy comparison to a dictionary."""
        return {
            "empirical_occupancy": self.empirical_occupancy,
            "synthetic_occupancy": self.synthetic_occupancy,
            "occupancy_difference": self.occupancy_difference,
            "total_variation_distance": self.total_variation_distance,
        }


@dataclass(frozen=True)
class TransitionValidationResult:
    """Comparison of empirical and synthetic transition dynamics."""

    empirical_matrix: np.ndarray
    synthetic_matrix: np.ndarray
    max_absolute_error: float
    mean_absolute_error: float
    frobenius_distance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert transition validation result to a dictionary."""
        return {
            "empirical_matrix": self.empirical_matrix.tolist(),
            "synthetic_matrix": self.synthetic_matrix.tolist(),
            "max_absolute_error": float(self.max_absolute_error),
            "mean_absolute_error": float(self.mean_absolute_error),
            "frobenius_distance": float(self.frobenius_distance),
        }


@dataclass(frozen=True)
class NumericFeatureComparison:
    """Statistical comparison of empirical vs synthetic continuous/numeric feature distributions."""

    feature_name: str
    empirical_mean: float
    synthetic_mean: float
    empirical_std: float
    synthetic_std: float
    mean_absolute_error: float
    relative_mean_error: float
    wasserstein_distance: float
    ks_statistic: float
    ks_pvalue: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert numeric feature comparison to a dictionary."""
        return {
            "feature_name": self.feature_name,
            "empirical_mean": self.empirical_mean,
            "synthetic_mean": self.synthetic_mean,
            "empirical_std": self.empirical_std,
            "synthetic_std": self.synthetic_std,
            "mean_absolute_error": self.mean_absolute_error,
            "relative_mean_error": self.relative_mean_error,
            "wasserstein_distance": self.wasserstein_distance,
            "ks_statistic": self.ks_statistic,
            "ks_pvalue": self.ks_pvalue,
        }


@dataclass(frozen=True)
class CategoricalFeatureComparison:
    """Statistical comparison of empirical vs synthetic categorical/discrete feature distributions."""

    feature_name: str
    empirical_frequencies: Dict[str, float]
    synthetic_frequencies: Dict[str, float]
    frequency_difference: Dict[str, float]
    total_variation_distance: float
    jensen_shannon_distance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert categorical feature comparison to a dictionary."""
        return {
            "feature_name": self.feature_name,
            "empirical_frequencies": self.empirical_frequencies,
            "synthetic_frequencies": self.synthetic_frequencies,
            "frequency_difference": self.frequency_difference,
            "total_variation_distance": self.total_variation_distance,
            "jensen_shannon_distance": self.jensen_shannon_distance,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Comprehensive, immutable validation report comparing empirical and synthetic behavioral traces."""

    structural: StructuralValidationResult
    state_occupancy: StateOccupancyResult
    transition: Optional[TransitionValidationResult]
    numeric_features: Dict[str, NumericFeatureComparison]
    categorical_features: Dict[str, CategoricalFeatureComparison]
    threshold_results: Dict[str, bool]
    is_valid: bool
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full validation report to a clean, nested dictionary."""
        return {
            "is_valid": self.is_valid,
            "structural": self.structural.to_dict(),
            "state_occupancy": self.state_occupancy.to_dict(),
            "transition": self.transition.to_dict() if self.transition is not None else None,
            "numeric_features": {
                k: v.to_dict() for k, v in self.numeric_features.items()
            },
            "categorical_features": {
                k: v.to_dict() for k, v in self.categorical_features.items()
            },
            "threshold_results": self.threshold_results,
            "summary": self.summary,
        }


def _validate_structural_profile(
    profile: Optional[Profile],
    declared_states: Sequence[str],
) -> StructuralValidationResult:
    """Perform structural checks on a calibrated Profile instance."""
    if profile is None:
        return StructuralValidationResult(
            is_valid=True,
            transition_matrix_valid=True,
            state_names_valid=True,
            emissions_valid=True,
            errors=(),
        )

    errors: List[str] = []
    trans_valid = True
    states_valid = True
    emiss_valid = True

    # Check transition matrix
    if profile.transition_matrix is not None:
        try:
            validate_transition_matrix(profile.transition_matrix)
            if profile.transition_matrix.shape != (len(declared_states), len(declared_states)):
                errors.append(
                    f"Transition matrix shape {profile.transition_matrix.shape} does not match "
                    f"number of states ({len(declared_states)})."
                )
                trans_valid = False
        except Exception as exc:
            errors.append(f"Invalid transition matrix: {exc}")
            trans_valid = False

    # Check state emissions alignment
    profile_states = set(profile.state_emissions.keys())
    declared_set = set(declared_states)
    if profile_states != declared_set:
        diff_missing = declared_set - profile_states
        diff_extra = profile_states - declared_set
        if diff_missing:
            errors.append(f"Profile state emissions missing states: {sorted(diff_missing)}")
            states_valid = False
        if diff_extra:
            errors.append(f"Profile state emissions contains undeclared states: {sorted(diff_extra)}")
            states_valid = False

    # Validate individual FeatureDistribution specifications
    for state_name, emissions in profile.state_emissions.items():
        if not isinstance(emissions, Mapping):
            errors.append(f"State '{state_name}' emissions must be a mapping.")
            emiss_valid = False
            continue
        for feat_name, dist_spec in emissions.items():
            if isinstance(dist_spec, FeatureDistribution):
                try:
                    validate_distribution_params(dist_spec.distribution_type, dist_spec.params)
                except Exception as exc:
                    errors.append(
                        f"State '{state_name}' feature '{feat_name}' has invalid distribution: {exc}"
                    )
                    emiss_valid = False

    overall_valid = trans_valid and states_valid and emiss_valid and len(errors) == 0
    return StructuralValidationResult(
        is_valid=overall_valid,
        transition_matrix_valid=trans_valid,
        state_names_valid=states_valid,
        emissions_valid=emiss_valid,
        errors=tuple(errors),
    )


def validate_calibration(
    empirical_data: Union[CalibrationData, pd.DataFrame],
    synthetic_data: Union[CalibrationData, pd.DataFrame],
    *,
    profile: Optional[Profile] = None,
    states: Optional[Sequence[str]] = None,
    state_column: str = "state",
    sequence_column: Optional[str] = "sequence_id",
    numeric_features: Optional[Sequence[str]] = None,
    categorical_features: Optional[Sequence[str]] = None,
    thresholds: Optional[Mapping[str, float]] = None,
) -> ValidationReport:
    """Validate calibrated models by comparing empirical traces against generated synthetic traces.

    Calculates structural integrity, state occupancy distributions, transition dynamics,
    and continuous/discrete feature emission alignments using rigorous statistical metrics.

    Args:
        empirical_data: Ground-truth observations as CalibrationData or pd.DataFrame.
        synthetic_data: Generated simulation output as CalibrationData or pd.DataFrame.
        profile: Optional fitted Profile instance to structurally validate.
        states: Sequence of declared state names. Inferred from empirical CalibrationData if omitted.
        state_column: Column name containing state labels (default 'state').
        sequence_column: Column name partitioning sequences for transition comparison.
        numeric_features: Feature names to validate using continuous distribution tests.
        categorical_features: Feature names to validate using discrete distribution tests.
        thresholds: Optional mapping of metric thresholds for configurable pass/fail evaluation.
            Supported threshold keys:
            - 'max_state_tvd': Maximum allowed state occupancy TVD.
            - 'max_transition_mae': Maximum allowed mean absolute transition error.
            - 'max_transition_max_error': Maximum allowed max absolute transition error.
            - 'max_numeric_mean_mae': Maximum allowed mean absolute error across numeric features.
            - 'max_categorical_tvd': Maximum allowed TVD across categorical features.

    Returns:
        Immutable, inspectable, and serializable ValidationReport.

    Raises:
        ValueError: If input datasets are empty or required columns/states are missing.
        TypeError: If dataset arguments are of invalid types.
    """
    # 1. Resolve empirical data and declarations
    if isinstance(empirical_data, CalibrationData):
        emp_df = empirical_data.data
        resolved_states = list(empirical_data.states)
        state_col = empirical_data.state_column
        seq_col = empirical_data.sequence_column
        num_feats = (
            list(empirical_data.numeric_features)
            if empirical_data.numeric_features is not None
            else []
        )
        cat_feats = (
            list(empirical_data.categorical_features)
            if empirical_data.categorical_features is not None
            else []
        )
    elif isinstance(empirical_data, pd.DataFrame):
        emp_df = empirical_data
        state_col = state_column
        seq_col = sequence_column
        num_feats = list(numeric_features) if numeric_features is not None else []
        cat_feats = list(categorical_features) if categorical_features is not None else []
    # 2. Resolve synthetic data
    if isinstance(synthetic_data, CalibrationData):
        syn_df = synthetic_data.data
    elif isinstance(synthetic_data, pd.DataFrame):
        syn_df = synthetic_data
    else:
        raise TypeError(
            f"synthetic_data must be CalibrationData or pd.DataFrame, got {type(synthetic_data).__name__}."
        )

    # Check empty DataFrames immediately
    if emp_df.empty:
        raise ValueError("Empirical dataset cannot be empty.")
    if syn_df.empty:
        raise ValueError("Synthetic dataset cannot be empty.")

    if isinstance(empirical_data, pd.DataFrame):
        if states is not None:
            resolved_states = list(states)
        elif profile is not None:
            resolved_states = list(profile.state_emissions.keys())
        elif state_col in emp_df.columns:
            resolved_states = sorted(emp_df[state_col].dropna().unique().tolist())
        else:
            raise ValueError(f"State column '{state_col}' not found in empirical DataFrame.")

    # Validate state column existence
    if state_col not in emp_df.columns:
        raise ValueError(f"State column '{state_col}' missing from empirical data.")
    if state_col not in syn_df.columns:
        raise ValueError(f"State column '{state_col}' missing from synthetic data.")

    # 3. Structural validation
    structural_result = _validate_structural_profile(profile, resolved_states)

    # 4. State occupancy validation
    emp_counts = emp_df[state_col].value_counts(normalize=True)
    syn_counts = syn_df[state_col].value_counts(normalize=True)

    emp_occ: Dict[str, float] = {}
    syn_occ: Dict[str, float] = {}
    occ_diff: Dict[str, float] = {}
    tvd_sum = 0.0

    all_observed_states = sorted(set(resolved_states).union(emp_counts.index).union(syn_counts.index))
    for s in all_observed_states:
        p_emp = float(emp_counts.get(s, 0.0))
        p_syn = float(syn_counts.get(s, 0.0))
        emp_occ[s] = p_emp
        syn_occ[s] = p_syn
        diff = p_syn - p_emp
        occ_diff[s] = float(diff)
        tvd_sum += abs(diff)

    state_tvd = float(0.5 * tvd_sum)
    state_occupancy_result = StateOccupancyResult(
        empirical_occupancy=emp_occ,
        synthetic_occupancy=syn_occ,
        occupancy_difference=occ_diff,
        total_variation_distance=state_tvd,
    )

    # 5. Transition dynamics validation
    transition_result: Optional[TransitionValidationResult] = None
    if seq_col in emp_df.columns and seq_col in syn_df.columns:
        try:
            emp_calib = (
                empirical_data
                if isinstance(empirical_data, CalibrationData)
                else CalibrationData(data=emp_df, states=resolved_states, state_column=state_col, sequence_column=seq_col)
            )
            syn_calib = (
                synthetic_data
                if isinstance(synthetic_data, CalibrationData)
                else CalibrationData(data=syn_df, states=resolved_states, state_column=state_col, sequence_column=seq_col)
            )
            p_emp = fit_transition_matrix(emp_calib, smoothing=0.0)
            p_syn = fit_transition_matrix(syn_calib, smoothing=0.0)

            diff_matrix = p_syn - p_emp
            max_abs_err = float(np.max(np.abs(diff_matrix)))
            mean_abs_err = float(np.mean(np.abs(diff_matrix)))
            frob_dist = float(np.linalg.norm(diff_matrix, ord="fro"))

            transition_result = TransitionValidationResult(
                empirical_matrix=p_emp,
                synthetic_matrix=p_syn,
                max_absolute_error=max_abs_err,
                mean_absolute_error=mean_abs_err,
                frobenius_distance=frob_dist,
            )
        except Exception:
            # Fall back to None if sequence structure is insufficient
            transition_result = None

    # 6. Numeric feature comparisons
    numeric_comparisons: Dict[str, NumericFeatureComparison] = {}
    for feat in num_feats:
        if feat in emp_df.columns and feat in syn_df.columns:
            emp_vals = pd.to_numeric(emp_df[feat], errors="coerce").dropna().values
            syn_vals = pd.to_numeric(syn_df[feat], errors="coerce").dropna().values

            if len(emp_vals) > 0 and len(syn_vals) > 0:
                emp_mean = float(np.mean(emp_vals))
                syn_mean = float(np.mean(syn_vals))
                emp_std = float(np.std(emp_vals, ddof=0))
                syn_std = float(np.std(syn_vals, ddof=0))
                mae = float(abs(syn_mean - emp_mean))
                denom = abs(emp_mean) if abs(emp_mean) > 1e-8 else 1.0
                rel_err = float(mae / denom)

                w_dist = float(wasserstein_distance(emp_vals, syn_vals))
                ks_res = ks_2samp(emp_vals, syn_vals)
                ks_stat = float(ks_res.statistic)
                ks_pval = float(ks_res.pvalue)

                numeric_comparisons[feat] = NumericFeatureComparison(
                    feature_name=feat,
                    empirical_mean=emp_mean,
                    synthetic_mean=syn_mean,
                    empirical_std=emp_std,
                    synthetic_std=syn_std,
                    mean_absolute_error=mae,
                    relative_mean_error=rel_err,
                    wasserstein_distance=w_dist,
                    ks_statistic=ks_stat,
                    ks_pvalue=ks_pval,
                )

    # 7. Categorical feature comparisons
    categorical_comparisons: Dict[str, CategoricalFeatureComparison] = {}
    for feat in cat_feats:
        if feat in emp_df.columns and feat in syn_df.columns:
            emp_s = emp_df[feat].astype(str)
            syn_s = syn_df[feat].astype(str)

            emp_f_series = emp_s.value_counts(normalize=True)
            syn_f_series = syn_s.value_counts(normalize=True)

            all_cats = sorted(set(emp_f_series.index).union(syn_f_series.index))
            emp_freqs: Dict[str, float] = {}
            syn_freqs: Dict[str, float] = {}
            diff_freqs: Dict[str, float] = {}

            p_vec: List[float] = []
            q_vec: List[float] = []
            tvd_cat_sum = 0.0

            for cat in all_cats:
                p_c = float(emp_f_series.get(cat, 0.0))
                q_c = float(syn_f_series.get(cat, 0.0))
                emp_freqs[cat] = p_c
                syn_freqs[cat] = q_c
                d = q_c - p_c
                diff_freqs[cat] = float(d)
                tvd_cat_sum += abs(d)
                p_vec.append(p_c)
                q_vec.append(q_c)

            cat_tvd = float(0.5 * tvd_cat_sum)
            # Jensen-Shannon distance using scipy (base 2)
            try:
                js_dist = float(jensenshannon(p_vec, q_vec, base=2))
                if np.isnan(js_dist):
                    js_dist = 0.0
            except Exception:
                js_dist = 0.0

            categorical_comparisons[feat] = CategoricalFeatureComparison(
                feature_name=feat,
                empirical_frequencies=emp_freqs,
                synthetic_frequencies=syn_freqs,
                frequency_difference=diff_freqs,
                total_variation_distance=cat_tvd,
                jensen_shannon_distance=js_dist,
            )

    # 8. Threshold evaluation
    threshold_results: Dict[str, bool] = {}
    thresholds_passed = True
    if thresholds is not None:
        if "max_state_tvd" in thresholds:
            max_tvd = thresholds["max_state_tvd"]
            passed = state_tvd <= max_tvd
            threshold_results["max_state_tvd"] = passed
            if not passed:
                thresholds_passed = False

        if "max_transition_mae" in thresholds and transition_result is not None:
            max_mae = thresholds["max_transition_mae"]
            passed = transition_result.mean_absolute_error <= max_mae
            threshold_results["max_transition_mae"] = passed
            if not passed:
                thresholds_passed = False

        if "max_transition_max_error" in thresholds and transition_result is not None:
            max_err = thresholds["max_transition_max_error"]
            passed = transition_result.max_absolute_error <= max_err
            threshold_results["max_transition_max_error"] = passed
            if not passed:
                thresholds_passed = False

        if "max_numeric_mean_mae" in thresholds and numeric_comparisons:
            max_num_mae = thresholds["max_numeric_mean_mae"]
            max_observed = max(c.mean_absolute_error for c in numeric_comparisons.values())
            passed = max_observed <= max_num_mae
            threshold_results["max_numeric_mean_mae"] = passed
            if not passed:
                thresholds_passed = False

        if "max_categorical_tvd" in thresholds and categorical_comparisons:
            max_cat_tvd = thresholds["max_categorical_tvd"]
            max_observed = max(c.total_variation_distance for c in categorical_comparisons.values())
            passed = max_observed <= max_cat_tvd
            threshold_results["max_categorical_tvd"] = passed
            if not passed:
                thresholds_passed = False

    is_overall_valid = structural_result.is_valid and thresholds_passed

    summary = {
        "empirical_records": len(emp_df),
        "synthetic_records": len(syn_df),
        "state_tvd": state_tvd,
        "transition_mae": transition_result.mean_absolute_error if transition_result else None,
        "numeric_features_evaluated": list(numeric_comparisons.keys()),
        "categorical_features_evaluated": list(categorical_comparisons.keys()),
        "thresholds_evaluated": len(threshold_results) > 0,
    }

    return ValidationReport(
        structural=structural_result,
        state_occupancy=state_occupancy_result,
        transition=transition_result,
        numeric_features=numeric_comparisons,
        categorical_features=categorical_comparisons,
        threshold_results=threshold_results,
        is_valid=is_overall_valid,
        summary=summary,
    )
