"""CLSI-Adapt personalization model.

Architecture
------------
One XGBoost binary classifier is trained per learner profile.

The model predicts the binary overload target defined in
:mod:`src.feature_engineering`.

Pipeline
--------
1. For each profile:
   a. Build features (11 columns) and overload labels using the feature
      engineering API.
   b. Apply the warm-up filter: only interactions at index >= WARMUP_INTERACTIONS
      (0-indexed within the profile) are eligible for *prediction*.  Earlier
      interactions may appear in the training portion of a fold.
   c. Run 5-fold TimeSeriesSplit cross-validation.
      * Inside each outer training fold, a nested TimeSeriesSplit selects the
        best (max_depth, learning_rate) pair by inner-fold AUC.
      * The outer validation fold is never used to pick hyperparameters.
   d. Collect out-of-fold predictions (probability + binary label).
   e. Train a final model on all eligible data using the best
      hyperparameters found most frequently across outer folds.

Warm-up / target alignment
---------------------------
The overload target for row t requires:
  * prior history  : t >= prior_window - 1 = 3  (default prior_window=4)
  * future window  : t + future_window < n       (default future_window=3)
    → last ``future_window`` rows receive NaN target.

So valid rows are those where overload target ∈ {0, 1} (not NaN).

The warm-up of 20 interactions means:

    warm_mask = (interaction_position_within_profile >= WARMUP_INTERACTIONS)

This mask is applied to the set of valid rows.  Rows that fail the warm-up
mask are excluded from CV folds **and from out-of-fold predictions**, but they
can still appear in the training portion of a fold (the temporal split does
not prevent warm-up rows from being training data).

No future observations enter training features.  ``build_features`` guarantees
this by construction.

Feature names
-------------
The 11 features (in order) are:
    nrt, accuracy, window_error_rate, retries, help_requested,
    confidence, streak_correct, streak_incorrect, nrt_variance,
    session_time, rolling_mean_nrt  (the 11th column appended by build_features)

CV strategy
-----------
Outer: TimeSeriesSplit(n_splits=5) on the warm-up-eligible valid rows.
Inner: TimeSeriesSplit(n_splits=3) on each outer training fold.

Both splits are strictly temporal (no shuffle).

Hyperparameter search grid
--------------------------
    max_depth     : {3, 5, 7}
    learning_rate : {0.01, 0.1}

Selection metric: mean inner-fold ROC-AUC.

XGBoost base config
-------------------
    n_estimators         : 200
    subsample            : 0.8
    colsample_bytree     : 0.8
    use_label_encoder    : False (deprecated in XGB ≥ 1.6)
    eval_metric          : logloss  (used for early stopping, not selection)
    objective            : binary:logistic
    scale_pos_weight     : computed per-fold from class ratio
    random_state / seed  : controlled via ``seed`` parameter (default 42)
    tree_method          : hist  (deterministic, fast)
    device               : cpu
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from src.feature_engineering import (
    FEATURE_COLUMNS,
    build_features,
    build_overload_target,
)
from src.simulator import LEARNER_PROFILES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of interactions to skip at the start of each profile's sequence
#: before making predictions.  Rows in [0, WARMUP) may still appear as
#: training data inside a fold.
WARMUP_INTERACTIONS: int = 20

#: Feature names: 10 base columns + rolling_mean_nrt appended by build_features.
FEATURE_NAMES: List[str] = FEATURE_COLUMNS + ["rolling_mean_nrt"]

#: Outer CV splits (TimeSeriesSplit).
N_OUTER_SPLITS: int = 5

#: Inner CV splits for nested hyperparameter search.
N_INNER_SPLITS: int = 3

#: Fixed number of boosting rounds (production default).
N_ESTIMATORS: int = 200

#: Reduced n_estimators used inside tests for speed. Never use in production.
TEST_N_ESTIMATORS: int = 20

#: Hyperparameter search grid.
HP_GRID: List[Dict[str, Any]] = [
    {"max_depth": md, "learning_rate": lr}
    for md in [3, 5, 7]
    for lr in [0.01, 0.1]
]

#: XGBoost base parameters (non-tunable). n_estimators is excluded here
#: so callers can override it at construction time.
XGB_BASE_PARAMS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cpu",
    "verbosity": 0,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Predictions and metrics for a single outer CV fold."""

    fold_idx: int
    profile: str
    #: Row positions (within the eligible array) of the validation set.
    val_indices: np.ndarray
    #: True binary overload labels for the validation set.
    y_true: np.ndarray
    #: Predicted probability of overload for the validation set.
    y_prob: np.ndarray
    #: Binary predicted labels (threshold 0.5).
    y_pred: np.ndarray
    #: Fold validation AUC.
    auc: float
    #: Best hyperparameters chosen by the inner search for this fold.
    best_params: Dict[str, Any]


@dataclass
class ProfileCVResult:
    """All CV output for a single learner profile."""

    profile: str
    fold_results: List[FoldResult]
    #: Out-of-fold prediction DataFrame with columns:
    #:   profile, interaction_id, row_position, y_true, y_prob, y_pred
    oof_predictions: pd.DataFrame
    #: Mean outer-fold AUC.
    mean_auc: float
    #: Std of outer-fold AUC.
    std_auc: float
    #: Best hyperparameters used for the final model.
    final_best_params: Dict[str, Any]
    #: Trained final XGBoost model (trained on all eligible data).
    final_model: Optional[xgb.XGBClassifier]
    #: Feature matrix used to train the final model.
    X_final: Optional[np.ndarray]
    #: Labels used to train the final model.
    y_final: Optional[np.ndarray]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_scale_pos_weight(y: np.ndarray) -> float:
    """Compute scale_pos_weight = count(negatives) / count(positives).

    If there are no positives, returns 1.0 to avoid division by zero.
    """
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def _build_xgb_classifier(
    extra_params: Dict[str, Any],
    scale_pos_weight: float,
    seed: int,
    n_estimators: int = N_ESTIMATORS,
) -> xgb.XGBClassifier:
    """Construct an XGBClassifier with merged base + tunable parameters."""
    params = {
        **XGB_BASE_PARAMS,
        "n_estimators": n_estimators,
        **extra_params,
        "scale_pos_weight": scale_pos_weight,
        "random_state": seed,
        "seed": seed,
    }
    return xgb.XGBClassifier(**params)


def _prepare_profile_data(
    df: pd.DataFrame,
    warmup: int = WARMUP_INTERACTIONS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Build feature matrix and labels for one profile's interaction sequence.

    Parameters
    ----------
    df:
        Interaction DataFrame (single or multi-learner for a profile).
    warmup:
        Number of interactions at the start of each learner's sequence that are excluded
        from *prediction* (but not from being training data).

    Returns
    -------
    X_eligible : np.ndarray, shape (n_eligible, 11)
        Feature matrix for eligible rows.
    y_eligible : np.ndarray, shape (n_eligible,)
        Binary overload labels for eligible rows.
    eligible_positions : np.ndarray of int
        0-indexed row positions within df of the eligible rows.
    interaction_ids : pd.Series
        interaction_id values aligned to eligible rows.
    learner_ids : pd.Series
        learner_id values aligned to eligible rows.
    """
    sub = df.reset_index(drop=True)

    # Build features (causal: row t uses only data from rows 0..t within learner)
    X_full, _ = build_features(sub)

    # Build overload target (NaN for last future_window rows per learner)
    y_full = build_overload_target(sub)

    n = len(sub)
    positions = np.arange(n)

    valid_mask = ~np.isnan(y_full.to_numpy())

    # Warm-up per learner sequence
    if "interaction_id" in sub.columns:
        warmup_mask = sub["interaction_id"].to_numpy() > warmup
    else:
        warmup_mask = positions >= warmup

    eligible_mask = valid_mask & warmup_mask

    eligible_positions = positions[eligible_mask]
    X_eligible = X_full[eligible_mask]
    y_eligible = y_full.to_numpy()[eligible_mask]

    if "interaction_id" in sub.columns:
        interaction_ids = sub["interaction_id"].iloc[eligible_positions].reset_index(drop=True)
    else:
        interaction_ids = pd.Series(eligible_positions + 1, name="interaction_id")

    if "learner_id" in sub.columns:
        learner_ids = sub["learner_id"].iloc[eligible_positions].reset_index(drop=True)
    else:
        learner_ids = pd.Series(1, index=np.arange(len(eligible_positions)), name="learner_id")

    return X_eligible, y_eligible, eligible_positions, interaction_ids, learner_ids


# ---------------------------------------------------------------------------
# Public API: hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    interaction_ids_train: Optional[np.ndarray] = None,
    seed: int = 42,
    n_inner_splits: int = N_INNER_SPLITS,
    hp_grid: List[Dict[str, Any]] = HP_GRID,
    n_estimators: int = N_ESTIMATORS,
) -> Dict[str, Any]:
    """Select the best hyperparameters via nested time-series cross-validation.

    Only the provided ``X_train`` / ``y_train`` are used.  The outer
    validation fold never enters this function; the contract is enforced
    by the caller.

    Because multiple synthetic learners share the same interaction-time index,
    forward-chaining splits are performed over unique interaction time steps
    rather than individual observation rows when ``interaction_ids_train`` is provided.

    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix for the outer training portion.
    y_train : np.ndarray
        Binary labels for the outer training portion.
    interaction_ids_train : np.ndarray, optional
        Interaction ID time step array aligned to X_train rows.
    seed : int
        Random seed for reproducibility.
    n_inner_splits : int
        Number of inner TimeSeriesSplit folds.
    hp_grid : list of dict
        Hyperparameter combinations to evaluate.
    n_estimators : int
        Number of boosting rounds.

    Returns
    -------
    best_params : dict
        The hyperparameter dict with the highest mean inner-fold AUC.
    """
    if interaction_ids_train is not None:
        unique_times = np.sort(np.unique(interaction_ids_train))
        if len(unique_times) >= n_inner_splits + 1:
            inner_cv = TimeSeriesSplit(n_splits=n_inner_splits)
            inner_splits = []
            for u_tr_idx, u_val_idx in inner_cv.split(unique_times):
                u_tr_set = set(unique_times[u_tr_idx])
                u_val_set = set(unique_times[u_val_idx])
                i_tr = np.where(np.isin(interaction_ids_train, list(u_tr_set)))[0]
                i_val = np.where(np.isin(interaction_ids_train, list(u_val_set)))[0]
                inner_splits.append((i_tr, i_val))
        else:
            inner_cv = TimeSeriesSplit(n_splits=n_inner_splits)
            inner_splits = list(inner_cv.split(X_train))
    else:
        inner_cv = TimeSeriesSplit(n_splits=n_inner_splits)
        inner_splits = list(inner_cv.split(X_train))

    best_auc = -1.0
    best_params = hp_grid[0]

    for params in hp_grid:
        fold_aucs: List[float] = []

        for inner_train_idx, inner_val_idx in inner_splits:
            if interaction_ids_train is not None:
                tr_iids = interaction_ids_train[inner_train_idx]
                val_iids = interaction_ids_train[inner_val_idx]
                assert tr_iids.max() < val_iids.min(), (
                    f"Inner split violation: inner validation starts before inner training ends. "
                    f"max_train_iid={tr_iids.max()}, min_val_iid={val_iids.min()}"
                )
                assert len(set(tr_iids).intersection(set(val_iids))) == 0, (
                    "Inner split violation: simultaneous time leakage in inner fold."
                )
            else:
                assert inner_train_idx.max() < inner_val_idx.min(), (
                    "Inner split violation: validation indices precede training indices."
                )

            X_i_tr, X_i_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_i_tr, y_i_val = y_train[inner_train_idx], y_train[inner_val_idx]

            # Skip if inner fold lacks both classes (can't compute AUC)
            if len(np.unique(y_i_tr)) < 2 or len(np.unique(y_i_val)) < 2:
                continue

            spw = _make_scale_pos_weight(y_i_tr)
            clf = _build_xgb_classifier(params, spw, seed, n_estimators=n_estimators)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_i_tr, y_i_tr)

            y_i_prob = clf.predict_proba(X_i_val)[:, 1]
            fold_aucs.append(roc_auc_score(y_i_val, y_i_prob))

        if fold_aucs:
            mean_auc = float(np.mean(fold_aucs))
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = params

    return best_params


# ---------------------------------------------------------------------------
# Public API: per-profile CV
# ---------------------------------------------------------------------------

def cross_validate_profile(
    df: pd.DataFrame,
    profile: str,
    seed: int = 42,
    warmup: int = WARMUP_INTERACTIONS,
    n_outer_splits: int = N_OUTER_SPLITS,
    n_estimators: int = N_ESTIMATORS,
) -> ProfileCVResult:
    """Run time-series cross-validation for one learner profile.

    Methodological Note:
    -------------------
    Because multiple synthetic learners share the same interaction-time index,
    forward-chaining splits are performed over unique interaction time steps
    rather than individual observation rows. This ensures that all learners at a given
    time step belong to the same temporal partition and guarantees strictly later
    validation times (max(train_interaction_id) < min(val_interaction_id)).

    CLSI-Adapt trains one pooled XGBoost model per behavioral profile using multiple
    independent learner traces of that profile.

    Parameters
    ----------
    df:
        Profile interaction DataFrame (may contain multiple learners).
    profile:
        Profile name string (used for labeling outputs).
    seed:
        Random seed for reproducibility.
    warmup:
        Warm-up interactions to exclude from prediction.
    n_outer_splits:
        Number of outer TimeSeriesSplit folds.
    n_estimators : int
        Number of boosting rounds.

    Returns
    -------
    ProfileCVResult
        Contains fold-level results, out-of-fold predictions, mean/std AUC,
        chosen final hyperparameters, and the trained final model.
    """
    X_elig, y_elig, positions, interaction_ids, learner_ids = _prepare_profile_data(df, warmup=warmup)

    iids_arr = interaction_ids.to_numpy()
    lids_arr = learner_ids.to_numpy()

    # Get unique, chronologically ordered interaction time steps
    unique_times = np.sort(np.unique(iids_arr))

    if len(unique_times) >= n_outer_splits + 1:
        outer_cv = TimeSeriesSplit(n_splits=n_outer_splits)
        outer_splits = []
        for u_tr_idx, u_val_idx in outer_cv.split(unique_times):
            u_tr_set = set(unique_times[u_tr_idx])
            u_val_set = set(unique_times[u_val_idx])
            train_idx = np.where(np.isin(iids_arr, list(u_tr_set)))[0]
            val_idx = np.where(np.isin(iids_arr, list(u_val_set)))[0]
            outer_splits.append((train_idx, val_idx))
    else:
        outer_cv = TimeSeriesSplit(n_splits=n_outer_splits)
        outer_splits = list(outer_cv.split(X_elig))

    fold_results: List[FoldResult] = []

    oof_y_prob = np.full(len(y_elig), np.nan)
    oof_y_pred = np.full(len(y_elig), np.nan)

    hp_votes: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_splits):
        tr_iids = iids_arr[train_idx]
        val_iids = iids_arr[val_idx]
        tr_lids = lids_arr[train_idx]
        val_lids = lids_arr[val_idx]

        # ── Rigorous outer-fold invariant assertions ────────────────────────
        assert tr_iids.max() < val_iids.min(), (
            f"Fold {fold_idx}: temporal ordering violation. "
            f"max_train_iid={tr_iids.max()} >= min_val_iid={val_iids.min()}"
        )

        assert len(set(zip(tr_lids, tr_iids)).intersection(set(zip(val_lids, val_iids)))) == 0, (
            f"Fold {fold_idx}: duplicate (learner_id, interaction_id) across train and validation."
        )

        assert len(set(tr_iids).intersection(set(val_iids))) == 0, (
            f"Fold {fold_idx}: simultaneous time leakage! Shared time steps present in both train and validation."
        )

        X_tr, X_val = X_elig[train_idx], X_elig[val_idx]
        y_tr, y_val = y_elig[train_idx], y_elig[val_idx]

        # Skip folds with insufficient class diversity
        if len(np.unique(y_tr)) < 2:
            continue

        # ── Nested hyperparameter search on outer TRAINING portion only ──
        best_params = tune_hyperparameters(
            X_tr, y_tr, interaction_ids_train=tr_iids, seed=seed,
            n_inner_splits=N_INNER_SPLITS, n_estimators=n_estimators,
        )
        hp_votes.append(best_params)

        # ── Train outer model ─────────────────────────────────────────────
        spw = _make_scale_pos_weight(y_tr)
        clf = _build_xgb_classifier(best_params, spw, seed, n_estimators=n_estimators)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)

        # ── Predict on outer validation ───────────────────────────────────
        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fold_auc = float(roc_auc_score(y_val, y_prob)) if len(np.unique(y_val)) >= 2 else float("nan")

        oof_y_prob[val_idx] = y_prob
        oof_y_pred[val_idx] = y_pred

        fold_results.append(
            FoldResult(
                fold_idx=fold_idx,
                profile=profile,
                val_indices=val_idx,
                y_true=y_val,
                y_prob=y_prob,
                y_pred=y_pred,
                auc=fold_auc,
                best_params=best_params,
            )
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────
    valid_aucs = [fr.auc for fr in fold_results if not np.isnan(fr.auc)]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    std_auc = float(np.std(valid_aucs)) if valid_aucs else float("nan")

    # ── Select final hyperparameters (majority vote across folds) ─────────
    final_best_params = _majority_vote_params(hp_votes)

    # ── Build OOF prediction DataFrame ───────────────────────────────────
    covered_mask = ~np.isnan(oof_y_prob)
    oof_df = pd.DataFrame(
        {
            "profile": profile,
            "learner_id": lids_arr[covered_mask],
            "interaction_id": iids_arr[covered_mask],
            "row_position": positions[covered_mask],
            "y_true": y_elig[covered_mask].astype(int),
            "y_prob": oof_y_prob[covered_mask],
            "y_pred": oof_y_pred[covered_mask].astype(int),
        }
    )

    # ── Train final model on all eligible data ────────────────────────────
    final_model, X_final, y_final = train_final_model(
        X_elig, y_elig, final_best_params, seed=seed, n_estimators=n_estimators
    )

    return ProfileCVResult(
        profile=profile,
        fold_results=fold_results,
        oof_predictions=oof_df,
        mean_auc=mean_auc,
        std_auc=std_auc,
        final_best_params=final_best_params,
        final_model=final_model,
        X_final=X_final,
        y_final=y_final,
    )


def _majority_vote_params(hp_votes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the hyperparameter dict that appeared most often across folds.

    Ties are broken by taking the first occurrence.  If hp_votes is empty,
    returns the first entry in HP_GRID.
    """
    if not hp_votes:
        return HP_GRID[0]

    # Convert each dict to a hashable key for counting
    counts: Dict[str, int] = {}
    key_to_params: Dict[str, Dict[str, Any]] = {}
    for params in hp_votes:
        key = str(sorted(params.items()))
        counts[key] = counts.get(key, 0) + 1
        key_to_params[key] = params

    best_key = max(counts, key=lambda k: counts[k])
    return key_to_params[best_key]


# ---------------------------------------------------------------------------
# Public API: final model training
# ---------------------------------------------------------------------------

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    best_params: Dict[str, Any],
    seed: int = 42,
    n_estimators: int = N_ESTIMATORS,
) -> Tuple[xgb.XGBClassifier, np.ndarray, np.ndarray]:
    """Train a final XGBoost model on all provided data.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (all eligible rows for a profile).
    y : np.ndarray
        Binary overload labels.
    best_params : dict
        Hyperparameters (max_depth, learning_rate) chosen by inner CV.
    seed : int
        Random seed.

    Returns
    -------
    model : xgb.XGBClassifier
        Fitted final model.
    X : np.ndarray
        The same feature matrix (passed through for SHAP convenience).
    y : np.ndarray
        The same labels (passed through).
    """
    spw = _make_scale_pos_weight(y)
    clf = _build_xgb_classifier(best_params, spw, seed, n_estimators=n_estimators)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    return clf, X, y


# ---------------------------------------------------------------------------
# Public API: profile model training (single-profile entry point)
# ---------------------------------------------------------------------------

def train_profile_model(
    df: pd.DataFrame,
    profile: str,
    seed: int = 42,
    warmup: int = WARMUP_INTERACTIONS,
    n_estimators: int = N_ESTIMATORS,
) -> ProfileCVResult:
    """Full train pipeline for a single learner profile.

    Convenience wrapper around ``cross_validate_profile``.

    Parameters
    ----------
    df:
        Single-profile interaction DataFrame.
    profile:
        Profile name.
    seed:
        Random seed.
    warmup:
        Warm-up interactions to exclude from prediction.
    n_estimators:
        Number of boosting rounds. Pass ``TEST_N_ESTIMATORS`` in unit tests
        to keep runtimes short without changing production behaviour.

    Returns
    -------
    ProfileCVResult
    """
    return cross_validate_profile(
        df, profile=profile, seed=seed, warmup=warmup, n_estimators=n_estimators
    )


# ---------------------------------------------------------------------------
# Public API: prediction
# ---------------------------------------------------------------------------

def predict_profile(
    model: xgb.XGBClassifier,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run predictions using a trained profile model.

    Parameters
    ----------
    model:
        Trained XGBClassifier.
    X:
        Feature matrix, shape (n, 11).

    Returns
    -------
    y_prob : np.ndarray, shape (n,)
        Predicted overload probability.
    y_pred : np.ndarray, shape (n,)
        Binary predicted label (threshold 0.5).
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return y_prob, y_pred


# ---------------------------------------------------------------------------
# Public API: SHAP
# ---------------------------------------------------------------------------

def _patch_shap_ubjson() -> None:
    """Monkey-patch shap's UBJSON decoder to handle XGBoost 3.x array-formatted base_score '[0E0]'."""
    try:
        import shap.explainers._tree as _shap_tree
        if hasattr(_shap_tree, "decode_ubjson_buffer") and not getattr(_shap_tree.decode_ubjson_buffer, "_bs_patched", False):
            orig_decode = _shap_tree.decode_ubjson_buffer
            def _wrapped_decode(*args, **kwargs):
                jmodel = orig_decode(*args, **kwargs)
                try:
                    lmp = jmodel.get("learner", {}).get("learner_model_param", {})
                    bs = lmp.get("base_score")
                    if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                        lmp["base_score"] = bs[1:-1]
                except Exception:
                    pass
                return jmodel
            _wrapped_decode._bs_patched = True
            _shap_tree.decode_ubjson_buffer = _wrapped_decode
    except Exception:
        pass


def compute_shap(
    model: Any,
    X: np.ndarray,
    feature_names: List[str] = FEATURE_NAMES,
    max_display: int = 11,
) -> Tuple[shap.Explanation, shap.TreeExplainer]:
    """Compute SHAP values using TreeExplainer on a trained XGBClassifier.

    Parameters
    ----------
    model:
        Trained XGBClassifier.
    X:
        Feature matrix used for SHAP calculation, shape (n, 11).
        Typically the full eligible training set (X_final).
    feature_names:
        Ordered list of feature names (must match X column order).
    max_display:
        Number of features to show in summary plot.

    Returns
    -------
    shap_values : shap.Explanation
        SHAP Explanation object (values, base_values, data).
    explainer : shap.TreeExplainer
        The fitted TreeExplainer instance.
    """
    _patch_shap_ubjson()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    # Attach feature names to the Explanation for plotting
    shap_values.feature_names = feature_names
    return shap_values, explainer


def plot_shap_summary(
    shap_values: shap.Explanation,
    X: np.ndarray,
    feature_names: List[str] = FEATURE_NAMES,
    max_display: int = 11,
    title: str = "SHAP Feature Importance",
    show: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """Generate a SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values:
        SHAP Explanation object returned by ``compute_shap``.
    X:
        Feature matrix (same as passed to ``compute_shap``).
    feature_names:
        Ordered feature names.
    max_display:
        Maximum features to display.
    title:
        Plot title.
    show:
        If True, call ``plt.show()``.
    save_path:
        If provided, save the figure to this path before showing.
    """
    import matplotlib.pyplot as plt

    # shap.summary_plot accepts either an Explanation object or raw values
    shap.summary_plot(
        shap_values,
        features=X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title(title, fontsize=13)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# Public API: full multi-profile training
# ---------------------------------------------------------------------------

def train_all_profiles(
    sim_data: Dict[str, pd.DataFrame],
    seed: int = 42,
    warmup: int = WARMUP_INTERACTIONS,
    save_dir: Optional[Path] = None,
) -> Dict[str, ProfileCVResult]:
    """Train CLSI-Adapt models for all learner profiles.

    Parameters
    ----------
    sim_data:
        Dictionary mapping profile name → single-profile DataFrame.
    seed:
        Global random seed.
    warmup:
        Warm-up interactions to exclude from prediction.
    save_dir:
        If provided, save each final model's JSON to this directory.
        XGBoost models are saved in the portable JSON format (not binary).

    Returns
    -------
    results : dict
        Maps profile name → ProfileCVResult.
    """
    results: Dict[str, ProfileCVResult] = {}

    for profile, df in sim_data.items():
        print(f"  Training profile: {profile} ...")
        result = train_profile_model(df, profile=profile, seed=seed, warmup=warmup)
        results[profile] = result
        print(
            f"    mean_auc={result.mean_auc:.4f} ± {result.std_auc:.4f}  "
            f"best_params={result.final_best_params}"
        )

        if save_dir is not None and result.final_model is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            model_path = save_dir / f"clsi_adapt_{profile}.json"
            result.final_model.save_model(str(model_path))
            print(f"    Saved model → {model_path}")

    return results


# ---------------------------------------------------------------------------
# CLSIAdaptModel class (sklearn-style wrapper for run_all compatibility)
# ---------------------------------------------------------------------------

class CLSIAdaptModel:
    """Sklearn-style wrapper for CLSI-Adapt per-profile XGBoost models.

    This class preserves backward compatibility with ``run_all.py`` and the
    ``src.models`` package export.  For the full training pipeline, use the
    functional API (``train_all_profiles``, ``cross_validate_profile``, etc.).

    Parameters
    ----------
    seed : int
        Random seed for all underlying XGBoost models.
    warmup : int
        Warm-up interactions to exclude from prediction.
    """

    def __init__(self, seed: int = 42, warmup: int = WARMUP_INTERACTIONS) -> None:
        self.seed = seed
        self.warmup = warmup
        # Populated after fit()
        self._profile_results: Dict[str, ProfileCVResult] = {}

    def fit(
        self,
        sim_data: Dict[str, pd.DataFrame],
        save_dir: Optional[Path] = None,
    ) -> "CLSIAdaptModel":
        """Train per-profile models.

        Parameters
        ----------
        sim_data:
            Dict mapping profile name → single-profile interaction DataFrame.
        save_dir:
            Optional directory to save model JSON files.

        Returns
        -------
        self
        """
        self._profile_results = train_all_profiles(
            sim_data, seed=self.seed, warmup=self.warmup, save_dir=save_dir
        )
        return self

    def predict(self, profile: str, X: np.ndarray) -> np.ndarray:
        """Binary overload prediction for a profile's feature matrix.

        Parameters
        ----------
        profile:
            Profile name (must have been fitted).
        X:
            Feature matrix, shape (n, 11).

        Returns
        -------
        np.ndarray of int, shape (n,)
        """
        model = self._get_model(profile)
        _, y_pred = predict_profile(model, X)
        return y_pred

    def predict_proba(self, profile: str, X: np.ndarray) -> np.ndarray:
        """Overload probability for a profile's feature matrix.

        Parameters
        ----------
        profile:
            Profile name.
        X:
            Feature matrix, shape (n, 11).

        Returns
        -------
        np.ndarray of shape (n, 2): [P(no overload), P(overload)]
        """
        model = self._get_model(profile)
        y_prob, _ = predict_profile(model, X)
        return np.column_stack([1.0 - y_prob, y_prob])

    def _get_model(self, profile: str) -> xgb.XGBClassifier:
        if profile not in self._profile_results:
            raise ValueError(
                f"Profile '{profile}' not found. Fitted profiles: "
                f"{list(self._profile_results.keys())}"
            )
        model = self._profile_results[profile].final_model
        if model is None:
            raise RuntimeError(f"Final model for profile '{profile}' was not trained.")
        return model

    def get_result(self, profile: str) -> ProfileCVResult:
        """Return the full ProfileCVResult for a given profile."""
        return self._profile_results[profile]

    @property
    def profiles(self) -> List[str]:
        """List of fitted profile names."""
        return list(self._profile_results.keys())
