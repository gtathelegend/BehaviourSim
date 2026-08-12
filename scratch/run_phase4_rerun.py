import sys
import os
import io
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Ensure src is importable
sys.path.insert(0, os.path.abspath('.'))

from src.config import Config, default_config
from src.feature_engineering import build_features, build_overload_target, FEATURE_COLUMNS
from src.models.clsi_adapt import (
    cross_validate_profile,
    _prepare_profile_data,
    compute_shap,
    WARMUP_INTERACTIONS,
    FEATURE_NAMES,
)

def run_phase4_audit():
    buf = io.StringIO()
    def p(text=""):
        buf.write(text + "\n")
        print(text)

    p("=" * 80)
    p("CLSI-ADAPT PHASE 4 CLEAN RERUN AUDIT")
    p("=" * 80)
    
    # Load dataset
    df_all = pd.read_csv("data/simulated_learners_all.csv")
    profiles = ["fast_accurate", "slow_accurate", "average", "fast_inaccurate", "slow_inaccurate"]
    
    p("\n1. DATASET OVERVIEW & TARGET DISTRIBUTION")
    p("-" * 80)
    target_dist = {}
    for prof in profiles:
        p_df = df_all[df_all["profile"] == prof].copy()
        n_learners = p_df["learner_id"].nunique()
        target = build_overload_target(p_df)
        valid_mask = ~target.isna()
        n_valid = int(valid_mask.sum())
        n_pos = int((target[valid_mask] == 1.0).sum())
        n_neg = int((target[valid_mask] == 0.0).sum())
        pos_rate = n_pos / n_valid if n_valid > 0 else 0.0
        
        zero_pos_learners = 0
        for l_id, l_df in p_df.groupby("learner_id"):
            l_target = build_overload_target(l_df)
            l_valid = ~l_target.isna()
            if (l_target[l_valid] == 1.0).sum() == 0:
                zero_pos_learners += 1
                
        target_dist[prof] = {
            "n_learners": n_learners,
            "n_valid": n_valid,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": pos_rate,
            "zero_pos_learners": zero_pos_learners
        }
        p(f"Profile: {prof:<16} | Learners: {n_learners} | Valid Targets: {n_valid} | "
          f"Positives: {n_pos:<4} | Negatives: {n_neg:<5} | Pos Rate: {pos_rate:.6f} | Zero-Pos Learners: {zero_pos_learners}")

    p("\n2. OUTER CROSS-VALIDATION DETAILS PER FOLD")
    p("-" * 80)
    
    cv_results = {}
    for prof in profiles:
        p(f"\n>>> Profile: {prof}")
        p_df = df_all[df_all["profile"] == prof].copy()
        
        X_elig, y_elig, positions, interaction_ids, learner_ids = _prepare_profile_data(p_df, warmup=WARMUP_INTERACTIONS)
        iids_arr = interaction_ids.to_numpy()
        lids_arr = learner_ids.to_numpy()
        
        cv_res = cross_validate_profile(p_df, profile=prof, seed=42, warmup=WARMUP_INTERACTIONS)
        cv_results[prof] = cv_res
        
        unique_times = np.sort(np.unique(iids_arr))
        from sklearn.model_selection import TimeSeriesSplit
        outer_cv = TimeSeriesSplit(n_splits=5)
        
        for fold_idx, (u_tr_idx, u_val_idx) in enumerate(outer_cv.split(unique_times)):
            u_tr_set = set(unique_times[u_tr_idx])
            u_val_set = set(unique_times[u_val_idx])
            train_idx = np.where(np.isin(iids_arr, list(u_tr_set)))[0]
            val_idx = np.where(np.isin(iids_arr, list(u_val_set)))[0]
            
            tr_iids = iids_arr[train_idx]
            val_iids = iids_arr[val_idx]
            
            max_train_time = int(tr_iids.max())
            min_val_time = int(val_iids.min())
            
            assert max_train_time < min_val_time, f"Fold {fold_idx} temporal ordering violation!"
            
            y_tr = y_elig[train_idx]
            y_val = y_elig[val_idx]
            
            n_tr = len(y_tr)
            n_val = len(y_val)
            n_tr_pos = int(y_tr.sum())
            n_val_pos = int(y_val.sum())
            
            fr = next((f for f in cv_res.fold_results if f.fold_idx == fold_idx), None)
            if fr is not None:
                if np.isnan(fr.auc):
                    auc_str = "undefined (single validation class)"
                else:
                    auc_str = f"{fr.auc:.4f}"
            else:
                if len(np.unique(y_tr)) < 2:
                    auc_str = "skipped (0 training positives)"
                elif len(np.unique(y_val)) < 2:
                    auc_str = "undefined (single validation class)"
                else:
                    auc_str = "skipped"
                    
            p(f"  Fold {fold_idx+1}: n_train={n_tr:<4} (pos={n_tr_pos:<3}), n_val={n_val:<4} (pos={n_val_pos:<3}) | "
              f"max_train_time={max_train_time:<4} < min_val_time={min_val_time:<4} | AUC = {auc_str}")

    p("\n3. OUT-OF-FOLD (OOF) PREDICTION COVERAGE & PROBABILITY STATS")
    p("-" * 80)
    oof_stats = {}
    for prof in profiles:
        cv_res = cv_results[prof]
        p_df = df_all[df_all["profile"] == prof].copy()
        X_elig, y_elig, positions, interaction_ids, learner_ids = _prepare_profile_data(p_df, warmup=WARMUP_INTERACTIONS)
        
        n_expected = len(y_elig)
        oof_df = cv_res.oof_predictions
        n_oof = len(oof_df)
        n_missing = n_expected - n_oof
        
        if n_oof > 0:
            probs = oof_df["y_prob"].to_numpy()
            n_pos_oof = int((oof_df["y_true"] == 1).sum())
            n_neg_oof = int((oof_df["y_true"] == 0).sum())
            n_uniq = len(np.unique(probs))
            min_p = float(np.min(probs))
            max_p = float(np.max(probs))
            mean_p = float(np.mean(probs))
        else:
            probs = np.array([])
            n_pos_oof = int(y_elig.sum())
            n_neg_oof = int((y_elig == 0).sum())
            n_uniq = 0
            min_p = np.nan
            max_p = np.nan
            mean_p = np.nan
            
        oof_stats[prof] = {
            "n_expected": n_expected,
            "n_oof": n_oof,
            "n_missing": n_missing,
            "n_pos": n_pos_oof,
            "n_neg": n_neg_oof,
            "n_uniq": n_uniq,
            "min_p": min_p,
            "max_p": max_p,
            "mean_p": mean_p,
        }
        
        p(f"Profile: {prof:<16} | Expected: {n_expected} | OOF Preds: {n_oof:<5} | Missing: {n_missing:<5} | "
          f"Prob Range: [{min_p:.6f}, {max_p:.6f}] | Mean Prob: {mean_p:.6f} | Unique Probs: {n_uniq}")

    p("\n4. OVERALL PER-PROFILE AUC")
    p("-" * 80)
    for prof in profiles:
        cv_res = cv_results[prof]
        if not np.isnan(cv_res.mean_auc):
            p(f"Profile: {prof:<16} | Mean AUC: {cv_res.mean_auc:.4f} ± {cv_res.std_auc:.4f}")
        else:
            p(f"Profile: {prof:<16} | Mean AUC: undefined (insufficient positive samples in validation folds)")

    p("\n5. FINAL MODEL STATISTICS PER PROFILE")
    p("-" * 80)
    final_stats = {}
    for prof in profiles:
        cv_res = cv_results[prof]
        model = cv_res.final_model
        X_final = cv_res.X_final
        
        if model is not None:
            booster = model.get_booster()
            dump = booster.get_dump()
            n_trees = len(dump)
            
            n_splits = 0
            for tree_str in dump:
                lines = tree_str.strip().split("\n")
                for line in lines:
                    if "leaf" not in line and "[" in line:
                        n_splits += 1
                        
            importances = model.feature_importances_
            imp_dict = dict(zip(FEATURE_NAMES, importances))
            
            y_prob_final = model.predict_proba(X_final)[:, 1]
            p_min = float(np.min(y_prob_final))
            p_max = float(np.max(y_prob_final))
            p_mean = float(np.mean(y_prob_final))
            p_std = float(np.std(y_prob_final))
            u_cnt = len(np.unique(y_prob_final))
        else:
            n_trees = 0
            n_splits = 0
            imp_dict = {f: 0.0 for f in FEATURE_NAMES}
            p_min = p_max = p_mean = p_std = np.nan
            u_cnt = 0
            
        final_stats[prof] = {
            "n_estimators": n_trees,
            "n_splits": n_splits,
            "importances": imp_dict,
            "p_min": p_min,
            "p_max": p_max,
            "p_mean": p_mean,
            "p_std": p_std,
            "unique_prediction_count": u_cnt,
        }
        
        p(f"\n>>> Profile: {prof}")
        p(f"  Trees: {n_trees} | Total Split Nodes: {n_splits} | Unique Preds: {u_cnt}")
        p(f"  Pred Stats: min={p_min:.6f}, max={p_max:.6f}, mean={p_mean:.6f}, std={p_std:.6f}")
        p("  Feature Importances:")
        for fname, val in imp_dict.items():
            if val > 0:
                p(f"    {fname:<20}: {val:.4f}")
            else:
                p(f"    {fname:<20}: 0.0000")

    p("\n6. SHAP DIAGNOSTICS (AVERAGE PROFILE)")
    p("-" * 80)
    avg_res = cv_results["average"]
    avg_model = avg_res.final_model
    X_avg = avg_res.X_final
    
    shap_vals, explainer = compute_shap(avg_model, X_avg)
    vals = shap_vals.values  # shape (n_samples, n_features)
    
    mean_abs_shap = np.mean(np.abs(vals), axis=0)
    max_abs_shap = float(np.max(np.abs(vals)))
    n_nonzero_shap = int(np.sum(vals != 0))
    
    mean_abs_shap_dict = dict(zip(FEATURE_NAMES, mean_abs_shap))
    
    p(f"Max |SHAP| across all features & samples: {max_abs_shap:.6f}")
    p(f"Number of Non-zero SHAP values: {n_nonzero_shap} / {vals.size}")
    p("Mean |SHAP| per feature:")
    for fname, val in mean_abs_shap_dict.items():
        p(f"  {fname:<20}: {val:.6f}")

    with open("scratch/phase4_raw_output.txt", "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

if __name__ == "__main__":
    run_phase4_audit()
