"""Run Phase 5 evaluation pipeline and generate all required output tables and audit report."""

import os
import sys
import pandas as pd
from pathlib import Path

# Ensure src is in pythonpath
sys.path.insert(0, os.path.abspath('.'))

from src.config import Config, default_config
from src.simulator import run_simulation
from src.feature_engineering import extract_features
from src.models import CLSIAdaptModel, RuleBasedCLSIModel, BKTModel
from src.evaluation import evaluate_models

def main():
    config = default_config
    print("[Phase 5] Running Simulation (seed=42)...")
    sim_data = run_simulation(config)
    
    print("[Phase 5] Extracting Features...")
    X, y, meta = extract_features(sim_data, window_size=config.feature_window_size)
    features = (X, y, meta)

    print("[Phase 5] Initializing Models (CLSI-Adapt, Rule-Based CLSI, BKT)...")
    clsi_adapt = CLSIAdaptModel(seed=config.seed)
    rule_based = RuleBasedCLSIModel(seed=config.seed)
    bkt = BKTModel(seed=config.seed)

    models = {
        "clsi_adapt": clsi_adapt,
        "rule_based_clsi": rule_based,
        "bkt": bkt,
    }

    print("[Phase 5] Evaluating Models & Generating Phase 5 Artifacts...")
    results = evaluate_models(models, features, config, sim_data=sim_data)
    
    print("\n" + "=" * 80)
    print("PHASE 5 EVALUATION COMPLETE")
    print("=" * 80)
    print("\n1. PROFILE CLASSIFICATION METRICS:")
    print(results["profile_metrics"].to_string(index=False))

    print("\n2. AGGREGATE METRICS:")
    print(results["aggregate_metrics"].to_string(index=False))

    print("\n3. RECOVERY METRICS:")
    print(results["recovery_metrics"].to_string(index=False))

    print("\n4. STATE STATISTICS:")
    print(results["state_statistics"].to_string(index=False))

    print("\nArtifact files saved to:")
    print(" - results/tables/profile_metrics.csv & .tex")
    print(" - results/tables/aggregate_metrics.csv & .tex")
    print(" - results/tables/recovery_metrics.csv & .tex")
    print(" - results/tables/state_statistics.csv & .tex")
    print(" - results/evaluation_phase5_audit.txt")

if __name__ == "__main__":
    main()
