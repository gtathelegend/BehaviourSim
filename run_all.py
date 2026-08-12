"""Run all pipeline stages for CLSI-Adapt Simulator.

Pipeline sequence:
  Simulation -> Feature Engineering -> Models -> Evaluation -> Visualization
"""

from src.config import Config, default_config
from src.simulator import run_simulation
from src.feature_engineering import extract_features
from src.models import CLSIAdaptModel, RuleBasedCLSIModel, BKTModel
from src.evaluation import evaluate_models
from src.visualize import generate_plots


def main(config: Config = default_config) -> None:
    """Execute end-to-end evaluation pipeline."""
    print("Starting CLSI-Adapt Simulator pipeline...")
    
    print("[1/5] Running Simulation...")
    sim_data = run_simulation(config)
    
    print("[2/5] Extracting Features...")
    features = extract_features(sim_data, config)
    
    print("[3/5] Initializing Models...")
    models = {
        "clsi_adapt": CLSIAdaptModel(seed=config.seed),
        "rule_based_clsi": RuleBasedCLSIModel(seed=config.seed),
        "bkt": BKTModel(seed=config.seed),
    }
    
    print("[4/5] Evaluating Models...")
    results = evaluate_models(models, features, config)
    
    print("[5/5] Generating Visualizations...")
    generate_plots(results, config)
    
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
