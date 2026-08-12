"""Configuration module for CLSI-Adapt Simulator."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Central configuration parameters for the CLSI-Adapt simulator and evaluation framework."""

    # Global random seed for reproducibility
    seed: int = 42

    # Simulation settings
    num_learners_per_profile: int = 10
    num_interactions_per_learner: int = 1000

    # Feature engineering settings
    feature_window_size: int = 5

    # Evaluation settings
    cv_split_count: int = 5

    # Output directories
    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    tables_dir: Path = field(default_factory=lambda: Path("results/tables"))
    figures_dir: Path = field(default_factory=lambda: Path("results/figures"))


default_config = Config()
