"""Synthetic Learner Simulator module for CLSI-Adapt.

Generates reproducible synthetic learner interaction traces with ground-truth cognitive states
(Optimal, Overload, Underload), performance-dependent state transitions, statistical properties,
and chronological feature tracking without future data leakage.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import Config, default_config


@dataclass
class LearnerProfile:
    """Explicit parameters defining a synthetic learner profile."""

    name: str
    theta: float
    rt_mean_optimal: float
    rt_sd_optimal: float
    overload_accuracy_drop_factor: float
    underload_rt_factor: float
    help_prob_overload: float
    help_prob_optimal: float


# Centralized registry of the 5 learner profiles
LEARNER_PROFILES: Dict[str, LearnerProfile] = {
    "fast_accurate": LearnerProfile(
        name="fast_accurate",
        theta=2.0,
        rt_mean_optimal=10.0,
        rt_sd_optimal=0.3,
        overload_accuracy_drop_factor=0.40,
        underload_rt_factor=0.60,
        help_prob_overload=0.50,
        help_prob_optimal=0.10,
    ),
    "fast_inaccurate": LearnerProfile(
        name="fast_inaccurate",
        theta=0.0,
        rt_mean_optimal=10.0,
        rt_sd_optimal=0.4,
        overload_accuracy_drop_factor=0.50,
        underload_rt_factor=0.50,
        help_prob_overload=0.60,
        help_prob_optimal=0.15,
    ),
    "slow_accurate": LearnerProfile(
        name="slow_accurate",
        theta=2.0,
        rt_mean_optimal=30.0,
        rt_sd_optimal=0.3,
        overload_accuracy_drop_factor=0.40,
        underload_rt_factor=0.70,
        help_prob_overload=0.40,
        help_prob_optimal=0.05,
    ),
    "slow_inaccurate": LearnerProfile(
        name="slow_inaccurate",
        theta=-1.0,
        rt_mean_optimal=30.0,
        rt_sd_optimal=0.5,
        overload_accuracy_drop_factor=0.60,
        underload_rt_factor=0.60,
        help_prob_overload=0.70,
        help_prob_optimal=0.20,
    ),
    "average": LearnerProfile(
        name="average",
        theta=1.0,
        rt_mean_optimal=20.0,
        rt_sd_optimal=0.4,
        overload_accuracy_drop_factor=0.50,
        underload_rt_factor=0.65,
        help_prob_overload=0.50,
        help_prob_optimal=0.10,
    ),
}

# State definitions
STATES = ["Optimal", "Overload", "Underload"]

# Base Hidden Markov transition matrix P(state_t | state_{t-1})
BASE_TRANSITION_MATRIX: Dict[str, Dict[str, float]] = {
    "Optimal": {"Optimal": 0.70, "Overload": 0.15, "Underload": 0.15},
    "Overload": {"Optimal": 0.30, "Overload": 0.60, "Underload": 0.10},
    "Underload": {"Optimal": 0.30, "Overload": 0.10, "Underload": 0.60},
}


def _logistic(x: float) -> float:
    """Standard logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def simulate_learner(
    profile_name: str,
    num_interactions: int = 100,
    seed: Optional[int] = None,
    config: Config = default_config,
) -> pd.DataFrame:
    """Simulate interaction log for a single learner profile.

    Args:
        profile_name: Name of the learner profile (must be one of LEARNER_PROFILES keys).
        num_interactions: Number of interactions to simulate.
        seed: Random seed for reproducibility. If None, uses config.seed.
        config: Configuration instance containing default parameters.

    Returns:
        DataFrame containing simulated interaction sequence and engineered features.
    """
    if profile_name not in LEARNER_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile_name}'. Must be one of {list(LEARNER_PROFILES.keys())}"
        )

    profile = LEARNER_PROFILES[profile_name]
    effective_seed = seed if seed is not None else config.seed
    rng = np.random.default_rng(effective_seed)
    window_size = config.feature_window_size

    # Simulation tracking containers
    states: List[str] = []
    difficulties: List[int] = []
    accuracies: List[int] = []
    nrts: List[float] = []
    raw_rts: List[float] = []
    retries_list: List[int] = []
    help_list: List[int] = []
    confidence_list: List[int] = []
    streak_correct_list: List[int] = []
    streak_incorrect_list: List[int] = []
    window_error_rate_list: List[float] = []
    nrt_variance_list: List[float] = []
    session_time_list: List[float] = []

    current_state = "Optimal"
    current_streak_correct = 0
    current_streak_incorrect = 0
    cumulative_session_time = 0.0

    for i in range(num_interactions):
        # 1. State Transition Determination
        # Precedence & Mutual Exclusivity Documentation:
        # Rule 1 (Overload Override): Triggered if recent 5-item mean accuracy < 0.4 (probability 0.70).
        # Rule 2 (Underload Override): Triggered if recent 5-item accuracy == 1.0 AND recent 5-item mean NRT < 0.4 (probability 0.60).
        # Mutual Exclusivity: Since mean accuracy < 0.4 and mean accuracy == 1.0 are mutually exclusive conditions,
        # Rule 1 and Rule 2 can never trigger simultaneously.
        # Insufficient History: If i < 5, overrides cannot be evaluated over 5 items; base HMM transition matrix is used.
        if i >= 5:
            last_5_acc = accuracies[-5:]
            last_5_nrt = nrts[-5:]
            mean_acc_5 = float(np.mean(last_5_acc))
            mean_nrt_5 = float(np.mean(last_5_nrt))

            if mean_acc_5 < 0.4:
                if rng.random() < 0.70:
                    current_state = "Overload"
                else:
                    # Transition via base HMM matrix
                    trans_probs = BASE_TRANSITION_MATRIX[current_state]
                    current_state = str(
                        rng.choice(
                            list(trans_probs.keys()), p=list(trans_probs.values())
                        )
                    )
            elif mean_acc_5 == 1.0 and mean_nrt_5 < 0.4:
                if rng.random() < 0.60:
                    current_state = "Underload"
                else:
                    trans_probs = BASE_TRANSITION_MATRIX[current_state]
                    current_state = str(
                        rng.choice(
                            list(trans_probs.keys()), p=list(trans_probs.values())
                        )
                    )
            else:
                trans_probs = BASE_TRANSITION_MATRIX[current_state]
                current_state = str(
                    rng.choice(
                        list(trans_probs.keys()), p=list(trans_probs.values())
                    )
                )
        else:
            if i > 0:
                trans_probs = BASE_TRANSITION_MATRIX[current_state]
                current_state = str(
                    rng.choice(
                        list(trans_probs.keys()), p=list(trans_probs.values())
                    )
                )
            else:
                current_state = "Optimal"

        states.append(current_state)

        # 2. Difficulty (Uniform discrete sampling from {1, 2, 3, 4, 5})
        difficulty = int(rng.choice([1, 2, 3, 4, 5]))
        difficulties.append(difficulty)

        # 3. Accuracy
        base_p = _logistic(profile.theta - difficulty)
        if current_state == "Optimal":
            state_factor = 1.0
        elif current_state == "Underload":
            state_factor = 0.90  # Explicit slight drop due to carelessness/boredom
        else:  # Overload
            state_factor = 1.0 - profile.overload_accuracy_drop_factor

        prob_correct = float(np.clip(base_p * state_factor, 0.0, 1.0))
        accuracy = int(rng.binomial(1, prob_correct))
        accuracies.append(accuracy)

        # Update Streaks
        if accuracy == 1:
            current_streak_correct += 1
            current_streak_incorrect = 0
        else:
            current_streak_incorrect += 1
            current_streak_correct = 0

        streak_correct_list.append(current_streak_correct)
        streak_incorrect_list.append(current_streak_incorrect)

        # 4. Response Time (NRT)
        # Log-normal distribution scaled relative to profile.rt_mean_optimal
        if current_state == "Optimal":
            mu_rt = np.log(profile.rt_mean_optimal * (1.0 + 0.05 * (difficulty - 3)))
            sigma_rt = profile.rt_sd_optimal
        elif current_state == "Overload":
            mu_rt = np.log(profile.rt_mean_optimal * 1.4 * (1.0 + 0.05 * (difficulty - 3)))
            sigma_rt = profile.rt_sd_optimal * 1.25
        else:  # Underload
            mu_rt = np.log(profile.rt_mean_optimal * profile.underload_rt_factor)
            sigma_rt = profile.rt_sd_optimal

        raw_rt = max(0.1, float(rng.lognormal(mean=mu_rt, sigma=sigma_rt)))
        raw_rts.append(raw_rt)

        nrt = max(0.01, float(raw_rt / profile.rt_mean_optimal))
        nrts.append(nrt)

        cumulative_session_time += raw_rt
        session_time_list.append(cumulative_session_time)

        # 5. Retries
        # Incorrect & Overloaded -> 40% probability of 1-3 retries; Otherwise 0-1 retry.
        if accuracy == 0 and current_state == "Overload":
            if rng.random() < 0.40:
                retries = int(rng.choice([1, 2, 3]))
            else:
                retries = 0
        else:
            if accuracy == 0:
                retries = int(rng.choice([0, 1]))
            else:
                retries = 0
        retries_list.append(retries)

        # 6. Help Requests
        if current_state == "Overload":
            p_help = profile.help_prob_overload
        elif current_state == "Optimal":
            p_help = profile.help_prob_optimal
        else:  # Underload (explicit low probability)
            p_help = 0.02
        help_requested = int(rng.random() < p_help)
        help_list.append(help_requested)

        # 7. Confidence (1-5, lower in Overload, higher in Underload)
        if current_state == "Overload":
            probs = [0.40, 0.35, 0.15, 0.07, 0.03]
        elif current_state == "Optimal":
            probs = [0.05, 0.15, 0.50, 0.20, 0.10]
        else:  # Underload
            probs = [0.02, 0.08, 0.20, 0.40, 0.30]

        confidence = int(rng.choice([1, 2, 3, 4, 5], p=probs))
        confidence_list.append(confidence)

        # 8. Chronological Windowed Metrics
        window_start = max(0, i + 1 - window_size)
        window_accs = accuracies[window_start : i + 1]
        window_nrts = nrts[window_start : i + 1]

        window_error_rate = float(1.0 - np.mean(window_accs))
        window_error_rate_list.append(window_error_rate)

        if len(window_nrts) >= 2:
            nrt_var = float(np.var(window_nrts, ddof=1))
        else:
            nrt_var = 0.0
        nrt_variance_list.append(nrt_var)

    # Build final output DataFrame
    df = pd.DataFrame(
        {
            "profile": profile_name,
            "interaction_id": np.arange(1, num_interactions + 1, dtype=int),
            "state": states,
            "difficulty": difficulties,
            "accuracy": accuracies,
            "nrt": nrts,
            "window_error_rate": window_error_rate_list,
            "retries": retries_list,
            "help_requested": help_list,
            "confidence": confidence_list,
            "streak_correct": streak_correct_list,
            "streak_incorrect": streak_incorrect_list,
            "nrt_variance": nrt_variance_list,
            "session_time": session_time_list,
        }
    )

    return df


def simulate_all_profiles(
    num_interactions: int = 100,
    seed: int = 42,
    config: Config = default_config,
) -> Dict[str, pd.DataFrame]:
    """Simulate interaction datasets for all registered learner profiles.

    Args:
        num_interactions: Number of interactions per profile.
        seed: Base random seed. Each profile derives a distinct deterministic seed.
        config: Configuration instance.

    Returns:
        Dictionary mapping profile_name to its simulated DataFrame.
    """
    results: Dict[str, pd.DataFrame] = {}
    for idx, profile_name in enumerate(LEARNER_PROFILES.keys()):
        # Profile-specific seed derivation ensures independence and determinism
        profile_seed = seed + idx * 1000
        results[profile_name] = simulate_learner(
            profile_name=profile_name,
            num_interactions=num_interactions,
            seed=profile_seed,
            config=config,
        )
    return results


def save_simulation(
    simulation_results: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None,
    config: Config = default_config,
) -> List[Path]:
    """Save simulation DataFrames to CSV files in output_dir.

    Args:
        simulation_results: Dictionary mapping profile names to DataFrames.
        output_dir: Target directory path. If None, uses config.data_dir.
        config: Configuration instance.

    Returns:
        List of saved CSV file paths.
    """
    target_dir = output_dir if output_dir is not None else config.data_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    all_dfs: List[pd.DataFrame] = []

    for profile_name, df in simulation_results.items():
        file_path = target_dir / f"simulated_learner_{profile_name}.csv"
        df.to_csv(file_path, index=False)
        saved_paths.append(file_path)
        all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_path = target_dir / "simulated_learners_all.csv"
        combined_df.to_csv(combined_path, index=False)
        saved_paths.append(combined_path)

    return saved_paths


def run_simulation(config: Config = default_config) -> Dict[str, pd.DataFrame]:
    """Run full simulation pipeline step and save outputs.

    Args:
        config: Configuration parameters for the simulation.

    Returns:
        Dictionary of simulated DataFrames for all profiles.
    """
    results = simulate_all_profiles(
        num_interactions=config.num_interactions_per_learner,
        seed=config.seed,
        config=config,
    )
    save_simulation(results, output_dir=config.data_dir, config=config)
    return results
