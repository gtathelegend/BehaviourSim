"""Education domain preset for BehaviorSim.

Generates reproducible synthetic learner interaction traces with ground-truth cognitive states
(Optimal, Overload, Underload), performance-dependent state transitions, statistical properties,
and chronological feature tracking without future data leakage.

Executed through the generic behaviorsim.core simulation engine.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.state import State
from behaviorsim.core.simulator import FeatureEvaluationContext, Simulator
from behaviorsim.core.transition import HistoryContext, TransitionRule
from behaviorsim.core.utils import derive_learner_seed


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

# Generic Core State instances for Education Domain
EDUCATION_CORE_STATES = [
    State("Optimal", description="Optimal cognitive load state"),
    State("Overload", description="High cognitive load / struggle state"),
    State("Underload", description="Low cognitive load / boredom state"),
]

# Legacy string state definitions exported for API compatibility
STATES: List[str] = [s.name for s in EDUCATION_CORE_STATES]

# Base Hidden Markov transition matrix P(state_t | state_{t-1})
BASE_TRANSITION_MATRIX: Dict[str, Dict[str, float]] = {
    "Optimal": {"Optimal": 0.70, "Overload": 0.15, "Underload": 0.15},
    "Overload": {"Optimal": 0.30, "Overload": 0.60, "Underload": 0.10},
    "Underload": {"Optimal": 0.30, "Overload": 0.10, "Underload": 0.60},
}

# Numerical 2D array representation of base transition matrix aligned with EDUCATION_CORE_STATES
EDUCATION_TRANSITION_MATRIX = np.array(
    [
        [0.70, 0.15, 0.15],
        [0.30, 0.60, 0.10],
        [0.30, 0.10, 0.60],
    ]
)


def _logistic(x: float) -> float:
    """Standard logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def build_education_transition_rules() -> List[TransitionRule]:
    """Construct domain transition rules for education performance-dependent overrides."""

    def overload_override_condition(ctx: HistoryContext) -> bool:
        recent_acc = ctx.get_recent("accuracy", 5)
        return len(recent_acc) >= 5 and float(np.mean(recent_acc)) < 0.4

    def underload_override_condition(ctx: HistoryContext) -> bool:
        recent_acc = ctx.get_recent("accuracy", 5)
        recent_nrt = ctx.get_recent("nrt", 5)
        return (
            len(recent_acc) >= 5
            and float(np.mean(recent_acc)) == 1.0
            and len(recent_nrt) >= 5
            and float(np.mean(recent_nrt)) < 0.4
        )

    return [
        TransitionRule(
            condition=overload_override_condition,
            target_state="Overload",
            probability=0.70,
        ),
        TransitionRule(
            condition=underload_override_condition,
            target_state="Underload",
            probability=0.60,
        ),
    ]


def _sample_accuracy(eval_ctx: FeatureEvaluationContext) -> int:
    """Sample item accuracy based on IRT ability theta, difficulty, and state factor."""
    difficulty = eval_ctx.history.get_recent("difficulty", 1)[-1]
    state = eval_ctx.state
    profile: LearnerProfile = eval_ctx.profile.metadata["learner_profile"]

    base_p = _logistic(profile.theta - difficulty)
    if state == "Optimal":
        state_factor = 1.0
    elif state == "Underload":
        state_factor = 0.90
    else:  # Overload
        state_factor = 1.0 - profile.overload_accuracy_drop_factor

    prob_correct = float(np.clip(base_p * state_factor, 0.0, 1.0))
    return int(eval_ctx.rng.binomial(1, prob_correct))


def _sample_nrt(eval_ctx: FeatureEvaluationContext) -> float:
    """Sample lognormal response time and return normalized response time (NRT)."""
    difficulty = eval_ctx.history.get_recent("difficulty", 1)[-1]
    state = eval_ctx.state
    profile: LearnerProfile = eval_ctx.profile.metadata["learner_profile"]

    if state == "Optimal":
        mu_rt = np.log(profile.rt_mean_optimal * (1.0 + 0.05 * (difficulty - 3)))
        sigma_rt = profile.rt_sd_optimal
    elif state == "Overload":
        mu_rt = np.log(profile.rt_mean_optimal * 1.4 * (1.0 + 0.05 * (difficulty - 3)))
        sigma_rt = profile.rt_sd_optimal * 1.25
    else:  # Underload
        mu_rt = np.log(profile.rt_mean_optimal * profile.underload_rt_factor)
        sigma_rt = profile.rt_sd_optimal

    raw_rt = max(0.1, float(eval_ctx.rng.lognormal(mean=mu_rt, sigma=sigma_rt)))
    eval_ctx.history.record("raw_rt", raw_rt)

    nrt = max(0.01, float(raw_rt / profile.rt_mean_optimal))
    return nrt


def _sample_retries(eval_ctx: FeatureEvaluationContext) -> int:
    """Sample retry count based on accuracy and cognitive state."""
    accuracy = eval_ctx.history.get_recent("accuracy", 1)[-1]
    state = eval_ctx.state
    rng = eval_ctx.rng

    if accuracy == 0 and state == "Overload":
        if rng.random() < 0.40:
            return int(rng.choice([1, 2, 3]))
        return 0
    else:
        if accuracy == 0:
            return int(rng.choice([0, 1]))
        return 0


def _sample_help(eval_ctx: FeatureEvaluationContext) -> int:
    """Sample help request flag based on cognitive state."""
    state = eval_ctx.state
    profile: LearnerProfile = eval_ctx.profile.metadata["learner_profile"]

    if state == "Overload":
        p_help = profile.help_prob_overload
    elif state == "Optimal":
        p_help = profile.help_prob_optimal
    else:  # Underload
        p_help = 0.02

    return int(eval_ctx.rng.random() < p_help)


def _sample_confidence(eval_ctx: FeatureEvaluationContext) -> int:
    """Sample confidence score (1-5) based on state-dependent categorical distribution."""
    state = eval_ctx.state
    rng = eval_ctx.rng

    if state == "Overload":
        probs = [0.40, 0.35, 0.15, 0.07, 0.03]
    elif state == "Optimal":
        probs = [0.05, 0.15, 0.50, 0.20, 0.10]
    else:  # Underload
        probs = [0.02, 0.08, 0.20, 0.40, 0.30]

    return int(rng.choice([1, 2, 3, 4, 5], p=probs))


def create_education_core_profile(learner_profile: LearnerProfile) -> Profile:
    """Construct generic Profile abstraction from an education LearnerProfile."""
    rules = build_education_transition_rules()
    state_emissions = {
        state_name: {
            "difficulty": FeatureDistribution("uniform_discrete", {"items": [1, 2, 3, 4, 5]}),
            "accuracy": _sample_accuracy,
            "nrt": _sample_nrt,
            "retries": _sample_retries,
            "help_requested": _sample_help,
            "confidence": _sample_confidence,
        }
        for state_name in STATES
    }
    return Profile(
        name=learner_profile.name,
        state_emissions=state_emissions,
        transition_matrix=EDUCATION_TRANSITION_MATRIX,
        transition_rules=rules,
        metadata={"learner_profile": learner_profile},
    )


def create_education_simulator(
    profile: Union[str, LearnerProfile] = "average",
    initial_state: str = "Optimal",
    **kwargs: Any,
) -> Simulator:
    """Factory creating a generic Simulator configured with Education domain specifications.

    Args:
        profile: Learner profile name (str) or LearnerProfile instance. Defaults to "average".
        initial_state: Name of initial state. Defaults to "Optimal".
        **kwargs: Additional parameters passed to Simulator (or 'profile_name').

    Returns:
        Configured Simulator instance for the education domain.
    """
    if "profile_name" in kwargs:
        profile = kwargs.pop("profile_name")

    if isinstance(profile, str):
        if profile not in LEARNER_PROFILES:
            raise ValueError(
                f"Unknown profile '{profile}'. Must be one of {list(LEARNER_PROFILES.keys())}"
            )
        lp = LEARNER_PROFILES[profile]
    elif isinstance(profile, LearnerProfile):
        lp = profile
    else:
        raise TypeError(
            f"profile must be a str or LearnerProfile, got {type(profile).__name__}."
        )

    core_profile = create_education_core_profile(lp)
    return Simulator(
        states=EDUCATION_CORE_STATES,
        profile=core_profile,
        initial_state=initial_state,
        **kwargs,
    )


def _get_default_config() -> Any:
    """Retrieve default configuration if src.config is available, otherwise a minimal dummy config."""
    try:
        from src.config import default_config  # type: ignore
        return default_config
    except ImportError:
        from dataclasses import make_dataclass
        DummyConfig = make_dataclass(
            "DummyConfig",
            [
                ("num_interactions_per_learner", int, 1000),
                ("num_learners_per_profile", int, 10),
                ("seed", int, 42),
                ("feature_window_size", int, 5),
                ("data_dir", Path, Path("data")),
            ],
        )
        return DummyConfig()


def simulate_learner(
    profile_name: str,
    num_interactions: Optional[int] = None,
    learner_id: int = 1,
    seed: Optional[int] = None,
    config: Optional[Any] = None,
) -> pd.DataFrame:
    """Simulate interaction log for a single independent synthetic learner via core Simulator.

    Args:
        profile_name: Name of the learner profile (must be one of LEARNER_PROFILES keys).
        num_interactions: Number of interactions to simulate. If None, uses config.num_interactions_per_learner.
        learner_id: Numeric identifier for the learner sequence.
        seed: Random seed for reproducibility. If None, uses config.seed.
        config: Configuration instance containing default parameters.

    Returns:
        DataFrame containing simulated interaction sequence and engineered features.
    """
    if profile_name not in LEARNER_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile_name}'. Must be one of {list(LEARNER_PROFILES.keys())}"
        )

    cfg = config if config is not None else _get_default_config()
    n_interactions = num_interactions if num_interactions is not None else cfg.num_interactions_per_learner
    learner_profile = LEARNER_PROFILES[profile_name]
    effective_seed = seed if seed is not None else cfg.seed
    window_size = cfg.feature_window_size

    simulator = create_education_simulator(profile=learner_profile, initial_state="Optimal")

    # Execute simulation using core Simulator engine
    sim_df = simulator.simulate(num_interactions=n_interactions, seed=effective_seed)

    # Post-process derived metrics to build exact education schema
    accuracies = sim_df["accuracy"].to_numpy(dtype=int)
    nrts = sim_df["nrt"].to_numpy(dtype=float)

    streak_correct_list: List[int] = []
    streak_incorrect_list: List[int] = []
    window_error_rate_list: List[float] = []
    nrt_variance_list: List[float] = []
    session_time_list: List[float] = []

    current_streak_correct = 0
    current_streak_incorrect = 0
    cumulative_session_time = 0.0

    # Extract raw_rt values from simulation history trace implicitly reconstructed
    raw_rts = nrts * learner_profile.rt_mean_optimal

    for i in range(n_interactions):
        acc = accuracies[i]
        if acc == 1:
            current_streak_correct += 1
            current_streak_incorrect = 0
        else:
            current_streak_incorrect += 1
            current_streak_correct = 0

        streak_correct_list.append(current_streak_correct)
        streak_incorrect_list.append(current_streak_incorrect)

        raw_rt = raw_rts[i]
        cumulative_session_time += raw_rt
        session_time_list.append(cumulative_session_time)

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

    # Construct final DataFrame matching exact historical schema
    df = pd.DataFrame(
        {
            "profile": profile_name,
            "learner_id": learner_id,
            "interaction_id": np.arange(1, n_interactions + 1, dtype=int),
            "state": sim_df["state"].to_list(),
            "difficulty": sim_df["difficulty"].astype(int).to_list(),
            "accuracy": accuracies,
            "nrt": nrts,
            "window_error_rate": window_error_rate_list,
            "retries": sim_df["retries"].astype(int).to_list(),
            "help_requested": sim_df["help_requested"].astype(int).to_list(),
            "confidence": sim_df["confidence"].astype(int).to_list(),
            "streak_correct": streak_correct_list,
            "streak_incorrect": streak_incorrect_list,
            "nrt_variance": nrt_variance_list,
            "session_time": session_time_list,
        }
    )

    return df


def simulate_all_profiles(
    num_interactions: Optional[int] = None,
    num_learners: Optional[int] = None,
    seed: int = 42,
    config: Optional[Any] = None,
) -> Dict[str, pd.DataFrame]:
    """Simulate interaction datasets for all registered learner profiles across multiple independent learners.

    Args:
        num_interactions: Number of interactions per learner sequence. If None, uses config.num_interactions_per_learner.
        num_learners: Number of independent learners per profile. If None, uses config.num_learners_per_profile.
        seed: Base random seed. Each profile and learner derives a distinct deterministic seed.
        config: Configuration instance.

    Returns:
        Dictionary mapping profile_name to its concatenated multi-learner DataFrame.
    """
    cfg = config if config is not None else _get_default_config()
    n_interactions = num_interactions if num_interactions is not None else cfg.num_interactions_per_learner
    n_learners = num_learners if num_learners is not None else cfg.num_learners_per_profile

    results: Dict[str, pd.DataFrame] = {}
    for p_idx, profile_name in enumerate(LEARNER_PROFILES.keys()):
        learner_dfs: List[pd.DataFrame] = []
        for l_idx in range(1, n_learners + 1):
            learner_seed = derive_learner_seed(seed, p_idx, l_idx)
            df_learner = simulate_learner(
                profile_name=profile_name,
                num_interactions=n_interactions,
                learner_id=l_idx,
                seed=learner_seed,
                config=cfg,
            )
            learner_dfs.append(df_learner)

        results[profile_name] = pd.concat(learner_dfs, ignore_index=True)
    return results


def save_simulation(
    simulation_results: Dict[str, pd.DataFrame],
    output_dir: Optional[Path] = None,
    config: Optional[Any] = None,
) -> List[Path]:
    """Save simulation DataFrames to CSV files in output_dir.

    Args:
        simulation_results: Dictionary mapping profile names to DataFrames.
        output_dir: Target directory path. If None, uses config.data_dir.
        config: Configuration instance.

    Returns:
        List of saved CSV file paths.
    """
    cfg = config if config is not None else _get_default_config()
    target_dir = output_dir if output_dir is not None else getattr(cfg, "data_dir", Path("data"))
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


def validate_simulation_diagnostics(
    sim_data: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Calculate and report diagnostic state and target distributions per profile and per learner.

    Args:
        sim_data: Dictionary mapping profile names to simulated DataFrames.

    Returns:
        Dict with keys 'profile_summary' and 'learner_summary' DataFrames.
    """
    try:
        from src.feature_engineering import build_overload_target  # type: ignore
    except ImportError:
        raise ImportError(
            "validate_simulation_diagnostics requires the legacy CLSI-Adapt 'src' package. "
            "Please ensure 'src' is accessible on sys.path to run this diagnostic."
        )

    profile_rows: List[Dict[str, Any]] = []
    learner_rows: List[Dict[str, Any]] = []

    for profile_name, df_profile in sim_data.items():
        learners = df_profile["learner_id"].unique() if "learner_id" in df_profile.columns else [1]
        n_learners = len(learners)
        total_interactions = len(df_profile)

        profile_valid = 0
        profile_pos = 0

        for l_id in learners:
            df_l = df_profile[df_profile["learner_id"] == l_id] if "learner_id" in df_profile.columns else df_profile
            target_l = build_overload_target(df_l)
            valid_mask = ~target_l.isna()
            n_valid = int(valid_mask.sum())
            n_pos = int((target_l[valid_mask] == 1.0).sum())

            profile_valid += n_valid
            profile_pos += n_pos

            state_counts = df_l["state"].value_counts().to_dict()
            opt_cnt = state_counts.get("Optimal", 0)
            ov_cnt = state_counts.get("Overload", 0)
            und_cnt = state_counts.get("Underload", 0)

            learner_rows.append(
                {
                    "profile": profile_name,
                    "learner_id": l_id,
                    "total_interactions": len(df_l),
                    "valid_targets": n_valid,
                    "positive_targets": n_pos,
                    "positive_rate": n_pos / max(1, n_valid),
                    "optimal_count": opt_cnt,
                    "overload_count": ov_cnt,
                    "underload_count": und_cnt,
                }
            )

        pos_rate = profile_pos / max(1, profile_valid)
        profile_rows.append(
            {
                "profile": profile_name,
                "n_learners": n_learners,
                "total_interactions": total_interactions,
                "valid_targets": profile_valid,
                "positive_targets": profile_pos,
                "positive_rate": pos_rate,
            }
        )

    profile_df = pd.DataFrame(profile_rows)
    learner_df = pd.DataFrame(learner_rows)

    return {
        "profile_summary": profile_df,
        "learner_summary": learner_df,
    }


def run_simulation(config: Optional[Any] = None) -> Dict[str, pd.DataFrame]:
    """Run full simulation pipeline step and save outputs.

    Args:
        config: Configuration parameters for the simulation.

    Returns:
        Dictionary of simulated DataFrames for all profiles.
    """
    cfg = config if config is not None else _get_default_config()
    results = simulate_all_profiles(
        num_interactions=cfg.num_interactions_per_learner,
        num_learners=cfg.num_learners_per_profile,
        seed=cfg.seed,
        config=cfg,
    )
    save_simulation(results, output_dir=getattr(cfg, "data_dir", Path("data")), config=cfg)
    return results
