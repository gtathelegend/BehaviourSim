"""Core simulator engine for BehaviorSim."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING, Union
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from behaviorsim.config import SimulationConfig

from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.state import State
from behaviorsim.core.transition import HistoryContext, TransitionRule
from behaviorsim.core.utils import (
    create_rng,
    sample_categorical,
    validate_transition_matrix,
)


@dataclass
class FeatureEvaluationContext:
    """Runtime context passed to dynamic parameter callables during feature emission."""

    state: str
    profile: Profile
    interaction_index: int
    history: HistoryContext
    rng: np.random.Generator


class SimulationHistory(HistoryContext):
    """In-memory sequence history tracker for causal rule and feature evaluation."""

    def __init__(self) -> None:
        self._feature_records: Dict[str, List[Any]] = {}

    def record(self, feature_name: str, value: Any) -> None:
        if feature_name not in self._feature_records:
            self._feature_records[feature_name] = []
        self._feature_records[feature_name].append(value)

    def get_recent(self, feature_name: str, window: int) -> Sequence[Any]:
        if feature_name not in self._feature_records:
            return []
        values = self._feature_records[feature_name]
        return values[-window:] if window > 0 else []

    def reset(self) -> None:
        self._feature_records.clear()


class Simulator:
    """Domain-neutral simulation engine for generating synthetic behavioral traces."""

    def __init__(
        self,
        states: Sequence[State],
        profile: Optional[Profile] = None,
        transition_matrix: Optional[np.ndarray] = None,
        initial_state: Optional[Union[str, State]] = None,
        *,
        profiles: Optional[Union[Sequence[Profile], Mapping[str, Profile]]] = None,
        profile_distribution: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Initialize Simulator engine with states and either a single Profile or profile mixture.

        Args:
            states: Non-empty sequence of discrete State instances with unique names.
            profile: Optional single Profile instance for single-profile simulation.
            transition_matrix: Optional fallback 2D stochastic transition matrix.
            initial_state: Optional initial state name or State instance. Defaults to states[0].
            profiles: Optional sequence or mapping of Profile instances for multi-profile simulation.
            profile_distribution: Optional mapping from profile name to selection probability.
                Must be specified when multiple profiles are provided (unless all profiles define
                probability). Probabilities must be non-negative, finite, and sum to 1.0.

        Raises:
            ValueError: If neither or both profile and profiles are supplied, if states or profiles
                are empty, or if matrix shapes, rules, or probabilities fail validation.
            TypeError: If arguments are of incorrect types.
        """
        if not states:
            raise ValueError("Simulator requires at least one State.")

        for i, s in enumerate(states):
            if not isinstance(s, State):
                raise TypeError(f"Element at index {i} of states is not a State instance.")

        state_names = [s.name for s in states]
        if len(state_names) != len(set(state_names)):
            raise ValueError(f"State names must be unique, got duplicate names in: {state_names}")

        self.states = list(states)
        self.state_names = state_names
        self.state_to_idx = {name: idx for idx, name in enumerate(state_names)}

        # Profile resolution
        if profile is not None and profiles is not None:
            raise ValueError("Cannot specify both 'profile' and 'profiles'. Provide one.")
        if profile is None and profiles is None:
            raise ValueError("Simulator requires either 'profile' or 'profiles'.")

        resolved_profiles: Dict[str, Profile] = {}

        if profile is not None:
            if not isinstance(profile, Profile):
                raise TypeError(f"profile must be a Profile instance, got {type(profile).__name__}.")
            resolved_profiles = {profile.name: profile}
        else:
            assert profiles is not None
            if isinstance(profiles, Mapping):
                if not profiles:
                    raise ValueError("profiles must contain at least one Profile.")
                for k, p in profiles.items():
                    if not isinstance(p, Profile):
                        raise TypeError(
                            f"All values in profiles mapping must be Profile instances, got {type(p).__name__}."
                        )
                    if k != p.name:
                        raise ValueError(f"Profile mapping key '{k}' does not match Profile.name '{p.name}'.")
                    resolved_profiles[k] = p
            elif isinstance(profiles, Sequence) and not isinstance(profiles, (str, bytes)):
                if not profiles:
                    raise ValueError("profiles must contain at least one Profile.")
                for i, p in enumerate(profiles):
                    if not isinstance(p, Profile):
                        raise TypeError(f"Element at index {i} of profiles is not a Profile instance.")
                    if p.name in resolved_profiles:
                        raise ValueError(f"Duplicate profile name '{p.name}' in profiles.")
                    resolved_profiles[p.name] = p
            else:
                raise TypeError(
                    f"profiles must be a Sequence or Mapping of Profile, got {type(profiles).__name__}."
                )

        # Profile distribution resolution & validation
        resolved_dist: Dict[str, float] = {}
        if profile_distribution is not None:
            if not isinstance(profile_distribution, Mapping):
                raise TypeError(
                    f"profile_distribution must be a mapping, got {type(profile_distribution).__name__}."
                )
            if not profile_distribution:
                raise ValueError("profile_distribution cannot be empty.")

            # Unknown profile names check
            for p_name in profile_distribution.keys():
                if p_name not in resolved_profiles:
                    raise ValueError(
                        f"Unknown profile name '{p_name}' in profile_distribution. "
                        f"Available profiles: {sorted(resolved_profiles.keys())}."
                    )

            # Missing profile names check
            for p_name in resolved_profiles.keys():
                if p_name not in profile_distribution:
                    raise ValueError(f"Missing profile '{p_name}' in profile_distribution.")

            # Probabilities validation
            total_prob = 0.0
            for p_name, prob in profile_distribution.items():
                if isinstance(prob, bool) or not isinstance(prob, (int, float)):
                    raise TypeError(
                        f"Profile probability for '{p_name}' must be numeric, got {type(prob).__name__}."
                    )
                prob_float = float(prob)
                if not np.isfinite(prob_float):
                    raise ValueError(f"Profile probability for '{p_name}' must be finite, got {prob}.")
                if prob_float < 0.0 or prob_float > 1.0:
                    raise ValueError(f"Profile probability for '{p_name}' must be in [0.0, 1.0], got {prob}.")
                total_prob += prob_float
                resolved_dist[p_name] = prob_float

            if not np.isclose(total_prob, 1.0, atol=1e-5):
                raise ValueError(f"Profile probabilities must sum to 1.0, got sum {total_prob}.")

        else:
            # profile_distribution is None
            if len(resolved_profiles) == 1:
                only_name = next(iter(resolved_profiles.keys()))
                resolved_dist = {only_name: 1.0}
            else:
                # Check if all profiles have explicit probability set on the Profile object
                if all(p.probability is not None for p in resolved_profiles.values()):
                    total_prob = 0.0
                    for p_name, p in resolved_profiles.items():
                        assert p.probability is not None
                        prob_float = float(p.probability)
                        total_prob += prob_float
                        resolved_dist[p_name] = prob_float
                    if not np.isclose(total_prob, 1.0, atol=1e-5):
                        raise ValueError(f"Profile probabilities must sum to 1.0, got sum {total_prob}.")
                else:
                    raise ValueError(
                        "profile_distribution must be explicitly specified when multiple profiles are provided."
                    )

        self.profiles = resolved_profiles
        self.profile_distribution = resolved_dist
        self.profile = profile if profile is not None else (
            next(iter(resolved_profiles.values())) if len(resolved_profiles) == 1 else None
        )

        # Matrix resolution & validation across all profiles
        if transition_matrix is not None:
            validate_transition_matrix(transition_matrix)
            n_states = len(self.states)
            if transition_matrix.shape != (n_states, n_states):
                raise ValueError(
                    f"Transition matrix shape {transition_matrix.shape} does not match state count ({n_states}, {n_states})."
                )

        self._profile_matrices: Dict[str, np.ndarray] = {}
        for p_name, p in self.profiles.items():
            if p.transition_matrix is not None:
                active_matrix = p.transition_matrix
            elif transition_matrix is not None:
                active_matrix = transition_matrix
            else:
                raise ValueError(
                    f"No transition matrix supplied for profile '{p_name}'. "
                    "Provide transition_matrix on Profile or Simulator."
                )

            validate_transition_matrix(active_matrix)
            n_states = len(self.states)
            if active_matrix.shape != (n_states, n_states):
                raise ValueError(
                    f"Transition matrix shape {active_matrix.shape} for profile '{p_name}' "
                    f"does not match state count ({n_states}, {n_states})."
                )
            self._profile_matrices[p_name] = active_matrix

        if self.profile is not None:
            self.transition_matrix = self._profile_matrices[self.profile.name]
        else:
            self.transition_matrix = transition_matrix

        # Initial state resolution
        if initial_state is not None:
            init_name = initial_state.name if isinstance(initial_state, State) else initial_state
            if init_name not in self.state_to_idx:
                raise ValueError(
                    f"Initial state '{init_name}' not found in registered states: {self.state_names}"
                )
            self.initial_state = init_name
        else:
            self.initial_state = self.state_names[0]

        # Validate profile transition rules target states across all profiles
        for p_name, p in self.profiles.items():
            if p.transition_rules:
                for rule in p.transition_rules:
                    if rule.target_state not in self.state_to_idx:
                        raise ValueError(
                            f"Transition rule target_state '{rule.target_state}' not found in registered states: {self.state_names}"
                        )

    def _sample_feature_value(
        self,
        dist_spec: Union[FeatureDistribution, Callable[[FeatureEvaluationContext], Any]],
        eval_ctx: FeatureEvaluationContext,
    ) -> Any:
        """Sample a single feature value using static distribution spec or dynamic callback."""
        if callable(dist_spec):
            val = dist_spec(eval_ctx)
            if isinstance(val, FeatureDistribution):
                return self._sample_feature_value(val, eval_ctx)
            return val

        if not isinstance(dist_spec, FeatureDistribution):
            raise TypeError(
                f"Feature distribution must be FeatureDistribution or callable, got {type(dist_spec).__name__}."
            )

        rng = eval_ctx.rng
        dtype = dist_spec.distribution_type.lower()
        params = dist_spec.params

        # Evaluate dynamic parameter callables inside params if any
        resolved_params: Dict[str, Any] = {}
        for k, v in params.items():
            if callable(v):
                resolved_params[k] = v(eval_ctx)
            else:
                resolved_params[k] = v

        if dtype == "normal":
            loc = float(resolved_params.get("loc", 0.0))
            scale = float(resolved_params.get("scale", 1.0))
            if scale < 0:
                raise ValueError(f"Normal distribution scale must be non-negative, got {scale}.")
            return float(rng.normal(loc=loc, scale=scale))

        elif dtype == "lognormal":
            mean = float(resolved_params.get("mean", 0.0))
            sigma = float(resolved_params.get("sigma", 1.0))
            if sigma < 0:
                raise ValueError(f"Lognormal distribution sigma must be non-negative, got {sigma}.")
            return float(rng.lognormal(mean=mean, sigma=sigma))

        elif dtype == "exponential":
            scale = float(resolved_params.get("scale", 1.0))
            if scale <= 0:
                raise ValueError(f"Exponential distribution scale must be positive, got {scale}.")
            return float(rng.exponential(scale=scale))

        elif dtype == "uniform":
            low = float(resolved_params.get("low", 0.0))
            high = float(resolved_params.get("high", 1.0))
            if high < low:
                raise ValueError(f"Uniform distribution high ({high}) must be >= low ({low}).")
            return float(rng.uniform(low=low, high=high))

        elif dtype == "uniform_discrete":
            if "items" in resolved_params:
                items = list(resolved_params["items"])
            elif "low" in resolved_params and "high" in resolved_params:
                low = int(resolved_params["low"])
                high = int(resolved_params["high"])
                if high < low:
                    raise ValueError(f"uniform_discrete high ({high}) must be >= low ({low}).")
                items = list(range(low, high + 1))
            else:
                raise ValueError("uniform_discrete requires 'items' or ('low', 'high') parameters.")
            return sample_categorical(rng, items)

        elif dtype == "bernoulli":
            p = float(resolved_params.get("p", 0.5))
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Bernoulli probability p must be in [0.0, 1.0], got {p}.")
            return int(rng.binomial(1, p))

        elif dtype == "poisson":
            lam = float(resolved_params.get("lam", 1.0))
            if lam < 0:
                raise ValueError(f"Poisson parameter lam must be non-negative, got {lam}.")
            return int(rng.poisson(lam=lam))

        elif dtype == "categorical":
            if "items" not in resolved_params:
                raise ValueError("categorical distribution requires 'items' parameter.")
            items = list(resolved_params["items"])
            probabilities = resolved_params.get("probabilities", None)
            return sample_categorical(rng, items, probabilities)

        else:
            raise ValueError(f"Unsupported distribution_type '{dist_spec.distribution_type}'.")

    def simulate(
        self,
        num_interactions: int = 100,
        num_sequences: int = 1,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Execute simulation loop across one or more independent sequences.

        Args:
            num_interactions: Number of interactions per sequence. Must be >= 1.
            num_sequences: Number of independent sequences to simulate. Must be >= 1.
            seed: Base random seed for deterministic execution.

        Returns:
            pd.DataFrame containing interaction traces across all generated sequences.
        """
        if num_interactions < 1:
            raise ValueError(f"num_interactions must be >= 1, got {num_interactions}.")

        if num_sequences < 1:
            raise ValueError(f"num_sequences must be >= 1, got {num_sequences}.")

        base_rng = create_rng(seed)

        # Pre-determine stable union feature column ordering across all profiles and states
        feature_names: List[str] = []
        for prof in self.profiles.values():
            for state_name, feature_map in prof.state_emissions.items():
                for f_name in feature_map.keys():
                    if f_name not in feature_names:
                        feature_names.append(f_name)

        sequence_dfs: List[pd.DataFrame] = []
        is_single = len(self.profiles) == 1
        single_profile = next(iter(self.profiles.values())) if is_single else None
        profile_names = list(self.profiles.keys())
        profile_probs = [self.profile_distribution[name] for name in profile_names]

        for seq_idx in range(1, num_sequences + 1):
            # Sequence seed derivation ensures determinism and sequence isolation
            seq_seed = int(base_rng.integers(0, 2**31 - 1)) if seed is None else seed + (seq_idx - 1) * 1000
            seq_rng = create_rng(seq_seed)
            history = SimulationHistory()

            if is_single:
                assert single_profile is not None
                selected_profile = single_profile
            else:
                profile_seed = int(base_rng.integers(0, 2**31 - 1)) if seed is None else seq_seed + 500
                profile_rng = create_rng(profile_seed)
                selected_profile_name = sample_categorical(profile_rng, profile_names, profile_probs)
                selected_profile = self.profiles[selected_profile_name]

            prof_matrix = self._profile_matrices[selected_profile.name]

            states_list: List[str] = []
            interaction_ids: List[int] = []
            sequence_ids: List[int] = []
            profiles_list: List[str] = []
            feature_data: Dict[str, List[Any]] = {f_name: [] for f_name in feature_names}

            current_state = self.initial_state

            for i in range(num_interactions):
                # 1. Transition Determination (for i > 0)
                if i > 0:
                    override_triggered = False
                    if selected_profile.transition_rules:
                        for rule in selected_profile.transition_rules:
                            if rule.condition(history):
                                override_triggered = True
                                if seq_rng.random() < rule.probability:
                                    current_state = rule.target_state
                                else:
                                    # Fallback to profile transition matrix if probability check fails
                                    curr_idx = self.state_to_idx[current_state]
                                    probs = prof_matrix[curr_idx]
                                    next_idx = sample_categorical(seq_rng, list(range(len(self.states))), probs)
                                    current_state = self.state_names[next_idx]
                                break  # First matching rule precedence

                    if not override_triggered:
                        curr_idx = self.state_to_idx[current_state]
                        probs = prof_matrix[curr_idx]
                        next_idx = sample_categorical(seq_rng, list(range(len(self.states))), probs)
                        current_state = self.state_names[next_idx]

                states_list.append(current_state)
                interaction_ids.append(i + 1)
                sequence_ids.append(seq_idx)
                profiles_list.append(selected_profile.name)

                # 2. Feature Emission Sampling
                eval_ctx = FeatureEvaluationContext(
                    state=current_state,
                    profile=selected_profile,
                    interaction_index=i,
                    history=history,
                    rng=seq_rng,
                )

                state_emissions = selected_profile.state_emissions.get(current_state, {})
                for f_name in feature_names:
                    if f_name in state_emissions:
                        dist_spec = state_emissions[f_name]
                        val = self._sample_feature_value(dist_spec, eval_ctx)
                    else:
                        val = None

                    feature_data[f_name].append(val)
                    history.record(f_name, val)

            # Build DataFrame for sequence
            seq_dict: Dict[str, Any] = {
                "profile": profiles_list,
                "sequence_id": sequence_ids,
                "interaction_id": interaction_ids,
                "state": states_list,
            }
            for f_name in feature_names:
                seq_dict[f_name] = feature_data[f_name]

            df_seq = pd.DataFrame(seq_dict)
            sequence_dfs.append(df_seq)

        return pd.concat(sequence_dfs, ignore_index=True)

    def generate(
        self,
        num_interactions: int = 100,
        num_sequences: int = 1,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic behavioral traces across one or more sequences.

        Ergonomic public alias for simulate().

        Args:
            num_interactions: Number of interactions per sequence. Must be >= 1.
            num_sequences: Number of independent sequences to simulate. Must be >= 1.
            seed: Base random seed for deterministic execution.

        Returns:
            pd.DataFrame containing interaction traces across all generated sequences.
        """
        return self.simulate(
            num_interactions=num_interactions,
            num_sequences=num_sequences,
            seed=seed,
        )

    @classmethod
    def from_preset(cls, name: str, **kwargs: Any) -> Any:
        """Construct a Simulator (or domain simulator) from a registered preset name.

        Args:
            name: Deterministic preset identifier (e.g., 'education').
            **kwargs: Additional parameters passed to the preset factory.

        Returns:
            Simulator or compatible high-level simulation object.

        Raises:
            TypeError: If name is not a string.
            ValueError: If preset name is not registered.
        """
        from behaviorsim.presets import get_preset

        factory = get_preset(name)
        return factory(**kwargs)

    @classmethod
    def from_config(
        cls,
        config: Union[SimulationConfig, Mapping[str, Any], str, Path],
        profile_name: Optional[str] = None,
        profile_distribution: Optional[Mapping[str, float]] = None,
    ) -> Simulator:
        """Construct a Simulator instance from a configuration specification.

        Args:
            config: SimulationConfig instance, configuration mapping/dict, or path to a YAML/JSON file.
            profile_name: Optional profile name to select from multi-profile configs.
                If None, defaults to the first profile in the configuration unless profile_distribution
                is specified.
            profile_distribution: Optional mapping of profile probabilities for mixture simulation.

        Returns:
            Configured Simulator instance.

        Raises:
            TypeError: If config is not a SimulationConfig, Mapping, or str/Path.
            ValueError: If configuration validation or profile selection fails.
            FileNotFoundError: If a file path is provided that does not exist.
        """
        from behaviorsim.config import SimulationConfig, load_config, parse_config

        if isinstance(config, SimulationConfig):
            cfg = config
        elif isinstance(config, Mapping):
            cfg = parse_config(config)
        elif isinstance(config, (str, Path)):
            cfg = load_config(config)
        else:
            raise TypeError(
                f"Unsupported config type: {type(config).__name__}. "
                "Expected SimulationConfig, Mapping, or file path (str/Path)."
            )

        return cfg.to_core(profile_name=profile_name, profile_distribution=profile_distribution)
