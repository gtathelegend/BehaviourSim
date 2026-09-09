"""Core simulator engine for BehaviorSim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union
import numpy as np
import pandas as pd

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
        profile: Profile,
        transition_matrix: Optional[np.ndarray] = None,
        initial_state: Optional[Union[str, State]] = None,
    ) -> None:
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
        self.profile = profile

        # Matrix resolution
        if profile.transition_matrix is not None:
            active_matrix = profile.transition_matrix
        elif transition_matrix is not None:
            active_matrix = transition_matrix
        else:
            raise ValueError(
                "No transition matrix supplied. Provide transition_matrix on Profile or Simulator."
            )

        validate_transition_matrix(active_matrix)
        n_states = len(self.states)
        if active_matrix.shape != (n_states, n_states):
            raise ValueError(
                f"Transition matrix shape {active_matrix.shape} does not match state count ({n_states}, {n_states})."
            )

        self.transition_matrix = active_matrix

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

        # Validate profile transition rules target states
        if profile.transition_rules:
            for rule in profile.transition_rules:
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

        # Pre-determine feature column ordering from emissions map across states
        feature_names: List[str] = []
        for state_name, feature_map in self.profile.state_emissions.items():
            for f_name in feature_map.keys():
                if f_name not in feature_names:
                    feature_names.append(f_name)

        sequence_dfs: List[pd.DataFrame] = []

        for seq_idx in range(1, num_sequences + 1):
            # Sequence seed derivation ensures determinism and sequence isolation
            seq_seed = int(base_rng.integers(0, 2**31 - 1)) if seed is None else seed + (seq_idx - 1) * 1000
            seq_rng = create_rng(seq_seed)
            history = SimulationHistory()

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
                    if self.profile.transition_rules:
                        for rule in self.profile.transition_rules:
                            if rule.condition(history):
                                override_triggered = True
                                if seq_rng.random() < rule.probability:
                                    current_state = rule.target_state
                                else:
                                    # Fallback to base transition matrix if probability check fails
                                    curr_idx = self.state_to_idx[current_state]
                                    probs = self.transition_matrix[curr_idx]
                                    next_idx = sample_categorical(seq_rng, list(range(len(self.states))), probs)
                                    current_state = self.state_names[next_idx]
                                break  # First matching rule precedence

                    if not override_triggered:
                        curr_idx = self.state_to_idx[current_state]
                        probs = self.transition_matrix[curr_idx]
                        next_idx = sample_categorical(seq_rng, list(range(len(self.states))), probs)
                        current_state = self.state_names[next_idx]

                states_list.append(current_state)
                interaction_ids.append(i + 1)
                sequence_ids.append(seq_idx)
                profiles_list.append(self.profile.name)

                # 2. Feature Emission Sampling
                eval_ctx = FeatureEvaluationContext(
                    state=current_state,
                    profile=self.profile,
                    interaction_index=i,
                    history=history,
                    rng=seq_rng,
                )

                state_emissions = self.profile.state_emissions.get(current_state, {})
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
