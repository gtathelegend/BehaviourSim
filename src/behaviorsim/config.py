"""Configuration data model, schema validation, and declarative rule compiler for BehaviorSim."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Union
import numpy as np
import pandas as pd
import yaml

from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.simulator import Simulator
from behaviorsim.core.state import State
from behaviorsim.core.transition import HistoryContext, TransitionRule
from behaviorsim.core.utils import validate_transition_matrix


# ---------------------------------------------------------------------------
# Supported Schema Primitives & Constants
# ---------------------------------------------------------------------------

SUPPORTED_DISTRIBUTIONS: Set[str] = {
    "normal",
    "lognormal",
    "exponential",
    "uniform",
    "uniform_discrete",
    "bernoulli",
    "poisson",
    "categorical",
}

SUPPORTED_AGGREGATIONS: Set[str] = {"mean", "sum", "min", "max", "last"}

SUPPORTED_OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


# ---------------------------------------------------------------------------
# Declarative Condition Compilation & Validation
# ---------------------------------------------------------------------------

def validate_condition_spec(condition_spec: Any) -> None:
    """Validate that a condition specification conforms to declarative condition schema.

    Arbitrary expression evaluation (eval/exec) is strictly forbidden.
    """
    if not isinstance(condition_spec, Mapping):
        raise TypeError(f"Condition specification must be a mapping, got {type(condition_spec).__name__}.")

    if "all" in condition_spec and "any" in condition_spec:
        raise ValueError("Condition specification cannot contain both 'all' and 'any'.")

    if "all" in condition_spec:
        subs = condition_spec["all"]
        if not isinstance(subs, Sequence) or isinstance(subs, (str, bytes)) or len(subs) == 0:
            raise ValueError("'all' condition must be a non-empty sequence of sub-conditions.")
        for sub in subs:
            validate_condition_spec(sub)
        return

    if "any" in condition_spec:
        subs = condition_spec["any"]
        if not isinstance(subs, Sequence) or isinstance(subs, (str, bytes)) or len(subs) == 0:
            raise ValueError("'any' condition must be a non-empty sequence of sub-conditions.")
        for sub in subs:
            validate_condition_spec(sub)
        return

    # Simple condition
    if "feature" not in condition_spec:
        raise ValueError("Condition specification missing required field: 'feature'.")
    feature = condition_spec["feature"]
    if not isinstance(feature, str) or not feature.strip():
        raise ValueError("Condition 'feature' must be a non-empty string.")

    if "operator" not in condition_spec:
        raise ValueError("Condition specification missing required field: 'operator'.")
    op = condition_spec["operator"]
    if op not in SUPPORTED_OPERATORS:
        raise ValueError(
            f"Unsupported condition operator: '{op}'. Supported operators: {sorted(SUPPORTED_OPERATORS.keys())}."
        )

    if "value" not in condition_spec:
        raise ValueError("Condition specification missing required field: 'value'.")

    window = condition_spec.get("window", 1)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"Condition 'window' must be an integer >= 1, got {window}.")

    agg = condition_spec.get("aggregation", "last")
    if not isinstance(agg, str) or agg.lower() not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported condition aggregation: '{agg}'. Supported aggregations: {sorted(SUPPORTED_AGGREGATIONS)}."
        )


def compile_condition(condition_spec: Mapping[str, Any]) -> Callable[[HistoryContext], bool]:
    """Compile a declarative condition specification into a HistoryContext predicate.

    Arbitrary expression evaluation (eval/exec) is strictly forbidden. Conditions
    are compiled into pure Python functions evaluating historical observations.

    Args:
        condition_spec: Declarative condition dictionary.

    Returns:
        Callable[[HistoryContext], bool] suitable for TransitionRule.
    """
    validate_condition_spec(condition_spec)

    if "all" in condition_spec:
        compiled_subs = [compile_condition(sub) for sub in condition_spec["all"]]
        return lambda ctx: all(sub(ctx) for sub in compiled_subs)

    if "any" in condition_spec:
        compiled_subs = [compile_condition(sub) for sub in condition_spec["any"]]
        return lambda ctx: any(sub(ctx) for sub in compiled_subs)

    feature = condition_spec["feature"]
    window = condition_spec.get("window", 1)
    agg = condition_spec.get("aggregation", "last").lower()
    op_str = condition_spec["operator"]
    op_func = SUPPORTED_OPERATORS[op_str]
    threshold = condition_spec["value"]

    def predicate(ctx: HistoryContext) -> bool:
        recent = ctx.get_recent(feature, window)
        if not recent:
            # When history is empty, condition safely evaluates to non-matching (False)
            return False

        if agg == "last":
            val = recent[-1]
        elif agg == "mean":
            val = float(np.mean(recent))
        elif agg == "sum":
            val = float(np.sum(recent))
        elif agg == "min":
            val = float(np.min(recent))
        elif agg == "max":
            val = float(np.max(recent))
        else:
            return False

        try:
            return bool(op_func(val, threshold))
        except (TypeError, ValueError):
            return False

    return predicate


# ---------------------------------------------------------------------------
# Distribution Parameter Validation
# ---------------------------------------------------------------------------

def validate_distribution_params(distribution: str, params: Mapping[str, Any]) -> None:
    """Validate parameter specifications for supported distribution families.

    Args:
        distribution: Name of the distribution family.
        params: Mapping of distribution parameters.

    Raises:
        ValueError: If distribution family is unsupported or parameters are invalid.
        TypeError: If params is not a mapping.
    """
    dtype = distribution.lower()
    if dtype not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"unsupported distribution type: '{distribution}'")

    if not isinstance(params, Mapping):
        raise TypeError(f"Distribution params must be a mapping, got {type(params).__name__}.")

    if dtype == "normal":
        if "scale" in params:
            scale = params["scale"]
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale < 0:
                raise ValueError(f"normal parameter 'scale' must be non-negative, got {scale}")
        if "loc" in params:
            loc = params["loc"]
            if isinstance(loc, bool) or not isinstance(loc, (int, float)):
                raise ValueError(f"normal parameter 'loc' must be numeric, got {loc}")

    elif dtype == "lognormal":
        if "sigma" in params:
            sigma = params["sigma"]
            if isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or sigma < 0:
                raise ValueError(f"lognormal parameter 'sigma' must be non-negative, got {sigma}")
        if "mean" in params:
            mean = params["mean"]
            if isinstance(mean, bool) or not isinstance(mean, (int, float)):
                raise ValueError(f"lognormal parameter 'mean' must be numeric, got {mean}")

    elif dtype == "exponential":
        if "scale" not in params:
            raise ValueError("exponential distribution requires 'scale' parameter")
        scale = params["scale"]
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or scale <= 0:
            raise ValueError(f"exponential parameter 'scale' must be positive, got {scale}")

    elif dtype == "uniform":
        if "low" not in params or "high" not in params:
            raise ValueError("uniform distribution requires 'low' and 'high' parameters")
        low = params["low"]
        high = params["high"]
        if isinstance(low, bool) or not isinstance(low, (int, float)):
            raise ValueError(f"uniform parameter 'low' must be numeric, got {low}")
        if isinstance(high, bool) or not isinstance(high, (int, float)):
            raise ValueError(f"uniform parameter 'high' must be numeric, got {high}")
        if high < low:
            raise ValueError(f"uniform parameter 'high' must be >= 'low', got high={high}, low={low}")

    elif dtype == "uniform_discrete":
        if "items" in params:
            items = params["items"]
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or len(items) == 0:
                raise ValueError("uniform_discrete parameter 'items' must be a non-empty sequence")
        elif "low" in params and "high" in params:
            low = params["low"]
            high = params["high"]
            if isinstance(low, bool) or not isinstance(low, int):
                raise ValueError(f"uniform_discrete parameter 'low' must be an integer, got {low}")
            if isinstance(high, bool) or not isinstance(high, int):
                raise ValueError(f"uniform_discrete parameter 'high' must be an integer, got {high}")
            if high < low:
                raise ValueError(f"uniform_discrete parameter 'high' must be >= 'low', got high={high}, low={low}")
        else:
            raise ValueError("uniform_discrete requires either 'items' or ('low', 'high') parameters")

    elif dtype == "bernoulli":
        if "p" not in params:
            raise ValueError("bernoulli distribution requires 'p' parameter")
        p = params["p"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not (0.0 <= float(p) <= 1.0):
            raise ValueError(f"bernoulli parameter 'p' must be between 0 and 1, got {p}")

    elif dtype == "poisson":
        if "lam" not in params:
            raise ValueError("poisson distribution requires 'lam' parameter")
        lam = params["lam"]
        if isinstance(lam, bool) or not isinstance(lam, (int, float)) or lam < 0:
            raise ValueError(f"poisson parameter 'lam' must be non-negative, got {lam}")

    elif dtype == "categorical":
        if "items" not in params:
            raise ValueError("categorical distribution requires 'items' parameter")
        items = params["items"]
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)) or len(items) == 0:
            raise ValueError("categorical parameter 'items' must be a non-empty sequence")
        if "probabilities" in params and params["probabilities"] is not None:
            probs = params["probabilities"]
            if not isinstance(probs, Sequence) or isinstance(probs, (str, bytes)):
                raise ValueError("categorical parameter 'probabilities' must be a sequence of floats")
            if len(probs) != len(items):
                raise ValueError(
                    f"categorical 'probabilities' length ({len(probs)}) must match 'items' length ({len(items)})"
                )
            for prob in probs:
                if (
                    isinstance(prob, bool)
                    or not isinstance(prob, (int, float))
                    or math.isnan(prob)
                    or math.isinf(prob)
                ):
                    raise ValueError("categorical probabilities must be finite numbers")
                if prob < 0:
                    raise ValueError(f"categorical probabilities must be non-negative, got {prob}")
            if not math.isclose(sum(probs), 1.0, abs_tol=1e-5):
                raise ValueError(f"categorical probabilities must sum to 1.0, got {sum(probs)}")


# ---------------------------------------------------------------------------
# Declarative Configuration Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StateConfig:
    """Configuration for a discrete cognitive/behavioral state."""

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("state 'name' must be a non-empty string")
        if not isinstance(self.description, str):
            raise ValueError("state 'description' must be a string")


@dataclass
class FeatureDistributionConfig:
    """Configuration for a feature emission statistical distribution."""

    distribution: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.distribution, str) or not self.distribution.strip():
            raise ValueError("distribution must be a non-empty string")
        validate_distribution_params(self.distribution, self.params)


@dataclass
class TransitionRuleConfig:
    """Declarative configuration for a history-dependent transition rule."""

    condition: Mapping[str, Any]
    target_state: str
    probability: float = 1.0

    def __post_init__(self) -> None:
        validate_condition_spec(self.condition)
        if not isinstance(self.target_state, str) or not self.target_state.strip():
            raise ValueError("transition rule 'target_state' must be a non-empty string")
        if isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)):
            raise TypeError("transition rule 'probability' must be numeric")
        prob_float = float(self.probability)
        if math.isnan(prob_float) or math.isinf(prob_float):
            raise ValueError("transition rule 'probability' must be finite")
        if not (0.0 <= prob_float <= 1.0):
            raise ValueError(f"transition rule 'probability' must be in [0, 1], got {self.probability}")


@dataclass
class ProfileConfig:
    """Configuration for an agent or persona profile."""

    name: str
    state_emissions: Mapping[str, Mapping[str, FeatureDistributionConfig]]
    transition_matrix: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None
    transition_rules: Optional[Sequence[TransitionRuleConfig]] = None
    metadata: Optional[Mapping[str, Any]] = None
    probability: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("profile 'name' must be a non-empty string")
        if not isinstance(self.state_emissions, Mapping):
            raise TypeError("profile 'state_emissions' must be a mapping")
        if self.probability is not None:
            if isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)):
                raise TypeError(f"profile probability must be numeric, got {type(self.probability).__name__}")
            p_val = float(self.probability)
            if not math.isfinite(p_val):
                raise ValueError("profile probability must be a finite number")
            if not (0.0 <= p_val <= 1.0):
                raise ValueError(f"profile probability must be between 0 and 1, got {self.probability}")


@dataclass
class SimulationRunConfig:
    """Execution parameters for simulation runs."""

    num_interactions: int = 100
    num_sequences: int = 1
    seed: Optional[int] = None
    initial_state: Optional[str] = None
    profile_distribution: Optional[Mapping[str, float]] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_interactions, bool)
            or not isinstance(self.num_interactions, int)
            or self.num_interactions < 1
        ):
            raise ValueError(f"num_interactions must be an integer >= 1, got {self.num_interactions}")
        if (
            isinstance(self.num_sequences, bool)
            or not isinstance(self.num_sequences, int)
            or self.num_sequences < 1
        ):
            raise ValueError(f"num_sequences must be an integer >= 1, got {self.num_sequences}")
        if self.seed is not None and (isinstance(self.seed, bool) or not isinstance(self.seed, int)):
            raise ValueError(f"seed must be an integer or None, got {self.seed}")
        if self.initial_state is not None and (
            not isinstance(self.initial_state, str) or not self.initial_state.strip()
        ):
            raise ValueError("initial_state must be a non-empty string or None")
        if self.profile_distribution is not None and not isinstance(self.profile_distribution, Mapping):
            raise TypeError("profile_distribution must be a mapping")


@dataclass
class SimulationConfig:
    """Root configuration object representing an entire simulation specification."""

    version: str
    states: Sequence[StateConfig]
    profiles: Sequence[ProfileConfig]
    simulation: SimulationRunConfig
    transition_matrix: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None
    profile_distribution: Optional[Mapping[str, float]] = None

    def to_core(
        self,
        profile_name: Optional[str] = None,
        profile_distribution: Optional[Mapping[str, float]] = None,
    ) -> Simulator:
        """Instantiate a generic Simulator from this configuration.

        Args:
            profile_name: Optional name of single profile to select. If provided, creates
                a single-profile Simulator regardless of other profiles.
            profile_distribution: Optional mapping of profile probabilities for mixture simulation.
                If None, uses distribution defined in configuration if present, or defaults to
                the first profile in self.profiles.

        Returns:
            Configured Simulator instance.
        """
        core_states = [State(name=s.name, description=s.description) for s in self.states]

        def _build_core_profile(p: ProfileConfig) -> Profile:
            core_state_emissions: Dict[str, Dict[str, FeatureDistribution]] = {}
            for s_name, feat_map in p.state_emissions.items():
                core_state_emissions[s_name] = {}
                for f_name, dist_cfg in feat_map.items():
                    core_state_emissions[s_name][f_name] = FeatureDistribution(
                        distribution_type=dist_cfg.distribution,
                        params=dist_cfg.params,
                    )

            core_rules = None
            if p.transition_rules:
                core_rules = [
                    TransitionRule(
                        condition=compile_condition(r.condition),
                        target_state=r.target_state,
                        probability=r.probability,
                    )
                    for r in p.transition_rules
                ]

            prof_matrix = None
            if p.transition_matrix is not None:
                prof_matrix = np.asarray(p.transition_matrix, dtype=float)

            return Profile(
                name=p.name,
                state_emissions=core_state_emissions,
                transition_matrix=prof_matrix,
                transition_rules=core_rules,
                metadata=p.metadata,
                probability=p.probability,
            )

        top_matrix = None
        if self.transition_matrix is not None:
            top_matrix = np.asarray(self.transition_matrix, dtype=float)

        # 1. Explicit profile_name selection takes precedence (single-profile mode)
        if profile_name is not None:
            matching = [p for p in self.profiles if p.name == profile_name]
            if not matching:
                raise ValueError(f"Profile '{profile_name}' not found in configuration.")
            core_profile = _build_core_profile(matching[0])
            return Simulator(
                states=core_states,
                profile=core_profile,
                transition_matrix=top_matrix,
                initial_state=self.simulation.initial_state,
            )

        # 2. Multi-profile mixture resolution
        active_dist = profile_distribution or self.profile_distribution or self.simulation.profile_distribution
        has_profile_probs = len(self.profiles) > 1 and all(p.probability is not None for p in self.profiles)

        if active_dist is not None or has_profile_probs:
            core_profiles = [_build_core_profile(p) for p in self.profiles]
            return Simulator(
                states=core_states,
                profiles=core_profiles,
                profile_distribution=active_dist,
                transition_matrix=top_matrix,
                initial_state=self.simulation.initial_state,
            )

        # 3. Default fallback to first profile (backward-compatible)
        core_profile = _build_core_profile(self.profiles[0])
        return Simulator(
            states=core_states,
            profile=core_profile,
            transition_matrix=top_matrix,
            initial_state=self.simulation.initial_state,
        )


# ---------------------------------------------------------------------------
# Schema Parsing & Validation Functions
# ---------------------------------------------------------------------------

def parse_config(raw_data: Mapping[str, Any]) -> SimulationConfig:
    """Parse and validate a raw configuration mapping into a typed SimulationConfig.

    Args:
        raw_data: Mapping representing configuration data (e.g., from YAML/JSON).

    Returns:
        Validated SimulationConfig instance.

    Raises:
        ValueError: On semantic or structural validation failures.
        TypeError: On unexpected data types.
    """
    if not isinstance(raw_data, Mapping):
        raise TypeError(f"Configuration must be a mapping, got {type(raw_data).__name__}.")

    version = raw_data.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("configuration 'version' must be a non-empty string")

    # 1. Parse & Validate States
    raw_states = raw_data.get("states")
    if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)) or len(raw_states) == 0:
        raise ValueError("states must contain at least one state")

    seen_states: Set[str] = set()
    state_configs: List[StateConfig] = []
    for i, s in enumerate(raw_states):
        if not isinstance(s, Mapping):
            raise TypeError(f"State at index {i} must be a mapping.")
        s_name = s.get("name")
        if not isinstance(s_name, str) or not s_name.strip():
            raise ValueError("state 'name' must be a non-empty string")
        if s_name in seen_states:
            raise ValueError(f"duplicate state name: '{s_name}'")
        seen_states.add(s_name)
        s_desc = s.get("description", "")
        state_configs.append(StateConfig(name=s_name, description=s_desc))

    n_states = len(seen_states)

    # 2. Parse & Validate Simulation Settings
    raw_sim = raw_data.get("simulation")
    if raw_sim is None or not isinstance(raw_sim, Mapping):
        raise ValueError("'simulation' must be a mapping")

    num_interactions = raw_sim.get("num_interactions", 100)
    num_sequences = raw_sim.get("num_sequences", 1)
    seed = raw_sim.get("seed")
    initial_state = raw_sim.get("initial_state")

    if initial_state is not None and initial_state not in seen_states:
        raise ValueError(f"initial_state '{initial_state}' does not exist in configured states")

    # Optional profile distribution from simulation or root config
    raw_dist = raw_data.get("profile_distribution")
    if raw_dist is None and isinstance(raw_sim, Mapping):
        raw_dist = raw_sim.get("profile_distribution")

    parsed_dist: Optional[Dict[str, float]] = None
    if raw_dist is not None:
        if not isinstance(raw_dist, Mapping):
            raise TypeError(f"profile_distribution must be a mapping, got {type(raw_dist).__name__}.")
        if not raw_dist:
            raise ValueError("profile_distribution cannot be empty")
        parsed_dist = {}
        total_p = 0.0
        for k, v in raw_dist.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("profile_distribution profile name must be a non-empty string")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"profile probability for '{k}' must be numeric, got {type(v).__name__}")
            v_float = float(v)
            if not math.isfinite(v_float):
                raise ValueError(f"profile probability for '{k}' must be finite")
            if v_float < 0.0 or v_float > 1.0:
                raise ValueError(f"profile probability for '{k}' must be between 0 and 1, got {v}")
            total_p += v_float
            parsed_dist[k] = v_float
        if not math.isclose(total_p, 1.0, abs_tol=1e-5):
            raise ValueError(f"profile_distribution probabilities must sum to 1.0, got {total_p}")

    sim_run_config = SimulationRunConfig(
        num_interactions=num_interactions,
        num_sequences=num_sequences,
        seed=seed,
        initial_state=initial_state,
        profile_distribution=parsed_dist,
    )

    # 3. Parse & Validate Top-level Transition Matrix
    top_matrix = None
    if raw_data.get("transition_matrix") is not None:
        try:
            top_matrix = np.asarray(raw_data["transition_matrix"], dtype=float)
        except Exception as e:
            raise ValueError(f"Failed to convert transition_matrix to array: {e}")
        validate_transition_matrix(top_matrix)
        if top_matrix.shape != (n_states, n_states):
            raise ValueError(f"transition_matrix must have shape ({n_states}, {n_states})")

    # 4. Parse & Validate Profiles
    raw_profiles = raw_data.get("profiles")
    if not isinstance(raw_profiles, Sequence) or isinstance(raw_profiles, (str, bytes)) or len(raw_profiles) == 0:
        raise ValueError("profiles must contain at least one profile")

    seen_profiles: Set[str] = set()
    profile_configs: List[ProfileConfig] = []

    for i, p in enumerate(raw_profiles):
        if not isinstance(p, Mapping):
            raise TypeError(f"Profile at index {i} must be a mapping.")
        p_name = p.get("name")
        if not isinstance(p_name, str) or not p_name.strip():
            raise ValueError("profile 'name' must be a non-empty string")
        if p_name in seen_profiles:
            raise ValueError(f"duplicate profile name: '{p_name}'")
        seen_profiles.add(p_name)

        # Optional probability on profile
        p_prob = p.get("probability")
        if p_prob is not None:
            if isinstance(p_prob, bool) or not isinstance(p_prob, (int, float)):
                raise TypeError(f"profile '{p_name}' probability must be numeric, got {type(p_prob).__name__}")
            p_prob_f = float(p_prob)
            if not math.isfinite(p_prob_f):
                raise ValueError(f"profile '{p_name}' probability must be finite")
            if p_prob_f < 0.0 or p_prob_f > 1.0:
                raise ValueError(f"profile '{p_name}' probability must be between 0 and 1, got {p_prob}")
            p_prob = p_prob_f

        # State emissions validation
        raw_emissions = p.get("state_emissions")
        if raw_emissions is None or not isinstance(raw_emissions, Mapping):
            raise ValueError(f"profile '{p_name}' state_emissions must be a mapping")

        emissions_config: Dict[str, Dict[str, FeatureDistributionConfig]] = {}
        for s_name, feat_map in raw_emissions.items():
            if s_name not in seen_states:
                raise ValueError(f"profile '{p_name}' references unknown state '{s_name}'")
            if not isinstance(feat_map, Mapping):
                raise TypeError(f"emissions for state '{s_name}' in profile '{p_name}' must be a mapping")

            emissions_config[s_name] = {}
            for f_name, d_spec in feat_map.items():
                if not isinstance(d_spec, Mapping):
                    raise TypeError(f"feature '{f_name}' distribution specification must be a mapping")
                dist_name = d_spec.get("distribution") or d_spec.get("distribution_type")
                if not dist_name or not isinstance(dist_name, str):
                    raise ValueError(f"feature '{f_name}' missing distribution type")
                d_params = d_spec.get("params", {})
                emissions_config[s_name][f_name] = FeatureDistributionConfig(
                    distribution=dist_name,
                    params=d_params,
                )

        # Profile transition matrix
        prof_matrix = None
        if p.get("transition_matrix") is not None:
            try:
                prof_matrix = np.asarray(p["transition_matrix"], dtype=float)
            except Exception as e:
                raise ValueError(f"profile '{p_name}' failed to convert transition_matrix to array: {e}")
            validate_transition_matrix(prof_matrix)
            if prof_matrix.shape != (n_states, n_states):
                raise ValueError(
                    f"profile '{p_name}' transition_matrix must have shape ({n_states}, {n_states})"
                )
        elif top_matrix is None:
            raise ValueError(
                f"profile '{p_name}' has no transition_matrix and no top-level transition_matrix is provided"
            )

        # Profile transition rules
        rule_configs = None
        if p.get("transition_rules") is not None:
            raw_rules = p["transition_rules"]
            if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes)):
                raise TypeError(f"profile '{p_name}' transition_rules must be a sequence")
            rule_configs = []
            for r_idx, r in enumerate(raw_rules):
                if not isinstance(r, Mapping):
                    raise TypeError(f"Transition rule at index {r_idx} in profile '{p_name}' must be a mapping.")
                target = r.get("target_state")
                if not target or not isinstance(target, str):
                    raise ValueError("transition rule missing 'target_state'")
                if target not in seen_states:
                    raise ValueError(f"transition rule target_state '{target}' does not exist")
                cond = r.get("condition")
                if cond is None:
                    raise ValueError("transition rule missing 'condition'")
                prob = r.get("probability", 1.0)
                rule_configs.append(
                    TransitionRuleConfig(
                        condition=cond,
                        target_state=target,
                        probability=prob,
                    )
                )

        profile_configs.append(
            ProfileConfig(
                name=p_name,
                state_emissions=emissions_config,
                transition_matrix=prof_matrix,
                transition_rules=rule_configs,
                metadata=p.get("metadata"),
                probability=p_prob,
            )
        )

    # Inferred distribution from profile probabilities if not explicitly given
    if parsed_dist is None and len(profile_configs) > 1 and all(p.probability is not None for p in profile_configs):
        total_p = sum(p.probability for p in profile_configs if p.probability is not None)
        if not math.isclose(total_p, 1.0, abs_tol=1e-5):
            raise ValueError(f"profile probabilities must sum to 1.0, got {total_p}")
        parsed_dist = {p.name: p.probability for p in profile_configs if p.probability is not None}

    # Cross-validation between parsed distribution and profiles
    if parsed_dist is not None:
        for k in parsed_dist.keys():
            if k not in seen_profiles:
                raise ValueError(f"profile_distribution references unknown profile '{k}'")
        for k in seen_profiles:
            if k not in parsed_dist:
                raise ValueError(f"profile_distribution missing profile '{k}'")

    return SimulationConfig(
        version=version,
        states=state_configs,
        profiles=profile_configs,
        simulation=sim_run_config,
        transition_matrix=top_matrix,
        profile_distribution=parsed_dist,
    )


def validate_config(config: Union[SimulationConfig, Mapping[str, Any]]) -> None:
    """Validate a SimulationConfig instance or raw configuration mapping.

    Args:
        config: SimulationConfig or raw configuration dictionary.

    Raises:
        ValueError: On semantic or structural validation failures.
        TypeError: On unexpected data types.
    """
    if isinstance(config, Mapping):
        parse_config(config)
    elif isinstance(config, SimulationConfig):
        # Validate internal consistency of SimulationConfig
        seen_states = {s.name for s in config.states}
        if len(seen_states) != len(config.states):
            raise ValueError("states must have unique names")

        n_states = len(seen_states)
        if config.transition_matrix is not None:
            mat = np.asarray(config.transition_matrix, dtype=float)
            validate_transition_matrix(mat)
            if mat.shape != (n_states, n_states):
                raise ValueError(f"transition_matrix must have shape ({n_states}, {n_states})")

        seen_profiles = {p.name for p in config.profiles}
        for p in config.profiles:
            for s_name in p.state_emissions:
                if s_name not in seen_states:
                    raise ValueError(f"profile '{p.name}' references unknown state '{s_name}'")
            if p.transition_matrix is not None:
                p_mat = np.asarray(p.transition_matrix, dtype=float)
                validate_transition_matrix(p_mat)
                if p_mat.shape != (n_states, n_states):
                    raise ValueError(
                        f"profile '{p.name}' transition_matrix must have shape ({n_states}, {n_states})"
                    )
            elif config.transition_matrix is None:
                raise ValueError(
                    f"profile '{p.name}' has no transition_matrix and no top-level transition_matrix is provided"
                )

            if p.transition_rules:
                for r in p.transition_rules:
                    if r.target_state not in seen_states:
                        raise ValueError(f"transition rule target_state '{r.target_state}' does not exist")
                    validate_condition_spec(r.condition)

        if config.simulation.initial_state is not None and config.simulation.initial_state not in seen_states:
            raise ValueError(f"initial_state '{config.simulation.initial_state}' does not exist in configured states")

        active_dist = config.profile_distribution or config.simulation.profile_distribution
        if active_dist is not None:
            if not isinstance(active_dist, Mapping):
                raise TypeError("profile_distribution must be a mapping")
            if not active_dist:
                raise ValueError("profile_distribution cannot be empty")
            total_p = 0.0
            for k, v in active_dist.items():
                if k not in seen_profiles:
                    raise ValueError(f"profile_distribution references unknown profile '{k}'")
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(f"profile probability for '{k}' must be numeric")
                v_float = float(v)
                if not math.isfinite(v_float) or v_float < 0.0 or v_float > 1.0:
                    raise ValueError(f"profile probability for '{k}' must be finite and within [0, 1]")
                total_p += v_float
            for k in seen_profiles:
                if k not in active_dist:
                    raise ValueError(f"profile_distribution missing profile '{k}'")
            if not math.isclose(total_p, 1.0, abs_tol=1e-5):
                raise ValueError(f"profile_distribution probabilities must sum to 1.0, got {total_p}")
    else:
        raise TypeError(f"Config must be a SimulationConfig or Mapping, got {type(config).__name__}.")


def build_simulator(
    config: SimulationConfig,
    profile_name: Optional[str] = None,
    profile_distribution: Optional[Mapping[str, float]] = None,
) -> Simulator:
    """Build a core Simulator from a validated SimulationConfig.

    Args:
        config: Validated SimulationConfig instance.
        profile_name: Optional profile name to select. If None, the first profile is used
            unless profile_distribution is configured.
        profile_distribution: Optional profile probabilities mapping for mixture simulation.

    Returns:
        Simulator instance.
    """
    return config.to_core(profile_name=profile_name, profile_distribution=profile_distribution)


def run_simulation(
    config: SimulationConfig,
    profile_name: Optional[str] = None,
    profile_distribution: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Execute a simulation run directly using settings from a SimulationConfig.

    Args:
        config: Validated SimulationConfig instance.
        profile_name: Optional profile name to select. If None, the first profile is used
            unless profile_distribution is configured.
        profile_distribution: Optional profile probabilities mapping for mixture simulation.

    Returns:
        pd.DataFrame containing the generated simulation trace.
    """
    sim = build_simulator(config, profile_name=profile_name, profile_distribution=profile_distribution)
    return sim.simulate(
        num_interactions=config.simulation.num_interactions,
        num_sequences=config.simulation.num_sequences,
        seed=config.simulation.seed,
    )


def load_config(path: Union[str, Path]) -> SimulationConfig:
    """Load, parse, and validate a simulation configuration from a YAML or JSON file.

    Supported formats: .yaml, .yml, .json (case-insensitive).

    Args:
        path: String or Path to the configuration file.

    Returns:
        Validated SimulationConfig instance.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If file format is unsupported, file is empty, or parsing/validation fails.
        TypeError: If root configuration is not a mapping/object.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext not in {".yaml", ".yml", ".json"}:
        raise ValueError(
            f"Unsupported configuration file format '{file_path.suffix}'. "
            "Supported formats: .yaml, .yml, .json"
        )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if ext in {".yaml", ".yml"}:
                raw_data = yaml.safe_load(f)
            else:
                raw_data = json.load(f)
    except (yaml.YAMLError, json.JSONDecodeError) as err:
        raise ValueError(f"Failed to parse configuration file '{file_path}': {err}") from err

    if raw_data is None:
        raise ValueError(f"Configuration file '{file_path}' is empty.")

    if not isinstance(raw_data, Mapping):
        raise TypeError(
            f"Root configuration in '{file_path}' must be a mapping/object, got {type(raw_data).__name__}."
        )

    return parse_config(raw_data)

