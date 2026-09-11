"""Core profile representation for BehaviorSim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence
import numpy as np

from behaviorsim.core.transition import TransitionRule
from behaviorsim.core.utils import validate_transition_matrix


@dataclass
class Profile:
    """Domain-neutral profile representing an agent or behavioral persona.

    Attributes:
        name: Non-empty string identifier for the profile.
        state_emissions: Mapping from state_name -> feature_name -> FeatureDistribution (or emission spec).
        transition_matrix: Optional 2D square stochastic matrix overriding default transitions.
        transition_rules: Optional sequence of TransitionRule objects for conditional overrides.
        metadata: Optional dictionary of arbitrary metadata.
    """

    name: str
    state_emissions: Mapping[str, Mapping[str, Any]]
    transition_matrix: Optional[np.ndarray] = None
    transition_rules: Optional[Sequence[TransitionRule]] = None
    metadata: Optional[Mapping[str, Any]] = None
    probability: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Profile name must be a non-empty string.")

        if not isinstance(self.state_emissions, Mapping):
            raise ValueError(
                f"Profile state_emissions must be a mapping, got {type(self.state_emissions).__name__}."
            )

        if self.transition_matrix is not None:
            validate_transition_matrix(self.transition_matrix)

        if self.transition_rules is not None:
            if not isinstance(self.transition_rules, Iterable) or isinstance(
                self.transition_rules, (str, bytes)
            ):
                raise TypeError("Profile transition_rules must be an iterable of TransitionRule.")

            for i, rule in enumerate(self.transition_rules):
                if not isinstance(rule, TransitionRule):
                    raise TypeError(
                        f"Element at index {i} of transition_rules is not a TransitionRule instance."
                    )

        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise ValueError("Profile metadata must be a mapping.")

        if self.probability is not None:
            if isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)):
                raise TypeError(
                    f"Profile probability must be numeric, got {type(self.probability).__name__}."
                )
            prob = float(self.probability)
            if not np.isfinite(prob):
                raise ValueError("Profile probability must be a finite number.")
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Profile probability must be in [0.0, 1.0], got {self.probability}.")
