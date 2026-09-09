"""Core feature representation and distribution primitives for BehaviorSim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FeatureDistribution:
    """Domain-neutral specification of a statistical distribution for feature emission.

    Attributes:
        distribution_type: String identifier of the distribution family (e.g., "normal",
            "lognormal", "bernoulli", "categorical", "uniform_discrete").
        params: Mapping containing fixed numerical or categorical distribution parameters.
    """

    distribution_type: str
    params: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.distribution_type, str):
            raise ValueError(
                f"Distribution type must be a string, got {type(self.distribution_type).__name__}."
            )
        if not self.distribution_type.strip():
            raise ValueError("Distribution type must be a non-empty string.")
        if not isinstance(self.params, Mapping):
            raise ValueError(
                f"Distribution params must be a mapping (e.g. dict), got {type(self.params).__name__}."
            )
