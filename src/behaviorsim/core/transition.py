"""Core transition matrix and transition rule logic for BehaviorSim."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class HistoryContext(Protocol):
    """Protocol for accessing chronological sequence history during rule evaluation."""

    def get_recent(self, feature_name: str, window: int) -> Sequence[Any]:
        """Return the most recent values for a given feature over a window length."""
        ...


@dataclass(frozen=True)
class TransitionRule:
    """Domain-neutral rule for performance/history dependent state transition overrides.

    Attributes:
        condition: Callable accepting a HistoryContext (or context object) returning bool.
        target_state: Non-empty string identifier of the state to transition to when condition holds.
        probability: Probability in [0, 1] of triggering the override when condition holds.
    """

    condition: Callable[[Any], bool]
    target_state: str
    probability: float = 1.0

    def __post_init__(self) -> None:
        if not callable(self.condition):
            raise TypeError("TransitionRule condition must be a callable.")

        if not isinstance(self.target_state, str) or not self.target_state.strip():
            raise ValueError("TransitionRule target_state must be a non-empty string.")

        if not isinstance(self.probability, (int, float)):
            raise TypeError(
                f"TransitionRule probability must be numeric, got {type(self.probability).__name__}."
            )

        prob_float = float(self.probability)
        if math.isnan(prob_float) or math.isinf(prob_float):
            raise ValueError("TransitionRule probability must be a finite number.")

        if not (0.0 <= prob_float <= 1.0):
            raise ValueError(
                f"TransitionRule probability must be within [0.0, 1.0], got {self.probability}."
            )
