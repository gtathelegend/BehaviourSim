"""Core state representation for BehaviorSim."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    """Domain-neutral discrete cognitive or behavioral state representation.

    Attributes:
        name: Non-empty string identifier for the state (e.g. "active", "inactive").
        description: Optional human-readable description of the state.
    """

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError(f"State name must be a string, got {type(self.name).__name__}.")
        if not self.name.strip():
            raise ValueError("State name must be a non-empty string.")
        if not isinstance(self.description, str):
            raise ValueError(
                f"State description must be a string, got {type(self.description).__name__}."
            )
