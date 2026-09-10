"""Domain-neutral preset registry for BehaviorSim."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

_PRESET_REGISTRY: Dict[str, Callable[..., Any]] = {}
_BUILTIN_INITIALIZED = False


def _ensure_builtins() -> None:
    """Ensure built-in presets are registered."""
    global _BUILTIN_INITIALIZED
    if not _BUILTIN_INITIALIZED:
        _BUILTIN_INITIALIZED = True
        import behaviorsim.presets  # noqa: F401


def register_preset(
    name: str,
    factory: Callable[..., Any],
    *,
    overwrite: bool = False,
) -> None:
    """Register a preset factory under a deterministic string name.

    Args:
        name: Deterministic string name for the preset.
        factory: Callable that instantiates and returns a Simulator or compatible object.
        overwrite: Whether to allow replacing an existing registered preset.

    Raises:
        ValueError: If name is not a non-empty string, or if already registered and overwrite is False.
        TypeError: If name is not a string or factory is not callable.
    """
    if not isinstance(name, str):
        raise TypeError(f"Preset name must be a string, got {type(name).__name__}.")
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("Preset name must be a non-empty string.")

    if not callable(factory):
        raise TypeError(f"Preset factory must be callable, got {type(factory).__name__}.")

    if cleaned_name in _PRESET_REGISTRY and not overwrite:
        raise ValueError(
            f"Preset '{cleaned_name}' is already registered. Use overwrite=True to replace."
        )

    _PRESET_REGISTRY[cleaned_name] = factory


def get_preset(name: str) -> Callable[..., Any]:
    """Retrieve the factory callable for a registered preset.

    Args:
        name: Name of the registered preset.

    Returns:
        The factory callable associated with the preset name.

    Raises:
        TypeError: If name is not a string.
        ValueError: If name is not registered.
    """
    if not isinstance(name, str):
        raise TypeError(f"Preset name must be a string, got {type(name).__name__}.")

    _ensure_builtins()

    cleaned_name = name.strip()
    if cleaned_name not in _PRESET_REGISTRY:
        available = list_presets()
        raise ValueError(
            f"Unknown preset '{cleaned_name}'. Available presets: {available}"
        )

    return _PRESET_REGISTRY[cleaned_name]


def list_presets() -> List[str]:
    """List all registered preset names in deterministic sorted order.

    Returns:
        Sorted list of registered preset names.
    """
    _ensure_builtins()
    return sorted(_PRESET_REGISTRY.keys())


def clear_presets() -> None:
    """Clear all registered presets (primarily for testing isolation)."""
    global _BUILTIN_INITIALIZED
    _PRESET_REGISTRY.clear()
    _BUILTIN_INITIALIZED = False
