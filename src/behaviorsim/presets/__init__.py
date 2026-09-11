"""Presets module containing domain-specific configurations for BehaviorSim."""

from behaviorsim.presets.registry import (
    clear_presets,
    get_preset,
    list_presets,
    register_preset,
)
from behaviorsim.presets.education import create_education_simulator
from behaviorsim.presets.healthcare import create_healthcare_simulator
from behaviorsim.presets.mobile_app import create_mobile_app_simulator

# Wire built-in presets that are constructible
register_preset("education", create_education_simulator, overwrite=True)
register_preset("healthcare", create_healthcare_simulator, overwrite=True)
register_preset("mobile_app", create_mobile_app_simulator, overwrite=True)

__all__ = [
    "register_preset",
    "get_preset",
    "list_presets",
    "clear_presets",
    "create_education_simulator",
    "create_healthcare_simulator",
    "create_mobile_app_simulator",
]
