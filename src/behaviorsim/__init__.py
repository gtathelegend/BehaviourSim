"""BehaviorSim: Synthetic Sequential Behavioral Data Generation Library."""

from behaviorsim.config import SimulationConfig
from behaviorsim.core.feature import FeatureDistribution
from behaviorsim.core.profile import Profile
from behaviorsim.core.simulator import Simulator
from behaviorsim.core.state import State
from behaviorsim.core.transition import TransitionRule

__version__ = "0.1.0"

__all__ = [
    "Simulator",
    "State",
    "Profile",
    "FeatureDistribution",
    "TransitionRule",
    "SimulationConfig",
    "__version__",
]
