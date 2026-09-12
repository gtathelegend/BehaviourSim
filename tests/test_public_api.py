"""Unit tests for BehaviorSim public API exports and Simulator facade methods."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest
import yaml

import behaviorsim
from behaviorsim import (
    FeatureDistribution,
    Profile,
    SimulationConfig,
    Simulator,
    State,
    TransitionRule,
    __version__,
)


# ---------------------------------------------------------------------------
# Public Exports
# ---------------------------------------------------------------------------

def test_public_package_exports() -> None:
    """Verify primary public API symbols and version are exposed directly from behaviorsim."""
    assert isinstance(__version__, str)
    assert __version__ == "1.0.1"

    assert Simulator is behaviorsim.core.simulator.Simulator
    assert State is behaviorsim.core.state.State
    assert Profile is behaviorsim.core.profile.Profile
    assert FeatureDistribution is behaviorsim.core.feature.FeatureDistribution
    assert TransitionRule is behaviorsim.core.transition.TransitionRule
    assert SimulationConfig is behaviorsim.config.SimulationConfig


# ---------------------------------------------------------------------------
# Simulator.generate()
# ---------------------------------------------------------------------------

def test_generate_matches_simulate() -> None:
    """Verify Simulator.generate() produces identical results to Simulator.simulate()."""
    states = [State("active"), State("idle")]
    emissions = {
        "active": {
            "val": FeatureDistribution("normal", {"loc": 5.0, "scale": 1.0}),
        },
        "idle": {
            "val": FeatureDistribution("normal", {"loc": 1.0, "scale": 0.5}),
        },
    }
    matrix = np.array([[0.8, 0.2], [0.3, 0.7]])
    profile = Profile(name="user", state_emissions=emissions, transition_matrix=matrix)

    sim = Simulator(states=states, profile=profile)

    df_simulate = sim.simulate(num_interactions=100, num_sequences=2, seed=12345)
    df_generate = sim.generate(num_interactions=100, num_sequences=2, seed=12345)

    pd.testing.assert_frame_equal(df_simulate, df_generate)


# ---------------------------------------------------------------------------
# Simulator.from_preset()
# ---------------------------------------------------------------------------

def test_from_preset_education() -> None:
    """Verify Simulator.from_preset('education') constructs a valid Simulator and runs."""
    sim = Simulator.from_preset("education")
    assert isinstance(sim, Simulator)
    assert [s.name for s in sim.states] == ["Optimal", "Overload", "Underload"]
    assert sim.profile.name == "average"

    df = sim.generate(num_interactions=50, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    assert "accuracy" in df.columns
    assert "difficulty" in df.columns
    assert "nrt" in df.columns
    assert "confidence" in df.columns
    assert set(df["state"].unique()).issubset({"Optimal", "Overload", "Underload"})


def test_from_preset_education_with_custom_profile() -> None:
    """Verify passing profile parameter to education preset works."""
    sim = Simulator.from_preset("education", profile="fast_accurate")
    assert isinstance(sim, Simulator)
    assert sim.profile.name == "fast_accurate"

    # Also supports profile_name kwarg
    sim2 = Simulator.from_preset("education", profile_name="slow_inaccurate")
    assert sim2.profile.name == "slow_inaccurate"


def test_from_preset_unknown_name() -> None:
    """Verify unknown preset name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown preset 'unknown_preset'"):
        Simulator.from_preset("unknown_preset")


def test_from_preset_invalid_profile() -> None:
    """Verify invalid profile parameter propagates ValueError from preset factory."""
    with pytest.raises(ValueError, match="Unknown profile 'nonexistent'"):
        Simulator.from_preset("education", profile="nonexistent")


# ---------------------------------------------------------------------------
# Simulator.from_config()
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_raw_config() -> dict:
    return {
        "version": "1.0",
        "states": [
            {"name": "state_1", "description": "First state"},
            {"name": "state_2", "description": "Second state"},
        ],
        "transition_matrix": [
            [0.6, 0.4],
            [0.2, 0.8],
        ],
        "profiles": [
            {
                "name": "profile_alpha",
                "state_emissions": {
                    "state_1": {"score": {"distribution": "normal", "params": {"loc": 10.0, "scale": 1.0}}},
                    "state_2": {"score": {"distribution": "normal", "params": {"loc": 20.0, "scale": 2.0}}},
                },
            },
            {
                "name": "profile_beta",
                "state_emissions": {
                    "state_1": {"score": {"distribution": "normal", "params": {"loc": 50.0, "scale": 1.0}}},
                    "state_2": {"score": {"distribution": "normal", "params": {"loc": 60.0, "scale": 2.0}}},
                },
            },
        ],
        "simulation": {
            "num_interactions": 30,
            "num_sequences": 1,
            "seed": 42,
            "initial_state": "state_1",
        },
    }


def test_from_config_mapping(minimal_raw_config: dict) -> None:
    """Verify constructing Simulator from a configuration dictionary."""
    sim = Simulator.from_config(minimal_raw_config)
    assert isinstance(sim, Simulator)
    assert [s.name for s in sim.states] == ["state_1", "state_2"]
    assert sim.profile.name == "profile_alpha"

    # Verify construction does not generate data automatically
    # (sim is a fresh Simulator instance)
    df = sim.generate(num_interactions=10, seed=123)
    assert len(df) == 10
    assert "score" in df.columns


def test_from_config_simulation_config_object(minimal_raw_config: dict) -> None:
    """Verify constructing Simulator directly from a SimulationConfig dataclass."""
    from behaviorsim.config import parse_config

    cfg = parse_config(minimal_raw_config)
    sim = Simulator.from_config(cfg, profile_name="profile_beta")
    assert isinstance(sim, Simulator)
    assert sim.profile.name == "profile_beta"


def test_from_config_yaml_file(minimal_raw_config: dict) -> None:
    """Verify constructing Simulator from a YAML file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "test_config.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(minimal_raw_config, f)

        sim = Simulator.from_config(yaml_path)
        assert isinstance(sim, Simulator)
        assert sim.profile.name == "profile_alpha"

        # Also test string path
        sim_str = Simulator.from_config(str(yaml_path))
        assert isinstance(sim_str, Simulator)


def test_from_config_json_file(minimal_raw_config: dict) -> None:
    """Verify constructing Simulator from a JSON file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_config.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(minimal_raw_config, f)

        sim = Simulator.from_config(json_path)
        assert isinstance(sim, Simulator)
        assert sim.profile.name == "profile_alpha"


def test_from_config_missing_file() -> None:
    """Verify FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        Simulator.from_config(Path("nonexistent_file_path_12345.yaml"))


def test_from_config_malformed_input() -> None:
    """Verify ValueError is raised on malformed configuration data."""
    bad_config = {"version": "1.0", "states": []}
    with pytest.raises(ValueError, match="states must contain at least one state"):
        Simulator.from_config(bad_config)


def test_from_config_unsupported_type() -> None:
    """Verify TypeError is raised when unsupported type is passed to from_config."""
    with pytest.raises(TypeError, match="Unsupported config type"):
        Simulator.from_config(12345)  # type: ignore
