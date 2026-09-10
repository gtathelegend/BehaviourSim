"""Unit tests for BehaviorSim configuration schema, validation, and rule compiler (behaviorsim.config)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List
import yaml

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from behaviorsim.config import (
    FeatureDistributionConfig,
    ProfileConfig,
    SimulationConfig,
    SimulationRunConfig,
    StateConfig,
    TransitionRuleConfig,
    build_simulator,
    compile_condition,
    load_config,
    parse_config,
    run_simulation,
    validate_condition_spec,
    validate_config,
    validate_distribution_params,
)
from behaviorsim.core.simulator import Simulator


# ---------------------------------------------------------------------------
# Helpers & Minimal Fixtures
# ---------------------------------------------------------------------------

def make_valid_raw_config() -> Dict[str, Any]:
    """Create a minimal valid raw configuration dictionary."""
    return {
        "version": "1.0",
        "states": [
            {"name": "active", "description": "Active state"},
            {"name": "idle", "description": "Idle state"},
        ],
        "transition_matrix": [
            [0.7, 0.3],
            [0.4, 0.6],
        ],
        "profiles": [
            {
                "name": "standard_user",
                "state_emissions": {
                    "active": {
                        "latency": {"distribution": "normal", "params": {"loc": 10.0, "scale": 2.0}},
                        "success": {"distribution": "bernoulli", "params": {"p": 0.8}},
                    },
                    "idle": {
                        "latency": {"distribution": "lognormal", "params": {"mean": 2.0, "sigma": 0.5}},
                        "success": {"distribution": "bernoulli", "params": {"p": 0.2}},
                    },
                },
            }
        ],
        "simulation": {
            "num_interactions": 20,
            "num_sequences": 1,
            "seed": 42,
            "initial_state": "active",
        },
    }


class DummyHistory:
    """Mock HistoryContext for testing compiled condition predicates."""

    def __init__(self, records: Dict[str, List[Any]] | None = None) -> None:
        self._records = records or {}

    def get_recent(self, feature_name: str, window: int) -> List[Any]:
        if feature_name not in self._records:
            return []
        vals = self._records[feature_name]
        return vals[-window:] if window > 0 else []


# ---------------------------------------------------------------------------
# Valid Configuration Tests
# ---------------------------------------------------------------------------

def test_minimal_valid_config() -> None:
    """Verify parsing and validation of a valid configuration."""
    raw = make_valid_raw_config()
    cfg = parse_config(raw)

    assert isinstance(cfg, SimulationConfig)
    assert cfg.version == "1.0"
    assert len(cfg.states) == 2
    assert cfg.states[0].name == "active"
    assert cfg.states[1].name == "idle"
    assert len(cfg.profiles) == 1
    assert cfg.profiles[0].name == "standard_user"
    assert cfg.simulation.num_interactions == 20
    assert cfg.simulation.seed == 42
    assert cfg.simulation.initial_state == "active"


def test_multiple_states_and_profiles() -> None:
    """Verify configuration with 3 states and multiple profiles."""
    raw = {
        "version": "1.0",
        "states": [
            {"name": "s1"},
            {"name": "s2"},
            {"name": "s3"},
        ],
        "transition_matrix": [
            [0.6, 0.2, 0.2],
            [0.1, 0.8, 0.1],
            [0.3, 0.3, 0.4],
        ],
        "profiles": [
            {
                "name": "p1",
                "state_emissions": {
                    "s1": {"v": {"distribution": "uniform", "params": {"low": 1, "high": 5}}},
                    "s2": {"v": {"distribution": "uniform", "params": {"low": 2, "high": 6}}},
                    "s3": {"v": {"distribution": "uniform", "params": {"low": 3, "high": 7}}},
                },
            },
            {
                "name": "p2",
                "state_emissions": {
                    "s1": {"v": {"distribution": "exponential", "params": {"scale": 1.5}}},
                    "s2": {"v": {"distribution": "exponential", "params": {"scale": 2.5}}},
                    "s3": {"v": {"distribution": "exponential", "params": {"scale": 3.5}}},
                },
                # Profile-level override matrix
                "transition_matrix": [
                    [0.8, 0.1, 0.1],
                    [0.2, 0.7, 0.1],
                    [0.1, 0.1, 0.8],
                ],
            },
        ],
        "simulation": {"num_interactions": 10},
    }
    cfg = parse_config(raw)
    assert len(cfg.profiles) == 2
    validate_config(cfg)  # Should not raise


def test_to_core_and_simulation_execution() -> None:
    """Verify that build_simulator generates a working Simulator instance."""
    raw = make_valid_raw_config()
    cfg = parse_config(raw)
    sim = build_simulator(cfg)

    assert isinstance(sim, Simulator)
    assert len(sim.states) == 2
    assert sim.initial_state == "active"

    df = sim.simulate(num_interactions=cfg.simulation.num_interactions, seed=cfg.simulation.seed)
    assert len(df) == 20
    assert "latency" in df.columns
    assert "success" in df.columns
    assert (df["profile"] == "standard_user").all()


def test_supported_distributions_valid_params() -> None:
    """Verify validation passes for all 8 supported distribution types with valid parameters."""
    valid_specs = [
        ("normal", {"loc": 0.0, "scale": 1.0}),
        ("lognormal", {"mean": 1.0, "sigma": 0.5}),
        ("exponential", {"scale": 2.0}),
        ("uniform", {"low": 0.0, "high": 10.0}),
        ("uniform_discrete", {"items": [1, 2, 3]}),
        ("uniform_discrete", {"low": 1, "high": 5}),
        ("bernoulli", {"p": 0.75}),
        ("poisson", {"lam": 3.5}),
        ("categorical", {"items": ["a", "b"], "probabilities": [0.3, 0.7]}),
        ("categorical", {"items": ["x", "y"]}),
    ]
    for dist_type, params in valid_specs:
        validate_distribution_params(dist_type, params)  # Should not raise


# ---------------------------------------------------------------------------
# Structural Failure Tests
# ---------------------------------------------------------------------------

def test_missing_or_empty_states() -> None:
    """Verify structural rejection when states are missing or empty."""
    raw1 = make_valid_raw_config()
    del raw1["states"]
    with pytest.raises(ValueError, match="states must contain at least one state"):
        parse_config(raw1)

    raw2 = make_valid_raw_config()
    raw2["states"] = []
    with pytest.raises(ValueError, match="states must contain at least one state"):
        parse_config(raw2)


def test_duplicate_state_names() -> None:
    """Verify error when duplicate state names are supplied."""
    raw = make_valid_raw_config()
    raw["states"] = [{"name": "active"}, {"name": "active"}]
    with pytest.raises(ValueError, match="duplicate state name: 'active'"):
        parse_config(raw)


def test_missing_or_empty_profiles() -> None:
    """Verify rejection when profiles are missing or empty."""
    raw1 = make_valid_raw_config()
    del raw1["profiles"]
    with pytest.raises(ValueError, match="profiles must contain at least one profile"):
        parse_config(raw1)

    raw2 = make_valid_raw_config()
    raw2["profiles"] = []
    with pytest.raises(ValueError, match="profiles must contain at least one profile"):
        parse_config(raw2)


def test_duplicate_profile_names() -> None:
    """Verify error on duplicate profile names."""
    raw = make_valid_raw_config()
    p = raw["profiles"][0]
    raw["profiles"] = [p, p]
    with pytest.raises(ValueError, match="duplicate profile name: 'standard_user'"):
        parse_config(raw)


def test_invalid_simulation_parameters() -> None:
    """Verify rejection of invalid simulation configuration values."""
    raw = make_valid_raw_config()
    raw["simulation"]["num_interactions"] = 0
    with pytest.raises(ValueError, match="num_interactions must be an integer >= 1"):
        parse_config(raw)

    raw["simulation"]["num_interactions"] = True  # Boolean must be rejected
    with pytest.raises(ValueError, match="num_interactions must be an integer >= 1"):
        parse_config(raw)

    raw["simulation"]["num_interactions"] = 10
    raw["simulation"]["num_sequences"] = -1
    with pytest.raises(ValueError, match="num_sequences must be an integer >= 1"):
        parse_config(raw)

    raw["simulation"]["num_sequences"] = 1
    raw["simulation"]["seed"] = True  # Boolean seed rejected
    with pytest.raises(ValueError, match="seed must be an integer or None"):
        parse_config(raw)


# ---------------------------------------------------------------------------
# Semantic Failure Tests
# ---------------------------------------------------------------------------

def test_transition_matrix_dimension_mismatch() -> None:
    """Verify error when matrix dimensions do not match state count."""
    raw = make_valid_raw_config()
    # 2 states, but 3x3 matrix
    raw["transition_matrix"] = [
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
    ]
    with pytest.raises(ValueError, match="transition_matrix must have shape"):
        parse_config(raw)


def test_unknown_state_reference_in_profile_emissions() -> None:
    """Verify error when profile state_emissions references an unregistered state."""
    raw = make_valid_raw_config()
    raw["profiles"][0]["state_emissions"]["unknown_state"] = {
        "x": {"distribution": "normal", "params": {"scale": 1}}
    }
    with pytest.raises(ValueError, match="references unknown state 'unknown_state'"):
        parse_config(raw)


def test_unknown_initial_state() -> None:
    """Verify error when simulation initial_state is not in configured states."""
    raw = make_valid_raw_config()
    raw["simulation"]["initial_state"] = "non_existent"
    with pytest.raises(ValueError, match="initial_state 'non_existent' does not exist in configured states"):
        parse_config(raw)


def test_profile_missing_transition_matrix_when_no_toplevel() -> None:
    """Verify error when neither top-level nor profile transition matrix is provided."""
    raw = make_valid_raw_config()
    del raw["transition_matrix"]
    with pytest.raises(ValueError, match="has no transition_matrix and no top-level transition_matrix is provided"):
        parse_config(raw)


def test_unsupported_distribution_type() -> None:
    """Verify error on unsupported distribution family."""
    raw = make_valid_raw_config()
    raw["profiles"][0]["state_emissions"]["active"]["latency"] = {
        "distribution": "cauchy",
        "params": {},
    }
    with pytest.raises(ValueError, match="unsupported distribution type: 'cauchy'"):
        parse_config(raw)


def test_invalid_distribution_parameters() -> None:
    """Verify parameter validation errors for each distribution type."""
    # Negative scale normal
    with pytest.raises(ValueError, match="normal parameter 'scale' must be non-negative"):
        validate_distribution_params("normal", {"scale": -1.0})

    # Negative sigma lognormal
    with pytest.raises(ValueError, match="lognormal parameter 'sigma' must be non-negative"):
        validate_distribution_params("lognormal", {"sigma": -0.5})

    # Non-positive scale exponential
    with pytest.raises(ValueError, match="exponential parameter 'scale' must be positive"):
        validate_distribution_params("exponential", {"scale": -2.0})
    with pytest.raises(ValueError, match="exponential parameter 'scale' must be positive"):
        validate_distribution_params("exponential", {"scale": 0.0})

    # Uniform high < low
    with pytest.raises(ValueError, match="uniform parameter 'high' must be >= 'low'"):
        validate_distribution_params("uniform", {"low": 10.0, "high": 5.0})

    # Uniform discrete high < low
    with pytest.raises(ValueError, match="uniform_discrete parameter 'high' must be >= 'low'"):
        validate_distribution_params("uniform_discrete", {"low": 10, "high": 5})

    # Bernoulli p out of [0, 1]
    with pytest.raises(ValueError, match="bernoulli parameter 'p' must be between 0 and 1"):
        validate_distribution_params("bernoulli", {"p": 1.5})

    # Poisson lam < 0
    with pytest.raises(ValueError, match="poisson parameter 'lam' must be non-negative"):
        validate_distribution_params("poisson", {"lam": -1})

    # Categorical probabilities sum != 1
    with pytest.raises(ValueError, match="categorical probabilities must sum to 1.0"):
        validate_distribution_params("categorical", {"items": ["a", "b"], "probabilities": [0.4, 0.4]})


# ---------------------------------------------------------------------------
# Declarative Transition Rules & Condition Tests
# ---------------------------------------------------------------------------

def test_condition_compilation_all_operators() -> None:
    """Verify all 6 supported comparison operators evaluate accurately."""
    cases = [
        ("<", 5.0, 4.0, True),
        ("<", 5.0, 5.0, False),
        ("<=", 5.0, 5.0, True),
        (">", 5.0, 6.0, True),
        (">=", 5.0, 5.0, True),
        ("==", 5.0, 5.0, True),
        ("!=", 5.0, 6.0, True),
        ("!=", 5.0, 5.0, False),
    ]
    for op, threshold, history_val, expected in cases:
        spec = {"feature": "metric", "operator": op, "value": threshold}
        pred = compile_condition(spec)
        history = DummyHistory({"metric": [history_val]})
        assert pred(history) == expected, f"Failed for {history_val} {op} {threshold}"


def test_condition_compilation_all_aggregations() -> None:
    """Verify mean, sum, min, max, last aggregations."""
    history = DummyHistory({"acc": [0.2, 0.4, 0.6, 0.8, 1.0]})

    # mean of last 5 is 0.6
    pred_mean = compile_condition({"feature": "acc", "window": 5, "aggregation": "mean", "operator": "==", "value": 0.6})
    assert pred_mean(history) is True

    # sum of last 5 is 3.0
    pred_sum = compile_condition({"feature": "acc", "window": 5, "aggregation": "sum", "operator": "==", "value": 3.0})
    assert pred_sum(history) is True

    # min of last 5 is 0.2
    pred_min = compile_condition({"feature": "acc", "window": 5, "aggregation": "min", "operator": "==", "value": 0.2})
    assert pred_min(history) is True

    # max of last 5 is 1.0
    pred_max = compile_condition({"feature": "acc", "window": 5, "aggregation": "max", "operator": "==", "value": 1.0})
    assert pred_max(history) is True

    # last is 1.0
    pred_last = compile_condition({"feature": "acc", "window": 5, "aggregation": "last", "operator": "==", "value": 1.0})
    assert pred_last(history) is True


def test_empty_history_safely_evaluates_to_false() -> None:
    """Verify empty history does not crash and safely evaluates to False."""
    spec = {"feature": "acc", "operator": "<", "value": 0.5}
    pred = compile_condition(spec)
    empty_history = DummyHistory({})
    assert pred(empty_history) is False


def test_compound_conditions_all_and_any() -> None:
    """Verify 'all' (AND) and 'any' (OR) compound condition evaluation."""
    history = DummyHistory({"score": [85], "errors": [2]})

    # all: score > 80 AND errors < 5 -> True
    spec_all_true = {
        "all": [
            {"feature": "score", "operator": ">", "value": 80},
            {"feature": "errors", "operator": "<", "value": 5},
        ]
    }
    assert compile_condition(spec_all_true)(history) is True

    # all: score > 90 AND errors < 5 -> False
    spec_all_false = {
        "all": [
            {"feature": "score", "operator": ">", "value": 90},
            {"feature": "errors", "operator": "<", "value": 5},
        ]
    }
    assert compile_condition(spec_all_false)(history) is False

    # any: score > 90 OR errors < 5 -> True
    spec_any_true = {
        "any": [
            {"feature": "score", "operator": ">", "value": 90},
            {"feature": "errors", "operator": "<", "value": 5},
        ]
    }
    assert compile_condition(spec_any_true)(history) is True

    # any: score > 90 OR errors > 5 -> False
    spec_any_false = {
        "any": [
            {"feature": "score", "operator": ">", "value": 90},
            {"feature": "errors", "operator": ">", "value": 5},
        ]
    }
    assert compile_condition(spec_any_false)(history) is False


def test_invalid_condition_specifications() -> None:
    """Verify validation errors for malformed condition specifications."""
    # Missing feature
    with pytest.raises(ValueError, match="missing required field: 'feature'"):
        validate_condition_spec({"operator": "<", "value": 5})

    # Unsupported operator
    with pytest.raises(ValueError, match="Unsupported condition operator"):
        validate_condition_spec({"feature": "f", "operator": "~=", "value": 5})

    # Invalid window
    with pytest.raises(ValueError, match="Condition 'window' must be an integer >= 1"):
        validate_condition_spec({"feature": "f", "operator": "==", "value": 5, "window": 0})

    # Unsupported aggregation
    with pytest.raises(ValueError, match="Unsupported condition aggregation"):
        validate_condition_spec({"feature": "f", "operator": "==", "value": 5, "aggregation": "std"})

    # Both all and any
    with pytest.raises(ValueError, match="cannot contain both 'all' and 'any'"):
        validate_condition_spec({"all": [], "any": []})


def test_invalid_transition_rule_target_state() -> None:
    """Verify error when transition rule targets an unregistered state."""
    raw = make_valid_raw_config()
    raw["profiles"][0]["transition_rules"] = [
        {
            "condition": {"feature": "latency", "operator": ">", "value": 10},
            "target_state": "non_existent_state",
            "probability": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="target_state 'non_existent_state' does not exist"):
        parse_config(raw)


def test_transition_rule_probabilities_and_boundaries() -> None:
    """Verify probability bounds [0, 1] on transition rules."""
    rule_0 = TransitionRuleConfig(
        condition={"feature": "f", "operator": "==", "value": 1},
        target_state="s",
        probability=0.0,
    )
    assert rule_0.probability == 0.0

    rule_1 = TransitionRuleConfig(
        condition={"feature": "f", "operator": "==", "value": 1},
        target_state="s",
        probability=1.0,
    )
    assert rule_1.probability == 1.0

    with pytest.raises(ValueError, match="probability' must be in"):
        TransitionRuleConfig(
            condition={"feature": "f", "operator": "==", "value": 1},
            target_state="s",
            probability=1.5,
        )


# ---------------------------------------------------------------------------
# Security & Safety Audit
# ---------------------------------------------------------------------------

def test_no_arbitrary_python_expression_execution() -> None:
    """Verify arbitrary Python code strings cannot be executed in conditions."""
    malicious_specs = [
        {"feature": "__import__('os').system('echo pwned')", "operator": "==", "value": 1},
        {"feature": "f", "operator": "==", "value": "1; import os; os.system('echo pwned')"},
    ]
    for spec in malicious_specs:
        # Should either compile safely as literal string matching or fail validation
        pred = compile_condition(spec)
        history = DummyHistory({"f": [1]})
        # Must not crash, execute code, or raise security exceptions
        res = pred(history)
        assert isinstance(res, bool)


# ---------------------------------------------------------------------------
# File Loading (YAML / JSON) Tests
# ---------------------------------------------------------------------------

def test_load_config_yaml_and_yml() -> None:
    """Verify loading from both .yaml and .yml files."""
    raw = make_valid_raw_config()
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_yaml = Path(tmp_dir) / "sim.yaml"
        p_yml = Path(tmp_dir) / "sim.yml"

        with open(p_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)
        with open(p_yml, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)

        cfg1 = load_config(p_yaml)
        cfg2 = load_config(p_yml)

        assert isinstance(cfg1, SimulationConfig)
        assert isinstance(cfg2, SimulationConfig)
        assert cfg1.version == "1.0"
        assert cfg2.version == "1.0"
        assert len(cfg1.states) == 2
        assert len(cfg2.states) == 2


def test_load_config_json() -> None:
    """Verify loading from a .json file."""
    raw = make_valid_raw_config()
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_json = Path(tmp_dir) / "sim.json"
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump(raw, f)

        cfg = load_config(p_json)
        assert isinstance(cfg, SimulationConfig)
        assert cfg.version == "1.0"
        assert cfg.simulation.num_interactions == 20


def test_load_config_case_insensitive_extension() -> None:
    """Verify loading handles uppercase file extensions (.YAML, .JSON, .YML)."""
    raw = make_valid_raw_config()
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_yaml_upper = Path(tmp_dir) / "CONFIG.YAML"
        p_json_upper = Path(tmp_dir) / "CONFIG.JSON"
        p_yml_upper = Path(tmp_dir) / "CONFIG.YML"

        with open(p_yaml_upper, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)
        with open(p_json_upper, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        with open(p_yml_upper, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)

        cfg_yaml = load_config(p_yaml_upper)
        cfg_json = load_config(p_json_upper)
        cfg_yml = load_config(p_yml_upper)

        assert cfg_yaml.version == "1.0"
        assert cfg_json.version == "1.0"
        assert cfg_yml.version == "1.0"


def test_load_config_unsupported_extension() -> None:
    """Verify rejection of unsupported configuration file extensions."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_toml = Path(tmp_dir) / "sim.toml"
        p_toml.write_text("version = '1.0'")
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            load_config(p_toml)

        p_txt = Path(tmp_dir) / "sim.txt"
        p_txt.write_text("version: 1.0")
        with pytest.raises(ValueError, match="Unsupported configuration file format"):
            load_config(p_txt)


def test_load_config_missing_file() -> None:
    """Verify FileNotFoundError is raised when target configuration file does not exist."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config("non_existent_config_file.yaml")


def test_load_config_malformed_yaml() -> None:
    """Verify ValueError is raised when YAML content is malformed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "bad.yaml"
        p.write_text("states: [unclosed_bracket", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse configuration file"):
            load_config(p)


def test_load_config_malformed_json() -> None:
    """Verify ValueError is raised when JSON content is malformed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "bad.json"
        p.write_text('{"states": [unclosed', encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse configuration file"):
            load_config(p)


def test_load_config_empty_file() -> None:
    """Verify ValueError is raised when configuration file is empty."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "empty.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="is empty"):
            load_config(p)


def test_load_config_non_mapping_root() -> None:
    """Verify TypeError is raised when root document is a list instead of a mapping."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_yaml = Path(tmp_dir) / "list.yaml"
        p_yaml.write_text("- state_a\n- state_b\n", encoding="utf-8")
        with pytest.raises(TypeError, match="must be a mapping/object"):
            load_config(p_yaml)

        p_json = Path(tmp_dir) / "list.json"
        p_json.write_text('["state_a", "state_b"]', encoding="utf-8")
        with pytest.raises(TypeError, match="must be a mapping/object"):
            load_config(p_json)


def test_yaml_json_equivalence() -> None:
    """Verify equivalent YAML and JSON files produce equivalent SimulationConfig objects."""
    raw = make_valid_raw_config()
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_yaml = Path(tmp_dir) / "sim.yaml"
        p_json = Path(tmp_dir) / "sim.json"

        with open(p_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)
        with open(p_json, "w", encoding="utf-8") as f:
            json.dump(raw, f)

        cfg_yaml = load_config(p_yaml)
        cfg_json = load_config(p_json)

        assert cfg_yaml.version == cfg_json.version
        assert [s.name for s in cfg_yaml.states] == [s.name for s in cfg_json.states]
        assert [p.name for p in cfg_yaml.profiles] == [p.name for p in cfg_json.profiles]
        np.testing.assert_array_equal(cfg_yaml.transition_matrix, cfg_json.transition_matrix)
        assert cfg_yaml.simulation == cfg_json.simulation


def test_end_to_end_loading_and_simulation() -> None:
    """Verify end-to-end flow: file -> load_config -> build_simulator -> simulate -> DataFrame."""
    raw = make_valid_raw_config()
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_yaml = Path(tmp_dir) / "model.yaml"
        with open(p_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)

        cfg = load_config(p_yaml)
        sim = build_simulator(cfg)
        df = sim.simulate(num_interactions=cfg.simulation.num_interactions, seed=cfg.simulation.seed)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20
        assert list(df["sequence_id"]) == [1] * 20
        assert list(df["interaction_id"]) == list(range(1, 21))
        assert set(df["state"].unique()).issubset({"active", "idle"})
        assert "latency" in df.columns
        assert "success" in df.columns


def test_deterministic_execution_from_loaded_config() -> None:
    """Verify identical seeds produce identical DataFrames, and different seeds differ."""
    raw = make_valid_raw_config()
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_yaml = Path(tmp_dir) / "det.yaml"
        with open(p_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)

        cfg = load_config(p_yaml)
        df1 = run_simulation(cfg)
        df2 = run_simulation(cfg)
        pd.testing.assert_frame_equal(df1, df2)

        # Alter seed
        cfg.simulation.seed = 9999
        df_diff = run_simulation(cfg)
        assert not df1.equals(df_diff)


def test_invalid_yaml_configuration_fails_validation() -> None:
    """Verify semantic validation errors are caught when loading invalid YAML files."""
    raw = make_valid_raw_config()
    raw["profiles"][0]["state_emissions"]["unknown_state"] = {
        "x": {"distribution": "normal", "params": {"scale": 1}}
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "invalid.yaml"
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)

        with pytest.raises(ValueError, match="references unknown state 'unknown_state'"):
            load_config(p)


def test_invalid_json_configuration_fails_validation() -> None:
    """Verify semantic validation errors are caught when loading invalid JSON files."""
    raw = make_valid_raw_config()
    raw["simulation"]["num_interactions"] = -5
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "invalid.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(raw, f)

        with pytest.raises(ValueError, match="num_interactions must be an integer >= 1"):
            load_config(p)


def test_load_config_accepts_str_and_path() -> None:
    """Verify load_config accepts both pathlib.Path and string path arguments."""
    raw = make_valid_raw_config()
    with tempfile.TemporaryDirectory() as tmp_dir:
        p_path = Path(tmp_dir) / "sim.yaml"
        with open(p_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f)

        cfg_from_path = load_config(p_path)
        cfg_from_str = load_config(str(p_path))

        assert isinstance(cfg_from_path, SimulationConfig)
        assert isinstance(cfg_from_str, SimulationConfig)
        assert cfg_from_path.version == cfg_from_str.version


# ---------------------------------------------------------------------------
# API & Validation Hardening Tests (Step 0.4.4)
# ---------------------------------------------------------------------------

def test_config_non_mutation_during_simulation() -> None:
    """Verify building and running a simulation does not mutate the source SimulationConfig."""
    raw = make_valid_raw_config()
    cfg = parse_config(raw)

    orig_version = cfg.version
    orig_states_count = len(cfg.states)
    orig_profiles_count = len(cfg.profiles)
    orig_num_interactions = cfg.simulation.num_interactions
    orig_seed = cfg.simulation.seed
    orig_matrix = np.array(cfg.transition_matrix, copy=True)

    sim = build_simulator(cfg)
    df = sim.simulate(num_interactions=cfg.simulation.num_interactions, seed=cfg.simulation.seed)
    assert len(df) == 20

    # Ensure all original configuration values remained unchanged
    assert cfg.version == orig_version
    assert len(cfg.states) == orig_states_count
    assert len(cfg.profiles) == orig_profiles_count
    assert cfg.simulation.num_interactions == orig_num_interactions
    assert cfg.simulation.seed == orig_seed
    np.testing.assert_array_equal(cfg.transition_matrix, orig_matrix)


def test_multiple_profiles_selection_and_unknown_profile() -> None:
    """Verify selecting profiles in multi-profile configurations and error handling."""
    raw = {
        "version": "1.0",
        "states": [{"name": "s1"}, {"name": "s2"}],
        "transition_matrix": [[0.5, 0.5], [0.5, 0.5]],
        "profiles": [
            {
                "name": "prof_alpha",
                "state_emissions": {
                    "s1": {"v": {"distribution": "normal", "params": {"loc": 10.0, "scale": 1.0}}},
                    "s2": {"v": {"distribution": "normal", "params": {"loc": 20.0, "scale": 1.0}}},
                },
            },
            {
                "name": "prof_beta",
                "state_emissions": {
                    "s1": {"v": {"distribution": "normal", "params": {"loc": 50.0, "scale": 1.0}}},
                    "s2": {"v": {"distribution": "normal", "params": {"loc": 60.0, "scale": 1.0}}},
                },
            },
        ],
        "simulation": {"num_interactions": 10, "seed": 42},
    }
    cfg = parse_config(raw)

    # Default selection uses first profile (prof_alpha)
    sim_default = build_simulator(cfg)
    assert sim_default.profile.name == "prof_alpha"

    # Explicit selection of prof_beta
    sim_beta = build_simulator(cfg, profile_name="prof_beta")
    assert sim_beta.profile.name == "prof_beta"

    df_beta = run_simulation(cfg, profile_name="prof_beta")
    assert (df_beta["profile"] == "prof_beta").all()
    assert (df_beta["v"] >= 45.0).all()  # Values centered around 50/60

    # Unknown profile name raises clear ValueError
    with pytest.raises(ValueError, match="Profile 'unknown' not found in configuration"):
        build_simulator(cfg, profile_name="unknown")


def test_transition_matrix_precedence_semantics() -> None:
    """Verify profile-specific transition matrix overrides top-level transition matrix."""
    matrix_top = [[0.9, 0.1], [0.1, 0.9]]
    matrix_profile_override = [[0.1, 0.9], [0.9, 0.1]]

    raw = {
        "version": "1.0",
        "states": [{"name": "s1"}, {"name": "s2"}],
        "transition_matrix": matrix_top,
        "profiles": [
            {
                "name": "default_user",
                "state_emissions": {"s1": {}, "s2": {}},
                # Uses top-level matrix
            },
            {
                "name": "override_user",
                "state_emissions": {"s1": {}, "s2": {}},
                "transition_matrix": matrix_profile_override,
            },
        ],
        "simulation": {"num_interactions": 10},
    }
    cfg = parse_config(raw)

    sim_default = build_simulator(cfg, profile_name="default_user")
    np.testing.assert_array_equal(sim_default.transition_matrix, matrix_top)

    sim_override = build_simulator(cfg, profile_name="override_user")
    np.testing.assert_array_equal(sim_override.transition_matrix, matrix_profile_override)


def test_no_toplevel_matrix_with_all_profile_matrices() -> None:
    """Verify configuration succeeds when top-level matrix is omitted but all profiles provide matrices."""
    raw = {
        "version": "1.0",
        "states": [{"name": "s1"}, {"name": "s2"}],
        "profiles": [
            {
                "name": "p1",
                "state_emissions": {"s1": {}, "s2": {}},
                "transition_matrix": [[0.5, 0.5], [0.5, 0.5]],
            },
            {
                "name": "p2",
                "state_emissions": {"s1": {}, "s2": {}},
                "transition_matrix": [[0.8, 0.2], [0.2, 0.8]],
            },
        ],
        "simulation": {"num_interactions": 10},
    }
    cfg = parse_config(raw)
    validate_config(cfg)
    sim1 = build_simulator(cfg, profile_name="p1")
    sim2 = build_simulator(cfg, profile_name="p2")
    assert sim1.transition_matrix.shape == (2, 2)
    assert sim2.transition_matrix.shape == (2, 2)


def test_nested_compound_conditions() -> None:
    """Verify nested all/any condition evaluations."""
    spec = {
        "all": [
            {"feature": "latency", "operator": ">", "value": 5.0},
            {
                "any": [
                    {"feature": "status", "operator": "==", "value": "failed"},
                    {"feature": "retries", "operator": ">", "value": 2},
                ]
            },
        ]
    }
    pred = compile_condition(spec)

    # latency > 5 and status == 'failed' -> True
    h1 = DummyHistory({"latency": [10.0], "status": ["failed"], "retries": [0]})
    assert pred(h1) is True

    # latency > 5 and retries > 2 -> True
    h2 = DummyHistory({"latency": [8.0], "status": ["ok"], "retries": [3]})
    assert pred(h2) is True

    # latency <= 5 -> False
    h3 = DummyHistory({"latency": [4.0], "status": ["failed"], "retries": [3]})
    assert pred(h3) is False

    # latency > 5, but neither failed nor retries > 2 -> False
    h4 = DummyHistory({"latency": [10.0], "status": ["ok"], "retries": [1]})
    assert pred(h4) is False


def test_condition_string_and_type_safety() -> None:
    """Verify string equality conditions and type-mismatch safety."""
    spec_str = {"feature": "mode", "operator": "==", "value": "fast"}
    pred_str = compile_condition(spec_str)

    assert pred_str(DummyHistory({"mode": ["fast"]})) is True
    assert pred_str(DummyHistory({"mode": ["slow"]})) is False

    # Incompatible comparison (e.g. comparing string value with '<' numeric threshold) safely returns False
    spec_incompatible = {"feature": "mode", "operator": "<", "value": 10.0}
    pred_incompatible = compile_condition(spec_incompatible)
    assert pred_incompatible(DummyHistory({"mode": ["fast"]})) is False


def test_yaml_safety_rejects_arbitrary_python_objects() -> None:
    """Verify YAML loader rejects unsafe Python object constructor tags."""
    unsafe_yaml = "states: !!python/object/apply:os.system ['echo unsafe']\n"
    with tempfile.TemporaryDirectory() as tmp_dir:
        p = Path(tmp_dir) / "unsafe.yaml"
        p.write_text(unsafe_yaml, encoding="utf-8")

        with pytest.raises(ValueError, match="Failed to parse configuration file"):
            load_config(p)


