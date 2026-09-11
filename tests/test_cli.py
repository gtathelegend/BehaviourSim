"""Comprehensive tests for the BehaviorSim CLI and packaging (behaviorsim.cli)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict
from unittest.mock import patch

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest
import yaml

from behaviorsim import __version__
from behaviorsim.cli import main


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

def make_sample_config_dict() -> Dict[str, Any]:
    """Create a minimal valid configuration dictionary for CLI tests."""
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
            },
            {
                "name": "power_user",
                "state_emissions": {
                    "active": {
                        "latency": {"distribution": "normal", "params": {"loc": 5.0, "scale": 1.0}},
                        "success": {"distribution": "bernoulli", "params": {"p": 0.95}},
                    },
                    "idle": {
                        "latency": {"distribution": "normal", "params": {"loc": 8.0, "scale": 1.5}},
                        "success": {"distribution": "bernoulli", "params": {"p": 0.5}},
                    },
                },
            },
        ],
        "simulation": {
            "num_interactions": 10,
            "num_sequences": 2,
            "seed": 42,
            "initial_state": "active",
        },
    }


@pytest.fixture
def yaml_config_file(tmp_path: Path) -> Path:
    cfg = make_sample_config_dict()
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return file_path


@pytest.fixture
def yml_config_file(tmp_path: Path) -> Path:
    cfg = make_sample_config_dict()
    file_path = tmp_path / "config.yml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return file_path


@pytest.fixture
def json_config_file(tmp_path: Path) -> Path:
    cfg = make_sample_config_dict()
    file_path = tmp_path / "config.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return file_path


# ---------------------------------------------------------------------------
# CLI Top-Level & Metadata Tests
# ---------------------------------------------------------------------------

def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify behaviorsim --help outputs usage information and exits with 0."""
    code = main(["--help"])
    assert code == 0
    captured = capsys.readouterr()
    assert "usage: behaviorsim" in captured.out
    assert "validate" in captured.out
    assert "run" in captured.out


def test_cli_no_args_shows_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify behaviorsim with no arguments displays help and exits with 0."""
    code = main([])
    assert code == 0
    captured = capsys.readouterr()
    assert "usage: behaviorsim" in captured.out


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify behaviorsim --version outputs package version and exits with 0."""
    code = main(["--version"])
    assert code == 0
    captured = capsys.readouterr()
    # In argparse, action="version" prints to stdout (or stderr in Python < 3.4)
    output = captured.out or captured.err
    assert f"BehaviorSim {__version__}" in output


# ---------------------------------------------------------------------------
# Validate Subcommand Tests
# ---------------------------------------------------------------------------

def test_validate_valid_yaml(yaml_config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify validate on valid YAML configuration succeeds with exit code 0."""
    code = main(["validate", str(yaml_config_file)])
    assert code == 0
    captured = capsys.readouterr()
    assert "is valid" in captured.out
    assert captured.err == ""


def test_validate_valid_yml(yml_config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify validate on valid .yml configuration succeeds with exit code 0."""
    code = main(["validate", str(yml_config_file)])
    assert code == 0
    captured = capsys.readouterr()
    assert "is valid" in captured.out
    assert captured.err == ""


def test_validate_valid_json(json_config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify validate on valid JSON configuration succeeds with exit code 0."""
    code = main(["validate", str(json_config_file)])
    assert code == 0
    captured = capsys.readouterr()
    assert "is valid" in captured.out
    assert captured.err == ""


def test_validate_invalid_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify validate on invalid configuration reports error cleanly and exits with 1."""
    bad_cfg = make_sample_config_dict()
    # Make transition matrix invalid (row does not sum to 1.0)
    bad_cfg["transition_matrix"] = [[0.5, 0.2], [0.4, 0.6]]
    bad_file = tmp_path / "bad_config.yaml"
    with open(bad_file, "w", encoding="utf-8") as f:
        yaml.dump(bad_cfg, f)

    code = main(["validate", str(bad_file)])
    assert code == 1
    captured = capsys.readouterr()
    assert "Configuration Error" in captured.err
    assert "Traceback" not in captured.err


def test_validate_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify validate on non-existent file reports clean error and exits with 1."""
    missing = tmp_path / "nonexistent.yaml"
    code = main(["validate", str(missing)])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: Configuration file not found" in captured.err
    assert "Traceback" not in captured.err


def test_validate_empty_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify validate on empty file reports clean error and exits with 1."""
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    code = main(["validate", str(empty_file)])
    assert code == 1
    captured = capsys.readouterr()
    assert "Configuration Error: Configuration file" in captured.err
    assert "is empty" in captured.err
    assert "Traceback" not in captured.err


# ---------------------------------------------------------------------------
# Run Subcommand Tests
# ---------------------------------------------------------------------------

def test_run_stdout_csv(yaml_config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify run without --output streams CSV formatted data directly to stdout."""
    code = main(["run", str(yaml_config_file)])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.err == ""

    # Parse stdout back as CSV
    df = pd.read_csv(io.StringIO(captured.out))
    assert len(df) == 20  # 2 sequences * 10 interactions
    assert "sequence_id" in df.columns
    assert "interaction_id" in df.columns
    assert "state" in df.columns
    assert "latency" in df.columns
    assert "success" in df.columns


def test_run_output_csv(yaml_config_file: Path, tmp_path: Path) -> None:
    """Verify run --output output.csv exports valid CSV."""
    out_file = tmp_path / "output.csv"
    code = main(["run", str(yaml_config_file), "--output", str(out_file)])
    assert code == 0
    assert out_file.exists()

    df = pd.read_csv(out_file)
    assert len(df) == 20
    assert "sequence_id" in df.columns
    assert "interaction_id" in df.columns


def test_run_output_json(yaml_config_file: Path, tmp_path: Path) -> None:
    """Verify run --output output.json exports valid JSON."""
    out_file = tmp_path / "output.json"
    code = main(["run", str(yaml_config_file), "--output", str(out_file)])
    assert code == 0
    assert out_file.exists()

    df = pd.read_json(out_file)
    assert len(df) == 20
    assert "sequence_id" in df.columns
    assert "interaction_id" in df.columns


def test_run_seed_override(yaml_config_file: Path, tmp_path: Path) -> None:
    """Verify run --seed produces deterministic results matching seed override."""
    out_1 = tmp_path / "out_1.csv"
    out_2 = tmp_path / "out_2.csv"

    # Run twice with the exact same overridden seed
    code1 = main(["run", str(yaml_config_file), "--seed", "999", "--output", str(out_1)])
    code2 = main(["run", str(yaml_config_file), "--seed", "999", "--output", str(out_2)])

    assert code1 == 0
    assert code2 == 0

    df1 = pd.read_csv(out_1)
    df2 = pd.read_csv(out_2)
    pd.testing.assert_frame_equal(df1, df2)

    # Run with different seed -> different emissions
    out_3 = tmp_path / "out_3.csv"
    code3 = main(["run", str(yaml_config_file), "--seed", "111", "--output", str(out_3)])
    assert code3 == 0
    df3 = pd.read_csv(out_3)
    assert not df1["latency"].equals(df3["latency"])


def test_run_interactions_override(yaml_config_file: Path, tmp_path: Path) -> None:
    """Verify run --interactions overrides interaction count per sequence."""
    out_file = tmp_path / "interactions.csv"
    code = main(["run", str(yaml_config_file), "--interactions", "5", "--output", str(out_file)])
    assert code == 0

    df = pd.read_csv(out_file)
    # 2 sequences * 5 interactions = 10 rows
    assert len(df) == 10
    assert df["interaction_id"].max() == 5


def test_run_sequences_override(yaml_config_file: Path, tmp_path: Path) -> None:
    """Verify run --sequences overrides number of sequences."""
    out_file = tmp_path / "sequences.csv"
    code = main(["run", str(yaml_config_file), "--sequences", "4", "--output", str(out_file)])
    assert code == 0

    df = pd.read_csv(out_file)
    # 4 sequences * 10 interactions = 40 rows
    assert len(df) == 40
    assert set(df["sequence_id"].unique()) == {1, 2, 3, 4}


def test_run_profile_override(yaml_config_file: Path, tmp_path: Path) -> None:
    """Verify run --profile selects the specified profile."""
    out_file = tmp_path / "power_user.csv"
    code = main(["run", str(yaml_config_file), "--profile", "power_user", "--output", str(out_file)])
    assert code == 0

    df = pd.read_csv(out_file)
    assert len(df) == 20
    assert (df["profile"] == "power_user").all()


# ---------------------------------------------------------------------------
# Error Handling & Validation Tests
# ---------------------------------------------------------------------------

def test_run_invalid_seed_negative(yaml_config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify negative seed is rejected with code 1 and clean error message."""
    code = main(["run", str(yaml_config_file), "--seed", "-5"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: --seed must be a non-negative integer" in captured.err
    assert "Traceback" not in captured.err


def test_run_invalid_interactions_negative(yaml_config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify negative interactions count is rejected with code 1 and clean error message."""
    code = main(["run", str(yaml_config_file), "--interactions", "-1"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: --interactions must be an integer >= 1" in captured.err
    assert "Traceback" not in captured.err


def test_run_invalid_interactions_zero(yaml_config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify zero interactions count is rejected with code 1."""
    code = main(["run", str(yaml_config_file), "--interactions", "0"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: --interactions must be an integer >= 1" in captured.err


def test_run_invalid_sequences_negative(yaml_config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify negative sequence count is rejected with code 1."""
    code = main(["run", str(yaml_config_file), "--sequences", "-2"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: --sequences must be an integer >= 1" in captured.err


def test_run_unknown_profile(yaml_config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify unknown profile is rejected with code 1 and list of available profiles."""
    code = main(["run", str(yaml_config_file), "--profile", "nonexistent_persona"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: Requested profile 'nonexistent_persona' not found in configuration." in captured.err
    assert "power_user" in captured.err
    assert "standard_user" in captured.err
    assert "Traceback" not in captured.err


def test_run_unsupported_output_format(yaml_config_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify unsupported file extension produces clean error and exits with 1."""
    out_file = tmp_path / "output.xml"
    code = main(["run", str(yaml_config_file), "--output", str(out_file)])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: Unsupported output format '.xml'" in captured.err
    assert "Supported formats: .csv, .json, .parquet" in captured.err
    assert "Traceback" not in captured.err


def test_run_parquet_missing_engine(yaml_config_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify attempting parquet export when pyarrow is missing gives clean message."""
    out_file = tmp_path / "output.parquet"
    # Even if pyarrow is missing or present, mock to_parquet raising ImportError to test handling
    with patch.object(pd.DataFrame, "to_parquet", side_effect=ImportError("No module named pyarrow")):
        code = main(["run", str(yaml_config_file), "--output", str(out_file)])
        assert code == 1
        captured = capsys.readouterr()
        assert "Parquet export requires an optional engine" in captured.err
        assert "pip install pyarrow" in captured.err
        assert "Traceback" not in captured.err


def test_cli_parsing_error_invalid_arg(yaml_config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify unknown flags trigger argparse parsing error with exit code 2."""
    code = main(["run", str(yaml_config_file), "--unknown-flag-xyz"])
    assert code == 2
    captured = capsys.readouterr()
    assert "unrecognized arguments" in captured.err


def test_cli_parsing_error_missing_required_config(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify missing positional config triggers argparse error with exit code 2."""
    code = main(["run"])
    assert code == 2
    captured = capsys.readouterr()
    assert "the following arguments are required: config" in captured.err


def test_run_case_insensitive_extensions(yaml_config_file: Path, tmp_path: Path) -> None:
    """Verify uppercase extensions (.CSV, .JSON) are correctly recognized and written."""
    out_csv = tmp_path / "output.CSV"
    out_json = tmp_path / "output.JSON"

    code1 = main(["run", str(yaml_config_file), "--output", str(out_csv)])
    assert code1 == 0
    assert out_csv.exists()
    df_csv = pd.read_csv(out_csv)
    assert len(df_csv) == 20

    code2 = main(["run", str(yaml_config_file), "--output", str(out_json)])
    assert code2 == 0
    assert out_json.exists()
    df_json = pd.read_json(out_json)
    assert len(df_json) == 20


def test_run_missing_output_parent_directory(yaml_config_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify writing to a non-existent parent directory fails cleanly with code 1."""
    out_file = tmp_path / "nonexistent_dir" / "out.csv"
    code = main(["run", str(yaml_config_file), "--output", str(out_file)])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error writing output to" in captured.err
    assert "Traceback" not in captured.err


def test_cli_entry_point_subprocess(yaml_config_file: Path, tmp_path: Path) -> None:
    """Verify behaviorsim console script works end-to-end via subprocess."""
    import subprocess

    out_file = tmp_path / "subprocess_out.csv"

    # Test version
    ver_res = subprocess.run(["behaviorsim", "--version"], capture_output=True, text=True)
    assert ver_res.returncode == 0
    assert f"BehaviorSim {__version__}" in (ver_res.stdout + ver_res.stderr)

    # Test validate
    val_res = subprocess.run(["behaviorsim", "validate", str(yaml_config_file)], capture_output=True, text=True)
    assert val_res.returncode == 0
    assert "is valid" in val_res.stdout

    # Test run
    run_res = subprocess.run(
        ["behaviorsim", "run", str(yaml_config_file), "--output", str(out_file), "--interactions", "3"],
        capture_output=True,
        text=True,
    )
    assert run_res.returncode == 0
    assert out_file.exists()
    df = pd.read_csv(out_file)
    assert len(df) == 6  # 2 sequences * 3 interactions
