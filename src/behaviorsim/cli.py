"""Command-line interface for BehaviorSim."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Optional, Sequence

from behaviorsim import __version__
from behaviorsim.config import build_simulator, load_config


def create_parser() -> argparse.ArgumentParser:
    """Build and return top-level argparse ArgumentParser for behaviorsim."""
    parser = argparse.ArgumentParser(
        prog="behaviorsim",
        description="BehaviorSim: Synthetic Sequential Behavioral Data Generation CLI",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"BehaviorSim {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Subcommands",
    )

    # Subcommand: validate
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a simulation configuration file (YAML/JSON)",
    )
    validate_parser.add_argument(
        "config",
        help="Path to YAML/JSON configuration file",
    )

    # Subcommand: run
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a simulation from a configuration file",
    )
    run_parser.add_argument(
        "config",
        help="Path to YAML/JSON configuration file",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (.csv, .json, or .parquet). If omitted, prints CSV to stdout.",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RNG seed (integer >= 0)",
    )
    run_parser.add_argument(
        "--interactions",
        type=int,
        default=None,
        help="Override number of interactions per sequence (integer >= 1)",
    )
    run_parser.add_argument(
        "--sequences",
        type=int,
        default=None,
        help="Override number of sequences to simulate (integer >= 1)",
    )
    run_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Target profile name for single-profile simulation override",
    )

    return parser


def handle_validate(args: argparse.Namespace) -> int:
    """Execute the validate subcommand. Returns exit code 0 or 1."""
    config_path = Path(args.config)
    try:
        load_config(config_path)
    except FileNotFoundError as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1
    except (ValueError, TypeError) as err:
        sys.stderr.write(f"Configuration Error: {err}\n")
        return 1
    except Exception as err:
        # Fallback for unexpected file I/O or system errors without bare traceback
        sys.stderr.write(f"Error loading configuration: {err}\n")
        return 1

    sys.stdout.write(f"Configuration '{config_path}' is valid.\n")
    return 0


def handle_run(args: argparse.Namespace) -> int:
    """Execute the run subcommand. Returns exit code 0 or 1."""
    config_path = Path(args.config)

    # Validate CLI overrides before running
    if args.seed is not None and args.seed < 0:
        sys.stderr.write(f"Error: --seed must be a non-negative integer, got {args.seed}\n")
        return 1
    if args.interactions is not None and args.interactions < 1:
        sys.stderr.write(
            f"Error: --interactions must be an integer >= 1, got {args.interactions}\n"
        )
        return 1
    if args.sequences is not None and args.sequences < 1:
        sys.stderr.write(f"Error: --sequences must be an integer >= 1, got {args.sequences}\n")
        return 1

    # Load and validate config via existing config API
    try:
        config = load_config(config_path)
    except FileNotFoundError as err:
        sys.stderr.write(f"Error: {err}\n")
        return 1
    except (ValueError, TypeError) as err:
        sys.stderr.write(f"Configuration Error: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Error loading configuration: {err}\n")
        return 1

    # Validate requested profile if supplied
    profile_override = args.profile
    if profile_override is not None:
        known_profiles = {p.name for p in config.profiles}
        if profile_override not in known_profiles:
            sys.stderr.write(
                f"Error: Requested profile '{profile_override}' not found in configuration. "
                f"Available profiles: {sorted(known_profiles)}\n"
            )
            return 1

    # Build simulator using existing configuration API
    try:
        simulator = build_simulator(config, profile_name=profile_override)
    except (ValueError, TypeError) as err:
        sys.stderr.write(f"Simulator Error: {err}\n")
        return 1

    # Determine simulation parameters with CLI overrides taking precedence
    num_interactions = (
        args.interactions
        if args.interactions is not None
        else config.simulation.num_interactions
    )
    num_sequences = (
        args.sequences if args.sequences is not None else config.simulation.num_sequences
    )
    seed = args.seed if args.seed is not None else config.simulation.seed

    # Run simulation
    try:
        df = simulator.simulate(
            num_interactions=num_interactions,
            num_sequences=num_sequences,
            seed=seed,
        )
    except Exception as err:
        sys.stderr.write(f"Simulation Error: {err}\n")
        return 1

    # Output generation
    if args.output is None:
        # Default: write CSV to stdout with no extraneous logging
        df.to_csv(sys.stdout, index=False)
        return 0

    output_path = Path(args.output)
    ext = output_path.suffix.lower()

    try:
        if ext == ".csv":
            df.to_csv(output_path, index=False)
        elif ext == ".json":
            # Practical, stable record-oriented JSON
            df.to_json(output_path, orient="records", indent=2)
        elif ext == ".parquet":
            try:
                df.to_parquet(output_path, index=False)
            except (ImportError, ModuleNotFoundError) as err:
                sys.stderr.write(
                    "Error: Parquet export requires an optional engine ('pyarrow' or 'fastparquet'). "
                    "Please install 'pyarrow' using 'pip install pyarrow'.\n"
                )
                return 1
        else:
            sys.stderr.write(
                f"Error: Unsupported output format '{output_path.suffix}'. "
                "Supported formats: .csv, .json, .parquet\n"
            )
            return 1
    except (ImportError, ModuleNotFoundError) as err:
        sys.stderr.write(f"Export Error: {err}\n")
        return 1
    except Exception as err:
        sys.stderr.write(f"Error writing output to '{output_path}': {err}\n")
        return 1

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point for BehaviorSim.

    Args:
        argv: Optional list of command-line arguments. If None, sys.argv[1:] is used.

    Returns:
        Exit code (0 for success, 1 for runtime/config error, 2 for CLI parse error).
    """
    parser = create_parser()

    # If no subcommand or argument is provided, show help and exit with code 0 or 2
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        parser.print_help()
        return 0

    try:
        args = parser.parse_args(argv)
    except SystemExit as err:
        # argparse raises SystemExit on --help (0), --version (0), or invalid syntax (2)
        return err.code if isinstance(err.code, int) else 2

    if args.command == "validate":
        return handle_validate(args)
    elif args.command == "run":
        return handle_run(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
