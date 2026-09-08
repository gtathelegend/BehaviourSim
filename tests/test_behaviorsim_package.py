"""Sanity test for BehaviorSim package imports and version."""

import sys
from pathlib import Path

# Ensure src/ is in sys.path so behaviorsim can be imported directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_behaviorsim_package_imports() -> None:
    import behaviorsim

    assert behaviorsim.__version__ == "0.1.0"
