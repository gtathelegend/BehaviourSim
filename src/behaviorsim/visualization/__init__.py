"""Visualization and analysis layer for BehaviorSim.

Provides publication-ready plotting utilities for:
- State occupancy and discrete trajectory dynamics
- Transition matrices and heatmaps
- Empirical vs synthetic distribution comparisons (numeric and categorical)
- Continuous and causal feature trajectories
- Calibration diagnostic dashboards and validation report visualizations
- Figure export utilities
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from behaviorsim.visualization.plot_states import (
    plot_state_occupancy,
    plot_transition_matrix,
)
from behaviorsim.visualization.plot_trajectories import (
    plot_state_trajectory,
    plot_feature_trajectory,
)
from behaviorsim.visualization.plot_distributions import (
    plot_feature_distribution,
    plot_feature_comparison,
    plot_categorical_distribution,
)
from behaviorsim.visualization.plot_calibration import (
    plot_occupancy_comparison,
    plot_transition_comparison,
    plot_numeric_comparison,
    plot_categorical_comparison,
    plot_calibration_summary,
)


def save_figure(
    fig: Figure,
    path: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = "tight",
    close: bool = False,
    **kwargs: Any,
) -> Path:
    """Save a matplotlib Figure to disk.

    Supports PNG, PDF, SVG, and other matplotlib-supported formats.
    Ensures parent directories are created automatically if they do not exist.
    Does not close caller-owned figures unless explicitly requested via close=True.

    Parameters
    ----------
    fig : Figure
        Matplotlib Figure to save.
    path : Union[str, Path]
        Target file path with extension (e.g. '.png', '.pdf', '.svg').
    dpi : int, default 300
        Resolution in dots per inch for raster formats.
    bbox_inches : str, default "tight"
        Bounding box specification to avoid clipped labels.
    close : bool, default False
        Whether to close the figure after saving.
    **kwargs : Any
        Additional keyword arguments forwarded to fig.savefig().

    Returns
    -------
    Path
        Path to the saved figure file.
    """
    if not isinstance(fig, Figure):
        raise TypeError(f"Expected matplotlib Figure, got {type(fig).__name__}.")

    target_path = Path(path)
    if not target_path.suffix:
        raise ValueError(
            f"Target path '{path}' must include a file extension (e.g. '.png', '.pdf', '.svg')."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(target_path), dpi=dpi, bbox_inches=bbox_inches, **kwargs)

    if close:
        plt.close(fig)

    return target_path


__all__ = [
    "plot_state_occupancy",
    "plot_state_trajectory",
    "plot_transition_matrix",
    "plot_feature_distribution",
    "plot_feature_comparison",
    "plot_categorical_distribution",
    "plot_feature_trajectory",
    "plot_occupancy_comparison",
    "plot_transition_comparison",
    "plot_numeric_comparison",
    "plot_categorical_comparison",
    "plot_calibration_summary",
    "save_figure",
]
