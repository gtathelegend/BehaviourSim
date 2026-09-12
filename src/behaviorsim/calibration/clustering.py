"""Profile clustering subsystem for BehaviorSim.

Clusters empirical entity/sequence interaction traces into distinct behavioral
personas based on summary behavioral representations (state occupancy, feature
moments, activity dynamics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from behaviorsim.calibration.fitter import (
    CalibrationData,
    fit_profile,
)
from behaviorsim.core.profile import Profile


@dataclass(frozen=True)
class ProfileClusteringResult:
    """Result of behavioral clustering across sequence/entity traces.

    Attributes:
        profiles: Mapping from profile name to calibrated Profile instance.
        profile_distribution: Mapping from profile name to mixture probability.
        assignments: pd.Series mapping sequence IDs to assigned profile name.
        cluster_features: Summary behavioral feature matrix used for clustering.
        metadata: Execution and provenance metadata.
    """

    profiles: Dict[str, Profile]
    profile_distribution: Dict[str, float]
    assignments: pd.Series
    cluster_features: pd.DataFrame
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert clustering result to a serializable dictionary."""
        return {
            "profiles": {k: v.name for k, v in self.profiles.items()},
            "profile_distribution": self.profile_distribution,
            "assignments": self.assignments.to_dict(),
            "metadata": self.metadata,
        }


def cluster_profiles(
    data: CalibrationData,
    n_profiles: int,
    *,
    seed: Optional[int] = None,
    name_prefix: str = "profile",
    distribution_types: Optional[Mapping[str, str]] = None,
    smoothing: float = 0.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ProfileClusteringResult:
    """Cluster empirical sequences/entities into discrete behavioral profiles.

    Aggregates interaction traces into sequence-level behavioral summary vectors
    (state occupancies, feature means, feature variances, activity statistics) and
    clusters entities using deterministic k-means. Each resulting cluster is fitted
    into a calibrated Profile and assigned a mixture probability matching cluster size.

    Important boundary:
        This performs empirical behavioral clustering of observed trajectories,
        not latent psychological, financial, or medical trait inference.

    Args:
        data: Validated CalibrationData container.
        n_profiles: Positive integer number of behavioral clusters to extract.
        seed: Optional deterministic random seed.
        name_prefix: Prefix string for generated profile names (e.g. 'profile_0').
        distribution_types: Optional feature-to-distribution mapping for profile emissions.
        smoothing: Additive smoothing parameter for transition fitting (default 0.0).
        metadata: Optional caller metadata dictionary.

    Returns:
        ProfileClusteringResult containing fitted Profiles, profile distribution,
        and sequence assignments.

    Raises:
        TypeError: If arguments are of invalid types.
        ValueError: If n_profiles < 1, n_profiles > number of sequences, or if
            sequence_column is missing for multi-profile clustering.
    """
    if not isinstance(data, CalibrationData):
        raise TypeError(f"data must be CalibrationData instance, got {type(data).__name__}.")

    if isinstance(n_profiles, bool) or not isinstance(n_profiles, int):
        raise TypeError(f"n_profiles must be an integer, got {type(n_profiles).__name__}.")

    if n_profiles < 1:
        raise ValueError(f"n_profiles must be at least 1, got {n_profiles}.")

    seq_col = data.sequence_column
    if seq_col is None or seq_col not in data.data.columns:
        if n_profiles > 1:
            raise ValueError(
                f"Cannot cluster into {n_profiles} profiles when sequence_column is None or missing. "
                "Entity clustering requires a sequence_column to partition entities."
            )
        seq_ids = ["single_entity"]
    else:
        seq_ids = list(data.data[seq_col].unique())

    num_sequences = len(seq_ids)
    if n_profiles > num_sequences:
        raise ValueError(
            f"Cannot form {n_profiles} clusters from {num_sequences} sequence entities "
            "(n_profiles > number of sequences)."
        )

    # 1. Deterministic sequence-level feature vector construction
    declared_states = sorted(data.states)
    numeric_feats = sorted(data.numeric_features or ())

    feature_rows: List[Dict[str, float]] = []
    for s_id in seq_ids:
        if seq_col is not None and seq_col in data.data.columns:
            seq_df = data.data[data.data[seq_col] == s_id]
        else:
            seq_df = data.data

        row_feat: Dict[str, float] = {}

        # State occupancy summaries
        seq_len = len(seq_df)
        for state_name in declared_states:
            occ = (seq_df[data.state_column] == state_name).mean() if seq_len > 0 else 0.0
            row_feat[f"state_occ_{state_name}"] = float(occ)

        # Numeric feature means and stds
        for nf in numeric_feats:
            if nf in seq_df.columns:
                vals = pd.to_numeric(seq_df[nf], errors="coerce").dropna().values
                if len(vals) > 0:
                    row_feat[f"mean_{nf}"] = float(np.mean(vals))
                    row_feat[f"std_{nf}"] = float(np.std(vals, ddof=0))
                else:
                    row_feat[f"mean_{nf}"] = 0.0
                    row_feat[f"std_{nf}"] = 0.0
            else:
                row_feat[f"mean_{nf}"] = 0.0
                row_feat[f"std_{nf}"] = 0.0

        # Activity statistics
        row_feat["activity_length"] = float(seq_len)
        feature_rows.append(row_feat)

    summary_df = pd.DataFrame(feature_rows, index=seq_ids)
    # Ensure stable column ordering
    summary_df = summary_df.reindex(sorted(summary_df.columns), axis=1)

    # 2. Clustering algorithm
    if n_profiles == 1:
        raw_labels = np.zeros(num_sequences, dtype=int)
    else:
        X = summary_df.values.astype(float)
        # Handle constant features safely during standardization
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0, ddof=0)
        stds_safe = np.where(stds > 1e-8, stds, 1.0)
        X_norm = (X - means) / stds_safe

        kmeans = KMeans(
            n_clusters=n_profiles,
            random_state=seed,
            n_init=10,
        )
        kmeans.fit(X_norm)
        raw_labels = kmeans.labels_

    # 3. Stable, deterministic cluster ordering
    # Group sequences by raw cluster label
    cluster_groups: Dict[int, List[Any]] = {}
    for seq_id, lbl in zip(seq_ids, raw_labels):
        cluster_groups.setdefault(lbl, []).append(seq_id)

    # Sort clusters deterministically:
    # 1st key: cluster size descending (largest cluster first)
    # 2nd key: string representation of first sequence ID in cluster ascending
    sorted_raw_labels = sorted(
        cluster_groups.keys(),
        key=lambda k: (-len(cluster_groups[k]), str(cluster_groups[k][0])),
    )

    # Remap to canonical cluster indices 0, 1, ..., n_profiles - 1
    raw_to_canonical = {raw_lbl: idx for idx, raw_lbl in enumerate(sorted_raw_labels)}
    canonical_assignments: Dict[Any, str] = {}
    cluster_proportions: Dict[str, float] = {}
    fitted_profiles: Dict[str, Profile] = {}

    for raw_lbl in sorted_raw_labels:
        canonical_idx = raw_to_canonical[raw_lbl]
        c_name = f"{name_prefix}_{canonical_idx}"
        c_seqs = cluster_groups[raw_lbl]
        c_prop = float(len(c_seqs) / num_sequences)
        cluster_proportions[c_name] = c_prop

        for s_id in c_seqs:
            canonical_assignments[s_id] = c_name

        # Sliced CalibrationData for this cluster
        if seq_col is not None and seq_col in data.data.columns:
            cluster_data = data.data[data.data[seq_col].isin(c_seqs)].copy()
        else:
            cluster_data = data.data.copy()

        cluster_calib = CalibrationData(
            data=cluster_data,
            states=data.states,
            state_column=data.state_column,
            feature_columns=data.feature_columns,
            sequence_column=data.sequence_column,
            profile_column=data.profile_column,
            numeric_features=data.numeric_features,
            categorical_features=data.categorical_features,
        )

        cluster_profile = fit_profile(
            data=cluster_calib,
            name=c_name,
            distribution_types=distribution_types,
            transition_smoothing=smoothing,
            metadata={
                "cluster_index": canonical_idx,
                "cluster_size": len(c_seqs),
                "cluster_proportion": c_prop,
                "calibrated_from": "cluster_profiles",
            },
        )
        object.__setattr__(cluster_profile, "probability", c_prop)
        fitted_profiles[c_name] = cluster_profile

    # Normalize cluster proportions so sum is exactly 1.0
    total_prop = sum(cluster_proportions.values())
    for k in cluster_proportions:
        cluster_proportions[k] = float(cluster_proportions[k] / total_prop)

    assignments_series = pd.Series(
        [canonical_assignments[s_id] for s_id in seq_ids],
        index=seq_ids,
        name="profile",
    )

    combined_metadata: Dict[str, Any] = {
        "n_profiles": n_profiles,
        "seed": seed,
        "num_sequences": num_sequences,
        "feature_names": list(summary_df.columns),
        "cluster_sizes": {
            f"{name_prefix}_{raw_to_canonical[k]}": len(v) for k, v in cluster_groups.items()
        },
    }
    if metadata is not None:
        combined_metadata["user_metadata"] = dict(metadata)

    return ProfileClusteringResult(
        profiles=fitted_profiles,
        profile_distribution=cluster_proportions,
        assignments=assignments_series,
        cluster_features=summary_df,
        metadata=combined_metadata,
    )
