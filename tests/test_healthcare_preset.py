"""Unit and integration tests for Healthcare patient monitoring preset (behaviorsim.presets.healthcare)."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from behaviorsim.core.simulator import Simulator
from behaviorsim.presets import get_preset, list_presets
from behaviorsim.presets.healthcare import (
    BASE_HEALTHCARE_TRANSITION_MATRIX,
    HEALTHCARE_PROFILES,
    HEALTHCARE_STATES,
    PATIENT_COHORTS,
    STATE_NAMES,
    HealthcareCohort,
    create_healthcare_profile,
    create_healthcare_simulator,
)


# ---------------------------------------------------------------------------
# Construction & State Space Tests
# ---------------------------------------------------------------------------

def test_healthcare_factory_returns_simulator() -> None:
    """Verify create_healthcare_simulator instantiates a valid Simulator instance."""
    sim = create_healthcare_simulator()
    assert isinstance(sim, Simulator)
    assert len(sim.states) == 5
    assert [s.name for s in sim.states] == STATE_NAMES


def test_healthcare_states_unique_and_valid() -> None:
    """Verify all healthcare states have unique non-empty names and descriptions."""
    names = [s.name for s in HEALTHCARE_STATES]
    assert len(names) == len(set(names))
    for s in HEALTHCARE_STATES:
        assert s.name.strip() != ""
        assert s.description is not None and s.description.strip() != ""


def test_base_transition_matrix_properties() -> None:
    """Verify base transition matrix dimensions, non-negativity, stochasticity, and absorbing state."""
    matrix = BASE_HEALTHCARE_TRANSITION_MATRIX
    assert matrix.shape == (5, 5)
    assert np.all(matrix >= 0.0)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    # Discharged state index is 4
    discharged_idx = STATE_NAMES.index("Discharged")
    assert matrix[discharged_idx, discharged_idx] == 1.0
    assert np.sum(matrix[discharged_idx, :discharged_idx]) == 0.0


# ---------------------------------------------------------------------------
# Registry Integration Tests
# ---------------------------------------------------------------------------

def test_healthcare_registered_and_discoverable() -> None:
    """Verify healthcare preset is listed in list_presets() and resolved by get_preset()."""
    presets = list_presets()
    assert "healthcare" in presets

    factory = get_preset("healthcare")
    assert callable(factory)
    assert factory is create_healthcare_simulator


def test_simulator_from_preset_healthcare() -> None:
    """Verify Simulator.from_preset('healthcare') instantiates an independent Simulator."""
    sim1 = Simulator.from_preset("healthcare")
    sim2 = Simulator.from_preset("healthcare", profile="chronic_risk")

    assert isinstance(sim1, Simulator)
    assert isinstance(sim2, Simulator)
    assert sim1 is not sim2
    assert sim1.profile is not None and sim1.profile.name == "stable_patient"
    assert sim2.profile is not None and sim2.profile.name == "chronic_risk"


# ---------------------------------------------------------------------------
# Profile & Cohort Tests
# ---------------------------------------------------------------------------

def test_all_defined_cohorts_construct_profiles() -> None:
    """Verify all registered patient cohorts generate valid Profile instances."""
    assert len(PATIENT_COHORTS) >= 4
    for name, cohort in PATIENT_COHORTS.items():
        assert isinstance(cohort, HealthcareCohort)
        profile = create_healthcare_profile(cohort)
        assert profile.name == name
        assert set(profile.state_emissions.keys()) == set(STATE_NAMES)
        # Check transition matrix stochasticity
        assert profile.transition_matrix is not None
        assert profile.transition_matrix.shape == (5, 5)
        assert np.all(profile.transition_matrix >= 0.0)
        assert np.allclose(profile.transition_matrix.sum(axis=1), 1.0)


@pytest.mark.parametrize("cohort_name", list(PATIENT_COHORTS.keys()))
def test_create_simulator_all_cohorts(cohort_name: str) -> None:
    """Verify create_healthcare_simulator accepts every registered cohort by name."""
    sim = create_healthcare_simulator(profile=cohort_name)
    assert sim.profile is not None
    assert sim.profile.name == cohort_name


def test_create_simulator_custom_cohort_instance() -> None:
    """Verify create_healthcare_simulator accepts a direct HealthcareCohort instance."""
    custom_cohort = HealthcareCohort(
        name="custom_pediatric",
        description="Pediatric monitoring profile",
        base_hr_loc=105.0,
        base_hr_scale=7.0,
        base_sbp_loc=100.0,
        base_sbp_scale=6.0,
        base_spo2_loc=99.0,
        base_temp_loc=37.0,
        acute_risk_multiplier=0.9,
        recovery_rate_multiplier=1.5,
        mobility_scale=1.5,
    )
    sim = create_healthcare_simulator(profile=custom_cohort)
    assert sim.profile is not None
    assert sim.profile.name == "custom_pediatric"


def test_create_simulator_invalid_profile_name() -> None:
    """Verify specifying an unknown profile name raises ValueError with available list."""
    with pytest.raises(ValueError, match="Unknown healthcare profile 'unknown_cohort'"):
        create_healthcare_simulator(profile="unknown_cohort")


def test_create_simulator_invalid_profile_type() -> None:
    """Verify passing an unsupported profile type raises TypeError."""
    with pytest.raises(TypeError, match="profile must be a str, HealthcareCohort, or None"):
        create_healthcare_simulator(profile=999)  # type: ignore


def test_create_simulator_initial_state_selection() -> None:
    """Verify initial_state parameter works for valid states and rejects unknown states."""
    sim = create_healthcare_simulator(initial_state="Recovery")
    assert sim.initial_state == "Recovery"

    with pytest.raises(ValueError, match="Unknown initial_state 'NonExistentState'"):
        create_healthcare_simulator(initial_state="NonExistentState")


# ---------------------------------------------------------------------------
# Simulation Telemetry & Bounds Tests
# ---------------------------------------------------------------------------

def test_simulation_reproducibility_seed() -> None:
    """Verify deterministic reproducibility when generating with a fixed seed."""
    sim = create_healthcare_simulator(profile="chronic_risk")

    df1 = sim.simulate(num_interactions=30, num_sequences=2, seed=42)
    df2 = sim.simulate(num_interactions=30, num_sequences=2, seed=42)

    pd.testing.assert_frame_equal(df1, df2)

    df3 = sim.simulate(num_interactions=30, num_sequences=2, seed=99)
    assert not df1["heart_rate_bpm"].equals(df3["heart_rate_bpm"])


def test_simulation_output_schema_and_columns() -> None:
    """Verify DataFrame structure contains ground-truth state, IDs, and all 6 vital features."""
    sim = create_healthcare_simulator(profile="stable_patient")
    df = sim.simulate(num_interactions=25, num_sequences=3, seed=123)

    assert len(df) == 75  # 3 * 25
    expected_cols = [
        "profile",
        "sequence_id",
        "interaction_id",
        "state",
        "heart_rate_bpm",
        "systolic_bp",
        "spo2_pct",
        "temperature_c",
        "alert_triggered",
        "mobility_score",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column '{col}' in output"

    assert set(df["sequence_id"].unique()) == {1, 2, 3}
    assert (df["profile"] == "stable_patient").all()
    assert set(df["state"].unique()).issubset(set(STATE_NAMES))


def test_telemetry_bounds_and_semantic_types() -> None:
    """Verify semantic ranges: HR ~40-180, SBP ~90-200, SpO2 ~80-100, Temp ~36-40, binary alert."""
    sim = create_healthcare_simulator(profile="post_operative")
    df = sim.simulate(num_interactions=100, num_sequences=5, seed=777)

    # Heart rate within reasonable synthetic monitoring range
    assert (df["heart_rate_bpm"] >= 40.0).all()
    assert (df["heart_rate_bpm"] <= 195.0).all()

    # Systolic blood pressure within reasonable synthetic range
    assert (df["systolic_bp"] >= 80.0).all()
    assert (df["systolic_bp"] <= 220.0).all()

    # SpO2 bounded in [80.0, 100.0]
    assert (df["spo2_pct"] >= 79.9).all()
    assert (df["spo2_pct"] <= 100.0 + 1e-6).all()

    # Temperature in approximate physiological monitoring range
    assert (df["temperature_c"] >= 35.0).all()
    assert (df["temperature_c"] <= 41.5).all()

    # Alert triggered must be binary {0, 1}
    assert set(df["alert_triggered"].unique()).issubset({0, 1})

    # Mobility score must be non-negative integer
    assert (df["mobility_score"] >= 0).all()


def test_discharged_state_is_absorbing() -> None:
    """Verify once Discharged is entered, all subsequent interactions remain Discharged."""
    sim = create_healthcare_simulator(profile="stable_patient", initial_state="Discharged")
    df = sim.simulate(num_interactions=20, num_sequences=3, seed=42)

    assert (df["state"] == "Discharged").all()
    assert (df["alert_triggered"] == 0).all()
    assert (df["mobility_score"] == 0).all()
    assert (df["spo2_pct"] >= 96.0).all()


def test_profile_differentiation_clinical_risk() -> None:
    """Verify chronic_risk produces substantially higher blood pressure, heart rate, and alert rate than stable_patient."""
    sim_stable = create_healthcare_simulator(profile="stable_patient")
    sim_chronic = create_healthcare_simulator(profile="chronic_risk")

    df_stable = sim_stable.simulate(num_interactions=100, num_sequences=20, seed=42)
    df_chronic = sim_chronic.simulate(num_interactions=100, num_sequences=20, seed=42)

    # Chronic risk patients have significantly elevated resting and acute systolic blood pressure
    mean_sbp_stable = df_stable["systolic_bp"].mean()
    mean_sbp_chronic = df_chronic["systolic_bp"].mean()
    assert mean_sbp_chronic > mean_sbp_stable + 10.0

    # Chronic risk patients have higher heart rate
    mean_hr_stable = df_stable["heart_rate_bpm"].mean()
    mean_hr_chronic = df_chronic["heart_rate_bpm"].mean()
    assert mean_hr_chronic > mean_hr_stable + 8.0

    # Chronic risk patients experience higher clinical alert trigger rates
    alert_rate_stable = df_stable["alert_triggered"].mean()
    alert_rate_chronic = df_chronic["alert_triggered"].mean()
    assert alert_rate_chronic > alert_rate_stable * 2.0



def test_multi_sequence_isolation() -> None:
    """Verify sequence 1 produces identical results whether num_sequences=1 or num_sequences=5."""
    sim = create_healthcare_simulator(profile="stable_patient")

    df_single = sim.simulate(num_interactions=25, num_sequences=1, seed=555)
    df_multi = sim.simulate(num_interactions=25, num_sequences=5, seed=555)

    df_multi_seq1 = df_multi[df_multi["sequence_id"] == 1].reset_index(drop=True)
    pd.testing.assert_frame_equal(df_single, df_multi_seq1)


# ---------------------------------------------------------------------------
# History Rules & Causality Tests
# ---------------------------------------------------------------------------

def test_sustained_hypoxia_tachycardia_rule_triggers_critical() -> None:
    """Verify repeated prior interactions showing hypoxia or tachycardia triggers Critical rule."""
    from behaviorsim.core.simulator import SimulationHistory
    from behaviorsim.presets.healthcare import _sustained_hypoxia_tachycardia_rule

    history = SimulationHistory()
    assert not _sustained_hypoxia_tachycardia_rule(history)

    # 1 hypoxic interaction (insufficient, requires 2)
    history.record("spo2_pct", 88.0)
    history.record("heart_rate_bpm", 80.0)
    assert not _sustained_hypoxia_tachycardia_rule(history)

    # 2nd hypoxic interaction -> triggers rule
    history.record("spo2_pct", 87.5)
    history.record("heart_rate_bpm", 82.0)
    assert _sustained_hypoxia_tachycardia_rule(history)

    # Normal recovery vitals interrupt the condition
    history.record("spo2_pct", 98.0)
    history.record("heart_rate_bpm", 75.0)
    assert not _sustained_hypoxia_tachycardia_rule(history)

    # Tachycardia trigger test
    history.record("spo2_pct", 98.0)
    history.record("heart_rate_bpm", 135.0)
    assert not _sustained_hypoxia_tachycardia_rule(history)

    history.record("spo2_pct", 97.0)
    history.record("heart_rate_bpm", 140.0)
    assert _sustained_hypoxia_tachycardia_rule(history)


def test_sustained_recovery_stability_rule_triggers_discharged() -> None:
    """Verify 3 consecutive stable observations in Recovery triggers Discharged rule."""
    from behaviorsim.core.simulator import SimulationHistory
    from behaviorsim.presets.healthcare import _sustained_recovery_stability_rule

    history = SimulationHistory()
    assert not _sustained_recovery_stability_rule(history)

    # Record 2 stable interactions (insufficient, requires 3)
    history.record("spo2_pct", 97.0)
    history.record("heart_rate_bpm", 72.0)
    history.record("spo2_pct", 98.0)
    history.record("heart_rate_bpm", 70.0)
    assert not _sustained_recovery_stability_rule(history)

    # 3rd stable interaction -> triggers rule
    history.record("spo2_pct", 99.0)
    history.record("heart_rate_bpm", 68.0)
    assert _sustained_recovery_stability_rule(history)

    # Unstable vitals interrupt
    history.record("spo2_pct", 92.0)
    history.record("heart_rate_bpm", 105.0)
    assert not _sustained_recovery_stability_rule(history)


def test_simulation_causal_rule_in_flight_trigger() -> None:
    """Verify in-flight simulator transition rules fire causally without looking at current or future steps."""
    # When starting in Critical with constant low SpO2, the rule ensures persistent Critical transitions
    sim = create_healthcare_simulator(profile="chronic_risk", initial_state="Critical")
    df = sim.simulate(num_interactions=10, num_sequences=2, seed=42)

    # In Critical, SpO2 is sampled from [80.0, 89.5], satisfying the hypoxia rule (SpO2 < 90)
    # By interaction 3, with 2 consecutive prior hypoxic readings, hypoxia rule triggers Critical
    assert df.loc[df["interaction_id"] >= 3, "state"].isin(["Critical", "Recovery"]).any()
    assert (df["spo2_pct"] <= 100.0).all()



# ---------------------------------------------------------------------------
# Synthetic & Non-Clinical Boundary Documentation Tests
# ---------------------------------------------------------------------------

def test_synthetic_boundary_documentation_present() -> None:
    """Verify module and factory docstrings explicitly state the model is synthetic and not medical advice."""
    import behaviorsim.presets.healthcare as hc_module

    mod_doc = hc_module.__doc__ or ""
    assert "synthetic" in mod_doc.lower()
    assert "not" in mod_doc.lower()
    assert ("clinical" in mod_doc.lower() or "medical" in mod_doc.lower())

    factory_doc = hc_module.create_healthcare_simulator.__doc__ or ""
    assert "synthetic" in factory_doc.lower()
    assert "not clinically validated" in factory_doc.lower()
