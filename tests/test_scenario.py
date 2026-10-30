"""Tests for scenario loading, validation, and the spacecraft state model.

All three example scenarios must load cleanly. Invalid scenarios must fail
with clear, field-level Pydantic error messages.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from leeway.plan import (
    AnomalyKind,
    Assumptions,
    NoradSatellite,
    Scenario,
    load_scenario,
)
from leeway.spacecraft import SpacecraftMode, SpacecraftState
from leeway.validator import ValidationResult, validate_scenario

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCENARIOS = Path(__file__).parent.parent / "scenarios"


def _scenario(name: str) -> Path:
    return SCENARIOS / f"{name}.yaml"


# ---------------------------------------------------------------------------
# Example scenario loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_name",
    [
        "eo_imaging_anticipated",
        "eo_imaging_mixed",
        "eo_sparse_coverage",
    ],
)
def test_example_scenarios_load(scenario_name):
    sc = load_scenario(_scenario(scenario_name))
    assert isinstance(sc, Scenario)
    assert sc.name == scenario_name


def test_anticipated_has_two_anomalies():
    sc = load_scenario(_scenario("eo_imaging_anticipated"))
    assert len(sc.anomalies) == 2
    assert all(a.anticipated for a in sc.anomalies)


def test_mixed_has_unanticipated_anomaly():
    sc = load_scenario(_scenario("eo_imaging_mixed"))
    anticipated = [a for a in sc.anomalies if a.anticipated]
    unanticipated = [a for a in sc.anomalies if not a.anticipated]
    assert len(anticipated) == 2
    assert len(unanticipated) == 1
    assert unanticipated[0].kind == AnomalyKind.software_fault


def test_sparse_coverage_uses_purdue_station():
    sc = load_scenario(_scenario("eo_sparse_coverage"))
    assert "purdue" in sc.ground_stations_file.lower()


def test_sparse_coverage_has_unanticipated_anomaly():
    sc = load_scenario(_scenario("eo_sparse_coverage"))
    unanticipated = [a for a in sc.anomalies if not a.anticipated]
    assert len(unanticipated) == 1


def test_all_scenarios_have_norad_satellite():
    for name in ("eo_imaging_anticipated", "eo_imaging_mixed", "eo_sparse_coverage"):
        sc = load_scenario(_scenario(name))
        assert isinstance(sc.satellite, NoradSatellite)
        assert sc.satellite.norad_id == 40697


def test_all_scenarios_seed_is_set():
    for name in ("eo_imaging_anticipated", "eo_imaging_mixed", "eo_sparse_coverage"):
        sc = load_scenario(_scenario(name))
        assert isinstance(sc.seed, int)
        assert sc.seed >= 0


# ---------------------------------------------------------------------------
# Semantic validation (validate_scenario)
# ---------------------------------------------------------------------------


def test_validate_anticipated_passes():
    sc = load_scenario(_scenario("eo_imaging_anticipated"))
    result = validate_scenario(sc, _scenario("eo_imaging_anticipated"))
    assert result.ok, f"Validation errors:\n{result}"


def test_validate_mixed_passes():
    sc = load_scenario(_scenario("eo_imaging_mixed"))
    result = validate_scenario(sc, _scenario("eo_imaging_mixed"))
    assert result.ok, f"Validation errors:\n{result}"


def test_validate_sparse_passes():
    sc = load_scenario(_scenario("eo_sparse_coverage"))
    result = validate_scenario(sc, _scenario("eo_sparse_coverage"))
    assert result.ok, f"Validation errors:\n{result}"


def test_validate_missing_tle_file_is_error(tmp_path):
    raw = {
        "name": "test",
        "satellite": {"tle_file": "does_not_exist.tle"},
        "ground_stations_file": "ground/polar_network.yaml",
        "time_span": {"start": "2024-01-10T12:00:00Z", "hours": 24},
        "seed": 0,
        "objectives": [{"id": "obj1", "kind": "image_target", "duration_s": 60}],
        "assumptions": {
            "ground_turnaround_s": 3600,
            "min_uplink_contact_s": 30,
            "min_downlink_contact_s": 120,
            "plan_upload_s": 180,
        },
    }
    import yaml

    scenario_file = tmp_path / "test.yaml"
    scenario_file.write_text(yaml.dump(raw))
    sc = load_scenario(scenario_file)
    result = validate_scenario(sc, scenario_file)
    assert not result.ok
    assert any("tle_file" in e for e in result.errors)


def test_validate_missing_ground_station_file_is_error(tmp_path):
    raw = {
        "name": "test",
        "satellite": {"norad_id": 25544},
        "ground_stations_file": "nonexistent_stations.yaml",
        "time_span": {"start": "2024-01-10T12:00:00Z", "hours": 24},
        "seed": 0,
        "objectives": [{"id": "obj1", "kind": "image_target", "duration_s": 60}],
        "assumptions": {
            "ground_turnaround_s": 3600,
            "min_uplink_contact_s": 30,
            "min_downlink_contact_s": 120,
            "plan_upload_s": 180,
        },
    }
    import yaml

    scenario_file = tmp_path / "test.yaml"
    scenario_file.write_text(yaml.dump(raw))
    sc = load_scenario(scenario_file)
    result = validate_scenario(sc, scenario_file)
    assert not result.ok
    assert any("ground_stations_file" in e for e in result.errors)


def test_validate_warns_on_missing_cycle_count():
    sc = load_scenario(_scenario("eo_imaging_anticipated"))
    # Create a modified assumptions without cycle entries
    from copy import deepcopy

    sc2 = deepcopy(sc)
    sc2.assumptions.command_cycles_per_anomaly.clear()
    from leeway.validator import _check_anomaly_kinds_in_assumptions

    result = ValidationResult()
    _check_anomaly_kinds_in_assumptions(sc2.anomalies, sc2.assumptions, result)
    assert len(result.warnings) == len(sc2.anomalies)


# ---------------------------------------------------------------------------
# Schema validation — invalid scenarios must fail with clear messages
# ---------------------------------------------------------------------------


def _base_valid() -> dict:
    return {
        "name": "valid_base",
        "satellite": {"norad_id": 25544},
        "ground_stations_file": "ground/polar_network.yaml",
        "time_span": {"start": "2024-01-10T12:00:00Z", "hours": 24},
        "seed": 42,
        "objectives": [{"id": "obj1", "kind": "image_target", "duration_s": 60}],
        "assumptions": {
            "ground_turnaround_s": 3600,
            "min_uplink_contact_s": 30,
            "min_downlink_contact_s": 120,
            "plan_upload_s": 180,
        },
    }


def test_invalid_missing_name():
    raw = _base_valid()
    del raw["name"]
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "name" in str(exc_info.value)


def test_invalid_negative_norad_id():
    raw = _base_valid()
    raw["satellite"] = {"norad_id": -1}
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "norad_id" in str(exc_info.value)


def test_invalid_zero_hours():
    raw = _base_valid()
    raw["time_span"]["hours"] = 0
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "hours" in str(exc_info.value)


def test_invalid_hours_too_large():
    raw = _base_valid()
    raw["time_span"]["hours"] = 10000
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "hours" in str(exc_info.value)


def test_invalid_no_objectives():
    raw = _base_valid()
    raw["objectives"] = []
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "objectives" in str(exc_info.value)


def test_invalid_duplicate_objective_ids():
    raw = _base_valid()
    raw["objectives"] = [
        {"id": "same", "kind": "image_target", "duration_s": 60},
        {"id": "same", "kind": "downlink_data", "duration_s": 60},
    ]
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "unique" in str(exc_info.value)


def test_invalid_objective_id_with_spaces():
    raw = _base_valid()
    raw["objectives"] = [{"id": "has spaces", "kind": "image_target", "duration_s": 60}]
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "spaces" in str(exc_info.value)


def test_invalid_duplicate_anomaly_ids():
    raw = _base_valid()
    raw["anomalies"] = [
        {"id": "dup", "kind": "payload_reset", "timing": {"kind": "fixed", "offset_s": 100},
         "anticipated": True},
        {"id": "dup", "kind": "missed_window", "timing": {"kind": "fixed", "offset_s": 200},
         "anticipated": True},
    ]
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "unique" in str(exc_info.value)


def test_invalid_anomaly_timing_exceeds_span():
    raw = _base_valid()
    raw["anomalies"] = [
        {
            "id": "late",
            "kind": "payload_reset",
            "timing": {"kind": "fixed", "offset_s": 999999},
            "anticipated": True,
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "exceeds" in str(exc_info.value)


def test_invalid_random_range_latest_before_earliest():
    raw = _base_valid()
    raw["anomalies"] = [
        {
            "id": "bad_range",
            "kind": "payload_reset",
            "timing": {"kind": "random", "earliest_s": 500, "latest_s": 100},
            "anticipated": True,
        }
    ]
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "latest_s" in str(exc_info.value)


def test_invalid_unknown_anomaly_kind():
    raw = _base_valid()
    raw["anomalies"] = [
        {
            "id": "bad_kind",
            "kind": "exploded",  # not in AnomalyKind
            "timing": {"kind": "fixed", "offset_s": 100},
            "anticipated": True,
        }
    ]
    with pytest.raises(ValidationError):
        Scenario.model_validate(raw)


def test_invalid_zero_turnaround():
    raw = _base_valid()
    raw["assumptions"]["ground_turnaround_s"] = 0
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "ground_turnaround_s" in str(exc_info.value)


def test_invalid_cycle_count_zero():
    raw = _base_valid()
    raw["assumptions"]["command_cycles_per_anomaly"] = {"payload_reset": 0}
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "command_cycles_per_anomaly" in str(exc_info.value)


def test_invalid_cycle_unknown_kind():
    raw = _base_valid()
    raw["assumptions"]["command_cycles_per_anomaly"] = {"not_a_kind": 1}
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "command_cycles_per_anomaly" in str(exc_info.value)


def test_invalid_negative_seed():
    raw = _base_valid()
    raw["seed"] = -1
    with pytest.raises(ValidationError) as exc_info:
        Scenario.model_validate(raw)
    assert "seed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Assumptions.cycles_for helper
# ---------------------------------------------------------------------------


def test_cycles_for_explicit():
    a = Assumptions(
        ground_turnaround_s=3600,
        command_cycles_per_anomaly={"payload_reset": 3},
        min_uplink_contact_s=30,
        min_downlink_contact_s=120,
        plan_upload_s=180,
    )
    assert a.cycles_for(AnomalyKind.payload_reset) == 3


def test_cycles_for_default():
    a = Assumptions(
        ground_turnaround_s=3600,
        command_cycles_per_anomaly={},
        min_uplink_contact_s=30,
        min_downlink_contact_s=120,
        plan_upload_s=180,
    )
    assert a.cycles_for(AnomalyKind.software_fault) == 1


# ---------------------------------------------------------------------------
# SpacecraftState
# ---------------------------------------------------------------------------


def test_initial_state():
    sc = SpacecraftState()
    assert sc.mode == SpacecraftMode.nominal
    assert sc.current_objective_id is None
    assert sc.active_anomaly_id is None
    assert sc.event_log == []


def test_transition_logs_event():
    sc = SpacecraftState()
    sc.transition(100.0, SpacecraftMode.safe_hold, "test")
    assert sc.mode == SpacecraftMode.safe_hold
    assert len(sc.event_log) == 1
    assert sc.event_log[0].kind == "mode_change"


def test_transition_noop_on_same_mode():
    sc = SpacecraftState()
    sc.transition(0.0, SpacecraftMode.nominal)
    assert sc.event_log == []


def test_start_complete_objective():
    sc = SpacecraftState()
    sc.start_objective(10.0, "obj1")
    assert sc.mode == SpacecraftMode.executing
    assert sc.current_objective_id == "obj1"

    sc.complete_objective(130.0)
    assert sc.mode == SpacecraftMode.nominal
    assert sc.current_objective_id is None


def test_detect_resolve_anomaly():
    sc = SpacecraftState()
    sc.detect_anomaly(500.0, "anom1", "payload_reset")
    assert sc.mode == SpacecraftMode.safe_hold
    assert sc.active_anomaly_id == "anom1"

    sc.resolve_anomaly(1000.0, "anom1")
    assert sc.mode == SpacecraftMode.nominal
    assert sc.active_anomaly_id is None


def test_event_log_ordering():
    sc = SpacecraftState()
    sc.detect_anomaly(10.0, "a1", "payload_reset")
    sc.resolve_anomaly(20.0, "a1")
    sc.start_objective(30.0, "obj1")
    sc.complete_objective(60.0)
    times = [e.time_s for e in sc.event_log]
    assert times == sorted(times)


def test_snapshot_returns_dict():
    sc = SpacecraftState()
    snap = sc.snapshot()
    assert snap["mode"] == "nominal"
    assert snap["current_objective_id"] is None
    assert snap["event_count"] == 0


def test_log_custom_event():
    sc = SpacecraftState()
    sc.log(42.0, "custom_event", "something happened", extra="data")
    assert sc.event_log[-1].kind == "custom_event"
    assert sc.event_log[-1].data["extra"] == "data"
