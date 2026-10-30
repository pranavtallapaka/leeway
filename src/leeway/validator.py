"""Leeway validator module.

Cross-field and cross-file validation for scenarios that Pydantic alone cannot check:
paths exist, satellite sources resolve, timing is self-consistent across modules.

Pydantic handles schema validation; this module handles semantic validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from leeway.plan import (
    Anomaly,
    Assumptions,
    NoradSatellite,
    Scenario,
    TLEFileSatellite,
)


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        lines: list[str] = []
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING {w}")
        return "\n".join(lines) if lines else "  OK"


def validate_scenario(scenario: Scenario, scenario_path: Path) -> ValidationResult:
    """Run semantic validation on a fully-parsed Scenario.

    Checks things Pydantic cannot: file existence, path resolution, and
    cross-field consistency between assumptions and anomaly definitions.

    Returns a ``ValidationResult``; callers decide whether to abort on errors.
    """
    result = ValidationResult()
    scenario_dir = scenario_path.parent

    _check_satellite_source(scenario, scenario_dir, result)
    _check_ground_stations_file(scenario, scenario_dir, result)
    _check_anomaly_kinds_in_assumptions(scenario.anomalies, scenario.assumptions, result)
    _check_objective_durations_vs_contact(scenario, result)

    return result


def _check_satellite_source(
    scenario: Scenario, scenario_dir: Path, result: ValidationResult
) -> None:
    src = scenario.satellite
    if isinstance(src, TLEFileSatellite):
        tle_path = scenario_dir / src.tle_file
        if not tle_path.exists():
            result.errors.append(
                f"satellite.tle_file {str(tle_path)!r} does not exist"
            )
    elif isinstance(src, NoradSatellite):
        if src.norad_id > 999_999:
            result.warnings.append(
                f"satellite.norad_id {src.norad_id} looks unusually large; "
                "verify it is a valid NORAD catalog number"
            )


def _check_ground_stations_file(
    scenario: Scenario, scenario_dir: Path, result: ValidationResult
) -> None:
    gs_path = scenario_dir / scenario.ground_stations_file
    if not gs_path.exists():
        result.errors.append(
            f"ground_stations_file {str(gs_path)!r} does not exist"
        )


def _check_anomaly_kinds_in_assumptions(
    anomalies: list[Anomaly],
    assumptions: Assumptions,
    result: ValidationResult,
) -> None:
    """Warn if an anomaly kind has no explicit cycle count — defaulting to 1."""
    for anomaly in anomalies:
        if anomaly.kind.value not in assumptions.command_cycles_per_anomaly:
            result.warnings.append(
                f"anomaly {anomaly.id!r} (kind={anomaly.kind.value!r}) has no entry in "
                f"command_cycles_per_anomaly — defaulting to 1 cycle"
            )


def _check_objective_durations_vs_contact(
    scenario: Scenario, result: ValidationResult
) -> None:
    """Warn if any objective's duration exceeds the minimum required contact time."""
    min_contact = scenario.assumptions.min_downlink_contact_s
    for obj in scenario.objectives:
        if obj.duration_s > min_contact * 10:
            result.warnings.append(
                f"objective {obj.id!r} duration {obj.duration_s}s is very long "
                f"relative to min_downlink_contact_s {min_contact}s — "
                "confirm this is intentional"
            )
