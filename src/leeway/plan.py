"""Leeway plan module.

Scenario schema (Pydantic v2) and scenario file loading.
All operational assumptions live in the scenario file — nothing is hard-coded here.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Satellite source
# ---------------------------------------------------------------------------


class NoradSatellite(BaseModel):
    norad_id: Annotated[int, Field(gt=0)]


class TLEFileSatellite(BaseModel):
    tle_file: str  # path relative to the scenario file


# ---------------------------------------------------------------------------
# Time span
# ---------------------------------------------------------------------------


class TimeSpan(BaseModel):
    start: str = Field(description="ISO-8601 UTC datetime, e.g. 2024-01-10T12:00:00Z")
    hours: Annotated[float, Field(gt=0, le=720)]


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


class ObjectiveKind(StrEnum):
    image_target = "image_target"
    downlink_data = "downlink_data"
    uplink_command = "uplink_command"
    calibration = "calibration"
    other = "other"


class Objective(BaseModel):
    id: str = Field(min_length=1)
    kind: ObjectiveKind
    duration_s: Annotated[float, Field(gt=0)]
    description: str = ""

    @field_validator("id")
    @classmethod
    def id_no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("objective id must not contain spaces")
        return v


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------


class AnomalyKind(StrEnum):
    payload_reset = "payload_reset"
    missed_window = "missed_window"
    attitude_error = "attitude_error"
    comms_dropout = "comms_dropout"
    power_anomaly = "power_anomaly"
    thermal_exceedance = "thermal_exceedance"
    software_fault = "software_fault"
    other = "other"


class FixedTime(BaseModel):
    kind: Literal["fixed"]
    offset_s: Annotated[float, Field(ge=0)]


class RandomRange(BaseModel):
    kind: Literal["random"]
    earliest_s: Annotated[float, Field(ge=0)]
    latest_s: float

    @model_validator(mode="after")
    def latest_after_earliest(self) -> RandomRange:
        if self.latest_s <= self.earliest_s:
            raise ValueError("latest_s must be greater than earliest_s")
        return self


AnomalyTiming = Annotated[FixedTime | RandomRange, Field(discriminator="kind")]


class Anomaly(BaseModel):
    id: str = Field(min_length=1)
    kind: AnomalyKind
    timing: AnomalyTiming
    anticipated: bool = Field(
        description=(
            "True if a contingency plan covers this anomaly. "
            "False means the spacecraft must wait for ground intervention."
        )
    )
    description: str = ""

    @field_validator("id")
    @classmethod
    def id_no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("anomaly id must not contain spaces")
        return v


# ---------------------------------------------------------------------------
# Operational assumptions — all editable, all in the scenario file
# ---------------------------------------------------------------------------


class Assumptions(BaseModel):
    ground_turnaround_s: Annotated[
        float,
        Field(
            gt=0,
            description=(
                "Seconds from when operators see anomaly telemetry to when "
                "a corrective command is ready to uplink."
            ),
        ),
    ]
    command_cycles_per_anomaly: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Ground command round trips required per anomaly kind. "
            "Keys are AnomalyKind values. Missing keys default to 1."
        ),
    )
    min_uplink_contact_s: Annotated[
        float, Field(gt=0, description="Minimum contact (s) to uplink a command.")
    ]
    min_downlink_contact_s: Annotated[
        float, Field(gt=0, description="Minimum contact (s) to downlink telemetry.")
    ]
    plan_upload_s: Annotated[
        float, Field(gt=0, description="Time (s) to uplink a contingency plan.")
    ]

    @field_validator("command_cycles_per_anomaly")
    @classmethod
    def cycles_positive(cls, v: dict[str, int]) -> dict[str, int]:
        valid_kinds = {k.value for k in AnomalyKind}
        for kind, cycles in v.items():
            if kind not in valid_kinds:
                raise ValueError(
                    f"unknown anomaly kind {kind!r} in command_cycles_per_anomaly"
                )
            if cycles < 1:
                raise ValueError(
                    f"command_cycles_per_anomaly[{kind!r}] must be >= 1, got {cycles}"
                )
        return v

    def cycles_for(self, anomaly_kind: AnomalyKind) -> int:
        """Return the number of ground command cycles needed for *anomaly_kind*."""
        return self.command_cycles_per_anomaly.get(anomaly_kind.value, 1)


# ---------------------------------------------------------------------------
# Top-level scenario
# ---------------------------------------------------------------------------


class Scenario(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    satellite: NoradSatellite | TLEFileSatellite
    ground_stations_file: str = Field(
        description="Path to ground station YAML, relative to the scenario file."
    )
    time_span: TimeSpan
    seed: Annotated[int, Field(ge=0)]
    objectives: Annotated[list[Objective], Field(min_length=1)]
    anomalies: list[Anomaly] = Field(default_factory=list)
    assumptions: Assumptions

    @model_validator(mode="after")
    def unique_ids(self) -> Scenario:
        obj_ids = [o.id for o in self.objectives]
        if len(obj_ids) != len(set(obj_ids)):
            raise ValueError("objective ids must be unique")
        anom_ids = [a.id for a in self.anomalies]
        if len(anom_ids) != len(set(anom_ids)):
            raise ValueError("anomaly ids must be unique")
        return self

    @model_validator(mode="after")
    def anomaly_timing_within_span(self) -> Scenario:
        span_s = self.time_span.hours * 3600
        for a in self.anomalies:
            if isinstance(a.timing, FixedTime):
                if a.timing.offset_s > span_s:
                    raise ValueError(
                        f"anomaly {a.id!r}: fixed offset {a.timing.offset_s}s "
                        f"exceeds scenario span {span_s}s"
                    )
            elif isinstance(a.timing, RandomRange):
                if a.timing.latest_s > span_s:
                    raise ValueError(
                        f"anomaly {a.id!r}: random range latest_s {a.timing.latest_s}s "
                        f"exceeds scenario span {span_s}s"
                    )
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_scenario(path: Path) -> Scenario:
    """Load and validate a scenario YAML file.

    Raises ``pydantic.ValidationError`` with field-level messages on any
    schema or constraint violation.
    """
    raw = yaml.safe_load(path.read_text())
    return Scenario.model_validate(raw)
