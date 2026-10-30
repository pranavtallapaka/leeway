"""Leeway spacecraft module.

Minimal spacecraft state model for Phase A simulation.
No physics beyond what the ops simulator needs: mode transitions and an event log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SpacecraftMode(StrEnum):
    nominal = "nominal"
    safe_hold = "safe_hold"
    executing = "executing"


@dataclass
class EventLogEntry:
    time_s: float  # seconds from scenario start
    kind: str  # e.g. "mode_change", "anomaly_detected", "objective_started"
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"t+{self.time_s:>8.1f}s  [{self.kind:<22}]  {self.detail}"


@dataclass
class SpacecraftState:
    """Minimal spacecraft state for Phase A.

    Tracks mode, current objective, and a log of everything that happened.
    The ops simulator reads and writes this; nothing else should mutate it directly.
    """

    mode: SpacecraftMode = SpacecraftMode.nominal
    current_objective_id: str | None = None
    active_anomaly_id: str | None = None
    event_log: list[EventLogEntry] = field(default_factory=list)

    def log(self, time_s: float, kind: str, detail: str, **data: Any) -> None:
        """Append an entry to the event log."""
        self.event_log.append(EventLogEntry(time_s=time_s, kind=kind, detail=detail, data=data))

    def transition(self, time_s: float, new_mode: SpacecraftMode, reason: str = "") -> None:
        """Change mode and record the transition."""
        if new_mode == self.mode:
            return
        old = self.mode
        self.mode = new_mode
        detail = f"{old.value} → {new_mode.value}"
        if reason:
            detail += f" ({reason})"
        self.log(time_s, "mode_change", detail, from_mode=old.value, to_mode=new_mode.value)

    def start_objective(self, time_s: float, objective_id: str) -> None:
        self.current_objective_id = objective_id
        self.transition(time_s, SpacecraftMode.executing, f"objective={objective_id}")
        self.log(time_s, "objective_started", f"started {objective_id!r}")

    def complete_objective(self, time_s: float) -> None:
        obj = self.current_objective_id
        self.current_objective_id = None
        self.transition(time_s, SpacecraftMode.nominal, f"completed {obj!r}")
        self.log(time_s, "objective_completed", f"completed {obj!r}")

    def detect_anomaly(self, time_s: float, anomaly_id: str, anomaly_kind: str) -> None:
        self.active_anomaly_id = anomaly_id
        self.transition(time_s, SpacecraftMode.safe_hold, f"anomaly={anomaly_id}")
        self.log(
            time_s,
            "anomaly_detected",
            f"{anomaly_kind} anomaly {anomaly_id!r} detected",
            anomaly_id=anomaly_id,
            anomaly_kind=anomaly_kind,
        )

    def resolve_anomaly(self, time_s: float, anomaly_id: str) -> None:
        self.active_anomaly_id = None
        self.transition(time_s, SpacecraftMode.nominal, f"anomaly {anomaly_id!r} resolved")
        self.log(time_s, "anomaly_resolved", f"anomaly {anomaly_id!r} resolved")

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the current state."""
        return {
            "mode": self.mode.value,
            "current_objective_id": self.current_objective_id,
            "active_anomaly_id": self.active_anomaly_id,
            "event_count": len(self.event_log),
        }
