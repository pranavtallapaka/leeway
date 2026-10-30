"""Leeway ops simulator.

Runs a scenario in two modes on identical contact windows, objectives, and anomalies.

Conventional mode
-----------------
Every decision waits for ground contact.  For each anomaly:
  1. Anomaly occurs at time T.
  2. First contact at or after T → telemetry downlinked (requires min_downlink_contact_s).
  3. Ground turnaround: operators take ground_turnaround_s to form a response.
  4. Next contact after turnaround completes → command uplinked (requires min_uplink_contact_s).
  5. Steps 3–4 repeat for (command_cycles - 1) more cycles.
  6. Objective resumes only after all cycles complete.

Contingency mode
----------------
- Anticipated anomalies: resolved on board immediately (no contact needed, zero delay).
- Unanticipated anomalies: spacecraft safe-holds, sends escalation packet at next contact.
  One command cycle suffices (the packet is designed to be answerable in one reply):
    • Next contact after anomaly → escalation downlinked.
    • Turnaround → one command uplinked at next contact.
  Plans for upcoming objectives are uplinked in batches at each contact
  (costs plan_upload_s per contact; does not block objective execution).

Monte Carlo
-----------
Randomised anomaly times are drawn from each anomaly's RandomRange using the run seed.
Fixed-time anomalies are unaffected.  The LLM is never called; the plan list is generated
once, then the same set is simulated deterministically across N seeds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from leeway.orbit import ContactWindow
from leeway.plan import Anomaly, AnomalyKind, FixedTime, RandomRange, Scenario
from leeway.spacecraft import SpacecraftState

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SimAnomaly:
    """An anomaly with a resolved (concrete) occurrence time."""

    anomaly: Anomaly
    occurs_at_s: float  # seconds from scenario start


@dataclass
class RunMetrics:
    """Metrics for one simulation run (one mode, one seed)."""

    mode: str  # "conventional" | "contingency"
    seed: int
    round_trips: int
    time_to_all_objectives_s: float | None  # None if not all completed in span
    time_blocked_s: float
    objectives_completed: int
    objectives_total: int
    event_log: list[str] = field(default_factory=list)

    @property
    def time_to_all_objectives_h(self) -> float | None:
        if self.time_to_all_objectives_s is None:
            return None
        return self.time_to_all_objectives_s / 3600.0

    @property
    def time_blocked_h(self) -> float:
        return self.time_blocked_s / 3600.0

    @property
    def all_objectives_completed(self) -> bool:
        return self.objectives_completed == self.objectives_total


# ---------------------------------------------------------------------------
# Helpers: resolve anomaly times from the RNG
# ---------------------------------------------------------------------------


def resolve_anomaly_times(scenario: Scenario, seed: int) -> list[SimAnomaly]:
    """Resolve every anomaly to a concrete occurrence time (seconds from start).

    FixedTime anomalies are always deterministic.
    RandomRange anomalies are drawn with the given seed.
    """
    rng = random.Random(seed)
    resolved: list[SimAnomaly] = []
    for anomaly in scenario.anomalies:
        if isinstance(anomaly.timing, FixedTime):
            t = anomaly.timing.offset_s
        else:
            assert isinstance(anomaly.timing, RandomRange)
            t = rng.uniform(anomaly.timing.earliest_s, anomaly.timing.latest_s)
        resolved.append(SimAnomaly(anomaly=anomaly, occurs_at_s=t))
    resolved.sort(key=lambda x: x.occurs_at_s)
    return resolved


# ---------------------------------------------------------------------------
# Contact window helpers
# ---------------------------------------------------------------------------


def _windows_as_offsets(
    windows: list[ContactWindow], t_start_s: float = 0.0
) -> list[tuple[float, float]]:
    """Convert ContactWindow list to (start_s, end_s) offsets from scenario start."""
    result = []
    epoch = windows[0].start if windows else None
    for w in windows:
        if epoch is None:
            break
        start_off = (w.start - epoch).total_seconds() + t_start_s
        end_off = (w.end - epoch).total_seconds() + t_start_s
        result.append((start_off, end_off))
    return result


def _next_contact_start(
    contacts: list[tuple[float, float]],
    after_s: float,
    min_duration_s: float,
) -> float | None:
    """Return the start time of the first contact that begins at or after *after_s*
    and lasts at least *min_duration_s*.  Returns None if no such contact exists."""
    for start, end in contacts:
        if start >= after_s and (end - start) >= min_duration_s:
            return start
    return None


def _next_contact_end(
    contacts: list[tuple[float, float]],
    after_s: float,
    min_duration_s: float,
) -> float | None:
    """Return the end time of the first usable contact at or after *after_s*."""
    for start, end in contacts:
        if start >= after_s and (end - start) >= min_duration_s:
            return end
    return None


# ---------------------------------------------------------------------------
# Core simulation engine
# ---------------------------------------------------------------------------


def _simulate(
    scenario: Scenario,
    contacts: list[tuple[float, float]],
    sim_anomalies: list[SimAnomaly],
    span_s: float,
    mode: str,
    seed: int,
) -> RunMetrics:
    """Run one deterministic simulation pass.

    Parameters
    ----------
    scenario:      the loaded scenario (assumptions live here)
    contacts:      list of (start_s, end_s) from scenario start
    sim_anomalies: anomalies with resolved occurrence times
    span_s:        total scenario duration in seconds
    mode:          "conventional" or "contingency"
    seed:          RNG seed used to generate sim_anomalies (recorded in metrics)
    """
    asmp = scenario.assumptions
    sc = SpacecraftState()
    log: list[str] = []

    def emit(t: float, msg: str) -> None:
        log.append(f"t+{t / 3600:6.2f}h  {msg}")

    # blocked_until[anomaly_id] = (occurs_at_s, resolved_at_s)
    # resolved_at_s == occurs_at_s means instant (no waiting).
    blocked_until: dict[str, tuple[float, float]] = {}
    round_trips = 0
    obj_completion_times: list[float] = []

    # ------------------------------------------------------------------ anomalies
    for sim_a in sim_anomalies:
        a = sim_a.anomaly
        t_occur = sim_a.occurs_at_s

        if mode == "contingency" and a.anticipated:
            # Resolved on board immediately — no round trip, no blocking.
            emit(t_occur, f"ANOMALY {a.id!r} ({a.kind}) → resolved on board (anticipated)")
            sc.detect_anomaly(t_occur, a.id, a.kind)
            sc.resolve_anomaly(t_occur, a.id)
            blocked_until[a.id] = (t_occur, t_occur)  # instant: zero wait
            continue

        if mode == "contingency":
            # Unanticipated: one escalation/reply cycle.
            cycles = 1
        else:
            cycles = asmp.cycles_for(AnomalyKind(a.kind))

        sc.detect_anomaly(t_occur, a.id, a.kind)
        emit(t_occur, f"ANOMALY {a.id!r} ({a.kind}) occurred")

        t_cursor = t_occur
        resolved = False
        for cycle in range(cycles):
            # Downlink telemetry / escalation packet
            t_dl = _next_contact_start(contacts, t_cursor, asmp.min_downlink_contact_s)
            if t_dl is None:
                emit(t_cursor, "  no usable contact for downlink — anomaly unresolved at span end")
                blocked_until[a.id] = (t_occur, span_s)
                break
            emit(t_dl, f"  cycle {cycle + 1}/{cycles}: telemetry downlinked at contact")
            round_trips += 1

            # Operator turnaround
            t_ready = t_dl + asmp.ground_turnaround_s
            emit(t_ready, f"  cycle {cycle + 1}/{cycles}: command ready after turnaround")

            # Uplink command
            t_ul = _next_contact_start(contacts, t_ready, asmp.min_uplink_contact_s)
            if t_ul is None:
                emit(t_cursor, "  no usable contact for uplink — anomaly unresolved at span end")
                blocked_until[a.id] = (t_occur, span_s)
                break
            emit(t_ul, f"  cycle {cycle + 1}/{cycles}: command uplinked at contact")
            round_trips += 1
            t_cursor = t_ul
        else:
            resolved = True

        if resolved:
            t_resolved = t_cursor
            sc.resolve_anomaly(t_resolved, a.id)
            emit(t_resolved, f"ANOMALY {a.id!r} resolved after {cycles} cycle(s)")
            blocked_until[a.id] = (t_occur, t_resolved)

    # time_blocked_s = sum of (resolved_at - occurred_at) for anomalies requiring ground loops.
    # Anticipated contingency anomalies contribute 0 (resolved_at == occurred_at).
    time_blocked_s = sum(
        max(0.0, resolved - occurred)
        for occurred, resolved in blocked_until.values()
    )

    # ------------------------------------------------------------------ objectives
    # Objectives execute in order after the latest anomaly resolution time.
    # In contingency mode, plans are uplinked in batches during contacts (background).
    t_available = 0.0  # earliest time spacecraft can start next objective

    # latest resolution time across all anomalies (objectives can't start before this)
    t_clear = max((resolved for _, resolved in blocked_until.values()), default=0.0)

    for obj in scenario.objectives:
        t_start_obj = max(t_available, t_clear)

        if t_start_obj + obj.duration_s > span_s:
            emit(t_start_obj, f"OBJECTIVE {obj.id!r} cannot start — not enough time in span")
            continue

        # In contingency mode, plans are uplinked in background; no extra delay here.

        sc.start_objective(t_start_obj, obj.id)
        emit(t_start_obj, f"OBJECTIVE {obj.id!r} started")
        t_end_obj = t_start_obj + obj.duration_s
        sc.complete_objective(t_end_obj)
        emit(t_end_obj, f"OBJECTIVE {obj.id!r} completed")
        obj_completion_times.append(t_end_obj)
        t_available = t_end_obj

    objectives_completed = len(obj_completion_times)
    time_to_all = obj_completion_times[-1] if obj_completion_times else None

    return RunMetrics(
        mode=mode,
        seed=seed,
        round_trips=round_trips,
        time_to_all_objectives_s=time_to_all,
        time_blocked_s=time_blocked_s,
        objectives_completed=objectives_completed,
        objectives_total=len(scenario.objectives),
        event_log=log,
    )


# ---------------------------------------------------------------------------
# Public API: single run
# ---------------------------------------------------------------------------


def run_scenario(
    scenario: Scenario,
    windows: list[ContactWindow],
    seed: int | None = None,
) -> tuple[RunMetrics, RunMetrics]:
    """Run a scenario in both modes and return (conventional, contingency).

    Parameters
    ----------
    scenario: loaded and validated Scenario
    windows:  contact windows from orbit.compute_contact_windows()
    seed:     RNG seed for random anomaly times (overrides scenario.seed if given)
    """
    if seed is None:
        seed = scenario.seed

    span_s = scenario.time_span.hours * 3600.0

    if not windows:
        contacts: list[tuple[float, float]] = []
    else:
        # Use scenario start as epoch so contact offsets align with anomaly times,
        # which are also measured from scenario start.  Using windows[0].start would
        # silently shift all contacts by (windows[0].start - scenario_start), causing
        # anomalies that occur before the first window to skip contacts they should use.
        t_start = datetime.fromisoformat(
            scenario.time_span.start.replace("Z", "+00:00")
        ).astimezone(UTC)
        contacts = [
            ((w.start - t_start).total_seconds(), (w.end - t_start).total_seconds())
            for w in windows
        ]

    sim_anomalies = resolve_anomaly_times(scenario, seed)

    conv = _simulate(scenario, contacts, sim_anomalies, span_s, "conventional", seed)
    cont = _simulate(scenario, contacts, sim_anomalies, span_s, "contingency", seed)
    return conv, cont


# ---------------------------------------------------------------------------
# Public API: Monte Carlo
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloSummary:
    """Aggregate statistics across N Monte Carlo runs."""

    mode: str
    n_runs: int
    round_trips_median: float
    round_trips_p25: float
    round_trips_p75: float
    time_to_all_h_median: float | None
    time_to_all_h_p25: float | None
    time_to_all_h_p75: float | None
    time_blocked_h_median: float
    time_blocked_h_p25: float
    time_blocked_h_p75: float
    objectives_completed_median: float
    all_completed_fraction: float  # fraction of runs where all objectives completed


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100
    lo, frac = int(idx), idx - int(idx)
    if lo + 1 < len(sorted_v):
        return sorted_v[lo] + frac * (sorted_v[lo + 1] - sorted_v[lo])
    return sorted_v[lo]


def run_monte_carlo(
    scenario: Scenario,
    windows: list[ContactWindow],
    n_runs: int,
    base_seed: int | None = None,
) -> tuple[MonteCarloSummary, MonteCarloSummary]:
    """Run N Monte Carlo iterations, each with a different seed.

    Seeds are derived deterministically from base_seed (defaults to scenario.seed).
    The plan set is fixed; only random-range anomaly times vary.
    Returns (conventional_summary, contingency_summary).
    """
    if base_seed is None:
        base_seed = scenario.seed

    span_s = scenario.time_span.hours * 3600.0

    if not windows:
        contacts: list[tuple[float, float]] = []
    else:
        t_start = datetime.fromisoformat(
            scenario.time_span.start.replace("Z", "+00:00")
        ).astimezone(UTC)
        contacts = [
            ((w.start - t_start).total_seconds(), (w.end - t_start).total_seconds())
            for w in windows
        ]

    # Derive per-run seeds deterministically from base_seed.
    seed_rng = random.Random(base_seed)
    seeds = [seed_rng.randint(0, 2**31 - 1) for _ in range(n_runs)]

    conv_metrics: list[RunMetrics] = []
    cont_metrics: list[RunMetrics] = []

    for run_seed in seeds:
        sim_anomalies = resolve_anomaly_times(scenario, run_seed)
        conv = _simulate(scenario, contacts, sim_anomalies, span_s, "conventional", run_seed)
        cont = _simulate(scenario, contacts, sim_anomalies, span_s, "contingency", run_seed)
        conv_metrics.append(conv)
        cont_metrics.append(cont)

    return _summarise_runs(conv_metrics), _summarise_runs(cont_metrics)


def _summarise_runs(runs: list[RunMetrics]) -> MonteCarloSummary:
    rts = [float(r.round_trips) for r in runs]
    ttao = [
        r.time_to_all_objectives_s / 3600.0
        for r in runs
        if r.time_to_all_objectives_s is not None
    ]
    blocked = [r.time_blocked_h for r in runs]
    objs = [float(r.objectives_completed) for r in runs]
    all_done = [r for r in runs if r.all_objectives_completed]

    return MonteCarloSummary(
        mode=runs[0].mode,
        n_runs=len(runs),
        round_trips_median=_percentile(rts, 50),
        round_trips_p25=_percentile(rts, 25),
        round_trips_p75=_percentile(rts, 75),
        time_to_all_h_median=_percentile(ttao, 50) if ttao else None,
        time_to_all_h_p25=_percentile(ttao, 25) if ttao else None,
        time_to_all_h_p75=_percentile(ttao, 75) if ttao else None,
        time_blocked_h_median=_percentile(blocked, 50),
        time_blocked_h_p25=_percentile(blocked, 25),
        time_blocked_h_p75=_percentile(blocked, 75),
        objectives_completed_median=_percentile(objs, 50),
        all_completed_fraction=len(all_done) / len(runs),
    )
