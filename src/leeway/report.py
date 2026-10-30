"""Leeway report module.

Renders side-by-side comparison tables for single runs and Monte Carlo summaries.
All formatting lives here; ops_sim.py contains only numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from leeway.ops_sim import MonteCarloSummary, RunMetrics


@dataclass
class ReproduceInfo:
    """Provenance metadata that lets anyone replay an identical simulation."""

    scenario_file: str
    seed: int
    tle_fetched_at: str  # ISO datetime string, "uploaded", or "unknown"


def _fmt_h(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    h = seconds / 3600.0
    return f"{h:.2f}h"


def _fmt_h_direct(h: float | None) -> str:
    if h is None:
        return "—"
    return f"{h:.2f}h"


def _pct(frac: float) -> str:
    return f"{frac * 100:.0f}%"


def _delta(conv: float | None, cont: float | None, *, lower_is_better: bool = True) -> str:
    """Return a delta string with direction arrow."""
    if conv is None or cont is None:
        return ""
    diff = conv - cont
    if lower_is_better:
        if diff > 0:
            return f"  ↓ {diff:.2f} saved"
        if diff < 0:
            return f"  ↑ {abs(diff):.2f} worse"
        return "  = same"
    else:
        if diff < 0:
            return f"  ↑ {abs(diff):.2f} gained"
        if diff > 0:
            return f"  ↓ {diff:.2f} lost"
        return "  = same"


# ---------------------------------------------------------------------------
# Single-run report
# ---------------------------------------------------------------------------


def render_comparison(
    conv: RunMetrics,
    cont: RunMetrics,
    *,
    verbose: bool = False,
    reproduce: ReproduceInfo | None = None,
) -> str:
    """Render a side-by-side table comparing conventional and contingency results."""
    lines: list[str] = []
    W = 28  # label column width
    C = 14  # value column width

    def row(label: str, cv: str, ct: str, note: str = "") -> None:
        lines.append(f"  {label:<{W}}  {cv:>{C}}  {ct:>{C}}  {note}")

    lines.append("")
    lines.append(f"  {'':>{W}}  {'Conventional':>{C}}  {'Contingency':>{C}}")
    lines.append("  " + "-" * (W + C * 2 + 8))

    rt_delta = _delta(float(conv.round_trips), float(cont.round_trips))
    row("Round trips", str(conv.round_trips), str(cont.round_trips), rt_delta)

    ttao_cv = _fmt_h(conv.time_to_all_objectives_s)
    ttao_ct = _fmt_h(cont.time_to_all_objectives_s)
    ttao_delta = _delta(
        conv.time_to_all_objectives_s,
        cont.time_to_all_objectives_s,
    )
    row("Time to all objectives", ttao_cv, ttao_ct, ttao_delta)

    blocked_delta = _delta(conv.time_blocked_s, cont.time_blocked_s)
    row(
        "Time blocked (waiting)",
        _fmt_h(conv.time_blocked_s),
        _fmt_h(cont.time_blocked_s),
        blocked_delta,
    )

    row(
        "Objectives completed",
        f"{conv.objectives_completed}/{conv.objectives_total}",
        f"{cont.objectives_completed}/{cont.objectives_total}",
    )

    lines.append("")

    if verbose:
        for m, label in ((conv, "CONVENTIONAL"), (cont, "CONTINGENCY")):
            lines.append(f"  {label} EVENT LOG (seed={m.seed})")
            for entry in m.event_log:
                lines.append(f"    {entry}")
            lines.append("")

    if reproduce is not None:
        lines.append(
            f"  Reproduce: scenario={reproduce.scenario_file}"
            f"  seed={reproduce.seed}"
            f"  TLE-fetched={reproduce.tle_fetched_at}"
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Monte Carlo report
# ---------------------------------------------------------------------------


def render_monte_carlo(
    conv: MonteCarloSummary,
    cont: MonteCarloSummary,
    *,
    reproduce: ReproduceInfo | None = None,
) -> str:
    """Render a side-by-side table for Monte Carlo results (median ± IQR)."""
    lines: list[str] = []
    W = 30
    C = 22

    def _iqr_str(med: float | None, p25: float | None, p75: float | None) -> str:
        if med is None:
            return "—"
        return f"{med:.2f}  [{p25:.2f}–{p75:.2f}]"

    def row(label: str, cv: str, ct: str, note: str = "") -> None:
        lines.append(f"  {label:<{W}}  {cv:>{C}}  {ct:>{C}}  {note}")

    lines.append("")
    lines.append(f"  Monte Carlo  N={conv.n_runs}  (median  [p25–p75])")
    lines.append(f"  {'':>{W}}  {'Conventional':>{C}}  {'Contingency':>{C}}")
    lines.append("  " + "-" * (W + C * 2 + 8))

    row(
        "Round trips",
        _iqr_str(conv.round_trips_median, conv.round_trips_p25, conv.round_trips_p75),
        _iqr_str(cont.round_trips_median, cont.round_trips_p25, cont.round_trips_p75),
        _delta(conv.round_trips_median, cont.round_trips_median),
    )

    row(
        "Time to all objectives (h)",
        _iqr_str(conv.time_to_all_h_median, conv.time_to_all_h_p25, conv.time_to_all_h_p75),
        _iqr_str(cont.time_to_all_h_median, cont.time_to_all_h_p25, cont.time_to_all_h_p75),
        _delta(conv.time_to_all_h_median, cont.time_to_all_h_median),
    )

    row(
        "Time blocked (h)",
        _iqr_str(
            conv.time_blocked_h_median, conv.time_blocked_h_p25, conv.time_blocked_h_p75
        ),
        _iqr_str(
            cont.time_blocked_h_median, cont.time_blocked_h_p25, cont.time_blocked_h_p75
        ),
        _delta(conv.time_blocked_h_median, cont.time_blocked_h_median),
    )

    row(
        "All objectives completed",
        _pct(conv.all_completed_fraction),
        _pct(cont.all_completed_fraction),
    )

    lines.append("")

    if reproduce is not None:
        lines.append(
            f"  Reproduce: scenario={reproduce.scenario_file}"
            f"  seed={reproduce.seed}"
            f"  TLE-fetched={reproduce.tle_fetched_at}"
        )
        lines.append("")

    return "\n".join(lines)
