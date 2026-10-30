"""Leeway HTML report generator.

Produces a self-contained HTML file (inline CSS + inline Plotly JS) suitable for
offline viewing or emailing to an operator after a call.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from leeway.ops_sim import MonteCarloSummary, RunMetrics
from leeway.plan import Scenario
from leeway.report import ReproduceInfo

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# CSS (kept as a plain string — not an f-string — to avoid escaping {}).
# ---------------------------------------------------------------------------

_CSS = (
    "* { box-sizing: border-box; margin: 0; padding: 0; }\n"
    "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;"
    " background: #f8fafc; color: #1e293b; line-height: 1.5; }\n"
    ".container { max-width: 980px; margin: 0 auto; padding: 2rem 1.5rem; }\n"
    "header { display: flex; justify-content: space-between; align-items: baseline;"
    " border-bottom: 2px solid #2563eb; padding-bottom: 0.75rem; margin-bottom: 1.5rem; }\n"
    ".brand { font-size: 1.1rem; font-weight: 700; color: #2563eb; }\n"
    ".generated { font-size: 0.8rem; color: #64748b; }\n"
    ".headline { font-size: 1.4rem; font-weight: 700; color: #2563eb; margin-bottom: 0.4rem; }\n"
    ".sub { font-size: 0.9rem; color: #64748b; margin-bottom: 2rem; }\n"
    "section { margin-bottom: 2.5rem; }\n"
    "h2 { font-size: 0.85rem; font-weight: 600; color: #475569; text-transform: uppercase;"
    " letter-spacing: 0.06em; margin-bottom: 1rem; padding-bottom: 0.35rem;"
    " border-bottom: 1px solid #e2e8f0; }\n"
    "table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }\n"
    "th { text-align: left; padding: 0.5rem 0.75rem; background: #f1f5f9;"
    " font-weight: 600; color: #475569; font-size: 0.8rem; }\n"
    "td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #e2e8f0; vertical-align: top; }\n"
    "td.val { font-weight: 600; white-space: nowrap; }\n"
    "td.note { color: #94a3b8; font-size: 0.82rem; }\n"
    ".saved { color: #16a34a; }\n"
    ".worse { color: #dc2626; }\n"
    ".caption { font-size: 0.82rem; color: #64748b; margin-bottom: 0.5rem; }\n"
    "footer { border-top: 1px solid #e2e8f0; padding-top: 1rem; margin-top: 0.5rem; }\n"
    ".note-text { font-size: 0.82rem; color: #64748b; margin-bottom: 0.6rem; }\n"
    ".reproduce { font-size: 0.78rem; color: #94a3b8; font-family: ui-monospace,"
    " 'Cascadia Code', monospace; background: #f1f5f9; padding: 0.5rem 0.75rem;"
    " border-radius: 4px; word-break: break-all; }\n"
)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """Minimal HTML escaping for user-supplied strings."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_h(seconds: float | None) -> str:
    if seconds is None:
        return "&#8212;"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def _delta_html(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return ""
    d = a - b
    if abs(d) < 0.01:
        return "same"
    if d > 0:
        return f'<span class="saved">&#8595; {d:.1f} saved</span>'
    return f'<span class="worse">&#8593; {abs(d):.1f} worse</span>'


def _metrics_table(conv: RunMetrics, cont: RunMetrics) -> str:
    rows = [
        (
            "Round trips",
            str(conv.round_trips),
            str(cont.round_trips),
            _delta_html(float(conv.round_trips), float(cont.round_trips)),
        ),
        (
            "Time to all objectives",
            _fmt_h(conv.time_to_all_objectives_s),
            _fmt_h(cont.time_to_all_objectives_s),
            _delta_html(conv.time_to_all_objectives_s, cont.time_to_all_objectives_s),
        ),
        (
            "Time blocked waiting",
            _fmt_h(conv.time_blocked_s),
            _fmt_h(cont.time_blocked_s),
            _delta_html(conv.time_blocked_s, cont.time_blocked_s),
        ),
        (
            "Objectives completed",
            f"{conv.objectives_completed}/{conv.objectives_total}",
            f"{cont.objectives_completed}/{cont.objectives_total}",
            "",
        ),
    ]
    parts = [
        '<table>',
        '<thead><tr><th>Metric</th><th>Conventional</th>',
        '<th>Contingency</th><th>&#916;</th></tr></thead>',
        '<tbody>',
    ]
    for label, cv, ct, delta in rows:
        parts.append(
            f'<tr><td>{_esc(label)}</td>'
            f'<td class="val">{cv}</td>'
            f'<td class="val">{ct}</td>'
            f'<td>{delta}</td></tr>'
        )
    parts += ["</tbody>", "</table>"]
    return "\n".join(parts)


def _mc_table(mc_conv: MonteCarloSummary, mc_cont: MonteCarloSummary) -> str:
    def iqr(med: float | None, p25: float | None, p75: float | None) -> str:
        if med is None:
            return "&#8212;"
        return f"{med:.1f} [{p25:.1f}&#8211;{p75:.1f}]"

    rows = [
        (
            "Round trips",
            iqr(mc_conv.round_trips_median, mc_conv.round_trips_p25, mc_conv.round_trips_p75),
            iqr(mc_cont.round_trips_median, mc_cont.round_trips_p25, mc_cont.round_trips_p75),
        ),
        (
            "Time to all objectives (h)",
            iqr(mc_conv.time_to_all_h_median, mc_conv.time_to_all_h_p25,
                mc_conv.time_to_all_h_p75),
            iqr(mc_cont.time_to_all_h_median, mc_cont.time_to_all_h_p25,
                mc_cont.time_to_all_h_p75),
        ),
        (
            "Time blocked (h)",
            iqr(mc_conv.time_blocked_h_median, mc_conv.time_blocked_h_p25,
                mc_conv.time_blocked_h_p75),
            iqr(mc_cont.time_blocked_h_median, mc_cont.time_blocked_h_p25,
                mc_cont.time_blocked_h_p75),
        ),
        (
            "All objectives completed",
            f"{mc_conv.all_completed_fraction * 100:.0f}%",
            f"{mc_cont.all_completed_fraction * 100:.0f}%",
        ),
    ]
    parts = [
        f'<p class="caption">Monte Carlo N={mc_conv.n_runs} &#8212; median [p25&#8211;p75]</p>',
        '<table>',
        '<thead><tr><th>Metric</th><th>Conventional</th><th>Contingency</th></tr></thead>',
        '<tbody>',
    ]
    for row in rows:
        cells = "".join(f'<td class="val">{v}</td>' for v in row)
        parts.append(f"<tr>{cells}</tr>")
    parts += ["</tbody>", "</table>"]
    return "\n".join(parts)


def _assumptions_table(scenario: Scenario) -> str:
    asmp = scenario.assumptions
    rows: list[tuple[str, str, str]] = [
        (
            "Ground turnaround time",
            f"{asmp.ground_turnaround_s / 3600:.2f} h",
            "Time from anomaly telemetry seen to command ready",
        ),
        (
            "Min contact for uplink",
            f"{asmp.min_uplink_contact_s:.0f} s",
            "Shortest usable pass for sending a command",
        ),
        (
            "Min contact for downlink",
            f"{asmp.min_downlink_contact_s:.0f} s",
            "Shortest usable pass for pulling telemetry",
        ),
        (
            "Plan upload time",
            f"{asmp.plan_upload_s:.0f} s",
            "Time to uplink a contingency plan",
        ),
    ]
    for kind, cyc in asmp.command_cycles_per_anomaly.items():
        rows.append((
            f"Command cycles &#8212; {kind.replace('_', ' ')}",
            str(cyc),
            "Downlink&#8594;uplink loops to resolve this anomaly type",
        ))
    parts = [
        '<table>',
        '<thead><tr><th>Parameter</th><th>Value</th><th>Notes</th></tr></thead>',
        '<tbody>',
    ]
    for label, value, note in rows:
        parts.append(
            f'<tr><td>{label}</td>'
            f'<td class="val">{_esc(value)}</td>'
            f'<td class="note">{note}</td></tr>'
        )
    parts += ["</tbody>", "</table>"]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_html_report(
    conv: RunMetrics,
    cont: RunMetrics,
    scenario: Scenario,
    sat_name: str,
    net_label: str,
    n_contacts: int,
    reproduce: ReproduceInfo,
    *,
    mc_conv: MonteCarloSummary | None = None,
    mc_cont: MonteCarloSummary | None = None,
    fig: Any | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Return a self-contained HTML report string.

    Parameters
    ----------
    conv / cont:    single-run metrics for conventional and contingency modes
    scenario:       the scenario that was simulated
    sat_name:       human-readable satellite name (from TLE or preset label)
    net_label:      human-readable ground-network label
    n_contacts:     number of contact windows found
    reproduce:      provenance metadata for the reproduce line
    mc_conv/cont:   optional Monte Carlo summaries (replaces single-run table when present)
    fig:            optional plotly.graph_objects.Figure for the timeline chart
    generated_at:   report generation time (defaults to now UTC)
    """
    if generated_at is None:
        generated_at = datetime.now(UTC)

    gen_str = generated_at.strftime("%Y-%m-%d %H:%M UTC")
    anticipated = sum(1 for a in scenario.anomalies if a.anticipated)
    unanticipated = len(scenario.anomalies) - anticipated

    # ── headline ────────────────────────────────────────────────────────────
    if mc_conv and mc_cont:
        rt_med = mc_conv.round_trips_median - mc_cont.round_trips_median
        wait_h = mc_conv.time_blocked_h_median - mc_cont.time_blocked_h_median
        headline = (
            f"Contingency plans remove {rt_med:.0f} of {mc_conv.round_trips_median:.0f} "
            f"round trips and {wait_h:.1f}h of waiting "
            f"(median across {mc_conv.n_runs} runs)."
        )
    else:
        rt_saved = conv.round_trips - cont.round_trips
        wait_h = (conv.time_blocked_s - cont.time_blocked_s) / 3600.0
        headline = (
            f"Contingency plans remove {rt_saved} of {conv.round_trips} round trips "
            f"and {wait_h:.1f}h of waiting."
        )

    sub = (
        f"{_esc(sat_name)} &middot; {_esc(net_label)} &middot; "
        f"{n_contacts} contacts in {scenario.time_span.hours:.0f}h &middot; "
        f"{anticipated} anticipated + {unanticipated} unanticipated anomalies"
    )

    # ── timeline chart ───────────────────────────────────────────────────────
    if fig is not None:
        chart_section = (
            "<section>\n"
            "<h2>Contact Windows &amp; Events</h2>\n"
            + fig.to_html(include_plotlyjs=True, full_html=False)
            + "\n</section>\n"
        )
    else:
        chart_section = ""

    # ── metrics table ────────────────────────────────────────────────────────
    if mc_conv and mc_cont:
        metrics_html = _mc_table(mc_conv, mc_cont)
    else:
        metrics_html = _metrics_table(conv, cont)

    # ── reproduce line ───────────────────────────────────────────────────────
    reproduce_text = (
        f"scenario={reproduce.scenario_file}"
        f"  seed={reproduce.seed}"
        f"  TLE-fetched={reproduce.tle_fetched_at}"
    )

    page_title = f"Leeway — {sat_name} — {gen_str}"

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"  <title>{_esc(page_title)}</title>\n"
        "  <style>\n"
        + _CSS
        + "  </style>\n"
        "</head>\n"
        "<body>\n"
        '<div class="container">\n'
        "<header>\n"
        f'  <div class="brand">&#128752; Leeway &middot; Round-Trip Calculator</div>\n'
        f'  <div class="generated">Generated {gen_str}</div>\n'
        "</header>\n"
        f'<div class="headline">{headline}</div>\n'
        f'<div class="sub">{sub}</div>\n'
        + chart_section
        + "<section>\n"
        "<h2>Metrics Comparison</h2>\n"
        + metrics_html
        + "\n</section>\n"
        "<section>\n"
        "<h2>Assumptions</h2>\n"
        + _assumptions_table(scenario)
        + "\n</section>\n"
        "<footer>\n"
        '<div class="note-text">&#9888; Simulated estimate based on public orbit data '
        "(CelesTrak) and the assumptions above. "
        "Actual operations depend on specific mission constraints, ground-station "
        "availability, and spacecraft health.</div>\n"
        f'<div class="reproduce">Reproduce: {_esc(reproduce_text)}</div>\n'
        "</footer>\n"
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )
