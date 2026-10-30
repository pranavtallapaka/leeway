"""Leeway — Phase A Streamlit app.

Run with: uv run streamlit run app.py
"""

from __future__ import annotations

import copy
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import yaml

from leeway.ground import load_stations
from leeway.html_report import build_html_report
from leeway.ops_sim import MonteCarloSummary, RunMetrics, run_monte_carlo, run_scenario
from leeway.orbit import (
    ContactWindow,
    compute_contact_windows,
    fetch_tle,
    parse_tle,
    tle_cache_timestamp,
)
from leeway.plan import (
    Anomaly,
    AnomalyKind,
    Assumptions,
    NoradSatellite,
    Objective,
    ObjectiveKind,
    RandomRange,
    Scenario,
    TimeSpan,
    load_scenario,
)
from leeway.report import ReproduceInfo

# ---------------------------------------------------------------------------
# Constants and presets
# ---------------------------------------------------------------------------

ACCENT = "#2563EB"  # single accent color throughout
CACHE_DIR = Path(__file__).parent / "data" / "cache"
_PRESETS_FILE = Path(__file__).parent / "scenarios" / "presets.yaml"


def _load_presets(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


_PRESETS = _load_presets(_PRESETS_FILE)

PRESET_SATELLITES: dict[str, int] = {s["name"]: s["norad_id"] for s in _PRESETS["satellites"]}
PRESET_NETWORKS: dict[str, Path] = {
    n["name"]: Path(__file__).parent / n["path"] for n in _PRESETS["ground_networks"]
}
PRESET_SCENARIOS: dict[str, Path] = {
    s["name"]: Path(__file__).parent / s["path"] for s in _PRESETS["scenarios"]
}

_ANOMALY_KIND_LABELS = {
    AnomalyKind.payload_reset: "Payload reset",
    AnomalyKind.missed_window: "Missed window",
    AnomalyKind.attitude_error: "Attitude error",
    AnomalyKind.comms_dropout: "Comms dropout",
    AnomalyKind.power_anomaly: "Power anomaly",
    AnomalyKind.thermal_exceedance: "Thermal exceedance",
    AnomalyKind.software_fault: "Software fault",
    AnomalyKind.other: "Other",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Leeway — Round-Trip Calculator",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .headline { font-size: 1.5rem; font-weight: 700; color: #2563EB; margin-bottom: 0.25rem; }
    .sub { color: #555; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .metric-box { background: #F0F4FF; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; }
    .assumption-row { font-size: 0.85rem; color: #444; padding: 0.15rem 0; }
    .replay-badge {
        background: #FFF3CD; color: #856404; border-radius: 4px;
        padding: 0.1rem 0.5rem; font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_h(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def _scenario_from_controls(
    norad_id: int,
    stations_path: Path,
    span_hours: float,
    start_iso: str,
    seed: int,
    n_anomalies: int,
    anomaly_kind: AnomalyKind,
    anticipated_fraction: float,
    turnaround_h: float,
    cycles: int,
    min_uplink_s: float,
    min_downlink_s: float,
    plan_upload_s: float,
) -> Scenario:
    """Build a Scenario purely from UI sliders (no file needed)."""
    objectives = [
        Objective(id="image_target_1", kind=ObjectiveKind.image_target, duration_s=120),
        Objective(id="downlink_batch", kind=ObjectiveKind.downlink_data, duration_s=600),
        Objective(id="calibration", kind=ObjectiveKind.calibration, duration_s=180),
    ]
    span_s = span_hours * 3600
    anomalies: list[Anomaly] = []
    chunk = span_s / max(n_anomalies + 1, 2)
    for i in range(n_anomalies):
        earliest = chunk * (i + 0.5)
        latest = min(chunk * (i + 1.0), span_s - 60)
        is_anticipated = i < int(n_anomalies * anticipated_fraction)
        anomalies.append(
            Anomaly(
                id=f"anomaly_{i + 1}",
                kind=anomaly_kind,
                timing=RandomRange(
                    kind="random",
                    earliest_s=max(0, earliest),
                    latest_s=max(earliest + 60, latest),
                ),
                anticipated=is_anticipated,
            )
        )
    return Scenario(
        name="ui_scenario",
        satellite=NoradSatellite(norad_id=norad_id),
        ground_stations_file=str(stations_path),
        time_span=TimeSpan(start=start_iso, hours=span_hours),
        seed=seed,
        objectives=objectives,
        anomalies=anomalies,
        assumptions=Assumptions(
            ground_turnaround_s=turnaround_h * 3600,
            command_cycles_per_anomaly={anomaly_kind.value: cycles},
            min_uplink_contact_s=min_uplink_s,
            min_downlink_contact_s=min_downlink_s,
            plan_upload_s=plan_upload_s,
        ),
    )


@st.cache_data(ttl=7200, show_spinner=False)
def _fetch_tle_cached(norad_id: int) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return fetch_tle(norad_id, CACHE_DIR)


@st.cache_data(ttl=7200, show_spinner=False)
def _compute_windows_cached(
    tle_text: str, stations_yaml: str, start_iso: str, hours: float
) -> list[ContactWindow]:
    sat = parse_tle(tle_text)
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(stations_yaml)
        tmp = Path(f.name)
    stations = load_stations(tmp)
    tmp.unlink()
    t_start = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(UTC)
    t_end = t_start + timedelta(hours=hours)
    return compute_contact_windows(sat, stations, t_start, t_end)


def _parse_event_log(log: list[str]) -> list[dict]:
    """Convert ops_sim event log strings into structured dicts for Plotly."""
    events: list[dict] = []
    for line in log:
        # format: "t+  H.HHh  MESSAGE"
        try:
            parts = line.split("h  ", 1)
            t_h = float(parts[0].replace("t+", "").strip())
            msg = parts[1].strip()
        except (ValueError, IndexError):
            continue
        events.append({"t_h": t_h, "msg": msg})
    return events


def _categorise(msg: str) -> str:
    m = msg.upper()
    if "ANOMALY" in m and "RESOLVED" in m:
        return "resolved"
    if "ANOMALY" in m and ("OCCURRED" in m or "ON BOARD" in m):
        return "anomaly"
    if "DOWNLINKED" in m:
        return "downlink"
    if "UPLINKED" in m:
        return "uplink"
    if "TURNAROUND" in m or "COMMAND READY" in m:
        return "turnaround"
    if "OBJECTIVE" in m and "STARTED" in m:
        return "obj_start"
    if "OBJECTIVE" in m and "COMPLETED" in m:
        return "obj_done"
    return "other"


_EVENT_COLORS = {
    "anomaly": "#EF4444",
    "resolved": "#22C55E",
    "downlink": "#F59E0B",
    "uplink": ACCENT,
    "turnaround": "#A78BFA",
    "obj_start": "#6B7280",
    "obj_done": "#059669",
    "other": "#CBD5E1",
}

_EVENT_SYMBOLS = {
    "anomaly": "x",
    "resolved": "circle",
    "downlink": "triangle-down",
    "uplink": "triangle-up",
    "turnaround": "diamond",
    "obj_start": "square",
    "obj_done": "star",
    "other": "circle-open",
}

# ---------------------------------------------------------------------------
# Timeline chart
# ---------------------------------------------------------------------------


def _build_timeline(
    windows: list[ContactWindow],
    conv: RunMetrics,
    cont: RunMetrics,
    span_hours: float,
) -> go.Figure:
    fig = go.Figure()

    if not windows:
        fig.add_annotation(text="No contact windows in this span.", x=0.5, y=0.5, showarrow=False)
        return fig

    epoch = windows[0].start

    def t_h(dt: datetime) -> float:
        return (dt - epoch).total_seconds() / 3600.0

    # ── contact windows (background bars) ────────────────────────────────
    for w in windows:
        s, e = t_h(w.start), t_h(w.end)
        fig.add_shape(
            type="rect", x0=s, x1=e, y0=0.55, y1=2.45,
            fillcolor=ACCENT, opacity=0.12, line_width=0,
            layer="below",
        )
        fig.add_shape(
            type="rect", x0=s, x1=e, y0=0.55, y1=2.45,
            fillcolor="rgba(0,0,0,0)", line=dict(color=ACCENT, width=1, dash="dot"),
        )

    # ── event traces ─────────────────────────────────────────────────────
    for row_y, metrics, label in ((2.0, conv, "Conventional"), (1.0, cont, "Contingency")):
        events = _parse_event_log(metrics.event_log)
        by_cat: dict[str, list] = {}
        for ev in events:
            cat = _categorise(ev["msg"])
            by_cat.setdefault(cat, []).append(ev)

        for cat, evs in by_cat.items():
            fig.add_trace(go.Scatter(
                x=[e["t_h"] for e in evs],
                y=[row_y] * len(evs),
                mode="markers+text",
                marker=dict(
                    symbol=_EVENT_SYMBOLS.get(cat, "circle"),
                    size=10,
                    color=_EVENT_COLORS.get(cat, "#CBD5E1"),
                    line=dict(color="white", width=1),
                ),
                text=[e["msg"][:40] for e in evs],
                textposition="top center",
                textfont=dict(size=8, color="#555"),
                hovertext=[f"[{label}] t+{e['t_h']:.2f}h<br>{e['msg']}" for e in evs],
                hoverinfo="text",
                name=cat.replace("_", " ").title(),
                legendgroup=cat,
                showlegend=(label == "Conventional"),
            ))

    # ── lane labels ───────────────────────────────────────────────────────
    for y, txt in ((2.0, "Conventional"), (1.0, "Contingency")):
        fig.add_annotation(
            x=-0.3, y=y, text=txt, showarrow=False,
            font=dict(size=11, color="#444"), xanchor="right",
        )

    fig.update_layout(
        height=320,
        margin=dict(l=110, r=20, t=30, b=40),
        xaxis=dict(
            title="Hours from scenario start",
            range=[-0.5, span_hours + 0.5],
            gridcolor="#F0F0F0",
        ),
        yaxis=dict(
            range=[0.4, 2.6],
            showticklabels=False,
            showgrid=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h", y=-0.25, x=0,
            font=dict(size=10),
            itemsizing="constant",
        ),
        hovermode="closest",
    )
    return fig


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _render_table(conv: RunMetrics, cont: RunMetrics) -> None:
    def saved(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return ""
        d = a - b
        if abs(d) < 0.01:
            return "same"
        return f"↓ {d:.1f} saved" if d > 0 else f"↑ {abs(d):.1f} worse"

    rows = [
        ("Round trips", str(conv.round_trips), str(cont.round_trips),
         saved(conv.round_trips, cont.round_trips)),
        ("Time to all objectives",
         _fmt_h(conv.time_to_all_objectives_s), _fmt_h(cont.time_to_all_objectives_s),
         saved(conv.time_to_all_objectives_s, cont.time_to_all_objectives_s)),
        ("Time blocked waiting",
         _fmt_h(conv.time_blocked_s), _fmt_h(cont.time_blocked_s),
         saved(conv.time_blocked_s, cont.time_blocked_s)),
        ("Objectives completed",
         f"{conv.objectives_completed}/{conv.objectives_total}",
         f"{cont.objectives_completed}/{cont.objectives_total}", ""),
    ]
    col_w = [2, 1.2, 1.2, 1.5]
    cols = st.columns(col_w)
    for col, hdr in zip(cols, ["Metric", "Conventional", "Contingency", "Δ"]):
        col.markdown(f"**{hdr}**")
    st.divider()
    for label, cv, ct, delta in rows:
        cols = st.columns(col_w)
        cols[0].write(label)
        cols[1].write(cv)
        cols[2].markdown(f"**{ct}**")
        if delta:
            color = "green" if "saved" in delta else ("red" if "worse" in delta else "gray")
            cols[3].markdown(f":{color}[{delta}]")


def _render_mc_table(mc_conv: MonteCarloSummary, mc_cont: MonteCarloSummary) -> None:
    def iqr(med: float | None, p25: float | None, p75: float | None) -> str:
        if med is None:
            return "—"
        return f"{med:.1f}  [{p25:.1f}–{p75:.1f}]"

    rows = [
        ("Round trips",
         iqr(mc_conv.round_trips_median, mc_conv.round_trips_p25, mc_conv.round_trips_p75),
         iqr(mc_cont.round_trips_median, mc_cont.round_trips_p25, mc_cont.round_trips_p75)),
        ("Time to all objectives (h)",
         iqr(mc_conv.time_to_all_h_median, mc_conv.time_to_all_h_p25, mc_conv.time_to_all_h_p75),
         iqr(mc_cont.time_to_all_h_median, mc_cont.time_to_all_h_p25, mc_cont.time_to_all_h_p75)),
        ("Time blocked (h)",
         iqr(mc_conv.time_blocked_h_median, mc_conv.time_blocked_h_p25,
             mc_conv.time_blocked_h_p75),
         iqr(mc_cont.time_blocked_h_median, mc_cont.time_blocked_h_p25,
             mc_cont.time_blocked_h_p75)),
        ("All objectives completed",
         f"{mc_conv.all_completed_fraction * 100:.0f}%",
         f"{mc_cont.all_completed_fraction * 100:.0f}%"),
    ]
    col_w = [2, 2, 2]
    cols = st.columns(col_w)
    for col, hdr in zip(cols, ["Metric  (median  [p25–p75])", "Conventional", "Contingency"]):
        col.markdown(f"**{hdr}**")
    st.divider()
    for row in rows:
        cols = st.columns(col_w)
        for col, val in zip(cols, row):
            col.write(val)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar() -> dict:
    st.sidebar.title("🛰️ Leeway")
    st.sidebar.caption("Phase A — Round-Trip Calculator")
    st.sidebar.divider()

    cfg: dict = {}

    # ── Satellite ─────────────────────────────────────────────────────────
    st.sidebar.subheader("Satellite")
    sat_mode = st.sidebar.radio(
        "Source", ["Preset", "NORAD ID", "Upload TLE"],
        horizontal=True, label_visibility="collapsed"
    )
    if sat_mode == "Preset":
        preset_name = st.sidebar.selectbox("Preset satellite", list(PRESET_SATELLITES))
        cfg["norad_id"] = PRESET_SATELLITES[preset_name]
        cfg["sat_label"] = preset_name
        cfg["tle_file"] = None
    elif sat_mode == "NORAD ID":
        cfg["norad_id"] = st.sidebar.number_input("NORAD ID", min_value=1, value=40697, step=1)
        cfg["sat_label"] = f"NORAD {cfg['norad_id']}"
        cfg["tle_file"] = None
    else:
        uploaded = st.sidebar.file_uploader("TLE file (.tle/.txt)", type=["tle", "txt"])
        cfg["norad_id"] = None
        cfg["tle_file"] = uploaded
        cfg["sat_label"] = uploaded.name if uploaded else "(no file)"

    # ── Ground network ────────────────────────────────────────────────────
    st.sidebar.subheader("Ground network")
    net_mode = st.sidebar.radio(
        "Network source", ["Preset", "Upload YAML"], horizontal=True, label_visibility="collapsed"
    )
    if net_mode == "Preset":
        net_name = st.sidebar.selectbox("Preset network", list(PRESET_NETWORKS))
        cfg["stations_path"] = PRESET_NETWORKS[net_name]
        cfg["stations_yaml"] = cfg["stations_path"].read_text()
        cfg["net_label"] = net_name
    else:
        up_net = st.sidebar.file_uploader("Ground stations YAML", type=["yaml", "yml"])
        if up_net:
            cfg["stations_yaml"] = up_net.read().decode()
            cfg["stations_path"] = None
            cfg["net_label"] = up_net.name
        else:
            cfg["stations_yaml"] = PRESET_NETWORKS["Polar pair (Svalbard + McMurdo)"].read_text()
            cfg["stations_path"] = PRESET_NETWORKS["Polar pair (Svalbard + McMurdo)"]
            cfg["net_label"] = "Polar pair (Svalbard + McMurdo)"

    # ── Scenario / time span ──────────────────────────────────────────────
    st.sidebar.subheader("Scenario")
    sc_mode = st.sidebar.radio(
        "Scenario source", ["Sliders", "Preset", "Upload YAML"],
        horizontal=True, label_visibility="collapsed",
    )
    cfg["sc_mode"] = sc_mode

    if sc_mode == "Preset":
        sc_name = st.sidebar.selectbox("Preset scenario", list(PRESET_SCENARIOS))
        cfg["scenario_path"] = PRESET_SCENARIOS[sc_name]
        cfg["scenario_label"] = sc_name
    elif sc_mode == "Upload YAML":
        up_sc = st.sidebar.file_uploader("Scenario YAML", type=["yaml", "yml"])
        cfg["scenario_upload"] = up_sc
        cfg["scenario_label"] = up_sc.name if up_sc else "(no file)"
    else:
        cfg["scenario_label"] = "Custom (sliders)"

    if sc_mode == "Sliders":
        cfg["span_hours"] = st.sidebar.slider("Span (hours)", 6, 72, 24, step=6)
        cfg["start_iso"] = "2024-01-10T12:00:00Z"
        cfg["seed"] = st.sidebar.number_input("Seed", min_value=0, value=42, step=1)

    # ── Assumptions (shown for Sliders; for file scenarios shown as overrides) ──
    st.sidebar.subheader("Assumptions")

    cfg["turnaround_h"] = st.sidebar.slider(
        "Ground turnaround (h)", 0.25, 8.0, 1.0, step=0.25,
        help="How long operators take to form a command after seeing telemetry.",
    )
    cfg["cycles"] = st.sidebar.slider(
        "Command cycles per anomaly", 1, 5, 2,
        help="How many downlink→uplink round trips to resolve one anomaly.",
    )
    cfg["n_anomalies"] = st.sidebar.slider("Number of anomalies", 0, 6, 2)
    cfg["anticipated_frac"] = st.sidebar.slider(
        "Fraction anticipated", 0.0, 1.0, 0.5, step=0.25,
        help="What fraction of anomalies have an on-board contingency plan.",
    )
    cfg["anomaly_kind"] = st.sidebar.selectbox(
        "Anomaly type",
        list(_ANOMALY_KIND_LABELS),
        format_func=lambda k: _ANOMALY_KIND_LABELS[k],
    )
    cfg["min_uplink_s"] = 30.0
    cfg["min_downlink_s"] = 120.0
    cfg["plan_upload_s"] = 180.0

    # ── Monte Carlo ───────────────────────────────────────────────────────
    st.sidebar.subheader("Monte Carlo")
    cfg["mc_on"] = st.sidebar.toggle("Enable Monte Carlo")
    if cfg["mc_on"]:
        cfg["mc_runs"] = st.sidebar.slider("Runs", 20, 500, 100, step=20)
    else:
        cfg["mc_runs"] = 1

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = _sidebar()

    st.title("Round-Trip Calculator")

    # ── Resolve TLE ───────────────────────────────────────────────────────
    tle_text: str | None = None
    sat_name = cfg.get("sat_label", "Unknown satellite")

    tle_fetched_at = "unknown"
    if cfg.get("tle_file") is not None:
        tle_text = cfg["tle_file"].read().decode()
        tle_fetched_at = "uploaded"
    elif cfg.get("norad_id"):
        with st.spinner(f"Fetching TLE for NORAD {cfg['norad_id']}…"):
            try:
                tle_text = _fetch_tle_cached(cfg["norad_id"])
                lines = tle_text.strip().splitlines()
                if len(lines) == 3:
                    sat_name = lines[0].strip()
                ts = tle_cache_timestamp(CACHE_DIR, cfg["norad_id"])
                if ts is not None:
                    tle_fetched_at = ts.strftime("%Y-%m-%dT%H:%MZ")
            except Exception as exc:
                st.error(f"Could not fetch TLE: {exc}")
                st.stop()

    if not tle_text:
        st.info("Select a satellite to begin.")
        st.stop()

    # ── Resolve scenario ──────────────────────────────────────────────────
    scenario: Scenario | None = None
    sc_mode = cfg["sc_mode"]

    if sc_mode == "Preset":
        try:
            scenario = load_scenario(cfg["scenario_path"])
            # Override assumptions from sliders
            scenario = copy.deepcopy(scenario)
            scenario.assumptions.ground_turnaround_s = cfg["turnaround_h"] * 3600
            for kind in scenario.assumptions.command_cycles_per_anomaly:
                scenario.assumptions.command_cycles_per_anomaly[kind] = cfg["cycles"]
        except Exception as exc:
            st.error(f"Could not load preset scenario: {exc}")
            st.stop()
    elif sc_mode == "Upload YAML":
        up = cfg.get("scenario_upload")
        if not up:
            st.info("Upload a scenario YAML file to continue.")
            st.stop()
        try:
            with tempfile.NamedTemporaryFile(suffix=".yaml", mode="wb", delete=False) as f:
                f.write(up.read())
                tmp = Path(f.name)
            scenario = load_scenario(tmp)
            tmp.unlink()
            scenario = copy.deepcopy(scenario)
            scenario.assumptions.ground_turnaround_s = cfg["turnaround_h"] * 3600
            for kind in scenario.assumptions.command_cycles_per_anomaly:
                scenario.assumptions.command_cycles_per_anomaly[kind] = cfg["cycles"]
        except Exception as exc:
            st.error(f"Could not load uploaded scenario: {exc}")
            st.stop()
    else:
        # Sliders
        try:
            scenario = _scenario_from_controls(
                norad_id=cfg["norad_id"],
                stations_path=cfg["stations_path"] or Path("scenarios/ground/polar_network.yaml"),
                span_hours=cfg["span_hours"],
                start_iso=cfg["start_iso"],
                seed=cfg["seed"],
                n_anomalies=cfg["n_anomalies"],
                anomaly_kind=cfg["anomaly_kind"],
                anticipated_fraction=cfg["anticipated_frac"],
                turnaround_h=cfg["turnaround_h"],
                cycles=cfg["cycles"],
                min_uplink_s=cfg["min_uplink_s"],
                min_downlink_s=cfg["min_downlink_s"],
                plan_upload_s=cfg["plan_upload_s"],
            )
        except Exception as exc:
            st.error(f"Invalid scenario parameters: {exc}")
            st.stop()

    # ── Compute contact windows ───────────────────────────────────────────
    span_hours = scenario.time_span.hours
    start_iso = scenario.time_span.start

    with st.spinner("Computing contact windows…"):
        try:
            windows = _compute_windows_cached(
                tle_text, cfg["stations_yaml"], start_iso, span_hours
            )
        except Exception as exc:
            st.error(f"Contact window computation failed: {exc}")
            st.stop()

    # ── Run simulation ────────────────────────────────────────────────────
    seed = int(getattr(scenario, "seed", 42))
    mc_on = cfg["mc_on"]
    mc_runs = cfg.get("mc_runs", 1)

    with st.spinner("Simulating…"):
        if mc_on and mc_runs > 1:
            mc_conv, mc_cont = run_monte_carlo(scenario, windows, n_runs=mc_runs, base_seed=seed)
            # Also compute a single run for the timeline
            conv, cont = run_scenario(scenario, windows, seed=seed)
        else:
            conv, cont = run_scenario(scenario, windows, seed=seed)
            mc_conv = mc_cont = None

    # ── Headline ──────────────────────────────────────────────────────────
    rt_saved = conv.round_trips - cont.round_trips
    wait_saved_s = conv.time_blocked_s - cont.time_blocked_s

    anticipated_count = sum(1 for a in scenario.anomalies if a.anticipated)
    unanticipated_count = len(scenario.anomalies) - anticipated_count

    if mc_on and mc_conv:
        rt_med = mc_conv.round_trips_median - mc_cont.round_trips_median
        wait_med_h = mc_conv.time_blocked_h_median - mc_cont.time_blocked_h_median
        headline = (
            f"Contingency plans remove **{rt_med:.0f} of {mc_conv.round_trips_median:.0f} "
            f"round trips** and **{wait_med_h:.1f}h of waiting** (median across {mc_runs} runs)."
        )
    else:
        rt_total = conv.round_trips
        headline = (
            f"Contingency plans remove **{rt_saved} of {rt_total} round trips** "
            f"and **{_fmt_h(wait_saved_s)} of waiting**."
        )

    st.markdown(f'<div class="headline">{headline}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub">'
        f"{sat_name} · {cfg['net_label']} · "
        f"{len(windows)} contacts in {span_hours:.0f}h · "
        f"{anticipated_count} anticipated + {unanticipated_count} unanticipated anomalies"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Timeline ──────────────────────────────────────────────────────────
    st.subheader("Contact windows & events")
    fig = _build_timeline(windows, conv, cont, span_hours)
    st.plotly_chart(fig, use_container_width=True)

    # ── Comparison table ──────────────────────────────────────────────────
    st.subheader("Metrics comparison")
    if mc_on and mc_conv:
        st.caption(f"Monte Carlo N={mc_runs} — median [p25–p75]")
        _render_mc_table(mc_conv, mc_cont)
    else:
        _render_table(conv, cont)

    # ── Assumptions panel ─────────────────────────────────────────────────
    with st.expander("Assumptions used in this simulation", expanded=False):
        asmp = scenario.assumptions
        rows = [
            ("Ground turnaround time",
             f"{asmp.ground_turnaround_s / 3600:.2f} h",
             "Time from anomaly telemetry seen to command ready"),
            ("Min contact for uplink",
             f"{asmp.min_uplink_contact_s:.0f} s",
             "Shortest usable pass for sending a command"),
            ("Min contact for downlink",
             f"{asmp.min_downlink_contact_s:.0f} s",
             "Shortest usable pass for pulling telemetry"),
            ("Plan upload time",
             f"{asmp.plan_upload_s:.0f} s",
             "Time to uplink a contingency plan"),
        ]
        for kind, cyc in asmp.command_cycles_per_anomaly.items():
            rows.append((
                f"Command cycles — {kind.replace('_', ' ')}",
                f"{cyc}",
                "Downlink→uplink loops to resolve this anomaly type",
            ))

        for label, value, note in rows:
            c1, c2, c3 = st.columns([2, 1, 3])
            c1.markdown(f'<div class="assumption-row">{label}</div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="assumption-row"><b>{value}</b></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="assumption-row" style="color:#888">{note}</div>',
                        unsafe_allow_html=True)

        st.caption(
            "🛰  Simulated estimate based on public orbit data (CelesTrak) and the "
            "assumptions shown above.  "
            "Actual operations depend on specific mission constraints."
        )

    # ── Sensitivity to turnaround time ───────────────────────────────────────
    with st.expander("Sensitivity to ground turnaround time", expanded=False):
        turnaround_values_h = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
        sweep_rows = []
        for t_h in turnaround_values_h:
            sc_sweep = copy.deepcopy(scenario)
            sc_sweep.assumptions.ground_turnaround_s = t_h * 3600
            c_sw, k_sw = run_scenario(sc_sweep, windows, seed=seed)
            sweep_rows.append({
                "t": t_h,
                "rt_saved": c_sw.round_trips - k_sw.round_trips,
                "hrs_saved": (c_sw.time_blocked_s - k_sw.time_blocked_s) / 3600,
            })

        current_ta_h = scenario.assumptions.ground_turnaround_s / 3600

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=[r["t"] for r in sweep_rows],
            y=[r["rt_saved"] for r in sweep_rows],
            mode="lines+markers",
            name="Round trips saved",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=6),
            yaxis="y1",
        ))
        fig_sens.add_trace(go.Scatter(
            x=[r["t"] for r in sweep_rows],
            y=[r["hrs_saved"] for r in sweep_rows],
            mode="lines+markers",
            name="Hours of waiting saved",
            line=dict(color="#F59E0B", width=2, dash="dot"),
            marker=dict(size=6, symbol="diamond"),
            yaxis="y2",
        ))
        fig_sens.add_hline(y=0, line=dict(color="#94A3B8", width=1, dash="dash"), yref="y1")
        fig_sens.add_vline(
            x=current_ta_h,
            line=dict(color="#94A3B8", width=1, dash="dot"),
            annotation_text="current",
            annotation_position="top right",
        )
        fig_sens.update_layout(
            height=260,
            margin=dict(l=60, r=60, t=20, b=40),
            xaxis=dict(title="Ground turnaround (h)", gridcolor="#F0F0F0"),
            yaxis=dict(title="Round trips saved", side="left", gridcolor="#F0F0F0"),
            yaxis2=dict(title="Hours saved", side="right", overlaying="y", showgrid=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
            hovermode="x unified",
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        if all(r["rt_saved"] == 0 for r in sweep_rows):
            st.caption(
                "Round trips saved is zero across all turnaround values "
                "— this scenario has no unanticipated anomalies."
            )
        elif all(r["hrs_saved"] < 0.1 for r in sweep_rows):
            st.caption(
                "Waiting-time savings are small here. "
                "Consider scenarios with longer gaps between contacts."
            )

    # ── Event log (debug) ─────────────────────────────────────────────────
    with st.expander("Event log (single run)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Conventional")
            st.code("\n".join(conv.event_log), language=None)
        with c2:
            st.caption("Contingency")
            st.code("\n".join(cont.event_log), language=None)

    # ── Export HTML report ────────────────────────────────────────────────
    st.divider()
    scenario_file_label = str(cfg.get("scenario_path") or "ui_scenario (sliders)")
    reproduce = ReproduceInfo(
        scenario_file=scenario_file_label,
        seed=seed,
        tle_fetched_at=tle_fetched_at,
    )
    html_bytes = build_html_report(
        conv,
        cont,
        scenario,
        sat_name,
        cfg["net_label"],
        len(windows),
        reproduce,
        mc_conv=mc_conv,
        mc_cont=mc_cont,
        fig=_build_timeline(windows, conv, cont, span_hours),
    ).encode("utf-8")
    filename = (
        "leeway_"
        + sat_name.replace(" ", "_").replace("/", "-")[:40]
        + "_"
        + datetime.now(UTC).strftime("%Y%m%d_%H%M")
        + ".html"
    )
    st.download_button(
        "Export HTML report",
        data=html_bytes,
        file_name=filename,
        mime="text/html",
        help="Self-contained HTML file — open offline or email to an operator.",
    )


if __name__ == "__main__" or True:
    main()
