"""Leeway command-line interface."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click

from leeway.ground import load_stations
from leeway.ops_sim import run_monte_carlo, run_scenario
from leeway.orbit import (
    PassSummary,
    compute_contact_windows,
    compute_gaps,
    fetch_tle,
    load_tle_file,
    parse_tle,
    summarise,
    tle_cache_timestamp,
)
from leeway.plan import NoradSatellite, TLEFileSatellite, load_scenario
from leeway.report import ReproduceInfo, render_comparison, render_monte_carlo
from leeway.validator import validate_scenario

_DEFAULT_CACHE_DIR = Path("data/cache")


@click.group()
def main() -> None:
    """Leeway — spacecraft contingency-plan round-trip calculator."""


@main.command()
@click.option("--norad", type=int, required=True, help="NORAD catalog number.")
@click.option(
    "--stations",
    "stations_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to ground station YAML file.",
)
@click.option(
    "--hours", type=float, default=48.0, show_default=True, help="Lookahead window (hours)."
)
@click.option(
    "--tle-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Local TLE file (skips CelesTrak fetch).",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=_DEFAULT_CACHE_DIR,
    show_default=True,
    help="Directory for cached TLE data.",
)
@click.option(
    "--start",
    type=str,
    default=None,
    help="UTC start time ISO-8601 (default: now).",
)
def passes(
    norad: int,
    stations_file: Path,
    hours: float,
    tle_file: Path | None,
    cache_dir: Path,
    start: str | None,
) -> None:
    """Compute and display contact windows and gaps for a satellite."""
    # --- resolve start time ---
    if start is not None:
        t_start = datetime.fromisoformat(start).replace(tzinfo=UTC)
    else:
        t_start = datetime.now(UTC).replace(second=0, microsecond=0)
    t_end = t_start + timedelta(hours=hours)

    # --- load TLE ---
    if tle_file is not None:
        tle_text = load_tle_file(tle_file)
        click.echo(f"Using local TLE file: {tle_file}", err=True)
    else:
        click.echo(f"Fetching TLE for NORAD {norad} from CelesTrak…", err=True)
        try:
            tle_text = fetch_tle(norad, cache_dir)
        except Exception as exc:
            click.echo(f"Error fetching TLE: {exc}", err=True)
            sys.exit(1)

    satellite = parse_tle(tle_text)
    tle_lines = tle_text.strip().splitlines()
    sat_name = tle_lines[0] if len(tle_lines) == 3 else f"NORAD {norad}"

    # --- load stations ---
    stations = load_stations(stations_file)

    # --- compute ---
    windows = compute_contact_windows(satellite, stations, t_start, t_end)
    gaps = compute_gaps(windows, t_start, t_end)
    summary = summarise(windows, gaps, hours)

    # --- render ---
    _print_header(sat_name, norad, t_start, t_end, stations)
    _print_passes_table(windows)
    _print_gaps_table(gaps)
    _print_summary(summary)


def _print_header(
    sat_name: str,
    norad: int,
    t_start: datetime,
    t_end: datetime,
    stations: list,
) -> None:
    click.echo()
    click.echo(f"  Satellite : {sat_name.strip()} (NORAD {norad})")
    start_s = t_start.strftime("%Y-%m-%d %H:%M")
    end_s = t_end.strftime("%Y-%m-%d %H:%M")
    click.echo(f"  Window    : {start_s} → {end_s} UTC")
    click.echo(f"  Stations  : {', '.join(s.name for s in stations)}")
    click.echo()


def _print_passes_table(windows: list) -> None:
    click.echo("  CONTACT WINDOWS")
    hdr = f"  {'#':>3}  {'Station':<28}  {'Start (UTC)':<17}  {'End (UTC)':<17}"
    click.echo(hdr + f"  {'Dur (min)':>9}  {'Max El':>7}")
    click.echo("  " + "-" * 94)
    for i, w in enumerate(windows, 1):
        click.echo(
            f"  {i:>3}  {w.station:<28}  "
            f"{w.start.strftime('%m-%d %H:%M:%S'):<17}  "
            f"{w.end.strftime('%m-%d %H:%M:%S'):<17}  "
            f"{w.duration_minutes:>9.1f}  "
            f"{w.max_elevation_deg:>6.1f}°"
        )
    if not windows:
        click.echo("       (no contacts in window)")
    click.echo()


def _print_gaps_table(gaps: list) -> None:
    click.echo("  GAPS (no-contact periods)")
    click.echo(f"  {'#':>3}  {'Start (UTC)':<17}  {'End (UTC)':<17}  {'Duration':>12}")
    click.echo("  " + "-" * 58)
    for i, g in enumerate(gaps, 1):
        click.echo(
            f"  {i:>3}  "
            f"{g.start.strftime('%m-%d %H:%M:%S'):<17}  "
            f"{g.end.strftime('%m-%d %H:%M:%S'):<17}  "
            f"{_fmt_duration(g.duration_s):>12}"
        )
    if not gaps:
        click.echo("       (no gaps — continuous coverage)")
    click.echo()


def _print_summary(s: PassSummary) -> None:
    click.echo("  SUMMARY")
    click.echo(f"  {'Span':<28} {s.span_hours:.1f} h")
    click.echo(f"  {'Contacts':<28} {s.contact_count}")
    click.echo(f"  {'Contacts/day':<28} {s.contacts_per_day:.1f}")
    click.echo(f"  {'Avg contact':<28} {s.avg_contact_minutes:.1f} min")
    click.echo(f"  {'Avg gap':<28} {_fmt_duration(s.avg_gap_s)}")
    click.echo(f"  {'Longest gap':<28} {_fmt_duration(s.longest_gap_s)}")
    click.echo(f"  {'Total in contact':<28} {_fmt_duration(s.total_contact_s)}")
    click.echo(f"  {'Total in gap (waiting)':<28} {_fmt_duration(s.total_gap_s)}")
    click.echo()


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# leeway run
# ---------------------------------------------------------------------------


@main.command()
@click.argument("scenario_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--runs",
    type=int,
    default=1,
    show_default=True,
    help="Monte Carlo iterations. 1 = single deterministic run.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Override the scenario seed.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=_DEFAULT_CACHE_DIR,
    show_default=True,
    help="Directory for cached TLE data.",
)
@click.option("--verbose", is_flag=True, help="Show per-mode event logs.")
def run(
    scenario_file: Path,
    runs: int,
    seed: int | None,
    cache_dir: Path,
    verbose: bool,
) -> None:
    """Run a scenario and compare conventional vs contingency operations."""
    # --- load and validate ---
    try:
        scenario = load_scenario(scenario_file)
    except Exception as exc:
        click.echo(f"Failed to load scenario: {exc}", err=True)
        sys.exit(1)

    result = validate_scenario(scenario, scenario_file)
    for err in result.errors:
        click.echo(f"  ERROR   {err}", err=True)
    for warn in result.warnings:
        click.echo(f"  WARNING {warn}", err=True)
    if not result.ok:
        sys.exit(1)

    effective_seed = seed if seed is not None else scenario.seed

    # --- resolve TLE ---
    sat = scenario.satellite
    tle_fetched_at = "unknown"
    if isinstance(sat, TLEFileSatellite):
        tle_path = scenario_file.parent / sat.tle_file
        tle_text = load_tle_file(tle_path)
        tle_fetched_at = "local-file"
        click.echo(f"Using local TLE: {tle_path}", err=True)
    else:
        assert isinstance(sat, NoradSatellite)
        click.echo(f"Fetching TLE for NORAD {sat.norad_id}…", err=True)
        try:
            tle_text = fetch_tle(sat.norad_id, cache_dir)
            ts = tle_cache_timestamp(cache_dir, sat.norad_id)
            if ts is not None:
                tle_fetched_at = ts.strftime("%Y-%m-%dT%H:%MZ")
        except Exception as exc:
            click.echo(f"TLE fetch failed: {exc}", err=True)
            sys.exit(1)

    satellite = parse_tle(tle_text)
    tle_lines = tle_text.strip().splitlines()
    sat_name = tle_lines[0].strip() if len(tle_lines) == 3 else str(sat)

    # --- compute contact windows ---

    t_start = datetime.fromisoformat(
        scenario.time_span.start.replace("Z", "+00:00")
    ).astimezone(UTC)
    t_end = t_start + timedelta(hours=scenario.time_span.hours)

    stations = load_stations(scenario_file.parent / scenario.ground_stations_file)
    click.echo(
        f"Computing contact windows for {sat_name} over "
        f"{len(stations)} station(s)…",
        err=True,
    )
    windows = compute_contact_windows(satellite, stations, t_start, t_end)
    click.echo(f"  {len(windows)} contact windows found.", err=True)

    # --- header ---
    click.echo()
    click.echo(f"  Scenario  : {scenario.name}")
    if scenario.description:
        desc = scenario.description.strip().replace("\n", " ")
        click.echo(f"  {'':10}  {desc[:80]}")
    click.echo(f"  Satellite : {sat_name}")
    start_s = t_start.strftime("%Y-%m-%d %H:%M")
    end_s = t_end.strftime("%Y-%m-%d %H:%M")
    click.echo(f"  Window    : {start_s} → {end_s} UTC  ({scenario.time_span.hours:.0f}h)")
    click.echo(f"  Stations  : {', '.join(s.name for s in stations)}")
    click.echo(
        f"  Contacts  : {len(windows)}  |  "
        f"Objectives: {len(scenario.objectives)}  |  "
        f"Anomalies: {len(scenario.anomalies)}"
        f"  ({sum(1 for a in scenario.anomalies if a.anticipated)} anticipated, "
        f"{sum(1 for a in scenario.anomalies if not a.anticipated)} unanticipated)"
    )
    click.echo(f"  Seed      : {effective_seed}")

    # --- simulate ---
    reproduce = ReproduceInfo(
        scenario_file=str(scenario_file),
        seed=effective_seed,
        tle_fetched_at=tle_fetched_at,
    )
    if runs == 1:
        conv, cont = run_scenario(scenario, windows, seed=effective_seed)
        click.echo(render_comparison(conv, cont, verbose=verbose, reproduce=reproduce))
    else:
        click.echo(f"\n  Running {runs} Monte Carlo iterations…", err=True)
        mc_conv, mc_cont = run_monte_carlo(scenario, windows, runs, base_seed=effective_seed)
        click.echo(render_monte_carlo(mc_conv, mc_cont, reproduce=reproduce))
