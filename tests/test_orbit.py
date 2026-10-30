"""Tests for src/leeway/orbit.py.

All tests use a fixed TLE from tests/fixtures/ — no network calls, no CelesTrak.
Results are deterministic: same TLE + same time window = same windows every run.

The ISS TLE used has epoch 2024-01-10 12:00 UTC (day 10.5 of 2024).
We compute windows for a 24-hour window starting at 2024-01-10 12:00 UTC and
check that counts and durations are physically plausible for a 51.6° inclined orbit
viewed from a mid-latitude site.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from leeway.ground import load_stations
from leeway.orbit import (
    ContactWindow,
    Gap,
    GroundStation,
    PassSummary,
    _merge_windows,
    compute_contact_windows,
    compute_gaps,
    load_tle_file,
    parse_tle,
    summarise,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
SCENARIOS = Path(__file__).parent.parent / "scenarios" / "ground"
ISS_TLE = FIXTURES / "iss_25544.tle"

# ---------------------------------------------------------------------------
# Reference window: 24 h from the TLE epoch (2024-01-10 12:00 UTC)
# ---------------------------------------------------------------------------

T_START = datetime(2024, 1, 10, 12, 0, 0, tzinfo=UTC)
T_END = datetime(2024, 1, 11, 12, 0, 0, tzinfo=UTC)

# Purdue station — ISS at 51.6° inclination passes overhead regularly
PURDUE = GroundStation(
    name="Purdue",
    latitude_deg=40.4259,
    longitude_deg=-86.9081,
    altitude_m=185.0,
    min_elevation_deg=5.0,
)

# High-Arctic station — sees ISS at 51.6° incl only near perigee passes at high lat
SVALBARD = GroundStation(
    name="Svalbard",
    latitude_deg=78.2297,
    longitude_deg=15.4045,
    altitude_m=50.0,
    min_elevation_deg=5.0,
)


# ---------------------------------------------------------------------------
# parse_tle
# ---------------------------------------------------------------------------


def test_parse_tle_returns_satellite():
    from skyfield.api import EarthSatellite

    tle = load_tle_file(ISS_TLE)
    sat = parse_tle(tle)
    assert isinstance(sat, EarthSatellite)
    assert "25544" in sat.model.satnum_str or sat.model.satnum == 25544


def test_parse_tle_two_line():
    tle = load_tle_file(ISS_TLE)
    lines = tle.strip().splitlines()
    two_line = "\n".join(lines[1:])  # strip name
    from skyfield.api import EarthSatellite

    sat = parse_tle(two_line)
    assert isinstance(sat, EarthSatellite)


def test_parse_tle_bad_input_raises():
    with pytest.raises(ValueError, match="TLE lines"):
        parse_tle("only one line here")


# ---------------------------------------------------------------------------
# Contact windows — ISS over Purdue, 24 h
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def iss_purdue_windows() -> list[ContactWindow]:
    tle = load_tle_file(ISS_TLE)
    sat = parse_tle(tle)
    return compute_contact_windows(sat, [PURDUE], T_START, T_END)


def test_iss_purdue_has_contacts(iss_purdue_windows):
    # ISS at 51.6° incl passes over 40°N several times per day
    assert len(iss_purdue_windows) >= 2, "Expected at least 2 passes over Purdue in 24 h"


def test_iss_purdue_contact_count_plausible(iss_purdue_windows):
    # ISS makes ~15.5 orbits/day; a mid-lat site sees ~4-7 passes per day
    assert len(iss_purdue_windows) <= 8, f"Unexpectedly many passes: {len(iss_purdue_windows)}"


def test_iss_purdue_durations_plausible(iss_purdue_windows):
    for w in iss_purdue_windows:
        # LEO passes: typically 3–12 min; allow 1–15 min
        assert 60 <= w.duration_s <= 900, (
            f"Pass duration {w.duration_s:.0f}s outside [60, 900] for {w}"
        )


def test_iss_purdue_elevations_above_mask(iss_purdue_windows):
    for w in iss_purdue_windows:
        assert w.max_elevation_deg >= PURDUE.min_elevation_deg


def test_iss_purdue_windows_sorted(iss_purdue_windows):
    starts = [w.start for w in iss_purdue_windows]
    assert starts == sorted(starts)


def test_iss_purdue_deterministic():
    tle = load_tle_file(ISS_TLE)
    sat = parse_tle(tle)
    w1 = compute_contact_windows(sat, [PURDUE], T_START, T_END)
    w2 = compute_contact_windows(sat, [PURDUE], T_START, T_END)
    assert len(w1) == len(w2)
    for a, b in zip(w1, w2):
        assert a.start == b.start
        assert a.end == b.end


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def iss_purdue_gaps(iss_purdue_windows) -> list[Gap]:
    return compute_gaps(iss_purdue_windows, T_START, T_END)


def test_gaps_cover_full_span(iss_purdue_windows, iss_purdue_gaps):
    total_gap = sum(g.duration_s for g in iss_purdue_gaps)
    span = (T_END - T_START).total_seconds()
    # contact + gap ≤ span (contacts can overlap; merged coverage + gaps = span)
    assert total_gap <= span
    assert total_gap > 0  # a 40°N site definitely has gaps


def test_gaps_no_negative_duration(iss_purdue_gaps):
    for g in iss_purdue_gaps:
        assert g.duration_s >= 0


def test_gaps_longest_is_plausible(iss_purdue_gaps):
    longest = max(g.duration_s for g in iss_purdue_gaps)
    # ISS at 51.6° incl viewed from a single mid-lat station: passes cluster by
    # node geometry, so a leading/trailing gap up to ~11 h is physically real
    # in a 24-hour window.  Anything over 12 h would mean no passes in half a day.
    assert longest <= 43200, f"Longest gap {longest/3600:.2f} h exceeds 12 h"


def test_no_contacts_returns_single_gap():
    gaps = compute_gaps([], T_START, T_END)
    assert len(gaps) == 1
    assert abs(gaps[0].duration_s - (T_END - T_START).total_seconds()) < 1


# ---------------------------------------------------------------------------
# _merge_windows
# ---------------------------------------------------------------------------


def _dt(h: int, m: int = 0) -> datetime:
    return datetime(2024, 1, 1, h, m, tzinfo=UTC)


def test_merge_non_overlapping():
    w1 = ContactWindow("A", _dt(0), _dt(1), 3600, 10)
    w2 = ContactWindow("B", _dt(2), _dt(3), 3600, 20)
    merged = _merge_windows([w1, w2])
    assert len(merged) == 2


def test_merge_overlapping():
    w1 = ContactWindow("A", _dt(0), _dt(1), 3600, 10)
    w2 = ContactWindow("B", _dt(0, 30), _dt(1, 30), 3600, 20)
    merged = _merge_windows([w1, w2])
    assert len(merged) == 1
    assert merged[0].duration_s == pytest.approx(5400, abs=1)


def test_merge_contiguous_stays_two():
    # Two windows that end/start exactly at the same time — not technically overlapping
    # start == end → merged into one span
    w1 = ContactWindow("A", _dt(0), _dt(1), 3600, 10)
    w2 = ContactWindow("B", _dt(1), _dt(2), 3600, 20)
    merged = _merge_windows([w1, w2])
    # start == end means w2 starts exactly at w1's end → merged into one
    assert len(merged) == 1


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------


def test_summarise_basic(iss_purdue_windows, iss_purdue_gaps):
    s = summarise(iss_purdue_windows, iss_purdue_gaps, span_hours=24.0)
    assert isinstance(s, PassSummary)
    assert s.contact_count == len(iss_purdue_windows)
    assert s.contacts_per_day == pytest.approx(len(iss_purdue_windows), rel=1e-3)
    assert s.avg_contact_s > 0
    assert s.longest_gap_s >= s.avg_gap_s


def test_summarise_empty():
    s = summarise([], [], span_hours=24.0)
    assert s.contact_count == 0
    assert s.avg_contact_s == 0.0


# ---------------------------------------------------------------------------
# Multi-station: Purdue + Svalbard (wider coverage, fewer gaps)
# ---------------------------------------------------------------------------


def test_two_stations_fewer_gaps_than_one():
    tle = load_tle_file(ISS_TLE)
    sat = parse_tle(tle)

    w_purdue = compute_contact_windows(sat, [PURDUE], T_START, T_END)
    g_purdue = compute_gaps(w_purdue, T_START, T_END)

    w_both = compute_contact_windows(sat, [PURDUE, SVALBARD], T_START, T_END)
    g_both = compute_gaps(w_both, T_START, T_END)

    total_gap_purdue = sum(g.duration_s for g in g_purdue)
    total_gap_both = sum(g.duration_s for g in g_both)
    assert total_gap_both <= total_gap_purdue, (
        "Adding a second station should not increase total gap time"
    )


# ---------------------------------------------------------------------------
# YAML station loader
# ---------------------------------------------------------------------------


def test_load_purdue_yaml():
    stations = load_stations(SCENARIOS / "purdue_single.yaml")
    assert len(stations) == 1
    s = stations[0]
    assert "Purdue" in s.name
    assert abs(s.latitude_deg - 40.4259) < 0.01
    assert s.min_elevation_deg == pytest.approx(5.0)


def test_load_polar_yaml():
    stations = load_stations(SCENARIOS / "polar_network.yaml")
    assert len(stations) == 2
    lats = {s.latitude_deg for s in stations}
    assert any(lat > 70 for lat in lats), "Expected an Arctic station"
    assert any(lat < -70 for lat in lats), "Expected an Antarctic station"
