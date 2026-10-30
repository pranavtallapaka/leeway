"""Leeway orbit module.

Orbit propagation and contact window computation using Skyfield.
Never write our own propagator — all TLE/SGP4 work goes through Skyfield.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from skyfield.api import EarthSatellite, Time, Topos, load, wgs84
from skyfield.timelib import Timescale

# ---------------------------------------------------------------------------
# CelesTrak GP API
# ---------------------------------------------------------------------------

_CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
_CACHE_MAX_AGE_SECONDS = 2 * 3600  # respect CelesTrak's 2-hour update cycle


def _tle_cache_path(cache_dir: Path, norad_id: int) -> Path:
    return cache_dir / f"tle_{norad_id}.txt"


def _tle_cache_meta_path(cache_dir: Path, norad_id: int) -> Path:
    return cache_dir / f"tle_{norad_id}.meta"


def _cache_is_fresh(cache_dir: Path, norad_id: int) -> bool:
    meta = _tle_cache_meta_path(cache_dir, norad_id)
    if not meta.exists():
        return False
    fetched_at = float(meta.read_text().strip())
    return (time.time() - fetched_at) < _CACHE_MAX_AGE_SECONDS


def fetch_tle(
    norad_id: int,
    cache_dir: Path,
    *,
    force_refresh: bool = False,
    timeout: float = 30.0,
) -> str:
    """Return a three-line TLE string for *norad_id*.

    Fetches from CelesTrak if the cache is absent or stale (>2 h).
    Writes the result to *cache_dir* for subsequent calls.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _tle_cache_path(cache_dir, norad_id)

    if not force_refresh and _cache_is_fresh(cache_dir, norad_id) and cache_path.exists():
        return cache_path.read_text()

    url = f"{_CELESTRAK_GP_URL}?CATNR={norad_id}&FORMAT=TLE"
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    tle_text = response.text.strip()

    if not tle_text or "No GP data found" in tle_text:
        raise ValueError(f"CelesTrak returned no TLE data for NORAD ID {norad_id}")

    cache_path.write_text(tle_text)
    _tle_cache_meta_path(cache_dir, norad_id).write_text(str(time.time()))
    return tle_text


def load_tle_file(path: Path) -> str:
    """Load TLE text from a local file (offline use)."""
    return path.read_text()


def tle_cache_timestamp(cache_dir: Path, norad_id: int) -> datetime | None:
    """Return the UTC datetime when the TLE for *norad_id* was last fetched from CelesTrak.

    Returns None if the cache metadata file does not exist.
    """
    meta = _tle_cache_meta_path(cache_dir, norad_id)
    if not meta.exists():
        return None
    return datetime.fromtimestamp(float(meta.read_text().strip()), tz=UTC)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_tle(tle_text: str) -> EarthSatellite:
    """Parse a 2- or 3-line TLE string into a Skyfield EarthSatellite."""
    lines = [ln for ln in tle_text.strip().splitlines() if ln.strip()]
    if len(lines) == 3:
        name, line1, line2 = lines[0], lines[1], lines[2]
    elif len(lines) == 2:
        name, line1, line2 = "", lines[0], lines[1]
    else:
        raise ValueError(f"Expected 2 or 3 TLE lines, got {len(lines)}")
    ts = load.timescale()
    return EarthSatellite(line1, line2, name, ts)


# ---------------------------------------------------------------------------
# Ground station
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundStation:
    name: str
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    min_elevation_deg: float

    def topos(self) -> Topos:
        return wgs84.latlon(
            self.latitude_deg,
            self.longitude_deg,
            elevation_m=self.altitude_m,
        )


# ---------------------------------------------------------------------------
# Contact windows
# ---------------------------------------------------------------------------


@dataclass
class ContactWindow:
    station: str
    start: datetime
    end: datetime
    duration_s: float
    max_elevation_deg: float

    @property
    def duration_minutes(self) -> float:
        return self.duration_s / 60.0


@dataclass
class Gap:
    start: datetime
    end: datetime
    duration_s: float

    @property
    def duration_minutes(self) -> float:
        return self.duration_s / 60.0

    @property
    def duration_hours(self) -> float:
        return self.duration_s / 3600.0


def _find_windows_for_station(
    satellite: EarthSatellite,
    station: GroundStation,
    t0: Time,
    t1: Time,
    ts: Timescale,
) -> list[ContactWindow]:
    """Find all contact windows between *satellite* and *station* in [t0, t1]."""
    topos = station.topos()
    t_arr, events = satellite.find_events(
        topos,
        t0,
        t1,
        altitude_degrees=station.min_elevation_deg,
    )
    # events: 0=rise, 1=culminate, 2=set
    windows: list[ContactWindow] = []
    current_rise: Time | None = None
    max_el: float = 0.0

    for t, ev in zip(t_arr, events):
        if ev == 0:  # rise
            current_rise = t
            max_el = 0.0
        elif ev == 1:  # culminate
            alt, _, _ = (satellite - topos).at(t).altaz()
            max_el = float(alt.degrees)
        elif ev == 2:  # set
            if current_rise is not None:
                duration = (t.tt - current_rise.tt) * 86400.0
                windows.append(
                    ContactWindow(
                        station=station.name,
                        start=current_rise.utc_datetime(),
                        end=t.utc_datetime(),
                        duration_s=duration,
                        max_elevation_deg=max_el,
                    )
                )
                current_rise = None

    return windows


def compute_contact_windows(
    satellite: EarthSatellite,
    stations: list[GroundStation],
    t_start: datetime,
    t_end: datetime,
) -> list[ContactWindow]:
    """Compute all contact windows for *satellite* over *stations* in [t_start, t_end].

    Windows from all stations are merged (overlapping windows are preserved — the
    satellite is in contact with *some* station, which is what matters for Leeway).
    Results are sorted by start time.
    """
    ts = load.timescale()
    t0 = ts.from_datetime(t_start.replace(tzinfo=UTC) if t_start.tzinfo is None else t_start)
    t1 = ts.from_datetime(t_end.replace(tzinfo=UTC) if t_end.tzinfo is None else t_end)

    all_windows: list[ContactWindow] = []
    for station in stations:
        all_windows.extend(_find_windows_for_station(satellite, station, t0, t1, ts))

    all_windows.sort(key=lambda w: w.start)
    return all_windows


def compute_gaps(
    windows: list[ContactWindow],
    t_start: datetime,
    t_end: datetime,
) -> list[Gap]:
    """Compute gaps between contact windows (the waiting time that matters for Leeway).

    Gaps are periods when the satellite is not in contact with any station in the network.
    The leading gap (before first contact) and trailing gap (after last contact) are included.
    """
    if not windows:
        total = (t_end - t_start).total_seconds()
        return [Gap(start=t_start, end=t_end, duration_s=total)]

    gaps: list[Gap] = []
    t_start_utc = t_start.replace(tzinfo=UTC) if t_start.tzinfo is None else t_start
    t_end_utc = t_end.replace(tzinfo=UTC) if t_end.tzinfo is None else t_end

    # Merge overlapping windows into coverage intervals before computing gaps
    merged = _merge_windows(windows)

    # Leading gap
    if merged[0].start > t_start_utc:
        gaps.append(Gap(
            start=t_start_utc,
            end=merged[0].start,
            duration_s=(merged[0].start - t_start_utc).total_seconds(),
        ))

    # Inter-contact gaps
    for i in range(len(merged) - 1):
        gap_start = merged[i].end
        gap_end = merged[i + 1].start
        if gap_end > gap_start:
            gaps.append(Gap(
                start=gap_start,
                end=gap_end,
                duration_s=(gap_end - gap_start).total_seconds(),
            ))

    # Trailing gap
    if merged[-1].end < t_end_utc:
        gaps.append(Gap(
            start=merged[-1].end,
            end=t_end_utc,
            duration_s=(t_end_utc - merged[-1].end).total_seconds(),
        ))

    return gaps


def _merge_windows(windows: list[ContactWindow]) -> list[ContactWindow]:
    """Merge overlapping/adjacent windows into non-overlapping coverage intervals.

    The returned windows have station="" and max_elevation=0 — they represent
    merged coverage spans, not individual station contacts.
    """
    if not windows:
        return []
    sorted_w = sorted(windows, key=lambda w: w.start)
    merged: list[ContactWindow] = []
    cur_start = sorted_w[0].start
    cur_end = sorted_w[0].end

    for w in sorted_w[1:]:
        if w.start <= cur_end:
            cur_end = max(cur_end, w.end)
        else:
            merged.append(ContactWindow(
                station="",
                start=cur_start,
                end=cur_end,
                duration_s=(cur_end - cur_start).total_seconds(),
                max_elevation_deg=0.0,
            ))
            cur_start = w.start
            cur_end = w.end

    merged.append(ContactWindow(
        station="",
        start=cur_start,
        end=cur_end,
        duration_s=(cur_end - cur_start).total_seconds(),
        max_elevation_deg=0.0,
    ))
    return merged


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


@dataclass
class PassSummary:
    span_hours: float
    contact_count: int
    contacts_per_day: float
    avg_contact_s: float
    avg_gap_s: float
    longest_gap_s: float
    total_contact_s: float
    total_gap_s: float

    @property
    def avg_contact_minutes(self) -> float:
        return self.avg_contact_s / 60.0

    @property
    def avg_gap_hours(self) -> float:
        return self.avg_gap_s / 3600.0

    @property
    def longest_gap_hours(self) -> float:
        return self.longest_gap_s / 3600.0


def summarise(
    windows: list[ContactWindow],
    gaps: list[Gap],
    span_hours: float,
) -> PassSummary:
    """Compute summary statistics over a set of windows and gaps."""
    contact_count = len(windows)
    total_contact_s = sum(w.duration_s for w in windows)
    total_gap_s = sum(g.duration_s for g in gaps)

    avg_contact_s = total_contact_s / contact_count if contact_count else 0.0
    avg_gap_s = total_gap_s / len(gaps) if gaps else 0.0
    longest_gap_s = max((g.duration_s for g in gaps), default=0.0)
    contacts_per_day = contact_count / (span_hours / 24.0) if span_hours > 0 else 0.0

    return PassSummary(
        span_hours=span_hours,
        contact_count=contact_count,
        contacts_per_day=contacts_per_day,
        avg_contact_s=avg_contact_s,
        avg_gap_s=avg_gap_s,
        longest_gap_s=longest_gap_s,
        total_contact_s=total_contact_s,
        total_gap_s=total_gap_s,
    )
