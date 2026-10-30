# Leeway — Contingency-Plan Round-Trip Calculator

Leeway quantifies how much time and how many ground-contact round trips an operator
saves by uploading contingency plans before a spacecraft goes over the horizon.
It runs a deterministic simulation of two operating modes — conventional and
contingency — on the same real orbit, ground-station network, and anomaly set,
then reports the difference.

---

## Quick start

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — `pip install uv` or `brew install uv`
- Internet access on first run (fetches TLE data from CelesTrak; cached for 2 h after that)

### Install

```sh
git clone https://github.com/your-org/leeway.git
cd leeway
uv sync
```

### Launch the Streamlit app

```sh
uv run streamlit run app.py
```

Open http://localhost:8501. Pick a satellite preset, a ground network, and a scenario;
the headline and timeline update immediately. Click **Export HTML report** to save a
self-contained file you can email to an operator.

### Run the CLI

```sh
# Compare conventional vs contingency for a scenario file:
uv run leeway run scenarios/eo_imaging_anticipated.yaml

# Monte Carlo across 100 seeds:
uv run leeway run scenarios/eo_imaging_mixed.yaml --runs 100

# Print contact windows for a satellite:
uv run leeway passes \
    --norad 40697 \
    --stations scenarios/ground/polar_network.yaml \
    --hours 24
```

---

## Running tests and linting

```sh
uv run pytest          # full test suite (all tests pass offline)
uv run ruff check .    # linter — zero warnings expected
```

---

## Preset satellites

Defined in `scenarios/presets.yaml`. NORAD IDs verified against the CelesTrak GP
catalogue on 2026-10-30.

| Satellite | NORAD ID | Orbit |
|---|---|---|
| Sentinel-2A | 40697 | 786 km SSO — ESA MultiSpectral Instrument |
| Sentinel-2B | 42063 | 786 km SSO — ESA MSI twin |
| Landsat 9 | 49260 | 705 km SSO — NASA/USGS OLI-2 + TIRS-2 |

---

## Scenario files

Scenarios are YAML files in `scenarios/`. Every operational assumption lives in the
scenario file — nothing is hard-coded in the Python source.

```yaml
name: eo_imaging_anticipated
satellite:
  norad_id: 40697                  # Sentinel-2A
ground_stations_file: ground/polar_network.yaml
time_span:
  start: "2024-01-10T12:00:00Z"
  hours: 24
seed: 42

objectives:
  - id: image_alps
    kind: image_target
    duration_s: 120

anomalies:
  - id: payload_reset_1
    kind: payload_reset
    timing:
      kind: random
      earliest_s: 7200
      latest_s: 28800
    anticipated: true              # covered by an on-board contingency plan

assumptions:
  ground_turnaround_s: 3600        # 1 h
  command_cycles_per_anomaly:
    payload_reset: 2
  min_uplink_contact_s: 30
  min_downlink_contact_s: 120
  plan_upload_s: 180
```

Set `anticipated: true` for anomaly kinds that have a pre-uploaded contingency plan.
Set `anticipated: false` for anomalies the spacecraft cannot handle alone — these
require at least one ground-contact cycle even in contingency mode.

---

## Modeling assumptions and defaults

| Assumption | Default | Notes |
|---|---|---|
| Ground turnaround time | 1 h | Seconds from anomaly downlink to command ready to uplink |
| Command cycles — payload\_reset | 2 | Downlink→uplink loops needed to resolve |
| Command cycles — software\_fault | 3 | |
| Command cycles — missed\_window | 1 | |
| Command cycles — all other kinds | 1 | Default fallback when not listed |
| Min contact duration for downlink | 120 s | Shortest usable pass for pulling telemetry |
| Min contact duration for uplink | 30 s | Shortest usable pass for sending a command |
| Plan upload time | 60 – 180 s | Pre-mission overhead; not on the critical path |
| Elevation mask | 5° | Applied to all ground stations |
| TLE source | CelesTrak GP API | Fetched per NORAD ID; cached 2 h in `data/cache/` |
| Simulation | Deterministic, seed-controlled | `RandomRange` anomaly times use a seeded RNG |
| Same-pass uplink + downlink | Not modelled | Each command cycle uses two distinct contacts |
| Objectives | Sequential, after all anomalies clear | Simplified scheduling |

**What each metric means:**

- **Round trips** — ground contacts required because of anomalies (downlink + uplink = 2 per cycle)
- **Time blocked** — seconds between an anomaly occurring and it being resolved; zero for
  anticipated anomalies in contingency mode
- **Objectives completed** — mission objectives that finish within the scenario time span

---

## Data and privacy

No credentials or API keys are required. TLE data is public and fetched from
[CelesTrak](https://celestrak.org). No user data is stored or transmitted.
Exported HTML reports are generated locally and never leave your machine unless
you share them.

---

## Deployment

### Streamlit Community Cloud

1. Push this repository to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set **Main file path** to `app.py`.
4. Click **Deploy**. No secrets or environment variables are needed.

TLEs are fetched at runtime; `data/cache/` is created automatically and is excluded
from git via `.gitignore`.

---

## Project structure

```
app.py                          Streamlit UI
src/leeway/
  orbit.py                      Contact window computation (Skyfield / SGP4)
  ops_sim.py                    Deterministic two-mode simulator
  plan.py                       Scenario schema (Pydantic v2) and loader
  report.py                     Text report rendering + ReproduceInfo
  html_report.py                Self-contained HTML report generator
  ground.py                     Ground station YAML loader
  validator.py                  Semantic scenario validation
  cli.py                        `leeway` CLI entry point
scenarios/
  presets.yaml                  Canonical preset satellites, networks, scenarios
  ground/                       Ground station YAML files
  eo_imaging_anticipated.yaml   All-anticipated anomaly example
  eo_imaging_mixed.yaml         Mixed anticipated + unanticipated example
  eo_sparse_coverage.yaml       Single mid-latitude station, worst-case gaps
data/cache/                     TLE and ephemeris cache (gitignored, auto-created)
tests/                          pytest suite
```
