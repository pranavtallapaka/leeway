# Leeway

## Product

Leeway cuts the number of command round trips between operators and spacecraft by giving the spacecraft contingency plans it can follow between ground passes.

**Phase A goal:** a "round-trip calculator" to show satellite operators. For a real satellite and ground-station network it compares:

- **Conventional operations** — every decision waits for the next ground pass
- **Contingency-plan operations** — the spacecraft follows pre-uploaded plans autonomously

The calculator shows round trips saved and hours saved.

---

## Architecture

```
src/leeway/
  orbit.py       — contact window computation (Skyfield only)
  spacecraft.py  — spacecraft state: modes, resources, health telemetry
  plan.py        — contingency plan data structures; scenario file loading
  validator.py   — schema & constraint validation for scenarios and plans
  executor.py    — command execution simulation (Phase A: deterministic state machine)
  agents.py      — autonomous agents that select contingency plans
  assurance.py   — domain-agnostic assurance: coverage, completeness, conflict detection
  report.py      — round-trip and hours-saved report generation
```

`scenarios/` — YAML/TOML scenario files (source of truth for every assumption)  
`data/cache/` — Skyfield ephemeris and TLE snapshots (gitignored, auto-downloaded)  
`tests/`      — pytest suite

---

## Rules

### Reproducibility
- Phase A is **fully deterministic** — no LLM calls, no randomness beyond a seeded RNG.
- Every run must be exactly reproducible from a **scenario file** plus a **seed**.

### Scenario files are the single source of truth for assumptions
Every assumption — ground turnaround time, anomaly-cycles-per-event, elevation mask, contact overhead, plan upload time, etc. — lives in the scenario file.  
**Never hard-code operational parameters in Python source.**

### Orbit propagation
- Use **Skyfield** for all orbit work (TLE ingestion, SGP4 propagation, contact window computation).  
- **Never write our own propagator.**

### Assurance module
- `assurance.py` must contain **no space-specific code** (no orbits, no TLEs, no spacecraft modes).  
- It receives abstract structures (plans, graphs, coverage sets) and returns verdicts.  
- Space knowledge belongs in `plan.py` / `validator.py`.

### Agents
- Agents (in `agents.py`) may **never** call the executor or send commands.  
- They observe state and return a selected plan or a recommendation; the executor acts on it separately.

### Code style
- Ruff for linting and import sorting (`uv run ruff check .`).
- pytest for all tests (`uv run pytest`).
- Type-annotate all public functions.
- No hard-coded file paths; use `pathlib.Path` and pass paths as arguments.

---

## Phase Roadmap

| Phase | Scope |
|-------|-------|
| **A** | Round-trip calculator — orbit windows + ops simulator + report |
| B | Contingency plan language + validator |
| C | On-board executor simulation |
| D | Autonomous agent (assurance-gated) |
