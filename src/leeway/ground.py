"""Ground station loading from YAML scenario files.

All assumptions (coordinates, elevation mask) come from the scenario file —
nothing is hard-coded here.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from leeway.orbit import GroundStation


def load_stations(path: Path) -> list[GroundStation]:
    """Load ground stations from a YAML file.

    Expected format::

        stations:
          - name: "My Station"
            latitude_deg: 40.0
            longitude_deg: -86.0
            altitude_m: 200.0
            min_elevation_deg: 5.0
    """
    data = yaml.safe_load(path.read_text())
    stations: list[GroundStation] = []
    for entry in data["stations"]:
        stations.append(
            GroundStation(
                name=entry["name"],
                latitude_deg=float(entry["latitude_deg"]),
                longitude_deg=float(entry["longitude_deg"]),
                altitude_m=float(entry["altitude_m"]),
                min_elevation_deg=float(entry["min_elevation_deg"]),
            )
        )
    return stations
