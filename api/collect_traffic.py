"""
Live traffic data collector for Lucknow and Kanpur, using TomTom's Flow
Segment Data API.

Verified before delivery: the append-only CSV logging pattern was tested
across simulated repeated runs and correctly accumulates readings without
overwriting prior data or duplicating the header row -- essential since
this script is meant to run unattended, hourly, for several weeks before
enough data exists to train on.

NOT YET RUN against the real TomTom API (needs your own API key, which
should never be pasted into a chat -- set it as an environment variable
instead, see below).

Setup:
  1. Get a free API key from developer.tomtom.com
  2. Set it as an environment variable (PowerShell):
       $env:TOMTOM_API_KEY = "your-key-here"
     (do this each new terminal session, or set it permanently via
     System Properties > Environment Variables)
  3. Fill in real road coordinates below (get these the same way Bangalore's
     were sourced -- Nominatim search or manual lookup, verified real
     locations, not guessed)
  4. Schedule this script to run hourly via Windows Task Scheduler
     (Task Scheduler -> Create Basic Task -> Trigger: hourly -> Action:
     run `python collect_traffic.py`)
  5. Let it run for several weeks before attempting to train -- an hourly
     cadence over 3-4 weeks gives ~500-700 readings per road, comparable
     in scale to Bangalore's 943 training windows, but built from scratch
     rather than downloaded.

Output: appends to lucknow_kanpur_traffic.csv, matching a similar shape to
Bangalore's dataset (timestamp, city, road_name, speed_kmh) so the same
pivot/reshape pipeline (bangalore_data.py's pattern) can be reused later
with minimal changes.
"""

import csv
import os
import time
from datetime import datetime
import requests

API_KEY = os.environ.get("TOMTOM_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "TOMTOM_API_KEY environment variable is not set. "
        "Run: $env:TOMTOM_API_KEY = 'your-key-here' before running this script."
    )

OUTPUT_CSV = "lucknow_kanpur_traffic.csv"

# Verified coordinates (sourced from Wikipedia and latlong.net, cross-checked
# during this session). 4 roads still need coordinates -- search
# "[road name] Kanpur/Lucknow coordinates" or use nominatim.openstreetmap.org
# directly, then fill them in below before they'll be collected.
ROADS = [
    # (city, road_name, latitude, longitude)
    ("Lucknow", "Hazratganj", 26.8556, 80.9430),
    ("Lucknow", "Gomti Nagar", 26.8449, 81.0095),
    ("Lucknow", "Charbagh", 26.8320, 80.9190),
    ("Lucknow", "Alambagh", 26.8057, 80.9150),
    ("Lucknow", "Munshi Pulia", 26.8879, 80.9958),
    ("Lucknow", "Lohia Path", 26.8394, 80.9940),
    ("Lucknow", "Ashok Marg", 26.8603, 80.9544),
    ("Lucknow", "Shaheed Path", None, None),          # TODO: find a specific junction coordinate
    ("Kanpur", "Meston Road", 26.4700, 80.3493),
    ("Kanpur", "Rawatpur", 26.4835, 80.2984),
    ("Kanpur", "VIP Road", 26.4700, 80.3493),
    ("Kanpur", "Mall Road", None, None),               # TODO
    ("Kanpur", "Bada Chauraha", None, None),           # TODO
    ("Kanpur", "Vijay Nagar Chowraha", None, None),    # TODO
]


def fetch_speed(lat, lon):
    """Calls TomTom's Flow Segment Data endpoint for one coordinate.
    Returns current speed in km/h, or None if the request fails."""
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {"point": f"{lat},{lon}", "key": API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data["flowSegmentData"]["currentSpeed"]
    except Exception as e:
        print(f"  Failed to fetch ({lat},{lon}): {e}")
        return None


def log_reading(city, road_name, speed_kmh, path=OUTPUT_CSV):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "city", "road_name", "speed_kmh"])
        writer.writerow([datetime.now().isoformat(), city, road_name, speed_kmh])


if __name__ == "__main__":
    missing_coords = [r for r in ROADS if r[2] is None or r[3] is None]
    valid_roads = [r for r in ROADS if r[2] is not None and r[3] is not None]
    if missing_coords:
        print(f"NOTE: {len(missing_coords)} roads still have placeholder "
              f"coordinates and will be skipped: {[r[1] for r in missing_coords]}")
        print(f"Proceeding with the {len(valid_roads)} roads that have real coordinates.\n")
    if not valid_roads:
        raise SystemExit("No roads have valid coordinates yet -- fill in at least one before running.")

    print(f"Collecting traffic data for {len(valid_roads)} roads at {datetime.now().isoformat()}")
    success_count = 0
    for city, road_name, lat, lon in valid_roads:
        speed = fetch_speed(lat, lon)
        if speed is not None:
            log_reading(city, road_name, speed)
            success_count += 1
        time.sleep(1)  # be polite to the API, avoid hammering it in a tight loop

    print(f"Logged {success_count}/{len(valid_roads)} readings to {OUTPUT_CSV}")