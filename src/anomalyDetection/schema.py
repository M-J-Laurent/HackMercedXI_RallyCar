"""Telemetry contract shared by generation/, training/ and the telemetry-handling code.

Every telemetry CSV (recorded or synthetic) must have these columns, sampled
against `lap_distance` (metres from stage start, strictly increasing).

Units / sign conventions:
    speed, wheel_speed_*   m/s
    throttle, brake        0..1
    steer                  -1..1, positive = steering left
    g_lat                  in g, positive = turning left
    g_long                 in g, positive = accelerating
    rpm                    engine rpm
    gear                   1..N (0 = neutral, 'R' etc. must be mapped by the parser)
    x, z                   horizontal world position in metres
"""

GRID_STEP_M = 2.0

DIST = "lap_distance"
WHEELS = ["fl", "fr", "rl", "rr"]
CHANNELS = [
    "time", "x", "z", "speed", "gear", "rpm",
    "throttle", "brake", "steer", "g_lat", "g_long",
    *[f"wheel_speed_{w}" for w in WHEELS],
]
TELEMETRY_COLUMNS = [DIST, *CHANNELS]

# Columns that only exist in generated / labelled datasets.
TRACK_ID = "track_id"
RUN_ID = "run_id"
LABEL_COL = "label"

NORMAL = "normal"
FAULT_LABELS = [
    "braked_too_early",
    "braked_too_late",
    "lifted_mid_corner",
    "late_throttle",
    "oversteer",
    "understeer",
    "wrong_gear",
]
LABELS = [NORMAL, *FAULT_LABELS]
